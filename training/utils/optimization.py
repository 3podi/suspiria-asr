from __future__ import annotations

import inspect
import math
from typing import Any

import torch


def build_optimizer_param_groups(
    model: torch.nn.Module,
    *,
    weight_decay: float,
) -> list[dict[str, Any]]:
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def should_use_fused_adamw(device: torch.device, requested: bool) -> bool:
    if not requested or device.type != "cuda":
        return False
    try:
        return "fused" in inspect.signature(torch.optim.AdamW).parameters
    except (TypeError, ValueError):
        return False


def build_adamw_optimizer(
    model: torch.nn.Module,
    *,
    device: torch.device,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    fused: bool,
) -> torch.optim.Optimizer:
    param_groups = build_optimizer_param_groups(model, weight_decay=weight_decay)
    optimizer_kwargs: dict[str, Any] = {
        "lr": float(lr),
        "betas": tuple(float(x) for x in betas),
        "eps": float(eps),
    }
    if should_use_fused_adamw(device, fused):
        optimizer_kwargs["fused"] = True
    return torch.optim.AdamW(param_groups, **optimizer_kwargs)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
    final_lr_scale: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(0, int(warmup_steps))
    total_steps = max(1, int(total_steps))
    schedule = str(schedule).strip().lower()
    if schedule not in {"cosine", "constant"}:
        raise ValueError(f"Unsupported lr schedule: {schedule}")
    final_lr_scale = float(final_lr_scale)
    if not (0.0 <= final_lr_scale <= 1.0):
        raise ValueError(f"final_lr_scale must be in [0, 1], got {final_lr_scale}")

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if schedule == "constant":
            return 1.0
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_scale + (1.0 - final_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    *,
    device: torch.device,
    train_cfg: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = build_adamw_optimizer(
        model,
        device=device,
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
        eps=float(train_cfg.get("eps", 1e-8)),
        fused=bool(train_cfg.get("fused_optimizer", True)),
    )
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 1000)),
        total_steps=int(train_cfg["max_steps"]),
        schedule=str(train_cfg.get("lr_schedule", "cosine")),
        final_lr_scale=float(train_cfg.get("final_lr_scale", 0.1)),
    )
    return optimizer, scheduler
