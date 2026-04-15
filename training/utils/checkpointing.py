from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from training.data.collator import SpecialTokenIds


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, *, decay: float) -> None:
        self.decay = float(decay)
        if not (0.0 < self.decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {self.decay}")
        self.model = copy.deepcopy(model)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_params = dict(self.model.named_parameters())
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ema_params[name].lerp_(param.detach(), 1.0 - self.decay)

        ema_buffers = dict(self.model.named_buffers())
        for name, buffer in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(buffer.detach())


def maybe_build_ema(model: torch.nn.Module, cfg: dict[str, Any]) -> ExponentialMovingAverage | None:
    ema_cfg = cfg.get("ema", {})
    if not bool(ema_cfg.get("enabled", False)):
        return None
    return ExponentialMovingAverage(model, decay=float(ema_cfg.get("decay", 0.999)))


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    cfg: dict[str, Any],
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": cfg,
    }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    if metric_name is not None:
        payload["metric_name"] = metric_name
    if metric_value is not None:
        payload["metric_value"] = float(metric_value)
    torch.save(payload, path)


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float | None, str | None]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if ema_model is not None and "ema_model" in payload:
        ema_model.load_state_dict(payload["ema_model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    step = int(payload.get("step", 0))
    metric_value = payload.get("metric_value")
    metric_name = payload.get("metric_name")
    return step, None if metric_value is None else float(metric_value), metric_name


def save_training_state(
    *,
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cfg: dict[str, Any],
    step: int,
    metric_name: str | None = None,
    metric_value: float | None = None,
) -> Path:
    out_dir = Path(cfg["runtime"].get("output_dir", "out/training")).expanduser().resolve()
    checkpoint_path = out_dir / f"checkpoint-step-{step:06d}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        ema_model=None if ema is None else ema.model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        cfg=cfg,
        metric_name=metric_name,
        metric_value=metric_value,
    )
    return checkpoint_path


def maybe_resume_training_state(
    *,
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cfg: dict[str, Any],
) -> tuple[int, float | None, str | None]:
    checkpoint_path = cfg["runtime"].get("checkpoint_path")
    if checkpoint_path in (None, "", "null", "None"):
        return 0, None, None
    return load_training_checkpoint(
        Path(str(checkpoint_path)).expanduser(),
        model=model,
        ema_model=None if ema is None else ema.model,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location="cpu",
    )


def save_tokenizer_artifacts(tokenizer, special_tokens: SpecialTokenIds, cfg: dict[str, Any]) -> Path:
    output_dir = Path(cfg["runtime"].get("output_dir", "out/training")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    with (output_dir / "resolved_special_tokens.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "bos_token_id": special_tokens.bos,
                "eos_token_id": special_tokens.eos,
                "pad_wait_token_id": special_tokens.pad_wait,
                "word_start_token_id": special_tokens.word_start,
            },
            f,
            indent=2,
        )
    return output_dir
