from __future__ import annotations

import math
from typing import Any

import torch

from training.data.collator import SpecialTokenIds
from training.utils.metrics import (
    MetricCounts,
    compute_batch_metric_counts,
    finalize_metric_counts,
    merge_metric_counts,
)

@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    special_tokens: SpecialTokenIds,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    counts = MetricCounts()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(batch)
        batch_counts = compute_batch_metric_counts(
            outputs["logits"],
            batch["packed_labels"],
            special_tokens=special_tokens,
            loss_value=float(outputs["loss"].detach().cpu()),
            unweighted_loss_value=float(outputs["unweighted_loss"].detach().cpu()),
        )
        merge_metric_counts(counts, batch_counts)

    if was_training:
        model.train()

    if counts.batch_count == 0:
        return {"loss": float("nan"), "perplexity": float("nan"), "num_batches": 0.0}
    return finalize_metric_counts(counts)


def select_eval_model(
    model: torch.nn.Module,
    *,
    ema,
    cfg: dict[str, Any],
) -> torch.nn.Module:
    eval_cfg = cfg.get("evaluation", {})
    if bool(eval_cfg.get("use_ema_for_eval", True)) and ema is not None:
        return ema.model
    return model
