from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Expected Hydra config to resolve to a mapping.")
    return plain


def resolve_torch_dtype(
    value: str | torch.dtype | None,
    *,
    default: torch.dtype | None = None,
) -> torch.dtype | None:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def resolve_device(runtime_cfg: dict[str, Any]) -> torch.device:
    device_name = runtime_cfg.get("device")
    if not device_name:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(str(device_name))


def set_random_seeds(
    seed: int,
    *,
    deterministic: bool = False,
) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
