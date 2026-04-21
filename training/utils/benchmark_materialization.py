from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.materialize_latents import (
    _is_empty_path,
    _materialize_shard_rows,
    _resolve_materialized_dtype,
    load_split_manifest_rows,
    resolve_manifest_root,
)
from training.utils.logging import silence_external_info_logs

silence_external_info_logs()

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark latent parquet materialization speed for different worker counts. "
            "Only the selected parquet shards are downloaded."
        )
    )
    parser.add_argument("--config-path", default="configs/training.yaml")
    parser.add_argument("--country", default=None, help="Override dataset.country from config.")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--shards-per-worker",
        type=int,
        default=1,
        help="Default benchmark size: each worker count uses workers * shards_per_worker parquet files.",
    )
    parser.add_argument(
        "--fixed-num-shards",
        type=int,
        default=None,
        help="If set, every worker count materializes the same number of parquet files.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Parquet record batch size for materialization.")
    parser.add_argument("--output-root", default=None, help="Root for benchmark outputs.")
    parser.add_argument("--include-speaker-prefix", action="store_true")
    parser.add_argument("--force-download", action="store_true", help="Ask HF to refresh selected parquet files.")
    parser.add_argument("--results-json", default=None, help="Optional explicit JSON result path.")
    return parser.parse_args()


def load_cfg(config_path: str) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Expected training config to resolve to a mapping.")
    return plain


def unique_shards_from_manifest(
    *,
    cfg: dict[str, Any],
    split: str,
) -> list[str]:
    dataset_cfg = cfg["dataset"]
    country = str(dataset_cfg["country"])
    manifest_root = resolve_manifest_root(dataset_cfg)
    rows = load_split_manifest_rows(manifest_root=manifest_root, country=country, split=split)
    seen = set()
    shards = []
    for row in rows:
        if row.latent_shard_path in seen:
            continue
        seen.add(row.latent_shard_path)
        shards.append(row.latent_shard_path)
    if not shards:
        raise RuntimeError(f"No latent shards found from manifest for country={country!r} split={split!r}.")
    return shards


def resolve_selected_shards(
    *,
    cfg: dict[str, Any],
    shard_rel_paths: list[str],
    cache_dir: Path,
    force_download: bool,
) -> dict[str, Path]:
    dataset_cfg = cfg["dataset"]
    local_root = dataset_cfg.get("local_dataset_root")
    if not _is_empty_path(local_root):
        dataset_root = Path(str(local_root)).expanduser().resolve()
        resolved = {rel_path: (dataset_root / rel_path).resolve() for rel_path in shard_rel_paths}
        missing = [str(path) for path in resolved.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing local parquet shards: {missing[:5]}")
        return resolved

    repo_id = dataset_cfg.get("repo_id")
    if _is_empty_path(repo_id):
        raise ValueError("dataset.repo_id must be set when local_dataset_root is not provided.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(
        snapshot_download(
            repo_id=str(repo_id),
            repo_type="dataset",
            revision=dataset_cfg.get("revision"),
            allow_patterns=shard_rel_paths,
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
            force_download=bool(force_download),
        )
    ).resolve()
    resolved = {rel_path: (snapshot_root / rel_path).resolve() for rel_path in shard_rel_paths}
    missing = [rel_path for rel_path, path in resolved.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"HF snapshot did not contain requested parquet shards: {missing[:5]}")
    return resolved


def run_materialization_once(
    *,
    shard_map: dict[str, Path],
    shard_rel_paths: list[str],
    materialized_root: Path,
    num_workers: int,
    materialization_batch_size: int,
    tensor_dtype,
    materialize_speaker_prefix: bool,
) -> dict[str, float]:
    materialized_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "shard_path": shard_map[rel_path],
            "latent_shard_path": rel_path,
            "materialized_root": materialized_root,
            "force_rematerialize": True,
            "materialize_speaker_prefix": materialize_speaker_prefix,
            "tensor_dtype": tensor_dtype,
            "materialization_batch_size": materialization_batch_size,
        }
        for rel_path in shard_rel_paths
    ]

    started = time.perf_counter()
    written = 0
    skipped = 0
    max_workers = min(max(1, int(num_workers)), len(jobs), max(1, os.cpu_count() or 1))
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_materialize_shard_rows, **job) for job in jobs]
            iterator = futures if tqdm is None else tqdm(futures, total=len(futures), desc=f"workers={num_workers}", unit="file")
            for future in iterator:
                shard_written, shard_skipped = future.result()
                written += shard_written
                skipped += shard_skipped
    else:
        iterator = jobs if tqdm is None else tqdm(jobs, total=len(jobs), desc="workers=1", unit="file")
        for job in iterator:
            shard_written, shard_skipped = _materialize_shard_rows(**job)
            written += shard_written
            skipped += shard_skipped

    elapsed = time.perf_counter() - started
    return {
        "requested_workers": float(num_workers),
        "actual_workers": float(max_workers),
        "num_shards": float(len(jobs)),
        "written_samples": float(written),
        "skipped_samples": float(skipped),
        "elapsed_sec": float(elapsed),
        "samples_per_sec": float(written) / max(elapsed, 1e-9),
        "shards_per_sec": float(len(jobs)) / max(elapsed, 1e-9),
    }


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_path)
    if args.country is not None:
        cfg["dataset"]["country"] = args.country

    workers = [int(value) for value in args.workers]
    if not workers or any(value <= 0 for value in workers):
        raise ValueError("--workers must contain positive integers.")

    dataset_cfg = cfg["dataset"]
    materialization_batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else dataset_cfg.get("materialization_batch_size", 128)
    )
    if materialization_batch_size <= 0:
        raise ValueError("materialization batch size must be positive.")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root or f"out/materialization_benchmark/{run_id}").expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    max_shards_needed = max(
        int(args.fixed_num_shards)
        if args.fixed_num_shards is not None
        else int(worker_count) * max(1, int(args.shards_per_worker))
        for worker_count in workers
    )
    all_shards = unique_shards_from_manifest(cfg=cfg, split=str(args.split))
    selected_union = all_shards[: min(max_shards_needed, len(all_shards))]
    if len(selected_union) < max_shards_needed:
        print(
            f"[BENCH] requested {max_shards_needed} shards but only {len(selected_union)} "
            f"are available for split={args.split!r}."
        )

    print(f"[BENCH] downloading/resolving {len(selected_union)} selected parquet shard(s)")
    shard_map = resolve_selected_shards(
        cfg=cfg,
        shard_rel_paths=selected_union,
        cache_dir=output_root / "_parquet_cache",
        force_download=bool(args.force_download),
    )

    tensor_dtype = _resolve_materialized_dtype(
        dataset_cfg.get("materialized_dtype", cfg["runtime"].get("data_dtype", "bf16")),
        default=torch.bfloat16,
    )
    materialize_speaker_prefix = bool(args.include_speaker_prefix)

    results = []
    for worker_count in workers:
        if args.fixed_num_shards is not None:
            num_shards = min(int(args.fixed_num_shards), len(selected_union))
        else:
            num_shards = min(int(worker_count) * max(1, int(args.shards_per_worker)), len(selected_union))
        shard_rel_paths = selected_union[:num_shards]
        materialized_root = output_root / f"workers-{worker_count:02d}_shards-{num_shards:04d}"
        print(
            f"[BENCH] workers={worker_count} shards={num_shards} "
            f"batch_size={materialization_batch_size} dtype={tensor_dtype}"
        )
        result = run_materialization_once(
            shard_map=shard_map,
            shard_rel_paths=shard_rel_paths,
            materialized_root=materialized_root,
            num_workers=worker_count,
            materialization_batch_size=materialization_batch_size,
            tensor_dtype=tensor_dtype,
            materialize_speaker_prefix=materialize_speaker_prefix,
        )
        result["materialized_root"] = str(materialized_root)
        results.append(result)
        print(
            "[BENCH_RESULT] "
            f"workers={int(result['requested_workers'])} "
            f"actual_workers={int(result['actual_workers'])} "
            f"shards={int(result['num_shards'])} "
            f"samples={int(result['written_samples'])} "
            f"elapsed={result['elapsed_sec']:.2f}s "
            f"samples_per_sec={result['samples_per_sec']:.2f}"
        )

    payload = {
        "config_path": str(args.config_path),
        "country": str(cfg["dataset"]["country"]),
        "split": str(args.split),
        "workers": workers,
        "fixed_num_shards": args.fixed_num_shards,
        "shards_per_worker": int(args.shards_per_worker),
        "materialization_batch_size": materialization_batch_size,
        "tensor_dtype": str(tensor_dtype),
        "selected_shards": selected_union,
        "results": results,
    }
    results_path = Path(args.results_json).expanduser().resolve() if args.results_json else output_root / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[BENCH] wrote {results_path}")


if __name__ == "__main__":
    main()
