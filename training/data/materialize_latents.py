from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from training.data.types import PairedManifestRow


def _is_empty_path(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "null", "None"}:
        return True
    return False


def load_manifest_rows(manifest_root: Path, manifest_glob: str) -> list[PairedManifestRow]:
    rows: list[PairedManifestRow] = []
    for path in sorted(manifest_root.glob(manifest_glob)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                rows.append(
                    PairedManifestRow(
                        key=str(payload["key"]),
                        country=str(payload.get("country", "unknown_country")),
                        split=str(payload.get("split", "unknown_split")),
                        transcription=str(payload.get("transcription", "")),
                        latent_shard_path=str(payload["latent_shard_path"]),
                        latent_row_idx=int(payload.get("latent_row_idx", 0)),
                        num_frames=None if payload.get("num_frames") is None else int(payload["num_frames"]),
                        speaker_prefix_frames=None
                        if payload.get("speaker_prefix_frames") is None
                        else int(payload["speaker_prefix_frames"]),
                        timestamps=payload.get("timestamps"),
                    )
                )
    if not rows:
        raise RuntimeError(
            f"No paired manifest rows found under {manifest_root} with glob {manifest_glob!r}."
        )
    return rows


def resolve_manifest_path(
    *,
    manifest_root: Path,
    country: str,
    split: str,
) -> Path:
    return manifest_root / "manifests" / country / split / "paired_manifest.jsonl"


def load_split_manifest_rows(
    *,
    manifest_root: Path,
    country: str,
    split: str,
) -> list[PairedManifestRow]:
    manifest_path = resolve_manifest_path(
        manifest_root=manifest_root,
        country=country,
        split=split,
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found for country={country!r} split={split!r}: {manifest_path}"
        )

    rows: list[PairedManifestRow] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                PairedManifestRow(
                    key=str(payload["key"]),
                    country=str(payload.get("country", country)),
                    split=str(payload.get("split", split)),
                    transcription=str(payload.get("transcription", "")),
                    latent_shard_path=str(payload["latent_shard_path"]),
                    latent_row_idx=int(payload.get("latent_row_idx", 0)),
                    num_frames=None if payload.get("num_frames") is None else int(payload["num_frames"]),
                    speaker_prefix_frames=None
                    if payload.get("speaker_prefix_frames") is None
                    else int(payload["speaker_prefix_frames"]),
                    timestamps=payload.get("timestamps"),
                )
            )
    return rows


def resolve_manifest_root(dataset_cfg: dict[str, Any]) -> Path:
    local_root = dataset_cfg.get("local_dataset_root")
    if not _is_empty_path(local_root):
        path = Path(str(local_root)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    repo_id = dataset_cfg.get("repo_id")
    if _is_empty_path(repo_id):
        raise ValueError("dataset.repo_id must be set when local_dataset_root is not provided.")

    manifest_glob = str(dataset_cfg.get("manifest_glob", "manifests/*/*/paired_manifest.jsonl"))
    local_dir = snapshot_download(
        repo_id=str(repo_id),
        repo_type="dataset",
        revision=dataset_cfg.get("revision"),
        allow_patterns=[manifest_glob, "summary.json"],
    )
    return Path(local_dir).resolve()


def _load_tensor_from_bytes(blob: bytes) -> torch.Tensor:
    buffer = io.BytesIO(blob)
    return torch.load(buffer, map_location="cpu")


def _download_shard_if_needed(
    row: PairedManifestRow,
    *,
    dataset_cfg: dict[str, Any],
    parquet_cache_dir: Path,
    dataset_root: Path | None,
) -> Path:
    if dataset_root is not None:
        local_path = (dataset_root / row.latent_shard_path).resolve()
        if local_path.exists():
            return local_path

    repo_id = dataset_cfg.get("repo_id")
    if _is_empty_path(repo_id):
        raise FileNotFoundError(
            f"Latent shard {row.latent_shard_path} was not found locally and dataset.repo_id is unset."
        )
    parquet_cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=str(repo_id),
            repo_type="dataset",
            revision=dataset_cfg.get("revision"),
            filename=row.latent_shard_path,
            local_dir=str(parquet_cache_dir),
            local_dir_use_symlinks=False,
        )
    ).resolve()


def _materialized_sample_path(materialized_root: Path, row: PairedManifestRow) -> Path:
    return materialized_root / row.country / row.split / f"{row.key}.pt"


def _resolve_single_shard(
    latent_shard_path: str,
    *,
    dataset_cfg: dict[str, Any],
    parquet_cache_dir: Path,
    dataset_root: Path | None,
) -> tuple[str, Path]:
    probe_row = PairedManifestRow(
        key="",
        country="",
        split="",
        transcription="",
        latent_shard_path=latent_shard_path,
        latent_row_idx=0,
        num_frames=None,
        speaker_prefix_frames=None,
        timestamps=None,
    )
    resolved = _download_shard_if_needed(
        probe_row,
        dataset_cfg=dataset_cfg,
        parquet_cache_dir=parquet_cache_dir,
        dataset_root=dataset_root,
    )
    return latent_shard_path, resolved


def _materialize_shard_rows(
    *,
    shard_path: Path,
    rows: list[PairedManifestRow],
    materialized_root: Path,
    force_rematerialize: bool,
    materialize_speaker_prefix: bool,
) -> tuple[int, int]:
    table = pq.read_table(shard_path)
    written = 0
    skipped = 0

    for row in rows:
        sample_path = _materialized_sample_path(materialized_root, row)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        if sample_path.exists() and not force_rematerialize:
            skipped += 1
            continue

        record = table.slice(row.latent_row_idx, 1).to_pylist()[0]
        payload = {
            "key": row.key,
            "country": row.country,
            "split": row.split,
            "projected": _load_tensor_from_bytes(record["projected_bytes"]).float(),
            "num_frames": int(row.num_frames if row.num_frames is not None else record["num_frames"]),
            "transcription": row.transcription,
            "timestamps": row.timestamps,
            "latent_shard_path": row.latent_shard_path,
            "latent_row_idx": row.latent_row_idx,
        }
        if materialize_speaker_prefix:
            payload["speaker_prefix_frames"] = int(
                row.speaker_prefix_frames
                if row.speaker_prefix_frames is not None
                else record["speaker_prefix_frames"]
            )
            payload["speaker_prefix_prequant"] = _load_tensor_from_bytes(
                record["speaker_prefix_prequant_bytes"]
            ).float()
        torch.save(payload, sample_path)
        written += 1

    return written, skipped


def materialize_latent_dataset(
    *,
    dataset_cfg: dict[str, Any],
    materialized_root: Path,
    force_rematerialize: bool = False,
    cleanup_parquet_after_materialize: bool = False,
) -> None:
    dataset_root = None
    local_root = dataset_cfg.get("local_dataset_root")
    if not _is_empty_path(local_root):
        dataset_root = Path(str(local_root)).expanduser().resolve()

    manifest_root = resolve_manifest_root(dataset_cfg)
    country = str(dataset_cfg["country"])
    materialize_speaker_prefix = bool(dataset_cfg.get("materialize_speaker_prefix", True))
    materialization_num_workers = max(1, int(dataset_cfg.get("materialization_num_workers", 1)))
    rows: list[PairedManifestRow] = []
    for split in ("train", "validation", "test"):
        try:
            rows.extend(load_split_manifest_rows(manifest_root=manifest_root, country=country, split=split))
        except FileNotFoundError:
            continue

    parquet_cache_dir = materialized_root / "_parquet_cache"
    rows_by_shard: dict[str, list[PairedManifestRow]] = defaultdict(list)
    for row in rows:
        rows_by_shard[row.latent_shard_path].append(row)

    shard_map: dict[str, Path] = {}
    shard_items = sorted(rows_by_shard.items(), key=lambda item: item[0])
    if materialization_num_workers > 1:
        with ThreadPoolExecutor(max_workers=materialization_num_workers) as pool:
            futures = [
                pool.submit(
                    _resolve_single_shard,
                    shard_rel_path,
                    dataset_cfg=dataset_cfg,
                    parquet_cache_dir=parquet_cache_dir,
                    dataset_root=dataset_root,
                )
                for shard_rel_path, _ in shard_items
            ]
            for future in futures:
                shard_rel_path, shard_path = future.result()
                shard_map[shard_rel_path] = shard_path
    else:
        for shard_rel_path, _ in shard_items:
            _, shard_path = _resolve_single_shard(
                shard_rel_path,
                dataset_cfg=dataset_cfg,
                parquet_cache_dir=parquet_cache_dir,
                dataset_root=dataset_root,
            )
            shard_map[shard_rel_path] = shard_path

    written = 0
    skipped = 0
    worker_jobs = [
        {
            "shard_path": shard_map[shard_rel_path],
            "rows": shard_rows,
            "materialized_root": materialized_root,
            "force_rematerialize": force_rematerialize,
            "materialize_speaker_prefix": materialize_speaker_prefix,
        }
        for shard_rel_path, shard_rows in shard_items
    ]
    if materialization_num_workers > 1:
        max_workers = min(materialization_num_workers, len(worker_jobs), max(1, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_materialize_shard_rows, **job) for job in worker_jobs]
            for future in futures:
                shard_written, shard_skipped = future.result()
                written += shard_written
                skipped += shard_skipped
    else:
        for job in worker_jobs:
            shard_written, shard_skipped = _materialize_shard_rows(**job)
            written += shard_written
            skipped += shard_skipped

    if cleanup_parquet_after_materialize and parquet_cache_dir.exists():
        for path in sorted(parquet_cache_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()

    print(
        f"[MATERIALIZE] country={country} shards={len(worker_jobs)} "
        f"written={written} skipped={skipped} workers={materialization_num_workers}"
    )
    return None
