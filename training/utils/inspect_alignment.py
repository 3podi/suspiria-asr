from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.alignment import build_delayed_target_stream
from training.data.collator import SpecialTokenIds
from training.data.dataset import MaterializedLatentDataset
from training.data.materialize_latents import resolve_manifest_root
from training.tokenizer import load_tokenizer
from training.utils.data import ensure_materialized_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect frame-synchronous token/latent alignment for one materialized sample."
    )
    parser.add_argument("--config-path", default="configs/training.yaml")
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument("--country", default=None, help="Override dataset.country from config.")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the selected split.")
    parser.add_argument("--key", default=None, help="Sample key to inspect. Overrides --index when set.")
    parser.add_argument("--delay-ms", type=int, default=None, help="Fixed delay in ms. Defaults to dataset.delay_max_ms.")
    parser.add_argument("--max-steps", type=int, default=None, help="Only print the first N aligned steps.")
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=100,
        help="Number of samples from the selected split to aggregate target percentages over.",
    )
    parser.add_argument("--output-path", default=None, help="Optional path to write the Markdown report.")
    return parser.parse_args()


def load_cfg(config_path: str) -> dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError("Expected training config to resolve to a mapping.")
    return plain


def find_sample_index(dataset: MaterializedLatentDataset, key: str) -> int:
    for idx, sample in enumerate(dataset.samples):
        if sample.key == key:
            return idx
    raise KeyError(f"Sample key {key!r} not found in split.")


def token_kind(token_id: int, special_tokens: SpecialTokenIds) -> str:
    if token_id == special_tokens.bos:
        return "BOS"
    if token_id == special_tokens.eos:
        return "EOS"
    if token_id == special_tokens.pad_wait:
        return "P"
    if token_id == special_tokens.word_start:
        return "W"
    return "TEXT"


def render_token(token_id: int, tokenizer, special_tokens: SpecialTokenIds) -> str:
    kind = token_kind(token_id, special_tokens)
    if kind != "TEXT":
        return f"[{kind}]" if kind not in {"P", "W"} else f"[{kind}]"
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    decoded = decoded.replace("\n", "\\n").replace("|", "\\|")
    return f"{token} / {decoded!r}"


def audio_region(step: int, *, left_pad_steps: int, real_steps: int) -> tuple[str, str]:
    latent_idx = step - left_pad_steps
    if step < left_pad_steps:
        return "left_pad", "-"
    if 0 <= latent_idx < real_steps:
        return "real", str(latent_idx)
    return "right_pad", "-"


def summarize_targets_after_left_padding(
    labels: list[int],
    *,
    left_pad_steps: int,
    special_tokens: SpecialTokenIds,
) -> dict[str, float]:
    counted = labels[left_pad_steps:]
    total = len(counted)
    if total == 0:
        return {
            "counted_steps": 0.0,
            "text_or_w_count": 0.0,
            "text_or_w_pct": float("nan"),
            "pad_count": 0.0,
            "pad_pct": float("nan"),
        }

    text_or_w = 0
    pad = 0
    for token_id in counted:
        kind = token_kind(int(token_id), special_tokens)
        if kind in {"TEXT", "W"}:
            text_or_w += 1
        elif kind == "P":
            pad += 1

    return {
        "counted_steps": float(total),
        "text_or_w_count": float(text_or_w),
        "text_or_w_pct": 100.0 * float(text_or_w) / float(total),
        "pad_count": float(pad),
        "pad_pct": 100.0 * float(pad) / float(total),
    }


def align_sample(
    *,
    cfg: dict[str, Any],
    sample: dict[str, Any],
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
):
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    if delay_ms % step_ms != 0:
        raise ValueError(f"--delay-ms must be a multiple of step_ms={step_ms}.")
    return build_delayed_target_stream(
        key=str(sample["key"]),
        latents=sample["projected"],
        transcript=str(sample["transcription"]),
        timestamps=sample.get("timestamps"),
        tokenizer=tokenizer,
        bos_token_id=special_tokens.bos,
        eos_token_id=special_tokens.eos,
        pad_wait_token_id=special_tokens.pad_wait,
        word_start_token_id=special_tokens.word_start,
        delay_steps=delay_ms // step_ms,
        left_pad_steps=int(cfg["dataset"].get("left_pad_steps", 0)),
        step_ms=step_ms,
    )


def aggregate_target_summary(
    *,
    cfg: dict[str, Any],
    dataset: MaterializedLatentDataset,
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
    num_samples: int,
) -> dict[str, float]:
    left_pad_steps = int(cfg["dataset"].get("left_pad_steps", 0))
    limit = min(max(0, int(num_samples)), len(dataset))
    counted_steps = 0
    text_or_w_count = 0
    pad_count = 0

    for sample_idx in range(limit):
        aligned = align_sample(
            cfg=cfg,
            sample=dataset[sample_idx],
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            delay_ms=delay_ms,
        )
        for token_id in aligned.labels.tolist()[left_pad_steps:]:
            counted_steps += 1
            kind = token_kind(int(token_id), special_tokens)
            if kind in {"TEXT", "W"}:
                text_or_w_count += 1
            elif kind == "P":
                pad_count += 1

    return {
        "num_samples": float(limit),
        "counted_steps": float(counted_steps),
        "text_or_w_count": float(text_or_w_count),
        "text_or_w_pct": float("nan") if counted_steps == 0 else 100.0 * text_or_w_count / counted_steps,
        "pad_count": float(pad_count),
        "pad_pct": float("nan") if counted_steps == 0 else 100.0 * pad_count / counted_steps,
    }


def render_aggregate_summary(summary: dict[str, float]) -> str:
    return "\n".join(
        [
            "## Aggregate Target Summary",
            "",
            "Counts exclude the left-padding region.",
            "",
            f"- summary_samples: `{int(summary['num_samples'])}`",
            f"- counted_steps: `{int(summary['counted_steps'])}`",
            f"- text_or_w_count: `{int(summary['text_or_w_count'])}`",
            f"- text_or_w_pct: `{summary['text_or_w_pct']:.2f}%`",
            f"- pad_count: `{int(summary['pad_count'])}`",
            f"- pad_pct: `{summary['pad_pct']:.2f}%`",
            "",
        ]
    )


def build_report(
    *,
    cfg: dict[str, Any],
    split: str,
    sample_index: int,
    sample: dict[str, Any],
    tokenizer,
    special_tokens: SpecialTokenIds,
    delay_ms: int,
    max_steps: int | None,
) -> str:
    step_ms = int(cfg["dataset"].get("step_ms", 80))
    if delay_ms % step_ms != 0:
        raise ValueError(f"--delay-ms must be a multiple of step_ms={step_ms}.")
    delay_steps = delay_ms // step_ms
    left_pad_steps = int(cfg["dataset"].get("left_pad_steps", 0))
    latents = sample["projected"]
    real_steps = int(latents.shape[0])

    aligned = align_sample(
        cfg=cfg,
        sample=sample,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
    )

    labels = aligned.labels.tolist()
    input_ids = aligned.input_ids.tolist()
    num_steps = len(labels)
    shown_steps = num_steps if max_steps is None else min(num_steps, int(max_steps))
    target_summary = summarize_targets_after_left_padding(
        labels,
        left_pad_steps=left_pad_steps,
        special_tokens=special_tokens,
    )

    lines = [
        "# Alignment Inspection",
        "",
        f"- key: `{sample['key']}`",
        f"- split: `{split}`",
        f"- sample_index: `{sample_index}`",
        f"- real_steps: `{real_steps}`",
        f"- left_pad_steps: `{left_pad_steps}`",
        f"- delay_ms: `{delay_ms}`",
        f"- delay_steps: `{delay_steps}`",
        f"- aligned_steps: `{num_steps}`",
        f"- shown_steps: `{shown_steps}`",
        "",
        "## Target Summary",
        "",
        "Counts exclude the left-padding region.",
        "",
        f"- counted_steps: `{int(target_summary['counted_steps'])}`",
        f"- text_or_w_count: `{int(target_summary['text_or_w_count'])}`",
        f"- text_or_w_pct: `{target_summary['text_or_w_pct']:.2f}%`",
        f"- pad_count: `{int(target_summary['pad_count'])}`",
        f"- pad_pct: `{target_summary['pad_pct']:.2f}%`",
        "",
        "## Transcript",
        "",
        str(sample["transcription"]),
        "",
        "## Word Timestamps",
        "",
    ]

    timestamps = sample.get("timestamps") or []
    if timestamps:
        lines.extend(
            f"- `{item.get('start', '?')}`-`{item.get('end', '?')}`: {item.get('text', '')}"
            for item in timestamps
        )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Aligned Steps",
            "",
            "| step | ms | audio | latent_idx | input_kind | input_token | target_kind | target_token |",
            "|---:|---:|---|---:|---|---|---|---|",
        ]
    )

    for step in range(shown_steps):
        region, latent_idx = audio_region(step, left_pad_steps=left_pad_steps, real_steps=real_steps)
        input_id = int(input_ids[step])
        target_id = int(labels[step])
        lines.append(
            "| "
            f"{step} | "
            f"{step * step_ms} | "
            f"{region} | "
            f"{latent_idx} | "
            f"{token_kind(input_id, special_tokens)} | "
            f"{render_token(input_id, tokenizer, special_tokens)} | "
            f"{token_kind(target_id, special_tokens)} | "
            f"{render_token(target_id, tokenizer, special_tokens)} |"
        )

    if shown_steps < num_steps:
        lines.extend(["", f"_Truncated: {num_steps - shown_steps} additional steps not shown._"])

    if len(aligned.input_ids) != len(aligned.labels) or len(aligned.labels) != len(aligned.audio_features):
        raise RuntimeError("Alignment invariant failed: input_ids, labels and audio_features lengths differ.")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config_path)
    if args.country is not None:
        cfg["dataset"]["country"] = args.country

    resolved_tokenizer = load_tokenizer(cfg["tokenizer"])
    tokenizer = resolved_tokenizer.tokenizer
    special_tokens = SpecialTokenIds(
        bos=resolved_tokenizer.bos_token_id,
        eos=resolved_tokenizer.eos_token_id,
        pad_wait=resolved_tokenizer.pad_wait_token_id,
        word_start=resolved_tokenizer.word_start_token_id,
    )

    materialized_root = ensure_materialized_dataset(cfg)
    manifest_root = resolve_manifest_root(cfg["dataset"])
    dataset = MaterializedLatentDataset(
        manifest_root=manifest_root,
        materialized_root=materialized_root,
        split=args.split,
        country=str(cfg["dataset"]["country"]),
    )

    sample_index = find_sample_index(dataset, args.key) if args.key else int(args.index)
    sample = dataset[sample_index]
    delay_ms = int(args.delay_ms if args.delay_ms is not None else cfg["dataset"].get("delay_max_ms", 2400))
    aggregate_summary = aggregate_target_summary(
        cfg=cfg,
        dataset=dataset,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
        num_samples=args.summary_samples,
    )

    report = build_report(
        cfg=cfg,
        split=args.split,
        sample_index=sample_index,
        sample=sample,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        delay_ms=delay_ms,
        max_steps=args.max_steps,
    )
    report = report + "\n" + render_aggregate_summary(aggregate_summary)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[INSPECT] wrote {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
