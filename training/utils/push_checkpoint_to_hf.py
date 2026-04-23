from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import safetensors.torch
import torch
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a training checkpoint and run artifacts to Hugging Face Hub.")
    parser.add_argument("--output-dir", required=True, help="Training output directory containing checkpoints.")
    parser.add_argument("--repo-id", required=True, help="Target HF model repo id, e.g. username/suspiria-decoder.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Defaults to latest checkpoint in output-dir.")
    parser.add_argument("--repo-type", default="model", choices=("model", "dataset", "space"))
    parser.add_argument("--private", action="store_true", help="Create the target repo as private if it does not exist.")
    parser.add_argument("--revision", default=None, help="Optional target branch/revision.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Defaults to logged-in token.")
    parser.add_argument("--commit-message", default="Upload training checkpoint")
    parser.add_argument("--upload-dir", default=None, help="Temporary staging directory. Defaults to output-dir/hf_upload.")
    parser.add_argument("--include-optimizer", action="store_true", help="Upload full resumable checkpoint as training_checkpoint.pt.")
    parser.add_argument("--prefer-ema", action="store_true", help="Export ema_model weights when available.")
    parser.add_argument("--weights-name", default="model.safetensors", help="Name of the exported safetensors weight file.")
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-step-(\d+)\.pt$", path.name)
    return -1 if match is None else int(match.group(1))


def resolve_checkpoint(output_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint not in (None, "", "null", "None"):
        path = Path(str(checkpoint)).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = sorted(output_dir.glob("checkpoint-step-*.pt"), key=checkpoint_step)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-step-*.pt files found under {output_dir}")
    return candidates[-1].resolve()


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def stage_checkpoint(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    staging_dir: Path,
    include_optimizer: bool,
    prefer_ema: bool,
    weights_name: str,
) -> None:
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_key = "ema_model" if prefer_ema and "ema_model" in payload else "model"
    safetensors.torch.save_file(payload[state_key], staging_dir / weights_name)

    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "exported_state": state_key,
        "step": int(payload.get("step", checkpoint_step(checkpoint_path))),
        "has_optimizer": "optimizer" in payload,
        "has_scheduler": "scheduler" in payload,
        "has_ema_model": "ema_model" in payload,
    }
    if "metric_name" in payload:
        metadata["metric_name"] = payload["metric_name"]
    if "metric_value" in payload:
        metadata["metric_value"] = payload["metric_value"]
    write_json(staging_dir / "checkpoint_metadata.json", metadata)

    if "config" in payload:
        write_json(staging_dir / "training_config.json", payload["config"])

    if include_optimizer:
        shutil.copy2(checkpoint_path, staging_dir / "training_checkpoint.pt")

    tokenizer_dir = output_dir / "tokenizer"
    if tokenizer_dir.exists():
        shutil.copytree(tokenizer_dir, staging_dir / "tokenizer")

    special_tokens_path = output_dir / "resolved_special_tokens.json"
    if special_tokens_path.exists():
        shutil.copy2(special_tokens_path, staging_dir / "resolved_special_tokens.json")

    readme = "\n".join(
        [
            "---",
            "library_name: pytorch",
            "tags:",
            "- automatic-speech-recognition",
            "- frame-synchronous",
            "---",
            "",
            "# Suspira ASR Decoder Checkpoint",
            "",
            f"- Source checkpoint: `{checkpoint_path.name}`",
            f"- Step: `{metadata['step']}`",
            f"- Exported state: `{state_key}`",
            "",
            "Files:",
            "",
            f"- `{weights_name}`: exported model state dict in safetensors format.",
            "- `training_config.json`: training configuration saved in the checkpoint, when available.",
            "- `checkpoint_metadata.json`: export metadata.",
            "- `tokenizer/`: tokenizer artifacts copied from the training output directory, when available.",
            "- `training_checkpoint.pt`: full resumable checkpoint, only when `--include-optimizer` is used.",
            "",
        ]
    )
    (staging_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Training output directory not found: {output_dir}")

    checkpoint_path = resolve_checkpoint(output_dir, args.checkpoint)
    staging_dir = (
        Path(args.upload_dir).expanduser().resolve()
        if args.upload_dir not in (None, "", "null", "None")
        else output_dir / "hf_upload"
    )

    stage_checkpoint(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        staging_dir=staging_dir,
        include_optimizer=bool(args.include_optimizer),
        prefer_ema=bool(args.prefer_ema),
        weights_name=str(args.weights_name),
    )

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=bool(args.private),
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        folder_path=str(staging_dir),
        commit_message=str(args.commit_message),
    )

    print(f"[HF] uploaded checkpoint={checkpoint_path}")
    print(f"[HF] staging_dir={staging_dir}")
    print(f"[HF] repo={args.repo_id}")


if __name__ == "__main__":
    main()
