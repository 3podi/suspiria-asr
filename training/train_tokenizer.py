from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from training.utils.config import to_plain_dict
from training.utils.tokenizer_training import (
    build_training_tokenizer_cfg,
    maybe_push_tokenizer,
    save_tokenizer_artifacts,
    train_bpe_tokenizer,
    validate_training_compatibility,
)


@hydra.main(version_base=None, config_path="../configs", config_name="tokenizer_training")
def main(cfg: DictConfig) -> None:
    cfg = to_plain_dict(cfg)

    tokenizer = train_bpe_tokenizer(cfg["tokenizer"], cfg["dataset"])
    output_dir = save_tokenizer_artifacts(tokenizer, cfg["output"]["dir"])

    compatibility = validate_training_compatibility(cfg, str(output_dir))
    training_cfg = build_training_tokenizer_cfg(cfg, str(output_dir))

    compatibility_path = output_dir / "training_stack_compatibility.json"
    with compatibility_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **compatibility,
                "training_tokenizer_config": training_cfg,
            },
            f,
            indent=2,
        )

    pushed_revision = maybe_push_tokenizer(
        tokenizer,
        output_dir=output_dir,
        push_cfg=cfg["output"],
    )

    print(f"[TOKENIZER] saved_to={output_dir}")
    print(
        "[TOKENIZER] "
        f"vocab_size={compatibility['vocab_size']} "
        f"bos={compatibility['bos_token_id']} "
        f"eos={compatibility['eos_token_id']} "
        f"pad_wait={compatibility['pad_wait_token_id']} "
        f"word_start={compatibility['word_start_token_id']}"
    )
    if pushed_revision is not None:
        print(f"[TOKENIZER] pushed_to={cfg['output']['repo_id']} revision={pushed_revision}")


if __name__ == "__main__":
    main()
