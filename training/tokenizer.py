from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer


@dataclass(frozen=True)
class ResolvedTokenizer:
    tokenizer: Any
    bos_token_id: int
    eos_token_id: int
    pad_wait_token_id: int
    word_start_token_id: int


def load_tokenizer(tokenizer_cfg: dict[str, Any]) -> ResolvedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["name"],
        revision=tokenizer_cfg.get("revision"),
        use_fast=bool(tokenizer_cfg.get("use_fast", True)),
    )

    extra_specials = list(tokenizer_cfg.get("additional_special_tokens", ["[P]", "[W]"]))
    additions = []
    for token_name in ("bos_token", "eos_token"):
        token_value = tokenizer_cfg.get(token_name)
        if token_value and getattr(tokenizer, token_name, None) is None:
            setattr(tokenizer, token_name, token_value)
            additions.append(token_name)

    existing = set(tokenizer.additional_special_tokens or [])
    missing = [tok for tok in extra_specials if tok not in existing]
    if missing or additions:
        tokenizer.add_special_tokens({"additional_special_tokens": list(existing) + missing})

    bos_token = tokenizer_cfg.get("bos_token", tokenizer.bos_token or "[BOS]")
    eos_token = tokenizer_cfg.get("eos_token", tokenizer.eos_token or "[EOS]")
    if tokenizer.bos_token is None or tokenizer.bos_token != bos_token:
        tokenizer.add_special_tokens({"bos_token": bos_token})
    if tokenizer.eos_token is None or tokenizer.eos_token != eos_token:
        tokenizer.add_special_tokens({"eos_token": eos_token})

    pad_wait = tokenizer_cfg.get("pad_wait_token", "[P]")
    word_start = tokenizer_cfg.get("word_start_token", "[W]")
    if pad_wait not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": list(tokenizer.additional_special_tokens) + [pad_wait]})
    if word_start not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": list(tokenizer.additional_special_tokens) + [word_start]})

    return ResolvedTokenizer(
        tokenizer=tokenizer,
        bos_token_id=int(tokenizer.convert_tokens_to_ids(bos_token)),
        eos_token_id=int(tokenizer.convert_tokens_to_ids(eos_token)),
        pad_wait_token_id=int(tokenizer.convert_tokens_to_ids(pad_wait)),
        word_start_token_id=int(tokenizer.convert_tokens_to_ids(word_start)),
    )

