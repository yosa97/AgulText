"""Sanitize tokenizer_config.json before loading via AutoTokenizer.

Known validator-anonymised model issues fixed here:
  1. ``extra_special_tokens`` shipped as a list instead of a dict —
     crashes newer transformers in ``_set_model_specific_special_tokens``.
  2. ``added_tokens_decoder`` values that are plain strings instead of
     the expected ``{"content": ..., "single_word": ..., ...}`` dicts.

All fixes are idempotent (safe to call multiple times on the same path).

Usage:
    from tokenizer_safe import safe_load_tokenizer
    tok = safe_load_tokenizer(model_path, trust_remote_code=True)
"""

from __future__ import annotations

import json
import os

from transformers import AutoTokenizer


def sanitize_tokenizer_config(model_path: str) -> bool:
    """Fix known malformed fields in tokenizer_config.json. Returns True if file changed."""
    if not isinstance(model_path, str) or not os.path.isdir(model_path):
        return False
    cfg_path = os.path.join(model_path, "tokenizer_config.json")
    if not os.path.exists(cfg_path):
        return False
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[tokenizer_safe] WARN failed to read {cfg_path}: {e}", flush=True)
        return False

    changed = False

    # Fix 1: extra_special_tokens must be a dict, not list
    est = cfg.get("extra_special_tokens")
    if isinstance(est, list):
        if all(isinstance(x, str) for x in est):
            cfg["extra_special_tokens"] = {x: x for x in est}
        elif all(isinstance(x, dict) for x in est):
            merged: dict = {}
            for d in est:
                merged.update(d)
            cfg["extra_special_tokens"] = merged
        else:
            cfg["extra_special_tokens"] = {}
        changed = True
        print(f"[tokenizer_safe] fixed extra_special_tokens (list→dict) in {cfg_path}", flush=True)

    # Fix 2: added_tokens_decoder values must be dicts, not bare strings
    atd = cfg.get("added_tokens_decoder")
    if isinstance(atd, dict):
        repaired: dict = {}
        any_repaired = False
        for tok_id, tok_val in atd.items():
            if isinstance(tok_val, str):
                repaired[tok_id] = {
                    "content": tok_val,
                    "single_word": False,
                    "lstrip": False,
                    "rstrip": False,
                    "normalized": False,
                    "special": True,
                }
                any_repaired = True
            else:
                repaired[tok_id] = tok_val
        if any_repaired:
            cfg["added_tokens_decoder"] = repaired
            changed = True
            print(f"[tokenizer_safe] fixed added_tokens_decoder (str→dict) in {cfg_path}", flush=True)

    if changed:
        try:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[tokenizer_safe] WARN failed to rewrite {cfg_path}: {e}", flush=True)
            return False
    return changed


def verify_tokenizer_health(tokenizer) -> list[str]:
    """Return a list of warning strings for common tokenizer misconfigurations.

    Call after loading to surface issues early rather than in the middle of
    training.  An empty list means no issues were found.
    """
    issues: list[str] = []

    if tokenizer.pad_token is None:
        issues.append("pad_token is None — will fall back to eos_token for padding")
    if tokenizer.eos_token is None:
        issues.append("eos_token is None — generation may not terminate properly")
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id:
        issues.append(
            "pad_token_id == eos_token_id — can cause label leakage in causal-LM training"
        )

    for issue in issues:
        print(f"[tokenizer_safe] HEALTH: {issue}", flush=True)

    return issues


def safe_load_tokenizer(pretrained_model_name_or_path: str, **kwargs):
    """AutoTokenizer.from_pretrained wrapper that sanitizes local model dirs first."""
    sanitize_tokenizer_config(pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
    verify_tokenizer_health(tokenizer)
    return tokenizer
