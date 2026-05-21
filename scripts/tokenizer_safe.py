"""Sanitize tokenizer_config.json before loading via AutoTokenizer.

Some validator-anonymized models ship with `extra_special_tokens` as a list,
which crashes newer transformers versions that expect a dict (their
`_set_model_specific_special_tokens` does `list(special_tokens.keys())`).

This helper auto-fixes the file in-place then loads. Idempotent — safe to
call multiple times on the same path.

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

    # Fix: extra_special_tokens must be a dict, not list
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
        print(f"[tokenizer_safe] fixed extra_special_tokens (list→dict) in {cfg_path}",
              flush=True)

    if changed:
        try:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[tokenizer_safe] WARN failed to rewrite {cfg_path}: {e}", flush=True)
            return False
    return changed


def safe_load_tokenizer(pretrained_model_name_or_path: str, **kwargs):
    """AutoTokenizer.from_pretrained wrapper that sanitizes local model dirs first."""
    sanitize_tokenizer_config(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
