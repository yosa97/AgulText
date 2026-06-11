import json
import os
import hashlib
import functools

current_dir = os.path.dirname(os.path.abspath(__file__))


def _build_lr_index(records: list) -> dict:
    """Build a hash→lr mapping for O(1) model lookup instead of linear scan."""
    return {rec["h"]: rec["lr"] for rec in records}


with open(os.path.join(current_dir, "lrs/dpo.json"), "r") as f:
    _dpo_lrs_raw = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo.json"), "r") as f:
    _grpo_lrs_raw = json.load(f)

with open(os.path.join(current_dir, "lrs/instruct.json"), "r") as f:
    _instruct_lrs_raw = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo_python.json"), "r") as f:
    _grpo_python_lrs_raw = json.load(f)

# Build O(1) lookup indexes
_dpo_index = _build_lr_index(_dpo_lrs_raw)
_grpo_index = _build_lr_index(_grpo_lrs_raw)
_instruct_index = _build_lr_index(_instruct_lrs_raw)
_grpo_python_index = _build_lr_index(_grpo_python_lrs_raw)

# Keep raw lists accessible (backward compat)
dpo_lrs = _dpo_lrs_raw
grpo_lrs = _grpo_lrs_raw
instruct_lrs = _instruct_lrs_raw
grpo_python_lrs = _grpo_python_lrs_raw


@functools.lru_cache(maxsize=512)
def hash_model(model: str) -> str:
    """SHA-256 hash of model name string, cached to avoid repeat computation."""
    return hashlib.sha256(model.encode("utf-8")).hexdigest()


def get_dpo_lr(model: str):
    return _dpo_index.get(hash_model(model))


def get_grpo_lr(model: str):
    return _grpo_index.get(hash_model(model))


def get_instruct_lr(model: str):
    return _instruct_index.get(hash_model(model))


def get_grpo_python_lr(model: str):
    return _grpo_python_index.get(hash_model(model))


def read_csv_ar(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 0]
    result = []
    for line in lines[1:]:  # Skip the header row
        parts = line.split(",")
        result.append({
            "size": int(parts[0]),
            "ar": parts[1].strip().lower(),
            "lr": float(parts[2])
        })
    return result


INSTRUCT_AR_CONFIG = read_csv_ar(os.path.join(current_dir, "lrs/instruct_ar.csv"))
DPO_AR_CONFIG = read_csv_ar(os.path.join(current_dir, "lrs/dpo_ar.csv"))


def get_lr_from_ar(architecture: str, param_nums: int, list_config: list):
    """Find the closest-size entry for the given architecture."""
    filtered_configs = [c for c in list_config if c["ar"] == architecture.lower()]
    if not filtered_configs:
        return None
    closest_config = min(filtered_configs, key=lambda c: abs(c["size"] - param_nums))
    print(f"Using lr from ar: {closest_config['lr']} for architecture: {architecture} and size: {param_nums}", flush=True)
    return closest_config["lr"]


def get_lr_from_ar_dpo(architecture: str, param_nums: int):
    return get_lr_from_ar(architecture.lower().strip(), param_nums, DPO_AR_CONFIG)


def get_lr_from_ar_instruct(architecture: str, param_nums: int):
    return get_lr_from_ar(architecture.lower().strip(), param_nums, INSTRUCT_AR_CONFIG)