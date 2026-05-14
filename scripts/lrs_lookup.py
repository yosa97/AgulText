import json 
import os 
import hashlib
current_dir = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(current_dir, "lrs/dpo.json"), "r") as f:
    dpo_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo.json"), "r") as f:
    grpo_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/instruct.json"), "r") as f:
    instruct_lrs = json.load(f)

with open(os.path.join(current_dir, "lrs/grpo_python.json"), "r") as f:
    grpo_python_lrs = json.load(f)


def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 


def get_dpo_lr(model: str):
    hashed_model = hash_model(model)
    for lr in dpo_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def get_grpo_lr(model: str):
    hashed_model = hash_model(model)
    for lr in grpo_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None

def get_instruct_lr(model: str):
    hashed_model = hash_model(model)
    for lr in instruct_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def get_grpo_python_lr(model: str):
    hashed_model = hash_model(model)
    for lr in grpo_python_lrs:
        if lr["h"] == hashed_model:
            return lr["lr"]
    return None


def read_csv_ar(path: str):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 0]
    result = []
    for line in lines[1: ]: # Skip the header
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
    # find the closest config from INSTRUCT_AR_CONFIG
    filtered_configs = [config for config in list_config if config["ar"] == architecture.lower()]
    if len(filtered_configs) == 0:
        return None
    closest_config = min(filtered_configs, key=lambda x: abs(x["size"] - param_nums))
    print(f"Using lr from ar: {closest_config['lr']} for architecture: {architecture} and size: {param_nums}", flush=True)
    return closest_config["lr"]


def get_lr_from_ar_dpo(architecture: str, param_nums: int):
    return get_lr_from_ar(architecture.lower().strip(), param_nums, DPO_AR_CONFIG)


def get_lr_from_ar_instruct(architecture: str, param_nums: int):
    return get_lr_from_ar(architecture.lower().strip(), param_nums, INSTRUCT_AR_CONFIG)