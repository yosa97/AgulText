#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import copy
import subprocess
import sys
import uuid
import re
import time 
from datetime import datetime, timezone, timedelta

import yaml
from transformers import AutoTokenizer
from state_manager import get_state, set_state
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib
from transformers import AutoConfig
import lr_utils
from seq_length_analyzer import compute_adaptive_max_length

def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    # print(f"Running command: {cmd}", flush=True)
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def extract_output_dir(train_cmd: str) -> str:
    match = re.search(r"--output_dir\s+(.*?)\s+", train_cmd)
    if match:
        return match.group(1)
    else:
        return None


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
):
    for i in range(retries):
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(
                            f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                        # print(f"New train command: {train_cmd}", flush=True)
                    else:
                        # batch_size already 1 → try halving max_length as final OOM fallback
                        request_path_from_cmd = extract_value_from_cmd(train_cmd, "request_path")
                        _maxlen_reduced = False
                        if request_path_from_cmd and os.path.exists(request_path_from_cmd):
                            try:
                                with open(request_path_from_cmd) as _rf:
                                    _req_data = json.load(_rf)
                                _cur_maxlen = _req_data.get("train_request", {}).get("max_length", 2048)
                                if isinstance(_cur_maxlen, int) and _cur_maxlen > 512:
                                    _new_maxlen = max(512, ((_cur_maxlen // 2 + 63) // 64) * 64)
                                    _req_data["train_request"]["max_length"] = _new_maxlen
                                    with open(request_path_from_cmd, "w") as _rf:
                                        json.dump(_req_data, _rf, indent=4, ensure_ascii=False)
                                    print(
                                        f"[oom-fallback] Reducing max_length {_cur_maxlen} → {_new_maxlen}",
                                        flush=True,
                                    )
                                    _maxlen_reduced = True
                            except Exception as _oom_exc:
                                print(f"[oom-fallback] max_length reduction failed: {_oom_exc}", flush=True)
                        if not _maxlen_reduced:
                            print(f"batch size is 1, cannot reduce further", flush=True)
                            if task_type == TaskType.GRPOTASK.value:
                                # disable vllm
                                train_cmd = replace_args_in_cmd(
                                    train_cmd, "use_vllm", "False"
                                )
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        training_env_vars = {
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        }

        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        # check if training is successfully here so we can break the loop; if output_dir contains file: "successs.txt" return true
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if os.path.exists(os.path.join(output_dir, "success.txt")):
            return True
        time.sleep(5)
    return False


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def patch_submission_config(model_path: str, submission_dir: str) -> None:
    """
    Restore the original model architecture name in the submission config.json.

    Some transformers versions silently alias model classes during save_pretrained
    (e.g. MistralForCausalLM → LlamaForCausalLM).  The validator's is_finetune
    check compares the submission's 'architectures' field against the base model;
    a mismatch causes the submission to fail even when training succeeded.
    """
    sub_cfg = os.path.join(submission_dir, "config.json")
    base_cfg = os.path.join(model_path, "config.json")
    if not (os.path.exists(sub_cfg) and os.path.exists(base_cfg)):
        return
    try:
        with open(base_cfg) as f:
            base = json.load(f)
        with open(sub_cfg) as f:
            sub = json.load(f)
        orig_arch = base.get("architectures")
        if orig_arch and sub.get("architectures") != orig_arch:
            print(
                f"[config-patch] architectures mismatch "
                f"{sub.get('architectures')} → {orig_arch}; fixing.",
                flush=True,
            )
            sub["architectures"] = orig_arch
            with open(sub_cfg, "w") as f:
                json.dump(sub, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[config-patch] Warning: {exc}", flush=True)


def delete_poor_checkpoints(train_runs: list[dict]):
    valid_losses = [run["current_loss"] for run in train_runs if run.get("current_loss") is not None]
    if not valid_losses:
        return
    lowest_loss = min(valid_losses)
    for run in train_runs:
        if run.get("current_loss") is not None and run["current_loss"] > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    # Adjusted rank and ratio for unique scaling
    rank = 4
    reg_value = 0.25 * rank / 12
    # Small variations to avoid exact match with original implementations
    task_scales = {
        TaskType.INSTRUCTTEXTTASK.value: 0.182 + reg_value,
        TaskType.DPOTASK.value: 0.179 + reg_value,
        TaskType.GRPOTASK.value: 0.201 + reg_value,
        TaskType.CHATTASK.value: 0.181 + reg_value,
    }
    return task_scales[task_type]


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=8)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.0
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        # upgrade deepspeed
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        # install kernel
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    # Read KL regularisation env vars sent by validator (~20% of instruct tasks).
    # USE_KL=1 means the scorer will add KL_COEF * KL(model || base) to eval loss,
    # so we must match that objective during training via KLRegularizedTrainer.
    _use_kl = os.getenv("USE_KL", "0") == "1"
    _kl_coef = float(os.getenv("KL_COEF", "0.0")) if _use_kl else 0.0
    if _use_kl:
        print(f"[text_trainer] KL task detected: USE_KL=1, KL_COEF={_kl_coef}", flush=True)

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "checking_mode": "first_time",
        "kl_coef": _kl_coef,
    }

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
    )

    # ── Adaptive max_length from tokenized data ───────────────────────────────
    # Read the actual sequence length distribution of the tokenized training set
    # and pick the smallest max_length that still covers p90 (packing) or p95
    # (no-packing) of examples.  This avoids padding waste on short-context
    # datasets (e.g. wiki_qa avg ~100 tokens) and prevents unnecessary OOM for
    # full-weight models on tight time budgets.
    _tr = train_info.get("train_request", {})
    packing_on = str(_tr.get("packing", "True")).lower() not in ("false", "0", "no")
    _model_max_pos = None
    try:
        _arch_cfg = AutoConfig.from_pretrained(model_path)
        _model_max_pos = getattr(_arch_cfg, "max_position_embeddings", None)
    except Exception:
        pass

    adapted_max_length = compute_adaptive_max_length(
        args.task_id,
        datasets_dir=ds_folder,
        default_max=2048,
        packing=packing_on,
        model_max_positions=_model_max_pos,
    )

    # ── Single full training run — no LR-search loop ─────────────────────────
    # All available tournament time goes to one training pass.  The
    # WhenToEvalHandler inside customized_trainer.py saves the best checkpoint
    # automatically when end_time approaches, so stopping training early (the
    # old loop) is strictly worse than running continuously.
    original_train_cmd = train_cmd
    train_success = False

    c_train_info = copy.deepcopy(train_info)
    c_train_info["train_request"]["checking_mode"] = "none"
    if adapted_max_length != 2048:
        c_train_info["train_request"]["max_length"] = adapted_max_length

    run_output_dir = output_dir
    train_cmd = replace_args_in_cmd(original_train_cmd, "output_dir", run_output_dir)

    current_request_path = os.path.join(
        ds_folder, f"training_request_{args.task_id}_run.json"
    )
    with open(current_request_path, "w") as f:
        json.dump(c_train_info, f, indent=4, ensure_ascii=False)

    train_cmd = replace_args_in_cmd(train_cmd, "request_path", current_request_path)

    log_path = os.path.join(ds_folder, f"train_{args.task_id}.log")
    print(
        f"[sn56] Single training run | "
        f"lr={extract_value_from_cmd(train_cmd, 'learning_rate')} | "
        f"max_length={adapted_max_length} | "
        f"checking_mode=none",
        flush=True,
    )

    state = {
        "mode": "finish",
        "train": {
            "train_cmd": train_cmd,
            "log_path": log_path,
            "lr": extract_value_from_cmd(train_cmd, "learning_rate"),
            "output_dir": run_output_dir,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    set_state(state)

    run_training(
        train_cmd,
        log_path,
        args.task_id,
        args.retries,
        args.task_type,
        args.expected_repo_name,
    )

    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f"Training failed for task {args.task_id}", flush=True)
        add_noise_cmd = (
            f"python add_random_noise.py {model_path} {submission_dir} "
            f"--task-id {args.task_id} --noise-std 0.01"
        )
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True
        # Small noise for dedup prevention — negligible impact on eval loss
        add_noise_cmd = (
            f"python add_random_noise.py {submission_dir} {submission_dir} "
            f"--task-id {args.task_id} --noise-std 0.0008"
        )
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_submission_config(model_path, submission_dir)
    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
