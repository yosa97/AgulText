from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
from state_manager import get_state, set_state
MAX_TRIES = 9


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)
    
class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none"
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self._end_time_fired = False   # set True when end_time trigger fires
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        # Adaptive eval timing: adjusted once after first eval completes
        self._eval_timing_adjusted = False
        # Cache original model config for architecture patching
        self._original_config = None
        _base_cfg_path = os.path.join(original_model_name, "config.json")
        if os.path.exists(_base_cfg_path):
            try:
                with open(_base_cfg_path) as _f:
                    self._original_config = json.load(_f)
            except Exception:
                pass

    def _patch_submission_architectures(self):
        """Patch architectures in submission config.json to match the base model.

        Some transformers versions alias model classes on load (e.g. MistralForCausalLM
        → LlamaForCausalLM). When save_pretrained writes config.json it uses the runtime
        class, which may differ from the original. The validator's is_finetune check
        compares architectures and silently fails on mismatch — submission is rejected
        even though the model is technically fine. This restores the original value.
        """
        if not self._original_config:
            return
        cfg_path = os.path.join(self.submission_dir, "config.json")
        if not os.path.exists(cfg_path):
            return
        orig_arch = self._original_config.get("architectures")
        if not orig_arch:
            return
        try:
            with open(cfg_path) as _f:
                cfg = json.load(_f)
            if cfg.get("architectures") != orig_arch:
                print(
                    f"[config-patch] architectures mismatch — "
                    f"restoring {cfg.get('architectures')} → {orig_arch}",
                    flush=True,
                )
                cfg["architectures"] = orig_arch
                with open(cfg_path, "w") as _f:
                    json.dump(cfg, _f, indent=2)
        except Exception as _e:
            print(f"[config-patch] WARNING: {_e}", flush=True)

    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Custom logic to decide whether to save or evaluate
        # print(f"************* on_step_end: {state.global_step}, check eval", flush=True)
        # TODO: implement the logic to save the model without evaluating if there is no check points --> avoid evaluating takes too much time
        # Check if the checking_step is reached
        # print(f"Checking the model at step: {state.global_step}, checking_step: {self.checking_step}, checking_mode: {self.checking_mode}", flush=True)
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            # print(f"Checking the model at step: {state.global_step}", flush=True)
            # check the time so far to estimate the training time in total 
            my_state = get_state()
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far: {time_so_far}"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\nTime for one step: {time_for_one_step}"
            # Now estimate the total training time for this training
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\nTotal remaining training time: {total_remaining_training_time}"
            # n * time_so_far + total_remaining_training_time = total_remaining_time
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time}"
            
            # n * time_so_far + (time_so_far + total_remaining_training_time) = total_remaining_time
            # time_so_far + total_remaining_training_time is the time it takes to finish the training (need to estimate the eval time and save time, assuming this is 15 minutes)
            # assuming time_so_far is + 5 minutes, just in case the checking step takes more time than expected
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar) # 300 = 5 minutes, assume that it extra time would be more or less 5 minutes
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0: # we should try more 
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = "no"
                # save the current loss of this step to the state;
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1 # including the current run
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time": # at second time, we don't estimate the training time again, just save the current_loss
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"            
            my_state = get_state()
            current_loss = None
            for log in reversed(state.log_history):
                if "loss" in log:
                    current_loss = log["loss"]
                    break
            
            if current_loss is None:
                current_loss = float('inf')
            my_state["train"]["current_loss"] = current_loss
                
            control.should_training_stop = True

            # Check if current_loss > current min_loss --> do not save to save time and space
            # 
            # if my_state["train"]["current_loss"] > current_min_loss:
            #     print(f"Current loss: {my_state['train']['current_loss']} is greater than the current min_loss: {current_min_loss}, do not save the checkpoint", flush=True)
            #     control.should_save = False
            # check if this is the last run and the current_loss is the lowest --> keep running the training
            current_is_the_best = False
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"Current loss: {my_state['train']['current_loss']} is greater than: {current_min_loss}", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                # print(log_content, flush=True)
        
            
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            # do not allow the pod to be stopped by any reason
                # first check if there is at least one checkpoint or not
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                control.should_training_stop = True   # stop after save completes
                self._end_time_fired = True
                if not self.has_checkpoint: # if there is no checkpoint, we just save the model, do not evaluate
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True

        # Skip evals before the model has trained 75% of one epoch.
        # Overfitting (and thus meaningful eval signal) is not possible yet.
        # Exception: end_time trigger is one-shot and must never be skipped.
        if (
            control.should_evaluate
            and when_to_eval["reason"] != "end_time"
            and self.total_steps_all_epochs > 0
            and args.num_train_epochs > 1
        ):
            steps_per_epoch = self.total_steps_all_epochs / max(1, args.num_train_epochs)
            if state.global_step < int(0.75 * steps_per_epoch):
                print(
                    f"[eval-skip] step={state.global_step} < 0.75*epoch={int(0.75 * steps_per_epoch)}, skipping eval",
                    flush=True,
                )
                control.should_evaluate = False
                control.should_save = False

        return control


    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        eval_loss = self.compute_loss(state, metrics)
        if eval_loss is None:
            print(f"WARNING: eval_loss is None at step: {state.global_step}, skipping best checkpoint update.", flush=True)
            return
        if state.global_step < 2:
            return 
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        if self.best_checkpoint_info is None or eval_loss < self.best_checkpoint_info["loss"]:
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f" At step: {state.global_step} The eval_loss: {eval_loss} is not smaller than the current best eval_loss: {self.best_checkpoint_info['loss']}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)
            self.update_best_checkpoint = False  # Reset flag — jangan biarkan stale True dari eval sebelumnya

        # Adaptive eval interval: setelah eval pertama selesai, ukur runtime-nya
        # dan sesuaikan eval_steps agar eval tidak memakan lebih dari 10% sisa waktu.
        # Hanya diperlebar (tidak dipersempit) untuk menjaga coverage minimum.
        if (
            not self._eval_timing_adjusted
            and self.end_time
            and self.total_steps_all_epochs > 0
        ):
            self._eval_timing_adjusted = True
            eval_runtime = metrics.get("eval_runtime", 0)
            if eval_runtime > 0:
                try:
                    _end_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
                    _remaining_s = max(0, (_end_obj - datetime.datetime.now()).total_seconds())
                except (ValueError, TypeError):
                    _remaining_s = 0
                if _remaining_s > 0:
                    _eval_budget_s = _remaining_s * 0.10
                    _max_evals = max(3, int(_eval_budget_s / eval_runtime))
                    _new_eval_steps = max(30, self.total_steps_all_epochs // _max_evals)
                    if _new_eval_steps > getattr(args, "eval_steps", 0):
                        print(
                            f"[eval-timing] eval={eval_runtime:.1f}s, sisa={_remaining_s:.0f}s "
                            f"→ eval_steps {getattr(args, 'eval_steps', '?')} → {_new_eval_steps}",
                            flush=True,
                        )
                        args.eval_steps = _new_eval_steps
                        args.save_steps = _new_eval_steps
                    else:
                        print(f"[eval-timing] eval cepat ({eval_runtime:.1f}s), interval tetap", flush=True)


    def _safe_copy_checkpoint(self, src_path: str, dst_path: str, step: int) -> bool:
        """Copy checkpoint src_path → dst_path dengan error handling lengkap.

        Menghindari skenario fatal: rmtree berhasil tapi copytree gagal → dst hilang.
        Menggunakan dirs_exist_ok=True sebagai fallback agar dst tidak perlu dihapus dulu.

        Returns True jika berhasil, False jika gagal.
        """
        if not os.path.exists(src_path):
            print(
                f"[on_save] WARNING: checkpoint-{step} tidak ditemukan di {src_path}",
                flush=True,
            )
            return False

        # Path 1: rmtree + copytree (clean copy — paling aman, hasilnya identik dengan src)
        try:
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
            print(f"[on_save] checkpoint-{step} → submission_dir OK", flush=True)
            return True
        except Exception as _e1:
            print(f"[on_save] copytree gagal (step={step}): {_e1}", flush=True)

        # Path 2: dirs_exist_ok fallback (Python 3.8+) — tidak perlu rmtree
        try:
            os.makedirs(dst_path, exist_ok=True)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            print(f"[on_save] checkpoint-{step} → submission_dir OK (dirs_exist_ok)", flush=True)
            return True
        except Exception as _e2:
            print(f"[on_save] dirs_exist_ok fallback juga gagal (step={step}): {_e2}", flush=True)
            return False

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):

        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True

        self.has_checkpoint = True

        if not is_main_process(LOCAL_RANK): # if not main process, skip this
            return

        # Diagnostik: log setiap kali on_save dipanggil
        print(
            f"[on_save] step={state.global_step} save_only={self.save_only} "
            f"update_best={self.update_best_checkpoint} "
            f"has_ckpt={self.has_checkpoint} end_time_fired={self._end_time_fired}",
            flush=True,
        )

        if self.save_only: # if only save, do not evaluate
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            src = os.path.join(self.output_dir, f"checkpoint-{current_step}")
            ok = self._safe_copy_checkpoint(src, self.submission_dir, current_step)
            if ok:
                self._patch_submission_architectures()
                with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                    f.write(f"{current_step},no_eval")
            else:
                print(f"[on_save] save_only GAGAL untuk step={current_step}", flush=True)
            self.update_best_checkpoint = False
            # release the flag
            self.save_only = False
            return

        # Custom logic after model is saved
        # You can trigger external services, logs, or backups here
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            best_step = self.best_checkpoint_info["step"]
            best_eval_loss = self.best_checkpoint_info["loss"]
            src = os.path.join(self.output_dir, f"checkpoint-{best_step}")
            ok = self._safe_copy_checkpoint(src, self.submission_dir, best_step)
            if ok:
                self._patch_submission_architectures()
                with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                    f.write(f"{best_step},{best_eval_loss}")
            else:
                print(f"[on_save] update_best GAGAL untuk step={best_step}", flush=True)
            self.update_best_checkpoint = False

        # end_time fallback: jika end_time sudah fire tapi submission_dir masih kosong
        # (misal eval_loss=None atau update_best_checkpoint tidak pernah True),
        # paksa simpan checkpoint saat ini agar miner selalu punya submission.
        if self._end_time_fired and is_main_process(LOCAL_RANK):
            self._end_time_fired = False
            sub_empty = (
                not os.path.exists(self.submission_dir)
                or len(os.listdir(self.submission_dir)) < 2
            )
            if sub_empty:
                current_step = state.global_step
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{current_step}")
                print(
                    f"[end_time fallback] submission_dir kosong, menyimpan checkpoint-{current_step}",
                    flush=True,
                )
                ok = self._safe_copy_checkpoint(checkpoint_path, self.submission_dir, current_step)
                if ok:
                    self._patch_submission_architectures()
                    with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                        f.write(f"{current_step},end_time_fallback")
                else:
                    print(f"[end_time fallback] GAGAL untuk step={current_step}", flush=True)


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)

        if eval_loss is not None:
            eval_loss = - eval_loss

        return eval_loss

    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool:
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)  # Make end_time timezone-aware in UTC
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        
        if self.steps_per_epoch > 0 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps > 0 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                # the eval time might be higher than the end_time, so we need to let the pod not stop by setting a flag for this
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    # set wandb_mode=offline; do not upload the data to wandb export WANDB_MODE=offline
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True