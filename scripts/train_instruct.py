from typing import Dict, Optional
import requests
import json
import random
from utility import log_info, MyDataset
from tokenizer_safe import safe_load_tokenizer
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer, BitsAndBytesConfig
import transformers
import torch
from transformers.trainer_utils import is_main_process
from dataclasses import dataclass, field
from transformers import Trainer
from customized_trainer import resize_if_needed, set_generation_config, CustomEvalSaveCallback, WhenToEvalHandler, init_wandb
from soup_callback import ModelSoupCallback
from final_dev_train import run_final_dev_train

# from packing.packed_dataset import PackedDataset
from transformers import (
    Trainer,
    TrainingArguments,
)

import os
import datetime
import shutil
from huggingface_hub import HfApi
from typing import Callable, Optional
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from state_manager import get_state, set_state

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    request_path: Optional[str] = field(default=None)
    packing: Optional[bool] = field(default=False)
    max_packed_size: Optional[int] = field(default=-1)
    use_liger: Optional[bool] = field(default=False)
    use_lora: Optional[bool] = field(default=False)
    disable_fa: Optional[bool] = field(default=False)
    use_attn_implementation: Optional[str] = field(default="")

@dataclass
class LoraArguments:
    lora_r: int = 128
    lora_alpha: int = 512
    lora_dropout: float = 0.1
    lora_target_modules: str = "all"  # all for all linear; "q_proj v_proj"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    
    
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def log_trainable_param_summary(model):
    """Log trainable vs frozen parameter breakdown for the model."""
    total_params = 0
    trainable_params = 0
    adapter_params = 0
    embedding_params = 0

    for name, param in model.named_parameters():
        n = param.numel() if param.numel() > 0 else getattr(param, "ds_numel", 0)
        total_params += n
        if param.requires_grad:
            trainable_params += n
            if any(k in name for k in ("lm_head", "embed_tokens", "embed_")):
                embedding_params += n
            else:
                adapter_params += n

    frozen_params = total_params - trainable_params
    pct = 100.0 * trainable_params / max(total_params, 1)
    log_info(
        f"Param summary | total={total_params:,d} | trainable={trainable_params:,d} ({pct:.2f}%) "
        f"| frozen={frozen_params:,d} | adapter={adapter_params:,d} | embedding={embedding_params:,d}"
    )


def _load_kl_ref_model(model_path: str, training_args, device: torch.device):
    """Muat salinan beku model asal untuk jalur KL full fine-tune.

    Berbeda dari winner (fungsi terpisah di kl_trainer.py) — implementasi ini
    langsung di train_instruct.py agar semuanya dalam satu file tanpa import ekstra.
    Hasilnya: model beku di device sama dengan model utama.
    """
    attn = "flash_attention_2" if not training_args.disable_fa else "eager"
    if training_args.use_attn_implementation:
        attn = training_args.use_attn_implementation
    ref = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    )
    ref.to(device)
    ref.eval()
    try:
        ref.config.use_cache = False
    except Exception:
        pass
    for p in ref.parameters():
        p.requires_grad_(False)
    log_info(f"[kl] model referensi beku dimuat dari {model_path} di {device}")
    return ref


class KLRegularizedTrainer(Trainer):
    """
    Trainer dengan penalti KL divergence untuk tugas G.O.D ber-flag KL.

    Loss = CE(finetuned, labels) + kl_coef * KL(P_ft || P_base)
    KL dihitung pada completion tokens (label != -100), sesuai validator evaluator.

    Perbedaan implementasi dari pendekatan lain:
    - Forward TANPA label agar logits selalu dimaterialkan (kompatibel liger kernel)
    - KL dihitung via F.kl_div(log_target=True) dalam float32 — lebih stabil
      numerik daripada exp() * diff manual di bf16
    - Mendukung dua jalur base-logits:
        * LoRA   → context manager disable_adapter() — nol memori ekstra
        * Full-FT → model referensi beku (ref_model) di device yang sama
    - model_accepts_loss_kwargs=False memastikan training_step selalu membagi
      loss dengan gradient_accumulation_steps (scaling akurat di semua versi HF)
    """

    def __init__(self, *args, kl_coef: float = 0.0, ref_model=None, use_lora: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_coef = kl_coef
        self.ref_model = ref_model      # frozen copy untuk full-FT path; None untuk LoRA
        self.use_lora = use_lora        # True → pakai disable_adapter(), bukan ref_model
        # Paksa training_step membagi loss dengan grad_accum (bukan compute_loss)
        # sehingga accumulation steps tidak memperbesar gradien secara salah
        self.model_accepts_loss_kwargs = False
        self._kl_first_step = True
        log_info(
            f"[kl] KLRegularizedTrainer: coef={kl_coef}, "
            f"sumber={'lora_adapter' if use_lora else 'frozen_copy'}"
        )

    def _kl_active(self):
        return self.kl_coef > 0.0 and (self.use_lora or self.ref_model is not None)

    def _base_logits(self, model, input_ids, attention_mask):
        """Logits dari model base, tanpa gradien."""
        with torch.no_grad():
            if self.use_lora:
                # LoRA: matikan adapter sementara, jalankan forward
                with model.disable_adapter():
                    return model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Full-FT: gunakan model referensi yang sudah dibekukan
            if self.ref_model.device != input_ids.device:
                self.ref_model.to(input_ids.device)
            return self.ref_model(input_ids=input_ids, attention_mask=attention_mask).logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs["labels"]

        # Forward TANPA labels — memastikan logits selalu ada (aman dengan liger)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # Cross-entropy causal-LM standar (shifted)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.args, "label_smoothing_factor", 0.0) or 0.0,
        )

        # Aux loss MoE jika model router aktif
        aux = getattr(outputs, "aux_loss", None)
        if aux is not None:
            aux_coef = getattr(getattr(model, "config", None), "router_aux_loss_coef", 0.0)
            ce_loss = ce_loss + aux_coef * aux

        if not self._kl_active():
            return (ce_loss, outputs) if return_outputs else ce_loss

        # KL(P_ft || P_base) pada completion tokens (unshifted — sesuai evaluator)
        mask = labels != -100  # [B, T]
        if mask.any():
            # Float32 upcast — bf16 terlalu lossy untuk log-softmax di kl_div
            ft_f32  = logits[mask].float()                                     # [N, V]
            ref_f32 = self._base_logits(model, input_ids, attention_mask)[mask].float()  # [N, V]

            log_ft   = torch.nn.functional.log_softmax(ft_f32,  dim=-1)   # log P_ft
            log_base = torch.nn.functional.log_softmax(ref_f32, dim=-1)   # log P_base

            # kl_div(input=log_Q, target=log_P, log_target=True):
            #   elemen[i,v] = exp(log_P[i,v]) * (log_P[i,v] - log_Q[i,v])
            #               = P_ft * (log P_ft - log P_base)
            # sum per token → [N], mean → scalar
            kl_per_token = torch.nn.functional.kl_div(
                log_base, log_ft, reduction="none", log_target=True
            ).sum(dim=-1)  # [N]
            kl_loss = kl_per_token.mean()
        else:
            kl_loss = logits.new_zeros(())

        total_loss = ce_loss + self.kl_coef * kl_loss

        if self._kl_first_step and is_main_process(LOCAL_RANK):
            log_info(
                f"[kl] langkah pertama: ce={ce_loss.item():.4f} "
                f"kl={float(kl_loss):.4f} total={total_loss.item():.4f} "
                f"(coef={self.kl_coef})"
            )
            self._kl_first_step = False

        return (total_loss, outputs) if return_outputs else total_loss



def load_lora_model(training_args: TrainingArguments, model_path: str, lora_args: LoraArguments, token_nums: int):
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        model_class = AutoLigerKernelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if lora_args.q_lora
            else None
        ),
    )
    # do not resize tokem embeddings in LOra --> will encounter size mismatch error in evaluation 
    # model.resize_token_embeddings(token_nums)
    # convert to lora
    if lora_args.lora_target_modules == "all":
        target_modules = find_all_linear_names(model)
    else:
        modules = lora_args.lora_target_modules.split(" ")
        target_modules = [mod.strip() for mod in modules if len(mod.strip()) > 0]

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
        # modules_to_save=["lm_head", "embed_tokens"],  # because we retrain the embedding
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    model.config.use_cache = False
    # Activate computing load balancing loss iin MixtralForCausalLM
    if hasattr(model.config, "output_router_logits"):
        setattr(model.config, "output_router_logits", True)

    print_trainable_parameters(model)
    return model


def load_model(training_args: TrainingArguments, model_path: str, token_nums: int):
    model_class = transformers.AutoModelForCausalLM
    
    if training_args.use_liger:
        from liger_kernel.transformers import AutoLigerKernelForCausalLM

        log_info("---------------using LIGER------------")
        model_class = AutoLigerKernelForCausalLM
    
    attn_implementation="flash_attention_2" if not training_args.disable_fa else "eager"
    if training_args.use_attn_implementation:
        attn_implementation = training_args.use_attn_implementation
        log_info(f"Using {attn_implementation} as the attention implementation")
    log_info(f"Using attn_implementation: {attn_implementation}")
    
    model = model_class.from_pretrained(
        model_path,
        # trust_remote_code=True, remove this because we already filter the model architecture, it will not be used with liger-kernel 
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    # model.resize_token_embeddings(token_nums)
    return model


def get_max_length_config():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(dir_path, "test_axolotl.yml")
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict["sequence_len"]


def main():
    """Format of training requests"""
    import sys as _sys
    _sys.stderr.write("[train_instruct] main() dimulai\n")
    _sys.stderr.flush()

    argument_parser = transformers.HfArgumentParser((TrainingArguments, LoraArguments))
    (training_args, lora_args) = argument_parser.parse_args_into_dataclasses()
    train_info = json.load(open(training_args.request_path, "r"))
    train_request = train_info["train_request"]
    # log_info(f"Training request: {train_request}", "start")
    task_id = train_request["task_id"]

    tokenizer = safe_load_tokenizer(train_request["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # wandb_init_success = init_wandb(train_request)
    # if not wandb_init_success:
    #     log_info("WANDB_API_KEY is not set, do not report to wandb")
    #     training_args.report_to = "none"    
    # else:
    #     log_info("WANDB_API_KEY is provided, we will report to wandb")
    #     training_args.report_to = "wandb"
        
    max_length = get_max_length_config()
    if "max_length" in train_request:
        max_length = train_request["max_length"]

    # we already tokenize the data and save it to train_tokenized.json and dev_tokenized.json
    train_ds = MyDataset(
        tokenizer,
       f"datasets/train_tokenized_{task_id}.json",
        max_length
    )

    dev_ds = MyDataset(
        tokenizer,
        f"datasets/dev_tokenized_{task_id}.json",
        max_length
    )
    log_info(f"train_size: {len(train_ds)}; dev_size: {len(dev_ds)}")
    
    
    donot_pack = False
    original_train_size = len(train_ds)
    original_steps = original_train_size // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )  # number of steps in the original training
    # min_steps here is per epoch
    if original_steps < train_request["min_steps"]:
        donot_pack = True
        log_info(f"original_steps: {original_steps} < min_steps: {train_request['min_steps']}, do not pack the dataset")

    min_data_size_num = (
        train_request["min_steps"]
        * training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    
        
    log_info(f"min_data_size_num: {min_data_size_num}; max_length: {max_length}")
    if training_args.packing and not donot_pack:
        from monkeypatch import monkey_patch_packing_for_model, PackedDataset
        log_info("Patching packing for model")

        monkey_patch_packing_for_model(train_request["model_path"])
        t1 = datetime.datetime.now()
        train_ds = PackedDataset(
            train_ds,
            tokenizer,
            max_input_length=max_length,
            max_packed_size=training_args.max_packed_size,
            min_item_num=min_data_size_num,
        )
        t2 = datetime.datetime.now()
        log_info(f"time for packing train_ds: {(t2 - t1).total_seconds()}")
        t1 = datetime.datetime.now()
        dev_ds = PackedDataset(
            dev_ds,
            tokenizer,
            max_input_length=max_length,
            max_packed_size=training_args.max_packed_size,
        )
        t2 = datetime.datetime.now()
        log_info(f"time for packing dev_ds: {(t2 - t1).total_seconds()}")
        log_info(f"train_ds: {train_ds.stat()}")
        log_info(f"dev_ds: {dev_ds.stat()}")

    log_info(f"world_size: {training_args.world_size}")
    total_steps_per_epoch = len(train_ds) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if total_steps_per_epoch == 0:
        total_steps_per_epoch = 1
    log_info(f"total_steps_per_epoch: {total_steps_per_epoch}")
    # consider reducing the batch_size if it is quite big
    # num_steps = len(train_ds) * training_args.num_train_epochs / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size)
    # num_steps > min_step ->
    max_batch_size_theory = len(train_ds) / (
        training_args.gradient_accumulation_steps
        * training_args.world_size
        * train_request["min_steps"]
    )
    max_batch_size_theory = int(max_batch_size_theory)
    if max_batch_size_theory == 0:
        max_batch_size_theory = 1

    original_batch_size = training_args.per_device_train_batch_size
    if training_args.per_device_train_batch_size > max_batch_size_theory:
        # if batch_size is quite big set it to this value to make sure that we have at least min_steps
        if train_request.get("adjust_batch_size", True):
            log_info(
                f"batch_size ({training_args.per_device_train_batch_size}) is quite big, reducing it to {max_batch_size_theory}"
            )
            training_args.per_device_train_batch_size = max_batch_size_theory
            # need to update total_steps_per_epoch
            total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
            log_info(f"updated total_steps_per_epoch: {total_steps_per_epoch}")

    if training_args.use_lora:
        model = load_lora_model(training_args, train_request["model_path"], lora_args, len(tokenizer))
    else:
        model = load_model(training_args, train_request["model_path"], len(tokenizer))
        # some model need to resize the token embeddings or encounter the size mismatch error; only for full-weight models
        resize_if_needed(train_request["model_name"], model, len(tokenizer))

    log_trainable_param_summary(model)

    try:
        model.config.use_cache = False
    except:
        pass

    # some model need to set the generation config or encounter the invalid generation config error
    set_generation_config(train_request["model_name"], model)

    # KL regularization: dua sumber — env var (USE_KL/KL_COEF, cara container
    # validator menyuntikkan) atau train_request["kl_coef"] (kompatibilitas).
    # Env var diprioritaskan karena itulah mekanisme resmi G.O.D task runner.
    _use_kl_env = os.environ.get("USE_KL") == "1"
    _kl_coef_env = os.environ.get("KL_COEF", "")
    kl_coef = 0.0
    if _use_kl_env and _kl_coef_env:
        try:
            kl_coef = float(_kl_coef_env)
        except (ValueError, TypeError):
            log_info(f"[kl] KL_COEF env var tidak valid ({_kl_coef_env!r}), dinonaktifkan")
    if kl_coef == 0.0:
        kl_coef = float(train_request.get("kl_coef", 0.0))
    log_info(
        f"[kl] kl_coef={kl_coef} "
        f"(USE_KL={_use_kl_env}, KL_COEF_env={_kl_coef_env!r}, "
        f"train_request={train_request.get('kl_coef', 0.0)})"
    )

    # Check if this is the main process and create the output directory
    if is_main_process(LOCAL_RANK):  # Only create directory on main process
        os.makedirs(training_args.output_dir, exist_ok=True)
        log_info(f"Created output directory: {training_args.output_dir}")
    
    periodic_save_steps = train_request.get("periodic_save_steps", -1)
    log_info(f"periodic_save_steps: {periodic_save_steps}")
    training_args.save_only_model = True  # only save the model, not the optimizer
    
    max_steps = train_request.get("max_steps", -1)
    log_info(f"max_steps: {max_steps}")
    
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state = get_state()
    state["train"]["start_train_time"] = start_time
    if is_main_process(LOCAL_RANK):
        set_state(state)
        
    total_steps_per_epoch = len(train_ds) // (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            )
    
    total_steps_all_epochs = total_steps_per_epoch * training_args.num_train_epochs
    log_info(f"total_steps_per_epoch: {total_steps_per_epoch}; total_steps_all_epochs: {total_steps_all_epochs}")

    # Dynamic warmup: 5% of total steps, clamped between 10 and 100
    dynamic_warmup = max(10, min(100, int(total_steps_all_epochs * 0.05)))
    if dynamic_warmup != training_args.warmup_steps:
        log_info(f"Overriding warmup_steps: {training_args.warmup_steps} -> {dynamic_warmup}")
        training_args.warmup_steps = dynamic_warmup

    success_file = os.path.join(training_args.output_dir, "success.txt")
    # remove the success file if it exists
    if is_main_process(LOCAL_RANK) and os.path.exists(success_file):
        os.remove(success_file)
    
    checking_step = train_request["checking_step"]
    if checking_step >= total_steps_per_epoch:
        checking_step = total_steps_per_epoch - 2

    # Guard: dataset terlalu kecil sehingga checking_step jadi <= 0 (misal -2).
    # Dalam kondisi ini LR-search loop di text_trainer.py tidak akan pernah selesai
    # karena on_step_end checking block tidak pernah dicapai dan state["mode"]
    # tidak pernah diubah dari "initial". Paksa mode="finish" agar loop keluar.
    if checking_step <= 0:
        log_info(
            f"Dataset too small for LR search (checking_step={checking_step}, "
            f"total_steps_per_epoch={total_steps_per_epoch}). "
            f"Forcing state mode='finish' to prevent infinite loop."
        )
        checking_step = 1  # fallback agar tidak ada nilai negatif di callback
        if is_main_process(LOCAL_RANK):
            _tiny_state = get_state()
            _tiny_state["mode"] = "finish"
            set_state(_tiny_state)

    _eval_callback = CustomEvalSaveCallback(
        WhenToEvalHandler(
            train_request["end_time"],
            train_request["save_before_remaining_time"],
            periodic_save_steps=periodic_save_steps,
            steps_per_epoch=total_steps_per_epoch,
            max_steps=max_steps,
        ),
        train_request["submission_dir"],
        training_args.output_dir,
        train_request["model_path"],   # local path untuk architecture patching config.json
        max_steps,
        checking_step=checking_step,
        total_steps_all_epochs=total_steps_all_epochs,
        end_time=train_request["end_time"],
        checking_mode=train_request.get("checking_mode", "none"),
    )

    # Soup callback: kumpulkan top-K checkpoint, rata-ratakan di akhir training.
    # Berbeda dari winner (greedy soup) — kita pakai uniform averaging
    # (lebih cepat, satu eval, tidak butuh iterasi per kandidat).
    _soup_cb = ModelSoupCallback(
        submission_dir=train_request["submission_dir"],
    )

    # ── Model referensi beku untuk jalur KL full fine-tune ──────────────────
    # LoRA tidak butuh ini (adapter bisa dinonaktifkan via context manager).
    # Full-FT: muat salinan beku sebelum trainer dibuat agar device sudah benar.
    # Error loading → KL dimatikan (training tetap lanjut tanpa penalti KL).
    _kl_ref_model = None
    if kl_coef > 0.0 and not training_args.use_lora:
        try:
            _kl_device = next(model.parameters()).device
            _kl_ref_model = _load_kl_ref_model(
                train_request["model_path"], training_args, _kl_device
            )
        except Exception as _kl_load_err:
            log_info(f"[kl] gagal muat model referensi ({_kl_load_err}), KL dinonaktifkan")
            kl_coef = 0.0

    # Gunakan KLRegularizedTrainer saat KL aktif.
    # Fallback ke Trainer biasa kalau kl_coef == 0 (identik, tanpa overhead).
    if kl_coef > 0.0:
        log_info(f"[kl] menggunakan KLRegularizedTrainer, coef={kl_coef}")
        trainer = KLRegularizedTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            kl_coef=kl_coef,
            ref_model=_kl_ref_model,          # None untuk LoRA, frozen copy untuk full-FT
            use_lora=bool(training_args.use_lora),
            callbacks=[_eval_callback, _soup_cb],
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            callbacks=[_eval_callback, _soup_cb],
        )

    # Berikan referensi trainer ke soup callback agar evaluate() bisa dipanggil
    # dari on_train_end (untuk menilai apakah rata-rata lebih baik dari best single).
    _soup_cb.trainer = trainer
    trainer.tokenizer = tokenizer

    import sys as _sys
    _sys.stderr.write(
        f"[train_instruct] trainer dibuat — bs={training_args.per_device_train_batch_size} "
        f"max_len={train_request.get('max_length','?')} "
        f"steps_per_epoch={total_steps_per_epoch} "
        f"output_dir={training_args.output_dir}\n"
    )
    _sys.stderr.flush()
    _sys.stderr.write(f"[train_instruct] Memulai trainer.train()\n")
    _sys.stderr.flush()
    trainer.train()
    _sys.stderr.write(
        f"[train_instruct] trainer.train() selesai, global_step={trainer.state.global_step}\n"
    )
    _sys.stderr.flush()

    # ── Emergency save: jika trainer.train() selesai tapi submission_dir kosong ──
    # Strategi 1 — copy checkpoint terakhir dari output_dir.
    # Strategi 2 — trainer.save_model() langsung ke submission_dir (fallback kalau
    #              tidak ada checkpoint sama sekali, misalnya training sangat singkat).
    if is_main_process(LOCAL_RANK):
        sub_dir = train_request["submission_dir"]
        sub_files = len(os.listdir(sub_dir)) if os.path.exists(sub_dir) else 0
        log_info(f"[emergency-check] submission_dir files={sub_files}")
        _sys.stderr.write(f"[emergency-check] sub_files={sub_files} sub_dir={sub_dir}\n")
        _sys.stderr.flush()
        if sub_files < 2:
            # ── Strategi 1: salin last_checkpoint ─────────────────────────────
            last_ckpt = get_last_checkpoint(training_args.output_dir)
            log_info(f"[emergency-check] last_checkpoint={last_ckpt}")
            _sys.stderr.write(f"[emergency-check] last_checkpoint={last_ckpt}\n")
            _sys.stderr.flush()
            if last_ckpt and os.path.isdir(last_ckpt):
                try:
                    log_info(f"[emergency-save] menyalin {last_ckpt} → {sub_dir}")
                    if os.path.exists(sub_dir):
                        shutil.rmtree(sub_dir)
                    shutil.copytree(last_ckpt, sub_dir)
                    with open(os.path.join(sub_dir, "loss.txt"), "w") as _f:
                        _f.write(f"{trainer.state.global_step},emergency_save")
                    log_info(f"[emergency-save] OK — {len(os.listdir(sub_dir))} files")
                    _sys.stderr.write(f"[emergency-save] strategi-1 OK, files={len(os.listdir(sub_dir))}\n")
                    _sys.stderr.flush()
                except Exception as _es_exc:
                    log_info(f"[emergency-save] strategi-1 GAGAL: {_es_exc}")
                    _sys.stderr.write(f"[emergency-save] strategi-1 GAGAL: {_es_exc}\n")
                    _sys.stderr.flush()

            # ── Strategi 2: trainer.save_model() langsung ────────────────────
            # Dipakai jika strategi-1 gagal ATAU tidak ada checkpoint sama sekali
            # (misalnya training berhenti sebelum checkpoint-1 selesai dibuat).
            _sub_files2 = len(os.listdir(sub_dir)) if os.path.exists(sub_dir) else 0
            if _sub_files2 < 2:
                try:
                    log_info(f"[emergency-save2] trainer.save_model({sub_dir})")
                    _sys.stderr.write(f"[emergency-save2] Memulai trainer.save_model\n")
                    _sys.stderr.flush()
                    os.makedirs(sub_dir, exist_ok=True)
                    trainer.save_model(sub_dir)
                    tokenizer.save_pretrained(sub_dir)
                    with open(os.path.join(sub_dir, "loss.txt"), "w") as _f:
                        _f.write(f"{trainer.state.global_step},emergency_save2")
                    _n2 = len(os.listdir(sub_dir))
                    log_info(f"[emergency-save2] OK — {_n2} files")
                    _sys.stderr.write(f"[emergency-save2] OK, files={_n2}\n")
                    _sys.stderr.flush()
                except Exception as _es2_exc:
                    log_info(f"[emergency-save2] GAGAL: {_es2_exc}")
                    _sys.stderr.write(f"[emergency-save2] GAGAL: {_es2_exc}\n")
                    _sys.stderr.flush()

    # ── Final dev pass: satu epoch terakhir LR kecil pada dev set ──────────
    # Dipanggil setelah emergency save agar submission_dir pasti berisi checkpoint.
    # Jika sisa waktu < 2 menit, run_final_dev_train() mengabaikan dirinya sendiri.
    # Semua rank memanggil ini (DDP-safe di dalam fungsi).
    try:
        run_final_dev_train(
            trainer,
            submission_dir=train_request["submission_dir"],
            end_time=train_request["end_time"],
            base_lr=float(training_args.learning_rate),
            local_rank=LOCAL_RANK,
            log=log_info,
        )
    except Exception as _fdt_exc:
        log_info(f"[final_dev] dilewati karena error: {_fdt_exc}")

    if is_main_process(LOCAL_RANK):
        success_file = os.path.join(training_args.output_dir, "success.txt")
        with open(success_file, "w") as f:
            f.write("Success")
    log_info("Training successfully done", "finish")

if __name__ == "__main__":
    main()
