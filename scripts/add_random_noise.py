import torch
from transformers import AutoModelForCausalLM
import os
import hashlib
import typer
from typing import Optional
from transformers import GenerationConfig
from tokenizer_safe import safe_load_tokenizer


def _seed_from_task(task_id: str) -> int:
    """Derive a deterministic but unique seed from task_id."""
    raw = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(raw[:4], byteorder="little")


def main(model_path: str, save_folder: str, task_id: Optional[str] = None, noise_std: float = 0.0008):
    """
    Load a model, add small uniqueness noise to embeddings, and save.
    noise_std: standard deviation of noise. Use small values (e.g. 0.0008) for
               trained models and larger (e.g. 0.01) for fallback (untrained) models.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig(temperature=None, top_p=None)
    tokenizer = safe_load_tokenizer(model_path)

    # Set per-task seed so each task submission is unique but reproducible
    if task_id:
        seed = _seed_from_task(task_id)
        torch.manual_seed(seed)
        print(f"Uniqueness noise: task_id={task_id}, seed={seed}, noise_std={noise_std}", flush=True)
    else:
        print(f"Uniqueness noise: no task_id, noise_std={noise_std}", flush=True)

    print("Adding uniqueness noise to model embeddings...", flush=True)
    with torch.no_grad():
        # Noise on input embeddings
        embed_in = model.get_input_embeddings()
        embed_in.weight.add_(torch.randn_like(embed_in.weight) * noise_std)

        # Noise on output embeddings (lm_head) if separate from input embeddings
        embed_out = model.get_output_embeddings()
        if embed_out is not None and embed_out is not embed_in:
            embed_out.weight.add_(torch.randn_like(embed_out.weight) * noise_std)

    os.makedirs(save_folder, exist_ok=True)
    print(f"Saving uniquified model to {save_folder}...", flush=True)
    model.save_pretrained(save_folder)
    tokenizer.save_pretrained(save_folder)
    print("Uniqueness noise applied successfully.", flush=True)


if __name__ == "__main__":
    typer.run(main)
