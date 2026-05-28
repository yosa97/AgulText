#!/bin/bash
# =============================================================================
#  AgulText — Test GRPO Training + Upload ke HuggingFace
#  Usage: bash examples/run_grpo.sh
#
#  Jalankan dari root repo:
#    git clone https://github.com/yosa97/AgulText
#    cd AgulText
#    bash examples/run_grpo.sh
#
#  Dengan upload HF otomatis:
#    HF_TOKEN=hf_xxx HF_REPO=yosa97/test-grpo bash examples/run_grpo.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

TASK_ID="test_grpo_$(date +%s)"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
HF_REPO="${HF_REPO:-}"
HOURS="${HOURS:-0.1}"
REPO_NAME="grpo-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="/ephemeral/agultext_cache"

echo "╔══════════════════════════════════════════╗"
echo "  AgulText — GRPO Training Test"
echo "  Model  : $MODEL"
echo "  Task ID: $TASK_ID"
echo "╚══════════════════════════════════════════╝"

mkdir -p "$CACHE_DIR/models" "$CACHE_DIR/datasets" "$CACHE_DIR/wandb_logs"
mkdir -p "$CACHE_DIR/checkpoints"

DATASET_PATH="$CACHE_DIR/datasets/${TASK_ID}_train_data.json"
cat > "$DATASET_PATH" << 'EOF'
[
  {"prompt": "Berapa 15 dikali 7? Jawab dengan angka saja."},
  {"prompt": "Berapa 8 ditambah 13? Jawab dengan angka saja."},
  {"prompt": "Berapa 100 dibagi 4? Jawab dengan angka saja."},
  {"prompt": "Berapa 9 pangkat 2? Jawab dengan angka saja."},
  {"prompt": "Berapa 50 dikurangi 17? Jawab dengan angka saja."},
  {"prompt": "Berapa 6 dikali 8? Jawab dengan angka saja."},
  {"prompt": "Berapa 144 dibagi 12? Jawab dengan angka saja."},
  {"prompt": "Berapa 7 pangkat 2? Jawab dengan angka saja."},
  {"prompt": "Berapa 25 ditambah 38? Jawab dengan angka saja."},
  {"prompt": "Berapa 1000 dibagi 8? Jawab dengan angka saja."},
  {"prompt": "Berapa 13 dikali 6? Jawab dengan angka saja."},
  {"prompt": "Berapa 200 dikurangi 75? Jawab dengan angka saja."},
  {"prompt": "Berapa 3 pangkat 3? Jawab dengan angka saja."},
  {"prompt": "Berapa 360 dibagi 9? Jawab dengan angka saja."},
  {"prompt": "Berapa 17 ditambah 44? Jawab dengan angka saja."}
]
EOF

python3 -c "
import json
with open('$DATASET_PATH') as f:
    data = json.load(f)
expanded = (data * 17)[:250]
with open('$DATASET_PATH', 'w') as f:
    json.dump(expanded, f, ensure_ascii=False)
print(f'Dataset diperluas ke {len(expanded)} samples')
"

# Reward function: beri nilai 1.0 jika jawaban mengandung angka
DATASET_TYPE='{"field_prompt":"prompt","reward_functions":[{"reward_func":"def reward_func(completions, **kwargs):\n    import re\n    return [1.0 if re.search(r\"\\\\d+\", c) else 0.0 for c in completions]","reward_weight":1.0,"func_hash":"test_grpo_hash","is_generic":false}]}'

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (pertama kali ~20-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
fi

MODEL_DIR_NAME="$(echo $MODEL | tr '/' '--')"

rm -rf "$CACHE_DIR/internal_datasets"
rm -rf "$CACHE_DIR/soutputs"
rm -rf "$CACHE_DIR/wandb"
mkdir -p "$CACHE_DIR/internal_datasets"
mkdir -p "$CACHE_DIR/soutputs"
mkdir -p "$CACHE_DIR/wandb"

echo ">>> Menjalankan container training GRPO..."
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v "$CACHE_DIR:/cache" \
    -v "$CACHE_DIR/checkpoints:/app/checkpoints" \
    -v "$REPO_ROOT/scripts:/workspace/scripts" \
    -v "$CACHE_DIR/internal_datasets:/workspace/scripts/datasets" \
    -v "$CACHE_DIR/soutputs:/workspace/scripts/soutputs" \
    -v "$CACHE_DIR/wandb:/workspace/scripts/wandb" \
    -e WANDB_MODE=offline \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TASK_ID="$TASK_ID" \
    -e MODEL="$MODEL" \
    -e MODEL_DIR_NAME="$MODEL_DIR_NAME" \
    -e DATASET_TYPE="$DATASET_TYPE" \
    -e HOURS="$HOURS" \
    -e REPO_NAME="$REPO_NAME" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HF_REPO="$HF_REPO" \
    --entrypoint bash \
    "$IMAGE_NAME" \
    -c '
        redis-server --daemonize yes && sleep 2

        cd /workspace/scripts
        python download_model_only.py "$MODEL"

        mkdir -p /workspace/input_data
        cp "/cache/datasets/${TASK_ID}_train_data.json" /workspace/input_data/

        python -m text_trainer \
            --task-id "$TASK_ID" \
            --model "$MODEL" \
            --dataset "/cache/datasets/${TASK_ID}_train_data.json" \
            --dataset-type "$DATASET_TYPE" \
            --task-type GrpoTask \
            --file-format json \
            --hours-to-complete "$HOURS" \
            --expected-repo-name "$REPO_NAME"

        if [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
            echo ">>> Mengupload ke HuggingFace: $HF_REPO ..."
            python -c "
from huggingface_hub import HfApi, create_repo
api = HfApi(token=\"$HF_TOKEN\")
create_repo(\"$HF_REPO\", token=\"$HF_TOKEN\", private=False, exist_ok=True)
api.upload_folder(
    folder_path=\"/app/checkpoints/$TASK_ID/$REPO_NAME\",
    repo_id=\"$HF_REPO\",
    token=\"$HF_TOKEN\",
    ignore_patterns=[\"*.log\", \"optimizer.pt\", \"rng_state*.pth\"],
)
print(\"Upload selesai: https://huggingface.co/$HF_REPO\")
"
        fi
    '

echo ""
echo "✓ Training selesai → $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"
if [ -n "$HF_REPO" ]; then
    echo "✓ Model terupload → https://huggingface.co/$HF_REPO"
else
    echo "TIP: Upload ke HF: HF_TOKEN=hf_xxx HF_REPO=yosa97/nama-repo bash examples/run_grpo.sh"
fi
