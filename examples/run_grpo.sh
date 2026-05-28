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
HOURS="${HOURS:-0.5}"
REPO_NAME="grpo-test-output"
IMAGE_NAME="agultext:latest"
CACHE_DIR="/tmp/agultext_cache"

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

echo ">>> Dataset GRPO ditulis: $DATASET_PATH (15 contoh)"

# Reward function: beri nilai 1.0 jika jawaban mengandung angka
DATASET_TYPE='{"field_prompt":"prompt","reward_functions":[{"reward_func":"def reward_func(completions, **kwargs):\n    import re\n    return [1.0 if re.search(r\"\\\\d+\", c) else 0.0 for c in completions]","reward_weight":1.0,"func_hash":"test_grpo_hash","is_generic":false}]}'

if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo ">>> Building Docker image (pertama kali ~20-30 menit)..."
    docker build -f dockerfiles/standalone-text-trainer.dockerfile -t "$IMAGE_NAME" .
fi

MODEL_DIR_NAME="$(echo $MODEL | tr '/' '--')"

echo ">>> Menjalankan container training GRPO..."
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=16g \
    -v "$CACHE_DIR:/cache" \
    -v "$CACHE_DIR/checkpoints:/app/checkpoints" \
    -e WANDB_MODE=offline \
    -e HF_HUB_ENABLE_HF_TRANSFER=1 \
    -e TASK_ID="$TASK_ID" \
    -e MODEL="$MODEL" \
    -e MODEL_DIR_NAME="$MODEL_DIR_NAME" \
    -e DATASET_TYPE="$DATASET_TYPE" \
    -e HOURS="$HOURS" \
    -e REPO_NAME="$REPO_NAME" \
    "$IMAGE_NAME" \
    bash -c '
        redis-server --daemonize yes && sleep 2
        MODEL_DIR="/cache/models/$MODEL_DIR_NAME"
        if [ ! -d "$MODEL_DIR" ]; then
            python -c "from huggingface_hub import snapshot_download; snapshot_download(\"$MODEL\", local_dir=\"$MODEL_DIR\", ignore_patterns=[\"*.gguf\"]); print(\"Model downloaded.\")"
        fi
        cd /workspace/scripts
        python -m text_trainer \
            --task-id "$TASK_ID" \
            --model "$MODEL" \
            --dataset "/cache/datasets/${TASK_ID}_train_data.json" \
            --dataset-type "$DATASET_TYPE" \
            --task-type GrpoTask \
            --file-format json \
            --hours-to-complete "$HOURS" \
            --expected-repo-name "$REPO_NAME"
    '

echo ""
echo "✓ Training selesai → $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME"

if [ -n "$HF_REPO" ]; then
    echo ""
    echo ">>> Mengupload ke HuggingFace: $HF_REPO ..."
    HF_TOKEN="$HF_TOKEN" bash examples/upload_to_hf.sh \
        "$CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME" \
        "$HF_REPO"
else
    echo ""
    echo "TIP: Upload ke HF:"
    echo "  bash examples/upload_to_hf.sh $CACHE_DIR/checkpoints/$TASK_ID/$REPO_NAME yosa97/nama-repo"
fi
