#!/bin/bash
# =============================================================================
#  Upload hasil training ke HuggingFace Hub
#  Usage: bash examples/upload_to_hf.sh <checkpoint_dir> <hf_repo_id>
#
#  Contoh:
#    bash examples/upload_to_hf.sh \
#      /tmp/agultext_cache/checkpoints/test_instruct_1234/instruct-test-output \
#      yosa97/agultext-qwen2.5-instruct-test
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CHECKPOINT_DIR="${1}"
HF_REPO_ID="${2}"
HF_TOKEN="${HF_TOKEN:-}"       # set via env atau akan ditanya

if [ -z "$CHECKPOINT_DIR" ] || [ -z "$HF_REPO_ID" ]; then
    echo "Usage: bash examples/upload_to_hf.sh <checkpoint_dir> <hf_repo_id>"
    echo ""
    echo "Contoh:"
    echo "  bash examples/upload_to_hf.sh \\"
    echo "    /tmp/agultext_cache/checkpoints/test_instruct_1234/instruct-test-output \\"
    echo "    yosa97/agultext-test"
    exit 1
fi

if [ -z "$HF_TOKEN" ]; then
    echo -n "Masukkan HuggingFace token (dari https://huggingface.co/settings/tokens): "
    read -s HF_TOKEN
    echo ""
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Direktori tidak ditemukan: $CHECKPOINT_DIR"
    echo ""
    echo "Cek daftar checkpoint yang ada:"
    ls /tmp/agultext_cache/checkpoints/ 2>/dev/null || echo "  (tidak ada checkpoint)"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "  Upload ke HuggingFace Hub"
echo "  From : $CHECKPOINT_DIR"
echo "  To   : $HF_REPO_ID"
echo "╚══════════════════════════════════════════╝"

# Cek loss.txt jika ada
if [ -f "$CHECKPOINT_DIR/loss.txt" ]; then
    echo ">>> Loss info: $(cat $CHECKPOINT_DIR/loss.txt)"
fi

# Upload via Python
python3 - << PYEOF
import os
from huggingface_hub import HfApi, create_repo

token = "$HF_TOKEN"
repo_id = "$HF_REPO_ID"
local_dir = "$CHECKPOINT_DIR"

api = HfApi(token=token)

# Buat repo jika belum ada
try:
    create_repo(repo_id, token=token, private=False, exist_ok=True)
    print(f">>> Repo: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f">>> Repo sudah ada atau error: {e}")

# Upload semua file
print(f">>> Mengupload dari {local_dir} ...")
api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    token=token,
    ignore_patterns=["*.log", "optimizer.pt", "rng_state*.pth"],
)

print("")
print(f"✓ Upload selesai!")
print(f"  https://huggingface.co/{repo_id}")
PYEOF
