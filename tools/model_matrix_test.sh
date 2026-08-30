#!/bin/bash
# =============================================================================
#  Matriks smoke-test lintas keluarga model — satu wakil per JALUR KODE unik.
#  Tujuan: bukan skor, tapi "tidak crash + loss.txt berisi angka" di tiap
#  cabang perilaku (FA-off, naive packing, MoE aux-loss, batch khusus, liger).
#
#  Estimasi: ~10 run x 15-25 mnt di H100 ≈ 3-4 jam total.
#  Usage:  bash tools/model_matrix_test.sh            # semua
#          ONLY=opt,bloom bash tools/model_matrix_test.sh   # subset
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# slug|model|jalur kode yang diuji
MATRIX=(
  "opt|facebook/opt-125m|naive packing (tanpa FA isolation)"
  "gptneo|EleutherAI/gpt-neo-125m|disable_flash_attention by arch"
  "bloom|bigscience/bloom-560m|FA off + FIXED_BS override"
  "pythia|EleutherAI/pythia-160m|gptneox + pengurangan batch khusus"
  "falcon|tiiuae/falcon-rw-1b|FA off by model-name"
  "mixtral|TitanML/tiny-mixtral|MoE + router aux loss + liger mixtral"
  "qwen3|Qwen/Qwen3-0.6B|liger qwen3 (keluarga boss-round)"
  "gemma2|unsloth/gemma-2-2b|liger gemma2"
  "phi3|microsoft/Phi-3-mini-4k-instruct|liger phi3, 3.8B bracket 2-4B"
)
# Sudah terbukti di tournament/kalibrasi (tidak perlu diulang):
#   Qwen2 (0.5B), Llama (SmolLM2/Llama-3.2-1B), LFM2 (2.6B), Phi (phi-1_5)

RESULTS_FILE="/tmp/model_matrix_results.txt"
: > "$RESULTS_FILE"

for entry in "${MATRIX[@]}"; do
    slug="${entry%%|*}"
    rest="${entry#*|}"
    model="${rest%%|*}"
    path_desc="${rest#*|}"

    if [ -n "${ONLY:-}" ] && ! echo ",$ONLY," | grep -q ",$slug,"; then
        continue
    fi

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  [$slug] $model"
    echo "  Jalur: $path_desc"
    echo "════════════════════════════════════════════════════"

    MODEL="$model" N_SAMPLES=3000 HOURS=0.35 bash examples/run_instruct.sh
    _rc=$?

    # Ambil loss.txt dari run terbaru
    _dir=$(ls -td /ephemeral/agultext_cache/checkpoints/test_instruct_* 2>/dev/null | head -1)
    _loss=$(cat "$_dir/instruct-test-output/loss.txt" 2>/dev/null || echo "TIDAK ADA")
    printf "%-10s %-40s exit=%s loss.txt=%s\n" "$slug" "$model" "$_rc" "$_loss" >> "$RESULTS_FILE"
done

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "  RINGKASAN MATRIKS MODEL"
echo "╚════════════════════════════════════════════════════╝"
cat "$RESULTS_FILE"
echo ""
echo "Kriteria LULUS per baris: loss.txt berisi 'step,angka' (bukan no_eval/TIDAK ADA)."
