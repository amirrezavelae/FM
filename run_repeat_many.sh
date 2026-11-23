#!/usr/bin/env bash
# The single-word repeat test (run_kv_repeat_test.py, "Word: x / Repeat: x" few-shot)
# on all 120 needle words instead of just `unhappiness`.
#
# Question: does "dropping the prefix pieces does no harm" (true for `unhappiness`)
# hold for many words, and does it depend on the word class (common / rare / novel)?
# `unhappiness` is added as word 0 so you can see the earlier result reproduced.
#
#   bash run_repeat_many.sh                          # ~30 min on the RTX 8000
#   VARIANTS="strict deploy" bash run_repeat_many.sh # both variants (~1 h)
#   SEEDS=3 bash run_repeat_many.sh                  # 3 random-control seeds (slower)
#   OUT=results_repeat120_b bash run_repeat_many.sh  # another output folder
#   ASSUME_YES=1 nohup bash run_repeat_many.sh > repeat120.nohup 2>&1 &
#
# Notes
#  * This test drops layers s..27 (keeps the early layers), s = 0, 4, ..., 24, 28.
#    s = 28 modifies nothing and is the baseline.
#  * Words the clean model already gets wrong are skipped by the script (the summary
#    prints how many were kept per class).
#  * Time is my estimate, not measured: ~40 short runs per word and variant.
#
# Needs: torch, transformers (same environment as the other experiments).
# If bash complains about '\r':  sed -i 's/\r$//' run_repeat_many.sh
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
NEEDLES="${NEEDLES:-needle_test/results/needle_v3/exp3_needles.csv}"
OUT="${OUT:-results_repeat120}"
VARIANTS="${VARIANTS:-strict}"
SEEDS="${SEEDS:-1}"
S_STEP="${S_STEP:-4}"

[ -f "$NEEDLES" ] || { echo "missing $NEEDLES"; exit 1; }
[ -f run_kv_repeat_test.py ] || { echo "missing run_kv_repeat_test.py"; exit 1; }
"$PYTHON" -c "import torch, transformers" || { echo "missing packages: pip install torch transformers"; exit 1; }

mkdir -p "$OUT"
WORDS="$OUT/words.txt"
{ echo "unhappiness"; tail -n +2 "$NEEDLES" | cut -d, -f1 | tr -d '\r'; } | awk '!seen[$0]++' > "$WORDS"
echo "$(wc -l < "$WORDS") words in $WORDS"

if [ -e "$OUT/kv_repeat_120.csv" ] && [ "${ASSUME_YES:-0}" != "1" ]; then
  read -r -p "$OUT/kv_repeat_120.csv exists and will be overwritten. Continue? [y/N] " ans
  [ "$ans" = "y" ] || exit 1
fi

# $VARIANTS is left unquoted on purpose (word splitting).
# shellcheck disable=SC2086
"$PYTHON" run_kv_repeat_test.py \
  --model "$MODEL" --words_file "$WORDS" \
  --variants $VARIANTS --sets prefix last word control --modes drop \
  --s_step "$S_STEP" --control_seeds "$SEEDS" \
  --dtype float32 --attn eager \
  --out "$OUT/kv_repeat_120.csv" 2>&1 | tee "$OUT/run.log"

"$PYTHON" summarize_repeat_many.py --csv "$OUT/kv_repeat_120.csv" --needles "$NEEDLES" \
  | tee "$OUT/summary.txt"

echo "done. results in $OUT:"
ls -1 "$OUT"
