#!/usr/bin/env bash
# Exp 3: needle retrieval after the needle's prefix pieces are dropped.
# Per layer window, the needle's prefix / last / all pieces, random haystack
# positions, and the original "every haystack word" design are compared, split
# by needle class (common / rare / novel), pieces, depth and prior_nll.
#
#   bash run_exp3.sh                         # preview, pause, then the full run (~1 h)
#   ASSUME_YES=1 nohup bash run_exp3.sh > exp3.nohup 2>&1 &
#   bash run_exp3.sh --resume                # continue an interrupted run
#   bash run_exp3.sh --overwrite             # start over in an existing OUT dir
#   OUT=results/needle_v2 bash run_exp3.sh   # keep an earlier run's folder (the first run,
#                                            # before the quote-aware scoring, is invalid)
#
# Useful overrides (environment variables):
#   N_PER_CLASS=20 bash run_exp3.sh          # quick pass (~30 min)
#   N_PER_CLASS=100 bash run_exp3.sh         # the written design's 100 needles per class
#   WINDOWS="0:24 24:28" bash run_exp3.sh    # other layer windows (a:b = layers a..b-1)
#   VARIANTS="strict" N_PER_CLASS=10 WINDOWS="0:28 0:24" CONDITIONS="needle_prefix random_hay" bash run_exp3.sh
#       strict = nothing after the needle ever sees its clean prefix; each condition
#       re-runs the rest of the prompt (~4 s each), so keep it small.
#   EXTRA="ar=../Tokens2Words/word_lists/top_5k_arabic_words.txt he=../Tokens2Words/word_lists/top_5k_hebrew_words_without_nikud.txt"
#       adds the Tokens2Words Arabic / Hebrew lists as extra needle classes.
#
# Uses ../subword_kv and ../opposite_test (shared code).
# Needs: torch, transformers, datasets, regex, numpy
# If bash complains about '\r':  sed -i 's/\r$//' run_exp3.sh
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # 1 Quadro RTX 8000 (48 GB)
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="${OUT:-results/needle}"
N_PER_CLASS="${N_PER_CLASS:-40}"
WINDOWS="${WINDOWS:-0:16 0:24 16:28 24:28 0:28}"
VARIANTS="${VARIANTS:-deploy}"
CONDITIONS="${CONDITIONS:-needle_prefix needle_last needle_all random_hay all_words random_all}"
SEEDS="${SEEDS:-3}"
EXTRA="${EXTRA:-}"
mkdir -p "$OUT"

for f in ../subword_kv/exp1_wikitext.py ../opposite_test/opposite_wikitext.py; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
"$PYTHON" -c "import torch, transformers, datasets, regex, numpy" \
  || { echo "missing packages: pip install torch transformers datasets regex numpy"; exit 1; }

# $WINDOWS / $CONDITIONS / $VARIANTS / $EXTRA are left unquoted on purpose (word splitting).
# shellcheck disable=SC2086
COMMON_ARGS=(--model "$MODEL" --out_dir "$OUT" --n_per_class "$N_PER_CLASS"
             --windows $WINDOWS --conditions $CONDITIONS --variants $VARIANTS
             --random_seeds "$SEEDS" --dtype float32 --attn eager)
[ -n "$EXTRA" ] && COMMON_ARGS+=(--extra_list $EXTRA)

# Check 2 (CPU only): read the needles and a few prompts by eye.
"$PYTHON" exp3_needle.py "${COMMON_ARGS[@]}" --preview 6 2>&1 | tee "$OUT/preview.txt"

if [ "${ASSUME_YES:-0}" != "1" ]; then
  read -r -p "Do the needles and prompts look right? [y/N] " ans
  [ "$ans" = "y" ] || { echo "stopped; fix the needle or prompt rules first"; exit 1; }
fi

# Main run. The hook check and the window check run first inside the script and abort on failure.
# Check 3 (Full exact match near 1.0) is printed in the summary at the end.
"$PYTHON" exp3_needle.py "${COMMON_ARGS[@]}" "$@" 2>&1 | tee -a "$OUT/run.log"

echo "done. results in $OUT:"
ls -1 "$OUT"
