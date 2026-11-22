#!/usr/bin/env bash
# Opposite test: drop the prefix pieces in a WINDOW of layers [a, b)
# (Exp 1 dropped layers s..27, i.e. kept the early layers; this drops the low
# layers instead and keeps the high ones, plus single 4-layer bands).
#
#   bash run_opposite.sh                       # ~12 min on the RTX 8000
#   ASSUME_YES=1 nohup bash run_opposite.sh > opposite.nohup 2>&1 &
#   bash run_opposite.sh --resume              # continue an interrupted run
#   bash run_opposite.sh --overwrite           # start over in an existing OUT dir
#   WINDOWS="0:14 14:28" bash run_opposite.sh  # your own windows (a:b = layers a..b-1)
#
# Uses ../subword_kv (shared code and the Exp 1 sweep for the comparison column).
# Same chunks, words, Random positions and metrics as Exp 1, so run it with the
# default 100 chunks / 1 random seed to compare with results/exp1_sweep.
#
# Needs: torch, transformers, datasets, regex, numpy
# If bash complains about '\r':  sed -i 's/\r$//' run_opposite.sh
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # 1 Quadro RTX 8000 (48 GB)
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="${OUT:-results/opposite}"
N_CHUNKS="${N_CHUNKS:-100}"
SEEDS="${SEEDS:-1}"
WINDOWS="${WINDOWS:-0:4 0:8 0:12 0:16 0:20 0:24 4:8 8:12 12:16 16:20 20:24 24:28}"
SWEEP="${SWEEP:-../subword_kv/results/exp1_sweep}"   # '' to skip the comparison column
mkdir -p "$OUT"

[ -f ../subword_kv/exp1_wikitext.py ] || { echo "missing ../subword_kv/exp1_wikitext.py"; exit 1; }
"$PYTHON" -c "import torch, transformers, datasets, regex, numpy" \
  || { echo "missing packages: pip install torch transformers datasets regex numpy"; exit 1; }

if [ ! -d "$SWEEP" ]; then
  echo "note: $SWEEP not found, the complement column will be skipped"
fi

# The hook check and the new window check run first inside the script and abort on failure.
# $WINDOWS is left unquoted on purpose so each "a:b" becomes its own argument.
# shellcheck disable=SC2086
"$PYTHON" opposite_wikitext.py \
  --model "$MODEL" \
  --out_dir "$OUT" \
  --n_chunks "$N_CHUNKS" --random_seeds "$SEEDS" \
  --windows $WINDOWS \
  --compare_sweep "$SWEEP" \
  --dtype float32 --attn eager \
  "$@" 2>&1 | tee -a "$OUT/run.log"

echo "done. results in $OUT:"
ls -1 "$OUT"
