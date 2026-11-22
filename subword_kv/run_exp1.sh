#!/usr/bin/env bash
# Exp 1: next-token distribution on WikiText-103 (Full vs Ours vs Random),
# now with per-position metrics, role slices and chunk inspection.
#
#   bash run_exp1.sh                      # word check, pause for you to read it, full run
#   ASSUME_YES=1 nohup bash run_exp1.sh > exp1.nohup 2>&1 &   # unattended
#   bash run_exp1.sh --resume             # continue an interrupted run
#   bash run_exp1.sh --overwrite          # start over in an existing OUT dir
#
# Output goes to results/exp1_pp by default, so the first run in results/exp1
# (no per-position data) stays intact. Override with OUT=...
#
# Needs: torch, transformers, datasets, regex, numpy
#   pip install datasets regex
# If the file came from Windows and bash complains about '\r':  sed -i 's/\r$//' run_exp1.sh
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # 1 Quadro RTX 8000 (48 GB)
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="${OUT:-results/exp1_pp}"
N_INSPECT="${N_INSPECT:-8}"          # worst chunks to print after the run
mkdir -p "$OUT"

"$PYTHON" -c "import torch, transformers, datasets, regex, numpy" \
  || { echo "missing packages: pip install torch transformers datasets regex numpy"; exit 1; }

# Check 2: word log, CPU only. Read the dropped/excluded words of 5 chunks by eye.
"$PYTHON" exp1_wikitext.py --model "$MODEL" --out_dir "$OUT" --words_only 5 \
  2>&1 | tee "$OUT/word_log_check.txt"

if [ "${ASSUME_YES:-0}" != "1" ]; then
  read -r -p "Do the dropped and excluded words look right? [y/N] " ans
  [ "$ans" = "y" ] || { echo "stopped; fix the word rules first"; exit 1; }
fi

# Main run: 200 chunks x (1 Full + 2 s values x (1 Ours + 3 Random)) = 1,800 runs.
# Check 1 (hook check) runs first inside the script and aborts on failure.
# The summary, paired and slice tables are printed and written at the end.
"$PYTHON" exp1_wikitext.py \
  --model "$MODEL" \
  --out_dir "$OUT" \
  --n_chunks 200 --ctx_len 512 --cont_len 64 \
  --s_values 0 16 \
  --random_seeds 3 \
  --dtype float32 --attn eager \
  "$@" 2>&1 | tee -a "$OUT/run.log"

# Failure examples, CPU only: raw text and the tokens where Ours raised the loss most.
# a18-t963 is the chunk with few repeated names but a large top-1 drop.
"$PYTHON" exp1_wikitext.py --model "$MODEL" --out_dir "$OUT" \
  --inspect_worst "$N_INSPECT" --inspect wt103-test-a18-t963 --inspect_s 0 \
  2>&1 | tee "$OUT/inspect_worst_s0.txt"

echo "done. results in $OUT:"
ls -1 "$OUT"
