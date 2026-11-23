#!/usr/bin/env bash
# Exp 3, strict variant: nothing after the needle ever sees its clean prefix.
# The word can only survive in the needle's last piece (' e' in ' vert'+'ebra'+'e').
# Compare with results/needle_v3 (deploy) on the same windows and conditions.
#
#   bash run_exp3_strict.sh                         # preview, pause, then the run
#   ASSUME_YES=1 nohup bash run_exp3_strict.sh > strict.nohup 2>&1 &
#   bash run_exp3_strict.sh --resume                # continue an interrupted run
#
# Defaults: 10 needles per class x 3 classes x 3 depths = 90 prompts, windows 0:16 and 24:28
# (where deploy showed needle_last harmless and needle_prefix harmful), 4 conditions, 1 seed.
# Each condition re-runs the rest of the prompt (~4 s), so expect roughly 1 hour. Not timed.
# Override anything from run_exp3.sh the same way, e.g. N_PER_CLASS=20 WINDOWS="0:16 0:24".
set -euo pipefail
cd "$(dirname "$0")"

export VARIANTS="${VARIANTS:-strict}"
export OUT="${OUT:-results/needle_v3_strict}"
export N_PER_CLASS="${N_PER_CLASS:-10}"
export WINDOWS="${WINDOWS:-0:16 24:28}"
export CONDITIONS="${CONDITIONS:-needle_prefix needle_last needle_all random_hay}"
export SEEDS="${SEEDS:-1}"

exec bash run_exp3.sh "$@"
