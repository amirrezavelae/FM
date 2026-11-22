# Opposite test: drop the low layers, keep the high ones

Exp 1 (`../subword_kv`) dropped the prefix pieces' KV in layers `s..27` and kept layers `0..s-1`, as in the proposal. Its sweep showed the cost sits in layers ≥ 20: dropping layers 20–27 cost as much as dropping all 28. This test asks the complementary question.

**Question:** can the prefix pieces be dropped in the *low* layers (`0..e-1`) while the high layers keep them?

A window `a:b` means: drop in layers `a, a+1, …, b-1`.

| Windows | Purpose |
|---|---|
| `0:4 0:8 0:12 0:16 0:20 0:24` | the opposite test proper: drop the bottom `e` layers |
| `4:8 8:12 12:16 16:20 20:24 24:28` | one 4-layer band at a time, to see which layers read the prefix pieces |

Memory saving of a window is `drop_frac × (b − a) / 28` (drop_frac ≈ 8.9% on WikiText), so `0:20` saves about 6.4% at most.

## Run

```bash
bash run_opposite.sh
```

About 12 minutes (100 chunks, 25 continuation runs per chunk). The script runs two checks first and aborts if either fails:

1. Exp 1's hook check (empty drop = default logits; a big drop changes them).
2. A window check: window `[0, 28)` equals the old "from layer 0" drop, window `[27, 28)` equals the old "from layer 27" drop, and `[0,1)`, `[0,14)`, `[14,28)` each visibly change the logits.

## Same as Exp 1

Data (WikiText-103 test, 512 context + 64 continuation tokens), word rules, protected positions, the Random control (same seeds, so the same positions), the metrics (KL to Full, top-1 agreement, change in log-loss) and the target-token roles (`rep_first`, `rep_rest`, `new_*`, `other`). The word logs are identical to Exp 1's, so they are not repeated here.

## Outputs (`results/opposite/`)

| File | Contents |
|---|---|
| `opposite_examples.jsonl` | one line per chunk, with per-position data |
| `opposite_summary.csv` | one row per condition |
| `opposite_paired.csv` | Ours−Random and Ours−Full per window, with chunk and article CIs |
| `opposite_slices.csv` | the same by target-token role |
| `opposite_meta.json` | arguments, versions, check results |

The console table has a "complement" column: the Exp 1 sweep's KL when the *other* layers `[b, 28)` are dropped instead (same chunks, same Random positions). Needs `../subword_kv/results/exp1_sweep`.

## Reading the result

| Result | Meaning |
|---|---|
| `0:e` Ours ≈ Random for e up to ~16–20 | the low layers do not need the prefix pieces: a cheap version of the method exists, saving ≈ `drop_frac × e/28` |
| `0:e` Ours ≫ Random | the prefix pieces are read in low layers too; there is no free window |
| a single band is costly (e.g. `20:24`) | that band reads the prefix pieces; compare with the `rep_first` column to see if it is the copy circuit |

Rebuild the tables without a GPU: `python opposite_wikitext.py --out_dir results/opposite --summarize_only`.
