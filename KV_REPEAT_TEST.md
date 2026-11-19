# KV-cache repeat test

Tests whether the model can still repeat a multi-subtoken word (e.g. `unhappiness`) after the KV entries of the word's **earlier subtokens** are dropped or quantized from layer `s` upward.

**Hypothesis:** once the last subtoken has been processed, it carries the whole word, so later tokens don't need the earlier subtokens' KV.

## Setup

1. Put `run_kv_repeat_test.py` in the Tokens2Words repo root. It only needs `torch` and `transformers`.
2. Log in to Hugging Face, because Llama is gated:
   ```bash
   huggingface-cli login
   ```

## Run

```bash
# one word, full sweep
python run_kv_repeat_test.py --words unhappiness

# quick pass
python run_kv_repeat_test.py --words unhappiness --modes drop q4 --s_step 4

# many words (one per line)
python run_kv_repeat_test.py --words_file words.txt
```

Results are written to `kv_repeat_results.csv`.

## What it does

The prompt is a few-shot repeat task:

```
Word: cat
Repeat: cat

Word: window
Repeat: window

Word: unhappiness
Repeat:
```

The script modifies the cache at layers `≥ s`, then generates. It records greedy exact match and the log-probability of the target.

| Option | Values | Meaning |
|---|---|---|
| `--variants` | `strict` | Modify the cache right after the word's last subtoken, before the rest of the prompt runs. This is the clean test of the hypothesis. |
| | `deploy` | Modify the cache after the prompt, just before its final token. Tokens after the word saw the clean KV, so the word can survive through them. |
| `--sets` | `prefix` | All subtokens except the last (the hypothesis). |
| | `last` | Only the last subtoken. This should hurt more. |
| | `word` | All subtokens. This should break the task. |
| | `control` | The same number of random earlier positions, never BOS, averaged over 3 seeds. |
| `--modes` | `drop` | Mask the positions out of attention. Positions and RoPE are unchanged. |
| | `q8` `q4` `q3` `q2` | Fake quantization: per-token asymmetric min-max, with groups along `head_dim` (`--group_size`). |

`s` runs from 0 to the number of layers in steps of `--s_step`. At the last value nothing is modified, so it serves as the baseline.

## Checks before reading results

- **Subtokens:** the printed split is the real one. The word has a leading space here, so it may differ from `un | h | appiness`. Change `--template` if needed.
- **Sanity:** both `argmax_equal` values must be `True`, and `maxdiff` should be at bf16 noise level.
- **Baseline:** `exact=True` is required. Otherwise the clean model can't do the task, and the word is skipped (override with `--keep_if_baseline_fails`).

## Reading the grid

Each row is one (variant, set, mode), and each column is a value of `s`. `✓` means an exact repeat and `·` means a failure.

| Observation | Meaning |
|---|---|
| `strict prefix` gives ✓ for `s ≥ s*` | The hypothesis holds from layer `s*`. This is the candidate layer. |
| `word` fails at every `s` | The test can detect damage. |
| `last` fails more than `prefix` | The word's information sits in the last subtoken. |
| `control` stays ✓ | The effect is specific to the word. |
| `strict` fails but `deploy` succeeds | The word survives through the tokens after it, not the last subtoken. |

## Caveat: copying vs. meaning

Copying text from context relies on induction heads, which read the earlier subtokens' KV directly. A failure in the repeat task therefore doesn't refute the hypothesis. To check the meaning instead of the spelling, run:

```bash
python run_kv_repeat_test.py --words unhappiness --target happiness \
  --template $'The opposite of sadness is joy.\nThe opposite of hot is cold.\nThe opposite of {word} is'
```

## Notes

- The quantization is fake (quantize, then dequantize), so it measures accuracy only. Estimate memory savings analytically: positions × layers ≥ `s` × bits.
- Use `--attn eager` (the default) or `sdpa`. Flash attention ignores the custom mask.
