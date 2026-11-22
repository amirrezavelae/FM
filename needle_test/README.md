# Exp 3: needle retrieval, with the three changes

A rare multi-token word (the needle) is hidden in about 2,000 tokens of WikiText as `The secret word is <needle>.` at 10%, 50% or 90% depth. The prompt ends with `What is the secret word? The secret word is`. The question is whether the model can still produce the needle after the KV of the needle's non-final pieces has been removed.

## The three changes from the written design

1. **Only the needle is dropped.** Random never touches the needle by chance, so matching only k would make Ours look catastrophic by construction. The conditions are now:

   | Condition | What is dropped | Role |
   |---|---|---|
   | `needle_prefix` | every needle piece except the last | the hypothesis |
   | `needle_last` | only the last piece | control: should hurt |
   | `needle_all` | every needle piece | control: should break (the test has teeth) |
   | `random_hay` | as many random haystack positions as `needle_prefix` | control: should be safe |
   | `all_words` | the prefix pieces of every eligible haystack word, plus the needle | the original design |
   | `random_all` | as many random haystack positions as `all_words` | control for `all_words` |

   Random positions never include position 0, the needle sentence, the question, the final token or the needle itself. 3 seeds, averaged.
2. **Layer windows `a:b` (drop in layers a..b-1).** Default `0:16 0:24 16:28 24:28 0:28`: the cheap windows from the opposite test (`0:16`, `0:24`), the proposal's `s=16` (`16:28`), the top four layers where Exp 1 found the cost (`24:28`), and every layer (`0:28`).
3. **Needle classes**, to test "common words survive, rare words don't":

   | Class | Words |
   |---|---|
   | `common` | lowercase English words that occur at least 10 times in WikiText val+test and that Qwen splits into 2+ tokens |
   | `rare` | real words that occur exactly once there (9+ letters) |
   | `novel` | invented pronounceable strings, never seen by the model |
   | `--extra_list name=path` | optional extra classes, for example the Tokens2Words Arabic and Hebrew lists |

   Each needle also gets **`prior_nll`**: how surprising it is to the model with no haystack (`The secret word is <needle>`). Low means the model knows it. Results are split by `prior_nll` tercile too, which is a model-based version of rarity.

   At least 30% of each class has 3+ pieces (as in the written design), when enough candidates exist; otherwise a warning is printed.

## Scoring

One teacher-forced pass per condition. `exact` means the argmax at every needle token equals the needle and the next argmax ends the answer (`.`, newline or a closing quote). That is the same as "greedy decoding reproduces the needle".

**Quote handling.** Qwen2.5-7B base opens its answer with a `"` in about 97% of prompts (`The secret word is "vertebrae`), and after the quote it spells the word WITHOUT a leading space: `vert`+`ebr`+`ae`, not the prompt's ` vert`+`ebra`+`e`. Two scoring bugs came from this. v1 compared the first token to the first needle piece and reported 3% Full accuracy. v2 fed the quote but still scored the spaced pieces, so Full was 0.03 again (first_ok = 0 for every Full prompt) although real greedy generation was right 92% of the time. Now, when Full's first answer token is a quote, that quote is fed to every condition and the needle is scored in its no-space tokenization after it (the quote's own probability is not counted). Full decides the format once, so NLLs are comparable across conditions. The run also stops after 12 prompts if Full exact match is below 0.5 (`--no_gate` disables this), so a scoring problem costs 2 minutes, not an hour.

**Full-succeeded restriction.** A prompt the clean model already misses cannot show whether dropping KV costs anything. The summary is therefore printed twice: on all prompts, and on the prompts where Full succeeded (files ending in `_fullok`). Read the second one for retention. The Full run also does real greedy generation (up to 10 tokens) as a cross-check, and `--generate_all` does it for every condition. The summary also reports the change in needle log-loss (`d_nll`).

## Variants

| Variant | Meaning |
|---|---|
| `deploy` (default) | prefill everything except the last prompt token, drop, then answer. The question tokens saw the clean KV, so the word can survive through them. |
| `strict` | prefill up to and including the needle's last piece, drop, then run the rest of the prompt. Nothing after the needle sees the clean prefix, so the word can only live in the last piece. Each condition re-runs the rest of the prompt (~4 s), so use few needles, windows and conditions. |

## Run

```bash
bash run_exp3.sh
```

It first prints the needles and 6 sample prompts (CPU only) and asks you to confirm. Then the main run does a hook check and a window check, aborting on failure. Roughly 1 hour with the defaults (40 needles × 3 classes × 3 depths = 360 prompts, about 11 s each). I have not timed it. `N_PER_CLASS=20` is the quick version; `N_PER_CLASS=100` is the written size. Other overrides are listed at the top of `run_exp3.sh`.

## Outputs (`results/needle/`)

| File | Contents |
|---|---|
| `exp3_examples.jsonl` | one line per prompt: needle, class, depth, pieces, positions, `prior_nll`, results for every window and condition |
| `exp3_summary.csv` | exact match and `d_nll` with 95% CIs, by variant, window, condition and group (all, class, depth, pieces, prior tercile) |
| `exp3_paired.csv` | paired differences: `needle_prefix − random_hay`, `needle_prefix − full`, `needle_last − needle_prefix`, `all_words − random_all` |
| `exp3_needles.csv` | every needle with its class, pieces and `prior_nll` |
| `*_fullok.csv` | the same summary, paired and needle tables restricted to prompts where Full succeeded |
| `exp3_meta.json` | arguments, versions, check results |
| `preview.txt` | the needle and prompt preview |

CIs are a bootstrap over needles, not prompts, because the three depths of one needle are not independent. Rebuild the tables without a GPU with `python exp3_needle.py --out_dir results/needle --summarize_only`.

## Checks before trusting a number

1. **Hook and window check** (automatic): empty drop matches default logits, a big drop changes them, window `[0,28)` equals the old "from layer 0" drop, `[27,28)` equals "from layer 27".
2. **Preview** (`preview.txt`): read the needles and a few prompts; the pieces and the question must look right.
3. **Baseline:** Full exact match should be close to 1.0. If it is not, fix the prompt before reading any other number. If one class (for example `novel`) fails on Full, the model cannot do the task for those needles even with a clean cache; report that class separately and don't read its dropped-KV results as evidence.

## Reading the result

| Result | Meaning |
|---|---|
| `needle_prefix` fine for `common` (or low `prior_nll`), fails for `rare` and `novel` | your theory: the last piece carries words the model knows, not arbitrary strings |
| `needle_prefix` fails for every class | the last piece does not carry the spelling; the hypothesis is about meaning only |
| fine in `0:24`, fails in `24:28` | the opposite design is usable; the top layers read the prefix pieces |
| `strict` fails where `deploy` passes | the word survives through the tokens after it (the question), not through the last piece |
| `needle_last` and `needle_all` do not fail | the test has no teeth: the model is answering without the needle's KV |
| `needle_prefix` ≈ `random_hay` ≈ Full | the needle's prefix is as safe to drop as any position |
