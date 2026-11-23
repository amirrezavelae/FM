# Dropping subword KV entries: what we tested and what we found

Model: Qwen2.5-7B (28 layers), float32, eager attention. "Drop" means the chosen cache positions are masked out of attention in the chosen layers. Positions and RoPE are unchanged.

## 1. Where we started

The proposal ("Dropping Subword KV Entries After Detokenization") rests on the claim from Kaplan et al. and Busigin & Pinter that the **last subtoken of a multi-token word carries the whole word** after the early layers. If so, the KV of the non-final subtokens (about 8.5–9% of cache positions on WikiText) can be dropped from layer `s` upward while keeping the early layers. At `s = 16` that would save about 3.7% of the cache, with no loss in generation quality.

I haven't read those papers myself. The claim is taken as the proposal states it.

## 2. What we did and what came out

| # | Experiment | Result |
|---|---|---|
| 0 | **Single-word repeat test** (`unhappiness`, few-shot "Repeat:") | Prefix drop was harmless at every `s`. Dropping the last piece or the whole word failed for `s ≤ 22`. This looked like support. |
| 1 | **WikiText next-token test** (200 chunks × 3 seeds; 512 context + 64 continuation tokens). Compares dropping the prefix pieces ("Ours") with the same number of random positions ("Random"). | **Ours is worse than Random.** At `s=16`: KL to Full 0.040 vs 0.011. At `s=0`: 0.039 vs 0.019. The cost is concentrated on tokens that re-copy a dropped word. |
| 1b | **`s` sweep** (100 chunks) | Ours stays at KL 0.034–0.039 for every `s ≤ 20` and only falls at `s=24` (0.021) and `s=27` (0.007). Random falls smoothly (0.021 → 0.0006). Keeping more layers helps only at the very top, which is not the gradual improvement the hypothesis predicts. |
| 2 | **Opposite test**: drop the low layers instead (windows `a:b`) | Dropping layers 0–23 looks cheap (7.25% saving, KL 0.0143), but Random is just as cheap there (0.0157). The cost sits in layers 24–27 (0.0214 vs 0.0025). The cheapness comes from ordinary tokens. On the ~5% of tokens that re-copy a dropped word, Ours is clearly worse than Random: later pieces at `0:16` KL 0.081 vs 0.0015, first piece at `24:28` 0.306 vs 0.005. |
| 3 | **Needle test**: hide `The secret word is <needle>.` in about 2,000 tokens, then ask for it (120 needles: common, rare, novel; 3 depths) | First two versions were invalid: a scoring bug (the model opens with a `"` and spells the word without a leading space). After the fix, Full = 0.89, and 1.00 on the 322 prompts used below. See the next table. |
| 4 | **Repeat test on 121 words** | The `unhappiness` result was an outlier. See the table after the needle one. |

**Needle test, exact match on the 322 prompts Full got right:**

| Window (layers dropped) | Prefix pieces dropped | Last piece only | All pieces | Random positions |
|---|---|---|---|---|
| 0:16 | 0.54 | 0.98 | 0.33 | 1.00 |
| 0:24 | 0.39 | 0.24 | 0.03 | 1.00 |
| 16:28 (the proposal's `s=16`) | 0.18 | 0.20 | 0.00 | 1.00 |
| 24:28 | 0.48 | 0.96 | 0.25 | 1.00 |
| 0:28 | 0.13 | 0.14 | 0.00 | 1.00 |

- By class at `0:16`, prefix dropped: common 0.79, rare 0.62, novel 0.20.
- By piece count: 2 pieces 0.90, 3 pieces 0.50, 4+ pieces 0.15.
- A small strict-variant check (30 needles, 2 windows) gave the same picture.

**Repeat test on 121 words, strict variant, prefix dropped (exact match by start layer `s`, layers `s..27` dropped):**

| Group | s=0 | s=20 | s=24 | s=28 (clean) |
|---|---|---|---|---|
| all (121) | 0.31 | 0.32 | 0.69 | 1.00 |
| common (40) | 0.57 | 0.57 | 0.85 | 1.00 |
| rare (40) | 0.35 | 0.38 | 0.80 | 1.00 |
| novel (40) | 0.00 | 0.00 | 0.42 | 1.00 |

- Random controls stay at 1.00. Dropping the last piece fails for `s ≤ 20` and is 1.00 at `s=24`.
- When the model cannot read the word it usually outputs the previous demonstration (`window`) or spells letters out. When the last piece is dropped, the answer is cut off or garbled at the missing end (`headhunting` → `headh`).

## 3. What we figured out

1. **Dropping the prefix pieces costs real accuracy whenever the word has to be reproduced.** Every experiment agrees. The `unhappiness` result in experiment 0 was one easy word.
2. **The part of the word whose KV you hide is the part that breaks.** Hiding the prefix in the top layers breaks the first piece. Hiding it in the low layers breaks the later pieces. Hiding the last piece breaks the end.
3. **The last piece carries the word only if the model already knows the word.** Dropping the prefix from layer 0 on the repeat test, split at the median `prior_nll` of each class:

   | Class | Familiar half | Unfamiliar half |
   |---|---|---|
   | common | 0.80 | 0.35 |
   | rare | 0.60 | 0.10 |
   | novel | 0.00 | 0.00 |

   More pieces means more loss.
4. **There's no free layer window.** The prefix pieces are needed in layers 20–23 and again in 24–27. The last piece is needed in layers 16–23 but not in 24–27.
5. **The average-token metrics hid all of this.** Exp 1's headline KL and the opposite test's "cheap" window are dominated by ordinary tokens, where dropping prefix pieces is harmless. The damage sits on the few tokens that re-use a dropped word.

## 4. Conclusion

The claim "the last subtoken carries the whole word, so the earlier pieces' KV is redundant" does **not** carry over to KV dropping as the proposal uses it. It holds partly, for familiar words with few pieces, and with a log-probability cost even there. It fails for rare or novel words and for anything that must be reproduced verbatim: names, identifiers, quoted spans. "Misleading" here means the claim doesn't transfer to this use. It does not mean the papers' own measurements are wrong, and I haven't checked what they measure.

## 5. Caveats

- One model, and mostly exact-match copying tasks. Meaning-only tasks are untested.
- The needle test has 40 needles per class. The strict variant was run on only 30 needles and 2 windows.
- The repeat test used one control seed, drawn from the demonstration region only, and a two-demonstration template.
- The opposite test, needle test and repeat test use different layer windows, so the numbers aren't directly comparable across them.
- Exp 2 (SQuAD) was not run. Extractive answers are copying, so I'd expect the same failure.

## 6. Open next steps

- **Quantize instead of drop.** The proposal says "approximate", and in the old one-word grid q4 and q2 never broke `unhappiness`. The repeat script already supports `q8 q4 q3 q2` and needs a `MODES` option in `run_repeat_many.sh`.
- **Non-copy probe.** Ask about the word's meaning, not its spelling, over real words only. That is the fair test of the papers' claim.
- **Band run for the prefix** (`0:8 8:16 16:24 24:28`) to separate overlapping windows.

## 7. Where things are

| Experiment | Code | Results |
|---|---|---|
| 0 | `run_kv_repeat_test.py`, `KV_REPEAT_TEST.md` | `results_qwen.csv`, `log_qwen.txt` |
| 1, 1b | `subword_kv/exp1_wikitext.py`, `run_exp1.sh` | `subword_kv/results/exp1_pp`, `exp1_sweep` |
| 2 | `opposite_test/` | `opposite_test/results/opposite` |
| 3 | `needle_test/exp3_needle.py`, `run_exp3.sh` | `needle_test/results/needle_v3`, `needle_v3_strict` (earlier `needle`, `needle_v2` are invalid) |
| 4 | `run_repeat_many.sh`, `summarize_repeat_many.py` | `results_repeat120/` |
