# Subword KV Dropping: Experiment Setup

Nov 22, 2026 · @Amirreza

## Goal

These experiments test whether, after prefill, the KV entries of the non-final subtokens of every multi-token word can be removed without hurting simple generation tasks. The claim holds only if removing them costs less than removing the same number of random entries.

The repeat test supports it for one word (`unhappiness` on Qwen2.5-7B: dropping `' unh'` at every layer left the output unchanged). These three experiments check whether it holds across many words and real prompts.

## Fixed settings

Every experiment uses the same settings, so only the condition changes between runs.

| Setting | Value | Why |
| --- | --- | --- |
| Model | Qwen2.5-7B (base), 28 layers | Ungated; Llama-3.1-8B later if access comes |
| Hardware | 1 Quadro RTX 8000, `CUDA_VISIBLE_DEVICES=``0` (gpustat GPU 6) | About 30 GB in float32 |
| Precision | float32, eager attention | Turing has poor bf16 support; the drop mask needs eager or sdpa |
| Intervention | Drop (mask out of attention), positions and RoPE unchanged | Quantization was not word-specific in the repeat test |
| Start layer s | 0 and 16 | s = 0 drops at every layer (largest saving); s = 16 keeps layers 0–15, where the word may still be read |
| When to drop | After prefill of the whole prompt (`deploy`) | The real use case |
| Decoding | Greedy | Deterministic, so conditions compare example by example |

Ours and Random each run at both values of s; Full runs once. Report every result separately for s = 0 and s = 16.

## Conditions

Each example runs under three conditions, from the same prefill.

| Condition | What is dropped | Role |
| --- | --- | --- |
| Full | Nothing | Reference |
| Ours | Every non-final subtoken of every eligible multi-token word in the prompt | The method |
| Random | The same number k of positions, sampled at random | Same memory saving, no word structure |

Rules for Random:

- k is matched per example: Random drops exactly as many entries as Ours dropped in that prompt.
- Positions are sampled from the prompt excluding the protected positions (below) and excluding the positions Ours drops, so the two sets never overlap.
- 3 seeds per example; report the mean over seeds.

Protected positions, never dropped in any condition:

- the first token of the sequence (attention sink);
- the final prompt token;
- special and template tokens.

If Ours drops nothing in an example (no eligible words), keep the example in Full but leave it out of the Ours vs Random comparison, and count how many such examples there were.

## Word definition

A word is a maximal run of letters in the prompt text. A word is eligible when the tokenizer splits it into 2 or more tokens that start and end exactly on the word's boundaries.

How to find them:

1. Find letter runs in the prompt string with the regex `\p{L}+` (the `regex` package).
2. Tokenize the prompt with `return_offsets_mapping=True`.
3. Map each word to the tokens whose character spans overlap it.
4. Keep the word only if every one of those tokens lies inside the word, except that the first token may also carry one leading space.
5. Drop positions = all of the word's tokens except the last.

Excluded words, logged with a reason:

| Reason | Example | Why excluded |
| --- | --- | --- |
| `single_token` | `cat` | Nothing to drop |
| `misaligned` | a token covering `ing,` | The token also carries non-letter text, so dropping it removes more than part of a word |
| `protected` | a word whose prefix is the sequence's first token | Overlaps a protected position |

Numbers and digit strings never qualify, because `\p{L}+` matches letters only. A word broken by an apostrophe or hyphen (`don't`, `well-known`) is treated as separate words, each checked on its own.

The repeat test placed each word after a space. Mid-sentence words with leading spaces are the common case, and step 4 handles them.

## Experiments

Three experiments, from cheapest to hardest. Each needs only the one model: no API and no larger model.

### Exp 1: Next-token distribution on WikiText

The question is how much dropping changes what the model predicts next.

- **Data:** WikiText-103 test set, 200 chunks. Each chunk is 512 context tokens followed by the next 64 tokens of the same text.
- **Procedure:** prefill the 512-token context, apply the condition to the context only, then feed the 64 real continuation tokens and record the next-token distribution at each of them. Nothing in the continuation is dropped.
- **Metrics,** averaged over the 64 positions and then over chunks:
  - KL(Full ‖ condition);
  - top-1 agreement with Full;
  - change in log-loss of the real next token relative to Full.

This uses the real text as the continuation, so it needs no reference answers and no other model.

### Exp 2: Extractive QA on SQuAD

The question is whether the model can still copy answer spans out of the context. This is the hardest case for the method, because answers are often multi-token names.

- **Data:** SQuAD v1.1 validation, 500 questions with a fixed seed.
- **Prompt:** 2 fixed examples from the training set, then `Context: … Question: … Answer:`.
- **Generation:** greedy, up to 20 tokens, stop at a newline.
- **Metrics:** exact match and token F1 against the gold answers, using the standard SQuAD normalization.
- **Key slice:** report the same metrics on questions whose gold answer contains a word that Ours dropped from the context. If the method fails, it fails here first.

### Exp 3: Needle retrieval

The question is whether a single rare word can be retrieved from a long context after its prefix entries are dropped.

- **Needles:** 100 words that Qwen splits into 2 or more tokens, at least 30 of them into 3 or more. Draw them from the Tokens2Words word lists.
- **Haystack:** about 2,000 tokens of WikiText filler. Insert `The secret word is <needle>.` at 10%, 50% or 90% depth.
- **Prompt end:** `What is the secret word? The secret word is`
- **Generation:** greedy, up to 10 tokens.
- **Metric:** exact match of the needle, reported overall, by depth and by number of subtokens.

Ours also drops the prefixes of the haystack's words, as it would in real use.

### Run size

| Experiment | Examples | Runs per example | Total runs |
| --- | --- | --- | --- |
| Exp 1 | 200 chunks | 9 | 1,800 |
| Exp 2 | 500 questions | 9 | 4,500 |
| Exp 3 | 100 needles × 3 depths | 9 | 2,700 |

Runs per example = 1 Full + 2 values of s × (1 Ours + 3 Random seeds).

Run the experiments in order, and stop to look after Exp 1.

## Logging

Every dropped entry is traceable to a word, and every excluded word to a reason.

**Per example:** one line of JSON in `<exp>_examples.jsonl`:

```json
{
  "exp": "squad", "id": "56be4db0acb8001400a502ec",
  "prompt_len": 312, "k": 21, "drop_frac": 0.067,
  "dropped_words": [
    {"word": "unhappiness", "pieces": [" unh", "appiness"], "dropped_positions": [148]}
  ],
  "excluded_words": [{"word": "Denver", "reason": "misaligned"}],
  "random_positions": {"0": [17, 90], "1": [33, 201], "2": [5, 260]},
  "outputs": {"full": "Denver Broncos", "ours": "Denver Broncos", "random_0": "Broncos"},
  "metrics": {"full": {"em": 1, "f1": 1.0}, "ours": {"em": 1, "f1": 1.0}}
}
```

The values above are illustrative. For Exp 1, `outputs` is replaced by the per-chunk KL, top-1 agreement and log-loss change.

**Per experiment:** three summary files.

| File | Contents |
| --- | --- |
| `<exp>_summary.csv` | One row per condition: each metric's mean and 95% CI, mean k, mean drop fraction |
| `<exp>_dropped_words.csv` | Every dropped word with its pieces and how often it was dropped, sorted by count |
| `<exp>_excluded_words.csv` | Every excluded word with its reason and count |

The memory saving equals the drop fraction when s = 0, since every layer is dropped. For s > 0, multiply it by (28 − s) / 28.

## Checks and reading the results

Run three checks before trusting any number.

1. **Hook check:** Full run with the drop hook installed and an empty drop list gives the same logits as Full without the hook (max difference near float32 noise).
2. **Word log check:** before any GPU run, print the dropped and excluded words for 5 prompts and read them by eye.
3. **Baseline check:** Full scores in the normal range for a 7B base model (roughly 75–85 F1 on SQuAD 2-shot, near 100% on the needle). If not, fix the prompt first.

**Statistics:** every comparison is paired by example. Report Ours − Random and Ours − Full as mean differences, with 95% bootstrap confidence intervals over examples (1,000 resamples).

**How to read the outcome:**

| Result | Meaning |
| --- | --- |
| Ours ≈ Full, Random clearly worse | Supports the method: the dropped entries were redundant, and that is specific to word structure |
| Ours ≈ Random ≈ Full | Inconclusive: at this drop fraction almost any entries can go; the result is not about words |
| Ours worse than Full, but better than Random | Partial support: word prefixes are cheaper to drop, but not free; try higher s |
| Ours worse than Random | The hypothesis fails for this task |

If the outcome is inconclusive because the drop fraction is low (likely about 10% of tokens in English), the natural next test is text with more subtokens per word, such as Persian or Turkish from FLORES-200.
