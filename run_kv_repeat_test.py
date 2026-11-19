"""
KV-cache repeat test for multi-subtoken words.

Question: once the model has processed the last subtoken of a word, can later
tokens still reproduce the word if the KV entries of the EARLIER subtokens are
dropped or quantized from layer s upward?

Two variants:
  strict  - prefill up to and including the word's last subtoken, modify the
            cache, then run the rest of the prompt and generate. Nothing after
            the word ever sees the clean earlier-subtoken KV, so the word can
            only survive through the last subtoken (and the layers below s).
  deploy  - prefill everything except the final prompt token, modify the
            cache, then generate. The tokens between the word and the answer
            saw the clean KV, so information can leak through them. This is
            closer to the proposal (modify after prefill).

Position sets that get modified:
  prefix  - all subtokens of the word except the last (the hypothesis)
  last    - only the last subtoken (should hurt more if the hypothesis holds)
  word    - all subtokens (should break the task; checks the test has teeth)
  control - the same number of random positions from the few-shot part
            (never BOS), averaged over a few seeds

Modes: drop (mask the positions out of attention at layers >= s; position ids
and RoPE are left untouched) and fake quantization at several bit widths
(per-token asymmetric min-max, groups along head_dim).

Metrics: greedy exact match and the teacher-forced log-probability of the
target continuation.

Drop this file in the Tokens2Words repo root (it only needs torch and
transformers) and run, e.g.:
  python run_kv_repeat_test.py --words unhappiness
"""

import argparse
import copy
import csv
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEMPLATE = (
    "Repeat each word exactly.\n\n"
    "Word: cat\nRepeat: cat\n\n"
    "Word: window\nRepeat: window\n\n"
    "Word: {word}\nRepeat:"
)


# --------------------------------------------------------------------------- #
# Cache access (works with both the old key_cache lists and the newer .layers)
# --------------------------------------------------------------------------- #
def get_kv(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def set_kv(cache, i, k, v):
    if hasattr(cache, "layers"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    else:
        cache.key_cache[i], cache.value_cache[i] = k, v


# --------------------------------------------------------------------------- #
# Fake quantization
# --------------------------------------------------------------------------- #
def fake_quant(x, bits, group_size):
    """Per-token asymmetric min-max quantize->dequantize along the last dim."""
    shape, dtype = x.shape, x.dtype
    d = shape[-1]
    g = group_size or d
    assert d % g == 0, f"head_dim {d} not divisible by group size {g}"
    xg = x.float().reshape(*shape[:-1], d // g, g)
    lo = xg.amin(-1, keepdim=True)
    hi = xg.amax(-1, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-8) / levels
    q = ((xg - lo) / scale).round().clamp(0, levels)
    return (q * scale + lo).reshape(shape).to(dtype)


def quantize_cache(cache, positions, start_layer, n_layers, bits, group_size):
    idx = torch.tensor(positions)
    for i in range(start_layer, n_layers):
        k, v = get_kv(cache, i)
        k, v = k.clone(), v.clone()
        p = idx.to(k.device)
        k[:, :, p] = fake_quant(k[:, :, p], bits, group_size)
        v[:, :, p] = fake_quant(v[:, :, p], bits, group_size)
        set_kv(cache, i, k, v)


# --------------------------------------------------------------------------- #
# Dropping: per-layer attention mask via forward pre-hooks
# --------------------------------------------------------------------------- #
class DropHooks:
    """Mask `positions` out of attention in layers >= start_layer.

    The hook replaces the attention mask with an explicit causal additive mask
    that also blocks the dropped positions. Batch size 1, no padding.
    """

    def __init__(self, model, positions, start_layer):
        self.handles = []
        self.positions = list(positions)
        layers = model.model.layers
        for i in range(start_layer, len(layers)):
            attn = layers[i].self_attn
            self.handles.append(
                attn.register_forward_pre_hook(self._make_hook(i), with_kwargs=True)
            )

    def _make_hook(self, layer_idx):
        def hook(module, args, kwargs):
            hs = kwargs.get("hidden_states", args[0] if args else None)
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            past = cache.get_seq_length(layer_idx) if cache is not None else 0
            q_len = hs.shape[1]
            kv_len = past + q_len
            q_pos = torch.arange(past, kv_len, device=hs.device)[:, None]
            k_pos = torch.arange(kv_len, device=hs.device)[None, :]
            allowed = k_pos <= q_pos
            if self.positions:
                allowed[:, torch.tensor(self.positions, device=hs.device)] = False
            mask = torch.zeros(q_len, kv_len, dtype=hs.dtype, device=hs.device)
            mask.masked_fill_(~allowed, torch.finfo(hs.dtype).min)
            kwargs["attention_mask"] = mask[None, None]
            return args, kwargs

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Prompt handling
# --------------------------------------------------------------------------- #
def encode_prompt(tok, template, word):
    prompt = template.format(word=word)
    start = template.index("{word}")
    end = start + len(word)
    enc = tok(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
        offs = [(0, 0)] + offs
    word_pos = [i for i, (a, b) in enumerate(offs) if b > a and a < end and b > start]
    if not word_pos:
        raise ValueError(f"could not locate {word!r} in the tokenized prompt")
    # The word must start and end on token boundaries (a leading space is fine).
    a0, b1 = offs[word_pos[0]][0], offs[word_pos[-1]][1]
    if a0 < start - 1 or b1 != end:
        raise ValueError(f"{word!r} is merged with neighbouring text by the tokenizer")
    return ids, word_pos


@torch.no_grad()
def forward(model, ids, cache):
    x = torch.tensor([ids], device=model.device)
    return model(input_ids=x, past_key_values=cache, use_cache=True).logits[0].float()


@torch.no_grad()
def continue_and_score(model, tok, cache, rest_ids, target_ids, max_new, stop_str):
    """From `cache`, run `rest_ids`, then (a) teacher-forced log-prob of
    target_ids and (b) greedy generation. Consumes a deep copy for (b)."""
    gen_cache = copy.deepcopy(cache)

    logits = forward(model, rest_ids + target_ids, cache)
    lp = F.log_softmax(logits[len(rest_ids) - 1 : -1], -1)
    logprob = lp.gather(1, torch.tensor(target_ids, device=lp.device)[:, None]).sum().item()

    logits = forward(model, rest_ids, gen_cache)
    out = []
    for _ in range(max_new):
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        if stop_str in tok.decode(out) or nxt == tok.eos_token_id:
            break
        logits = forward(model, [nxt], gen_cache)
    text = tok.decode(out, skip_special_tokens=True).split(stop_str)[0].strip()
    return text, logprob


def run_condition(model, tok, clean_cache, rest_ids, target_ids, mode, positions,
                  start_layer, n_layers, bits, group_size, max_new, stop_str):
    cache = copy.deepcopy(clean_cache)
    hooks = None
    if positions and start_layer < n_layers:
        if mode == "drop":
            hooks = DropHooks(model, positions, start_layer)
        else:
            quantize_cache(cache, positions, start_layer, n_layers, bits, group_size)
    try:
        text, logprob = continue_and_score(
            model, tok, cache, rest_ids, target_ids, max_new, stop_str)
    finally:
        if hooks:
            hooks.remove()
    return text, logprob


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sanity_checks(model, ids, split):
    from transformers import DynamicCache
    one = forward(model, ids, None)[-1]
    cache = DynamicCache()
    forward(model, ids[:split], cache)
    two = forward(model, ids[split:], cache)[-1]

    cache = DynamicCache()
    forward(model, ids[:split], cache)
    h = DropHooks(model, [], 0)  # explicit mask, nothing dropped
    try:
        hooked = forward(model, ids[split:], cache)[-1]
    finally:
        h.remove()
    return {
        "split_vs_onepass_maxdiff": (one - two).abs().max().item(),
        "split_vs_onepass_argmax_equal": bool(one.argmax() == two.argmax()),
        "mask_hook_vs_default_maxdiff": (two - hooked).abs().max().item(),
        "mask_hook_argmax_equal": bool(two.argmax() == hooked.argmax()),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--words", nargs="*", default=[])
    ap.add_argument("--words_file", help="one word per line")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE,
                    help="prompt with a {word} slot; the model should answer after it")
    ap.add_argument("--target", default="{word}",
                    help="expected answer; '{word}' for the repeat task")
    ap.add_argument("--variants", nargs="+", default=["strict", "deploy"])
    ap.add_argument("--sets", nargs="+", default=["prefix", "last", "word", "control"])
    ap.add_argument("--modes", nargs="+", default=["drop", "q8", "q4", "q3", "q2"])
    ap.add_argument("--s_step", type=int, default=2)
    ap.add_argument("--group_size", type=int, default=0, help="0 = head_dim")
    ap.add_argument("--control_seeds", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="eager", help="eager or sdpa (not flash)")
    ap.add_argument("--keep_if_baseline_fails", action="store_true",
                    help="run the sweep even if the clean model gets the word wrong")
    ap.add_argument("--out", default="kv_repeat_results.csv")
    args = ap.parse_args()

    words = list(args.words)
    if args.words_file:
        with open(args.words_file) as f:
            words += [w.strip() for w in f if w.strip()]
    if not words:
        words = ["unhappiness"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attn,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    n_layers = model.config.num_hidden_layers
    s_values = sorted(set(list(range(0, n_layers, args.s_step)) + [n_layers]))
    stop_str = "\n"

    from transformers import DynamicCache
    rows = []
    for word in words:
        try:
            ids, word_pos = encode_prompt(tok, args.template, word)
        except ValueError as e:
            print(f"[skip] {e}")
            continue
        pieces = [tok.decode([ids[i]]) for i in word_pos]
        print(f"\n=== {word!r}  subtokens: {' | '.join(repr(p) for p in pieces)}")
        if len(word_pos) < 2:
            print("[skip] single-token word, nothing to drop")
            continue
        target = args.target.format(word=word)
        target_ids = tok(" " + target, add_special_tokens=False)["input_ids"]
        print(f"target continuation tokens: {[tok.decode([t]) for t in target_ids]}")

        sets = {
            "prefix": word_pos[:-1],
            "last": word_pos[-1:],
            "word": word_pos,
        }
        n_ctrl = len(sets["prefix"])
        ctrl_pool = list(range(1, word_pos[0]))  # few-shot region, never BOS

        for variant in args.variants:
            split = word_pos[-1] + 1 if variant == "strict" else len(ids) - 1
            rest = ids[split:]

            sc = sanity_checks(model, ids, split)
            print(f"[{variant}] sanity: {sc}")

            clean = DynamicCache()
            forward(model, ids[:split], clean)
            base_text, base_lp = run_condition(
                model, tok, clean, rest, target_ids, "drop", [], n_layers, n_layers,
                0, args.group_size, args.max_new_tokens, stop_str)
            base_ok = base_text == target
            print(f"[{variant}] baseline: {base_text!r} exact={base_ok} logprob={base_lp:.2f}")
            if not base_ok and not args.keep_if_baseline_fails:
                print(f"[{variant}] model fails the clean task -> skipping this word")
                continue

            for set_name in args.sets:
                if set_name == "control":
                    pos_lists = [(seed, sorted(random.Random(seed).sample(ctrl_pool, n_ctrl)))
                                 for seed in range(args.control_seeds)]
                else:
                    pos_lists = [(None, sets[set_name])]
                for mode in args.modes:
                    bits = int(mode[1:]) if mode.startswith("q") else 0
                    for s in s_values:
                        for seed, pos in pos_lists:
                            text, lp = run_condition(
                                model, tok, clean, rest, target_ids,
                                "drop" if mode == "drop" else "quant", pos, s, n_layers,
                                bits, args.group_size, args.max_new_tokens, stop_str)
                            rows.append(dict(
                                word=word, variant=variant, set=set_name, seed=seed,
                                positions=" ".join(map(str, pos)), mode=mode, s=s,
                                generated=text, exact=int(text == target),
                                logprob=round(lp, 4), baseline_logprob=round(base_lp, 4)))
                    print_grid(rows, word, variant, set_name, mode, s_values)

    if rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")
        if len(words) > 1:
            print_summary(rows, s_values)


def print_grid(rows, word, variant, set_name, mode, s_values):
    sel = [r for r in rows if r["word"] == word and r["variant"] == variant
           and r["set"] == set_name and r["mode"] == mode]
    marks, lps = [], []
    for s in s_values:
        rs = [r for r in sel if r["s"] == s]
        acc = sum(r["exact"] for r in rs) / len(rs)
        marks.append("✓" if acc == 1 else ("·" if acc == 0 else f"{acc:.1f}"))
        lps.append(sum(r["logprob"] for r in rs) / len(rs))
    print(f"  {variant:6s} {set_name:7s} {mode:4s} | "
          + " ".join(f"{m:>3s}" for m in marks)
          + " | logp@s: " + " ".join(f"{s}:{lp:.1f}" for s, lp in zip(s_values, lps)))


def print_summary(rows, s_values):
    print("\nExact-match rate over words, by (variant, set, mode) and s:")
    print(" " * 26 + " ".join(f"{s:>4d}" for s in s_values))
    keys = sorted({(r["variant"], r["set"], r["mode"]) for r in rows})
    for key in keys:
        vals = []
        for s in s_values:
            rs = [r for r in rows if (r["variant"], r["set"], r["mode"]) == key and r["s"] == s]
            vals.append(sum(r["exact"] for r in rs) / max(len(rs), 1))
        print(f"{key[0]:6s} {key[1]:7s} {key[2]:4s}      " + " ".join(f"{v:4.2f}" for v in vals))


if __name__ == "__main__":
    main()
