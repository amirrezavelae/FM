#!/usr/bin/env python
"""
Exp 3: needle retrieval after the needle's prefix pieces are dropped.

A rare multi-token word ("needle") is hidden in ~2,000 tokens of WikiText:
    The secret word is <needle>.
inserted at 10% / 50% / 90% depth. The prompt ends with
    What is the secret word? The secret word is
and the model must produce the needle.

Three changes relative to the written Exp 3 design (see README.md):

 1. Only the NEEDLE is dropped (not every word of the haystack), so Random is
    not trivially safe by never touching the needle. Conditions per layer window:
      needle_prefix  all pieces of the needle except the last   (the hypothesis)
      needle_last    only the last piece      (should hurt: control)
      needle_all     every piece of the needle (should break: has teeth)
      random_hay     as many random haystack positions as needle_prefix drops
      all_words      the original design: prefix pieces of every eligible
                     multi-token haystack word + the needle
      random_all     as many random haystack positions as all_words drops
 2. Layer WINDOWS [a, b) instead of only "from layer s": e.g. 0:24 (drop low
    layers, the cheap one in the opposite test), 24:28, 16:28 (the proposal's
    s=16), 0:28 (every layer).
 3. Needle CLASSES, to test whether common words survive but rare ones do not:
      common  frequent lowercase English words (WikiText val+test count >= N)
      rare    real but rare words (count == 1)
      novel   invented pronounceable strings (never seen by the model)
      <name>  optional extra lists (e.g. Tokens2Words Arabic/Hebrew)
    Each needle also gets prior_nll = how surprising it is to the model with no
    haystack ("The secret word is <needle>"), a model-based rarity measure.

Scoring is one teacher-forced pass per condition: exact = the argmax at every
needle token equals the needle AND the next argmax is a stop token ('.' or
newline). That equals "greedy generation reproduces the needle". The Full run
also does real greedy generation (up to 10 tokens) as a baseline check, and
--generate_all does it for every condition.

Variants:
  deploy  prefill everything except the last prompt token, drop, then answer.
          The question tokens saw the clean KV (the real use case).
  strict  prefill up to and including the needle's last piece, drop, then run
          the rest of the prompt. Nothing after the needle ever sees the clean
          prefix, so the word can only survive through the last piece. Much
          slower (each condition re-runs the rest of the prompt).

Modes: --preview N (CPU only: needles and N prompts), --summarize_only, --resume.
"""

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DynamicCache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "subword_kv"))
sys.path.insert(0, os.path.join(HERE, "..", "opposite_test"))

import exp1_wikitext as E  # noqa: E402  (test-split articles, model loading, hook check)
import opposite_wikitext as O  # noqa: E402  (window check)
from subword_kv import DropHooks, WordIndex, cluster_ratio_ci  # noqa: E402

EXP = "needle"
SENTENCE = "The secret word is {w}.\n"
LEAD = "The secret word is "
QUESTION = "\n\nWhat is the secret word? The secret word is"
ALL_CONDS = ["needle_prefix", "needle_last", "needle_all", "random_hay", "all_words", "random_all"]
RANDOM_CONDS = {"random_hay", "random_all"}
DEFAULT_WINDOWS = ["0:16", "0:24", "16:28", "24:28", "0:28"]
PAIRS = [("needle_prefix", "random_hay"), ("needle_prefix", "full"),
         ("needle_last", "needle_prefix"), ("all_words", "random_all")]


def wtag(a, b):
    return f"{a}:{b}"


# --------------------------------------------------------------------------- #
# Needles
# --------------------------------------------------------------------------- #
def load_split_text(name):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=name)
    return "".join(ds["text"])


def lowercase_word_counts():
    """Counts of all-lowercase words (5-20 letters) in WikiText-103 val + test."""
    text = load_split_text("validation") + load_split_text("test")
    return Counter(re.findall(r"(?<![A-Za-z])[a-z]{5,20}(?![A-Za-z])", text))


def n_pieces(tok, words):
    enc = tok([" " + w for w in words], add_special_tokens=False)["input_ids"]
    return [len(x) for x in enc]


def novel_word(rng):
    cons, vow = "bdfgklmnprstvz", "aeiou"
    parts = []
    for _ in range(rng.choice([3, 4, 5])):
        parts.append(rng.choice(cons) + rng.choice(vow)
                     + (rng.choice("nrlsm") if rng.random() < 0.25 else ""))
    return "".join(parts)


def pick(cands, n, frac3, rng, label):
    """n needles from (word, pieces) candidates, with >= frac3 having 3+ pieces."""
    rng.shuffle(cands)
    ge3 = [c for c in cands if c[1] >= 3]
    two = [c for c in cands if c[1] == 2]
    q3 = min(len(ge3), math.ceil(frac3 * n))
    if q3 < math.ceil(frac3 * n):
        print(f"[warn] class {label}: only {len(ge3)} candidates with 3+ pieces "
              f"(wanted {math.ceil(frac3 * n)})")
    chosen = ge3[:q3]
    rest = ge3[q3:] + two
    rng.shuffle(rest)
    chosen += rest[: n - len(chosen)]
    if len(chosen) < n:
        print(f"[warn] class {label}: only {len(chosen)} needles available (wanted {n})")
    return chosen


def select_needles(tok, args):
    rng = random.Random(args.seed)
    banned = {"secret", "word", "words", "what"}
    needles = []
    counts = lowercase_word_counts()
    vocab = sorted(w for w in counts if w not in banned)

    for cls in args.classes:
        if cls == "common":
            words = [w for w in vocab if counts[w] >= args.common_min_count and len(w) >= 7]
        elif cls == "rare":
            words = [w for w in vocab if counts[w] == 1 and len(w) >= 9]
            rng.shuffle(words)
            words = words[:6000]
        elif cls == "novel":
            seen, words = set(), []
            while len(words) < 3000:
                w = novel_word(rng)
                if w not in counts and w not in seen:
                    seen.add(w)
                    words.append(w)
        else:
            raise SystemExit(f"unknown class {cls!r}")
        pcs = n_pieces(tok, words)
        cands = [(w, p) for w, p in zip(words, pcs) if p >= 2]
        for w, p in pick(cands, args.n_per_class, args.frac_3plus, rng, cls):
            needles.append({"word": w, "cls": cls, "pieces": p})

    for spec in args.extra_list or []:
        name, path = spec.split("=", 1)
        import regex
        with open(path, encoding="utf-8") as f:
            words = [w.strip() for w in f if regex.fullmatch(r"\p{L}+", w.strip())]
        words = sorted(set(words))
        rng.shuffle(words)
        words = words[:6000]
        pcs = n_pieces(tok, words)
        cands = [(w, p) for w, p in zip(words, pcs) if p >= 2]
        for w, p in pick(cands, args.n_per_class, args.frac_3plus, rng, name):
            needles.append({"word": w, "cls": name, "pieces": p})
    return needles


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def filler_pool(articles, n_hay):
    return [a for a in articles if len(a) >= n_hay + 50]


def build_prompt(tok, pool, rng, needle, depth, n_hay):
    """Haystack + needle sentence + question. Returns a dict or None."""
    hay = None
    for _ in range(40):
        art = rng.choice(pool)
        lo = rng.randint(0, len(art) - n_hay - 1)
        c0, c1 = int(art.starts[lo]), int(art.ends[lo + n_hay - 1])
        h = art.text[c0:c1]
        m = re.search(r"\s", h[:200])
        if m:
            h = h[m.end():]
        k = max(h.rfind("\n"), h.rfind(" "))
        if k > 0:
            h = h[:k]
        if re.search(r"(?<![A-Za-z])" + re.escape(needle) + r"(?![A-Za-z])", h, re.I):
            continue  # the needle word already occurs in the filler
        if "secret word" in h.lower():
            continue
        hay = h
        break
    if hay is None:
        return None

    nl = [m.start() for m in re.finditer("\n", hay)]
    target = int(depth * len(hay))
    ins = (min(nl, key=lambda i: abs(i - target)) + 1) if nl else target
    sentence = SENTENCE.format(w=needle)
    text = hay[:ins] + sentence + hay[ins:]
    q_char = len(text.rstrip())
    full = text.rstrip() + QUESTION
    n_char = ins + len(LEAD)
    w_start, w_end = n_char, n_char + len(needle)
    assert full[w_start:w_end] == needle

    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    idx = WordIndex(full, ids, offs, tok)
    t0 = int(np.searchsorted(idx.ends, w_start, side="right"))
    t1 = int(np.searchsorted(idx.starts, w_end, side="left")) - 1
    if t1 <= t0 or not idx._aligned(w_start, w_end, t0, t1):
        return None
    sent_end_char = ins + len(sentence.rstrip("\n"))          # just after the '.'
    s0 = int(np.searchsorted(idx.ends, ins, side="right"))
    s1 = int(np.searchsorted(idx.starts, sent_end_char, side="left")) - 1
    q0 = int(np.searchsorted(idx.ends, q_char, side="right"))
    return {"text": full, "ids": ids, "idx": idx, "t0": t0, "t1": t1, "sent": (s0, s1),
            "q0": q0, "n": len(ids), "depth_char": ins / max(len(text), 1)}


def make_positions(P, needle, args, pid, variant):
    """Position sets for every condition (absolute token positions)."""
    n, t0, t1 = P["n"], P["t0"], P["t1"]
    s0, s1 = P["sent"]
    prefix = list(range(t0, t1))
    sets = {"needle_prefix": prefix, "needle_last": [t1], "needle_all": list(range(t0, t1 + 1))}

    dropped, _ = P["idx"].analyze(0, n, {0, n - 1})
    allw = sorted({p for w in dropped for p in w["dropped_positions"]} | set(prefix))
    sets["all_words"] = allw
    P["n_all_words"] = len(dropped)

    protected = {0, n - 1} | set(range(s0, s1 + 1)) | set(range(P["q0"], n))
    pool_hay = [p for p in range(n) if p not in protected and p not in set(prefix)]
    pool_all = [p for p in pool_hay if p not in set(allw)]
    rand = {}
    for r in range(args.random_seeds):
        rng = random.Random(f"{args.seed}-{pid}-{variant}-hay-{r}")
        rand.setdefault("random_hay", []).append(sorted(rng.sample(pool_hay, min(len(prefix), len(pool_hay)))))
        rng = random.Random(f"{args.seed}-{pid}-{variant}-all-{r}")
        rand.setdefault("random_all", []).append(sorted(rng.sample(pool_all, min(len(allw), len(pool_all)))))
    return sets, rand


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #
def fwd(model, ids, cache, keep):
    """Feed `ids` on `cache`; return logits of the last `keep` positions."""
    x = torch.tensor([ids], device=model.device)
    try:
        out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=keep)
    except TypeError:
        out = model(input_ids=x, past_key_values=cache, use_cache=True)
    return out.logits[0, -keep:].float()


def prefill(model, ids):
    cache = DynamicCache()
    fwd(model, ids, cache, 1)
    return cache


QUOTES = "\"'“”"


def make_token_tests(tok):
    """is_quote(t): the token is a bare quote mark. is_stop(t): once quotes and
    spaces are removed it is empty or '.', i.e. the answer has ended."""
    cache = {}

    def text(t):
        if t not in cache:
            cache[t] = tok.decode([t])
        return cache[t]

    def is_quote(t):
        s = text(t).strip()
        return len(s) == 1 and s in QUOTES

    def is_stop(t):
        return text(t).strip(QUOTES + " \n\t").strip() in ("", ".")

    return is_quote, is_stop


def after_quote_ids(tok, q_id, needle, fallback):
    """Token ids of the needle as it is tokenized right after the quote token q_id.
    In the prompt the needle follows a space (' vert','ebra','e'); after a quote
    there is no space, so the model's first piece is 'vert', not ' vert'. Scoring
    the spaced ids there was the bug that kept Full at 3%."""
    q = tok.decode([q_id])
    ids = tok(q + needle, add_special_tokens=False)["input_ids"]
    if ids and ids[0] == q_id and tok.decode(ids[1:]) == needle:
        return ids[1:]
    ids = tok(needle, add_special_tokens=False)["input_ids"]
    return ids if tok.decode(ids) == needle else fallback


@torch.no_grad()
def eval_condition(model, tok, cache, tail, target, tests, positions, a, b, generate, max_new, needle,
                   fmt="auto"):
    """Teacher-forced exact match + NLL of the needle; optional greedy text.

    The base model usually opens its answer with a quote mark, and then spells the
    needle WITHOUT a leading space (a different tokenization from the prompt's).
    So: if the argmax at the first answer position is a quote, the quote is fed and
    the needle is scored after it in its no-space tokenization (the quote's own
    probability is not counted). `fmt` fixes the format: "auto" decides from this
    run (used for Full); Full's choice (res["fmt"]: a quote id or None) is then
    passed to every other condition, so NLLs are comparable across conditions.
    exact = every needle token is the argmax AND the next argmax ends the answer
    ('.', newline or a closing quote)."""
    is_quote, is_stop = tests
    hooks = DropHooks(model, positions, a, b) if positions else None
    try:
        if fmt == "auto":
            T = len(target)
            lp = F.log_softmax(fwd(model, tail + target, copy.deepcopy(cache), T + 1), -1)
            a0 = int(lp[0].argmax())
            fmt = a0 if (a0 != target[0] and is_quote(a0)) else None
        quoted = fmt is not None
        if quoted:
            target = after_quote_ids(tok, fmt, needle, target)
            T = len(target)
            lp = F.log_softmax(fwd(model, tail + [fmt] + target, copy.deepcopy(cache), T + 1), -1)
        else:
            T = len(target)
            lp = F.log_softmax(fwd(model, tail + target, copy.deepcopy(cache), T + 1), -1)
        tgt = torch.tensor(target, device=lp.device)
        ok = lp[:-1].argmax(-1) == tgt
        nll = -lp[:-1].gather(1, tgt[:, None])[:, 0]
        stop_ok = is_stop(int(lp[-1].argmax()))
        res = {"exact": float(bool(ok.all()) and stop_ok), "nll": float(nll.sum()),
               "first_ok": float(ok[0]), "frac_tok_ok": float(ok.float().mean()),
               "quoted": float(quoted), "fmt": fmt}
        if generate:
            c2 = copy.deepcopy(cache)
            lg = fwd(model, tail, c2, 1)[-1]
            out = []
            for _ in range(max_new):
                nxt = int(lg.argmax())
                out.append(nxt)
                if "\n" in tok.decode(out):
                    break
                lg = fwd(model, [nxt], c2, 1)[-1]
            text = re.split(r"[.\n]", tok.decode(out))[0].strip().strip(QUOTES).strip()
            res["gen"] = text
            res["gen_exact"] = float(text == needle)
        return res
    finally:
        if hooks:
            hooks.remove()


def mean_res(rs):
    out = {k: float(np.mean([r[k] for r in rs]))
           for k in ("exact", "nll", "first_ok", "frac_tok_ok", "quoted")}
    if "gen_exact" in rs[0]:
        out["gen_exact"] = float(np.mean([r["gen_exact"] for r in rs]))
    return out


@torch.no_grad()
def prior_nll(model, tok, needle, target_ids):
    ids = tok(LEAD.rstrip() + " " + needle, add_special_tokens=False)["input_ids"]
    if ids[-len(target_ids):] != target_ids:
        return None
    lp = F.log_softmax(fwd(model, ids, DynamicCache(), len(target_ids) + 1), -1)
    tgt = torch.tensor(target_ids, device=lp.device)
    return float(-lp[:-1].gather(1, tgt[:, None])[:, 0].sum())


@torch.no_grad()
def run_prompt(model, tok, P, nd, depth, variant, windows, conds, args, pid, prior):
    ids, t0, t1, n = P["ids"], P["t0"], P["t1"], P["n"]
    target = ids[t0: t1 + 1]
    tests = make_token_tests(tok)
    split = n - 1 if variant == "deploy" else t1 + 1
    cache = prefill(model, ids[:split])
    tail = ids[split:]
    sets, rand = make_positions(P, nd["word"], args, pid, variant)

    full = eval_condition(model, tok, cache, tail, target, tests, [], 0, 1, True,
                          args.max_new_tokens, nd["word"])
    rec = {"exp": EXP, "id": pid, "variant": variant, "needle": nd["word"], "cls": nd["cls"],
           "depth": depth, "n_pieces": len(target), "prompt_len": n,
           "needle_positions": list(range(t0, t1 + 1)), "k_prefix": t1 - t0,
           "k_all_words": len(sets["all_words"]), "n_all_words": P["n_all_words"],
           "prior_nll": prior, "full": full, "results": {}}
    for a, b in windows:
        wr = {}
        for cond in conds:
            if cond in RANDOM_CONDS:
                rs = [eval_condition(model, tok, cache, tail, target, tests, p, a, b,
                                     args.generate_all, args.max_new_tokens, nd["word"],
                                     fmt=full["fmt"])
                      for p in rand[cond]]
                wr[cond] = mean_res(rs)
            else:
                wr[cond] = eval_condition(model, tok, cache, tail, target, tests, sets[cond],
                                          a, b, args.generate_all, args.max_new_tokens, nd["word"],
                                          fmt=full["fmt"])
        rec["results"][wtag(a, b)] = wr
    return rec


# --------------------------------------------------------------------------- #
# Preview (CPU only)
# --------------------------------------------------------------------------- #
def preview(tok, needles, pool, args, n_prompts):
    print("\nNEEDLES")
    by = defaultdict(list)
    for nd in needles:
        by[nd["cls"]].append(nd)
    for cls, lst in by.items():
        pc = [x["pieces"] for x in lst]
        print(f"  {cls:<8s} n={len(lst):3d}  pieces: mean {np.mean(pc):.1f}, "
              f"3+: {sum(p >= 3 for p in pc)}   e.g. "
              + ", ".join(f"{x['word']}({x['pieces']})" for x in lst[:8]))
    rng = random.Random(args.seed + 1)
    shown = 0
    print("\nPROMPTS")
    for nd in needles:
        if shown >= n_prompts:
            break
        depth = rng.choice(args.depths)
        P = build_prompt(tok, pool, random.Random(f"{args.seed}-{nd['word']}-{depth}"),
                         nd["word"], depth, args.n_hay_tokens)
        if P is None:
            print(f"  {nd['word']}: could not build a prompt (needle merged or filler clash)")
            continue
        sets, rand = make_positions(P, nd["word"], args, f"prev-{nd['word']}", "deploy")
        t0, t1 = P["t0"], P["t1"]
        pieces = [tok.decode([P["ids"][t]]) for t in range(t0, t1 + 1)]
        ctx = tok.decode(P["ids"][max(t0 - 8, 0): t1 + 4])
        print(f"  [{nd['cls']}] {nd['word']} depth={depth} len={P['n']} pieces={pieces} "
              f"positions={t0}-{t1}  k_prefix={len(sets['needle_prefix'])} "
              f"k_all_words={len(sets['all_words'])}")
        print(f"      context: {ctx!r}")
        print(f"      question: {tok.decode(P['ids'][P['q0']:])!r}")
        shown += 1


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def npieces_group(p):
    return "2" if p == 2 else ("3" if p == 3 else "4+")


def summarize(out_dir, args):
    recs = E.read_jsonl(os.path.join(out_dir, "exp3_examples.jsonl"))
    if not recs:
        print("no records")
        return
    _summarize(recs, out_dir, args, "")
    ok = [r for r in recs if r["full"]["exact"] == 1.0]
    if 0 < len(ok) < len(recs):
        print("\n" + "#" * 100)
        print(f"RESTRICTED TO PROMPTS WHERE FULL SUCCEEDED ({len(ok)} of {len(recs)}): "
              "the usual way to measure what dropping the KV costs, since a prompt "
              "the clean model already misses cannot show retention. Files end in _fullok.")
        _summarize(ok, out_dir, args, "_fullok")


def _summarize(recs, out_dir, args, suffix):
    priors = sorted({r["needle"]: r["prior_nll"] for r in recs if r.get("prior_nll") is not None}.values())
    q1, q2 = (np.quantile(priors, [1 / 3, 2 / 3]) if priors else (0, 0))

    def groups(r):
        g = [("all", "all"), ("class", r["cls"]), ("depth", str(r["depth"])),
             ("npieces", npieces_group(r["n_pieces"]))]
        if r.get("prior_nll") is not None:
            g.append(("prior", "low" if r["prior_nll"] <= q1 else ("mid" if r["prior_nll"] <= q2 else "high")))
        return g

    # per (variant, window, cond, group): lists of exact / dnll / needle
    acc = defaultdict(lambda: ([], [], []))
    pair_acc = defaultdict(lambda: ([], []))
    for r in recs:
        gs = groups(r)
        full_exact, full_nll = r["full"]["exact"], r["full"]["nll"]
        for gt, g in gs:
            e, d, nd = acc[(r["variant"], "-", "full", gt, g)]
            e.append(full_exact)
            d.append(0.0)
            nd.append(r["needle"])
        for w, wr in r["results"].items():
            for cond, v in wr.items():
                for gt, g in gs:
                    e, d, nd = acc[(r["variant"], w, cond, gt, g)]
                    e.append(v["exact"])
                    d.append(v["nll"] - full_nll)
                    nd.append(r["needle"])
            for ca, cb in PAIRS:
                if ca not in wr or (cb != "full" and cb not in wr):
                    continue
                va = wr[ca]["exact"]
                vb = full_exact if cb == "full" else wr[cb]["exact"]
                for gt, g in gs:
                    diffs, nds = pair_acc[(r["variant"], w, f"{ca}_minus_{cb}", gt, g)]
                    diffs.append(va - vb)
                    nds.append(r["needle"])

    rows = []
    for (var, w, cond, gt, g), (e, d, nd) in sorted(acc.items()):
        est, lo, hi = cluster_ratio_ci(e, np.ones(len(e)), nd, args.bootstrap)
        dm, dlo, dhi = cluster_ratio_ci(d, np.ones(len(d)), nd, args.bootstrap)
        rows.append({"variant": var, "window": w, "condition": cond, "group_type": gt, "group": g,
                     "n": len(e), "exact": est, "ci_lo": lo, "ci_hi": hi,
                     "d_nll": dm, "d_nll_lo": dlo, "d_nll_hi": dhi})
    E.write_csv(os.path.join(out_dir, f"exp3_summary{suffix}.csv"), rows)

    prow = []
    for (var, w, cmp_, gt, g), (diffs, nds) in sorted(pair_acc.items()):
        est, lo, hi = cluster_ratio_ci(diffs, np.ones(len(diffs)), nds, args.bootstrap)
        prow.append({"variant": var, "window": w, "comparison": cmp_, "group_type": gt, "group": g,
                     "n": len(diffs), "mean_diff": est, "ci_lo": lo, "ci_hi": hi})
    E.write_csv(os.path.join(out_dir, f"exp3_paired{suffix}.csv"), prow)

    nrow = {}
    for r in recs:
        nrow[r["needle"]] = {"needle": r["needle"], "class": r["cls"], "n_pieces": r["n_pieces"],
                             "prior_nll": r.get("prior_nll")}
    E.write_csv(os.path.join(out_dir, f"exp3_needles{suffix}.csv"), list(nrow.values()))

    # ---------------- console ------------------------------------------------ #
    S = {(r["variant"], r["window"], r["condition"], r["group_type"], r["group"]): r for r in rows}
    PR = {(r["variant"], r["window"], r["comparison"], r["group_type"], r["group"]): r for r in prow}
    wins = sorted({r["window"] for r in rows if r["window"] != "-"},
                  key=lambda s: tuple(int(x) for x in s.split(":")))
    for var in sorted({r["variant"] for r in rows}):
        fr = S[(var, "-", "full", "all", "all")]
        print("\n" + "=" * 100)
        print(f"variant: {var}   prompts: {fr['n']}   needles: {len({r['needle'] for r in recs if r['variant'] == var})}")
        print(f"[check 3] Full exact match (teacher-forced): {fr['exact']:.3f}  "
              f"-> should be near 1.0; if not, fix the prompt before trusting anything")
        gen = [r["full"].get("gen_exact") for r in recs if r["variant"] == var and "gen_exact" in r["full"]]
        if gen:
            print(f"          Full exact match (real greedy generation):  {np.mean(gen):.3f}")
        qd = [r["full"]["quoted"] for r in recs if r["variant"] == var and "quoted" in r["full"]]
        if qd:
            print(f"          Full answers that open with a quote mark:   {np.mean(qd):.3f} "
                  "(handled: the needle is scored after the quote)")

        def cell(cond, gt, g, w):
            r = S.get((var, w, cond, gt, g))
            return f"{r['exact']:.2f}" if r else "  - "

        print(f"\nExact match by condition (all prompts); Full = {fr['exact']:.2f}")
        print(f"  {'condition':<15s}" + "".join(f"{w:>9s}" for w in wins))
        for cond in ALL_CONDS:
            if (var, wins[0], cond, "all", "all") in S:
                print(f"  {cond:<15s}" + "".join(f"{cell(cond, 'all', 'all', w):>9s}" for w in wins))
        print("\nneedle_prefix - random_hay (exact match, article/needle-clustered 95% CI):")
        for w in wins:
            p = PR.get((var, w, "needle_prefix_minus_random_hay", "all", "all"))
            if p:
                print(f"  {w:>6s}: {p['mean_diff']:+.3f} [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]")

        for gt, title in (("class", "needle class"), ("npieces", "pieces in the needle"),
                          ("depth", "depth"), ("prior", "prior_nll tercile (low = easy for the model)")):
            gl = sorted({r["group"] for r in rows if r["variant"] == var and r["group_type"] == gt})
            if not gl:
                continue
            print(f"\nneedle_prefix exact match by {title}   (Full / needle_prefix per window; random_hay in brackets)")
            print(f"  {'group':<10s}{'n':>5s}{'Full':>7s}" + "".join(f"{w:>16s}" for w in wins))
            for g in gl:
                n = S[(var, "-", "full", gt, g)]["n"]
                line = f"  {g:<10s}{n:>5d}{S[(var, '-', 'full', gt, g)]['exact']:>7.2f}"
                for w in wins:
                    a, b = cell("needle_prefix", gt, g, w), cell("random_hay", gt, g, w)
                    line += f"{a + ' [' + b + ']':>16s}"
                print(line)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--classes", nargs="+", default=["common", "rare", "novel"])
    ap.add_argument("--extra_list", nargs="*", default=None, metavar="NAME=PATH",
                    help="extra needle classes from word-list files (one word per line)")
    ap.add_argument("--n_per_class", type=int, default=40)
    ap.add_argument("--frac_3plus", type=float, default=0.3,
                    help="at least this share of each class has 3+ pieces (if available)")
    ap.add_argument("--common_min_count", type=int, default=10)
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--n_hay_tokens", type=int, default=2000)
    ap.add_argument("--windows", nargs="+", default=DEFAULT_WINDOWS, metavar="A:B")
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDS, choices=ALL_CONDS)
    ap.add_argument("--variants", nargs="+", default=["deploy"], choices=["deploy", "strict"])
    ap.add_argument("--random_seeds", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=10)
    ap.add_argument("--generate_all", action="store_true",
                    help="also run real greedy generation for every condition (slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--hook_tol", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_dir", default="results/needle")
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="CPU only: print the needles and N prompts, then exit")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--no_gate", action="store_true",
                    help="do not stop early when Full exact match is low on the first 12 prompts")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl = os.path.join(args.out_dir, "exp3_examples.jsonl")
    if args.summarize_only:
        summarize(args.out_dir, args)
        return
    windows = O.parse_windows(args.windows)

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise SystemExit("a fast tokenizer is needed for offset mappings")
    needles = select_needles(tok, args)
    articles = E.load_articles(tok)
    pool = filler_pool(articles, args.n_hay_tokens)
    print(f"{len(needles)} needles, {len(pool)} filler articles (of {len(articles)}), "
          f"depths {args.depths}, windows {[wtag(a, b) for a, b in windows]}, variants {args.variants}")

    if args.preview:
        preview(tok, needles, pool, args, args.preview)
        return

    done = set()
    if os.path.exists(jsonl):
        if args.resume:
            done = {r["id"] for r in E.read_jsonl(jsonl)}
            print(f"resuming: {len(done)} prompts already done")
        elif args.overwrite:
            os.remove(jsonl)
        else:
            raise SystemExit(f"{jsonl} exists: use --resume or --overwrite")

    model = E.load_model(args)
    n_layers = model.config.num_hidden_layers
    if max(b for _, b in windows) > n_layers:
        raise SystemExit(f"windows must end at or below {n_layers}")

    # Check 1: hook check + window check on the first buildable prompt.
    P0 = None
    for nd in needles:
        P0 = build_prompt(tok, pool, random.Random("check"), nd["word"], 0.5, args.n_hay_tokens)
        if P0:
            break
    hc = E.hook_check(model, P0["ids"][:512], P0["ids"][512:528], args.hook_tol)
    wc = O.window_check(model, P0["ids"][:512], P0["ids"][512:528], args.hook_tol, n_layers)
    print(f"[check 1] hook check: {hc}\n[check 1b] window check: {wc}")
    if not (hc["ok"] and wc["ok"]):
        raise SystemExit("hook/window check failed: do not trust any number from this setup")

    import transformers
    with open(os.path.join(args.out_dir, "exp3_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "n_layers": n_layers, "hook_check": hc, "window_check": wc,
                   "needles": needles, "torch": torch.__version__,
                   "transformers": transformers.__version__}, f, indent=2)

    t0 = time.time()
    n_run = n_skip = 0
    total = len(needles) * len(args.depths) * len(args.variants)
    prior_cache = {}
    gate = []
    with open(jsonl, "a", encoding="utf-8") as out:
        for nd in needles:
            for depth in args.depths:
                P = build_prompt(tok, pool, random.Random(f"{args.seed}-{nd['word']}-{depth}"),
                                 nd["word"], depth, args.n_hay_tokens)
                if P is None:
                    n_skip += 1
                    print(f"[skip] {nd['word']} depth {depth}: could not build prompt")
                    continue
                if nd["word"] not in prior_cache:
                    prior_cache[nd["word"]] = prior_nll(model, tok, nd["word"], P["ids"][P["t0"]: P["t1"] + 1])
                for variant in args.variants:
                    pid = f"{variant}|{nd['cls']}|{nd['word']}|d{depth}"
                    if pid in done:
                        continue
                    rec = run_prompt(model, tok, P, nd, depth, variant, windows, args.conditions,
                                     args, pid, prior_cache[nd["word"]])
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n_run += 1
                    gate.append(rec["full"]["exact"])
                    if len(gate) == 12 and np.mean(gate) < 0.5 and not args.no_gate:
                        raise SystemExit(f"[check 3] Full exact match on the first 12 prompts is "
                                         f"{np.mean(gate):.2f}: the scoring or prompt is still wrong. "
                                         f"Stopped early (use --no_gate to override).")
                    if n_run % 10 == 0:
                        el = time.time() - t0
                        print(f"  {n_run}/{total} prompts  {el / n_run:.1f}s/prompt  "
                              f"last: {nd['word']} ({nd['cls']}) full_exact={rec['full']['exact']:.0f}",
                              flush=True)
    print(f"{n_run} prompts run, {n_skip} skipped")
    summarize(args.out_dir, args)


if __name__ == "__main__":
    main()
