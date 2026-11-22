#!/usr/bin/env python
"""
Exp 1: next-token distribution on WikiText-103 (test).

For each chunk (512 context tokens + 64 continuation tokens + 1 target token):
  1. prefill the 512-token context with the clean model;
  2. apply a condition to the context's KV cache (drop = mask out of attention
     at layers >= s);
  3. feed the 64 real continuation tokens and compare the next-token
     distribution at each of them with the Full (nothing dropped) run.

Conditions per chunk (9 continuation runs when words are found):
  full                              nothing dropped
  ours_s{s}   for s in --s_values   all non-final pieces of every eligible
                                    multi-token word in the context
  random_s{s} for s in --s_values   same number k of random positions, never
                                    protected ones, never Ours' positions;
                                    --random_seeds seeds, averaged

Protected positions (never dropped anywhere): context position 0 (attention
sink) and the last context token. No special or template tokens occur in this
experiment.

Metrics per chunk, averaged over the 64 positions: KL(Full || cond), top-1
agreement with Full, change in log-loss of the real next token vs Full.

Outputs in --out_dir:
  exp1_examples.jsonl       one JSON line per chunk, incl. per-position KL / dNLL
                            and the role of each target token ("per_position")
  exp1_summary.csv          one row per condition (mean + 95% bootstrap CI)
  exp1_paired.csv           paired differences (Ours-Random, Ours-Full); CIs over
                            chunks and over articles
  exp1_slices.csv           metrics split by target-token role (rep_first, rep_rest,
                            new_first, new_rest, other); article-clustered CIs
  exp1_dropped_words.csv    every dropped word, sorted by count
  exp1_excluded_words.csv   every excluded word with its reason, sorted by count
  exp1_meta.json            args, versions, hook-check result

Modes:
  --words_only N     CPU only: print dropped/excluded words for N chunks, exit
  --inspect ID ...   CPU only: raw text + the tokens where Ours hurt most
  --inspect_worst N  same, for the N chunks with the largest Ours-Random KL gap
  --summarize_only   rebuild the summary files from an existing jsonl, exit
  --resume           continue a run, skipping chunks already in the jsonl
"""

import argparse
import copy
import csv
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from subword_kv import DropHooks, WordIndex, bootstrap_ci, cluster_ratio_ci

EXP = "wikitext"
TITLE_RE = re.compile(r"^= [^=].* =$")  # level-1 article heading, after strip()
METRICS = ("kl", "top1", "dloss", "loss")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_articles(tok):
    """WikiText-103 raw test split, split into articles, tokenized with offsets."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    texts, cur = [], []
    for row in ds["text"]:
        if TITLE_RE.match(row.strip()) and "".join(cur).strip():
            texts.append("".join(cur))
            cur = []
        cur.append(row)
    if "".join(cur).strip():
        texts.append("".join(cur))

    articles = []
    for t in texts:
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        articles.append(WordIndex(t, enc["input_ids"], enc["offset_mapping"], tok))
    return articles


def sample_chunks(articles, args):
    """Non-overlapping windows of ctx+cont+1 tokens inside articles, seeded."""
    win = args.ctx_len + args.cont_len + 1
    rng = random.Random(args.seed)
    cands = []
    for ai, art in enumerate(articles):
        m = len(art) // win
        if m == 0:
            continue
        off = rng.randint(0, len(art) - m * win)
        cands += [(ai, off + i * win) for i in range(m)]
    rng.shuffle(cands)
    if len(cands) < args.n_chunks:
        print(f"[warn] only {len(cands)} windows available, wanted {args.n_chunks}")
    chosen = sorted(cands[: args.n_chunks])
    return [{"id": f"wt103-test-a{ai}-t{lo}", "article": ai, "lo": lo}
            for ai, lo in chosen]


def analyze_chunk(articles, chunk, args):
    art = articles[chunk["article"]]
    protected = {0, args.ctx_len - 1}
    dropped, excluded = art.analyze(chunk["lo"], chunk["lo"] + args.ctx_len, protected)
    ours = sorted({p for w in dropped for p in w["dropped_positions"]})
    return art, dropped, excluded, ours


# --------------------------------------------------------------------------- #
# Word log check (CPU only)
# --------------------------------------------------------------------------- #
def print_word_log(articles, chunks, args, n):
    for chunk in chunks[:n]:
        art, dropped, excluded, ours = analyze_chunk(articles, chunk, args)
        lo = chunk["lo"]
        c0 = int(art.starts[lo])
        c1 = int(art.ends[lo + args.ctx_len - 1])
        print("=" * 100)
        print(f"{chunk['id']}  k={len(ours)}  drop_frac={len(ours) / args.ctx_len:.3f}")
        print("context starts: " + repr(art.text[c0:c0 + 160]))
        print(f"\n  DROPPED WORDS ({len(dropped)}):  word | pieces | dropped positions")
        for w in dropped[:40]:
            print(f"    {w['word']:<20s} {str(w['pieces']):<42s} {w['dropped_positions']}")
        if len(dropped) > 40:
            print(f"    ... {len(dropped) - 40} more")
        by_reason = Counter(e["reason"] for e in excluded)
        print(f"\n  EXCLUDED WORDS: {dict(by_reason)}")
        for reason in by_reason:
            if reason == "single_token":
                continue
            ex = [e["word"] for e in excluded if e["reason"] == reason][:15]
            print(f"    {reason}: {ex}")
        print(f"\n  (context text span: chars {c0}-{c1})")


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #
def load_model(args):
    kw = dict(attn_implementation=args.attn,
              device_map="auto" if torch.cuda.is_available() else None)
    dt = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, **kw)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, **kw)
    model.eval()
    return model


def prefill(model, ctx_ids):
    cache = DynamicCache()
    x = torch.tensor([ctx_ids], device=model.device)
    model(input_ids=x, past_key_values=cache, use_cache=True)
    return cache


def continuation_logits(model, cache, cont_ids, positions, start_layer):
    """Feed the continuation on a copy of the cache, optionally with a drop."""
    c = copy.deepcopy(cache)
    x = torch.tensor([cont_ids], device=model.device)
    hooks = DropHooks(model, positions, start_layer) if positions else None
    try:
        return model(input_ids=x, past_key_values=c, use_cache=True).logits[0].float()
    finally:
        if hooks:
            hooks.remove()


@torch.no_grad()
def hook_check(model, ctx_ids, cont_ids, tol):
    """Check 1: the hook with an empty drop list must reproduce the default
    logits; a large drop must visibly change them (the hook is really active)."""
    cache = prefill(model, ctx_ids)
    x = torch.tensor([cont_ids], device=model.device)
    plain = model(input_ids=x, past_key_values=copy.deepcopy(cache), use_cache=True
                  ).logits[0].float()
    h = DropHooks(model, [], 0)
    try:
        hooked = model(input_ids=x, past_key_values=copy.deepcopy(cache), use_cache=True
                       ).logits[0].float()
    finally:
        h.remove()
    big = continuation_logits(model, cache, cont_ids, list(range(1, len(ctx_ids) - 1)), 0)
    res = {
        "empty_drop_vs_default_maxdiff": (plain - hooked).abs().max().item(),
        "empty_drop_argmax_equal": bool((plain.argmax(-1) == hooked.argmax(-1)).all()),
        "big_drop_vs_default_maxdiff": (plain - big).abs().max().item(),
    }
    res["ok"] = (res["empty_drop_vs_default_maxdiff"] <= tol
                 and res["empty_drop_argmax_equal"]
                 and res["big_drop_vs_default_maxdiff"] > 10 * tol)
    return res


def compare(lp_full, lp_cond, targets):
    """Per-chunk metrics, averaged over continuation positions."""
    kl = (lp_full.exp() * (lp_full - lp_cond)).sum(-1)
    top1 = (lp_full.argmax(-1) == lp_cond.argmax(-1)).float()
    nll_c = -lp_cond.gather(1, targets[:, None])[:, 0]
    nll_f = -lp_full.gather(1, targets[:, None])[:, 0]
    metrics = {"kl": kl.mean().item(), "top1": top1.mean().item(),
               "dloss": (nll_c - nll_f).mean().item(), "loss": nll_c.mean().item()}
    # per continuation position (for the slice analysis)
    pos = {"kl": kl.tolist(), "dnll": (nll_c - nll_f).tolist(), "top1": top1.tolist()}
    return metrics, pos


def mean_metrics(ms):
    return {k: float(np.mean([m[k] for m in ms])) for k in METRICS}


def mean_pos(ps):
    return {k: np.mean([p[k] for p in ps], axis=0).tolist() for k in ps[0]}


def round_pos(p, nd=5):
    return {k: [round(v, nd) for v in vals] for k, vals in p.items()}


@torch.no_grad()
def run_chunk(model, art, chunk, dropped, excluded, ours, args):
    lo, C, L = chunk["lo"], args.ctx_len, args.cont_len
    ctx_ids = art.ids[lo: lo + C]
    cont_ids = art.ids[lo + C: lo + C + L]
    tgt = torch.tensor(art.ids[lo + C + 1: lo + C + L + 1], device=model.device)
    k = len(ours)

    cache = prefill(model, ctx_ids)
    lp_full = F.log_softmax(continuation_logits(model, cache, cont_ids, [], 0), -1)
    full = {"kl": 0.0, "top1": 1.0, "dloss": 0.0,
            "loss": (-lp_full.gather(1, tgt[:, None])[:, 0]).mean().item()}

    rec = {"exp": EXP, "id": chunk["id"], "prompt_len": C, "cont_len": L,
           "eligible": k > 0, "k": k, "drop_frac": k / C,
           "dropped_words": dropped, "excluded_words": excluded,
           "random_positions": {}, "metrics": {"full": full}, "metrics_by_seed": {}}
    if k == 0:  # no eligible words: stays in Full, out of Ours-vs-Random
        return rec

    pool = [p for p in range(C) if p not in {0, C - 1} and p not in set(ours)]
    k_rand = min(k, len(pool))
    rand_sets = {}
    for r in range(args.random_seeds):
        rng = random.Random(f"{args.seed}-{chunk['id']}-{r}")
        rand_sets[r] = sorted(rng.sample(pool, k_rand))
    rec["random_positions"] = {str(r): p for r, p in rand_sets.items()}

    # Per-position record. role = how the target token relates to multi-token
    # words; "rep_*" = a word Ours dropped in the context that re-appears here.
    known = {w["word"] for w in dropped}
    t_lo = lo + C + 1
    pp = {"target_ids": art.ids[t_lo: t_lo + L],
          "roles": art.token_roles(t_lo, t_lo + L, known),
          "full_nll": [round(v, 5) for v in (-lp_full.gather(1, tgt[:, None])[:, 0]).tolist()]}

    for s in args.s_values:
        lp = F.log_softmax(continuation_logits(model, cache, cont_ids, ours, s), -1)
        m, p = compare(lp_full, lp, tgt)
        rec["metrics"][f"ours_s{s}"] = m
        pp[f"ours_s{s}"] = round_pos(p)
        ms, ps = [], []
        for r in range(args.random_seeds):
            lp = F.log_softmax(continuation_logits(model, cache, cont_ids, rand_sets[r], s), -1)
            m, p = compare(lp_full, lp, tgt)
            ms.append(m)
            ps.append(p)
        rec["metrics"][f"random_s{s}"] = mean_metrics(ms)
        rec["metrics_by_seed"][f"random_s{s}"] = ms
        pp[f"random_s{s}"] = round_pos(mean_pos(ps))  # mean over seeds
    rec["per_position"] = pp
    return rec


# --------------------------------------------------------------------------- #
# Summaries (built from the jsonl, so they work after a resume too)
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def article_of(rec):
    """Article index from a chunk id like wt103-test-a18-t963."""
    return int(re.search(r"-a(\d+)-t", rec["id"]).group(1))


ROLES = ["rep_first", "rep_rest", "new_first", "new_rest", "other"]


def slice_analysis(recs, args, out_dir):
    """Per-position metrics split by the role of the target token.

    rep_*  = pieces of a multi-token word that Ours dropped in the context and
             that re-appears in the continuation (copying a word whose prefix
             KV was removed). new_* = multi-token words not seen as drops.
    CIs bootstrap over articles (chunks of one article are not independent).
    """
    pp = [r for r in recs if r["eligible"] and "per_position" in r]
    if not pp:
        print("\n[slices] no per-position data in this jsonl (run made before it was added)")
        return
    groups = [article_of(r) for r in pp]
    rows = []
    for s in args.s_values:
        for role in ROLES + ["all"]:
            masks = [np.array([role == "all" or x == role for x in r["per_position"]["roles"]])
                     for r in pp]
            counts = np.array([m.sum() for m in masks], dtype=float)
            if counts.sum() == 0:
                continue
            for metric in ("kl", "dnll", "top1"):
                tot = {c: np.array([np.array(r["per_position"][f"{c}_s{s}"][metric])[m].sum()
                                    for r, m in zip(pp, masks)]) for c in ("ours", "random")}
                for label, v in (("ours", tot["ours"]), ("random", tot["random"]),
                                 ("ours_minus_random", tot["ours"] - tot["random"])):
                    est, lo, hi = cluster_ratio_ci(v, counts, groups, args.bootstrap)
                    rows.append({"s": s, "role": role, "metric": metric, "quantity": label,
                                 "n_positions": int(counts.sum()),
                                 "n_chunks": int((counts > 0).sum()),
                                 "mean": est, "ci_lo": lo, "ci_hi": hi})
    write_csv(os.path.join(out_dir, "exp1_slices.csv"), rows)

    print("\nSlices by target-token role (article-clustered bootstrap CIs)")
    print("  rep_* = pieces of a word Ours dropped that re-appears in the continuation")
    for s in args.s_values:
        print(f"\n  s={s}   {'role':<10s}{'n_pos':>7s}{'KL ours':>10s}{'KL rand':>10s}"
              f"{'ours-rand KL [95% CI]':>34s}{'dNLL ours-rand':>18s}")
        for role in ROLES + ["all"]:
            g = {(r["metric"], r["quantity"]): r for r in rows if r["s"] == s and r["role"] == role}
            if not g:
                continue
            kd, nd = g[("kl", "ours_minus_random")], g[("dnll", "ours_minus_random")]
            print(f"       {role:<10s}{kd['n_positions']:>7d}"
                  f"{g[('kl', 'ours')]['mean']:>10.4f}{g[('kl', 'random')]['mean']:>10.4f}"
                  f"{kd['mean']:>+14.4f} [{kd['ci_lo']:+.4f},{kd['ci_hi']:+.4f}]"
                  f"{nd['mean']:>+10.4f}")


def inspect_chunks(articles, tok, args, recs, ids):
    """CPU only: show one chunk's dropped words and where the loss rose."""
    by_id = {r["id"]: r for r in recs}
    s, C, L = args.inspect_s, args.ctx_len, args.cont_len
    for cid in ids:
        rec = by_id.get(cid)
        if rec is None:
            print(f"[inspect] {cid} not in the jsonl")
            continue
        ai, lo = map(int, re.search(r"-a(\d+)-t(\d+)", cid).groups())
        art = articles[ai]
        m = rec["metrics"]
        print("=" * 100)
        print(f"{cid}  k={rec['k']}  s={s}")
        if f"ours_s{s}" in m:
            print(f"  KL   ours={m[f'ours_s{s}']['kl']:.3f}  random={m[f'random_s{s}']['kl']:.3f}   "
                  f"top1 ours={m[f'ours_s{s}']['top1']:.2f}  random={m[f'random_s{s}']['top1']:.2f}   "
                  f"dloss ours={m[f'ours_s{s}']['dloss']:+.3f}  random={m[f'random_s{s}']['dloss']:+.3f}")
        cnt = Counter((w["word"], "|".join(w["pieces"])) for w in rec["dropped_words"])
        print("  dropped words (most frequent): "
              + "; ".join(f"{w} [{p}] x{n}" for (w, p), n in cnt.most_common(10)))
        c_end = int(art.ends[lo + C - 1])
        print("\n  context ends : ..." + repr(art.text[int(art.starts[max(lo + C - 40, lo)]): c_end]))
        t_lo = lo + C + 1
        print("  continuation : " + repr(art.text[int(art.starts[t_lo]): int(art.ends[t_lo + L - 1])]))
        pp = rec.get("per_position")
        if not pp or f"ours_s{s}" not in pp:
            print("\n  (no per-position data for this run; rerun with the current script)")
            continue
        d_o, d_r = pp[f"ours_s{s}"]["dnll"], pp[f"random_s{s}"]["dnll"]
        kl_o = pp[f"ours_s{s}"]["kl"]
        order = sorted(range(L), key=lambda j: -d_o[j])[:15]
        print(f"\n  15 target tokens where Ours raised the loss most "
              f"(dNLL = loss change vs Full, in nats):")
        print(f"  {'j':>3s} {'role':<10s}{'full NLL':>9s}{'dNLL ours':>11s}{'dNLL rand':>11s}"
              f"{'KL ours':>9s}  context -> [target]")
        for j in order:
            t = t_lo + j
            left = tok.decode(art.ids[max(t - 6, lo + C): t]) if t - 1 >= lo + C else ""
            print(f"  {j:>3d} {pp['roles'][j]:<10s}{pp['full_nll'][j]:>9.2f}{d_o[j]:>+11.2f}"
                  f"{d_r[j]:>+11.2f}{kl_o[j]:>9.3f}  {left!r} -> [{tok.decode([art.ids[t]])!r}]")
        by_role = {}
        for j in range(L):
            by_role.setdefault(pp["roles"][j], []).append(j)
        print("\n  mean dNLL by role (ours / random):  " + "   ".join(
            f"{r}: {np.mean([d_o[j] for j in js]):+.3f}/{np.mean([d_r[j] for j in js]):+.3f} (n={len(js)})"
            for r, js in sorted(by_role.items())))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(out_dir, args):
    recs = read_jsonl(os.path.join(out_dir, "exp1_examples.jsonl"))
    meta_path = os.path.join(out_dir, "exp1_meta.json")
    n_layers = 28
    if os.path.exists(meta_path):
        n_layers = json.load(open(meta_path)).get("n_layers", n_layers)

    elig = [r for r in recs if r["eligible"]]
    n_total, n_none = len(recs), len(recs) - len(elig)
    print(f"\n{n_total} chunks, {n_none} without eligible words (kept in Full only)")

    conds = [("full", recs)]
    for s in args.s_values:
        conds += [(f"ours_s{s}", elig), (f"random_s{s}", elig)]

    rows = []
    for name, rs in conds:
        s = int(name.split("_s")[1]) if "_s" in name else None
        row = {"condition": name, "n": len(rs), "n_chunks_total": n_total,
               "n_chunks_no_words": n_none,
               "mean_k": float(np.mean([r["k"] for r in rs])) if rs else float("nan"),
               "mean_drop_frac": float(np.mean([r["drop_frac"] for r in rs])) if rs else float("nan")}
        if name == "full":
            row["mean_k"], row["mean_drop_frac"] = 0.0, 0.0
        frac = row["mean_drop_frac"]
        row["mem_saving"] = frac * (n_layers - s) / n_layers if s is not None else 0.0
        for m in METRICS:
            mean, lo, hi = bootstrap_ci([r["metrics"][name][m] for r in rs], args.bootstrap)
            row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        rows.append(row)
    write_csv(os.path.join(out_dir, "exp1_summary.csv"), rows)

    paired = []
    for s in args.s_values:
        for m in ("kl", "top1", "dloss"):
            ours = np.array([r["metrics"][f"ours_s{s}"][m] for r in elig])
            rand = np.array([r["metrics"][f"random_s{s}"][m] for r in elig])
            full = np.array([r["metrics"]["full"][m] for r in elig])
            groups = [article_of(r) for r in elig]
            for label, diff in (("ours_minus_random", ours - rand),
                                ("ours_minus_full", ours - full)):
                mean, lo, hi = bootstrap_ci(diff, args.bootstrap)
                _, alo, ahi = cluster_ratio_ci(diff, np.ones(len(diff)), groups, args.bootstrap)
                paired.append({"s": s, "comparison": label, "metric": m, "n": len(diff),
                               "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                               "ci_lo_article": alo, "ci_hi_article": ahi})
    write_csv(os.path.join(out_dir, "exp1_paired.csv"), paired)

    dropped = Counter()
    for r in recs:
        for w in r["dropped_words"]:
            dropped[(w["word"], json.dumps(w["pieces"], ensure_ascii=False))] += 1
    write_csv(os.path.join(out_dir, "exp1_dropped_words.csv"),
              [{"word": w, "pieces": p, "n_pieces": len(json.loads(p)), "count": c}
               for (w, p), c in sorted(dropped.items(), key=lambda kv: (-kv[1], kv[0]))])

    excluded = Counter()
    for r in recs:
        for e in r["excluded_words"]:
            excluded[(e["word"], e["reason"])] += 1
    write_csv(os.path.join(out_dir, "exp1_excluded_words.csv"),
              [{"word": w, "reason": why, "count": c}
               for (w, why), c in sorted(excluded.items(), key=lambda kv: (-kv[1], kv[0]))])

    # --- console report -------------------------------------------------- #
    full_loss = rows[0]["loss_mean"]
    print(f"[check 3] Full log-loss {full_loss:.3f} nats/token (ppl {np.exp(full_loss):.1f}); "
          "sanity-check this against a 7B base model on WikiText.")
    print(f"\n{'condition':<12s}{'n':>5s}{'k':>7s}{'drop%':>7s}{'KL':>9s}{'top1':>8s}{'dloss':>9s}")
    for r in rows:
        print(f"{r['condition']:<12s}{r['n']:>5d}{r['mean_k']:>7.1f}{100 * r['mean_drop_frac']:>7.1f}"
              f"{r['kl_mean']:>9.4f}{r['top1_mean']:>8.3f}{r['dloss_mean']:>9.4f}")
    print("\nPaired differences (mean [95% CI]); lower KL/dloss and higher top1 is better:")
    for p in paired:
        if p["comparison"] == "ours_minus_random":
            print(f"  s={p['s']:<3d}{p['metric']:<6s}ours-random = {p['mean_diff']:+.4f} "
                  f"[{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}] chunks  "
                  f"[{p['ci_lo_article']:+.4f}, {p['ci_hi_article']:+.4f}] articles  n={p['n']}")

    slice_analysis(recs, args, out_dir)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--n_chunks", type=int, default=200)
    ap.add_argument("--ctx_len", type=int, default=512)
    ap.add_argument("--cont_len", type=int, default=64)
    ap.add_argument("--s_values", type=int, nargs="+", default=[0, 16])
    ap.add_argument("--random_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="chunk sampling + random positions")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default="eager", help="eager or sdpa (not flash)")
    ap.add_argument("--hook_tol", type=float, default=1e-3,
                    help="max allowed logit difference in the hook check")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_dir", default="results/exp1")
    ap.add_argument("--words_only", type=int, default=0, metavar="N",
                    help="CPU only: print the word log for N chunks and exit")
    ap.add_argument("--inspect", nargs="*", default=None, metavar="CHUNK_ID",
                    help="CPU only: show these chunks (ids from the jsonl) and exit")
    ap.add_argument("--inspect_worst", type=int, default=0, metavar="N",
                    help="CPU only: show the N chunks where Ours-Random KL is largest")
    ap.add_argument("--inspect_s", type=int, default=0, help="s value used by --inspect")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl = os.path.join(args.out_dir, "exp1_examples.jsonl")
    if args.summarize_only:
        summarize(args.out_dir, args)
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise SystemExit("a fast tokenizer is needed for offset mappings")
    articles = load_articles(tok)
    chunks = sample_chunks(articles, args)
    print(f"{len(articles)} test articles, {len(chunks)} chunks "
          f"({args.ctx_len} ctx + {args.cont_len} cont + 1 target tokens each)")

    if args.words_only:
        print_word_log(articles, chunks, args, args.words_only)
        return

    if args.inspect is not None or args.inspect_worst:
        recs = read_jsonl(jsonl)
        ids = list(args.inspect or [])
        if args.inspect_worst:
            s = args.inspect_s
            el = [r for r in recs if r["eligible"]]
            el.sort(key=lambda r: r["metrics"][f"random_s{s}"]["kl"] - r["metrics"][f"ours_s{s}"]["kl"])
            ids += [r["id"] for r in el[: args.inspect_worst]]
        inspect_chunks(articles, tok, args, recs, ids)
        return

    done = set()
    if os.path.exists(jsonl):
        if args.resume:
            done = {r["id"] for r in read_jsonl(jsonl)}
            print(f"resuming: {len(done)} chunks already done")
        elif args.overwrite:
            os.remove(jsonl)
        else:
            raise SystemExit(f"{jsonl} exists: use --resume or --overwrite")

    model = load_model(args)
    n_layers = model.config.num_hidden_layers
    if max(args.s_values) >= n_layers:
        raise SystemExit(f"s values must be < {n_layers}")

    # Check 1: hook check on the first chunk.
    c0 = chunks[0]
    art0 = articles[c0["article"]]
    lo = c0["lo"]
    hc = hook_check(model, art0.ids[lo: lo + args.ctx_len],
                    art0.ids[lo + args.ctx_len: lo + args.ctx_len + args.cont_len], args.hook_tol)
    print(f"[check 1] hook check: {hc}")
    if not hc["ok"]:
        raise SystemExit("hook check failed: do not trust any number from this setup")

    import transformers
    with open(os.path.join(args.out_dir, "exp1_meta.json"), "w") as f:
        json.dump({"args": vars(args), "n_layers": n_layers, "hook_check": hc,
                   "n_articles": len(articles), "torch": torch.__version__,
                   "transformers": transformers.__version__}, f, indent=2)

    t0 = time.time()
    n_run = 0
    with open(jsonl, "a", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            if chunk["id"] in done:
                continue
            art, dropped, excluded, ours = analyze_chunk(articles, chunk, args)
            rec = run_chunk(model, art, chunk, dropped, excluded, ours, args)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_run += 1
            if n_run % 10 == 0 or i == len(chunks) - 1:
                el = time.time() - t0
                print(f"  {i + 1}/{len(chunks)} chunks  {el / n_run:.1f}s/chunk  "
                      f"last k={rec['k']}", flush=True)

    summarize(args.out_dir, args)


if __name__ == "__main__":
    main()
