#!/usr/bin/env python
"""
Opposite test: drop the prefix pieces in a WINDOW of layers [a, b), not from a
start layer upward.

Exp 1 dropped layers s..27 (keep the early layers, prune the late ones; the
proposal's design). Its sweep showed the cost sits in layers >= 20. This test
asks the complementary question: what if the prefix pieces are dropped in the
LOW layers (0..e-1) and kept in the high layers?

Same data, chunks, word definition, protected positions, Random control and
metrics as Exp 1 (it imports them from ../subword_kv). A window is "a:b" and
means: drop in layers a, a+1, ..., b-1.

Default windows:
  0:4 0:8 0:12 0:16 0:20 0:24   prefix windows  (the opposite test proper)
  4:8 8:12 12:16 16:20 20:24 24:28  single 4-layer bands (which layers matter)

Random uses the same positions as Exp 1 for the same --seed and chunk, so the
runs are comparable with the Exp 1 sweep (use --random_seeds 1 there too).

Outputs in --out_dir:
  opposite_examples.jsonl   one JSON line per chunk (with per-position data)
  opposite_summary.csv      one row per condition
  opposite_paired.csv       Ours-Random and Ours-Full per window (chunk and article CIs)
  opposite_slices.csv       the same by target-token role (rep_first, ..., other)
  opposite_meta.json        args, versions, hook/window check
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "subword_kv"))

import exp1_wikitext as E  # noqa: E402  (shared data, word rules, metrics)
from subword_kv import DropHooks, bootstrap_ci, cluster_ratio_ci  # noqa: E402

DEFAULT_WINDOWS = ["0:4", "0:8", "0:12", "0:16", "0:20", "0:24",
                   "4:8", "8:12", "12:16", "16:20", "20:24", "24:28"]
EXP = "wikitext_opposite"


def tag(a, b):
    return f"w{a}-{b}"


def parse_windows(specs):
    out = []
    for sp in specs:
        a, b = (int(x) for x in sp.split(":"))
        if not 0 <= a < b:
            raise SystemExit(f"bad window {sp!r}: need 0 <= a < b")
        out.append((a, b))
    return list(dict.fromkeys(out))


# --------------------------------------------------------------------------- #
# Model side
# --------------------------------------------------------------------------- #
def cont_logits(model, cache, cont_ids, positions, a, b):
    """Continuation logits on a copy of the cache, dropping `positions` in layers [a, b)."""
    import copy
    c = copy.deepcopy(cache)
    x = torch.tensor([cont_ids], device=model.device)
    hooks = DropHooks(model, positions, a, b) if positions else None
    try:
        return model(input_ids=x, past_key_values=c, use_cache=True).logits[0].float()
    finally:
        if hooks:
            hooks.remove()


@torch.no_grad()
def window_check(model, ctx_ids, cont_ids, tol, n_layers):
    """Checks for the new layer-window hook on top of Exp 1's hook check.

      A: window [0, L) equals the old 'from layer 0' drop      (must be identical)
      B: window [L-1, L) equals the old 'from layer L-1' drop  (must be identical)
      C: window [0, 1) alone visibly changes the logits        (the window is active)
      D: the two halves [0, L/2) and [L/2, L) both change the logits
    """
    cache = E.prefill(model, ctx_ids)
    pos = list(range(1, len(ctx_ids) - 1))
    plain = cont_logits(model, cache, cont_ids, [], 0, n_layers)
    old0 = E.continuation_logits(model, cache, cont_ids, pos, 0)
    oldl = E.continuation_logits(model, cache, cont_ids, pos, n_layers - 1)
    maxd = lambda x, y: (x - y).abs().max().item()  # noqa: E731
    h = n_layers // 2
    res = {
        "A_window_full_vs_old_s0": maxd(cont_logits(model, cache, cont_ids, pos, 0, n_layers), old0),
        "B_window_last_vs_old_s_last": maxd(cont_logits(model, cache, cont_ids, pos, n_layers - 1, n_layers), oldl),
        "C_layer0_only_vs_plain": maxd(cont_logits(model, cache, cont_ids, pos, 0, 1), plain),
        "D_lower_half_vs_plain": maxd(cont_logits(model, cache, cont_ids, pos, 0, h), plain),
        "D_upper_half_vs_plain": maxd(cont_logits(model, cache, cont_ids, pos, h, n_layers), plain),
    }
    res["ok"] = (res["A_window_full_vs_old_s0"] <= tol
                 and res["B_window_last_vs_old_s_last"] <= tol
                 and res["C_layer0_only_vs_plain"] > 10 * tol
                 and res["D_lower_half_vs_plain"] > 10 * tol
                 and res["D_upper_half_vs_plain"] > 10 * tol)
    return res


@torch.no_grad()
def run_chunk(model, art, chunk, dropped, excluded, ours, args, windows):
    lo, C, L = chunk["lo"], args.ctx_len, args.cont_len
    ctx_ids = art.ids[lo: lo + C]
    cont_ids = art.ids[lo + C: lo + C + L]
    tgt = torch.tensor(art.ids[lo + C + 1: lo + C + L + 1], device=model.device)
    k = len(ours)
    n_layers = args.n_layers

    cache = E.prefill(model, ctx_ids)
    lp_full = F.log_softmax(cont_logits(model, cache, cont_ids, [], 0, n_layers), -1)
    nll_full = -lp_full.gather(1, tgt[:, None])[:, 0]
    full = {"kl": 0.0, "top1": 1.0, "dloss": 0.0, "loss": nll_full.mean().item()}

    rec = {"exp": EXP, "id": chunk["id"], "prompt_len": C, "cont_len": L,
           "eligible": k > 0, "k": k, "drop_frac": k / C,
           "dropped_words": dropped, "excluded_words": excluded,
           "random_positions": {}, "metrics": {"full": full}, "metrics_by_seed": {}}
    if k == 0:
        return rec

    # Random: same sampling (and seeds) as Exp 1, so positions match that run.
    pool = [p for p in range(C) if p not in {0, C - 1} and p not in set(ours)]
    k_rand = min(k, len(pool))
    rand_sets = {}
    for r in range(args.random_seeds):
        rng = random.Random(f"{args.seed}-{chunk['id']}-{r}")
        rand_sets[r] = sorted(rng.sample(pool, k_rand))
    rec["random_positions"] = {str(r): p for r, p in rand_sets.items()}

    known = {w["word"] for w in dropped}
    t_lo = lo + C + 1
    pp = {"target_ids": art.ids[t_lo: t_lo + L],
          "roles": art.token_roles(t_lo, t_lo + L, known),
          "full_nll": [round(v, 5) for v in nll_full.tolist()]}

    for a, b in windows:
        t = tag(a, b)
        lp = F.log_softmax(cont_logits(model, cache, cont_ids, ours, a, b), -1)
        m, p = E.compare(lp_full, lp, tgt)
        rec["metrics"][f"ours_{t}"] = m
        pp[f"ours_{t}"] = E.round_pos(p)
        ms, ps = [], []
        for r in range(args.random_seeds):
            lp = F.log_softmax(cont_logits(model, cache, cont_ids, rand_sets[r], a, b), -1)
            m, p = E.compare(lp_full, lp, tgt)
            ms.append(m)
            ps.append(p)
        rec["metrics"][f"random_{t}"] = E.mean_metrics(ms)
        rec["metrics_by_seed"][f"random_{t}"] = ms
        pp[f"random_{t}"] = E.round_pos(E.mean_pos(ps))
    rec["per_position"] = pp
    return rec


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarize(out_dir, args, windows, n_layers):
    recs = E.read_jsonl(os.path.join(out_dir, "opposite_examples.jsonl"))
    elig = [r for r in recs if r["eligible"]]
    groups = [E.article_of(r) for r in elig]
    n_total, n_none = len(recs), len(recs) - len(elig)
    print(f"\n{n_total} chunks, {n_none} without eligible words (kept in Full only)")

    # ---- summary ---------------------------------------------------------- #
    conds = [("full", recs, 0)]
    for a, b in windows:
        conds += [(f"ours_{tag(a, b)}", elig, b - a), (f"random_{tag(a, b)}", elig, b - a)]
    rows = []
    for name, rs, width in conds:
        row = {"condition": name, "n": len(rs), "n_chunks_total": n_total,
               "n_chunks_no_words": n_none,
               "mean_k": float(np.mean([r["k"] for r in rs])) if name != "full" and rs else 0.0,
               "mean_drop_frac": float(np.mean([r["drop_frac"] for r in rs])) if name != "full" and rs else 0.0}
        row["layers_dropped"] = width
        row["mem_saving"] = row["mean_drop_frac"] * width / n_layers
        for m in E.METRICS:
            mean, lo, hi = bootstrap_ci([r["metrics"][name][m] for r in rs], args.bootstrap)
            row[f"{m}_mean"], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        rows.append(row)
    E.write_csv(os.path.join(out_dir, "opposite_summary.csv"), rows)
    by_name = {r["condition"]: r for r in rows}

    # ---- paired ----------------------------------------------------------- #
    paired = []
    for a, b in windows:
        t = tag(a, b)
        for m in ("kl", "top1", "dloss"):
            ours = np.array([r["metrics"][f"ours_{t}"][m] for r in elig])
            rand = np.array([r["metrics"][f"random_{t}"][m] for r in elig])
            full = np.array([r["metrics"]["full"][m] for r in elig])
            for label, diff in (("ours_minus_random", ours - rand), ("ours_minus_full", ours - full)):
                mean, lo, hi = bootstrap_ci(diff, args.bootstrap)
                _, alo, ahi = cluster_ratio_ci(diff, np.ones(len(diff)), groups, args.bootstrap)
                paired.append({"window": t, "a": a, "b": b, "comparison": label, "metric": m,
                               "n": len(diff), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                               "ci_lo_article": alo, "ci_hi_article": ahi})
    E.write_csv(os.path.join(out_dir, "opposite_paired.csv"), paired)

    # ---- slices by target-token role -------------------------------------- #
    pp = [r for r in elig if "per_position" in r]
    srows = []
    pgroups = [E.article_of(r) for r in pp]
    for a, b in windows:
        t = tag(a, b)
        for role in E.ROLES + ["all"]:
            masks = [np.array([role == "all" or x == role for x in r["per_position"]["roles"]])
                     for r in pp]
            counts = np.array([m.sum() for m in masks], dtype=float)
            if counts.sum() == 0:
                continue
            for metric in ("kl", "dnll", "top1"):
                tot = {c: np.array([np.array(r["per_position"][f"{c}_{t}"][metric])[m].sum()
                                    for r, m in zip(pp, masks)]) for c in ("ours", "random")}
                for label, v in (("ours", tot["ours"]), ("random", tot["random"]),
                                 ("ours_minus_random", tot["ours"] - tot["random"])):
                    est, lo, hi = cluster_ratio_ci(v, counts, pgroups, args.bootstrap)
                    srows.append({"window": t, "a": a, "b": b, "role": role, "metric": metric,
                                  "quantity": label, "n_positions": int(counts.sum()),
                                  "mean": est, "ci_lo": lo, "ci_hi": hi})
    E.write_csv(os.path.join(out_dir, "opposite_slices.csv"), srows)

    # ---- console ---------------------------------------------------------- #
    full_loss = by_name["full"]["loss_mean"]
    print(f"Full log-loss {full_loss:.3f} nats/token (ppl {np.exp(full_loss):.1f})")

    def sl(t, role, metric, q="ours_minus_random"):
        for r in srows:
            if r["window"] == t and r["role"] == role and r["metric"] == metric and r["quantity"] == q:
                return r
        return None

    sweep = {}
    sp = os.path.join(args.compare_sweep, "exp1_summary.csv") if args.compare_sweep else ""
    if sp and os.path.exists(sp):
        import csv
        with open(sp, encoding="utf-8") as f:
            sweep = {r["condition"]: r for r in csv.DictReader(f)}
        print(f"(complement column: Exp 1 sweep, same chunks, drop layers [b, {n_layers}) instead)")

    print(f"\n{'window':<8s}{'saving%':>8s}{'KL ours':>9s}{'KL rand':>9s}"
          f"{'ours-rand KL [95% CI, articles]':>36s}{'rep_first':>10s}{'other':>8s}"
          f"{'complement KL ours/rand':>26s}")
    for a, b in windows:
        t = tag(a, b)
        po = by_name[f"ours_{t}"]
        pr = by_name[f"random_{t}"]
        d = next(p for p in paired if p["window"] == t and p["comparison"] == "ours_minus_random"
                 and p["metric"] == "kl")
        rf, ot = sl(t, "rep_first", "kl"), sl(t, "other", "kl")
        comp = ""
        if a == 0 and f"ours_s{b}" in sweep:
            comp = f"{float(sweep[f'ours_s{b}']['kl_mean']):.4f}/{float(sweep[f'random_s{b}']['kl_mean']):.4f}"
        print(f"{t:<8s}{100 * po['mem_saving']:>8.2f}{po['kl_mean']:>9.4f}{pr['kl_mean']:>9.4f}"
              f"{d['mean_diff']:>+14.4f} [{d['ci_lo_article']:+.4f},{d['ci_hi_article']:+.4f}]"
              f"{rf['mean'] if rf else float('nan'):>+10.3f}{ot['mean'] if ot else float('nan'):>+8.4f}"
              f"{comp:>26s}")
    print("\nrep_first / other columns: Ours-Random KL on first pieces of re-appearing "
          "dropped words / on ordinary tokens.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--n_chunks", type=int, default=100)
    ap.add_argument("--ctx_len", type=int, default=512)
    ap.add_argument("--cont_len", type=int, default=64)
    ap.add_argument("--windows", nargs="+", default=DEFAULT_WINDOWS, metavar="A:B",
                    help="layer windows [A,B) in which the prefix pieces are dropped")
    ap.add_argument("--random_seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--hook_tol", type=float, default=1e-3)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out_dir", default="results/opposite")
    ap.add_argument("--compare_sweep",
                    default=os.path.join(HERE, "..", "subword_kv", "results", "exp1_sweep"),
                    help="Exp 1 sweep folder, for the complement column ('' to skip)")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    jsonl = os.path.join(args.out_dir, "opposite_examples.jsonl")
    meta_path = os.path.join(args.out_dir, "opposite_meta.json")
    windows = parse_windows(args.windows)

    if args.summarize_only:
        n_layers = 28
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, encoding="utf-8"))
            n_layers = meta.get("n_layers", n_layers)
            if args.windows == DEFAULT_WINDOWS and "windows" in meta:
                windows = [tuple(w) for w in meta["windows"]]
        summarize(args.out_dir, args, windows, n_layers)
        return

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise SystemExit("a fast tokenizer is needed for offset mappings")
    articles = E.load_articles(tok)
    chunks = E.sample_chunks(articles, args)
    print(f"{len(articles)} test articles, {len(chunks)} chunks; windows: "
          + " ".join(tag(a, b) for a, b in windows))

    done = set()
    if os.path.exists(jsonl):
        if args.resume:
            done = {r["id"] for r in E.read_jsonl(jsonl)}
            print(f"resuming: {len(done)} chunks already done")
        elif args.overwrite:
            os.remove(jsonl)
        else:
            raise SystemExit(f"{jsonl} exists: use --resume or --overwrite")

    model = E.load_model(args)
    n_layers = model.config.num_hidden_layers
    args.n_layers = n_layers
    if max(b for _, b in windows) > n_layers:
        raise SystemExit(f"windows must end at or below {n_layers}")

    # Check 1 (hook, from Exp 1) plus the window checks.
    c0 = chunks[0]
    art0 = articles[c0["article"]]
    lo = c0["lo"]
    ctx0 = art0.ids[lo: lo + args.ctx_len]
    cont0 = art0.ids[lo + args.ctx_len: lo + args.ctx_len + args.cont_len]
    hc = E.hook_check(model, ctx0, cont0, args.hook_tol)
    print(f"[check 1] hook check: {hc}")
    wc = window_check(model, ctx0, cont0, args.hook_tol, n_layers)
    print(f"[check 1b] window check: {wc}")
    if not (hc["ok"] and wc["ok"]):
        raise SystemExit("hook/window check failed: do not trust any number from this setup")

    import transformers
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"args": {k: v for k, v in vars(args).items()}, "n_layers": n_layers,
                   "windows": windows, "hook_check": hc, "window_check": wc,
                   "torch": torch.__version__, "transformers": transformers.__version__},
                  f, indent=2)

    t0 = time.time()
    n_run = 0
    with open(jsonl, "a", encoding="utf-8") as out:
        for i, chunk in enumerate(chunks):
            if chunk["id"] in done:
                continue
            art, dropped, excluded, ours = E.analyze_chunk(articles, chunk, args)
            rec = run_chunk(model, art, chunk, dropped, excluded, ours, args, windows)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n_run += 1
            if n_run % 10 == 0 or i == len(chunks) - 1:
                print(f"  {i + 1}/{len(chunks)} chunks  {(time.time() - t0) / n_run:.1f}s/chunk",
                      flush=True)

    summarize(args.out_dir, args, windows, n_layers)


if __name__ == "__main__":
    main()
