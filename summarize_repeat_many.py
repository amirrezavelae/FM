"""Summarize run_kv_repeat_test.py output over many words, split by needle class.

For each variant and set (prefix / last / word / control): the share of words whose
greedy output is still the exact word, per start layer s (layers s..27 modified,
s = 28 is the unmodified baseline). Words are grouped by class from exp3_needles.csv
('ref' = words not in that file, e.g. unhappiness). Also prints the mean log-prob drop.
Plain Python, no pandas needed.
"""
import argparse
import csv
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--needles", default=None)
    ap.add_argument("--mode", default="drop")
    args = ap.parse_args()

    cls = {}
    if args.needles:
        with open(args.needles, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cls[r["needle"]] = (r["class"], int(r["n_pieces"]))

    with open(args.csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] == args.mode]
    if not rows:
        print("no rows")
        return

    # average over control seeds first, so each word counts once
    acc = defaultdict(list)
    for r in rows:
        acc[(r["word"], r["variant"], r["set"], int(r["s"]))].append(
            (float(r["exact"]), float(r["logprob"]) - float(r["baseline_logprob"])))
    cell = {k: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for k, v in acc.items()}

    variants = sorted({k[1] for k in cell})
    sets = [s for s in ["prefix", "last", "word", "control"] if any(k[2] == s for k in cell)]
    svals = sorted({k[3] for k in cell})

    def group_of(word):
        return cls.get(word, ("ref", 0))

    for variant in variants:
        words = sorted({k[0] for k in cell if k[1] == variant})
        by_class = defaultdict(list)
        for w in words:
            by_class[group_of(w)[0]].append(w)
        print("=" * 100)
        print(f"variant: {variant}   words kept (clean model correct): "
              + ", ".join(f"{c}={len(v)}" for c, v in sorted(by_class.items())))
        groups = [("all", words)] + sorted(by_class.items())
        for st in sets:
            print(f"\n  set = {st}   exact-match share by start layer s (layers s..27 dropped)")
            print("  " + f"{'group':<10s}{'n':>5s}  " + "".join(f"{s:>7d}" for s in svals))
            for name, ws in groups:
                vals = []
                for s in svals:
                    xs = [cell[(w, variant, st, s)][0] for w in ws if (w, variant, st, s) in cell]
                    vals.append(f"{sum(xs) / len(xs):7.2f}" if xs else f"{'-':>7s}")
                print("  " + f"{name:<10s}{len(ws):>5d}  " + "".join(vals))
        # by number of pieces, prefix and last only
        for st in [s for s in ["prefix", "last"] if s in sets]:
            print(f"\n  set = {st}   by number of prompt pieces (needles only)")
            print("  " + f"{'pieces':<10s}{'n':>5s}  " + "".join(f"{s:>7d}" for s in svals))
            for lab, test in [("2", lambda p: p == 2), ("3", lambda p: p == 3), ("4+", lambda p: p >= 4)]:
                ws = [w for w in words if w in cls and test(cls[w][1])]
                if not ws:
                    continue
                vals = []
                for s in svals:
                    xs = [cell[(w, variant, st, s)][0] for w in ws if (w, variant, st, s) in cell]
                    vals.append(f"{sum(xs) / len(xs):7.2f}" if xs else f"{'-':>7s}")
                print("  " + f"{lab:<10s}{len(ws):>5d}  " + "".join(vals))
        print("\n  mean log-prob change vs clean (prefix / last / control), all words")
        print("  " + f"{'set':<10s}" + "".join(f"{s:>7d}" for s in svals))
        for st in [s for s in ["prefix", "last", "control"] if s in sets]:
            vals = []
            for s in svals:
                xs = [cell[(w, variant, st, s)][1] for w in words if (w, variant, st, s) in cell]
                vals.append(f"{sum(xs) / len(xs):7.2f}" if xs else f"{'-':>7s}")
            print("  " + f"{st:<10s}" + "".join(vals))


if __name__ == "__main__":
    main()
