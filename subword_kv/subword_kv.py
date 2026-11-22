"""
Shared pieces for the subword KV-dropping experiments.

  * WordIndex  - finds eligible multi-token words in a tokenized text and says
                 which token positions "Ours" drops (all but the last piece).
  * DropHooks  - masks chosen cache positions out of attention at layers >= s
                 (positions and RoPE are untouched).
  * bootstrap_ci - mean and 95% bootstrap CI.

Word definition (see "Subword KV Dropping Experiment Setup.md"):
  a word is a maximal run of letters (regex \\p{L}+). It is eligible when the
  tokenizer splits it into >= 2 tokens that lie exactly inside the word; the
  first token may additionally carry one leading space.
"""

import numpy as np
import regex
import torch

WORD_RE = regex.compile(r"\p{L}+")

# Reasons a word is excluded (logged for every excluded word).
SINGLE = "single_token"      # one token, nothing to drop
MISALIGNED = "misaligned"    # a token also carries non-letter text
PROTECTED = "protected"      # a would-be dropped piece is a protected position
TRUNCATED = "truncated"      # the word runs past the edge of the context window


class WordIndex:
    """Words of one text, with the token offsets of the same text.

    `offsets` is the list of (char_start, char_end) per token from a fast
    tokenizer run over `text` with add_special_tokens=False.
    """

    def __init__(self, text, ids, offsets, tok):
        self.text = text
        self.ids = list(ids)
        self.tok = tok
        self.offsets = [tuple(o) for o in offsets]
        self.starts = np.array([a for a, _ in self.offsets], dtype=np.int64)
        self.ends = np.array([b for _, b in self.offsets], dtype=np.int64)
        spans = [(m.start(), m.end()) for m in WORD_RE.finditer(text)]
        self.spans = spans
        self.word_starts = np.array([a for a, _ in spans], dtype=np.int64)
        self.word_ends = np.array([b for _, b in spans], dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    # ------------------------------------------------------------------ #
    def _aligned(self, ws, we, t0, t1):
        """Every token lies inside the word; the first may carry one leading space."""
        for j in range(t0, t1 + 1):
            a, b = self.offsets[j]
            if j == t0:
                lead_space = a == ws - 1 and ws > 0 and self.text[a] == " "
                if not (a == ws or lead_space):
                    return False
            elif a < ws:
                return False
            if b > we:
                return False
        return self.offsets[t1][1] == we

    def token_roles(self, lo, hi, known_words):
        """Role of each token in [lo, hi) with respect to multi-token words.

        "rep_first" / "rep_rest": first / later piece of a multi-token word whose
        spelling is in `known_words` (the words Ours dropped in the context, so
        the word re-appears after its prefix KV was removed).
        "new_first" / "new_rest": same, for multi-token words not in `known_words`.
        "other": everything else (single-token words, punctuation, spaces).
        A word that starts before `lo` has its in-window pieces marked "_rest".
        """
        roles = ["other"] * (hi - lo)
        c0, c1 = int(self.starts[lo]), int(self.ends[hi - 1])
        i0 = int(np.searchsorted(self.word_ends, c0, side="right"))
        i1 = int(np.searchsorted(self.word_starts, c1, side="left"))
        for ws, we in self.spans[i0:i1]:
            t0 = int(np.searchsorted(self.ends, ws, side="right"))
            t1 = int(np.searchsorted(self.starts, we, side="left")) - 1
            if t1 <= t0:
                continue
            kind = "rep" if self.text[ws:we] in known_words else "new"
            for t in range(max(t0, lo), min(t1, hi - 1) + 1):
                roles[t - lo] = f"{kind}_{'first' if t == t0 else 'rest'}"
        return roles

    def analyze(self, lo, hi, protected):
        """Classify every word that overlaps the token window [lo, hi).

        `protected` holds window-relative positions that may never be dropped.
        Returns (dropped, excluded):
          dropped  = [{"word", "pieces", "dropped_positions"}]   (window-relative)
          excluded = [{"word", "reason"}]
        """
        c0, c1 = int(self.starts[lo]), int(self.ends[hi - 1])
        i0 = int(np.searchsorted(self.word_ends, c0, side="right"))
        i1 = int(np.searchsorted(self.word_starts, c1, side="left"))
        dropped, excluded = [], []
        for ws, we in self.spans[i0:i1]:
            word = self.text[ws:we]
            t0 = int(np.searchsorted(self.ends, ws, side="right"))
            t1 = int(np.searchsorted(self.starts, we, side="left")) - 1
            if t1 < t0:
                excluded.append({"word": word, "reason": MISALIGNED})
            elif t1 == t0:
                excluded.append({"word": word, "reason": SINGLE})
            elif t0 < lo or t1 > hi - 1:
                excluded.append({"word": word, "reason": TRUNCATED})
            elif not self._aligned(ws, we, t0, t1):
                excluded.append({"word": word, "reason": MISALIGNED})
            else:
                pos = [t - lo for t in range(t0, t1)]  # all pieces but the last
                if any(p in protected for p in pos):
                    excluded.append({"word": word, "reason": PROTECTED})
                else:
                    pieces = [self.tok.decode([self.ids[t]]) for t in range(t0, t1 + 1)]
                    dropped.append({"word": word, "pieces": pieces,
                                    "dropped_positions": pos})
        return dropped, excluded


# --------------------------------------------------------------------------- #
# Dropping: per-layer attention mask through forward pre-hooks
# --------------------------------------------------------------------------- #
class DropHooks:
    """Mask `positions` out of attention in layers >= start_layer.

    The pre-hook replaces the attention mask of each such layer with an explicit
    additive causal mask that also blocks the dropped cache positions. Position
    ids and RoPE are untouched. Batch size 1, no padding. With an empty
    `positions` list the mask is plain causal (used by the hook check).
    """

    def __init__(self, model, positions, start_layer):
        self.handles = []
        self.positions = sorted(set(int(p) for p in positions))
        layers = model.model.layers
        for i in range(start_layer, len(layers)):
            self.handles.append(layers[i].self_attn.register_forward_pre_hook(
                self._make_hook(i), with_kwargs=True))

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
# Statistics
# --------------------------------------------------------------------------- #
def cluster_ratio_ci(sums, counts, groups, n_boot=1000, seed=0, alpha=0.05):
    """Ratio estimator sum(sums)/sum(counts) with a bootstrap over clusters.

    sums, counts: per-unit arrays (a unit is a chunk). groups: per-unit cluster
    label (the article). Clusters are resampled with replacement, so repeated
    chunks from one article are not treated as independent. Per-unit means use
    counts = 1. Returns (estimate, lo, hi); NaN when nothing is counted.
    """
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    labels, gi = np.unique(np.asarray(groups), return_inverse=True)
    gs = np.bincount(gi, weights=sums, minlength=len(labels))
    gc = np.bincount(gi, weights=counts, minlength=len(labels))
    if gc.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labels), size=(n_boot, len(labels)))
    den = gc[idx].sum(axis=1)
    ok = den > 0
    est = gs[idx].sum(axis=1)[ok] / den[ok]
    lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(gs.sum() / gc.sum()), float(lo), float(hi)


def bootstrap_ci(x, n_boot=1000, seed=0, alpha=0.05):
    """Mean and percentile bootstrap CI over examples. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)
