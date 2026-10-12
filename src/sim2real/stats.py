"""Statistics for the sim-to-real experiment.

Everything here works on per-utterance numbers so that comparisons stay
paired: the same sentence is scored under every voice, channel and ASR model.

Contents
- Text scoring: ``normalize``, ``wer_counts``, ``utterance_wer``,
  ``corpus_wer``, ``keyword_hits``, ``keyword_recall``.
- Paired comparison: ``paired_bootstrap_ci`` (difference of means).
- Keyword detection: ``mcnemar`` (exact), ``rule_of_three``,
  ``clopper_pearson`` and ``clopper_pearson_upper_zero``.
- Lab vs reality: ``decision_flip``.
- Prediction-powered inference: ``ppi_mean`` (Angelopoulos et al., 2023,
  "Prediction-powered inference", Science), with optional power tuning
  (PPI++, Angelopoulos, Duchi and Zrnic, 2023).

All bootstrap functions take an explicit ``seed`` so results are repeatable.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

import jiwer
import numpy as np
from scipy import stats as _st

# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------

#: Emergency keywords (multi-word phrases allowed). Matched on normalized
#: text as whole-word sequences.
EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "help",
    "bleeding",
    "blood",
    "gun",
    "shot",
    "knife",
    "stabbed",
    "fire",
    "smoke",
    "not breathing",
    "chest pain",
    "unconscious",
    "overdose",
    "heart attack",
    "stroke",
    "seizure",
    "ambulance",
    "police",
    "choking",
    "fell",
)

_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (keep apostrophes), collapse spaces."""
    t = text.lower().replace("-", " ")
    t = _NON_WORD.sub(" ", t)
    return " ".join(t.split())


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """Return ``(errors, n_ref_words)`` after normalization.

    errors = substitutions + deletions + insertions. An empty reference with
    a non-empty hypothesis counts every hypothesis word as an insertion.
    """
    r, h = normalize(ref), normalize(hyp)
    n_ref = len(r.split())
    if n_ref == 0:
        return len(h.split()), 0
    if not h:
        return n_ref, n_ref
    o = jiwer.process_words(r, h)
    return o.substitutions + o.deletions + o.insertions, n_ref


def utterance_wer(ref: str, hyp: str) -> float:
    """Word error rate of one utterance (can exceed 1 with insertions)."""
    e, n = wer_counts(ref, hyp)
    return e / n if n else float(e > 0)


def corpus_wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Corpus WER = total errors / total reference words."""
    pairs = [wer_counts(r, h) for r, h in zip(refs, hyps, strict=True)]
    tot_e = sum(e for e, _ in pairs)
    tot_n = sum(n for _, n in pairs)
    return tot_e / tot_n if tot_n else float("nan")


def _contains(text_norm: str, phrase: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(normalize(phrase))}(?!\S)", text_norm) is not None


def keyword_hits(
    ref: str, hyp: str, keywords: Iterable[str] = EMERGENCY_KEYWORDS
) -> dict[str, bool]:
    """For each keyword present in ``ref``, whether it is also found in ``hyp``.

    Keywords absent from the reference are left out (recall only).
    """
    r, h = normalize(ref), normalize(hyp)
    return {k: _contains(h, k) for k in keywords if _contains(r, k)}


def keyword_recall(
    refs: Sequence[str], hyps: Sequence[str], keywords: Iterable[str] = EMERGENCY_KEYWORDS
) -> dict:
    """Pooled keyword recall over many utterances.

    Returns ``{"recall", "hits", "total", "per_keyword": {kw: (hits, total)}}``.
    """
    kws = tuple(keywords)
    per: dict[str, list[int]] = {}
    for r, h in zip(refs, hyps, strict=True):
        for k, ok in keyword_hits(r, h, kws).items():
            c = per.setdefault(k, [0, 0])
            c[0] += int(ok)
            c[1] += 1
    hits = sum(c[0] for c in per.values())
    total = sum(c[1] for c in per.values())
    return {
        "recall": hits / total if total else float("nan"),
        "hits": hits,
        "total": total,
        "per_keyword": {k: (c[0], c[1]) for k, c in per.items()},
    }


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def _boot_idx(n_items: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n_items, size=(n_boot, n_items))


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for ``mean(a) - mean(b)`` over paired items.

    ``a[i]`` and ``b[i]`` must be scores of the same utterance. Utterances are
    resampled with replacement, keeping pairs together.

    Returns ``{"diff", "lo", "hi", "p_a_better", "p_two_sided", "n_items"}``
    where ``p_a_better`` is the bootstrap share of replicates with
    ``mean(a) < mean(b)`` (a is better when lower, as for WER) and
    ``p_two_sided`` is a bootstrap sign test: ``2 * min(P(d<=0), P(d>=0))``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("a and b must be non-empty 1-D arrays of equal length")
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[_boot_idx(d.size, n, rng)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p2 = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {
        "diff": float(d.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_a_better": float((boots < 0).mean()),
        "p_two_sided": float(p2),
        "n_items": int(d.size),
    }


# ---------------------------------------------------------------------------
# Keyword detection tests
# ---------------------------------------------------------------------------


def mcnemar(b01: int, b10: int) -> dict:
    """Exact (binomial) two-sided McNemar test on discordant pairs.

    ``b01``: items system A missed but B caught; ``b10``: A caught, B missed.
    Under H0 each discordant pair is a fair coin. Returns
    ``{"b01", "b10", "n_discordant", "p_value"}`` (p = 1 when no discordant).
    """
    if b01 < 0 or b10 < 0:
        raise ValueError("counts must be non-negative")
    m = b01 + b10
    p = 1.0 if m == 0 else float(_st.binomtest(min(b01, b10), m, 0.5).pvalue)
    return {"b01": int(b01), "b10": int(b10), "n_discordant": int(m), "p_value": min(1.0, p)}


def rule_of_three(n: int) -> float:
    """Approximate one-sided 95% upper bound on a rate after 0 events in n."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 3.0 / n


def clopper_pearson_upper_zero(n: int, alpha: float = 0.05) -> float:
    """Exact one-sided ``1 - alpha`` upper bound after 0 events in n trials.

    Solves ``(1 - p) ** n = alpha``, i.e. ``p = 1 - alpha ** (1 / n)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 1.0 - alpha ** (1.0 / n)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided ``1 - alpha`` Clopper-Pearson interval for k/n."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError("need 0 <= k <= n and n > 0")
    lo = 0.0 if k == 0 else float(_st.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# Decision flip: does the lab pick the same winner as reality?
# ---------------------------------------------------------------------------


def decision_flip(
    scores_by_condition: Mapping[str, Mapping[str, Sequence[float]]],
    human_key: str = "human",
    systems: tuple[str, str] | None = None,
    n: int = 10000,
    seed: int = 0,
    lower_is_better: bool = True,
) -> dict:
    """Compare the winner of two systems under each condition to the human one.

    ``scores_by_condition[cond][system]`` is a per-utterance score array (for
    example WER). ``human_key`` names the reference condition (real speech).

    When a condition has the same number of utterances as the human condition,
    items are assumed aligned (same sentence index) and the bootstrap resamples
    the same indices in both, which keeps the sentence-difficulty correlation.
    Otherwise the two conditions are resampled independently.

    Returns ``{"systems", "human": {...}, "conditions": {cond: {...}}}`` where
    each entry has the observed ``mean`` per system, ``gap`` (A minus B),
    ``winner``; conditions add ``flip`` (observed winner differs from human)
    and ``p_flip`` (bootstrap share of replicates in which the winners differ).
    The human entry adds ``p_winner_stable``: how often the human winner holds
    under resampling, a hint of whether reality itself has a clear winner.
    """
    if human_key not in scores_by_condition:
        raise KeyError(f"missing human condition {human_key!r}")
    human = scores_by_condition[human_key]
    if systems is None:
        names = list(human)
        if len(names) != 2:
            raise ValueError("pass systems=(A, B) when there are not exactly two")
        systems = (names[0], names[1])
    sa, sb = systems
    sign = 1.0 if lower_is_better else -1.0
    rng = np.random.default_rng(seed)

    def gap_arr(cond: Mapping[str, Sequence[float]]) -> np.ndarray:
        a, b = np.asarray(cond[sa], float), np.asarray(cond[sb], float)
        if a.shape != b.shape:
            raise ValueError("both systems must be scored on the same utterances")
        return a - b

    def winner(g: float) -> str:
        if g == 0:
            return "tie"
        return sa if sign * g < 0 else sb

    def summary(cond: Mapping[str, Sequence[float]]) -> dict:
        g = gap_arr(cond)
        return {
            "mean": {sa: float(np.mean(cond[sa])), sb: float(np.mean(cond[sb]))},
            "gap": float(g.mean()),
            "winner": winner(float(g.mean())),
            "n_items": int(g.size),
        }

    gh = gap_arr(human)
    idx_h = _boot_idx(gh.size, n, rng)
    boot_h = gh[idx_h].mean(axis=1)
    hs = summary(human)
    hs["p_winner_stable"] = float((np.sign(boot_h) == np.sign(hs["gap"])).mean())

    out: dict = {"systems": [sa, sb], "human": hs, "conditions": {}}
    for name, cond in scores_by_condition.items():
        if name == human_key:
            continue
        g = gap_arr(cond)
        idx = idx_h if g.size == gh.size else _boot_idx(g.size, n, rng)
        boot = g[idx].mean(axis=1)
        s = summary(cond)
        s["flip"] = s["winner"] != hs["winner"]
        s["p_flip"] = float((np.sign(boot) != np.sign(boot_h)).mean())
        s["paired_with_human"] = bool(g.size == gh.size)
        out["conditions"][name] = s
    return out


# ---------------------------------------------------------------------------
# Prediction-powered inference for a mean
# ---------------------------------------------------------------------------


def ppi_mean(
    labeled_human: Sequence[float],
    labeled_synth: Sequence[float],
    unlabeled_synth: Sequence[float],
    alpha: float = 0.05,
    power_tuning: bool = False,
) -> dict:
    """Estimate the human-speech mean from synthetic proxies plus few humans.

    Setup, for one ASR system: a large set of sentences is scored on synthetic
    audio only (``unlabeled_synth``, size N); a small subset is scored on both
    human audio (``labeled_human``, Y) and synthetic audio (``labeled_synth``,
    f), aligned by sentence (size n). The synthetic score plays the role of
    the "prediction" in Angelopoulos et al. (2023).

    Estimator: ``theta = lam * mean(f_unl) + mean(Y - lam * f_lab)`` with
    variance ``lam^2 var(f_unl)/N + var(Y - lam f_lab)/n`` and a normal CI.
    Classic PPI uses ``lam = 1``. With ``power_tuning`` (PPI++), ``lam`` is the
    plug-in optimum ``cov(Y, f) / ((1 + n/N) var(f))`` clipped to [0, 1], so
    PPI is never much worse than the human-only estimate.

    The synthetic bias (sim-to-real gap) is corrected by the rectifier
    ``mean(Y - f)``; PPI is valid whatever the bias, it only gains width when
    synthetic and human scores are correlated per sentence.

    Returns ``{"estimate", "lo", "hi", "width", "lam", "naive": {...},
    "synth_only": {...}, "width_ratio", "n_labeled", "n_unlabeled"}``.
    """
    y = np.asarray(labeled_human, float)
    f = np.asarray(labeled_synth, float)
    fu = np.asarray(unlabeled_synth, float)
    if y.shape != f.shape or y.ndim != 1:
        raise ValueError("labeled_human and labeled_synth must be aligned 1-D arrays")
    n_l, n_u = y.size, fu.size
    if n_l < 2 or n_u < 2:
        raise ValueError("need at least 2 labeled and 2 unlabeled items")
    z = float(_st.norm.ppf(1 - alpha / 2))

    lam = 1.0
    if power_tuning:
        vf = np.var(np.concatenate([f, fu]), ddof=1)
        lam = 0.0 if vf == 0 else float(np.cov(y, f, ddof=1)[0, 1] / ((1 + n_l / n_u) * vf))
        lam = float(np.clip(lam, 0.0, 1.0))

    rect = y - lam * f
    est = lam * fu.mean() + rect.mean()
    se = float(np.sqrt(lam**2 * fu.var(ddof=1) / n_u + rect.var(ddof=1) / n_l))

    se_naive = float(y.std(ddof=1) / np.sqrt(n_l))
    naive = {
        "estimate": float(y.mean()),
        "lo": float(y.mean() - z * se_naive),
        "hi": float(y.mean() + z * se_naive),
        "width": 2 * z * se_naive,
    }
    se_s = float(fu.std(ddof=1) / np.sqrt(n_u))
    synth_only = {
        "estimate": float(fu.mean()),
        "lo": float(fu.mean() - z * se_s),
        "hi": float(fu.mean() + z * se_s),
        "width": 2 * z * se_s,
        "note": "ignores the sim-to-real bias; shown only for contrast",
    }
    width = 2 * z * se
    return {
        "estimate": float(est),
        "lo": float(est - z * se),
        "hi": float(est + z * se),
        "width": width,
        "lam": lam,
        "naive": naive,
        "synth_only": synth_only,
        "width_ratio": width / naive["width"] if naive["width"] else float("nan"),
        "n_labeled": int(n_l),
        "n_unlabeled": int(n_u),
    }
