"""Calibrating synthetic voices against a few human recordings (follow-up to v2).

Three questions, all on existing per-utterance results (no new transcription):

1. Voice fidelity: does a synthetic voice's per-sentence error predict the human
   per-sentence error (correlation, calibration slope, agreement on model differences)?
2. Calibrated weighting: with a small human-labeled calibration set from speakers
   disjoint from the evaluation speakers, do fidelity weights, a per-voice linear
   correction or PPI / PPI++ remove the false confidence of the plain voice average?
3. Active selection: does choosing which sentences to record from synthetic
   disagreement give narrower valid intervals than random selection at equal n?

Per-sentence scores are ``errors / c`` with a constant ``c`` (a mean word count), so the
mean of the scores over a set of sentences is the corpus WER of that set when ``c`` is the
set's mean word count. All functions take plain arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _st

from . import stats

SYNTH = ["kokoro:af_heart", "kokoro:am_michael", "piper:en_US-lessac-medium",
         "piper:en_US-ryan-medium", "mms:eng", "say:Samantha", "say:Daniel"]
RATE = [f"{v}@rate" for v in SYNTH]


# --------------------------------------------------------------------------- data

@dataclass
class LibriData:
    ids: list
    n: np.ndarray          # reference words per sentence
    speaker: np.ndarray    # speaker id per sentence (str)
    E: dict                # (voice, condition, model) -> errors per sentence (float array)

    def has(self, *keys) -> bool:
        return all(k in self.E for k in keys)


def load_librispeech(utterances_csv) -> LibriData:
    u = pd.read_csv(utterances_csv, low_memory=False)
    d = u[u["set"] == "librispeech"]
    ids = sorted(d["id"].unique())
    first = d.drop_duplicates("id").set_index("id").loc[ids]
    piv = d.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                        aggfunc="first").reindex(ids)
    E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
    return LibriData(ids, first["n_ref"].to_numpy(float), first["speaker"].astype(str).to_numpy(), E)


def synth_mean(D: LibriData, cond: str, model: str, voices=SYNTH) -> np.ndarray:
    return np.mean([D.E[(v, cond, model)] for v in voices], axis=0)


# --------------------------------------------------------------------------- weighted moments

def wcorr(x: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Pearson correlation of x and y under each row of a weight matrix W (B x N)."""
    sw = W.sum(1)
    mx, my = W @ x / sw, W @ y / sw
    sxx = W @ (x * x) / sw - mx ** 2
    syy = W @ (y * y) / sw - my ** 2
    sxy = W @ (x * y) / sw - mx * my
    den = np.sqrt(np.clip(sxx * syy, 0, None))
    return np.where(den > 0, sxy / np.where(den > 0, den, 1), np.nan)


def wslope(f: np.ndarray, y: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS slope and intercept of y on f under each row of W; slope 0 when f is constant."""
    sw = W.sum(1)
    mf, my = W @ f / sw, W @ y / sw
    vf = W @ (f * f) / sw - mf ** 2
    cf = W @ (f * y) / sw - mf * my
    b = np.where(vf > 1e-15, cf / np.where(vf > 1e-15, vf, 1), 0.0)
    return b, my - b * mf


def pct_ci(boot: np.ndarray) -> tuple[float, float]:
    b = boot[np.isfinite(boot)]
    if b.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- Q1 fidelity

def fidelity_cell(f: np.ndarray, y: np.ndarray, W: np.ndarray) -> dict:
    """Fidelity of synthetic per-sentence scores f against human scores y (same sentences)."""
    one = np.ones((1, f.size))
    r = float(wcorr(f, y, one)[0])
    b, a = (float(z[0]) for z in wslope(f, y, one))
    rb = wcorr(f, y, W)
    bb, _ = wslope(f, y, W)
    rho = float(_st.spearmanr(f, y).statistic) if np.std(f) > 0 and np.std(y) > 0 else float("nan")
    return {"pearson_r": r, "pearson_lo": pct_ci(rb)[0], "pearson_hi": pct_ci(rb)[1],
            "spearman_rho": rho, "calib_slope": b, "calib_slope_lo": pct_ci(bb)[0],
            "calib_slope_hi": pct_ci(bb)[1], "calib_intercept": a}


def diff_agreement(df: np.ndarray, dy: np.ndarray, n: np.ndarray, speaker: np.ndarray,
                   W: np.ndarray) -> dict:
    """Agreement of synthetic and human per-sentence model differences (A minus B).

    sign_agree: among sentences where the human difference is non-zero, share where the
    synthetic difference has the same sign (a zero synthetic difference counts as disagreement,
    which penalizes single voices against the 7-voice mean, whose difference is rarely exactly 0;
    sign_agree_both_nonzero drops those sentences).
    speaker_r: correlation of per-speaker corpus-WER differences (40 points).
    """
    nz = dy != 0
    same = (np.sign(df) == np.sign(dy)) & nz
    sb = (W @ same) / np.maximum(W @ nz, 1)
    spk = pd.Series(speaker)
    g_df = pd.Series(df * 1.0).groupby(spk).sum() / pd.Series(n).groupby(spk).sum()
    g_dy = pd.Series(dy * 1.0).groupby(spk).sum() / pd.Series(n).groupby(spk).sum()
    rs = float(np.corrcoef(g_df, g_dy)[0, 1]) if g_df.std() > 0 and g_dy.std() > 0 else float("nan")
    rb = wcorr(df, dy, W)
    return {"diff_r": float(np.corrcoef(df, dy)[0, 1]) if df.std() > 0 and dy.std() > 0 else float("nan"),
            "diff_r_lo": pct_ci(rb)[0], "diff_r_hi": pct_ci(rb)[1],
            "n_human_nonzero": int(nz.sum()), "sign_agree": float(same.sum() / max(nz.sum(), 1)),
            "sign_agree_lo": pct_ci(sb)[0], "sign_agree_hi": pct_ci(sb)[1],
            "sign_agree_both_nonzero": float(same.sum() / max((nz & (df != 0)).sum(), 1)),
            "n_both_nonzero": int((nz & (df != 0)).sum()),
            "speaker_diff_r": rs}


# --------------------------------------------------------------------------- Q2 estimators

def decide(lo, hi):
    """Three-way decision on A minus B (lower WER is better): 'A', 'B' or 'none'."""
    lo, hi = np.asarray(lo), np.asarray(hi)
    return np.where(hi < 0, "A", np.where(lo > 0, "B", "none"))


def two_sample_rectified(y_cal: np.ndarray, f_cal: np.ndarray, f_eval_mean: float, n_eval: int,
                         lam: float = 1.0, alpha: float = 0.05) -> dict:
    """PPI-style estimate of the human mean over a fixed evaluation set from labels on a
    disjoint calibration sample.

    est = lam * mean_eval(f) + mean_cal(y - lam f). For lam fixed, est - truth equals
    mean_cal(r) - mean_eval(r) with r = y - lam f, whose variance is estimated by
    S_r^2 (1/n + 1/N_eval) with S_r from the calibration sample (sentences treated as
    exchangeable across the two sets; speaker clustering is ignored, so coverage is checked
    by simulation). t interval with n - 1 df. lam = 0 gives the human-only estimate.
    """
    n = y_cal.size
    r = y_cal - lam * f_cal
    est = lam * f_eval_mean + r.mean()
    se = r.std(ddof=1) * np.sqrt(1 / n + 1 / n_eval)
    t = _st.t.ppf(1 - alpha / 2, n - 1)
    return {"est": float(est), "lo": float(est - t * se), "hi": float(est + t * se)}


def ppipp_lambda(y_cal: np.ndarray, f_cal: np.ndarray) -> float:
    """Plug-in PPI++ weight cov(y, f) / var(f) on the calibration sample (not clipped)."""
    vf = np.var(f_cal, ddof=1)
    return 0.0 if vf <= 1e-15 else float(np.cov(y_cal, f_cal, ddof=1)[0, 1] / vf)


def fidelity_weights(df_cal: np.ndarray, dy_cal: np.ndarray, floor_frac: float = 0.05) -> np.ndarray:
    """Inverse mean-squared-error weights per voice from calibration sentences.

    df_cal: (voices, n) synthetic per-sentence model differences; dy_cal: (n,) human ones.
    The MSE is floored at ``floor_frac`` of its mean so a voice that happens to match all
    calibration sentences exactly does not take all the weight.
    """
    mse = ((df_cal - dy_cal[None, :]) ** 2).mean(1)
    mse = np.maximum(mse, floor_frac * max(mse.mean(), 1e-12))
    w = 1.0 / mse
    return w / w.sum()


def linear_corrected(F_cal: dict, y_cal: dict, F_eval: dict, Wc: np.ndarray | None = None,
                     We: np.ndarray | None = None) -> np.ndarray:
    """Per-voice linear correction, averaged over voices, of the model difference.

    F_cal[m]: (voices, n) synthetic scores on calibration sentences for model m in (A, B);
    y_cal[m]: (n,) human scores; F_eval[m]: (voices, N_eval). For each voice and model,
    y = a + b f is fit by OLS on the calibration sentences and applied to the mean of f over
    the evaluation set; the prediction for A minus that for B is averaged over voices.
    With weight matrices Wc (B x n) and We (B x N_eval) it returns B bootstrap replicates.
    """
    Wc = np.ones((1, next(iter(y_cal.values())).size)) if Wc is None else Wc
    We = np.ones((1, next(iter(F_eval.values())).shape[1])) if We is None else We
    out = 0.0
    ms = list(F_cal)
    for sgn, m in zip((1.0, -1.0), ms):
        V = F_cal[m].shape[0]
        acc = 0.0
        for v in range(V):
            b, a = wslope(F_cal[m][v], y_cal[m], Wc)
            fe = We @ F_eval[m][v] / We.sum(1)
            acc = acc + a + b * fe
        out = out + sgn * acc / V
    return np.atleast_1d(out)


# --------------------------------------------------------------------------- Q3 designs

def srs_draws(rng, N: int, n: int, reps: int) -> np.ndarray:
    return np.argsort(rng.random((reps, N)), axis=1)[:, :n]


def make_strata(u: np.ndarray, H: int = 5) -> np.ndarray:
    """Equal-size strata by rank of the uncertainty score u (ties broken by sentence order)."""
    order = np.argsort(u, kind="stable")
    s = np.empty(u.size, int)
    s[order] = np.arange(u.size) * H // u.size
    return s


def neyman_allocation(strata: np.ndarray, u: np.ndarray, n: int, min_per: int = 2) -> np.ndarray:
    """Neyman allocation n_h proportional to N_h * S_h, with S_h proxied by the mean of u
    in the stratum plus a small floor; at least ``min_per`` per stratum, at most N_h."""
    H = strata.max() + 1
    Nh = np.bincount(strata, minlength=H).astype(float)
    Sh = np.array([u[strata == h].mean() for h in range(H)])
    Sh = Sh + 0.05 * max(Sh.mean(), 1e-12)
    raw = n * Nh * Sh / (Nh * Sh).sum()
    nh = np.maximum(min_per, np.floor(raw)).astype(int)
    nh = np.minimum(nh, Nh.astype(int))
    while nh.sum() < n:
        gap = np.where(nh < Nh, raw - nh, -np.inf)
        nh[int(np.argmax(gap))] += 1
    while nh.sum() > n:
        gap = np.where(nh > min_per, nh - raw, -np.inf)
        nh[int(np.argmax(gap))] -= 1
    return nh


def stratified_ppi(y: np.ndarray, f: np.ndarray, strata: np.ndarray, nh: np.ndarray, rng,
                   reps: int, alpha: float = 0.05, use_f: bool = True) -> dict:
    """Stratified sampling without replacement (n_h per stratum, strata fixed from synthetic
    scores only) with the PPI rectifier per stratum. Design-unbiased for the finite-population
    mean of y; variance sum_h W_h^2 (1 - n_h/N_h) s_{r,h}^2 / n_h; t with n - H df.
    use_f=False gives the stratified human-only estimator."""
    N = y.size
    r = y - f if use_f else y
    est = np.full(reps, f.mean() if use_f else 0.0)
    var = np.zeros(reps)
    H = strata.max() + 1
    for h in range(H):
        idx = np.flatnonzero(strata == h)
        Nh, k = idx.size, int(nh[h])
        pick = idx[srs_draws(rng, Nh, k, reps)]
        rh = r[pick]
        est += (Nh / N) * rh.mean(1)
        var += (Nh / N) ** 2 * (1 - k / Nh) * rh.var(1, ddof=1) / k
    t = _st.t.ppf(1 - alpha / 2, int(nh.sum()) - H)
    se = np.sqrt(var)
    return {"est": est, "lo": est - t * se, "hi": est + t * se, "width": 2 * t * se}


def poisson_probs(u: np.ndarray, n: int, mix: float = 0.2) -> np.ndarray:
    """Inclusion probabilities proportional to u, mixed with uniform, capped at 1, summing to n."""
    N = u.size
    s = u / u.sum() if u.sum() > 0 else np.full(N, 1 / N)
    p = n * ((1 - mix) * s + mix / N)
    for _ in range(50):
        over = p > 1
        if not over.any():
            break
        p[over] = 1.0
        rest = ~over
        p[rest] *= (n - over.sum()) / p[rest].sum()
    return np.clip(p, 1e-9, 1.0)


def poisson_ipw_ppi(y: np.ndarray, f: np.ndarray, pi: np.ndarray, rng, reps: int,
                    alpha: float = 0.05) -> dict:
    """Active-inference style estimator: each sentence is recorded independently with
    probability pi_i (Poisson sampling); est = mean(f) + (1/N) sum xi_i (y_i - f_i) / pi_i
    (Horvitz-Thompson rectifier), variance (1/N^2) sum xi_i (1 - pi_i) r_i^2 / pi_i^2, normal
    interval. Unbiased by design for any pi > 0; the sample size is random with mean sum(pi)."""
    N = y.size
    r = y - f
    xi = rng.random((reps, N)) < pi[None, :]
    est = f.mean() + (xi * (r / pi)).sum(1) / N
    var = (xi * ((1 - pi) * r ** 2 / pi ** 2)).sum(1) / N ** 2
    z = _st.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var)
    return {"est": est, "lo": est - z * se, "hi": est + z * se, "width": 2 * z * se,
            "n_drawn": xi.sum(1)}


def srs_ppi(y: np.ndarray, f: np.ndarray, n: int, rng, reps: int, use_f: bool = True,
            alpha: float = 0.05) -> dict:
    """Simple random sample without replacement; classic PPI (lam = 1) or human-only,
    finite-population correction, t with n - 1 df (same as stats.ppi_mean_finite)."""
    lab = srs_draws(rng, y.size, n, reps)
    r = stats.ppi_mean_finite(y[lab], f[lab], f.mean(), y.size, alpha=alpha,
                              method="ppi" if use_f else "naive")
    return {"est": r["estimate"], "lo": r["lo"], "hi": r["hi"], "width": r["width"]}


def equal_width_n(n_grid, widths_ref, target_width) -> float:
    """n at which a reference design reaches target_width (log-log interpolation on its
    curve of median CI width, or of RMSE, against n; the curve is made non-increasing first to
    absorb Monte Carlo noise); nan when outside the grid."""
    x = np.log(np.asarray(n_grid, float))
    w = np.log(np.minimum.accumulate(np.asarray(widths_ref, float)))
    t = np.log(target_width)
    if t > w[0] or t < w[-1]:
        return float("nan")
    return float(np.exp(np.interp(t, w[::-1], x[::-1])))
