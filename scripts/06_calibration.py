"""Follow-up: calibrating synthetic voices with a few human recordings (CPU only, existing results).

Reads results/utterances.csv (LibriSpeech set) and writes results/followups/calibration/:
- fidelity.csv            Q1 per voice x condition x model: Pearson r (speaker-cluster CI), Spearman,
                          calibration slope of human on synthetic (CI), intercept, gap
- fidelity_diff.csv       Q1 per voice x condition x model pair: correlation of per-sentence model
                          differences (CI), sign agreement on sentences where humans differ (CI),
                          per-speaker correlation of differences
- weighting_reps.csv.gz   Q2 every simulated split (estimate, CI, decisions) per method
- weighting.csv           Q2 summary: decision accuracy, winner rate, coverage, bias, RMSE, width
- active.csv              Q3 coverage, width, bias per target x design x n
- active_savings.csv      Q3 recordings saved at equal median CI width and at equal RMSE, vs random PPI and human-only
- fig_*.png, summary.json

Q2 (Whisper vs Parakeet, the 4 conditions Whisper ran): each replicate splits the 40 speakers into
16 calibration and 24 evaluation speakers, draws n = 10, 25 or 50 calibration sentences from the
calibration speakers and evaluates on all sentences of the evaluation speakers. The target is the
human corpus-WER difference on the evaluation sentences. Q3: finite population of the 250 sentences,
budgets n = 10 to 100, random vs uncertainty-driven selection (stratified Neyman and Poisson IPW).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from sim2real import calibrate as cal
from sim2real import stats
from sim2real.paths import RESULTS

OUT = RESULTS / "followups" / "calibration"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
B1 = 2000           # Q1 bootstrap replicates
R2 = 1000           # Q2 simulated splits per n
BE, BC = 1000, 500  # Q2 bootstrap replicates (evaluation-only, joint for linear)
R3 = 2000           # Q3 simulated samples per design and n
WH, PA, W2V = "whisper-large-v3-turbo", "parakeet-tdt-0.6b-v2", "wav2vec2-base-960h"
MODELS = [WH, PA, W2V]
C4 = ["clean", "phone", "phone+noise10", "realistic"]
CONDS = ["clean", "phone", "g722-wb", "opus-nb", "phone+noise10", "phone+noise5",
         "phone+muffle", "phone+loss5", "opus-nb+loss5", "realistic"]
SHORT = {"kokoro:af_heart": "Kokoro af_heart", "kokoro:am_michael": "Kokoro am_michael",
         "piper:en_US-lessac-medium": "Piper lessac", "piper:en_US-ryan-medium": "Piper ryan",
         "mms:eng": "Meta MMS", "say:Samantha": "macOS Samantha", "say:Daniel": "macOS Daniel",
         "synthetic_mean": "mean of 7 voices"}
MSHORT = {WH: "Whisper", PA: "Parakeet", W2V: "wav2vec2 (control)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})


# --------------------------------------------------------------------------- Q1

def q1(D: cal.LibriData) -> tuple[pd.DataFrame, pd.DataFrame]:
    nbar = D.n.mean()
    W = stats.cluster_weights(D.speaker, B1, seed=SEED)
    rows, drows = [], []
    for m in MODELS:
        for c in CONDS:
            if not D.has(("human", c, m)):
                continue
            y = D.E[("human", c, m)] / nbar
            voices = [v for v in cal.SYNTH + cal.RATE if D.has((v, c, m))]
            if all(D.has((v, c, m)) for v in cal.SYNTH):
                voices.append("synthetic_mean")
            for v in voices:
                f = (cal.synth_mean(D, c, m) if v == "synthetic_mean" else D.E[(v, c, m)]) / nbar
                r = cal.fidelity_cell(f, y, W)
                rows.append({"model": m, "condition": c, "voice": v, "human_wer": y.mean(),
                             "synth_wer": f.mean(), "gap": f.mean() - y.mean(), **r})
    for a, b in [(WH, PA), (WH, W2V), (PA, W2V)]:
        for c in CONDS:
            if not D.has(("human", c, a), ("human", c, b)):
                continue
            dy = D.E[("human", c, a)] - D.E[("human", c, b)]
            voices = [v for v in cal.SYNTH + cal.RATE if D.has((v, c, a), (v, c, b))]
            if all(D.has((v, c, a), (v, c, b)) for v in cal.SYNTH):
                voices.append("synthetic_mean")
            for v in voices:
                if v == "synthetic_mean":
                    df = cal.synth_mean(D, c, a) - cal.synth_mean(D, c, b)
                else:
                    df = D.E[(v, c, a)] - D.E[(v, c, b)]
                drows.append({"model_a": a, "model_b": b, "condition": c, "voice": v,
                              **cal.diff_agreement(df, dy, D.n, D.speaker, W)})
    t, td = pd.DataFrame(rows), pd.DataFrame(drows)
    t.to_csv(OUT / "fidelity.csv", index=False, float_format="%.5f")
    td.to_csv(OUT / "fidelity_diff.csv", index=False, float_format="%.5f")
    return t, td


# --------------------------------------------------------------------------- Q2

def full_reference(D: cal.LibriData, c: str) -> dict:
    W = stats.cluster_weights(D.speaker, 10000, seed=SEED)
    d = D.E[("human", c, WH)] - D.E[("human", c, PA)]
    boot = (W @ d) / (W @ D.n)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(d.sum() / D.n.sum()), "lo": float(lo), "hi": float(hi),
            "decision": str(cal.decide(lo, hi))}


def q2(D: cal.LibriData) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spk_all = np.array(sorted(set(D.speaker)))
    refs = {c: full_reference(D, c) for c in C4}
    rows = []
    rng = np.random.default_rng(SEED + 2)
    splits = []
    for n in (10, 25, 50):
        for rep in range(R2):
            perm = rng.permutation(spk_all)
            cal_spk = set(perm[:16])
            in_cal = np.array([s in cal_spk for s in D.speaker])
            pool, ev = np.flatnonzero(in_cal), np.flatnonzero(~in_cal)
            lab = rng.choice(pool, size=n, replace=False)
            splits.append((n, rep, lab, ev, int(rng.integers(2**31))))
    for c in C4:
        t0 = time.time()
        EA = np.stack([D.E[(v, c, WH)] for v in cal.SYNTH])
        EB = np.stack([D.E[(v, c, PA)] for v in cal.SYNTH])
        hA, hB = D.E[("human", c, WH)], D.E[("human", c, PA)]
        for n, rep, lab, ev, s in splits:
            k = 1.0 / D.n[ev].mean()
            FA, FB, yA, yB = EA * k, EB * k, hA * k, hB * k
            dY, dF = yA - yB, FA - FB
            dFbar = dF.mean(0)
            truth = float(dY[ev].mean())
            We = stats.cluster_weights(D.speaker[ev], BE, seed=s)
            swe = We.sum(1)
            hb = We @ dY[ev] / swe
            ev_ref = str(cal.decide(*np.percentile(hb, [2.5, 97.5])))
            res = {}
            g = dFbar[ev]
            boot = We @ g / swe
            res["plain average"] = (g.mean(), *np.percentile(boot, [2.5, 97.5]))
            w = cal.fidelity_weights(dF[:, lab], dY[lab])
            g = w @ dF[:, ev]
            boot = We @ g / swe
            res["fidelity-weighted"] = (g.mean(), *np.percentile(boot, [2.5, 97.5]))
            Fc, yc, Fe = {WH: FA[:, lab], PA: FB[:, lab]}, {WH: yA[lab], PA: yB[lab]}, {WH: FA[:, ev], PA: FB[:, ev]}
            est = float(cal.linear_corrected(Fc, yc, Fe)[0])
            Wc = stats.cluster_weights(D.speaker[lab], BC, seed=s + 1)
            boot = cal.linear_corrected(Fc, yc, Fe, Wc, We[:BC])
            res["linear-corrected"] = (est, *np.percentile(boot, [2.5, 97.5]))
            fm = float(dFbar[ev].mean())
            for name, lam in (("human only", 0.0), ("PPI", 1.0),
                              ("PPI++", cal.ppipp_lambda(dY[lab], dFbar[lab]))):
                r = cal.two_sample_rectified(dY[lab], dFbar[lab], fm, ev.size, lam=lam)
                res[name] = (r["est"], r["lo"], r["hi"])
            for meth, (e, lo, hi) in res.items():
                rows.append({"condition": c, "n_cal": n, "rep": rep, "method": meth, "estimate": e,
                             "lo": lo, "hi": hi, "truth_eval": truth, "decision": str(cal.decide(lo, hi)),
                             "ref_full": refs[c]["decision"], "ref_eval": ev_ref, "n_eval": ev.size})
        print(f"Q2 {c}: {time.time() - t0:.0f} s", flush=True)
    reps = pd.DataFrame(rows)
    reps.to_csv(OUT / "weighting_reps.csv.gz", index=False, float_format="%.6f")
    reps["covered"] = (reps["lo"] <= reps["truth_eval"]) & (reps["truth_eval"] <= reps["hi"])
    reps["err"] = reps["estimate"] - reps["truth_eval"]
    reps["width"] = reps["hi"] - reps["lo"]
    reps["acc_full"] = reps["decision"] == reps["ref_full"]
    reps["acc_eval"] = reps["decision"] == reps["ref_eval"]
    reps["declares_winner"] = reps["decision"] != "none"
    reps["wrong_winner"] = reps["declares_winner"] & (reps["decision"] != reps["ref_full"])
    summ = reps.groupby(["condition", "n_cal", "method"]).agg(
        ref_full=("ref_full", "first"), decision_acc_vs_full=("acc_full", "mean"),
        decision_acc_vs_eval=("acc_eval", "mean"), share_declares_winner=("declares_winner", "mean"),
        share_wrong_or_false_winner=("wrong_winner", "mean"), coverage=("covered", "mean"),
        bias=("err", "mean"), rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
        median_width=("width", "median"), mean_truth=("truth_eval", "mean"),
        share_eval_ref_none=("ref_eval", lambda s: float((s == "none").mean()))).reset_index()
    order = {m: i for i, m in enumerate(METHODS)}
    summ = summ.sort_values(["condition", "n_cal", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    summ.to_csv(OUT / "weighting.csv", index=False, float_format="%.5f")
    return reps, summ, refs


METHODS = ["plain average", "fidelity-weighted", "linear-corrected", "human only", "PPI", "PPI++"]


# --------------------------------------------------------------------------- Q3

N_GRID = [10, 15, 20, 25, 30, 40, 50, 75, 100]


def q3(D: cal.LibriData) -> tuple[pd.DataFrame, pd.DataFrame]:
    nbar = D.n.mean()
    targets = []
    for c in C4:
        for m in MODELS:
            F = np.stack([D.E[(v, c, m)] for v in cal.SYNTH]) / nbar
            targets.append((f"WER {MSHORT[m]}", c, D.E[("human", c, m)] / nbar, F))
        F = np.stack([D.E[(v, c, WH)] - D.E[(v, c, PA)] for v in cal.SYNTH]) / nbar
        targets.append(("Whisper minus Parakeet", c, (D.E[("human", c, WH)] - D.E[("human", c, PA)]) / nbar, F))
    rows = []
    for ti, (tname, c, y, F) in enumerate(targets):
        f = F.mean(0)
        truth = float(y.mean())
        scores = {"disagreement": F.std(0), "magnitude": np.sqrt((F ** 2).mean(0))}
        for n in N_GRID:
            rng = np.random.default_rng(SEED + 1000 * ti + n)
            res = {"random, human only": cal.srs_ppi(y, f, n, rng, R3, use_f=False),
                   "random, PPI": cal.srs_ppi(y, f, n, rng, R3)}
            for sname, u in scores.items():
                strata = cal.make_strata(u, 5)
                nh = cal.neyman_allocation(strata, u, n)
                res[f"stratified {sname}, PPI"] = cal.stratified_ppi(y, f, strata, nh, rng, R3)
                res[f"Poisson IPW {sname}, PPI"] = cal.poisson_ipw_ppi(y, f, cal.poisson_probs(u, n), rng, R3)
            for d, r in res.items():
                cov = (r["lo"] <= truth) & (truth <= r["hi"])
                rows.append({"target": tname, "condition": c, "design": d, "n": n, "truth": truth,
                             "coverage": float(cov.mean()), "median_width": float(np.median(r["width"])),
                             "mean_width": float(np.mean(r["width"])),
                             "bias": float(np.mean(r["est"] - truth)),
                             "rmse": float(np.sqrt(np.mean((r["est"] - truth) ** 2))),
                             "mean_n_drawn": float(np.mean(r.get("n_drawn", n)))})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "active.csv", index=False, float_format="%.6f")
    sv = []
    for (tname, c), g in t.groupby(["target", "condition"]):
        ref = g[g["design"] == "random, PPI"].sort_values("n")
        refh = g[g["design"] == "random, human only"].sort_values("n")
        for d in g["design"].unique():
            for n in (25, 50):
                row = g[(g["design"] == d) & (g["n"] == n)].iloc[0]
                n_eq = cal.equal_width_n(ref["n"], ref["median_width"], row["median_width"])
                n_eqh = cal.equal_width_n(refh["n"], refh["median_width"], row["median_width"])
                # true precision: RMSE of the estimate over the simulated samples (independent of
                # whether the design's own variance estimate is right)
                n_eq_r = cal.equal_width_n(ref["n"], ref["rmse"], row["rmse"])
                n_eqh_r = cal.equal_width_n(refh["n"], refh["rmse"], row["rmse"])
                sv.append({"target": tname, "condition": c, "design": d, "n": n,
                           "coverage": row["coverage"], "median_width": row["median_width"],
                           "width_vs_random_ppi": row["median_width"] / ref[ref["n"] == n]["median_width"].iloc[0],
                           "rmse_vs_random_ppi": row["rmse"] / ref[ref["n"] == n]["rmse"].iloc[0],
                           "n_random_ppi_same_width": n_eq, "saved_vs_random_ppi_width": n_eq - n,
                           "n_random_ppi_same_rmse": n_eq_r, "saved_vs_random_ppi_rmse": n_eq_r - n,
                           "n_random_human_same_width": n_eqh, "saved_vs_random_human_width": n_eqh - n,
                           "n_random_human_same_rmse": n_eqh_r, "saved_vs_random_human_rmse": n_eqh_r - n})
    s = pd.DataFrame(sv)
    s.to_csv(OUT / "active_savings.csv", index=False, float_format="%.4f")
    return t, s


# --------------------------------------------------------------------------- figures

def fig_fidelity(t: pd.DataFrame) -> None:
    voices = cal.SYNTH + ["synthetic_mean"]
    cols = [(m, c) for m in MODELS for c in C4]
    M = np.full((len(voices), len(cols)), np.nan)
    for i, v in enumerate(voices):
        for j, (m, c) in enumerate(cols):
            r = t[(t.voice == v) & (t.model == m) & (t.condition == c)]
            if len(r):
                M[i, j] = r["pearson_r"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#123f78"])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [c for _, c in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(voices)), [SHORT[v] for v in voices])
    for k, m in enumerate(MODELS):
        ax.text(k * 4 + 1.5, -0.9, MSHORT[m], ha="center", fontsize=8, color=INK)
        if k:
            ax.axvline(k * 4 - 0.5, color="white", lw=2)
    ax.axhline(len(voices) - 1.5, color="white", lw=2)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r, synthetic vs human per sentence")
    ax.set_title("Per-sentence fidelity: how well each voice's errors track human errors", loc="left",
                 pad=18, fontsize=10)
    fig.savefig(OUT / "fig_fidelity.png")
    plt.close(fig)


MARK = {"plain average": "o", "fidelity-weighted": "s", "linear-corrected": "D", "human only": "^",
        "PPI": "v", "PPI++": "P"}


def fig_weighting(summ: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharey="row", sharex=True)
    for j, c in enumerate(C4):
        g = summ[summ.condition == c]
        for k, meth in enumerate(METHODS):
            h = g[g.method == meth].sort_values("n_cal")
            for i, col in enumerate(["decision_acc_vs_full", "coverage"]):
                axes[i, j].plot(h["n_cal"], h[col], marker=MARK[meth], color=CAT[k], lw=1.5, ms=5, label=meth)
        axes[0, j].set_title(f"{c}\n(human: {'Parakeet better' if g['ref_full'].iloc[0] == 'B' else 'no winner'})",
                             fontsize=9)
        axes[1, j].axhline(0.95, color=INK2, lw=0.8, ls="--")
        axes[1, j].set_xticks([10, 25, 50])
    fig.supxlabel("human-labeled calibration sentences (n)", fontsize=9)
    axes[0, 0].set_ylabel("decision matches human\n(full 250 sentences)")
    axes[1, 0].set_ylabel("CI covers human difference\n(held-out speakers)")
    axes[0, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(ncol=6, loc="lower left", bbox_to_anchor=(0, 1.25), fontsize=8)
    fig.suptitle("Whisper vs Parakeet: correcting the 7-voice lab with a few human recordings",
                 x=0.01, ha="left", y=1.07, fontsize=10)
    fig.savefig(OUT / "fig_weighting.png")
    plt.close(fig)


def fig_active(t: pd.DataFrame) -> None:
    designs = ["random, human only", "random, PPI", "stratified magnitude, PPI", "Poisson IPW magnitude, PPI"]
    panels = [("WER Whisper", "realistic"), ("WER Parakeet", "realistic"),
              ("WER wav2vec2 (control)", "realistic"), ("Whisper minus Parakeet", "realistic")]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4), sharex=True)
    for j, (tg, c) in enumerate(panels):
        g = t[(t.target == tg) & (t.condition == c)]
        for k, d in enumerate(designs):
            h = g[g.design == d].sort_values("n")
            for i, col in enumerate(["median_width", "rmse"]):
                axes[i, j].plot(h["n"], 100 * h[col], marker="o", ms=3.5, lw=1.5, color=CAT[k], label=d)
        for i in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].set_yscale("log")
            axes[i, j].set_xticks([10, 25, 50, 100], ["10", "25", "50", "100"])
            axes[i, j].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axes[i, j].yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
            axes[i, j].tick_params(axis="y", which="minor", labelsize=7)
        axes[0, j].set_title(tg.replace(" (control)", "\n(control)").replace(" minus ", " minus\n") + f", {c}", fontsize=9)
    fig.supxlabel("human recordings (n)", fontsize=9)
    axes[0, 0].set_ylabel("reported: median 95 % CI\nwidth (points)")
    axes[1, 0].set_ylabel("actual: RMSE of the\nestimate (points)")
    axes[0, 0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, 1.2), fontsize=8)
    fig.suptitle("Which sentences to record: active designs shrink the reported CI more than the actual error",
                 x=0.01, ha="left", y=1.06, fontsize=10)
    fig.savefig(OUT / "fig_active.png")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def main() -> None:
    D = cal.load_librispeech(RESULTS / "utterances.csv")
    t0 = time.time()
    f1, f1d = q1(D)
    print(f"Q1 done {time.time() - t0:.0f} s", flush=True)
    _, s2, refs = q2(D)
    print(f"Q2 done {time.time() - t0:.0f} s", flush=True)
    a3, s3 = q3(D)
    print(f"Q3 done {time.time() - t0:.0f} s", flush=True)
    fig_fidelity(f1)
    fig_weighting(s2)
    fig_active(a3)
    summary = {"q2_reference_full_250": refs, "settings": {
        "q1_bootstrap": B1, "q2_splits_per_n": R2, "q2_boot_eval": BE, "q2_boot_joint_linear": BC,
        "q3_samples": R3, "q2_cal_speakers": 16, "q2_eval_speakers": 24, "seed": SEED}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
