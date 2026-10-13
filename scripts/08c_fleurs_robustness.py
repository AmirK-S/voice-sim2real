"""FLEURS follow-up, review fixes B2, I1, I3, I4 (CPU only, existing transcripts).

1. Break-even widening (B2, I1): for every decision, gap and damage cell, the factor by which its
   95 % CI (sentence bootstrap) would have to widen around the point estimate before it includes 0.
2. Proxy speaker clusters (B2): FLEURS has no speaker ids. Each human clip gets a voice-and-channel
   fingerprint: long-term average log-mel spectrum of its quietest frames (microphone and room),
   mean and SD of 19 MFCCs over its speech frames, and median F0. Within each gender, Ward
   agglomerative clustering into k clusters (k = 10, 20, 40 in total, split by gender share). The
   headline CIs are recomputed with a cluster bootstrap over these proxy clusters. The same
   procedure is validated on the LibriSpeech human clips, where the true speakers are known
   (adjusted Rand index, and CI widths with proxy vs true speaker clusters).
   A second grouping, runs of consecutive FLoRes sentence ids (same source article), is also used.
3. LibriSpeech counts on the same bootstrap unit (I3): the v2 per-cell gaps recounted with the
   sentence-bootstrap CIs already stored in results/tables/wer_by_cell.csv.
4. Own-SNR of FLEURS recordings (I4): per clip, 95th minus 10th percentile of 30 ms phone-band
   frame energies (dB), a simple SNR proxy. Channel damage is recomputed on the best-SNR tercile.

Output: results/followups/fleurs/robustness/
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sim2real import channel, stats
from sim2real.paths import RESULTS

spec = importlib.util.spec_from_file_location("fl08", ROOT / "scripts" / "08_fleurs.py")
fl08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl08)

OUT = RESULTS / "followups" / "fleurs" / "robustness"
B = 10000
SYNTH, MEAN = fl08.SYNTH, fl08.MEAN
WH, PA, W2V = fl08.WHISPER, fl08.PARAKEET, fl08.W2V


def break_even(d: float, lo: float, hi: float) -> float:
    """Widening factor (around the point) at which the CI first includes 0; nan if it already does."""
    if lo > 0:
        return d / (d - lo)
    if hi < 0:
        return d / (d - hi)
    return np.nan


# ----------------------------------------------------------------------------- fingerprints

def fingerprint(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    import librosa

    x = np.asarray(x, np.float32)
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=512, hop_length=160, n_mels=40, fmin=50, fmax=7600)
    logS = np.log(S + 1e-10)
    e = logS.mean(0)
    quiet = e <= np.quantile(e, 0.15)
    loud = e >= np.quantile(e, 0.40)
    ltas = logS[:, quiet].mean(1)
    ltas = ltas - ltas.mean()
    mf = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=20)[1:]
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=160)
    n = min(len(f0), len(loud))
    f0v = f0[:n][loud[:n]]
    return np.concatenate([ltas, mf[:, loud].mean(1), mf[:, loud].std(1), [np.log(np.median(f0v))]])


def proxy_clusters(F: np.ndarray, gender: np.ndarray, k: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(F)
    # three blocks (room spectrum, MFCC mean+SD, F0) get equal total weight
    blocks = [slice(0, 40), slice(40, 78), slice(78, 79)]
    for b in blocks:
        Z[:, b] /= np.sqrt(Z[:, b].shape[1])
    lab = np.empty(len(F), object)
    gs = sorted(set(gender))
    for g in gs:
        idx = np.flatnonzero(gender == g)
        kg = max(1, int(round(k * len(idx) / len(F))))
        kg = min(kg, len(idx))
        cl = AgglomerativeClustering(n_clusters=kg, linkage="ward").fit_predict(Z[idx])
        lab[idx] = [f"{g}{c}" for c in cl]
    return lab


# ----------------------------------------------------------------------------- data

def fleurs_E():
    u = pd.read_csv(fl08.OUT / "utterances.csv.gz", keep_default_na=False)
    ids = sorted(u["id"].unique())
    first = u.drop_duplicates("id").set_index("id").loc[ids]
    n = first["n_ref"].to_numpy(float)
    piv = u.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors", aggfunc="first").reindex(ids)
    E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
    for c, m in {(c, m) for (_, c, m) in E}:
        if all((v, c, m) in E for v in SYNTH):
            E[(MEAN, c, m)] = np.mean([E[(v, c, m)] for v in SYNTH], axis=0)
    return ids, first, n, E


def headline(E: dict, n: np.ndarray, W: np.ndarray) -> dict:
    """The FLEURS headline quantities under bootstrap weights W."""
    out = {}
    for c in ("clean", "realistic"):
        r = fl08._paired(E[("human", c, WH)], E[("human", c, PA)], n, W)
        out[f"human W-P {c}"] = (r["diff"], r["lo"], r["hi"])
    hg = E[("human", "realistic", WH)] - E[("human", "realistic", PA)]
    sg = E[(MEAN, "realistic", WH)] - E[(MEAN, "realistic", PA)]
    r = fl08._paired(sg, hg, n, W)
    out["mean-voice DiD realistic"] = (r["diff"], r["lo"], r["hi"])
    r = fl08._paired(E[(MEAN, "realistic", WH)], E[(MEAN, "realistic", PA)], n, W)
    out["mean-voice W-P realistic"] = (r["diff"], r["lo"], r["hi"])
    # gaps: non-MMS cells significantly easier
    k_easy = k_sig = k_cells = 0
    for (v, c, m) in E:
        if v in SYNTH and v != "mms:eng" and ("human", c, m) in E:
            r = fl08._paired(E[(v, c, m)], E[("human", c, m)], n, W)
            k_cells += 1
            k_sig += (r["lo"] > 0) or (r["hi"] < 0)
            k_easy += r["hi"] < 0
    out["non-MMS gap cells easier"] = (k_easy, k_cells, k_sig)
    # channel damage: cells whose CI excludes 0 (no BH here: CI-based count), and understate
    k_u = k_o = k_d = 0
    for (v, c, m) in E:
        if v in SYNTH and c != "clean" and all(k in E for k in [("human", c, m), (v, "clean", m), ("human", "clean", m)]):
            ds = E[(v, c, m)] - E[(v, "clean", m)]
            dh = E[("human", c, m)] - E[("human", "clean", m)]
            r = fl08._paired(ds, dh, n, W)
            k_d += 1
            k_u += r["hi"] < 0
            k_o += r["lo"] > 0
    out["damage cells understate / overstate"] = (k_u, k_o, k_d)
    return out


def main() -> None:
    import soundfile as sf

    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ---- 1. break-even widening for every FLEURS decision, gap and damage cell
    dc = pd.read_csv(fl08.OUT / "decisions.csv")
    rows = []
    for r in dc.itertuples():
        for q in ("human_gap", "synth_gap", "did"):
            d, lo, hi = getattr(r, q), getattr(r, f"{q}_lo"), getattr(r, f"{q}_hi")
            rows.append({"table": "decisions", "model_a": r.model_a, "model_b": r.model_b, "condition": r.condition,
                         "voice": r.voice, "quantity": q, "estimate": d, "lo": lo, "hi": hi,
                         "break_even_widening": break_even(d, lo, hi)})
    gp = pd.read_csv(fl08.OUT / "gaps.csv")
    for r in gp.itertuples():
        rows.append({"table": "gaps", "model_a": r.model, "model_b": "", "condition": r.condition, "voice": r.voice,
                     "quantity": "gap", "estimate": r.gap, "lo": r.gap_lo, "hi": r.gap_hi,
                     "break_even_widening": break_even(r.gap, r.gap_lo, r.gap_hi)})
    dm = pd.read_csv(fl08.OUT / "channel_damage.csv")
    for r in dm.itertuples():
        rows.append({"table": "channel_damage", "model_a": r.model, "model_b": "", "condition": r.condition,
                     "voice": r.voice, "quantity": "damage_error", "estimate": r.damage_error, "lo": r.lo, "hi": r.hi,
                     "break_even_widening": break_even(r.damage_error, r.lo, r.hi)})
    be = pd.DataFrame(rows)
    be.to_csv(OUT / "break_even_widening.csv", index=False, float_format="%.5f")
    sel = lambda t, q: be[(be["table"] == t) & (be["quantity"] == q)]
    g = sel("gaps", "gap")
    gnm = g[g["voice"].isin([v for v in SYNTH if v != "mms:eng"])]
    d = sel("channel_damage", "damage_error")
    d = d[d["voice"].isin(SYNTH)]
    summary["break_even"] = {
        "human W-P realistic": float(be[(be["table"] == "decisions") & (be["quantity"] == "human_gap") &
                                        (be["model_a"] == WH) & (be["model_b"] == PA) &
                                        (be["condition"] == "realistic")]["break_even_widening"].iloc[0]),
        "mean-voice DiD realistic": float(be[(be["table"] == "decisions") & (be["quantity"] == "did") &
                                             (be["model_a"] == WH) & (be["model_b"] == PA) &
                                             (be["condition"] == "realistic") & (be["voice"] == MEAN)]["break_even_widening"].iloc[0]),
        "non-MMS gaps: min": float(gnm["break_even_widening"].min()),
        "non-MMS gaps significant at 2x": int((gnm["break_even_widening"] >= 2).sum()), "non-MMS gaps": len(gnm),
        "damage cells significant (CI) at 1x": int(d["break_even_widening"].notna().sum()),
        "damage cells significant at 1.5x": int((d["break_even_widening"] >= 1.5).sum()),
        "damage cells significant at 2x": int((d["break_even_widening"] >= 2).sum()), "damage cells": len(d)}

    # ---- 2. proxy speaker clusters, validated on LibriSpeech
    ids, first, n, E = fleurs_E()
    fl_man = pd.read_csv(fl08.MANIFEST, keep_default_na=False)
    fl_man = fl_man[fl_man["voice"] == "human"].set_index("id").loc[ids]
    F = np.stack([fingerprint(sf.read(ROOT / p, dtype="float32")[0]) for p in fl_man["path"]])
    gender = fl_man["gender"].to_numpy()
    sub = pd.read_csv(fl08.SUBSET, dtype={"sentence_id": str}).set_index("id").loc[ids]
    sid = sub["sentence_id"].astype(int).to_numpy()
    order = np.argsort(sid)
    run = np.zeros(len(sid), int)
    r_id = 0
    for a, b in zip(order[:-1], order[1:]):
        run[a] = r_id
        if sid[b] - sid[a] > 1:
            r_id += 1
    run[order[-1]] = r_id
    groupings = {"sentence (as reported)": np.arange(len(ids)).astype(str),
                 "consecutive sentence ids": run.astype(str)}
    for k in (10, 20, 40):
        groupings[f"proxy speakers k={k}"] = proxy_clusters(F, gender, k)
    hrows = []
    for gname, cl in groupings.items():
        W = stats.cluster_weights(cl, B, seed=0)
        h = headline(E, n, W)
        hrows.append({"grouping": gname, "n_clusters": len(set(cl)), **{k: json.dumps([round(float(x), 5) for x in v]) for k, v in h.items()}})
    ht = pd.DataFrame(hrows)
    ht.to_csv(OUT / "headline_by_grouping.csv", index=False)
    summary["headline_by_grouping"] = hrows

    # validation on LibriSpeech: true speakers vs proxy clusters vs sentences
    from sklearn.metrics import adjusted_rand_score

    utt = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, low_memory=False)
    ls = utt[(utt["set"] == "librispeech")]
    man = pd.read_csv(ROOT / "data" / "corpus" / "manifest.csv", keep_default_na=False)
    lsh = man[(man["set"] == "librispeech") & (man["voice"] == "human")].sort_values("id")
    spk_file = ROOT / "data" / "librispeech" / "LibriSpeech" / "SPEAKERS.TXT"
    sexmap = {}
    for line_ in spk_file.read_text().splitlines():
        if line_.startswith(";") or "|" not in line_:
            continue
        parts = [x.strip() for x in line_.split("|")]
        sexmap[parts[0]] = parts[1]
    Fl = np.stack([fingerprint(sf.read(ROOT / p, dtype="float32")[0]) for p in lsh["path"]])
    true = lsh["speaker"].astype(str).to_numpy()
    lgender = np.array([sexmap.get(s, "U") for s in true])
    val = []
    lsids = lsh["id"].tolist()
    ls_piv = ls.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors", aggfunc="first").reindex(lsids)
    nl = ls[(ls["voice"] == "human")].drop_duplicates("id").set_index("id").loc[lsids, "n_words" if "n_words" in ls else "n_ref"].to_numpy(float)
    def ls_e(v, c, m):
        return ls_piv[(v, c, m)].to_numpy(float)
    cells = [(v, c, m) for (v, c, m) in ls_piv.columns if v in SYNTH and m in (WH, PA) and c in ("clean", "realistic")]
    for k in (10, 20, 40, 80):
        pc = proxy_clusters(Fl, lgender, k)
        widths = {"true speakers": [], "proxy": [], "sentence": []}
        Wt = stats.cluster_weights(true, 4000, seed=0)
        Wp = stats.cluster_weights(pc, 4000, seed=0)
        Ws = stats.cluster_weights(np.arange(len(true)), 4000, seed=0)
        for cell in cells:
            for name, Wx in (("true speakers", Wt), ("proxy", Wp), ("sentence", Ws)):
                r = fl08._paired(ls_e(*cell), ls_e("human", cell[1], cell[2]), nl, Wx)
                widths[name].append(r["hi"] - r["lo"])
        wt, wp, ws = (np.array(widths[x]) for x in ("true speakers", "proxy", "sentence"))
        val.append({"k": k, "ari_vs_true_speakers": adjusted_rand_score(true, pc),
                    "median_width_proxy_over_true": float(np.median(wp / wt)),
                    "median_width_sentence_over_true": float(np.median(ws / wt)),
                    "q10_proxy_over_true": float(np.quantile(wp / wt, 0.1)),
                    "q90_proxy_over_true": float(np.quantile(wp / wt, 0.9)), "n_cells": len(cells)})
    vt = pd.DataFrame(val)
    vt.to_csv(OUT / "proxy_cluster_validation_librispeech.csv", index=False, float_format="%.4f")
    summary["proxy_validation_librispeech"] = val

    # ---- 3. LibriSpeech counts with the sentence bootstrap (same unit as FLEURS)
    lsc = pd.read_csv(RESULTS / "tables" / "wer_by_cell.csv")
    gfl = pd.read_csv(fl08.OUT / "gaps.csv")
    keys = set(zip(gfl[gfl["voice"].isin(SYNTH)]["model"], gfl[gfl["voice"].isin(SYNTH)]["condition"],
                   gfl[gfl["voice"].isin(SYNTH)]["voice"]))
    lsc = lsc[(lsc["set"] == "librispeech").to_numpy() & np.array([(m, c, v) in keys for m, c, v in zip(lsc["model"], lsc["condition"], lsc["voice"])])]
    cnt = lambda lo, hi: {"harder": int((lo > 0).sum()), "easier": int((hi < 0).sum()), "ns": int(((lo <= 0) & (hi >= 0)).sum())}
    summary["librispeech_same_70_cells"] = {"speaker_cluster_ci": cnt(lsc["gap_lo"], lsc["gap_hi"]),
                                            "sentence_ci": cnt(lsc["gap_lo_sentence_boot"], lsc["gap_hi_sentence_boot"]),
                                            "n": len(lsc)}

    # ---- 4. own-SNR of the FLEURS recordings, damage on the best-SNR tercile
    snr = []
    for p in fl_man["path"]:
        x = sf.read(ROOT / p, dtype="float32")[0]
        y = channel._bandlimit_16k(np.asarray(x, np.float64))
        f = 480
        m = len(y) // f
        e = 10 * np.log10(np.mean(y[: m * f].reshape(m, f) ** 2, axis=1) + 1e-20)
        snr.append(float(np.percentile(e, 95) - np.percentile(e, 10)))
    snr = np.array(snr)
    pd.DataFrame({"id": ids, "gender": gender, "snr_proxy_db": snr}).to_csv(OUT / "fleurs_own_snr.csv", index=False)
    terc = np.digitize(snr, np.quantile(snr, [1 / 3, 2 / 3]))  # 0 worst, 2 best
    drows = []
    for t_ in (0, 1, 2):
        s = terc == t_
        W = stats.cluster_weights(np.arange(int(s.sum())), B, seed=0)
        for (v, c, m) in sorted(E):
            if v == "human" or c == "clean" or not all(k in E for k in [("human", c, m), (v, "clean", m), ("human", "clean", m)]):
                continue
            ds = (E[(v, c, m)] - E[(v, "clean", m)])[s]
            dh = (E[("human", c, m)] - E[("human", "clean", m)])[s]
            r = fl08._paired(ds, dh, n[s], W)
            drows.append({"snr_tercile": ["worst", "middle", "best"][t_], "snr_lo_db": float(snr[s].min()),
                          "snr_hi_db": float(snr[s].max()), "n_sentences": int(s.sum()), "model": m, "condition": c,
                          "voice": v, "human_damage": dh.sum() / n[s].sum(), "synth_damage": ds.sum() / n[s].sum(),
                          "damage_error": r["diff"], "lo": r["lo"], "hi": r["hi"], "p": r["p"]})
    dd = pd.DataFrame(drows)
    for t_, gg in dd[dd["voice"].isin(SYNTH)].groupby("snr_tercile"):
        dd.loc[gg.index, "p_bh"] = stats.bh(gg["p"])
    dd.to_csv(OUT / "channel_damage_by_own_snr.csv", index=False, float_format="%.5f")
    ss = {}
    for t_, gg in dd[dd["voice"].isin(SYNTH)].groupby("snr_tercile"):
        ss[t_] = {"cells": len(gg), "understate_bh": int(((gg["p_bh"] < 0.05) & (gg["damage_error"] < 0)).sum()),
                  "overstate_bh": int(((gg["p_bh"] < 0.05) & (gg["damage_error"] > 0)).sum()),
                  "snr_range_db": [float(gg["snr_lo_db"].iloc[0]), float(gg["snr_hi_db"].iloc[0])]}
        for key in [(PA, "realistic", "kokoro:af_heart"), (PA, "realistic", MEAN), (WH, "realistic", "kokoro:af_heart")]:
            q = dd[(dd["snr_tercile"] == t_) & (dd["model"] == key[0]) & (dd["condition"] == key[1]) & (dd["voice"] == key[2])].iloc[0]
            ss[t_][" | ".join(key)] = {"human_damage": q["human_damage"], "synth_damage": q["synth_damage"],
                                       "error": q["damage_error"], "lo": q["lo"], "hi": q["hi"]}
    summary["damage_by_own_snr"] = ss
    summary["own_snr_by_gender_median"] = pd.Series(snr).groupby(gender).median().to_dict()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(json.dumps(summary["break_even"], indent=1))
    print(ht.to_string())
    print(vt.to_string())
    print(json.dumps(summary["librispeech_same_70_cells"]))
    print(json.dumps(ss, indent=1, default=float))


if __name__ == "__main__":
    main()
