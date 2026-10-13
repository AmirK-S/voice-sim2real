"""Follow-up: main-run speech re-transcribed with Whisper's standard pipeline.

Why
  Every main-run Whisper WER went through asr._whisper_batch, a raw decoder whose 30 s window is
  padded with the log-mel of digital silence, not the way mlx_whisper.transcribe pads clips under
  30 s. On noise-only clips that padding changes the phantom rate from 34 to 94 of 100
  (results/followups/phantom/v2/NOTES.md). Its effect on speech WER was never measured.

What
  The same clips as the main run (scripts/04_run_experiment.py: channel.prepare, then
  channel.apply with the sentence-id seed, so pairing with every other model is preserved),
  transcribed with mlx_whisper.transcribe at its defaults plus language="en" (the arm
  "transcribe_defaults_en" of scripts/09c_whisper_gate_silero.py, same call). Model id in the
  transcript cache: whisper-large-v3-turbo-std (data/cache/asr/whisper-large-v3-turbo-std/).

Steps
  run      GPU. LibriSpeech human first, then the 7 voices, on clean, phone, phone+noise10,
           realistic; then the emergency caller set (realistic first). Shards in
           results/raw/followups/whisper_std/<set>__<human|synth>__<condition>.csv. Resumable.
  analyze  CPU. Scores the shards exactly as the main merge does and writes
           results/followups/whisper_std/ (CSV, figure, summary.json, transcripts_std.csv.gz).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sim2real import asr, channel, data, stats
from sim2real.paths import RAW, RESULTS


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run04 = _load("run04", "04_run_experiment.py")

STD = "whisper-large-v3-turbo-std"
RAWW = "whisper-large-v3-turbo"
PARAKEET = "parakeet-tdt-0.6b-v2"
W2V = "wav2vec2-base-960h"
CONDS = ["clean", "phone", "phone+noise10", "realistic"]
SYNTH = ["kokoro:af_heart", "kokoro:am_michael", "piper:en_US-lessac-medium",
         "piper:en_US-ryan-medium", "mms:eng", "say:Samantha", "say:Daniel"]
SHARDS = RAW / "followups" / "whisper_std"
OUT = RESULTS / "followups" / "whisper_std"
B, SEED = 10000, 0


# ----------------------------------------------------------------------------- transcription

def transcribe_std(arrays: list[np.ndarray]) -> list[str]:
    """mlx_whisper.transcribe, defaults + language "en" (09c arm transcribe_defaults_en), cached."""
    import mlx_whisper

    out = []
    for a in arrays:
        a = asr._as16k(a)
        key = asr._key(STD, a.tobytes())
        t = asr._cache_get(STD, key)
        if t is None:
            o = mlx_whisper.transcribe(a, path_or_hf_repo=asr.WHISPER_ID, verbose=None, language="en")
            t = o["text"].strip()
            asr._cache_put(STD, key, t)
        out.append(t)
    return out


def run_step() -> None:
    SHARDS.mkdir(parents=True, exist_ok=True)
    m = run04.select(run04.load_manifest(), "main", ["librispeech", "emergency"])
    plan = [("librispeech", ["human"], c) for c in CONDS]
    plan += [("librispeech", SYNTH, c) for c in CONDS]
    plan += [("emergency", SYNTH, c) for c in ["realistic", "clean", "phone", "phone+noise10"]]
    for s, voices, cond in plan:
        tag = "human" if voices == ["human"] else "synth"
        shard = SHARDS / f"{s}__{tag}__{cond}.csv"
        g = m[(m["set"] == s) & m["voice"].isin(voices)]
        if shard.exists() and len(pd.read_csv(shard)) == len(g):
            print(f"[{s} {tag} {cond}] already done", flush=True)
            continue
        t0, rows = time.time(), []
        for voice, gv in g.groupby("voice", sort=True):
            arrays = []
            for p, i in zip(gv["path"], gv["id"]):
                x = channel.prepare(run04.read16k(p), seed=run04.utt_seed(i))
                arrays.append(channel.apply(cond, x, seed=run04.utt_seed(i), meta={}))
            hyps = transcribe_std(arrays)
            rows += [{"set": s, "id": i, "voice": voice, "condition": cond, "hyp": h}
                     for i, h in zip(gv["id"], hyps)]
        tmp = shard.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        tmp.replace(shard)
        print(f"[{s} {tag} {cond}] {len(rows)} clips in {time.time() - t0:.0f} s", flush=True)


# ----------------------------------------------------------------------------- scoring

def score(d: pd.DataFrame) -> pd.DataFrame:
    """Same scoring as scripts/04_run_experiment.py::merge (errors, n_ref, keywords)."""
    import jiwer

    ref_n = {r: asr.normalize(r) for r in d["ref"].unique()}
    errs, nref = [], []
    for r, h in zip(d["ref"], d["hyp"]):
        rn, hn = ref_n[r], asr.normalize(h)
        n = len(rn.split())
        if not hn:
            e = n
        else:
            o = jiwer.process_words(rn, hn)
            e = o.substitutions + o.deletions + o.insertions
        errs.append(e)
        nref.append(n)
    d = d.copy()
    d["errors"], d["n_ref"] = errs, nref
    ref_kws = [data.find_keywords(r) for r in d["ref"]]
    hits = [run04._kw_hits(k, h) for k, h in zip(ref_kws, d["hyp"])]
    d["kw_total"] = [len(k) for k in ref_kws]
    d["kw_hits"] = [len(k) for k in hits]
    d["kw_detected_any"] = [len(data.find_keywords(h)) > 0 for h in d["hyp"]]
    return d


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    u = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, dtype={"speaker": str, "category": str})
    u = u[u["condition"].isin(CONDS) & ~u["voice"].str.endswith("@rate")]
    sh = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(SHARDS.glob("*.csv"))],
                   ignore_index=True)
    base = u[u["model"] == RAWW][["set", "id", "speaker", "category", "voice", "condition", "ref"]]
    s = base.merge(sh, on=["set", "id", "voice", "condition"], how="inner", validate="one_to_one")
    s = score(s)
    s["model"] = STD
    return u, s


# ----------------------------------------------------------------------------- analysis

def analyze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    u, s = load_all()
    s[["set", "id", "voice", "condition", "model", "hyp", "errors", "n_ref", "kw_total", "kw_hits"]].to_csv(
        OUT / "transcripts_std.csv.gz", index=False)
    # check: the main merge's scores are reproduced by score() on the raw-decoder rows
    chk = score(u[u["model"] == RAWW].head(2000)[["set", "id", "voice", "condition", "ref", "hyp"]])
    assert (chk["errors"].to_numpy() == u[u["model"] == RAWW].head(2000)["errors"].to_numpy()).all()

    ls = pd.concat([u[u["set"] == "librispeech"], s[s["set"] == "librispeech"]], ignore_index=True)
    ids = sorted(ls["id"].unique())
    first = ls.drop_duplicates("id").set_index("id").loc[ids]
    n = first["n_ref"].to_numpy(float)
    piv = ls.pivot_table(index="id", columns=["voice", "condition", "model"], values="errors",
                         aggfunc="first").reindex(ids)
    E = {c: piv[c].to_numpy(float) for c in piv.columns if not piv[c].isna().any()}
    W = stats.cluster_weights(first["speaker"].astype(str).to_numpy(), B, seed=SEED)
    Wn = W @ n
    summary: dict = {"n_sentences": len(ids), "B": B,
                     "std_rows": {f"{a}|{b}": int(v) for (a, b), v in s.groupby(["set", "condition"]).size().items()}}

    def wer(k):
        return float(E[k].sum() / n.sum())

    def pdiff(x):  # x = per-sentence error difference
        bt = (W @ x) / Wn
        return float(x.sum() / n.sum()), *np.percentile(bt, [2.5, 97.5]).tolist(), stats.boot_p(bt), bt

    # (1) WER per voice x condition, std vs raw, paired
    rows = []
    for v in ["human"] + SYNTH:
        for c in CONDS:
            if (v, c, STD) not in E:
                continue
            ws, wr = wer((v, c, STD)), wer((v, c, RAWW))
            bs = (W @ E[(v, c, STD)]) / Wn
            br = (W @ E[(v, c, RAWW)]) / Wn
            d, lo, hi, p, _ = pdiff(E[(v, c, STD)] - E[(v, c, RAWW)])
            rows.append({"voice": v, "condition": c, "wer_std": ws,
                         "wer_std_lo": np.percentile(bs, 2.5), "wer_std_hi": np.percentile(bs, 97.5),
                         "wer_raw": wr, "wer_raw_lo": np.percentile(br, 2.5), "wer_raw_hi": np.percentile(br, 97.5),
                         "std_minus_raw": d, "lo": lo, "hi": hi, "p": p,
                         "wer_parakeet": wer((v, c, PARAKEET))})
    t1 = pd.DataFrame(rows)
    t1["p_holm_human"] = np.nan
    hm = t1["voice"] == "human"
    t1.loc[hm, "p_holm_human"] = stats.holm(t1.loc[hm, "p"])
    t1.to_csv(OUT / "wer_std_vs_raw.csv", index=False, float_format="%.5f")

    # sentence-level agreement
    ag = []
    for v in ["human"] + SYNTH:
        for c in CONDS:
            a = s[(s["set"] == "librispeech") & (s["voice"] == v) & (s["condition"] == c)].set_index("id")
            r = u[(u["set"] == "librispeech") & (u["model"] == RAWW) & (u["voice"] == v) & (u["condition"] == c)].set_index("id")
            if a.empty:
                continue
            j = a.join(r[["hyp", "errors"]], rsuffix="_raw")
            same = [asr.normalize(x) == asr.normalize(y) for x, y in zip(j["hyp"], j["hyp_raw"])]
            ag.append({"voice": v, "condition": c, "n": len(j), "identical_normalized": int(sum(same)),
                       "std_more_errors": int((j["errors"] > j["errors_raw"]).sum()),
                       "std_fewer_errors": int((j["errors"] < j["errors_raw"]).sum()),
                       "std_empty": int(sum(not asr.normalize(x) for x in j["hyp"])),
                       "raw_empty": int(sum(not asr.normalize(x) for x in j["hyp_raw"])),
                       "max_extra_errors_one_sentence": int((j["errors"] - j["errors_raw"]).max())})
    pd.DataFrame(ag).to_csv(OUT / "sentence_agreement.csv", index=False)

    # (2) Whisper vs Parakeet, std and raw, per voice + human + DiD with Holm over 28 voice decisions
    rows = []
    for wm in (STD, RAWW):
        for c in CONDS:
            if ("human", c, wm) not in E:
                continue
            h = E[("human", c, wm)] - E[("human", c, PARAKEET)]
            hd, hlo, hhi, hp, hb = pdiff(h)
            for v in SYNTH:
                if (v, c, wm) not in E:
                    continue
                x = E[(v, c, wm)] - E[(v, c, PARAKEET)]
                sd, slo, shi, sp, sb = pdiff(x)
                dd, dlo, dhi, dp, _ = pdiff(x - h)
                rows.append({"whisper": wm, "condition": c, "voice": v,
                             "human_gap": hd, "human_lo": hlo, "human_hi": hhi, "human_p": hp,
                             "lab_gap": sd, "lab_lo": slo, "lab_hi": shi,
                             "did": dd, "did_lo": dlo, "did_hi": dhi, "did_p": dp,
                             "confident_flip": bool((hlo > 0 or hhi < 0) and (slo > 0 or shi < 0)
                                                    and np.sign(sd) != np.sign(hd))})
    t2 = pd.DataFrame(rows)
    for wm, g in t2.groupby("whisper"):
        t2.loc[g.index, "did_p_holm"] = stats.holm(g["did_p"]) if len(g) else np.nan
    t2.to_csv(OUT / "whisper_vs_parakeet_by_voice.csv", index=False, float_format="%.5f")

    # 7-voice mean with BOOT-BOTH, TOST/MDE, decisions: reuse 10_literature_checks functions
    lit = _load("lit10", "10_literature_checks.py")
    rng = np.random.default_rng(lit.VSEED)
    VW = {"flat": lit.voice_weights("flat", rng), "nested": lit.voice_weights("nested", rng)}
    parts_mean, parts_tost, parts_dec, parts_mvh = [], [], [], []
    for wm in (STD, RAWW):
        lit.WHISPER = wm
        lit.MODELS = [wm, PARAKEET, W2V]
        mt, boots = lit.mean_rows("librispeech", E, n, W, CONDS, VW)
        mt.insert(0, "whisper", wm)
        parts_mean.append(mt)
        tt = lit.tost_rows("librispeech", E, n, W, CONDS)
        tt.insert(0, "whisper", wm)
        parts_tost.append(tt)
        dec, _ = lit.decision_rows("librispeech", E, n, W, CONDS, boots)
        dec.insert(0, "whisper", wm)
        parts_dec.append(dec)
        mvh = lit.mean_vs_human_rows("librispeech", E, n, W, CONDS, VW)
        mvh.insert(0, "whisper", wm)
        parts_mvh.append(mvh)
    mt = pd.concat(parts_mean, ignore_index=True)
    # Holm over the 4 conditions for the mean gap (Whisper vs Parakeet), per scheme
    mt["mean_gap_p_holm4"] = np.nan
    for (wm, a, b, sc), g in mt.groupby(["whisper", "model_a", "model_b", "scheme"]):
        mt.loc[g.index, "mean_gap_p_holm4"] = stats.holm(g["mean_gap_p"])
    mt.to_csv(OUT / "mean_boot_both.csv", index=False, float_format="%.5f")
    tt = pd.concat(parts_tost, ignore_index=True)
    tt.to_csv(OUT / "human_gap_tost_mde.csv", index=False, float_format="%.5f")
    dec = pd.concat(parts_dec, ignore_index=True)
    dec.to_csv(OUT / "decisions_by_voice.csv", index=False, float_format="%.5f")
    mvh = pd.concat(parts_mvh, ignore_index=True)
    mvh.to_csv(OUT / "mean_vs_human_boot_both.csv", index=False, float_format="%.5f")

    # R2: Whisper vs wav2vec2 ratio lab/real; R3a gaps vs human (clean), R3b Kokoro damage error
    r2 = dec[(dec.model_b == W2V) & (dec.model_a.isin([STD, RAWW])) & ~dec.voice.str.startswith("mean")].copy()
    r2["ratio"] = r2["lab_gap"] / r2["human_gap"]
    r3 = []
    for wm in (STD, RAWW):
        for v in SYNTH:
            for c in CONDS:
                if (v, c, wm) not in E:
                    continue
                g_, glo, ghi, gp, _ = pdiff(E[(v, c, wm)] - E[("human", c, wm)])
                row = {"whisper": wm, "voice": v, "condition": c, "gap_vs_human": g_, "lo": glo, "hi": ghi, "p": gp}
                if c != "clean":
                    ds = E[(v, c, wm)] - E[(v, "clean", wm)]
                    dh = E[("human", c, wm)] - E[("human", "clean", wm)]
                    de, dlo, dhi, dp, _ = pdiff(ds - dh)
                    row.update({"human_damage": float(dh.sum() / n.sum()), "synth_damage": float(ds.sum() / n.sum()),
                                "damage_error": de, "damage_lo": dlo, "damage_hi": dhi, "damage_p": dp})
                r3.append(row)
    r3 = pd.DataFrame(r3)
    r3.to_csv(OUT / "gap_and_damage_vs_human.csv", index=False, float_format="%.5f")

    # keyword recall (emergency set), realistic and others if present
    kw = []
    em_s = s[s["set"] == "emergency"]
    em_u = u[(u["set"] == "emergency") & (u["model"].isin([RAWW, PARAKEET]))]
    em = pd.concat([em_u, em_s], ignore_index=True)
    em = em[em["category"].isin(["emergency", "fall_help"]) & (em["kw_total"] > 0)].copy()
    em["kw_detected_any"] = em["kw_detected_any"].astype(str) == "True"
    for (mdl, c), g in em.groupby(["model", "condition"]):
        if c not in set(em_s["condition"]):
            continue
        kw.append({"model": mdl, "condition": c, "renderings": len(g),
                   "kw_total": int(g["kw_total"].sum()), "kw_hits": int(g["kw_hits"].sum()),
                   "recall": g["kw_hits"].sum() / g["kw_total"].sum(),
                   "call_misses": int((~g["kw_detected_any"]).sum())})
    kwt = pd.DataFrame(kw)
    # sentence-cluster bootstrap CI of recall (by id), as in 05_analyze pooled table
    if not kwt.empty:
        los, his = [], []
        for r in kwt.itertuples():
            g = em[(em["model"] == r.model) & (em["condition"] == r.condition)]
            byid = g.groupby("id")[["kw_hits", "kw_total"]].sum()
            Wk = stats.cluster_weights(byid.index.to_numpy(), B, seed=SEED)
            bt = (Wk @ byid["kw_hits"].to_numpy(float)) / (Wk @ byid["kw_total"].to_numpy(float))
            los.append(np.percentile(bt, 2.5)); his.append(np.percentile(bt, 97.5))
        kwt["lo"], kwt["hi"] = los, his
        kwt.to_csv(OUT / "keyword_recall.csv", index=False, float_format="%.5f")

    # summary
    summary["human_std_vs_raw"] = t1[hm][["condition", "wer_std", "wer_std_lo", "wer_std_hi", "wer_raw",
                                         "std_minus_raw", "lo", "hi", "p", "p_holm_human"]].to_dict("records")
    summary["voice_std_minus_raw_range"] = [float(t1[~hm]["std_minus_raw"].min()), float(t1[~hm]["std_minus_raw"].max())]
    for wm, g in t2.groupby("whisper"):
        summary[f"single_voice_did_{wm}"] = {"nominal_p_lt_05": int((g["did_p"] < 0.05).sum()),
                                             "holm_lt_05": int((g["did_p_holm"] < 0.05).sum()),
                                             "min_holm_p": float(g["did_p_holm"].min()),
                                             "confident_flips": int(g["confident_flip"].sum()), "n": len(g)}
    wp = mt[(mt.model_b == PARAKEET)]
    summary["mean_whisper_vs_parakeet"] = wp[["whisper", "condition", "scheme", "human_gap", "human_lo", "human_hi",
                                              "mean_gap", "mean_gap_lo", "mean_gap_hi", "mean_gap_p", "mean_gap_p_holm4",
                                              "did", "did_lo", "did_hi"]].to_dict("records")
    summary["tost"] = tt.to_dict("records")
    summary["r2_ratio_range"] = {wm: [float(g[g.human_sig]["ratio"].min()), float(g[g.human_sig]["ratio"].max())]
                                 for wm, g in r2.groupby("model_a")}
    summary["r2_realistic_human_whisper_vs_w2v"] = {wm: float(g[(g.condition == "realistic")]["human_gap"].iloc[0])
                                                   for wm, g in r2.groupby("model_a")}
    summary["r2_realistic_lab_range"] = {wm: [float(g[g.condition == "realistic"]["lab_gap"].min()),
                                              float(g[g.condition == "realistic"]["lab_gap"].max())]
                                         for wm, g in r2.groupby("model_a")}
    summary["confident_wrong_sign"] = {wm: int(g["confident_wrong_sign"].sum()) for wm, g in dec.groupby("whisper")}
    if not kwt.empty:
        summary["keyword_recall"] = kwt.to_dict("records")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if set(CONDS) <= set(mt[(mt.whisper == STD) & (mt.model_b == PARAKEET)].condition):
        fig(t1, t2, mt)
    pd.set_option("display.width", 220)
    print(t1.round(4).to_string())
    print(pd.DataFrame(ag).to_string())
    print(t2.round(4).to_string())
    print(wp[["whisper", "condition", "scheme", "human_gap", "human_lo", "human_hi", "mean_gap", "mean_gap_lo",
              "mean_gap_hi", "mean_gap_p_holm4", "did", "did_lo", "did_hi"]].round(4).to_string())
    print(tt.round(4).to_string())
    print(r3[(r3.voice == "kokoro:af_heart") | (r3.condition == "clean")].round(4).to_string())
    print(json.dumps({k: summary[k] for k in summary if k.startswith(("single", "r2", "confident"))}, indent=1, default=float))
    if not kwt.empty:
        print(kwt.round(4).to_string())


def fig(t1: pd.DataFrame, t2: pd.DataFrame, mt: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, GREY, INK2 = "#2a78d6", "#8a8984", "#52514e"
    fig_, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: human WER per condition, std vs raw
    h = t1[t1.voice == "human"].set_index("condition").loc[CONDS]
    x = np.arange(len(CONDS))
    for off, col, lab, key in [(-0.12, GREY, "raw decoder (main run)", "raw"), (0.12, BLUE, "transcribe() defaults", "std")]:
        y = h[f"wer_{key}"] * 100
        ax[0].errorbar(x + off, y, yerr=[y - h[f"wer_{key}_lo"] * 100, h[f"wer_{key}_hi"] * 100 - y],
                       fmt="o", color=col, label=lab, capsize=3)
    ax[0].set_xticks(x, CONDS)
    ax[0].set_ylabel("Whisper WER on human LibriSpeech (%)")
    ax[0].set_title("Human WER: standard vs raw decoder", fontsize=10)
    ax[0].legend(fontsize=8)
    # right: Whisper minus Parakeet, human and 7-voice mean (both_flat), std vs raw
    for off, col, wm, lab in [(-0.15, GREY, RAWW, "raw"), (0.15, BLUE, STD, "std")]:
        g = mt[(mt.whisper == wm) & (mt.model_b == PARAKEET) & (mt.scheme == "both_flat")].set_index("condition").loc[CONDS]
        y = g["human_gap"] * 100
        ax[1].errorbar(x + off - 0.05, y, yerr=[y - g["human_lo"] * 100, g["human_hi"] * 100 - y], fmt="o",
                       color=col, capsize=3, label=f"human ({lab})")
        y2 = g["mean_gap"] * 100
        ax[1].errorbar(x + off + 0.05, y2, yerr=[y2 - g["mean_gap_lo"] * 100, g["mean_gap_hi"] * 100 - y2], fmt="s",
                       color=col, mfc="white", capsize=3, label=f"7-voice mean, voices resampled ({lab})")
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_xticks(x, CONDS)
    ax[1].set_ylabel("WER Whisper - WER Parakeet (points)")
    ax[1].set_title("Whisper vs Parakeet: >0 means Parakeet better", fontsize=10)
    ax[1].legend(fontsize=7)
    fig_.tight_layout()
    fig_.savefig(OUT / "fig_whisper_std.png", dpi=150, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "analyze"])
    a = ap.parse_args()
    run_step() if a.step == "run" else analyze()


if __name__ == "__main__":
    main()
