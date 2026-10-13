"""FLEURS follow-up, review fix B3: does the TTS input text explain the FLEURS vs LibriSpeech contrast?

In the main run the TTS voices read LibriSpeech-style text (sim2real.data.tts_text: sentence case,
no inner punctuation, final period, no digits because LibriSpeech spells numbers out). In the
FLEURS follow-up they read the raw FLEURS text (case, commas, digits, abbreviations). Meta MMS-TTS
also lowercases everything and its vocabulary has only the digits 0 to 6, so "1990" is misread.

This script re-synthesizes the 250 FLEURS sentences with LibriSpeech-style text for the 7 voices:
  FLEURS normalized transcription (lowercase, no punctuation)
  -> digits spelled out with num2words (years as years, "40,000" as a cardinal, "%" as percent)
  -> sim2real.data.tts_text (sentence case, "I", final period).
Whisper and Parakeet then transcribe the clean audio (same preparation and seeds as
scripts/08_fleurs.py). Scoring uses the same references as the FLEURS follow-up.

Steps
  synth            synthesize data/audio/fleurs_lsstyle/<voice>/<id>.wav
  run --model M    clean condition only, one shard per model
  analyze          gaps with raw vs LibriSpeech-style text, on all 250 sentences and on the 212
                   without digits; voice spread on the same subsets
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sim2real import asr, channel, data, fleurs, stats
from sim2real.paths import DATA, RAW, RESULTS

spec = importlib.util.spec_from_file_location("fl08", ROOT / "scripts" / "08_fleurs.py")
fl08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl08)

OUT = RESULTS / "followups" / "fleurs" / "textprep"
SHARDS = RAW / "followups" / "fleurs_textprep"
AUDIO = DATA / "audio" / "fleurs_lsstyle"
SYNTH = fl08.SYNTH
B = 10000


def spell_numbers(t: str) -> str:
    from num2words import num2words

    t = t.replace("%", " percent").replace(":", " ").replace("’", "'").replace("[", "").replace("]", "")
    t = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-rt-z])", " ", t)  # m16 -> m 16, 2.4ghz -> 2.4 ghz

    def rep(m: re.Match) -> str:
        s, plural = m.group(1), m.group(2)
        v = s.replace(",", "")
        if re.fullmatch(r"\d+\.\d+", v):
            return num2words(float(v))
        n = int(v)
        w = num2words(n, to="year") if ("," not in s and 1100 <= n <= 2099 and len(v) == 4) else num2words(n)
        if plural:  # 1920s -> nineteen twenties
            w = w[:-1] + "ies" if w.endswith("y") else w + "s"
        return w

    t = re.sub(r"(\d[\d,]*(?:\.\d+)?)(s\b)?", rep, t)
    t = re.sub(r"[^a-zà-ÿ' ]", " ", t)  # LibriSpeech-style: letters, apostrophes, spaces only
    return re.sub(r"\s+", " ", t).strip()


def ls_style(transcription: str) -> str:
    return data.tts_text(spell_numbers(transcription).upper())


def subset() -> pd.DataFrame:
    s = pd.read_csv(fl08.SUBSET, dtype={"sentence_id": str}, keep_default_na=False)
    idx = fleurs.index()[["file", "transcription"]]
    s = s.merge(idx, on="file", how="left", validate="one_to_one")
    s["tts_ls"] = [ls_style(t) for t in s["transcription"]]
    s["has_digit"] = s["text"].str.contains(r"\d")
    return s


def synth() -> None:
    s = subset()
    OUT.mkdir(parents=True, exist_ok=True)
    s[["id", "text", "tts_text", "tts_ls", "has_digit"]].to_csv(OUT / "tts_text_variants.csv", index=False)
    for voice in SYNTH:
        t0 = time.perf_counter()
        d = AUDIO / voice.replace(":", "_")
        jobs = [(r.tts_ls, d / f"{r.id}.wav") for r in s.itertuples()]
        with ThreadPoolExecutor(6 if voice.startswith("say:") else 1) as ex:
            st = list(ex.map(lambda j: fleurs.synth_one(j[0], voice, j[1]), jobs))
        bad = [x for x in st if x not in ("ok", "skip")]
        print(f"{voice}: {st.count('ok')} new, {st.count('skip')} cached, {len(bad)} failed "
              f"({time.perf_counter() - t0:.0f} s) {bad[:2]}", flush=True)


def run(model: str) -> None:
    model = asr.resolve(model)
    s = subset()
    SHARDS.mkdir(parents=True, exist_ok=True)
    shard = SHARDS / f"{model}__clean.csv"
    if shard.exists():
        print("cached")
        return
    rows, t0 = [], time.time()
    for voice in SYNTH:
        d = AUDIO / voice.replace(":", "_")
        arrays = []
        for i in s["id"]:
            x = channel.prepare(fl08.read16k(str((d / f"{i}.wav").relative_to(ROOT))), seed=fl08.utt_seed(i))
            arrays.append(channel.apply("clean", x, seed=fl08.utt_seed(i), meta={}))
        hyps = asr.transcribe_arrays(arrays, model)
        rows += [{"id": i, "voice": voice, "hyp": h} for i, h in zip(s["id"], hyps)]
        print(f"[{model}] {voice} done ({time.time() - t0:.0f} s)", flush=True)
    pd.DataFrame(rows).to_csv(shard, index=False)


def score(t: pd.DataFrame, refs: dict) -> pd.DataFrame:
    import jiwer

    errs, nref = [], []
    for i, h in zip(t["id"], t["hyp"]):
        rn = asr.normalize(refs[i])
        hn = asr.normalize(fleurs.scoring_ref(h if isinstance(h, str) else ""))
        n = len(rn.split())
        if not hn:
            e = n
        else:
            o = jiwer.process_words(rn, hn)
            e = o.substitutions + o.deletions + o.insertions
        errs.append(e)
        nref.append(n)
    t = t.copy()
    t["errors"], t["n_ref"] = errs, nref
    return t


def analyze() -> None:
    s = subset()
    ids = sorted(s["id"])
    s = s.set_index("id").loc[ids]
    refs = {i: fleurs.scoring_ref(s.loc[i, "text"]) for i in ids}
    u = pd.read_csv(fl08.OUT / "utterances.csv.gz", keep_default_na=False)
    u = u[u["condition"] == "clean"]
    rows = []
    for m in (fl08.WHISPER, fl08.PARAKEET):
        new = score(pd.read_csv(SHARDS / f"{m}__clean.csv", keep_default_na=False), refs)
        new = new.set_index(["voice", "id"])
        old = u[u["model"] == m].set_index(["voice", "id"])
        n = old.loc["human"].loc[ids, "n_ref"].to_numpy(float)
        eh = old.loc["human"].loc[ids, "errors"].to_numpy(float)
        E_raw = {v: old.loc[v].loc[ids, "errors"].to_numpy(float) for v in SYNTH}
        E_ls = {v: new.loc[v].loc[ids, "errors"].to_numpy(float) for v in SYNTH}
        E_raw[fl08.MEAN] = np.mean([E_raw[v] for v in SYNTH], axis=0)
        E_ls[fl08.MEAN] = np.mean([E_ls[v] for v in SYNTH], axis=0)
        for label, sel in (("all 250", np.ones(len(ids), bool)),
                           ("212 without digits", ~s["has_digit"].to_numpy()),
                           ("38 with digits", s["has_digit"].to_numpy())):
            W = stats.cluster_weights(np.arange(int(sel.sum())), B, seed=0)
            for v in SYNTH + [fl08.MEAN]:
                g_raw = fl08._paired(E_raw[v][sel], eh[sel], n[sel], W)
                g_ls = fl08._paired(E_ls[v][sel], eh[sel], n[sel], W)
                ch = fl08._paired(E_ls[v][sel], E_raw[v][sel], n[sel], W)
                rows.append({"model": m, "subset": label, "n_sentences": int(sel.sum()), "voice": v,
                             "human_wer": eh[sel].sum() / n[sel].sum(),
                             "wer_raw_text": E_raw[v][sel].sum() / n[sel].sum(),
                             "wer_ls_text": E_ls[v][sel].sum() / n[sel].sum(),
                             "gap_raw_text": g_raw["diff"], "gap_raw_lo": g_raw["lo"], "gap_raw_hi": g_raw["hi"],
                             "gap_ls_text": g_ls["diff"], "gap_ls_lo": g_ls["lo"], "gap_ls_hi": g_ls["hi"],
                             "change_ls_minus_raw": ch["diff"], "change_lo": ch["lo"], "change_hi": ch["hi"]})
    g = pd.DataFrame(rows)
    g.to_csv(OUT / "gaps_by_text_prep.csv", index=False, float_format="%.5f")

    # voice spread (worst / best) with each text, per subset
    sp = []
    for m in (fl08.WHISPER, fl08.PARAKEET):
        for label in g["subset"].unique():
            d = g[(g["model"] == m) & (g["subset"] == label) & g["voice"].isin(SYNTH)].set_index("voice")
            for col in ("wer_raw_text", "wer_ls_text"):
                w = d[col]
                sp.append({"model": m, "subset": label, "text": col.replace("wer_", ""),
                           "best": w.idxmin(), "best_wer": w.min(), "worst": w.idxmax(), "worst_wer": w.max(),
                           "worst_over_best": w.max() / w.min(),
                           "worst_over_best_excl_mms": w.drop("mms:eng").max() / w.drop("mms:eng").min()})
    pd.DataFrame(sp).to_csv(OUT / "voice_spread_by_text_prep.csv", index=False, float_format="%.4f")
    # bootstrap CI for the spread ratio with LS-style text, all 250 and 212
    summ = {}
    for m in (fl08.WHISPER, fl08.PARAKEET):
        new = score(pd.read_csv(SHARDS / f"{m}__clean.csv", keep_default_na=False), refs).set_index(["voice", "id"])
        old = u[u["model"] == m].set_index(["voice", "id"])
        n = old.loc["human"].loc[ids, "n_ref"].to_numpy(float)
        for label, sel in (("all 250", np.ones(len(ids), bool)), ("212 without digits", ~s["has_digit"].to_numpy())):
            W = stats.cluster_weights(np.arange(int(sel.sum())), B, seed=0)
            for tag, src in (("raw", old), ("ls", new)):
                boots = np.stack([(W @ src.loc[v].loc[ids, "errors"].to_numpy(float)[sel]) / (W @ n[sel]) for v in SYNTH])
                pts = np.array([src.loc[v].loc[ids, "errors"].to_numpy(float)[sel].sum() / n[sel].sum() for v in SYNTH])
                rb = boots.max(0) / np.maximum(boots.min(0), 1e-9)
                summ[f"{fl08.MSHORT[m]} | {label} | {tag}"] = {
                    "worst_over_best": float(pts.max() / pts.min()), "lo": float(np.percentile(rb, 2.5)),
                    "hi": float(np.percentile(rb, 97.5)), "worst": SYNTH[int(pts.argmax())], "best": SYNTH[int(pts.argmin())]}
    sig = lambda lo, hi: np.where(lo > 0, "harder", np.where(hi < 0, "easier", "ns"))
    gv = g[g["voice"].isin(SYNTH)].copy()
    gv["sig_raw"], gv["sig_ls"] = sig(gv["gap_raw_lo"], gv["gap_raw_hi"]), sig(gv["gap_ls_lo"], gv["gap_ls_hi"])
    summ["significance_counts"] = {f"{fl08.MSHORT[m]} | {lab}": {
        "raw": gg["sig_raw"].value_counts().to_dict(), "ls": gg["sig_ls"].value_counts().to_dict()}
        for (m, lab), gg in gv.groupby(["model", "subset"])}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    fig_textprep(g)
    pd.set_option("display.width", 220)
    print(g[g["subset"] != "38 with digits"][["model", "subset", "voice", "gap_raw_text", "gap_ls_text", "gap_ls_lo",
                                              "gap_ls_hi", "change_ls_minus_raw", "change_lo", "change_hi"]].round(4).to_string())
    print(json.dumps(summ, indent=1))


def fig_textprep(g: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = SYNTH + [fl08.MEAN]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, m in zip(axes, (fl08.WHISPER, fl08.PARAKEET)):
        d = g[(g["model"] == m) & (g["subset"] == "all 250")].set_index("voice").reindex(order)
        y = np.arange(len(order))[::-1]
        ax.axvline(0, color=fl08.INK2, lw=0.8)
        ax.errorbar(d["gap_raw_text"] * 100, y + 0.15, xerr=[(d["gap_raw_text"] - d["gap_raw_lo"]) * 100,
                    (d["gap_raw_hi"] - d["gap_raw_text"]) * 100], fmt="o", ms=5, color=fl08.GRAY, lw=1.5,
                    label="raw FLEURS text (as in the follow-up)")
        ax.errorbar(d["gap_ls_text"] * 100, y - 0.15, xerr=[(d["gap_ls_text"] - d["gap_ls_lo"]) * 100,
                    (d["gap_ls_hi"] - d["gap_ls_text"]) * 100], fmt="o", ms=5, color=fl08.BLUE, lw=1.5,
                    label="LibriSpeech-style text (lowercase, no punctuation, numbers in words)")
        ax.set_yticks(y, [fl08.SHORT[v] for v in order])
        ax.set_title(f"{fl08.MSHORT[m]}, clean", loc="left", fontsize=10)
        ax.set_xlabel("synthetic minus FLEURS human WER (points)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2,
               fontsize=8, frameon=False)
    fig.suptitle("Does the TTS input text explain the FLEURS gaps? Same audio pipeline, two text preparations",
                 x=0.02, ha="left", fontsize=10.5)
    fig.text(0.02, -0.12, "95 % CIs, sentence bootstrap (250 sentences). Negative = the voice is easier than human speech.",
             fontsize=7.5, color=fl08.INK2)
    fig.tight_layout()
    fig.savefig(fl08.OUT / "fig4_tts_text_prep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["synth", "run", "analyze", "show"])
    ap.add_argument("--model")
    a = ap.parse_args()
    if a.step == "synth":
        synth()
    elif a.step == "run":
        run(a.model)
    elif a.step == "show":
        for r in subset().head(40).itertuples():
            if r.has_digit or "," in r.text:
                print(repr(r.text), "->", repr(r.tts_ls))
    else:
        analyze()


if __name__ == "__main__":
    main()
