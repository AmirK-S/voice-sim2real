"""TTS-error follow-up, review fix I1: a blind listening check of the rendering-error flag.

The precision check in results/followups/tts_errors/ was made from the same transcripts that define
the flag, without audio, by the person who designed it. A valid check needs a different person
listening blind. This script prepares that check and scores it; it cannot do the listening.

  make   60 items, shuffled: the 30 flagged words of hand_check_sample.csv and 30 words that at
         least one strong model (Whisper or Parakeet) got wrong on the same voice's clean clip but
         that the R2 flag did not flag, with the same number per voice (seed 1). Writes
           listening_kit/sheet.csv  item, audio file, reference sentence with the word in [brackets],
                                    and an empty "judgement" column (render_error / fine / unsure)
           listening_kit/key.csv    item -> voice, sentence id, word index, flagged or not
         and copies the 60 TTS clips to data/listening_kit/ (gitignored). The listener must not
         open key.csv, the transcripts or the flag tables.
  score  reads the filled sheet.csv: precision of the flag (flagged items judged render_error),
         the share of unflagged errors judged render_error, and a recall estimate
         (P_flag N_flag) / (P_flag N_flag + P_unflag N_unflag), with N the population counts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sim2real import stats
from sim2real.asr import normalize
from sim2real.paths import DATA, RESULTS

spec = importlib.util.spec_from_file_location("te07", ROOT / "scripts" / "07_tts_errors.py")
te = importlib.util.module_from_spec(spec)
spec.loader.exec_module(te)

KIT = RESULTS / "followups" / "tts_errors" / "listening_kit"
AUDIO = DATA / "listening_kit"


def load():
    u = pd.read_csv(RESULTS / "utterances.csv", keep_default_na=False, dtype={"speaker": str}, low_memory=False)
    ls = u[u["set"] == "librispeech"]
    ids = sorted(ls["id"].unique())
    return u, ids


def unflagged_population(u, ids, flags):
    cl = u[(u["set"] == "librispeech") & (u["condition"] == "clean")]
    err = {(r.voice, r.id, r.model): te.idx_set(r.err_ref_idx) for r in cl.itertuples()}
    pop = []
    for v in te.SYNTH:
        for i in ids:
            s = err[(v, i, te.WH)] | err[(v, i, te.PK)]
            s -= flags["R2"][v].get(i, set())
            pop += [(v, i, k) for k in sorted(s)]
    return pop


def make() -> None:
    KIT.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    u, ids = load()
    flags = te.librispeech_flags(u, ids)
    flagged = pd.read_csv(te.OUT / "hand_check_sample.csv")
    pop = unflagged_population(u, ids, flags)
    rng = np.random.default_rng(1)
    picks = []
    for v, n in flagged.groupby("voice").size().items():
        cand = [p for p in pop if p[0] == v]
        for j in rng.choice(len(cand), size=n, replace=False):
            picks.append(cand[j])
    sub = pd.read_csv(DATA / "corpus" / "librispeech_subset.csv").set_index("id")
    man = pd.read_csv(DATA / "corpus" / "manifest.csv", keep_default_na=False)
    path = {(r.voice, r.id): r.path for r in man[man["set"] == "librispeech"].itertuples()}
    items = [(r.voice, r.id, int(r.word_idx), True) for r in flagged.itertuples()] + [(v, i, k, False) for v, i, k in picks]
    order = rng.permutation(len(items))
    sheet, key = [], []
    for n, j in enumerate(order, 1):
        v, i, k, f = items[j]
        toks = normalize(sub.loc[i, "text"]).split()
        item = f"L{n:02d}"
        dst = AUDIO / f"{item}.wav"
        shutil.copyfile(ROOT / path[(v, i)], dst)
        sheet.append({"item": item, "audio": str(dst.relative_to(ROOT)),
                      "sentence": " ".join(toks[:k] + [f"[{toks[k]}]"] + toks[k + 1:]), "judgement": "", "comment": ""})
        key.append({"item": item, "voice": v, "id": i, "word_idx": k, "word": toks[k], "flagged_R2": f})
    pd.DataFrame(sheet).to_csv(KIT / "sheet.csv", index=False)
    pd.DataFrame(key).to_csv(KIT / "key.csv", index=False)
    n_flag = sum(len(s) for v in te.SYNTH for s in flags["R2"][v].values())
    meta = {"n_items": len(items), "n_flagged_items": int(len(flagged)), "n_unflagged_items": len(picks),
            "population_R2_flagged_words": n_flag, "population_unflagged_error_words": len(pop),
            "instructions": "Play each clip. Judge only the bracketed word: render_error if the voice "
                            "mispronounces, garbles, skips or runs it into the next word so that a careful "
                            "human listener would not hear the reference word; fine if it is said "
                            "clearly; unsure otherwise. Do not open key.csv."}
    (KIT / "README.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def score() -> None:
    sheet = pd.read_csv(KIT / "sheet.csv", keep_default_na=False)
    key = pd.read_csv(KIT / "key.csv")
    meta = json.loads((KIT / "README.json").read_text())
    d = sheet.merge(key, on="item")
    if (d["judgement"] == "").any():
        raise SystemExit(f"{int((d['judgement'] == '').sum())} items not judged yet")
    out = {}
    for f, g in d.groupby("flagged_R2"):
        k, n = int((g["judgement"] == "render_error").sum()), len(g)
        lo, hi = stats.clopper_pearson(k, n)
        out["flagged" if f else "unflagged"] = {"render_error": k, "n": n, "share": k / n, "lo": lo, "hi": hi,
                                                "unsure": int((g["judgement"] == "unsure").sum())}
    pf, pu = out["flagged"]["share"], out["unflagged"]["share"]
    nf, nu = meta["population_R2_flagged_words"], meta["population_unflagged_error_words"]
    out["recall_estimate"] = pf * nf / (pf * nf + pu * nu) if (pf * nf + pu * nu) > 0 else float("nan")
    (KIT / "scores.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["make", "score"])
    a = ap.parse_args()
    make() if a.step == "make" else score()
