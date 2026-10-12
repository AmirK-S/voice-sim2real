"""Rate-matched TTS variants of the LibriSpeech subset (review fix I2: speaking rate confound).

For every sentence and every TTS voice, re-synthesize so that the speech duration (edge silence
trimmed, sim2real.channel.speech_duration) matches the human recording of the same sentence:
    factor = duration_default_tts / duration_human
Kokoro speed, Piper 1/length_scale and MMS speaking_rate start from `factor`; macOS `say`
takes words per minute and starts from -r 180. Up to 3 correction passes then multiply the
speed by the remaining duration ratio until it is within 4 % of the human duration.

Audio: data/audio/librispeech/<voice>@rate/<id>.wav. Manifest rows are appended to
data/corpus/manifest.csv with voice "<voice>@rate"; data/corpus/rate_matched.csv records the
factor and the achieved duration ratio.

Usage: python scripts/01b_rate_matched.py [--voices ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import soundfile as sf

from sim2real import tts
from sim2real.channel import speech_duration
from sim2real.paths import DATA

CORPUS = DATA / "corpus"
AUDIO = DATA / "audio" / "librispeech"
SAY_REF_WPM = 180.0


def read(p) -> np.ndarray:
    x, sr = sf.read(ROOT / p, dtype="float32")
    assert sr == 16000
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=",".join(tts.VOICES))
    a = ap.parse_args()
    man = pd.read_csv(CORPUS / "manifest.csv", keep_default_na=False)
    man = man[~man["voice"].str.endswith("@rate")]
    ls = man[man["set"] == "librispeech"]
    human = ls[ls["voice"] == "human"].set_index("id")
    d_h = {i: speech_duration(read(p)) for i, p in human["path"].items()}
    texts = pd.read_csv(CORPUS / "librispeech_subset.csv").set_index("id")["tts_text"]
    rows, recs = [], []
    for voice in a.voices.split(","):
        t0 = time.time()
        g = ls[ls["voice"] == voice].set_index("id")
        out_dir = AUDIO / (voice.replace(":", "_") + "@rate")
        out_dir.mkdir(parents=True, exist_ok=True)

        def job(uid):
            path = out_dir / f"{uid}.wav"
            d_def = speech_duration(read(g.loc[uid, "path"]))
            factor = float(np.clip(d_def / d_h[uid], 0.6, 1.7))
            if not path.exists():
                # up to 4 passes: correct the speed by the remaining duration ratio each time
                if voice.startswith("say:"):
                    wpm = SAY_REF_WPM
                    y = tts.synthesize(texts[uid], voice, say_wpm=wpm)
                else:
                    sp = factor
                    y = tts.synthesize(texts[uid], voice, speed=sp)
                for _ in range(3):
                    ratio = speech_duration(y) / d_h[uid]
                    if abs(ratio - 1) < 0.04:
                        break
                    if voice.startswith("say:"):
                        wpm = float(np.clip(wpm * ratio, 90, 450))
                        y = tts.synthesize(texts[uid], voice, say_wpm=wpm)
                    else:
                        sp = float(np.clip(sp * ratio, 0.5, 2.0))
                        y = tts.synthesize(texts[uid], voice, speed=sp)
                tmp = path.with_suffix(".tmp.wav")
                sf.write(tmp, y, 16000, subtype="PCM_16")
                tmp.rename(path)
            y = read(path.relative_to(ROOT))
            return uid, path, factor, d_def, speech_duration(y), len(y) / 16000

        workers = 6 if voice.startswith("say:") else 1
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(job, list(g.index)))
        for uid, path, factor, d_def, d_new, dur in res:
            r = g.loc[uid]
            rows.append(dict(set="librispeech", id=uid, voice=f"{voice}@rate",
                             path=str(path.relative_to(ROOT)), text=r["text"], keywords="", category="",
                             speaker=r["speaker"], duration_s=round(dur, 3)))
            recs.append(dict(id=uid, voice=voice, factor=factor, d_human=d_h[uid], d_default=d_def,
                             d_rate_matched=d_new, ratio_default=d_def / d_h[uid], ratio_matched=d_new / d_h[uid]))
        rr = pd.DataFrame([x for x in recs if x["voice"] == voice])
        print(f"{voice}: {len(res)} clips in {time.time() - t0:.0f} s; duration ratio vs human "
              f"default median {rr['ratio_default'].median():.2f}, matched median {rr['ratio_matched'].median():.2f} "
              f"(IQR {rr['ratio_matched'].quantile(.25):.2f}-{rr['ratio_matched'].quantile(.75):.2f})", flush=True)
    full = pd.read_csv(CORPUS / "manifest.csv", keep_default_na=False)
    full = full[~full["voice"].isin({r["voice"] for r in rows})]
    pd.concat([full, pd.DataFrame(rows)], ignore_index=True).to_csv(CORPUS / "manifest.csv", index=False)
    pd.DataFrame(recs).to_csv(CORPUS / "rate_matched.csv", index=False)
    print(f"manifest: +{len(rows)} rate-matched rows")


if __name__ == "__main__":
    main()
