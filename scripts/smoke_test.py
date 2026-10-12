"""Smoke test every component on a ~2 s sample. Writes results/smoke_test.json and WAVs in results/raw/smoke/."""
import json
import time
import traceback

import jiwer
import numpy as np
import soundfile as sf

from sim2real import asr, audio, datasets, tts
from sim2real.paths import RAW, RESULTS

TEXT = "Help, he is not breathing."
OUT = RAW / "smoke"
OUT.mkdir(parents=True, exist_ok=True)
report = {"text": TEXT, "tts": {}, "asr": {}, "other": {}}
norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])


def timed(fn, *a, **k):
    t = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t


clips = {}
tts_jobs = [(f"kokoro:{v}", tts.kokoro, {"voice": v}) for v in tts.KOKORO_VOICES]
tts_jobs += [(f"piper:{v}", tts.piper, {"voice": v}) for v in tts.PIPER_VOICE_IDS]
tts_jobs += [("mms-tts-eng", tts.mms, {})]
tts_jobs += [(f"say:{v}", tts.say, {"voice": v}) for v in tts.SAY_VOICES]
for name, fn, kw in tts_jobs:
    try:
        (a, sr), t_first = timed(fn, TEXT, **kw)
        (a, sr), t_warm = timed(fn, TEXT, **kw)
        a16 = audio.resample(a, sr)
        clips[name] = a16
        sf.write(OUT / f"{name.replace(':', '_')}.wav", a16, 16000)
        report["tts"][name] = {"ok": True, "native_sr": sr, "dur_s": round(len(a) / sr, 2),
                               "t_first_s": round(t_first, 2), "t_warm_s": round(t_warm, 3)}
    except Exception as e:
        report["tts"][name] = {"ok": False, "error": repr(e), "tb": traceback.format_exc()[-800:]}
    print(name, report["tts"][name])

# one real human clip (~2-3 s) from LibriSpeech
try:
    (idx, t_idx) = timed(datasets.librispeech_index)
    idx["dur"] = [sf.info(p).duration for p in idx["path"]]
    row = idx.iloc[(idx["dur"] - 2.5).abs().argmin()]
    (h, sr), _ = timed(datasets.load, row["path"])
    clips["human:librispeech"] = audio.resample(h, sr)
    report["other"]["librispeech"] = {"ok": True, "n_utts": len(idx), "n_speakers": int(idx["speaker"].nunique()),
                                      "hours": round(idx["dur"].sum() / 3600, 2), "index_s": round(t_idx, 2),
                                      "sample_id": row["id"], "sample_text": row["text"], "sample_dur": round(row["dur"], 2)}
    human_ref = row["text"]
except Exception as e:
    report["other"]["librispeech"] = {"ok": False, "error": repr(e)}
    human_ref = None
print("librispeech", report["other"]["librispeech"])

try:
    (meta, t) = timed(datasets.esc50_meta)
    r = meta[meta["category"] == "crying_baby"].iloc[0]
    (n, nsr), t_load = timed(datasets.load, r["path"])
    report["other"]["esc50"] = {"ok": True, "n_clips": len(meta), "n_categories": int(meta["category"].nunique()),
                                "sample": r["filename"], "sr": nsr, "dur_s": round(len(n) / nsr, 2), "load_s": round(t + t_load, 3)}
except Exception as e:
    report["other"]["esc50"] = {"ok": False, "error": repr(e)}
print("esc50", report["other"]["esc50"])

try:
    src = clips.get("kokoro:af_heart")
    g, t = timed(audio.g711_roundtrip, src, 16000)
    sf.write(OUT / "kokoro_af_heart_g711.wav", g, 16000)
    clips["kokoro:af_heart+g711"] = g
    snr = 10 * np.log10(np.sum(src[: len(g)] ** 2) / np.sum((src[: len(g)] - g[: len(src)]) ** 2))
    report["other"]["g711"] = {"ok": True, "t_s": round(t, 3), "len_in": len(src), "len_out": len(g), "snr_vs_input_db": round(float(snr), 1)}
except Exception as e:
    report["other"]["g711"] = {"ok": False, "error": repr(e)}
print("g711", report["other"]["g711"])

for aname, fn in asr.ASR.items():
    rep = {"results": {}}
    try:
        _, t_load = timed(fn, np.zeros(16000, dtype=np.float32))
        rep["t_first_call_incl_load_s"] = round(t_load, 2)
        for cname, a in clips.items():
            hyp, t = timed(fn, a)
            ref = human_ref if cname.startswith("human") else TEXT
            rep["results"][cname] = {"hyp": hyp, "wer": round(jiwer.wer(norm(ref), norm(hyp)), 3), "t_s": round(t, 3),
                                     "rtf": round(t / (len(a) / 16000), 3)}
        rep["ok"] = True
    except Exception as e:
        rep.update(ok=False, error=repr(e), tb=traceback.format_exc()[-800:])
    report["asr"][aname] = rep
    print(aname, json.dumps(rep, indent=1))

(RESULTS / "smoke_test.json").write_text(json.dumps(report, indent=2))
print("WROTE", RESULTS / "smoke_test.json")
