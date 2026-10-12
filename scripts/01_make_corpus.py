"""Build the corpus: LibriSpeech subset (human + every TTS voice) and the emergency caller set
(every TTS voice). Audio goes to data/audio/<set>/<voice>/<id>.wav (16 kHz mono); existing files
are skipped. Writes data/corpus/{librispeech_subset.csv, emergency_callers.json, manifest.csv}.

Usage: uv run python scripts/01_make_corpus.py [--n 250] [--voices kokoro:af_heart,...] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import soundfile as sf

from sim2real import tts
from sim2real.audio import resample
from sim2real.data import emergency_callers, select_librispeech
from sim2real.datasets import load
from sim2real.paths import DATA

AUDIO = DATA / "audio"
CORPUS = DATA / "corpus"


def voice_dir(set_name: str, voice: str):
    return AUDIO / set_name / voice.replace(":", "_")


def synth_one(text: str, voice: str, path) -> str:
    """Synthesize to `path` unless it exists. Returns 'skip', 'ok' or an error string."""
    if path.exists() and path.stat().st_size > 1000:
        return "skip"
    try:
        a = tts.synthesize(text, voice)
        if len(a) < 1600:
            return f"too short ({len(a)} samples)"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.wav")
        sf.write(tmp, a, 16000, subtype="PCM_16")
        tmp.rename(path)
        return "ok"
    except Exception as e:  # keep going, report at the end
        return repr(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--voices", default=",".join(tts.VOICES))
    ap.add_argument("--limit", type=int, default=None, help="only the first N items per set (quick test)")
    args = ap.parse_args()
    voices = args.voices.split(",")
    CORPUS.mkdir(parents=True, exist_ok=True)

    ls = select_librispeech(n=args.n)
    ls.to_csv(CORPUS / "librispeech_subset.csv", index=False)
    em = emergency_callers()
    (CORPUS / "emergency_callers.json").write_text(json.dumps(em, indent=1))
    print(f"librispeech: {len(ls)} utts, {ls['speaker'].nunique()} speakers, {ls['duration_s'].sum() / 60:.1f} min")
    print(f"emergency: {len(em)} utterances")

    items = []  # (set, id, text_for_tts, reference_text, keywords, category, speaker)
    for r in ls.itertuples():
        items.append(("librispeech", r.id, r.tts_text, r.text, [], "", r.speaker))
    for c in em:
        items.append(("emergency", c["id"], c["text"], c["text"], c["keywords"], c["category"], ""))
    if args.limit:
        items = [it for s in ("librispeech", "emergency") for it in [i for i in items if i[0] == s][: args.limit]]

    rows, errors = [], []

    # human audio (LibriSpeech only), resampled copy at 16 kHz
    paths_by_id = dict(zip(ls["id"], ls["path"]))
    for s, uid, _, ref, kw, cat, spk in items:
        if s != "librispeech":
            continue
        out = voice_dir(s, "human") / f"{uid}.wav"
        if not out.exists():
            a, sr = load(paths_by_id[uid])
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, resample(a, sr), 16000, subtype="PCM_16")
        rows.append(dict(set=s, id=uid, voice="human", path=str(out.relative_to(DATA.parent)), text=ref,
                         keywords="", category=cat, speaker=spk))

    for voice in voices:
        t0 = time.perf_counter()
        jobs = [(it, voice_dir(it[0], voice) / f"{it[1]}.wav") for it in items]
        # `say` is a subprocess: run several at once. Model-based engines run sequentially.
        workers = 6 if voice.startswith("say:") else 1
        with ThreadPoolExecutor(workers) as ex:
            status = list(ex.map(lambda j: synth_one(j[0][2], voice, j[1]), jobs))
        n_ok = status.count("ok")
        n_skip = status.count("skip")
        for (it, path), st in zip(jobs, status):
            s, uid, _, ref, kw, cat, spk = it
            if st in ("ok", "skip"):
                rows.append(dict(set=s, id=uid, voice=voice, path=str(path.relative_to(DATA.parent)), text=ref,
                                 keywords="|".join(kw), category=cat, speaker=spk))
            else:
                errors.append(dict(set=s, id=uid, voice=voice, error=st))
        print(f"{voice}: {n_ok} new, {n_skip} cached, {len(jobs) - n_ok - n_skip} failed, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)

    man = pd.DataFrame(rows)
    man["duration_s"] = [round(sf.info(DATA.parent / p).duration, 3) for p in man["path"]]
    man.to_csv(CORPUS / "manifest.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(CORPUS / "errors.csv", index=False)
        print(f"{len(errors)} failures, see {CORPUS / 'errors.csv'}")
    print(f"manifest: {len(man)} rows -> {CORPUS / 'manifest.csv'}")
    print(man.groupby(["set", "voice"])["duration_s"].agg(["count", "mean"]).round(2).to_string())


if __name__ == "__main__":
    main()
