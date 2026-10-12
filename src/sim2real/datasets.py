from __future__ import annotations

import pandas as pd
import soundfile as sf

from .paths import ESC50, LIBRISPEECH


def librispeech_index() -> pd.DataFrame:
    rows = []
    for trans in sorted(LIBRISPEECH.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            spk, chap, _ = uid.split("-")
            rows.append({"id": uid, "speaker": spk, "chapter": chap, "text": text,
                         "path": str(trans.parent / f"{uid}.flac")})
    return pd.DataFrame(rows)


def esc50_meta() -> pd.DataFrame:
    df = pd.read_csv(ESC50 / "meta" / "esc50.csv")
    df["path"] = [str(ESC50 / "audio" / f) for f in df["filename"]]
    return df


def load(path: str):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr
