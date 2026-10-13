"""FLEURS English (en_us) test split as an out-of-domain human baseline.

FLEURS (Conneau et al. 2022, CC BY 4.0) is read speech: Wikipedia-style FLoRes sentences read by
volunteers with varied accents, microphones and rooms. It is not phone-call speech.

What the release provides, and what it does not:
- one row per recording: FLoRes sentence id, file name, raw transcription (case and punctuation),
  normalized transcription, number of samples (16 kHz), speaker gender;
- **no speaker id**. Most sentences have two recordings (different readers), some have one.

Selection used here (select_fleurs): one recording per sentence id, raw duration 3 to 12 s (same
window as the LibriSpeech subset), gender-balanced round robin with a fixed seed. Because speaker
ids are missing, every item is its own bootstrap cluster (sentence-level bootstrap); this is the
"provided grouping" (sentence id), not a speaker grouping.

Only data/en_us/test.tsv and data/en_us/audio/test.tar.gz are downloaded, to data/fleurs/hf/.
Only the selected wav files are extracted, to data/fleurs/wav/.
"""
from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from . import tts
from .paths import DATA

REPO = "google/fleurs"
FLEURS = DATA / "fleurs"
HF_DIR = FLEURS / "hf"
TSV = HF_DIR / "data" / "en_us" / "test.tsv"
TAR = HF_DIR / "data" / "en_us" / "audio" / "test.tar.gz"
WAV_DIR = FLEURS / "wav"
AUDIO = DATA / "audio" / "fleurs"
SR = 16000
TSV_COLS = ["sentence_id", "file", "raw", "transcription", "chars", "num_samples", "gender"]


def download() -> None:
    """Fetch only the en_us test TSV and audio archive from the Hugging Face Hub (cached)."""
    from huggingface_hub import hf_hub_download

    for f in ("data/en_us/test.tsv", "data/en_us/audio/test.tar.gz"):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=str(HF_DIR))


def index() -> pd.DataFrame:
    d = pd.read_csv(TSV, sep="\t", header=None, names=TSV_COLS, quoting=3, keep_default_na=False,
                    dtype={"sentence_id": str})
    d["duration_s"] = d["num_samples"].astype(float) / SR
    return d


def select_fleurs(n: int = 250, min_s: float = 3.0, max_s: float = 12.0, seed: int = 0) -> pd.DataFrame:
    """One recording per sentence id, raw duration in [min_s, max_s], gender-balanced.

    For each sentence id one in-range recording is drawn at random (seeded). The sentences are
    then taken in a round robin over gender (MALE, FEMALE, MALE, ...), random order within a
    gender, until n are chosen or a gender runs out (then the other gender fills the rest)."""
    d = index()
    d = d[(d["duration_s"] >= min_s) & (d["duration_s"] <= max_s)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    d = d.iloc[rng.permutation(len(d))]
    one = d.drop_duplicates("sentence_id", keep="first")
    queues = {g: list(grp.index) for g, grp in one.groupby("gender", sort=True)}
    order = sorted(queues, key=lambda g: len(queues[g]))  # the scarcer gender first
    picked: list[int] = []
    while len(picked) < n and any(queues.values()):
        for g in order:
            if queues[g] and len(picked) < n:
                picked.append(queues[g].pop(0))
    s = one.loc[picked].copy()
    s["id"] = "fl" + s["sentence_id"]
    s["text"] = s["raw"].str.strip()
    s["tts_text"] = s["text"]  # raw transcription is already sentence case with punctuation
    # speaker ids are not in the release: each item is its own cluster
    s["speaker"] = s["id"]
    return s.sort_values("id").reset_index(drop=True)


def scoring_ref(text: str) -> str:
    """Reference used for scoring. Whisper's English normalizer (sim2real.asr.normalize) deletes
    any text between parentheses or brackets, which is right for Whisper's own annotations but
    wrong for FLEURS, where parentheses are part of the read sentence (25 of the 250 selected
    sentences). Applied to references and hypotheses alike.
    Brackets are turned into spaces so their content is kept; nothing else changes."""
    return re.sub(r"\s+", " ", re.sub(r"[()\[\]]", " ", text)).strip()


def extract(files: list[str]) -> dict[str, Path]:
    """Extract the named wav files (base names) from the test archive into WAV_DIR.
    Members are matched by base name and written flat, so archive paths cannot escape WAV_DIR."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    want = {f: WAV_DIR / f for f in files}
    todo = {f for f, p in want.items() if not p.exists()}
    if todo:
        with tarfile.open(TAR, "r:gz") as tf:
            for m in tf:
                base = Path(m.name).name
                if m.isfile() and base in todo:
                    src = tf.extractfile(m)
                    assert src is not None
                    tmp = WAV_DIR / f".{base}.tmp"
                    tmp.write_bytes(src.read())
                    tmp.rename(WAV_DIR / base)
                    todo.discard(base)
                    if not todo:
                        break
    missing = [f for f in files if not want[f].exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files not found in {TAR}: {missing[:3]}")
    return want


def voice_dir(voice: str) -> Path:
    return AUDIO / voice.replace(":", "_")


def synth_one(text: str, voice: str, path: Path) -> str:
    """Same as scripts/01_make_corpus.synth_one: synthesize unless cached; 'skip', 'ok' or error."""
    if path.exists() and path.stat().st_size > 1000:
        return "skip"
    try:
        a = tts.synthesize(text, voice)
        if len(a) < 1600:
            return f"too short ({len(a)} samples)"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.wav")
        sf.write(tmp, a, SR, subtype="PCM_16")
        tmp.rename(path)
        return "ok"
    except Exception as e:  # keep going, report at the end
        return repr(e)
