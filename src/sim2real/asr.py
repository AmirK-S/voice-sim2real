"""ASR backends behind one interface, with a disk cache and a shared scoring normalizer.

Main entry points:
    transcribe(paths, model_id)          -> list[str]   audio files on disk
    transcribe_arrays(arrays, model_id)  -> list[str]   16 kHz float32 arrays in memory
    normalize(text) / wer(refs, hyps)                   identical scoring for every model

Models (ids, aliases in MODEL_ALIASES):
    whisper-large-v3-turbo  mlx-whisper, language en, temperature 0, no fallback.
                            Clips up to 30 s are decoded in batches of 30 s windows;
                            longer clips go through mlx_whisper.transcribe (sliding window).
    wav2vec2-base-960h      transformers, greedy CTC, one utterance per forward pass.
    parakeet-tdt-0.6b-v2    parakeet-mlx, greedy TDT, one utterance per forward pass.

wav2vec2 and Parakeet run one utterance at a time on purpose: neither encoder masks
padding here, so a padded batch would make a transcript depend on its batch neighbours,
which would break the (model, audio hash) cache. They are fast enough (RTF < 0.1).

Cache: data/cache/asr/<model>/<hh>/<sha1>.txt, key = sha1(CACHE_VERSION, model, audio).
For files the audio part is the raw file bytes; for arrays it is the float32 samples.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .paths import DATA

WHISPER_ID = "mlx-community/whisper-large-v3-turbo"
WAV2VEC2_ID = "facebook/wav2vec2-base-960h"
PARAKEET_ID = "mlx-community/parakeet-tdt-0.6b-v2"

MODELS = ("whisper-large-v3-turbo", "wav2vec2-base-960h", "parakeet-tdt-0.6b-v2")
MODEL_ALIASES = {"whisper": MODELS[0], "wav2vec2": MODELS[1], "parakeet": MODELS[2]}
CACHE_DIR = DATA / "cache" / "asr"
CACHE_VERSION = "v1"  # bump when decoding settings change, to invalidate old transcripts
SR = 16000
WHISPER_BATCH = 8
WHISPER_MAX_S = 30.0


def resolve(model_id: str) -> str:
    """Map an alias ("whisper") or a full id to a canonical model id."""
    mid = MODEL_ALIASES.get(model_id, model_id)
    if mid not in MODELS:
        raise ValueError(f"unknown ASR model {model_id!r}; choose from {MODELS} or {tuple(MODEL_ALIASES)}")
    return mid


# ---------------------------------------------------------------- normalization

@lru_cache(maxsize=1)
def _whisper_normalizer():
    """Whisper's English normalizer from transformers (number words -> digits, contractions,
    fillers). The British-to-American spelling map is not shipped locally, so it is empty."""
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        return EnglishTextNormalizer({})
    except Exception:  # pragma: no cover, fallback keeps scoring usable
        return None


def normalize(text: str) -> str:
    """Scoring normalization applied identically to references and every model's output:
    Whisper English normalizer when available, then lowercase, strip punctuation,
    collapse spaces. Returns "" for empty input."""
    text = unicodedata.normalize("NFKC", text or "")
    norm = _whisper_normalizer()
    if norm is not None:
        text = norm(text)
    text = text.lower().replace("'", "")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(refs: Sequence[str] | str, hyps: Sequence[str] | str) -> float:
    """Corpus WER (total edits / total reference words) after normalize().
    Accepts single strings or parallel lists."""
    import jiwer

    if isinstance(refs, str):
        refs, hyps = [refs], [hyps]
    r = [normalize(x) for x in refs]
    h = [normalize(x) for x in hyps]
    keep = [i for i, x in enumerate(r) if x]
    if not keep:
        raise ValueError("all references are empty after normalization")
    return float(jiwer.wer([r[i] for i in keep], [h[i] for i in keep]))


# ---------------------------------------------------------------- backends (single clip)

def _as16k(audio16k: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(audio16k, dtype=np.float32).reshape(-1))


def whisper(audio16k: np.ndarray) -> str:
    """Whisper large-v3-turbo on one clip (same decoding as the batched path)."""
    return _whisper_batch([_as16k(audio16k)])[0]


@lru_cache(maxsize=1)
def _whisper_model():
    import mlx.core as mx
    from mlx_whisper.transcribe import ModelHolder

    return ModelHolder.get_model(WHISPER_ID, mx.float16)


def _whisper_long(audio: np.ndarray) -> str:
    import mlx_whisper

    out = mlx_whisper.transcribe(
        audio, path_or_hf_repo=WHISPER_ID, language="en", task="transcribe",
        temperature=0.0, condition_on_previous_text=False, verbose=None,
    )
    return out["text"].strip()


def _whisper_batch(arrays: list[np.ndarray]) -> list[str]:
    """Batched single-window decoding: mel computed exactly as mlx_whisper.transcribe does
    (audio + 30 s of zero padding, first 3000 frames), greedy, timestamps on, no
    temperature fallback and no no-speech gating (an empty or silent clip still decodes)."""
    import mlx.core as mx
    from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions, decode

    model = _whisper_model()
    out: list[str | None] = [None] * len(arrays)
    short = []
    for i, a in enumerate(arrays):
        if len(a) / SR > WHISPER_MAX_S:
            out[i] = _whisper_long(a)
        else:
            short.append(i)
    opts = DecodingOptions(language="en", task="transcribe", temperature=0.0, fp16=True)
    for s in range(0, len(short), WHISPER_BATCH):
        idx = short[s : s + WHISPER_BATCH]
        mels = []
        for i in idx:
            mel = log_mel_spectrogram(arrays[i], n_mels=model.dims.n_mels, padding=N_SAMPLES)
            mels.append(pad_or_trim(mel[:N_FRAMES], N_FRAMES, axis=-2).astype(mx.float16))
        results = decode(model, mx.stack(mels), opts)
        for i, r in zip(idx, results):
            out[i] = r.text.strip()
    return [x or "" for x in out]


@lru_cache(maxsize=1)
def _w2v():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_ID).eval().to(dev)
    return Wav2Vec2Processor.from_pretrained(WAV2VEC2_ID), model, dev


def wav2vec2(audio16k: np.ndarray) -> str:
    """wav2vec2-base-960h, greedy CTC (argmax then collapse), one clip."""
    import torch

    audio = _as16k(audio16k)
    if len(audio) < 400:  # shorter than one conv receptive field
        return ""
    proc, model, dev = _w2v()
    inp = proc(audio, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        ids = model(inp).logits.argmax(-1)
    return proc.batch_decode(ids.cpu())[0].strip()


@lru_cache(maxsize=1)
def _parakeet():
    from parakeet_mlx import from_pretrained

    return from_pretrained(PARAKEET_ID)


def parakeet(audio16k: np.ndarray) -> str:
    """Parakeet TDT 0.6B v2 (MLX), greedy decoding, one clip."""
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    audio = _as16k(audio16k)
    if len(audio) < 1600:
        return ""
    model = _parakeet()
    mel = get_logmel(mx.array(audio), model.preprocessor_config)
    return model.generate(mel)[0].text.strip()


ASR = {MODELS[0]: whisper, MODELS[1]: wav2vec2, MODELS[2]: parakeet}


def _run_batch(model_id: str, arrays: list[np.ndarray]) -> list[str]:
    if model_id == MODELS[0]:
        return _whisper_batch(arrays)
    fn = ASR[model_id]
    return [fn(a) for a in arrays]


# ---------------------------------------------------------------- cache + public API

def _key(model_id: str, payload: bytes) -> str:
    h = hashlib.sha1(f"{CACHE_VERSION}|{model_id}|".encode())
    h.update(payload)
    return h.hexdigest()


def _cache_path(model_id: str, key: str) -> Path:
    return CACHE_DIR / model_id / key[:2] / f"{key}.txt"


def _cache_get(model_id: str, key: str) -> str | None:
    p = _cache_path(model_id, key)
    return p.read_text(encoding="utf-8") if p.exists() else None


def _cache_put(model_id: str, key: str, text: str) -> None:
    p = _cache_path(model_id, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{np.random.randint(1 << 30)}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)  # atomic, safe if several processes share the cache


def _load16k(path: str | Path) -> np.ndarray:
    from .audio import resample
    from .datasets import load

    audio, sr = load(str(path))
    return resample(audio, sr, SR)


def _transcribe_keys(keys: list[str], loaders, model_id: str, use_cache: bool) -> list[str]:
    out: list[str | None] = [_cache_get(model_id, k) if use_cache else None for k in keys]
    todo = [i for i, x in enumerate(out) if x is None]
    chunk = WHISPER_BATCH * 4  # bounded memory: load and decode a slice at a time
    for s in range(0, len(todo), chunk):
        idx = todo[s : s + chunk]
        texts = _run_batch(model_id, [loaders(i) for i in idx])
        for i, t in zip(idx, texts):
            out[i] = t
            if use_cache:
                _cache_put(model_id, keys[i], t)
    return [x or "" for x in out]


def transcribe(paths: Iterable[str | Path], model_id: str, use_cache: bool = True) -> list[str]:
    """Transcribe audio files (any format soundfile reads, any rate, resampled to 16 kHz).
    Returns raw model text in input order; score it with normalize()/wer()."""
    model_id = resolve(model_id)
    paths = [Path(p) for p in paths]
    keys = [_key(model_id, p.read_bytes()) for p in paths]
    return _transcribe_keys(keys, lambda i: _load16k(paths[i]), model_id, use_cache)


def transcribe_arrays(arrays: Iterable[np.ndarray], model_id: str, use_cache: bool = True) -> list[str]:
    """Same as transcribe() for in-memory 16 kHz float32 mono arrays."""
    model_id = resolve(model_id)
    arrays = [_as16k(a) for a in arrays]
    keys = [_key(model_id, a.tobytes()) for a in arrays]
    return _transcribe_keys(keys, lambda i: arrays[i], model_id, use_cache)
