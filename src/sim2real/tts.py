"""Text-to-speech backends. Every function returns (float32 mono numpy array, sample_rate)."""
from __future__ import annotations

import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import PIPER_VOICES

KOKORO_VOICES = ["af_heart", "am_michael"]
PIPER_VOICE_IDS = list(PIPER_VOICES)
SAY_VOICES = ["Samantha", "Daniel"]
MMS_TTS_ID = "facebook/mms-tts-eng"
KOKORO_ID = "hexgrad/Kokoro-82M"


@lru_cache(maxsize=1)
def _kokoro():
    from kokoro import KPipeline

    return KPipeline(lang_code="a", repo_id=KOKORO_ID)


def kokoro(text: str, voice: str = "af_heart", speed: float = 1.0):
    pipe = _kokoro()
    chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=voice, speed=speed)]
    return np.concatenate(chunks), 24000


@lru_cache(maxsize=4)
def _piper(voice: str):
    from piper import PiperVoice

    return PiperVoice.load(str(PIPER_VOICES[voice]))


def piper(text: str, voice: str = "en_US-lessac-medium", speed: float = 1.0):
    from piper.config import SynthesisConfig

    v = _piper(voice)
    cfg = None if speed == 1.0 else SynthesisConfig(length_scale=(v.config.length_scale or 1.0) / speed)
    chunks = list(v.synthesize(text, syn_config=cfg))
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return audio, chunks[0].sample_rate


@lru_cache(maxsize=1)
def _mms():
    from transformers import AutoTokenizer, VitsModel

    return VitsModel.from_pretrained(MMS_TTS_ID).eval(), AutoTokenizer.from_pretrained(MMS_TTS_ID)


def mms(text: str, seed: int = 0, speed: float = 1.0):
    import torch

    model, tok = _mms()
    torch.manual_seed(seed)
    model.speaking_rate = speed  # VITS duration predictor: length_scale = 1 / speaking_rate
    with torch.no_grad():
        wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
    model.speaking_rate = 1.0
    return wav.astype(np.float32), model.config.sampling_rate


def say(text: str, voice: str = "Samantha", rate_wpm: float | None = None):
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "o.aiff"
        rate = [] if rate_wpm is None else ["-r", str(int(round(rate_wpm)))]
        subprocess.run(["say", "-v", voice, *rate, "-o", str(aiff), text], check=True)
        audio, sr = sf.read(aiff, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


# ---------------------------------------------------------------------------
# Unified entry point used by the corpus builder and the experiments.
# ---------------------------------------------------------------------------

VOICES = (
    [f"kokoro:{v}" for v in KOKORO_VOICES]
    + [f"piper:{v}" for v in PIPER_VOICE_IDS]
    + ["mms:eng"]
    + [f"say:{v}" for v in SAY_VOICES]
)
"""Every synthetic voice of the experiment, as 'engine:voice' ids."""


def synthesize(text: str, voice_id: str, speed: float = 1.0, say_wpm: float | None = None) -> np.ndarray:
    """Synthesize `text` with `voice_id` ('kokoro:af_heart', 'piper:en_US-lessac-medium', 'mms:eng',
    'say:Samantha', ...) and return 16 kHz mono float32 audio clipped to [-1, 1].

    `speed` > 1 speaks faster (Kokoro speed, Piper 1/length_scale, MMS speaking_rate).
    macOS `say` takes an absolute rate in words per minute instead (`say_wpm`)."""
    from .audio import resample

    engine, _, voice = voice_id.partition(":")
    if engine == "kokoro":
        a, sr = kokoro(text, voice=voice, speed=speed)
    elif engine == "piper":
        a, sr = piper(text, voice=voice, speed=speed)
    elif engine == "mms":
        if voice != "eng":
            raise ValueError(f"only mms:eng is supported, got {voice_id!r}")
        a, sr = mms(text, speed=speed)
    elif engine == "say":
        a, sr = say(text, voice=voice, rate_wpm=say_wpm)
    else:
        raise ValueError(f"unknown TTS engine in {voice_id!r}")
    a = np.asarray(a, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    return np.clip(resample(a, sr, 16000), -1.0, 1.0).astype(np.float32)
