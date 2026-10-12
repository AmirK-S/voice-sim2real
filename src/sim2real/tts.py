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


def kokoro(text: str, voice: str = "af_heart"):
    pipe = _kokoro()
    chunks = [np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=voice)]
    return np.concatenate(chunks), 24000


@lru_cache(maxsize=4)
def _piper(voice: str):
    from piper import PiperVoice

    return PiperVoice.load(str(PIPER_VOICES[voice]))


def piper(text: str, voice: str = "en_US-lessac-medium"):
    v = _piper(voice)
    chunks = list(v.synthesize(text))
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return audio, chunks[0].sample_rate


@lru_cache(maxsize=1)
def _mms():
    from transformers import AutoTokenizer, VitsModel

    return VitsModel.from_pretrained(MMS_TTS_ID).eval(), AutoTokenizer.from_pretrained(MMS_TTS_ID)


def mms(text: str, seed: int = 0):
    import torch

    model, tok = _mms()
    torch.manual_seed(seed)
    with torch.no_grad():
        wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
    return wav.astype(np.float32), model.config.sampling_rate


def say(text: str, voice: str = "Samantha"):
    with tempfile.TemporaryDirectory() as d:
        aiff = Path(d) / "o.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        audio, sr = sf.read(aiff, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr
