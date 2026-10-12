from __future__ import annotations

import subprocess

import numpy as np
import soxr


def resample(audio: np.ndarray, sr: int, target: int = 16000) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32)
    return soxr.resample(audio.astype(np.float32), sr, target).astype(np.float32)


def g711_roundtrip(audio: np.ndarray, sr: int, out_sr: int = 16000) -> np.ndarray:
    """Encode to 8 kHz G.711 mu-law with ffmpeg, decode, return float32 at out_sr."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
         "-ar", "8000", "-c:a", "pcm_mulaw", "-f", "mulaw", "pipe:1"],
        input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "mulaw", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
         "-ar", str(out_sr), "-f", "s16le", "pipe:1"],
        input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(dec, dtype="<i2").astype(np.float32) / 32768.0
