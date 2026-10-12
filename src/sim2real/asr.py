"""ASR backends. Every function takes (float32 mono audio at 16 kHz) and returns text."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

WHISPER_ID = "mlx-community/whisper-large-v3-turbo"
WAV2VEC2_ID = "facebook/wav2vec2-base-960h"
PARAKEET_ID = "mlx-community/parakeet-tdt-0.6b-v2"


def whisper(audio16k: np.ndarray) -> str:
    import mlx_whisper

    out = mlx_whisper.transcribe(
        audio16k.astype(np.float32), path_or_hf_repo=WHISPER_ID, language="en",
        temperature=0.0, condition_on_previous_text=False, verbose=None,
    )
    return out["text"].strip()


@lru_cache(maxsize=1)
def _w2v():
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return Wav2Vec2Processor.from_pretrained(WAV2VEC2_ID), Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_ID).eval().to(dev), dev


def wav2vec2(audio16k: np.ndarray) -> str:
    import torch

    proc, model, dev = _w2v()
    inp = proc(audio16k, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
    with torch.no_grad():
        ids = model(inp).logits.argmax(-1)
    return proc.batch_decode(ids.cpu())[0].strip()


@lru_cache(maxsize=1)
def _parakeet():
    from parakeet_mlx import from_pretrained

    return from_pretrained(PARAKEET_ID)


def parakeet(audio16k: np.ndarray) -> str:
    import mlx.core as mx
    from parakeet_mlx.audio import get_logmel

    model = _parakeet()
    mel = get_logmel(mx.array(audio16k.astype(np.float32)), model.preprocessor_config)
    res = model.generate(mel)
    return res[0].text.strip()


ASR = {"whisper-large-v3-turbo": whisper, "wav2vec2-base-960h": wav2vec2, "parakeet-tdt-0.6b-v2": parakeet}
