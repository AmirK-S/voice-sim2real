from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
LIBRISPEECH = DATA / "librispeech" / "LibriSpeech" / "test-clean"
ESC50 = DATA / "esc50" / "ESC-50-master"
PIPER = DATA / "piper" / "en" / "en_US"
PIPER_VOICES = {
    "en_US-lessac-medium": PIPER / "lessac" / "medium" / "en_US-lessac-medium.onnx",
    "en_US-ryan-medium": PIPER / "ryan" / "medium" / "en_US-ryan-medium.onnx",
}
