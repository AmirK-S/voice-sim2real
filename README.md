# voice-sim2real

[![tests](https://github.com/AmirK-S/voice-sim2real/actions/workflows/tests.yml/badge.svg)](https://github.com/AmirK-S/voice-sim2real/actions/workflows/tests.yml)

**Do synthetic callers change which model wins?** A local, reproducible measurement of one slice of the sim-to-real gap in voice-agent evaluation: the same scripts read by real people and by 7 TTS voices, sent through simulated phone channels, transcribed by 3 public ASR models. About 74,000 scored transcriptions.

**TL;DR** (reconciled 13/10/2026 after the follow-ups; see `CLAIMS-STATUS.md`)
- No single synthetic voice confidently flipped the winner between two close models, on LibriSpeech or FLEURS, and no voice or average picked the wrong winner with confidence. Whether a single voice's lab-vs-reality difference survives correction depends on how Whisper is decoded: none with the main run's decoder, 3 of 28 with Whisper's standard `transcribe()` (MMS and Piper ryan, exaggerating Parakeet's lead).
- **Averaging voices gave intervals that were too narrow**, because the voices were held fixed. With voices resampled and a multiple-testing correction, the average still favours Parakeet on 2 of 3 noisy conditions (3 of 3 with Whisper's standard `transcribe()`), in the human estimate's direction, which 250 human sentences can neither confirm nor rule out. A lab bias is neither shown nor excluded. (Corrected: the first wording, "manufactured certainty", overstated it.)
- The voice changes **how big** a difference looks: 0.37x to 3.72x the real gap on LibriSpeech (descriptive extremes, all in pairs with the in-domain wav2vec2 control); 0.26x to 0.92x on FLEURS, where the lab always understates.
- **The lab is optimistic about noise.** Kokoro hides about half the damage noise does to LibriSpeech speech; against FLEURS readers, voices understate it in 43 of 49 cells, and every voice except MMS is easier than humans.
- **Whisper writes text on 95 to 100 % of noise clips after a phone line** with its standard `transcribe` settings (mlx-whisper 0.4.3; its no-speech gate never fired). A default Silero VAD cuts that to 2 to 3 %. (Corrected: the first figure, 33 of 100, came from a raw decoder with non-standard padding.)
- About 25 real recordings from about 20 different speakers, plus synthetic runs of the same scripts (PPI), give a valid interval on the real-speech WER level, tighter than the recordings alone. They do not help a close model-vs-model comparison.

One-page summary: [`REPORT.md`](REPORT.md) (with a dated update box). Every number, CI and caveat: [`results/FINDINGS.md`](results/FINDINGS.md). Follow-ups: [`FOLLOWUPS.md`](FOLLOWUPS.md).

![Whisper vs Parakeet, lab vs reality](results/figures/fig4_decision_whisper_vs_parakeet.png)

## Why

Voice agents are often tested with simulated callers: a script, a TTS voice, an automated run. Simulated callers do not sound or behave like real callers, so a change can look better in the lab than in production. This repo measures the acoustic half of that gap with public models standing in for the agent's ear:

- does the WER change depending on which synthetic voice reads the script?
- does the lab pick a different winner between two ASR models than real speech does (a decision flip)?
- does the lab estimate the cost of a phone channel (codec, noise, packet loss) correctly?
- how often are emergency keywords ("bleeding", "not breathing", "chest pain", "help", ...) missed, per voice and per channel?

It does not measure caller behaviour (panic, hesitation, off-script answers, talking over the agent), a streaming pipeline (VAD, endpointing, barge-in) or real emergency calls. "Human" here means LibriSpeech read speech.

## Quickstart

Apple Silicon Mac, `uv`, `ffmpeg`, system libopus (`brew install opus`). Python 3.12 is pinned.

```
uv sync                                          # install dependencies (uv.lock)
uv run python scripts/smoke_test.py              # end-to-end check on a few clips
uv run python scripts/01_make_corpus.py          # 250 LibriSpeech sentences x (human + 7 TTS), 122 caller scripts x 7 TTS
uv run python scripts/01b_rate_matched.py        # TTS re-synthesized at the human speaking rate
uv run python scripts/02_channel_selftest.py     # channel checks (G.711, levels, SNR, loss and concealment, Opus, G.722)
uv run python scripts/03_asr_selftest.py         # ASR checks
uv run python scripts/04_run_experiment.py       # full grid, cached and resumable
uv run python scripts/05_analyze.py              # tables, figures, summary.json
uv run --with pytest python -m pytest tests      # stats unit tests
```

Useful variants of step 4: `--model whisper --conditions clean,realistic` runs one model on a subset, `--group rate` runs the rate-matched voices, `--noise-only` runs the hallucination check on pure noise, `--merge-only` rebuilds `results/utterances.csv` from the transcript shards. `scripts/reproduce.sh` (or `make reproduce`) runs the exact published grid.

Data goes to `data/` (gitignored): LibriSpeech test-clean (openslr.org, CC BY 4.0), ESC-50 noise (GitHub, CC BY-NC 3.0), Piper voices. Models live in the Hugging Face cache.

The same steps are available as `make setup`, `make smoke`, `make corpus`, `make run`, `make analyze` and `make test`.

## What is in the experiment

| Part | What |
|---|---|
| Human speech | 250 LibriSpeech test-clean sentences, 40 speakers |
| Caller scripts | 122 written scripts: 64 emergency, 16 fall needing help, 16 fall but fine, 26 non-emergency. TTS only, no human audio |
| TTS voices (7) | Kokoro af_heart, am_michael; Piper lessac, ryan; Meta MMS-TTS eng; macOS `say` Samantha, Daniel |
| ASR models (3) | Whisper large-v3-turbo (MLX), Parakeet TDT 0.6B v2 (MLX), wav2vec2-base-960h (in-domain control: trained on LibriSpeech) |
| Clip preparation | edge silence trimmed, 250 ms lead and trail, active level -26 dBov, -65 dBFS noise floor |
| Channels (10) | clean; phone (8 kHz, handset filter, G.711 mu-law); G.722 wideband; Opus narrowband about 10 kb/s; phone + ESC-50 noise at 10 and 5 dB SNR measured after the line; muffled pendant-like audio; 5 % packet loss with G.711 Appendix I-style concealment; Opus with 5 % loss and real libopus concealment; "realistic" (street noise 10 dB, phone line, 3 % bursty loss) |
| Metrics | corpus WER (Whisper English normalizer), gap vs human on the same sentences, channel damage, emergency keyword recall, call-level detection, false alarms, text produced on pure noise |
| Statistics | speaker-cluster bootstrap CIs, difference-in-differences test for lab-vs-reality decisions with Holm correction, Benjamini-Hochberg on large tables, exact McNemar for keyword detection, rule of three for zero-miss claims, prediction-powered inference (PPI, PPI++) |

The full run is 73,746 scored transcriptions. Whisper ran 4 of the 10 channels (clean, phone, phone+noise10, realistic) for GPU-time reasons; Parakeet and wav2vec2 ran all 10.

## Headline results

All numbers in WER percentage points ("points") with 95 % CIs; details in `results/FINDINGS.md`.

| Question | Result |
|---|---|
| Does a single synthetic voice flip the winner between Whisper and Parakeet? | No confident flip. Real speech only has a winner on clean audio (Parakeet by 0.37 [0.12, 0.67]; 0.25 [0.04, 0.52] with Whisper's standard `transcribe()`). No voice's lab-vs-reality difference survives Holm correction across 28 decisions with the main run's decoder; with standard Whisper 3 do (MMS clean and phone+noise10, Piper ryan clean), all overstating Parakeet's lead (`results/followups/whisper_std`). |
| Does averaging 7 voices help? | It picks the right winner on clean audio. On the 3 phone and noisy conditions it declares Parakeet better where the human CI includes 0; with voices resampled and Holm over the 4 conditions, on 2 of 3 (3 of 3 with standard Whisper). The human sample cannot confirm or rule out gaps of that size, and the lab offset is not significant, so this is over-precision; a bias is neither shown nor excluded (`results/followups/literature_checks`, `results/followups/calibration`). |
| Large gaps (Whisper or Parakeet vs wav2vec2) | Winner never flips, but the lab's gap ranges from 0.37x to 3.72x the real gap. Realistic channel: real Whisper vs wav2vec2 gap 21.7 points; lab says 9.6 (Kokoro) to 35.8 (macOS Daniel). |
| Synthetic vs human WER, clean audio | LibriSpeech: Kokoro -0.2 to -0.4 points (easier than humans), Piper ryan +0.8 to +1.4, Meta MMS +3.6 to +4.6 (harder; a third to two thirds of it flagged rendering errors, not checked by listening). FLEURS: every voice except MMS is easier than humans. |
| Cost of noise | Kokoro shows about half the WER increase that the same noise causes on human speech (Whisper phone+noise10: real +2.13, Kokoro error -1.18 [-2.11, -0.31]). On FLEURS every voice understates it (43 of 49 cells). |
| Emergency keyword recall, realistic channel | Whisper 95.5 %, Parakeet 94.6 %, wav2vec2 55.2 %. Call-level misses: 7 and 9 of 546 renderings for Whisper and Parakeet. |
| Text on pure noise (100 clips) | Main run (raw decoder, non-standard padding): Whisper 33, Parakeet 2, wav2vec2 21. With Whisper's standard `transcribe` settings: 95 of 100, and 100 of 100 at random lengths; Silero VAD default: 2 to 3 (`results/followups/phantom/v2`). |
| PPI | About 25 human recordings plus TTS runs of the same scripts give a valid interval on the real-speech WER, 9 to 23 % narrower than the 25 human recordings alone. Validated with about 19 distinct speakers per draw; no gain for a model-vs-model difference. |

## Follow-ups

Four follow-ups ran after the main run, plus published statistical checks and a re-transcription with Whisper's standard settings, and some of them correct the first write-up:
- [`FOLLOWUPS.md`](FOLLOWUPS.md): what each follow-up found (calibration and active selection, FLEURS, TTS rendering errors, phantom speech and guards), with limits and the closest prior work.
- [`CLAIMS-STATUS.md`](CLAIMS-STATUS.md): every claim of the report and the follow-ups, marked holds, weakened, replaced or new, with its source file.
- [`RELATED-WORK.md`](RELATED-WORK.md): where this sits in the literature (149 checked references), and what is and is not new.
- Code and outputs: `results/followups/{calibration,fleurs,tts_errors,phantom,literature_checks,whisper_std}/`, each with a `NOTES.md`.

## Charts

| | |
|---|---|
| ![WER by voice](results/figures/fig1_wer_by_voice.png) | ![WER by condition](results/figures/fig2_wer_by_condition.png) |
| ![Gap vs human by voice](results/figures/fig3_gap_by_voice.png) | ![Whisper vs Parakeet, lab vs reality](results/figures/fig4_decision_whisper_vs_parakeet.png) |
| ![Keyword recall](results/figures/fig5_keyword_recall.png) | ![PPI interval width](results/figures/fig6_ppi_ci_width.png) |
| ![Rate-matched voices](results/figures/fig7_rate_matched.png) | |

Tables behind every number: `results/tables/` (18 CSV files) and `results/summary.json`.

## Limitations

- **Behaviour is absent.** Every voice reads the identical script. The behavioural half of the gap is probably the larger one, and it is not here.
- **The human baseline is in-domain.** LibriSpeech is wav2vec2's training domain and likely part of the others' training data. FLEURS has since been run (`results/followups/fleurs`): against it the voices look easier and the lab more optimistic, so LibriSpeech-based gaps are neither upper nor lower bounds (corrected 13/10/2026; an earlier version said "upper bounds"). Real phone recordings of the caller scripts remain the fix.
- **Whisper decoding.** The main run decoded Whisper with a raw decoder whose window padding differs from `transcribe()` for clips under 30 s. Re-decoded with standard `transcribe()` (`results/followups/whisper_std`): Whisper WERs move by at most 0.35 points (human -0.12 to -0.25, not significant), but that is enough to change which close-comparison results pass the significance threshold (see above). FLEURS and the rate-matched voices still use the raw decoder.
- **Small human sample for close comparisons.** 250 sentences from 40 speakers detect a Whisper vs Parakeet gap of about 1 point at 80 % power on the noisy channels, so "no winner" means "cannot tell".
- **Offline ASR on single utterances.** No streaming, endpointing, barge-in or turn-taking.
- **Simulated channels.** The handset filter approximates P.48 and the concealment is modelled on G.711 Appendix I, not bit-exact. ESC-50 noise approximates street and room noise.
- **Partial grid.** Whisper ran 4 of 10 channels; rate-matched voices ran on clean audio only.
- **Nobody listened to the clips.** TTS rendering errors (MMS, "hell" for "help" on macOS Daniel) are inferred from transcripts and a shared-error filter; a blind listening kit exists (`results/followups/tts_errors`) and has not been run.
- **Phantom-speech scope.** The standard-settings rates cover 200 noise clips, four noise types at one level, under mlx-whisper 0.4.3; Silero was applied offline on whole clips.
- **Keyword recall has no human reference.** The caller scripts exist only as TTS audio, so keyword comparisons are between voices and channels.
- **Licence.** ESC-50 is CC BY-NC.

## How to extend

The pipeline is three plain functions, so each piece can be swapped.

**Your own TTS.** Add a branch in `synthesize()` in `src/sim2real/tts.py` that returns 16 kHz mono float32 audio, and add its `engine:voice` id to `VOICES`. Then rerun steps 1, 4 and 5. Recorded human audio of the caller scripts can enter the same way, as a voice that loads files instead of synthesizing.

**Your own ASR.** Write a function `my_asr(audio16k: np.ndarray) -> str` in `src/sim2real/asr.py`, add its id to `MODELS` and to the `ASR` dict (and an alias in `MODEL_ALIASES`). Transcripts are cached by audio hash, so only new clips are decoded.

**Your own channel.** Add an entry to `CONDITIONS` in `src/sim2real/channel.py`: a function `(audio16k, seed, meta) -> audio16k`. The seed comes from the sentence id, so every voice gets the same noise bed and loss pattern and comparisons stay paired.

**Your own agent.** Replace the ASR step with a call to your stack (for example, stream the channel-processed audio into it and keep the final transcript, the detected intent or the escalation decision), and score that output instead of WER. The statistics in `src/sim2real/stats.py` (paired and cluster bootstrap, difference in differences, McNemar, rule of three, PPI) take plain arrays and do not depend on ASR.

**Your own scripts.** The caller set lives in `src/sim2real/data.py` (`emergency_callers()`, `KEYWORDS`). Keyword recall and call-level detection use that keyword list.

## Project layout

```
src/sim2real/   tts.py (voices), channel.py (phone line, noise, loss, codecs), asr.py (models, cache),
                stats.py (bootstrap, DiD, McNemar, rule of three, PPI), data.py (corpus, caller scripts)
scripts/        numbered pipeline steps, self-tests, reproduce.sh
results/        FINDINGS.md, tables/ (CSV), figures/ (PNG), summary.json
tests/          statistics unit tests (run in CI)
```

## Troubleshooting

If `import sim2real` fails under `uv run`, macOS may have flagged the editable `.pth` file in `.venv` as hidden. The scripts add `src/` to `sys.path` themselves; `chflags nohidden .venv/lib/python3.12/site-packages/*.pth` also fixes it.

## Licences

Code: MIT. LibriSpeech: CC BY 4.0. ESC-50: CC BY-NC 3.0 (non-commercial; mention it if results are shared). Whisper English spelling map vendored from openai-whisper (MIT).
