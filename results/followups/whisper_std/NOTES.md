# Whisper standard settings check: speech WER

Code: `scripts/11_whisper_std.py` (steps `run`, `analyze`). Run of 13/10/2026 on the same Mac, Whisper alone on the GPU, about 2 h 10 min for 11,416 clips.

**Why.** Every main-run Whisper WER went through `asr._whisper_batch`, a raw decoder (greedy, temperature 0, no fallback, no no-speech gate) whose 30 s window is padded with the log-mel of digital silence. `mlx_whisper.transcribe` pads clips under 30 s differently, and on noise-only clips that alone moves the phantom rate from 34 to 94 of 100 (`../phantom/v2/NOTES.md`). Its effect on speech had not been measured.

**What.** The same clips as the main run (`channel.prepare`, then `channel.apply` with the sentence-id seed, so every clip is identical to the one the other models heard and pairing is preserved), transcribed with `mlx_whisper.transcribe` 0.4.3 at its defaults plus `language="en"`: the arm `transcribe_defaults_en` of `scripts/09c_whisper_gate_silero.py` (temperature fallback, condition on previous text, no-speech gate on). Called "standard" below; the main-run decoder is "raw". Cache model id `whisper-large-v3-turbo-std`.
- LibriSpeech: 250 sentences x (human + 7 voices) x 4 conditions (clean, phone, phone+noise10, realistic) = 8,000 clips.
- Caller scripts: 122 x 7 voices x the same 4 conditions = 3,416 clips (keyword recall).

**Statistics.** The existing helpers, unchanged: speaker-cluster bootstrap (40 speakers, 10,000 replicates, seed 0, the weights of `05_analyze.py`), BOOT-BOTH for the 7-voice mean (`10_literature_checks.py`, flat and nested voice resampling, seed 1), DiD with Holm over the 28 single-voice decisions, TOST and minimum detectable effect. The raw-decoder rows recomputed here reproduce the published numbers exactly (0.37 [0.12, 0.67]; 5 of 28 nominal, smallest Holm p 0.157; `mean_boot_both.csv` of `literature_checks`; keyword recall 95.5 %, 7 call misses). "Points" are WER percentage points. Whisper minus Parakeet > 0 means Parakeet is better.

## Short answer

The decoder barely changes Whisper's WER (within 0.35 points everywhere, no human difference significant), but it moves the close Whisper vs Parakeet comparison by a few tenths of a point, in a direction that **strengthens two of the report's statistical claims and makes one sentence wrong**:
- Human speech gets slightly easier for Whisper (-0.12 to -0.25 points), TTS voices slightly harder (-0.12 to +0.35). So with standard Whisper the real Parakeet lead shrinks and the lab's Parakeet lead grows.
- **R1a, second half, no longer holds:** 3 of 28 single-voice DiDs now survive Holm (it was 0). All three are voices that exaggerate Parakeet's lead (MMS clean and phone+noise10, Piper ryan clean); none is a flip. No voice and no average picks the wrong winner with confidence under either decoder.
- **R1c:** the 7-voice mean, voices resampled and Holm over 4 conditions, is significant on **3 of 3** noisy conditions (and clean), not 2 of 3.
- **R1d:** the lab offset is larger (+0.34 to +0.73 points) and, with speakers-only intervals, now excludes 0 on phone and phone+noise10; with voices resampled (BOOT-BOTH, the correct interval) it still includes 0 everywhere. "A bias is neither shown nor excluded" holds, with a larger point estimate.
- **R1b:** the human clean gap is 0.25 [0.04, 0.52] instead of 0.37 [0.12, 0.67], still Parakeet; the noisy conditions still cannot tell.
- R2, R3a, R3b and keyword recall hold, with small shifts.

Which decoder is right? Neither is "the" Whisper: standard is what a user who calls `transcribe()` gets. The claims that changed sit near the significance threshold; their direction never changed.

## 1. Human WER, standard vs raw (`wer_std_vs_raw.csv`)

Paired speaker-cluster bootstrap on the same 250 sentences; Holm over the 4 conditions.

| Condition | Raw (main run) | Standard | Standard minus raw | p (Holm) | Transcripts identical after normalization |
|---|---|---|---|---|---|
| clean | 1.99 % [1.42, 2.59] | 1.87 % [1.37, 2.42] | -0.12 [-0.35, +0.05] | 0.87 | 241 / 250 |
| phone | 1.96 % [1.43, 2.54] | 1.85 % [1.35, 2.40] | -0.12 [-0.35, +0.04] | 0.87 | 242 / 250 |
| phone+noise10 | 4.11 % [3.13, 5.22] | 3.86 % [2.98, 4.83] | -0.25 [-0.66, +0.11] | 0.81 | 221 / 250 |
| realistic | 4.32 % [3.45, 5.29] | 4.14 % [3.30, 5.07] | -0.18 [-0.54, +0.16] | 0.87 | 215 / 250 |

- **No significant difference on human speech.** All four point estimates favour the standard pipeline; no empty transcript under either decoder; at most 3 extra errors on one sentence (`sentence_agreement.csv`).
- **TTS voices go the other way, slightly** (28 cells, standard minus raw from -0.12 to +0.35 points; 24 of 28 positive). Nominally significant in 2 cells: Kokoro af_heart realistic +0.23 [+0.07, +0.40] and macOS Daniel phone +0.23 [+0.02, +0.46], neither corrected for 28 tests. MMS has the most changed transcripts (188 to 197 identical of 250).
- Absolute Whisper WERs in the main run are therefore within about 0.35 points of off-the-shelf Whisper on these clips.

## 2. Whisper vs Parakeet with standard Whisper

### Human gap (`human_gap_tost_mde.csv`)

| Condition | Raw: human gap | Standard: human gap | Standard: 90 % CI | TOST within 0.5 points | Standard: MDE at 80 % |
|---|---|---|---|---|---|
| clean | +0.37 [0.12, 0.67] | **+0.25 [0.04, 0.52]**, p 0.027 | [0.07, 0.47] | yes (and different from 0) | 0.34 |
| phone | +0.16 [-0.23, +0.57] | +0.05 [-0.31, +0.41] | [-0.25, +0.35] | **yes** | 0.51 |
| phone+noise10 | +0.49 [-0.15, +1.22] | +0.23 [-0.32, +0.81] | [-0.23, +0.70] | no | 0.79 |
| realistic | +0.60 [-0.04, +1.29] | +0.42 [-0.31, +1.17] | [-0.21, +1.04] | no | 1.06 |

Parakeet still wins on clean, by less. On phone the real difference is now equivalent to 0 within 0.5 points (it was not with the raw decoder). On the two noisy conditions the human sample still cannot tell.

### Single voices: DiD (lab gap minus human gap), Holm over 28 (`whisper_vs_parakeet_by_voice.csv`)

| | Raw (main run) | Standard |
|---|---|---|
| Nominal p < 0.05 | 5 of 28 | 7 of 28 |
| Holm p < 0.05 | **0** (smallest 0.157) | **3** (smallest 0.0056) |
| Confident flips (both CIs exclude 0, opposite signs) | 0 | 0 |

The three Holm survivors under standard Whisper (DiD in points):
- MMS, clean: lab +1.71 [0.98, 2.49] vs human +0.25; DiD +1.46 [0.74, 2.22], Holm p 0.0056 (raw: +0.99 [0.24, 1.84], Holm 0.21).
- MMS, phone+noise10: lab +2.15 [1.27, 3.06] vs human +0.23; DiD +1.92 [0.88, 2.95], Holm p 0.011 (raw: +1.60 [0.52, 2.62], Holm 0.16).
- Piper ryan, clean: lab +1.20 [0.61, 1.85] vs human +0.25; DiD +0.95 [0.34, 1.62], Holm p 0.036 (raw: +0.62 [0.00, 1.30], Holm 1.0).

All three overstate Parakeet's lead, in the human direction; MMS is the voice with the most flagged rendering errors (`../tts_errors/NOTES.md`; its Whisper vs Parakeet distortion was not explained by shared rendering errors). Next closest: Piper ryan phone (Holm 0.055) and MMS phone (0.077).

### The 7-voice mean (`mean_boot_both.csv`)

Mean gap with voices resampled as well as speakers (flat; nested in brackets after), Holm over the 4 conditions:

| Condition | Raw: mean gap, Holm p | Standard: mean gap (flat) | Standard: nested | Standard: Holm p (flat) |
|---|---|---|---|---|
| clean | +0.48 [0.02, 1.02], 0.070 | +0.59 [0.09, 1.21] | [0.07, 1.38] | 0.018 |
| phone | +0.44 [0.02, 0.97], 0.070 | +0.52 [0.11, 1.02] | [0.10, 1.14] | 0.018 |
| phone+noise10 | +0.86 [0.38, 1.47], 0.0016 | +0.96 [0.44, 1.57] | [0.42, 1.74] | 0.0016 |
| realistic | +0.75 [0.20, 1.38], 0.019 | +0.95 [0.36, 1.64] | [0.33, 1.80] | 0.003 |

Lab offset (mean DiD), standard Whisper:

| Condition | Speakers only | Voices resampled (flat) | Raw, speakers only |
|---|---|---|---|
| clean | +0.34 [-0.00, +0.68] (p 0.052) | [-0.19, +0.94] | +0.11 [-0.28, +0.49] |
| phone | **+0.47 [+0.04, +0.89]** | [-0.07, +1.04] | +0.28 [-0.13, +0.69] |
| phone+noise10 | **+0.73 [+0.10, +1.33]** | [-0.05, +1.52] | +0.38 [-0.40, +1.05] |
| realistic | +0.53 [-0.18, +1.27] | [-0.31, +1.44] | +0.15 [-0.48, +0.74] |

- **R1c changes from 2 of 3 to 3 of 3** noisy conditions (all 4 with clean), under both voice schemes.
- **R1d holds in wording but is closer to the line.** With voices resampled the offset includes 0 on every condition, so a bias is still neither shown nor excluded. With the speakers-only interval (the one that the report itself showed to be too narrow) it excludes 0 on phone and phone+noise10. All point estimates rise, to +0.34 to +0.73 points.
- Confident wrong-sign decisions (`decisions_by_voice.csv`): 0 under both decoders.

## 3. Other claims

- **R2 (large gaps).** Real Whisper vs wav2vec2 gap on realistic: 21.9 points (raw 21.7); lab 9.4 (Kokoro) to 35.7 (Daniel) (raw 9.6 to 35.8). The Whisper vs wav2vec2 ratio range becomes 0.35x to 3.43x (raw 0.38x to 3.71x), so the overall LibriSpeech range is 0.35x to 3.72x (the upper extreme is a Parakeet pair, unchanged). Holds.
- **R3a (clean, synthetic minus human, Whisper)** (`gap_and_damage_vs_human.csv`): Kokoro af_heart -0.14 [-0.52, +0.27], am_michael -0.35 [-0.62, -0.07]; Piper ryan +1.71 [0.93, 2.60] (raw +1.39); MMS +5.06 [3.98, 6.16] (raw +4.60). With Parakeet unchanged, the report's ranges become Kokoro -0.1 to -0.4, Piper ryan +0.8 to +1.7, MMS +3.6 to +5.1. Direction holds; the harder voices look a little harder.
- **R3b (Kokoro and noise).** Whisper phone+noise10: real damage +1.99 points (raw +2.13), Kokoro af_heart shows +1.04, error -0.95 [-1.81, -0.11] (raw -1.18 [-2.11, -0.31]). Realistic: -0.74 [-1.53, -0.02] (raw -0.99). Still about half; holds.
- **R4a (keyword recall, caller scripts, 735 instances in 546 renderings)** (`keyword_recall.csv`, sentence-cluster bootstrap): realistic 95.8 % [93.9, 97.6] with 6 call-level misses (raw 95.5 % [93.7, 97.2], 7 misses; Parakeet 94.6 %, 9). Clean 98.0 %, phone 97.0 %, phone+noise10 96.7 % (raw 97.4, 97.4, 96.5). Holds.

## Limits

- One extra decoding path (mlx-whisper 0.4.3, defaults + English). openai-whisper and faster-whisper were not run.
- FLEURS and the rate-matched voices were not re-decoded; their Whisper numbers still use the raw decoder.
- Whisper decoding is not bit-reproducible across batches (`../phantom/v2/NOTES.md`); `transcribe()` decodes one clip at a time, so these transcripts do not depend on batch neighbours.
- The changes in significance are of a few tenths of a point on a human sample that detects about 0.5 to 1 point (MDE above). They show how fragile the close-comparison significance claims are, not that one decoder is the truth.

## Files

- `wer_std_vs_raw.csv`: WER per voice x condition, both decoders, paired difference
- `sentence_agreement.csv`: identical transcripts, more or fewer errors per sentence
- `whisper_vs_parakeet_by_voice.csv`: human gap, lab gap, DiD and Holm per voice, both decoders
- `mean_boot_both.csv`, `mean_vs_human_boot_both.csv`, `human_gap_tost_mde.csv`, `decisions_by_voice.csv`: the `10_literature_checks.py` tables for both decoders (column `whisper`); `mean_gap_p_holm4` is Holm over the 4 conditions
- `gap_and_damage_vs_human.csv`: R3a and R3b quantities for Whisper
- `keyword_recall.csv`: caller-script keyword recall and call-level misses
- `transcripts_std.csv.gz`: every standard transcript with its error count
- `fig_whisper_std.png`: human WER by decoder; Whisper vs Parakeet, human and 7-voice mean, by decoder
- `summary.json`
- Raw shards (gitignored): `results/raw/followups/whisper_std/`
