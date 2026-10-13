# Follow-ups

After sharing `REPORT.md` I kept pulling on the thread. Four follow-ups ran on the same data and the same laptop, plus a pass of published statistical checks. Here is what came out, with limits. Some of it corrects the report; those corrections are now marked in `REPORT.md` itself. Every number and its source file is in `CLAIMS-STATUS.md`; prior work is in `RELATED-WORK.md` (bracketed numbers).

## 1. Calibration with a few human recordings, and active selection

**Decision-relevant number: with 25 human recordings, PPI covers the held-out human Whisper vs Parakeet difference in 94 to 96 % of splits; the plain 7-voice average covers it in 47 to 86 %.**

- **What I asked.** Can a few real recordings correct a synthetic test set, and can I choose which ones to record?
- **What came out.** No voice tracks the human per-sentence *difference* between two close models (r = -0.15 to +0.27, no CI excludes 0). The plain average is over-precise: its interval is 0.7 to 0.9 points wide. Weighting voices by fidelity keeps that problem. PPI fixes coverage, but with no correlation to exploit, its interval is as wide as the human-only one. For the WER *level*, PPI does help (about 9.5 recordings saved at n = 25).
- **This weakened my own claim.** The report said averaging "manufactured certainty" and did not remove "the bias". The lab-minus-human offset is not significant (+0.15 to +0.38 points, every CI includes 0), so a bias is neither shown nor excluded. Resampling voices as well as speakers (BOOT-BOTH) makes the average's interval about twice as wide. With that and a Holm correction, the average still favours Parakeet on 2 of 3 noisy conditions, in the same direction as the human estimate. The human side is underpowered: about 1 point is detectable at 80 % power, and the lab reports 0.75 to 0.86.
- **Active selection** shrinks the *reported* CI by 15 to 19 % but the actual error by only 1 to 7 %, and coverage drops to 83 to 90 %. Random PPI is the safe default.
- **Limits.** One dataset of 40 speakers; coverage is conditional on them, and the held-out human target is itself noisy (about 150 sentences).
- **Closest prior work.** PPI and PPI++ [34, 35], small-n behaviour [38], the gain ceiling of Dorner et al. [39]; active inference [52, 53]; the averaging issue is Clark's language-as-fixed-effect fallacy [106] and BOOT-BOTH [28].

## 2. FLEURS as a second human set

**Decision-relevant number: against FLEURS readers, every voice except MMS was easier than humans in 60 of 60 cells, and voices understated the cost of noise in 43 of 49.**

- **What I asked.** LibriSpeech is in-domain for these models. Do the conclusions hold with out-of-domain read speech (250 FLEURS en_us sentences, same 7 voices, same channels)?
- **What came out.** Human WER roughly doubles for the strong models and wav2vec2 collapses (16.8 % vs 3.3 % clean). The voices that were harder than LibriSpeech readers become easier than FLEURS readers. Large gaps are always understated (0.26x to 0.92x). So my "gaps against humans are upper bounds" was wrong: LibriSpeech-based gaps are neither upper nor lower bounds.
- **The close comparison.** No voice flips with confidence (0 of 14 after Holm). On the realistic channel, the human sample shows Parakeet ahead (+1.23 [0.36, 2.10]); the average agrees in sign at about a quarter of the size, and its interval includes 0 once voices are resampled.
- **Limits.** FLEURS has no speaker ids, so CIs are sentence-level. The human winner breaks under a 1.42x widening; "60 of 60" holds up to 1.94x (59 of 60 at 2x); damage holds in 36 of 49 at 2x and 35 in the cleanest third of recordings, which carry their own noise. Readers who deviate from the text inflate human WER and make every voice look easier.
- **Closest prior work.** 2M-BELEBELE [1] (same corpus, one TTS, rank swaps in French and Spanish); "TTS is easier than humans" [2, 3, 4, 5]; Lau et al. [58] found TTS harder on LibriSpeech, as I did.

## 3. TTS rendering errors

**Decision-relevant number: removing flagged words cuts the MMS gap by 35 to 45 % (symmetric rule) to 49 to 63 % (stricter rule); MMS stays clearly harder than humans (+3.00 [2.10, 3.95] points for Whisper after symmetric cleaning).**

- **What I asked.** How much of a voice's gap is the voice saying the word wrong, rather than the ASR failing?
- **What came out.** Only MMS and Piper ryan have clearly more flagged words than the human mirror baseline. Cleaning only the synthetic side makes Kokoro look significantly easier than humans; that is a selection artifact, and it disappears with symmetric cleaning. The MMS distortion of Whisper vs Parakeet is not explained by shared rendering errors, so "rendering errors hurt Whisper more" remains untested. The averaged-lab pattern and the Kokoro noise result survive cleaning.
- **Limits.** The flag was checked on paper by its author: 30 items, 77 % [58, 90] plausible, no audio. A blind listening kit is ready and has not been run. Daniel's "hell" for "help" may be a British-voice trait, so it is not counted as an error.
- **Closest prior work.** Lau et al. [58] and CrossASR [59] (cross-ASR detection of TTS faults); 2M-BELEBELE [1] supports the direction for MMS.

## 4. Phantom speech and guards

**Decision-relevant number: off the shelf, Whisper large-v3-turbo wrote text on 95 to 100 % of noise clips after a phone line, not 33 %; a default Silero VAD cut that to 2 to 3 % without dropping any of 1,000 human clips.**

- **What I asked.** What does Whisper do on non-speech with its standard settings, and which guard removes phantom text at the lowest cost on real speech?
- **What came out.** The report's 33 of 100 came from a raw decoder that pads the 30 s window differently from `transcribe()`; changing only the padding takes it to 94. With `mlx_whisper.transcribe` 0.4.3 defaults: 95 / 100 at 4.00 s and 100 / 100 at random lengths, 80 of 95 "thank you". The no-speech gate never fired (no_speech_prob about 1e-10 on every clip). On a larger set of 17 kinds of non-speech, excluding the bot's echoed prompt (real speech): Whisper 86.7 % (raw decoder, probably a low estimate), wav2vec2 29.4 %, Parakeet 16.4 %, which splits into about 30 % on 0.5 to 2 s clips and about 2 % on long ones. So Parakeet is robust on long non-speech, not on short bursts. With Silero at its default, non-echo phantoms fall to 1.5 % (Whisper), 0.2 % (Parakeet) and 1.5 % (wav2vec2). Tuned energy and score rules did worse.
- **Emergency keywords.** No model invented one on non-speech except by transcribing the bot's own prompt. Zero in 1.36 h is a bound, not a rate: at most about 2.2 per hour (95 %, rule of three).
- **Limits.** The default-pipeline arms cover 200 clips, four noise types at one level; "the gate never fires" is shown for mlx-whisper 0.4.3 only (openai-whisper and faster-whisper not tested). Silero ran offline on whole clips, not streaming. Main-run Whisper speech WERs used the same non-standard padding; the effect is not measured (one GPU run on 250 human clips would settle it).
- **Closest prior work.** Barański et al. [71] (large-v3, 40.3 % of non-speech files, "thank you" dominant, Silero the best simple guard); Calm-Whisper [72] (99.97 % on UrbanSound8K, consistent with 95 to 100 %); Careless Whisper [70] for the harm motivation. What may be new: the padding mechanism, the inert gate for turbo, and Parakeet's short-clip rate.

## What still holds, in one paragraph

No single voice confidently flipped Whisper vs Parakeet, on either corpus, and no voice or average picked the wrong winner with confidence. The voice changes how big a large gap looks. Kokoro hides about half of the noise damage on LibriSpeech, and every voice understates it on FLEURS. A synthetic run cannot stand in for the real WER level, the real cost of noise, or a close model comparison; a few real recordings with PPI fix the level, not the comparison.
