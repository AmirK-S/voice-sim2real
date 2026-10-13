# Do synthetic callers change which model wins? A small measurement

I got curious about the sim-to-real gap in voice-agent testing: simulated callers do not sound or behave like real ones, so a change can look better in the lab than in production. I measured one slice of it, the acoustic slice, with public models on a laptop. Here is what came out. It is one setup, read speech and simulated phone lines, so it may or may not transfer to your stack.

**Setup in one paragraph.** 250 LibriSpeech sentences read by real people and by 7 TTS voices (Kokoro x2, Piper x2, Meta MMS, macOS x2), plus 122 written caller scripts read by the same TTS voices. Every clip goes through 10 simulated channels (G.711 phone line, Opus, G.722, street and crowd noise at 10 and 5 dB, packet loss, muffled audio, and a "realistic" mix). Three public ASR models stand in for the agent's ear: Whisper large-v3-turbo, Parakeet TDT 0.6B v2, and wav2vec2-base as a weak, in-domain control. About 74,000 scored transcriptions.

## The numbers that would matter for a decision

1. **No single voice confidently flipped the winner between two close models, but averaging voices manufactured certainty.** On real speech, Parakeet beats Whisper only on clean audio (by 0.37 WER points [0.12, 0.67]); on the phone and noisy channels the real difference is within noise. No voice's lab-vs-reality difference survived correction across 28 decisions. The average of all 7 voices, though, declared Parakeet significantly better on all 3 phone and noisy conditions. Averaging many renderings of the same sentence narrows the interval without removing the bias.

2. **The voice changes how big a difference looks.** For a pair with a large real gap (Whisper vs wav2vec2 on the realistic channel, 21.7 points on human speech), the lab said 9.6 points with Kokoro and 35.8 with macOS Daniel. The winner never flipped, but the size did, by 0.37x to 3.72x across cells.

3. **Synthetic speech is not uniformly "too easy".** On clean audio Kokoro was slightly easier than humans (-0.2 to -0.4 points), Piper ryan harder (+0.8 to +1.4) and Meta MMS clearly harder (+3.6 to +4.6, about half of it word-level rendering errors). Where it misleads most is noise: Kokoro showed about half the WER increase that the same noise caused on human speech (Whisper, phone + 10 dB noise: real +2.1 points, Kokoro understated it by 1.2 [0.3, 2.1]).

4. **Emergency keywords under a bad phone line.** On the realistic channel, Whisper caught 95.5 % of keyword instances ("bleeding", "not breathing", "chest pain", "help", ...) and Parakeet 94.6 %; at call level they missed 7 and 9 of 546 renderings. The weak model caught 55 %. Two small surprises: macOS Daniel says "hell" for "help" often enough to make "help" the worst keyword, and Whisper produced text ("thank you", mostly) on 33 of 100 clips of pure noise.

## Two statistics points I found useful

- **Zero misses is not zero risk.** With 105 keyword instances and no miss, the rule of three only bounds the miss rate below about 3 %, and that assumes independent instances, which keywords in the same script are not.
- **A few real recordings go a long way (PPI).** Recording about 25 scripts with real people and running every script through TTS gave a valid interval on real-speech WER, 9 to 23 % narrower than the 25 recordings alone. Human-only intervals at 10 recordings under-covered (80 to 89 %). The gain is modest because per-sentence agreement between synthetic and human results was only 0.5 to 0.7, and it needs scripted calls with a synthetic twin.

## Limits, plainly

- Behaviour is not measured: every voice reads the same script. No panic, hesitation, off-script answers or barge-in. That half of the gap is probably the larger one.
- "Human" is audiobook read speech, which is close to the training data of these models. Gaps against humans are upper bounds; the clean fix is real phone recordings of the caller scripts.
- Offline ASR on single utterances, simulated codecs and noise, Whisper on 4 of 10 channels, and nobody listened to the clips.

Code, tables and all caveats: `README.md` and `results/FINDINGS.md`. Happy to be told where this does not match what you see with real traffic.

## Corrections

**13/10/2026, on item 1.** The interval behind "averaging voices manufactured certainty" resampled speakers but held the 7 voices fixed. Resampling the voices too (BOOT-BOTH, the standard fix when the test items are themselves a sample; Deutsch, Dror and Roth, TACL 2021) makes it about twice as wide. The average still favours Parakeet on the noisy channels, in the same direction as the human estimate, and the lab-minus-real difference is not significant. The 250 human sentences are too few to confirm or rule out a gap of this size: about 1 point is detectable at 80 % power, and the lab says 0.75 to 0.86. So "manufactured certainty" overstates what the data show. The correct reading: the averaged lab claims a winner that the human sample cannot check, and its v2 interval was too narrow. No voice, and no average, ever picked the wrong winner with confidence. Details: `results/followups/literature_checks/NOTES.md`.
