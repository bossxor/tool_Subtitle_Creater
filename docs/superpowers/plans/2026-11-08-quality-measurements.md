# Quality investigation and bounded fix

Improve Japanese recognition and Korean translation using measured comparisons and targeted regressions. Preserve existing subtitles, model choices and user data.

## Continuation scope

- [x] Reproduce and fix deterministic corruption before changing models: number/repeated-word loss, incorrect short-cue timing, broad blocklist deletion and false polite-register matches.
- [x] Separate glossary allowances from actual source text in translation validation; reject acronym-only sentence loss and narrow meta-response exemptions.
- [x] Check commentary titles as well as stream disposition when choosing dialogue audio.
- [x] Compare Japanese-tuned recognition against large-v3 using the same fixed inputs and decoding settings, including separate reading, Common Voice and broadcast samples.
- [x] Compare TranslateGemma 27B against 12B with fixed semantic cases covering negation, subject, amounts, literal/figurative senses and register.
- [x] Offer supported alternatives, retain previous defaults and document speed/quality tradeoffs.
- [x] Independent review and focused regressions.
- [ ] Complete full-pipeline verification of the newly combined Japanese/27B settings. The run was interrupted; do not treat it as passed.

## Evidence

- Existing large-v3: 20/192 normalized character edits (10.42%) on the 40.8-second Japanese reading sample.
- large-v2: 38/192 (19.79%) on the same sample; not adopted.
- Turbo, beam-search and Kotoba variants did not beat the large-v3 baseline on this sample. This is a small reading benchmark, not an overall subtitle accuracy measurement.
- Hy-MT2, Qwen3/Qwen3.5, longer style instructions and English pivot experiments still produced meaning, subject or register errors. None were adopted as replacement translators.
- Dictionary grounding and unrestricted transcript rewriting have no demonstrated safe improvement. Neither is enabled. Downloaded experimental models remain unused assets.
- Japanese-tuned 760M versus large-v3, CPU int8/beam 1/batch 1: on 48 fixed Common Voice/broadcast clips, 155 versus 215 edits over 1,091 normalized reference characters. The 40.8-second literary sample worsened from 20 to 28 edits over 192 characters. The Japanese model is therefore optional, with large-v3 retained as default. Small samples and transcription/spelling differences prevent generalizing these numbers to overall accuracy.
- TranslateGemma 27B versus 12B, 32 fixed inputs: manual review found improvements in inclusive requests, some idioms and honorifics, plus regressions in nuance. CPU totals were 366.23 versus 115.17 seconds. Keep 12B as default and offer slower 27B explicitly. Greetings remain overly formal in both models.
- Source glossary hints reduced literary-sample edits from 20 to 12, but these hints were chosen from known reference vocabulary. This demonstrates the optional hint path, not unprompted recognition accuracy; an unrelated word also worsened.

## Applied correction

The extraction path previously let ffmpeg choose audio automatically. In multilingual videos this could select an English track while recognition was configured for Japanese. It now selects a track tagged with the source language, prefers non-commentary dialogue and uses the default track when language metadata is absent, with a warning. Missing or unreadable audio fails before publishing a WAV.

Both audio and recognition caches use a new version, preventing reuse of audio from the wrong track. Old cache files and existing subtitle files are preserved. Recognition may consequently need to run again once.

## Checks

- Real two-track MKV: English 440 Hz default plus Japanese 880 Hz. Extracted audio peak is 880 Hz with Japanese selected.
- Regression coverage: language aliases, commentary, untagged defaults, absent audio, failed probing, and atomic extraction failure.
- Combined audio, translation, subtitle review, GUI, cache, folder selection and encoding checks: 81 passing tests.
- Reopening review preserves row-specific glossary choices; targeted Gemma translation retains bounded full-scene context.

## Remaining limits

Raw large-v3 recognition still confuses Japanese homophones on the reference sample. Translation still has idiom and register errors. Structural validation cannot establish semantic correctness. Real drama dialogue, overlapping voices and background music remain insufficiently evaluated. No claim of universally improved recognition or translation accuracy is made.
