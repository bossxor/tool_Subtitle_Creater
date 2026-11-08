# Subtitle Quality Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. User requested immediate implementation; no additional approval handoffs.

**Goal:** Fix false translation success, add tested context handling, preserve quality state, and enable safe existing-subtitle review/update.
**Architecture:** Keep current pipeline and return contracts; optional quality reports flow to GUI and versioned cache. A focused subtitle-review module owns parsing, provenance and atomic backed-up updates; a Qt dialog owns inspection/preview.
**Tech Stack:** Python, standard library, existing PySide6/faster-whisper/requests.
**Spec:** ../specs/2026-11-08-subtitle-quality-design.md

## Global Constraints
- Local execution, no new dependencies. Preserve existing subtitles until explicit selected apply.
- No accuracy claims without reference transcripts. Use installed models; adopt context only after comparison.
- Maintain CLI, cancellation, output formats and existing public return types.

## Review Focus
- English input acronyms must not bypass language validation; test quality validator.
- Verification of existing Korean text must not act as translation; test verify_cues.
- Cached failures must remain visible after restart; test pipeline reports/cache.
- Modified/external captions must not be auto-selected; test provenance and stale-file apply.
- ASS/SMI formatting that cannot be preserved must be read-only; test parser boundaries.

## Tasks
- [x] 1. Add failing quality regression tests. Implement translation_issue(), restricted sound fallback, contextual prompt, and both backend callers. Run new and textfix tests.
- [x] 2. Version translation/segment caches, serialize reports, add reprocess modes and optional run_batch reports. Preserve repeated spoken lines; test cache/status/cancellation.
- [x] 3. Add subtitle parsers, generated-file sidecars, legacy exact-content matching, read-only inspection and backed-up atomic text updates. Test modified files, formats, timestamps and failures.
- [x] 4. Add background review/translation dialog, file selection, issue list, before/after preview, selected apply and original-video regeneration queue. Connect main-window quality results and reprocess mode.
- [x] 5. Run installed-model context comparison and STT reference investigation. Keep settings unchanged unless evidence supports a change. Add reproducible benchmark and results.
- [x] 6. Run relevant unit/GUI suites, capture actual UI, review diff, update README and build distributable while preserving user data.

## Decisions
- Implement inline in the existing workspace; user requested immediate action. Do not move or delete user caches.
- Existing external files are inspectable but never auto-selected. Styled content that cannot be safely patched remains read-only.
- Translation-only updates preserve current times. Full recognition is explicit and needs video.

## Verification (2026-11-08)
- 58 selected unittest cases passed together; textfix assert suite passed.
- Real 40.8s GPU pipeline completed: 10 cues, 0 failed / 0 transliterated, manifest and timestamps checked. This proves execution, not semantic correctness.
- Final executable built; --selftest exit 0 with CUDA/STT load.
- 759 user-cache files preserved: sizes/mtimes unchanged and JSON SHA256 hashes identical.
- Final independent review found no remaining P1/P2 blockers.
- Ruling: retain large-v3 batch/beam 1; sequential beam 5 did not improve reference CER (10.42% vs 10.94%) on the single literary clip.
- Ruling: reject expanded style prompts and multi-line translation after real-model regressions. Ship bounded reference context/current-cue translation, with context-free retries.
- Ruling: preserve read-only handling for unsupported styled/multilingual files; do not flatten and overwrite their format.
- No user subtitles were bulk updated; no commit, push or release publication was performed.
