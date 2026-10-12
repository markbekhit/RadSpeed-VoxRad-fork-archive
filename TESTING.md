# Testing RadSpeed

RadSpeed uses layered tests so failures are caught at the cheapest reliable
level while the clinical and browser-critical paths still receive end-to-end
coverage. The test data is synthetic and must never contain patient information.

## Test layers

- **Python unit/integration:** `python -m unittest discover -v`
  covers transcription, formatting/templates, clinical safety rules, silent
  failures, worksheet-aware source-omission checks, and file-drop integrations.
- **Browser E2E:** `pytest e2e --browser chromium`
  starts an isolated local server with `VOXRAD_MOCK_MODE=1`, uses HTTP Basic
  Auth, and exercises public Impressions plus authenticated transcription and
  streamed formatting. It also covers keyboard-first reporting, automatic
  laterality-aware QA, atomic worklist case switching, compact patient focus,
  worksheet-aware source coverage, deterministic follow-up prompting,
  streaming-final recovery, transcript omission warnings, quality-retention
  consent, numeral preferences, compact report lists, and manual score
  insertion. External model calls are routed to local mock endpoints.
- **Clinical quality:** `python -m evals.clinical_quality`
  validates the reviewed synthetic references. Use
  `python -m evals.clinical_quality --live --minimum-pass-rate 0.80` only in an
  environment with the configured text-model key. Live evals use six synthetic
  cases and never patient data.
- **Live production QA:** browser smoke testing confirms the deployed UI and
  synthetic Impressions flow after material web changes.

## Microphone filtering

Settings → Microphone enables background filtering by default. Balanced is the
default sensitivity; Low blocks more quiet audio in a busy room, and High accepts
quieter dictation. Preferences save automatically in this device/browser's local
storage and apply when the next recording starts, including the Windows compact
window and Refine. They do not change another workstation's preferences.

The browser requests noise suppression and echo cancellation, without automatic
gain control. A local audio worklet then gates lower-level audio before all
transcription providers receive it. The gate has a 350 ms release and a 2 ms gain
ramp, with no delayed audio buffer to lose at Stop. If the filter cannot start,
recording stops with an error rather than silently sending unfiltered audio.

This is not speaker recognition or competing-speech separation. Another speaker
at a similar level can still pass, particularly while the gate is open. Use a
close microphone, test both your own speech and room speech in Settings, and
reduce filtering if quiet words are lost. The 30-second microphone test displays
levels locally; it never records, uploads, or transcribes audio. No clinical
recordings were used to select or validate these thresholds.

`e2e/test_microphone.py` covers device-local preferences, all four recording
entry points, the real Web Audio graph using synthetic tones, level thresholds,
quiet tails, release, raw-track cleanup, failure without fallback, filter-off
behavior, and the local-only Settings test.

## Setup

```bash
python -m pip install -r requirements-test.txt
python -m playwright install chromium
```

For a CI-equivalent local run:

```bash
python -m unittest discover -v
python -m evals.clinical_quality
pytest e2e --browser chromium --tracing=retain-on-failure
```

## Conventions

- Add a regression test for every bug fix.
- Mock external providers in automated tests; never spend API credits in the
  standard CI gate.
- Assertions must verify user-visible behavior or clinical facts, not merely
  that a value exists.
- Clinical cases protect dictated findings, negation, laterality,
  measurements, and report section order without requiring one exact wording.
- Keep live model evaluation separate from deployment gating because model
  output is probabilistic; its scheduled workflow is an early-warning signal.
