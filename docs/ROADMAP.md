# RadSpeed Roadmap

Updated: 2026-10-02

This document is the canonical product roadmap for RadSpeed. It is intended to
survive context resets — refer back to this file when picking up work
mid-stream.

## Market context (April 2026)

Primary launch market: **Australia + New Zealand**. US/HIPAA is not a near-term
constraint, so audit logging is still important (medico-legal + AU privacy
principles), but we are not pursuing BAA-grade compliance documentation yet.

**Competitive landscape snapshot:**

- **Rad AI** — enterprise replacement (US dominant). $68M Series C, ~1/3 of US
  health systems. Shipped own STT Dec 2025. Best-in-class follow-up tracking
  (Rad AI Continuity).
- **RADPAIR** — bootstrapped, product-led; agentic voice control of PACS
  viewport (Fovia partnership), Groq-accelerated. Open SDK strategy.
- **Scriptor (rScriptor)** — Windows desktop **overlay** that wraps PowerScribe /
  Dragon / M-Modal. <10 min setup per site. Free **rScriptor Impressions**
  copy-paste tool drives funnel. Unique NLP QA layer (laterality / gender /
  anatomy).
- **FinalScribe** — indie, ~100 users mid-2025. 10-language. Pre-PMF.

**Important AU/NZ-specific note**: most radiologists here run **PowerScribe as
a Windows desktop app**, not the web version. A Chrome-extension overlay has
limited reach. The right overlay path for our market is a **Windows desktop
companion** (window monitoring + clipboard / merge field injection, no audio
tap), in the style of Scriptor — not a browser extension.

## What RadSpeed already has (shipped on `main`)

These are confirmed in code on `main` as of this update — not aspirations.

### Radiologist workstation (web)

- **Current production models** — GPT-6 Sol with low reasoning effort for report generation and
  worksheet vision; Deepgram Nova-3 Medical or AssemblyAI Universal-3 Pro
  Medical Mode for streaming STT; Groq Whisper Large V3 Turbo as the segment
  fallback. Provider model IDs are centralised and routine deployments
  preserve the operator-owned production model secret.
- **Multi-provider streaming STT** — Deepgram Nova-3 Medical, AssemblyAI
  Universal-3 Pro Medical Mode, Groq Whisper Large V3 Turbo segment fallback.
  Automatic mode prefers AssemblyAI Medical when configured, then Deepgram
  Medical, and uses Groq only as the fallback. Signed-in owner choices persist
  across restarts.
  Voice editing by selection works including the short-utterance edge case
  (`web/stt_providers/`, `web/app.py` /ws/transcribe).
- **Voice refinement** — select a passage, speak corrections, regenerate.
- **Vocab learning loop** — repeated edits become per-user keyword boosts in
  the streaming STT call (`/vocab`, `/vocab/add`).
- **Style suggestion learning** — repeated style-pattern edits prompt the
  user to adopt the matching style preference
  (`/api/style-suggestion/apply` and friends).
- **Per-user style preferences** with the most granular control of any
  vendor in this segment: spelling (BR/AM), numerals (Roman/Arabic),
  measurement units + separators + decimal precision, laterality, impression
  style, negation phrasing, date format, paste format
  (`config/config.py`, `llm/format.py`).
- **42 bundled radiology templates + 5 guidelines** (BIRADS, TIRADS, PIRADS,
  LIRADS, Fleischner) — see `templates/` and `guidelines/`.
- **CT CAP staging layout trial** — the staging template can produce either
  component-based findings or compact Chest and Abdomen/Pelvis sections. A
  comparison mode creates both non-exported drafts and lets the radiologist
  choose one before editing, QA, copying or sign-off.
- **Streaming report generation** with patient context block
  (`stream_format_text`, `format_text(patient_context=...)`).
- **Impression-only refresh** in the signed-in report screen. It replaces only
  the current Impression from the existing Findings and applies matching
  Fleischner, BI-RADS, LI-RADS, PI-RADS or TI-RADS guidance by default.
- **Smart paste** — rich / plain / markdown clipboard payloads for
  different RIS text fields, plus one-keystroke "Next Case" reset (Alt+N).
- **Local indication screenshot transcription** — the shared screenshot area
  accepts pasted, chosen or dropped worksheet and indication images. A separate
  action beside worksheet report generation transcribes the indication for
  review, editing and plain-text PowerScribe copy. Recognition uses RadSpeed's
  bundled browser-only OCR and does not upload the screenshot or extracted text.
- **Sonographer worksheet screenshots** — paste, choose or drop up to four
  overlapping worksheet snips. A table-aware vision pass binds measurements
  and marks to their row/column/laterality headers, excludes identifiers and
  unchecked options, then generates a report through a worksheet-native
  template. Blank or unmarked fields remain unknown rather than being filled
  with normal defaults. Images are processed in memory and are not persisted
  by RadSpeed.
- **Keyboard-first reporting loop** — Alt/Option+R record/pause,
  Alt/Option+S stop, Ctrl/Cmd+Enter generate, Ctrl/Cmd+Shift+C copy, and
  Alt/Option+N next case. Shortcuts are shown beside the recorder.
- **Atomic worklist case switching** — moving to another order clears the
  completed case and replaces the whole patient context; unfinished work is
  protected so demographics cannot be mixed across two orders.
- **Compact active-patient focus** — patient / RIS details are closed by default
  behind a disclosure, and loaded demographics collapse to a pinned patient /
  MRN / accession / study banner, keeping identity visible while returning
  vertical space to the report.
- **Local prior comparison** — signed RadSpeed reports for the same MRN are
  surfaced per user. A prior reaches the formatting prompt only after explicit
  selection and is strongly delimited as reference-only context.
- **Radiologist-owned follow-up register** — explicit recommendation language
  is highlighted after generation, but tracking begins only after confirmation.
  Outstanding tasks are user-scoped, due-date aware and auditable.
- **Standardised assessment inserter** — manually selected BI-RADS, PI-RADS
  v2.1, LI-RADS CT/MRI v2018 and ACR TI-RADS categories emit a consistent
  report fragment. It does not calculate a category or management plan.
- **OAuth (Google + Microsoft)** with per-user settings persisted in SQLite.
- **Fracture Lab** — authenticated experimental review with an ephemeral live
  path for one to four radiograph screenshots or DICOM views, selected as
  individual files, a whole folder or a ZIP. DICOM files and ZIP archives are
  decoded and windowed locally, and only their pixel canvases proceed to the
  privacy preview, so the original files, filenames and identifying metadata
  never leave the device. A bundled browser-only text
  recogniser conservatively blacks out identifier-labelled lines and ID-like
  strings before upload, preserves standard laterality/view markers, and
  supports manual drag-to-blackout plus removal of any incorrect blackout. The
  analysis endpoint requires confirmation of the exact cleaned previews. Only
  those generic-filename copies leave the browser; RadSpeed strips metadata as
  a second layer, reviews the views together with the configured frontier
  vision model in one independent pass. That pass identifies the anatomy
  automatically. When it sees a chest or rib study, RadSpeed runs a local
  KAD-512 fracture-probability estimate for each
  original view and displays its public-dataset estimate separately from the
  frontier answer. All per-view open-model scores are returned to the
  signed-in interface. Dedicated wrist studies are automatically routed to a
  local YOLOv9-C locator trained on public GRAZPEDWRI-DX paediatric wrist data;
  its top three boxes covered 99.3% of labelled fractures on the authors'
  patient-held-out 2,029-image test split at IoU 0.5. This does not establish
  adult-wrist performance. Other non-chest studies use a local RT-DETR locator
  trained on public FracAtlas data. Both locators display ranked, untrusted
  attention cues separately; raw scores are neither displayed nor treated as
  fracture probabilities, and their output is not silently fused into the
  frontier answer. The stronger broad SigLIP/MedSigLIP classifier is connected
  through a private, outbound-only Mac worker: RadSpeed places a short-lived
  encrypted job in AWS, the Mac processes it with Apple GPU acceleration when
  awake, and the result is deleted after collection. It reproduces the
  externally evaluated OrthoFrac-XR inference path (AUC 0.892 across 1,132
  images). If the Mac is offline, the interface says so and the frontier and
  other available models continue normally. Fracture Lab returns
  cautious four-state wording, explicitly uncalibrated confidence and suggested
  regions. The page also
  includes the 1,132-case public OrthoFrac-XR benchmark. Direct PACS/DICOMweb
  connectivity remains future separate-service work.

### PACS / RIS / EHR integration (already shipped — needs partner adoption)

This is genuinely bidirectional. The framework is in place; the open work is
deployment and partner sign-on, not new code.

- **HL7 v2.4 ORU^R01 export** — drop final reports to a file-drop inbox for
  RIS integration engines to pick up; atomic writes, collision-safe filenames
  (`llm/hl7_export.py`).
- **HL7 v2.4 ORM^O01 ingestion** — parse inbound orders from integration
  engines, surface them in the worklist; malformed / oversize / mid-write
  files are quarantined (`llm/hl7_import.py`).
- **DICOM Basic Text SR export** — finalised reports written as standard SR
  (SOP Class `1.2.840.10008.5.1.4.1.1.88.11`) for PACS that ingest SR
  directly (`llm/dicom_sr_export.py`).
- **DICOM Modality Worklist (MWL) bridge agent** — on-prem Python agent runs
  C-FIND against the clinic's PACS and pushes orders to the cloud RadSpeed
  instance over HTTPS, avoiding the inbound-firewall problem
  (`agents/voxrad_mwl_agent.py`, `docs/mwl-bridge-agent.md`).
- **FHIR R4 DiagnosticReport export** per report (`llm/fhir_export.py`).
- **FHIR RIS patient lookup** — `/patient/{accession}` queries any FHIR R4
  server for ImagingStudy + Patient (`web/app.py`).
- **In-app worklist panel** — modality filter chips (CT/MR/US/XR/Other),
  waiting-time labels, one-click archive (`/api/hl7/worklist`,
  `/api/hl7/worklist/{order_id}/archive`, `/api/worklist/push`).

### Public free wedge tool (just shipped)

- **`/impressions`** — public, no sign-up. Findings → guideline-aware
  impression in <2s, auto-copy to clipboard. Modality field, optional
  Fleischner/BIRADS/LIRADS/PIRADS/TIRADS toggle, browser-stored style
  preferences. Per-IP hourly rate limit
  (`RADSPEED_IMPRESSIONS_HOURLY_LIMIT`, default 20/hr).
- **`POST /api/impressions/stream`** — public SSE endpoint backing the page.
- **`llm/impressions.py`** — purpose-built impression-only system prompt.
- Knee impressions close with one supported pertinent negative: “Intact
  cruciate, collateral ligaments and menisci.” The full-report and
  impression-only paths exclude injured, uncertain and unassessed structures
  from that statement and name the remaining intact structures.
  These lists use a comma after “cruciate” and one “and” before the final
  structure. Impression post-processing also corrects the double-“and” form
  without adding structures or changing the Findings.

### Clinical governance (Phase 1, just shipped)

- **`web/audit.py`** — `reports` and `audit_log` tables in `users.db`.
  Tamper-evident hash chain catches prev_hash, row_hash, and metadata-vs-
  payload-hash mismatches.
- **Sign-off** — `POST /api/reports/sign-off` locks a report as `final`,
  runs the HL7/SR/FHIR export pipeline, audits each step. UI: green
  "Sign off" button per case.
- **Amendments** — `POST /api/reports/amend` creates a versioned successor
  with a required reason; prior version is preserved.
- **Audit trail UI** — "Audit" button opens a per-accession modal listing
  every version and every event.
- **Status badge** — Draft → Preliminary → Final → Amended.

### NLP QA layer (Phase 2, just shipped)

- **`web/qa.py`** — deterministic laterality / gender / unit-drift /
  modality-anatomy checks plus a conservative transcript-to-report pathology
  coverage check. Worksheet sources suppress a polarity warning only for an
  ambiguous unmarked label that a negative report matches without extra
  finding terms; qualifier mismatches, structured values, measurements and
  explicit positive statuses still warn.
  Flag-only, never rewrites.
- **`POST /api/qa-check`** — runs all checks, returns a flat list of
  severity-tagged flags. Its `source_kind` field accepts `dictation` or
  `worksheet` and defaults to `dictation`. The browser sends `worksheet` only
  for the worksheet workflow and falls back to `dictation` for any unexpected
  internal source state.
- QA runs automatically after generation and again before sign-off, while
  remaining advisory and never rewriting the report. Laterality is inferred
  from body-part labels such as "Right knee". The manual "QA Check" button
  remains available; each flag is dismissible.
- **Opt-in quality samples** — each user can retain de-identified transcript,
  draft and voice-correction pairs for internal checks, evaluation and training
  of RadSpeed's own models. The setting is off by default, samples expire after
  365 days, withdrawal deletes retained pairs, and no pair is shared with
  outside model providers for training.

### Deployment

- **AWS Sydney** — auto-deploy on push to `main` via GitHub Actions and OIDC;
  persistent Lightsail storage for users.db and working data; immutable images
  in ECR; running at `https://radspeed.com.au`.
- **Docker** — `docker compose up -d` for self-hosted.

### Automated quality coverage

- **387 Python tests + 98 Chromium E2E workflows** run before deployment and on
  pull requests. Coverage includes
  silent-failure diagnostics, HL7 file-drop hardening, template selection,
  all bundled template rendering, patient/style prompt construction,
  streaming output cleanup, and the encrypted desktop transcription pipeline.
- The transcription pipeline tests exercise real Fernet encryption/decryption
  and temporary-file cleanup around mocked external model calls. They also
  protect the last good report when formatting fails.
- Browser tests start an isolated mock-mode server and exercise public
  Impressions validation/generation, authenticated audio-segment transcription
  through streamed formatting, worksheet screenshot paste-to-report,
  worksheet-aware QA, indication screenshot OCR/copy, patient-detail disclosure
  defaults, mobile overflow, and authentication rejection.
- A six-case synthetic clinical corpus gates dictated concepts, negation,
  measurements, laterality, and section order. Reference validation runs in CI;
  the deployed production model is evaluated weekly and on demand.

## Roadmap — sequenced

### Phase 0 (just landed): RadSpeed Impressions wedge tool

**Done.** Live at `/impressions` after the next deploy. Cheapest customer
acquisition mechanism in the segment. Validates demand and warms users up
for the full RadSpeed dictation workstation.

### Phase 0.5 (just landed): Windows desktop helper for the Impressions tool

**Done.** AutoHotkey v2 script in `desktop-helper/RadSpeedImpressions.ahk`.
Removes the copy/paste round-trip from the web tool — radiologist selects
findings in PowerScribe One (or any Windows app), presses **Ctrl+I**, and
the impression appears in the IMPRESSION section of the report. Backed by
the existing public `POST /api/impressions/text` endpoint.

Configurable: hotkey, paste mode (goto_impression / after_selection /
replace_selection / at_cursor), and a `JumpKeys` field for the keystrokes
that navigate from FINDINGS to IMPRESSION in the user's specific
PowerScribe template. Default is `goto_impression` with empty JumpKeys,
which falls back gracefully to the `after_selection` behaviour.

The native Tauri companion has now superseded this proof-of-concept for routine
use; the script remains available as a lightweight fallback.

### Phase 1 (just landed): Audit log + sign-off + amendments

**Done.** Medico-legal posture for AU/NZ practices.

- `web/audit.py` — `reports` and `audit_log` tables in `users.db`, both living
  on persistent AWS Lightsail storage.
- Tamper-evident hash chain on `audit_log`. `verify_chain()` catches
  prev_hash mismatches, row_hash mismatches, AND payload-vs-metadata
  mismatches (so retroactive metadata edits without bumping the hash also
  fail verification).
- Explicit sign-off step — `POST /api/reports/sign-off` locks the report
  as `final`, runs the HL7 / SR / FHIR export pipeline against the
  signed text, and writes a `sign_off` audit event (plus per-export
  events). Sign-off requires OAuth (Basic Auth has no `users.id` to FK
  on, so it returns 403).
- Amendment flow — `POST /api/reports/amend` creates a versioned `amended`
  successor pointing at the prior report, with a required reason. The
  prior version is preserved.
- Audit-trail view — `GET /api/audit-log?accession=...` and
  `GET /api/reports?accession=...`. UI: an "Audit" button per case opens
  a modal listing every version + every event for that accession.
- UI status badge cycles **Draft → Preliminary → Final → Amended**.

### Phase 2 (just landed): NLP QA layer

**Done.** PowerScribe One ships its own QA pass, so this is parity rather
than a moat — but it's table stakes for users on Dragon, M-Modal, browser-
only setups, and useful belt-and-braces verification anywhere.

- `web/qa.py` — deterministic checks only (no LLM call):
  - **Laterality** vs ordered side ("left" / "right" / "bilateral").
  - **Gender mismatch** — female-only anatomy (uterus / ovary / cervix) in
    a male patient, and vice versa.
  - **Modality / anatomy mismatch** — flags anatomy from a different region
    than the ordered body part (e.g. "cardiac chambers" on a knee MR).
  - **Unit drift** — a single measurement that mixes mm and cm.
  - **Source coverage** — flags dictated pathology omitted from the report;
    ambiguous unmarked worksheet labels are not treated as positive findings
    only when the negative report matches the same finding terms without extra
    qualifiers. A qualifier mismatch remains eligible for a warning.
- `POST /api/qa-check` — flag-only, never rewrites the report. The optional
  `source_kind` request field is `dictation` by default and accepts `worksheet`;
  the browser falls back to `dictation` for an unexpected internal source state.
- UI: "QA Check" button → flag panel above the report. Each flag is
  dismissible; severity-coloured (error / warning / info).
- The deterministic pass now runs automatically after report generation and
  before sign-off. It infers ordered laterality from body-part context, so
  "Right knee" versus a left-only report is flagged without extra data entry.
- An LLM cross-check pass for things regex can't see (e.g. a lesion that
  legitimately spans both sides) is a future iteration if users ask for it.

### Phase 3 (in progress, Q3-2026): Windows desktop overlay

Reach the AU/NZ PowerScribe **desktop** install base, not just web. The
AHK helper proved demand; the native companion makes it shippable to a
practice rather than a tinkerer.

- **Shipped:** native Tauri 2 tray companion, global hotkey, clipboard capture,
  PowerScribe jump-key paste modes, embedded RadSpeed web view, local settings,
  signed updater artifacts and automatic updates. Version 0.2.29 added native
  Windows RTF lists, so numbered and bulleted conclusions remain editable lists
  in compatible PowerScribe versions. Version 0.2.30 adds native Windows CF_HTML
  beside RTF because PowerScribe versions can select different rich clipboard
  formats. Browser-only copy now states that PowerScribe can remove its formatting.
  The companion includes the focus-free status HUD and production web app.
  Version 0.2.31 starts as a compact, always-on-top reporting controller with
  Record, Pause, Stop, Copy and Next Case actions. The full workstation and
  settings remain available without covering the reporting setup by default.
  Version 0.2.32 starts automatically after Windows sign-in. Closing a window
  hides it to the system tray, so the controller and hotkey remain available.
  Version 0.2.33 preserves bold inline anatomical subheadings through native
  RTF and HTML report copy, while keeping the finding text normal weight.
  Version 0.2.34 keeps the compact controller after sign-in redirects and
  allows resizing from its edges, with a 420 by 200 minimum control area.
  Version 0.2.35 adds a direct Reporting preferences action, including the
  grade and liver-segment numeral choice.
  Version 0.2.36 adds Refine / Stop refine and an editable tail of the live
  transcript to the compact controller. Its default size is 520 by 300 with
  a 420 by 280 minimum. Native report lists use a smaller hanging indent and
  no literal leading space after the list marker.
  The compact transcript shows two lines at the default window size and grows
  when the window is made taller.
  Pausing dictation keeps the voice line flat until recording resumes.
  All report Copy actions omit the EXAM section from every clipboard format,
  including native Windows rich text. The displayed and saved report keep it.
- **Remaining external dependency:** commercial Authenticode / EV certificate
  for a verified Windows publisher identity and removal of the SmartScreen
  “Unknown publisher” warning. Tauri update signing is already configured but
  is not a substitute for Authenticode.
- **Later if demanded by practices:** richer Win32 UI Automation/window-text
  monitoring. The current clipboard approach avoids audio capture and has the
  smaller integration/security surface.
- Pricing: per-radiologist subscription. Ride on top of practice's existing
  PowerScribe contract.

### Phase 4 (foundation shipped; integration next): Critical findings tracking

Rad AI Continuity parity for the AU/NZ market.

- **Shipped:** conservative deterministic detection of explicit follow-up
  language, radiologist confirmation, persistent user-scoped register, optional
  due dates, complete/dismiss actions and audit events.
- **Next:** validated notification channel to the ordering provider and reliable
  reconciliation against incoming orders/results. Notifications are deliberately
  not sent until practice routing, responsibility and escalation rules exist.
- **Later:** hybrid incidental-finding detection and closure analytics.

### Phase 5 (foundation shipped; calculators later): Structured scoring widgets

- **Shipped:** interactive manual category inserter for BI-RADS, PI-RADS v2.1,
  LI-RADS CT/MRI v2018 and ACR TI-RADS. The radiologist remains the classifier;
  the widget standardises wording only.
- **Next:** validated feature-entry calculators one system at a time, with
  source/version display, explainable intermediate scoring and regression cases.
  Fleischner management remains guideline-reference driven until a sufficiently
  complete and clinically reviewed input model is built.

### Phase 6 (shipped 2026-09-22): Australian practice profile

**Done.** Privacy and regulatory posture for selling to AU radiology
practices, driven by the 22 Sep 2026 compliance review (Privacy Act APP 8 /
APP 11, NSW HRIPA, Vic HRA, TGA digital-scribe guidance, RANZCR Standards
v12). Everything is opt-in behind `RADSPEED_PROFILE=practice`; the personal
profile is unchanged.

- `config/practice.py` — profile loader, per-control env overrides, AU
  residency host allow-list, start-up validation that refuses offshore
  endpoints or a missing SSO client.
- `llm/text_client.py` — provider factory. `bedrock-anthropic` adapts the
  Anthropic Messages API on Bedrock (`au.anthropic.claude-sonnet-5`, Sydney +
  Melbourne only) to the chat-completions shape the rest of `llm/` uses.
- `web/stt_providers/deepgram.py` — `api.au.deepgram.com` + `mip_opt_out`.
  AssemblyAI refused under AU residency.
- `llm/format.py` — identifier minimisation: placeholders in the prompt,
  age derived from DOB, prior report scrubbed, values restored on output
  (format, stream, feedback paths).
- `llm/disclosure.py` — "AI-assisted draft" line appended at sign-off and
  amendment, so HL7 / SR / FHIR copies carry it.
- `web/retention.py` — scheduled scrub of report rows and export files,
  audited as `retention_purge`; audit chain still verifies after a purge.
- `web/app.py` — SSO-required mode, Basic-auth lockout, session max age +
  idle timeout, Secure cookie, strict-origin check, `no-store` on private
  responses, research-feature gate (Impression generator, Fracture Lab → 404),
  transcript text removed from logs, `/api/retention` admin endpoints.
- `deploy/aws/Caddyfile` — plain HTTP now redirects except `/health`.
- Docs: `docs/practice-deployment.md`, `deploy/practice.env.example`.

Still to do before a practice pilot (Tier 2 of the review): privacy policy
and terms, DPA + sub-processor list, security statement in the PainTrack /
Qscan format, breach response plan, RANZCR AI-tool evidence pack, TGA scoping
statement, and removing the "HIPAA-ready" / "No PHI stored" claims from the
unpublished `landing/` mockups.

## Explicitly NOT doing (and why)

- **HIPAA BAA documentation** — not relevant for AU/NZ launch. Revisit if/when
  US enters scope. (AU privacy work is Phase 6 above, not this item.)
- **Chrome extension overlay for PowerScribe Web** — AU/NZ market is desktop
  PowerScribe. Browser extension is the wrong bet here.
- **Agentic voice control of the PACS viewport** — RADPAIR's fight; requires
  a deep PACS partnership we don't have.
- **Multi-language support beyond English** — FinalScribe's niche, too narrow
  to anchor brand positioning.
- **Out-marketing Rad AI on follow-up tracking in the US** — they raised $68M
  for that fight. Phase 4 is "table stakes" parity for our market, not a
  US-style enterprise sales motion.

## Notes for future Claude sessions

- `main` is the deployment branch — push to main triggers the AWS Lightsail
  auto-deploy to `radspeed.com.au`.
- The Impressions page lives at `/impressions` (public, no auth).
- The PACS/RIS/EHR integration framework is ALREADY SHIPPED on main (HL7
  ORU/ORM, DICOM SR, MWL bridge, FHIR R4). Don't re-plan it as a future
  feature — it's deployment-and-partners work, not new code.
- Audit log + sign-off + amendments are shipped (Phase 1). Don't re-plan.
- NLP QA layer is shipped (Phase 2, deterministic only). Don't re-plan
  unless extending with an LLM cross-check pass — note that PowerScribe One
  has its own QA out of the box, so extension here is parity-driven, not
  competitive-moat work.
- The user prefers concise updates and direct technical communication.
