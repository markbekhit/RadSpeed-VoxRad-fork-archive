"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  uiMode: "idle",
  mediaRecorder: null,
  stream: null,          // kept open between segments so mic stays active
  audioChunks: [],
  sessionId: null,
  isRecording: false,
  isPaused: false,
  timerInterval: null,
  timerSeconds: 0,
  audioCtx: null,
  analyser: null,
  animFrameId: null,
  // Silence-triggered real-time transcription
  silenceStart: null,
  isSegmentTranscribing: false,
  // VAD: true only if RMS exceeded the microphone speech threshold
  speechDetected: false,
  // Count of in-flight submitAudioSegment calls — auto-format fires only when 0
  pendingSegments: 0,
  // Streaming STT state
  streamingSupported: false,
  streamingWs: null,
  streamingWorkletNode: null,
  streamingAudioCtx: null,
  confirmedText: "",
  interimText: "",
  // All provider finals received in chronological order. Unlike confirmedText,
  // this is not cleared when the user moves the insertion cursor.
  streamingServerConfirmedText: "",
  streamingBakedServerText: "",
  streamingCursorRepositioned: false,
  // Cursor-aware insertion: text before/after the insert point when recording started.
  // Updated live as the user moves the cursor during streaming.
  streamingBefore: "",
  streamingAfter: "",
  streamingAnchorPos: 0,   // cursor start we last set programmatically
  streamingAnchorEnd: 0,   // cursor end we last set programmatically (for selection highlight)
  streamingSelectedText: "",  // text user selected for replacement (kept visible until speech arrives)
  // Voice editing: {elId, start, end, selectedText} — used for segment (non-streaming) mode
  voiceEditTarget: null,
  // The last report produced by the LLM — stored so we can diff against the
  // user's edits on Copy and detect style-setting drift (impression style, date format).
  reportLlmOutput: "",
  // Selection made DURING an active recording session. Set by the selectionchange
  // listener so the VAD/onstop handler can promote it to a voice-edit on the next cut.
  pendingVoiceEditSelection: null,
  // True once the current report has been copied to the clipboard. Reset when a
  // new report is produced or edited, so Next Case / tab-close can warn about
  // losing an uncopied report.
  reportCopied: false,
  // Set while we've sent {"type":"stop"} and are awaiting session_complete, so
  // an unexpected socket close (or a timeout) during finalization can recover
  // the transcript already in the box instead of hanging on "Processing…".
  streamingFinalizing: false,
  streamingFinalizeTimer: null,
  // Bumped on every Next Case. In-flight streaming/formatting captures the
  // epoch at start and bails if it changes, so a late SSE token or transcript
  // can't land in the next patient's case.
  caseEpoch: 0,
  formatAbort: null,
  ctCapDrafts: {},
  // Dictation mode may complete missing template anatomy with normal defaults.
  // Worksheet mode never does: blank/unmarked worksheet cells remain unknown.
  sourceKind: "dictation",
};

// Suppress our own programmatic cursor moves from triggering the selectionchange handler.
let _suppressStreamingSelChange = false;

const SILENCE_THRESHOLD   = 0.01;   // RMS below this = silence
const SILENCE_DURATION_MS = 800;    // 800 ms pause triggers segment
const MIN_SEGMENT_BYTES   = 12000;  // ignore blobs smaller than this
// Require at least this many ms of cumulative speech-level audio before we
// trust a segment. A single RMS spike (breath, chair creak, keyboard click)
// is not enough — prevents accidental voice-edit replacement on noise.
// 120 ms is safely above any single click/pop but short enough to catch
// one-syllable replacements like "right" or "no" whose voiced-above-threshold
// portion is often only 150-200 ms.
const MIN_SPEECH_MS       = 120;

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function setStatus(msg, type = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = type ? `${type}` : "";
}

// ---------------------------------------------------------------------------
// Vocabulary correction suggestion banner
// ---------------------------------------------------------------------------
// After a voice edit, the server may flag the replacement as a likely ASR
// correction (phonetically similar to the original, not a content change).
// We show a small dismissable banner offering to remember the new spelling.
function showVocabSuggest({ old, new: neu }) {
  const banner = $("vocab-suggest");
  const text = $("vocab-suggest-text");
  if (!banner || !text) return;
  text.innerHTML = `Looks like a spelling fix — remember <b>${escapeHtml(neu)}</b> for future dictations?`;
  banner.dataset.term = neu;
  banner.style.display = "";
}

function hideVocabSuggest() {
  const banner = $("vocab-suggest");
  if (banner) {
    banner.style.display = "none";
    delete banner.dataset.term;
  }
}

async function acceptVocabSuggest() {
  const banner = $("vocab-suggest");
  const term = banner && banner.dataset.term;
  hideVocabSuggest();
  if (!term) return;
  try {
    const resp = await fetch("/vocab/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ term }),
    });
    if (resp.ok) {
      setStatus(`Added "${term}" to your vocabulary.`, "success");
    } else {
      setStatus(`Could not save vocabulary term.`, "error");
    }
  } catch (err) {
    setStatus(`Vocabulary save failed: ${err.message}`, "error");
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ---------------------------------------------------------------------------
// Style-setting suggestion banner
// ---------------------------------------------------------------------------
// After voice edits or on Copy, the server may flag a repeated correction
// pattern (e.g. always changing British→American spelling) and suggest
// updating the matching style setting. The suggestion is shown in a
// dismissable banner; dismiss is permanent (stored server-side).

function showStyleSuggest(suggestion) {
  const banner = $("style-suggest");
  const text = $("style-suggest-text");
  if (!banner || !text || !suggestion) return;
  text.innerHTML = suggestion.message || "";
  banner.dataset.suggestion = JSON.stringify(suggestion);
  banner.style.display = "";
}

function hideStyleSuggest() {
  const banner = $("style-suggest");
  if (banner) {
    banner.style.display = "none";
    delete banner.dataset.suggestion;
  }
}

async function applyStyleSuggest() {
  const banner = $("style-suggest");
  const raw = banner && banner.dataset.suggestion;
  hideStyleSuggest();
  if (!raw) return;
  let suggestion;
  try { suggestion = JSON.parse(raw); } catch { return; }
  try {
    const resp = await fetch("/api/style-suggestion/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(suggestion),
    });
    if (resp.ok) {
      setStatus("Style preference saved.", "success");
    } else {
      setStatus("Could not save style preference.", "error");
    }
  } catch (err) {
    setStatus(`Error saving preference: ${err.message}`, "error");
  }
}

async function dismissStyleSuggest() {
  const banner = $("style-suggest");
  const raw = banner && banner.dataset.suggestion;
  hideStyleSuggest();
  if (!raw) return;
  let suggestion;
  try { suggestion = JSON.parse(raw); } catch { return; }
  fetch("/api/style-suggestion/dismiss", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pattern_key: suggestion.pattern_key }),
  }).catch(() => {});
}

function setUI(mode) {
  state.uiMode = mode;
  // mode: idle | recording | paused | processing | transcribed | formatting | done
  const rec = $("btn-record");
  rec.disabled = !["idle", "transcribed", "done", "recording", "paused"].includes(mode);
  if (mode === "recording") {
    rec.innerHTML = '<span>&#10073;&#10073;</span> Pause';
  } else if (mode === "paused") {
    rec.innerHTML = '<span>&#9654;</span> Resume';
  } else {
    rec.innerHTML = '<span>&#9654;</span> Record';
  }
  $("btn-stop").disabled        = !["recording", "paused"].includes(mode);
  $("btn-format").disabled      = !["transcribed", "done"].includes(mode);
  $("btn-copy").disabled        = mode !== "done";
  const overlayCopy = $("btn-overlay-copy");
  if (overlayCopy) overlayCopy.disabled = mode !== "done";
  syncDesktopRefine(mode);
  scrollDesktopTranscript();
  const btnCopyFromComparison = $("btn-copy-from-comparison");
  if (btnCopyFromComparison) btnCopyFromComparison.disabled = mode !== "done";
  $("btn-edit-toggle").disabled = mode !== "done";
  $("btn-refine").disabled      = mode !== "done" || fbState.isRecording;
  // Phase 1 / 2 buttons — gated on having a report to act on.
  const hasReport = mode === "done" && !!($("report-raw")?.value || "").trim();
  const btnImpression = $("btn-impression");
  if (btnImpression) btnImpression.disabled = !hasReport || _signedReportId !== null;
  const btnQa = $("btn-qa");        if (btnQa) btnQa.disabled = !hasReport;
  const btnSign = $("btn-sign-off"); if (btnSign) btnSign.disabled = !hasReport || _signedReportId !== null;
  const btnAudit = $("btn-audit");  if (btnAudit) btnAudit.disabled = !($("accession")?.value || "").trim();
  const btnAmend = $("btn-amend");  if (btnAmend) btnAmend.style.display = _signedReportId !== null ? "" : "none";
  const refineHint = $("report-refine-hint");
  if (refineHint) refineHint.style.display = mode === "done" ? "" : "none";

  const dot = $("rec-dot");
  if (dot) dot.style.display = (mode === "recording" || mode === "paused") ? "inline-block" : "none";
  if (dot) dot.style.opacity = mode === "paused" ? "0.4" : "1";

  if (mode === "processing" || mode === "formatting") {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    spinner.id = "spinner-tmp";
    const existing = $("spinner-tmp");
    if (existing) existing.remove();
    $("status").prepend(spinner);
    $("status").prepend(" ");
  } else {
    const s = $("spinner-tmp");
    if (s) s.remove();
  }
}

// ---------------------------------------------------------------------------
// Windows desktop compact overlay
// ---------------------------------------------------------------------------
function desktopInvoke(command, args = {}) {
  const invoke = window.__TAURI__?.core?.invoke;
  if (!invoke) return Promise.resolve(false);
  return invoke(command, args).then(() => true);
}

function scrollDesktopTranscript() {
  const transcript = $("transcription");
  if (document.body.classList.contains("desktop-overlay") && transcript
      && document.activeElement !== transcript) {
    transcript.scrollTop = transcript.scrollHeight;
  }
}

function syncDesktopRefine(mode = state.uiMode) {
  const button = $("btn-overlay-refine");
  if (!button) return;
  button.textContent = fbState.isRecording ? "■ Stop refine" : "🎤 Refine";
  button.disabled = fbState.isProcessing || (!fbState.isRecording
    && (mode !== "done" || state.isRecording || !$("report-raw")?.value.trim()));
  const busy = fbState.isRecording || fbState.isProcessing;
  $("btn-overlay-copy").disabled = busy || mode !== "done";
  $("btn-overlay-next").disabled = busy;
  if (busy) $("btn-record").disabled = true;
}

async function setDesktopCompactMode(compact) {
  document.body.classList.toggle("desktop-overlay", compact);
  try {
    await desktopInvoke("cmd_set_compact_mode", { compact });
  } catch (err) {
    console.warn("desktop window resize failed:", err);
  }
  window.setTimeout(initCanvasResize, 120);
  scrollDesktopTranscript();
}

function initDesktopOverlay() {
  const params = new URLSearchParams(window.location.search);
  // OAuth callbacks return to /app without the original query. Identify the
  // native app window as well, including already-installed desktop versions.
  const nativeAppWindow = window.__RADSPEED_DESKTOP_OVERLAY__ === true
    || window.__TAURI__?.window?.getCurrentWindow?.().label === "app";
  if (params.get("desktop") !== "overlay" && !nativeAppWindow) return;

  document.body.classList.add("desktop-host", "desktop-overlay");
  $("btn-desktop-expand")?.addEventListener("click", () => setDesktopCompactMode(false));
  $("btn-desktop-collapse")?.addEventListener("click", () => setDesktopCompactMode(true));
  $("btn-desktop-settings")?.addEventListener("click", async () => {
    try { await desktopInvoke("cmd_show_settings"); }
    catch (err) { console.warn("desktop settings failed:", err); }
  });
  $("btn-overlay-copy")?.addEventListener("click", () => $("btn-copy")?.click());
  $("btn-overlay-next")?.addEventListener("click", () => $("btn-next-case")?.click());
  $("btn-overlay-refine")?.addEventListener("click", () => {
    if (fbState.isRecording) stopFeedback();
    else startFeedback();
  });
}

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
function startTimer() {
  state.timerSeconds = 0;
  $("timer").textContent = "0:00";
  state.timerInterval = setInterval(() => {
    if (state.isPaused) return;  // don't count time while paused
    state.timerSeconds++;
    const m = Math.floor(state.timerSeconds / 60);
    const s = String(state.timerSeconds % 60).padStart(2, "0");
    $("timer").textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(state.timerInterval);
  state.timerInterval = null;
}

// ---------------------------------------------------------------------------
// Waveform + silence detection (Web Audio API AnalyserNode)
// ---------------------------------------------------------------------------
function getRMS(dataArr) {
  let sum = 0;
  for (let i = 0; i < dataArr.length; i++) {
    const v = (dataArr[i] - 128) / 128.0;
    sum += v * v;
  }
  return Math.sqrt(sum / dataArr.length);
}

function startWaveform(stream) {
  const canvas = $("waveform");
  const ctx = canvas.getContext("2d");

  state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = state.audioCtx.createMediaStreamSource(stream);
  state.analyser = state.audioCtx.createAnalyser();
  state.analyser.fftSize = 256;
  source.connect(state.analyser);

  const bufLen = state.analyser.frequencyBinCount;
  const dataArr = new Uint8Array(bufLen);

  function draw() {
    state.animFrameId = requestAnimationFrame(draw);
    if (state.isPaused) {
      // Keep the paused display flat. The microphone can still be live while
      // the recorder is paused, but its audio must not move the voice line.
      dataArr.fill(128);
    } else {
      state.analyser.getByteTimeDomainData(dataArr);
    }

    // ── VAD + Silence detection ───────────────────────────────────────────────────────
    if (state.isRecording && !state.isPaused && !state.isSegmentTranscribing) {
      const rms = getRMS(dataArr);
      const now = Date.now();

      const speechThreshold = RadSpeedMicrophone.speechThreshold(state.stream);
      if (rms >= speechThreshold) {
        // Accumulate elapsed speech time so a single noise spike (breath,
        // keyboard click) can't flip speechDetected on its own.
        if (state._speechLastFrameTs) {
          const dt = now - state._speechLastFrameTs;
          // Cap per-frame delta at 100 ms so a slow frame (background tab)
          // doesn't single-handedly cross the threshold.
          state.speechMs = (state.speechMs || 0) + Math.min(dt, 100);
        }
        state._speechLastFrameTs = now;
        if (state.speechMs >= MIN_SPEECH_MS) {
          state.speechDetected = true;
        }
      } else {
        state._speechLastFrameTs = null;
      }
      if (rms >= Math.min(SILENCE_THRESHOLD, speechThreshold)) {
        state.silenceStart = null;
      } else {
        if (!state.silenceStart) {
          state.silenceStart = now;
        } else if (now - state.silenceStart >= SILENCE_DURATION_MS) {
          const approxBytes = state.audioChunks.reduce((s, c) => s + c.size, 0);
          if (state.speechDetected && approxBytes >= MIN_SEGMENT_BYTES) {
            state.silenceStart = null;
            state.isSegmentTranscribing = true;
            if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
              state.mediaRecorder.stop();
            }
          } else if (!state.speechDetected && approxBytes >= MIN_SEGMENT_BYTES * 3) {
            state.silenceStart = null;
            state.audioChunks = [];
            if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
              state.mediaRecorder.stop();
            }
          }
        }
      }
    }

    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0E1118";
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = "#FFD700";
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    const sliceW = W / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
      const v = dataArr[i] / 128.0;
      const y = (v * H) / 2;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      x += sliceW;
    }
    ctx.lineTo(W, H / 2);
    ctx.stroke();
  }
  draw();
}

function stopWaveform() {
  cancelAnimationFrame(state.animFrameId);
  state.animFrameId = null;
  if (state.audioCtx) {
    state.audioCtx.close().catch(() => {});
    state.audioCtx = null;
    state.analyser = null;
  }
  const canvas = $("waveform");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0E1118";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#FFD700";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// MediaRecorder management
// ---------------------------------------------------------------------------
function _startMediaRecorder() {
  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";

  state.mediaRecorder = new MediaRecorder(state.stream, { mimeType });
  state.audioChunks = [];
  state.speechDetected = false;  // reset VAD flag for new segment
  state.speechMs = 0;             // reset cumulative speech duration
  state._speechLastFrameTs = null;
  state.silenceStart = null;

  state.mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) state.audioChunks.push(e.data);
  };

  state.mediaRecorder.onstop = () => {
    const chunks = state.audioChunks.splice(0); // take and clear
    const hadSpeech = state.speechDetected;

    // Promote a mid-recording selection ONLY when this segment had real speech
    // — i.e. the user actually said the replacement. A silent VAD cut (user
    // paused to think after selecting) must NOT consume the selection or end
    // recording; we keep both alive so the NEXT cut (with their replacement
    // utterance) does the splice.
    const promoteNow = !state.voiceEditTarget && state.pendingVoiceEditSelection && hadSpeech;
    if (promoteNow) {
      // Mark this voice-edit as mid-recording so submitAudioSegment knows to
      // keep the UI in "recording" mode after splicing.
      state.voiceEditTarget = { ...state.pendingVoiceEditSelection, keepRecording: true };
      state.pendingVoiceEditSelection = null;
      console.log("[voice-edit] promoted mid-recording selection to voice-edit:",
        JSON.stringify(state.voiceEditTarget));
    }

    const isVoiceEdit = !!state.voiceEditTarget;
    // Voice edit always treats the segment as final (one utterance, then done).
    const isFinal = !state.isRecording || isVoiceEdit;
    // Mid-recording promotion: the user wants to keep dictating after the
    // splice (PowerScribe style). Restart the mediaRecorder so the mic stays
    // hot while we send the voice-edit segment to the server.
    const keepRecording = isVoiceEdit && state.voiceEditTarget.keepRecording && state.isRecording;

    if ((state.isRecording && !isVoiceEdit) || keepRecording) {
      // Silence-triggered segment OR mid-recording voice-edit — restart
      // recorder immediately so the mic stays active.
      _startMediaRecorder();
    } else {
      // Final segment or explicit voice-edit (started with selection + Record):
      // release the microphone.
      if (isVoiceEdit && state.isRecording) {
        state.isRecording = false;
        stopTimer();
        stopWaveform();
      }
      if (state.stream) {
        RadSpeedMicrophone.stop(state.stream);
        state.stream = null;
      }
    }

    // VAD gate: only send to Whisper if speech was detected in this segment
    if (!isFinal && !hadSpeech) {
      state.isSegmentTranscribing = false;
      return; // discard silent segment silently
    }

    submitAudioSegment(chunks, isFinal);
  };

  state.mediaRecorder.start(250);
}

// ---------------------------------------------------------------------------
// Recording — branches to voice-edit, streaming, or segment mode
// ---------------------------------------------------------------------------
// Record button: starts recording when idle, toggles pause/resume while active.
async function onRecordClick() {
  if (state.isRecording) {
    if (state.isPaused) resumeRecording();
    else pauseRecording();
    return;
  }
  return startRecording();
}

function pauseRecording() {
  if (!state.isRecording || state.isPaused) return;
  state.isPaused = true;
  // Groq segment mode: pause the MediaRecorder so no more audio is collected
  // and the VAD silence-cut doesn't fire. Streaming mode handles pause via
  // the worklet sending zero-filled PCM in place of real audio.
  if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
    try { state.mediaRecorder.pause(); } catch (_) {}
  }
  state.silenceStart = null;
  setUI("paused");
  setStatus("Paused — press Resume to continue.", "");
}

function resumeRecording() {
  if (!state.isRecording || !state.isPaused) return;
  state.isPaused = false;
  if (state.mediaRecorder && state.mediaRecorder.state === "paused") {
    try { state.mediaRecorder.resume(); } catch (_) {}
  }
  state.silenceStart = null;
  setUI("recording");
  setStatus(state.streamingWs
    ? "Streaming STT active — words appear in real time."
    : "Recording… pause briefly to see live transcription.", "active");
}

async function startRecording() {
  if (fbState.isRecording || fbState.isProcessing) return;
  if (state.isRecording) return; // Stop button ends recording
  state.isPaused = false;
  // Once voice is added, use strict dictation QA for the mixed source. This can
  // create an extra warning, but it must not hide a genuine dictated conflict.
  state.sourceKind = "dictation";

  // Belt-and-suspenders: re-read selections at click time.
  // _grabVoiceEdit() on pointerdown is the primary path; this catches keyboard
  // activation and cross-browser edge cases.
  if (!state.voiceEditTarget) {
    // Check report-rendered div first (non-textarea, survives button click)
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed) {
      const reportDiv = $("report-rendered");
      if (reportDiv && reportDiv.contains(sel.anchorNode)) {
        const selectedText = sel.toString().trim();
        if (selectedText) {
          state.voiceEditTarget = { elId: "report-rendered", selectedText };
          console.log("[voice-edit] captured rendered (belt):", JSON.stringify(selectedText));
        }
      }
    }
  }
  if (!state.voiceEditTarget) {
    for (const id of ["transcription", "report-raw"]) {
      const _el = $(id);
      if (_el && _el.selectionStart !== _el.selectionEnd) {
        state.voiceEditTarget = {
          elId: id,
          start: _el.selectionStart,
          end: _el.selectionEnd,
          selectedText: _el.value.slice(_el.selectionStart, _el.selectionEnd),
        };
        console.log("[voice-edit] captured (belt):", id, _el.selectionStart, _el.selectionEnd);
        break;
      }
    }
  }
  console.log("[voice-edit] startRecording — voiceEditTarget:",
    state.voiceEditTarget ? JSON.stringify({
      elId: state.voiceEditTarget.elId,
      start: state.voiceEditTarget.start,
      end: state.voiceEditTarget.end,
      text: state.voiceEditTarget.selectedText,
    }) : "null",
    "streamingSupported:", state.streamingSupported);

  if (state.voiceEditTarget) {
    // Rendered-report edits must use segment mode — no textarea cursor position
    // exists, so streaming insertion doesn't apply.
    if (state.voiceEditTarget.elId === "report-rendered") {
      await startVoiceEditRecording();
    } else if (state.streamingSupported) {
      await startStreamingRecording();
    } else {
      await startVoiceEditRecording();
    }
  } else if (state.streamingSupported) {
    await startStreamingRecording();
  } else {
    await startSegmentRecording();
  }
}

// Voice edit: record one utterance, replace the saved textarea selection.
async function startVoiceEditRecording() {
  // If the previous streaming session's WebSocket is still open waiting for
  // session_complete, close it now. Otherwise, when session_complete arrives
  // mid-voice-edit, _cleanupStreaming() would kill the voice edit microphone.
  if (state.streamingWs) {
    state.streamingWs.close();
    state.streamingWs = null;
  }
  try {
    state.stream = await RadSpeedMicrophone.open();
    startWaveform(state.stream);
    state.silenceStart = null;
    state.isSegmentTranscribing = false;
    state.speechDetected = false;
    state.pendingSegments = 0;
    _startMediaRecorder();
    state.isRecording = true;
    startTimer();
    setUI("recording");
    const label = state.voiceEditTarget.elId === "transcription" ? "transcript" : "report";
    setStatus(`Voice editing ${label} (replacing "${state.voiceEditTarget.selectedText}") — dictate replacement, then pause.`, "active");
  } catch (err) {
    state.voiceEditTarget = null;
    RadSpeedMicrophone.stop(state.stream);
    state.stream = null;
    stopWaveform();
    setStatus(`Could not start microphone: ${err.message}`, "error");
  }
}

async function startSegmentRecording() {
  state.pendingVoiceEditSelection = null; // fresh session — don't inherit old selection
  try {
    state.stream = await RadSpeedMicrophone.open();
    startWaveform(state.stream);
    state.silenceStart = null;
    state.isSegmentTranscribing = false;
    state.speechDetected = false;
    state.pendingSegments = 0;
    _startMediaRecorder();
    state.isRecording = true;
    startTimer();
    setUI("recording");
    setStatus("Recording… pause briefly to see live transcription. Select any word to voice-edit it.", "active");
  } catch (err) {
    RadSpeedMicrophone.stop(state.stream);
    state.stream = null;
    stopWaveform();
    setStatus(
      `Could not start microphone: ${err.message}`,
      "error"
    );
  }
}

// ---------------------------------------------------------------------------
// Streaming STT recording
// ---------------------------------------------------------------------------
async function startStreamingRecording() {
  // If a previous streaming session's WebSocket is still open (e.g. server was
  // slow to send session_complete), detach its handlers and close it now so
  // stale messages can't call _cleanupStreaming() and destroy this new session.
  if (state.streamingWs) {
    const oldWs = state.streamingWs;
    state.streamingWs = null;
    oldWs.onopen    = null;  // prevent stale onopen from firing if connection was still pending
    oldWs.onmessage = null;
    oldWs.onerror   = null;
    oldWs.onclose   = null;
    try { oldWs.close(); } catch (_) {}
  }

  // Capture cursor position / selection BEFORE the mic opens.
  // This makes streaming PowerScribe-style: text is inserted at the cursor,
  // replacing any selection, rather than always appending to the end.
  const tx = $("transcription");
  if (state.voiceEditTarget && state.voiceEditTarget.elId === "transcription") {
    state.streamingBefore = tx.value.slice(0, state.voiceEditTarget.start);
    state.streamingAfter  = tx.value.slice(state.voiceEditTarget.end);
    state.streamingSelectedText = state.voiceEditTarget.selectedText || "";
  } else {
    // No selection — default: append after existing transcript.
    state.streamingBefore = tx.value;
    state.streamingAfter  = "";
    state.streamingSelectedText = "";   // always clear stale selection from any prior session
  }
  state.voiceEditTarget = null;

  const wsToken = document.body.dataset.wsToken || "";
  const proto   = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl   = `${proto}//${location.host}/ws/transcribe?token=${encodeURIComponent(wsToken)}`;

  try {
    state.stream = await RadSpeedMicrophone.open();
  } catch (err) {
    setStatus(`Could not start microphone: ${err.message}`, "error");
    return;
  }

  state.streamingWs = new WebSocket(wsUrl);
  state.streamingWs.binaryType = "arraybuffer";

  state.streamingWs.onopen = async () => {
    // Send config
    state.streamingWs.send(JSON.stringify({
      template_name: $("template-select").value || null,
    }));

    // Set up AudioContext + AudioWorklet
    try {
      state.streamingAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      await state.streamingAudioCtx.audioWorklet.addModule("/static/pcm-worklet.js");
      // Guard: if _cleanupStreaming() was called by onerror during the await above
      // (e.g. Deepgram rejected the connection), streamingAudioCtx is now null.
      // The error status is already set; exit cleanly without crashing.
      if (!state.streamingAudioCtx || !state.stream) {
        stopTimer();
        stopWaveform();
        setUI("idle");
        return;
      }
      const source = state.streamingAudioCtx.createMediaStreamSource(state.stream);
      state.streamingWorkletNode = new AudioWorkletNode(state.streamingAudioCtx, "pcm-processor");
      state.streamingWorkletNode.port.onmessage = (e) => {
        if (!state.streamingWs || state.streamingWs.readyState !== WebSocket.OPEN) return;
        if (state.isPaused) {
          // Send zero-filled PCM of the same length so the provider's session
          // stays alive during a pause but sees only silence.
          state.streamingWs.send(new ArrayBuffer(e.data.byteLength));
          return;
        }
        state.streamingWs.send(e.data);
      };
      source.connect(state.streamingWorkletNode);
      // Connect to destination to keep AudioContext alive in some browsers
      state.streamingWorkletNode.connect(state.streamingAudioCtx.destination);
    } catch (err) {
      setStatus(`AudioWorklet setup failed: ${err.message}`, "error");
      _cleanupStreaming();
      stopTimer();
      stopWaveform();
      setUI("idle");
      return;
    }

    // Also start visual waveform using the same stream
    startWaveform(state.stream);

    state.confirmedText = "";
    state.interimText   = "";
    state.streamingServerConfirmedText = "";
    state.streamingBakedServerText = "";
    state.streamingCursorRepositioned = false;
    state.isRecording   = true;
    startTimer();
    setUI("recording");
    const sdot = $("streaming-dot");
    if (sdot) sdot.style.display = "inline-block";
    setStatus("Streaming STT active — words appear in real time.", "active");
  };

  state.streamingWs.onmessage = (e) => handleStreamingMessage(JSON.parse(e.data));

  state.streamingWs.onerror = (e) => {
    _teardownStreamingUnexpected("Streaming connection error — transcript preserved.");
  };

  state.streamingWs.onclose = (e) => {
    if (e.code === 4001) {
      _teardownStreamingUnexpected("Streaming auth failed. Please reload.");
      return;
    }
    // A drop mid-dictation (network blip, server redeploy) or before the server
    // finalizes would otherwise leave the UI stuck recording/processing with the
    // timer running and every subsequent word silently discarded.
    if (state.isRecording) {
      _teardownStreamingUnexpected("Connection lost — transcript preserved up to the interruption.");
    } else if (state.streamingFinalizing) {
      _teardownStreamingUnexpected("Connection lost before finalizing — using the transcript captured so far.");
    }
  };
}

// Full teardown after an unexpected streaming failure. Preserves whatever text
// is already in the transcription box (the live display wrote it as words
// arrived) and returns the UI to a usable state instead of a stuck spinner.
function _teardownStreamingUnexpected(statusMsg) {
  _cleanupStreaming();   // also clears the finalize flag/timer and stops the mic
  stopTimer();
  stopWaveform();
  const hasText = ($("transcription").value || "").trim();
  setUI(hasText ? "transcribed" : "idle");
  setStatus(statusMsg, "error");
}

function handleStreamingMessage(msg) {
  switch (msg.type) {
    case "interim":
      state.interimText = (msg.text || "").replace(/\s*—\s*/g, " ").trim();
      _updateStreamingDisplay();
      break;
    case "final": {
      const chunk = (msg.text || "").replace(/\s*—\s*/g, " ").trim();
      state.confirmedText = state.confirmedText
        ? state.confirmedText + (chunk ? " " + chunk : "")
        : chunk;
      state.streamingServerConfirmedText = state.streamingServerConfirmedText
        ? state.streamingServerConfirmedText + (chunk ? " " + chunk : "")
        : chunk;
      state.interimText = "";
      _updateStreamingDisplay();
      break;
    }
    case "session_complete": {
      if (msg.session_id) state.sessionId = msg.session_id;
      // The server can recover a final provider interim during shutdown and
      // applies the last ASR correction pass. Prefer that canonical speech for
      // this recording. Previous text is already in streamingBefore/After.
      state.streamingSelectedText = "";
      const canonical = (msg.transcription || "").trim();
      const rawServer = (msg.raw_transcription || canonical).trim();
      let speech;
      if (!state.streamingCursorRepositioned) {
        speech = canonical || state.confirmedText;
      } else if (state.streamingBakedServerText && rawServer.startsWith(state.streamingBakedServerText)) {
        // Cursor moves bake earlier finals into before/after. Insert only the
        // chronological suffix that arrived after the latest move.
        speech = rawServer.slice(state.streamingBakedServerText.length).trim();
      } else {
        // Compatibility/fallback when an older server omits raw_transcription.
        const recovered = (msg.recovered_text || "").trim();
        speech = state.confirmedText + (recovered ? (state.confirmedText ? " " : "") + recovered : "");
      }
      speech = (speech || "").trim();
      const before = state.streamingBefore;
      const after  = state.streamingAfter;
      const sep1 = (before && !/\s$/.test(before) && speech) ? " " : "";
      // Don't insert a space before punctuation (e.g. after voice-editing a word
      // mid-sentence, `after` starts with "." or "," — no space needed).
      const sep2 = (after && !/^\s/.test(after) && !/^[.,;:!?)]/.test(after) && speech) ? " " : "";
      $("transcription").value = before + sep1 + speech + sep2 + after;
      _cleanupStreaming();
      if (speech || before || after) {
        setUI("transcribed");
        if (speech) formatReport();
      } else {
        setUI("idle");
        setStatus("No speech detected.", "error");
      }
      break;
    }
    case "error":
      setStatus(`Streaming error: ${msg.message}`, "error");
      _cleanupStreaming();
      setUI("idle");
      break;
    default:
      break;
  }
}

function _updateStreamingDisplay() {
  const tx      = $("transcription");
  const before  = state.streamingBefore;
  const after   = state.streamingAfter;
  const confirmed = state.confirmedText;
  // Only show interim when inserting at the end of the text (no after-text).
  // This keeps the display equal to "stableText" during mid-text insertion,
  // so cursor positions map 1-to-1 without any offset arithmetic.
  const speech  = after
    ? confirmed
    : confirmed + (state.interimText ? (confirmed ? " " : "") + state.interimText : "");

  // If user selected text for replacement and no speech has arrived yet,
  // re-display the selected text with its highlight preserved.
  if (!speech && state.streamingSelectedText) {
    const sel = state.streamingSelectedText;
    const newValue = before + sel + after;
    const selStart = before.length;
    const selEnd   = selStart + sel.length;
    _suppressStreamingSelChange = true;
    tx.value = newValue;
    tx.selectionStart = selStart;
    tx.selectionEnd   = selEnd;
    state.streamingAnchorPos = selStart;
    state.streamingAnchorEnd = selEnd;
    setTimeout(() => { _suppressStreamingSelChange = false; }, 0);
    return;
  }
  // Clear selected text once speech arrives to replace it
  if (speech && state.streamingSelectedText) {
    state.streamingSelectedText = "";
  }

  const sep1 = (before && !/\s$/.test(before) && speech) ? " " : "";
  const sep2 = (after && !/^\s/.test(after) && !/^[.,;:!?)]/.test(after) && speech) ? " " : "";
  const newValue  = before + sep1 + speech + sep2 + after;
  // Anchor: right after confirmed text (interim trails the cursor; after-text follows).
  const anchorPos = before.length + sep1.length + confirmed.length;

  _suppressStreamingSelChange = true;
  tx.value = newValue;
  tx.selectionStart = tx.selectionEnd = anchorPos;
  scrollDesktopTranscript();
  state.streamingAnchorPos = anchorPos;
  state.streamingAnchorEnd = anchorPos;
  // Release the suppress flag after the browser has dispatched any
  // selectionchange events queued by the value/cursor assignment above.
  setTimeout(() => { _suppressStreamingSelChange = false; }, 0);
}

function _cleanupStreaming() {
  const sdot = $("streaming-dot");
  if (sdot) sdot.style.display = "none";

  if (state.streamingWorkletNode) {
    try { state.streamingWorkletNode.disconnect(); } catch (_) {}
    state.streamingWorkletNode = null;
  }
  if (state.streamingAudioCtx) {
    state.streamingAudioCtx.close().catch(() => {});
    state.streamingAudioCtx = null;
  }
  if (state.stream) {
    RadSpeedMicrophone.stop(state.stream);
    state.stream = null;
  }
  if (state.streamingWs) {
    if (state.streamingWs.readyState === WebSocket.OPEN ||
        state.streamingWs.readyState === WebSocket.CONNECTING) {
      state.streamingWs.close();
    }
    state.streamingWs = null;
  }
  if (state.streamingFinalizeTimer) {
    clearTimeout(state.streamingFinalizeTimer);
    state.streamingFinalizeTimer = null;
  }
  state.streamingFinalizing = false;
  state.isRecording   = false;
  state.isPaused      = false;
  state.confirmedText   = "";
  state.interimText     = "";
  state.streamingServerConfirmedText = "";
  state.streamingBakedServerText = "";
  state.streamingCursorRepositioned = false;
  state.streamingBefore = "";
  state.streamingAfter  = "";
  state.streamingAnchorPos = 0;
  state.streamingAnchorEnd = 0;
  state.streamingSelectedText = "";
}

function stopRecording() {
  if (!state.isRecording) return;

  // If we stopped while paused, resume the MediaRecorder first so its onstop
  // fires with any buffered audio. Clear the flag so worklet frames stop being
  // zeroed out and the VAD can evaluate normally.
  if (state.isPaused) {
    state.isPaused = false;
    if (state.mediaRecorder && state.mediaRecorder.state === "paused") {
      try { state.mediaRecorder.resume(); } catch (_) {}
    }
  }

  if (state.streamingSupported && state.streamingWs) {
    stopStreamingRecording();
    return;
  }

  // Segment (Groq Whisper) path
  state.isRecording = false;
  stopTimer();
  stopWaveform();
  setUI("processing");
  setStatus("Processing remaining audio…", "active");

  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop(); // onstop handles stream teardown + final transcription
  } else {
    // Edge case: recorder already stopped mid-silence-segment restart.
    // Stream cleanup happened (or will happen) in that onstop. Just update UI.
    if (state.stream) {
      RadSpeedMicrophone.stop(state.stream);
      state.stream = null;
    }
    state.isSegmentTranscribing = false;
    // Defer to _maybeAutoFormat — a mid-recording segment may still be in-flight
    _maybeAutoFormat();
  }
}

function stopStreamingRecording() {
  state.isRecording = false;
  stopTimer();
  stopWaveform();
  setUI("processing");
  setStatus("Processing final audio…", "active");

  // Disconnect worklet so no more audio frames are sent
  if (state.streamingWorkletNode) {
    try { state.streamingWorkletNode.disconnect(); } catch (_) {}
  }

  // Tell server to flush and finalize
  if (state.streamingWs && state.streamingWs.readyState === WebSocket.OPEN) {
    state.streamingFinalizing = true;
    state.streamingWs.send(JSON.stringify({ type: "stop" }));
    // Don't wait forever for session_complete — if the server dies after we
    // send stop, fall back to the transcript already captured in the box.
    if (state.streamingFinalizeTimer) clearTimeout(state.streamingFinalizeTimer);
    state.streamingFinalizeTimer = setTimeout(() => {
      if (state.streamingFinalizing) {
        _teardownStreamingUnexpected("Finalizing timed out — using the transcript captured so far.");
      }
    }, 10000);
  } else {
    // WS already closed; treat as no speech
    _cleanupStreaming();
    setUI("idle");
    setStatus("No speech detected.", "error");
  }
}

// ---------------------------------------------------------------------------
// Auto-format gate: fire only when recording stopped AND no segments in-flight
// ---------------------------------------------------------------------------
function _maybeAutoFormat() {
  if (state.isRecording || state.pendingSegments > 0) return;
  const hasText = $("transcription").value.trim();
  if (hasText) {
    setUI("transcribed");
    formatReport();
  } else {
    setUI("idle");
    setStatus("No speech detected.", "error");
  }
}

// ---------------------------------------------------------------------------
// Transcribe segment (called from onstop for both mid-recording and final)
// ---------------------------------------------------------------------------
async function submitAudioSegment(chunks, isFinal) {
  // Capture voice edit target at call time — async gaps could clear state later.
  const editTarget = isFinal ? state.voiceEditTarget : null;
  console.log("[voice-edit] submitAudioSegment — isFinal:", isFinal,
    "editTarget:", editTarget ? JSON.stringify({ elId: editTarget.elId, start: editTarget.start, end: editTarget.end, text: editTarget.selectedText }) : "null");

  state.pendingSegments++;
  const blob = new Blob(chunks, { type: "audio/webm" });

  if (blob.size < MIN_SEGMENT_BYTES) {
    state.isSegmentTranscribing = false;
    state.pendingSegments--;
    if (!editTarget) {
      _maybeAutoFormat();
    } else {
      const wasMidRecording = !!editTarget.keepRecording;
      state.voiceEditTarget = null;
      if (wasMidRecording && state.isRecording) {
        setStatus("No speech detected — edit unchanged. Keep dictating.", "active");
      } else {
        setUI(_inferUIMode());
        setStatus("No speech detected — edit unchanged.", "error");
      }
    }
    return;
  }

  const formData = new FormData();
  formData.append("audio", blob, "segment.webm");
  if (editTarget) {
    // Voice edit: pass the text before the selection as Whisper context rather
    // than the vocabulary list — prevents Whisper hallucinating prompt completions.
    const el = $(editTarget.elId);
    // editTarget on report-rendered has no start/end; only the selectedText string.
    if (typeof editTarget.start === "number" && el && "value" in el) {
      const before = el.value.slice(
        Math.max(0, editTarget.start - 300), editTarget.start
      ).trim();
      formData.append("whisper_prompt", before);
    } else {
      formData.append("whisper_prompt", "");
    }
    if (editTarget.selectedText) {
      formData.append("selected_text", editTarget.selectedText);
    }
  } else {
    const templateName = $("template-select").value;
    if (templateName) formData.append("template_name", templateName);
  }

  try {
    const resp = await fetch("/transcribe", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      setStatus(`Transcription error: ${err.detail}`, "error");
      // On a voice edit, clear the target and unlock the UI — otherwise it stays
      // stuck in "processing" and the stale target hijacks the next recording.
      if (editTarget) { state.voiceEditTarget = null; setUI(_inferUIMode()); }
      return;
    }
    const data = await resp.json();
    if (data.session_id) state.sessionId = data.session_id;

    const newText = data.transcription ? data.transcription.trim() : "";
    console.log("[voice-edit] /transcribe returned:", JSON.stringify(newText), "editTarget still:", editTarget ? editTarget.elId : "null");

    if (editTarget) {
      const wasMidRecording = !!editTarget.keepRecording;

      // If the server returned empty text (genuine silence, or hallucination
      // filter rejected what Whisper fabricated), preserve the user's
      // selection unchanged. Never delete highlighted text on empty.
      if (!newText) {
        state.voiceEditTarget = null;
        if (wasMidRecording && state.isRecording) {
          setStatus("No speech detected — edit unchanged. Keep dictating.", "active");
        } else {
          setUI(_inferUIMode());
          setStatus("No speech detected — edit unchanged.", "error");
        }
        return;
      }

      if (editTarget.elId === "report-rendered") {
        // Voice edit on the rendered report: find-and-replace in raw markdown.
        const raw = $("report-raw").value;
        const updated = _spliceRenderedEdit(raw, editTarget.selectedText, newText);
        state.voiceEditTarget = null;
        if (updated !== null) {
          $("report-raw").value = updated;
          $("report-rendered").innerHTML = renderMarkdown(updated);
          _markReportChanged();
          setUI(_inferUIMode());
          setStatus("Voice edit applied.", "success");
          runQaCheck({ quiet: true });
          if (data.suggest_vocab) showVocabSuggest(data.suggest_vocab);
          if (data.suggest_setting) showStyleSuggest(data.suggest_setting);
        } else {
          setUI(_inferUIMode());
          setStatus("Could not locate selected text in report — edit manually.", "error");
        }
      } else {
        // Voice edit on a textarea: splice at selection position.
        const el = $(editTarget.elId);
        const { start, end } = editTarget;
        el.value = el.value.slice(0, start) + newText + el.value.slice(end);
        el.selectionStart = el.selectionEnd = start + newText.length;
        el.focus();
        if (editTarget.elId === "report-raw") {
          $("report-rendered").innerHTML = renderMarkdown(el.value);
          _markReportChanged();
          runQaCheck({ quiet: true });
        }
        state.voiceEditTarget = null;
        if (wasMidRecording && state.isRecording) {
          setStatus("Voice edit applied — keep dictating.", "active");
        } else {
          setUI(_inferUIMode());
          setStatus("Voice edit applied.", "success");
        }
        if (data.suggest_vocab) showVocabSuggest(data.suggest_vocab);
      }
    } else {
      if (newText) {
        const existing = $("transcription").value.trim();
        $("transcription").value = existing ? existing + " " + newText : newText;
        scrollDesktopTranscript();
      }
      if (state.isRecording) {
        setStatus("Recording… pause briefly to see live transcription.", "active");
      }
    }
  } catch (err) {
    setStatus(`Transcription error: ${err.message}`, "error");
    if (editTarget) { state.voiceEditTarget = null; setUI(_inferUIMode()); }
  } finally {
    state.isSegmentTranscribing = false;
    state.pendingSegments--;
    if (!editTarget) _maybeAutoFormat();
  }
}

// Find selectedText in raw markdown and replace with replacement.
// Returns the updated string, or null if the text was not found OR is ambiguous.
//
// Radiology reports repeat short phrases constantly ("right", "normal", "No
// focal lesion."). Blindly replacing the FIRST occurrence when the radiologist
// selected a later one silently rewrites the wrong sentence — a laterality
// error exactly where it matters most. When the selection is not unique we
// refuse and let the caller tell the user to edit manually, rather than guess.
function _countOccurrences(haystack, needle) {
  if (!needle) return 0;
  let n = 0, i = 0;
  while ((i = haystack.indexOf(needle, i)) !== -1) { n++; i += needle.length; }
  return n;
}

function _spliceRenderedEdit(raw, selectedText, replacement) {
  // Verbatim match — only act if it is unambiguous.
  const exact = _countOccurrences(raw, selectedText);
  if (exact === 1) {
    const idx = raw.indexOf(selectedText);
    return raw.slice(0, idx) + replacement + raw.slice(idx + selectedText.length);
  }
  if (exact > 1) return null;  // ambiguous — don't guess which one
  // Case-insensitive fallback (capitalisation post-processing may have changed case)
  const lower = raw.toLowerCase();
  const lowerSel = selectedText.toLowerCase();
  if (_countOccurrences(lower, lowerSel) !== 1) return null;
  const idxLow = lower.indexOf(lowerSel);
  return raw.slice(0, idxLow) + replacement + raw.slice(idxLow + selectedText.length);
}

// Infer the correct UI mode from current content after a voice edit.
function _inferUIMode() {
  if ($("report-raw").value.trim()) return "done";
  if ($("transcription").value.trim()) return "transcribed";
  return "idle";
}

// ---------------------------------------------------------------------------
// Case focus — compact patient identity + explicitly selected local prior
// ---------------------------------------------------------------------------
let _priorReports = [];
let _priorRefreshToken = 0;

function updatePatientSummary({ collapse = false } = {}) {
  const summary = $("patient-summary");
  if (!summary) return;
  const name = ($("patient-name")?.value || "").trim();
  const dob = ($("patient-dob")?.value || "").trim();
  const mrn = ($("patient-id")?.value || "").trim();
  const accession = ($("accession")?.value || "").trim();
  const study = [$("modality")?.value, $("body-part")?.value]
    .map((v) => (v || "").trim()).filter(Boolean).join(" ");
  const bits = [name, dob ? `DOB ${dob}` : "", mrn ? `MRN ${mrn}` : "",
    accession ? `#${accession}` : "", study].filter(Boolean);
  summary.textContent = bits.length ? bits.join(" · ") : "No case loaded";
  const details = $("patient-context-details");
  if (collapse && details && bits.length) details.open = false;
}

function _resetPriors({ invalidate = true } = {}) {
  if (invalidate) _priorRefreshToken++;
  _priorReports = [];
  const panel = $("prior-panel");
  const select = $("prior-select");
  const preview = $("prior-preview");
  if (select) select.innerHTML = '<option value="">Do not include a prior</option>';
  if (preview) { preview.textContent = ""; preview.style.display = "none"; }
  if (panel) panel.style.display = "none";
}

async function refreshPriorReports() {
  const patientId = ($("patient-id")?.value || "").trim();
  const accession = ($("accession")?.value || "").trim();
  const token = ++_priorRefreshToken;
  _resetPriors({ invalidate: false });
  if (!patientId) return;
  try {
    const qs = new URLSearchParams({ patient_id: patientId });
    if (accession) qs.set("exclude_accession", accession);
    const resp = await fetch(`/api/reports/priors?${qs}`);
    if (!resp.ok || token !== _priorRefreshToken) return;
    const data = await resp.json();
    if (token !== _priorRefreshToken) return;
    _priorReports = data.reports || [];
    if (!_priorReports.length) return;
    const select = $("prior-select");
    for (const report of _priorReports) {
      const option = document.createElement("option");
      option.value = String(report.id);
      const date = report.signed_at || report.created_at || "Unknown date";
      const study = [report.modality, report.body_part].filter(Boolean).join(" ");
      option.textContent = [date.slice(0, 10), study, report.accession ? `#${report.accession}` : ""]
        .filter(Boolean).join(" · ");
      select.appendChild(option);
    }
    $("prior-panel").style.display = "";
  } catch (_) { /* Prior lookup is optional; reporting must remain available. */ }
}

function onPriorSelected() {
  const selected = _priorReports.find((r) => String(r.id) === $("prior-select").value);
  const preview = $("prior-preview");
  if (!selected) {
    preview.textContent = "";
    preview.style.display = "none";
    return;
  }
  preview.textContent = selected.report_text || "";
  preview.style.display = "";
}

function _selectedPriorPayload() {
  const selected = _priorReports.find((r) => String(r.id) === $("prior-select")?.value);
  return selected ? {
    comparison_report: selected.report_text || null,
    comparison_date: selected.signed_at || selected.created_at || null,
  } : { comparison_report: null, comparison_date: null };
}

// ---------------------------------------------------------------------------
// FHIR patient lookup — populates patient context fields from RIS
// ---------------------------------------------------------------------------
async function lookupPatient() {
  const accession = $("accession").value.trim();
  if (!accession) {
    setStatus("Enter an accession number first.", "error");
    return;
  }
  const btn = $("btn-lookup");
  btn.disabled = true;
  const prevText = btn.textContent;
  btn.textContent = "…";
  try {
    const resp = await fetch(`/patient/${encodeURIComponent(accession)}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      setStatus(`FHIR lookup: ${err.detail}`, "error");
      return;
    }
    const data = await resp.json();
    if (data.patient_name)        $("patient-name").value        = data.patient_name;
    if (data.patient_dob)         $("patient-dob").value         = data.patient_dob;
    if (data.patient_id)          $("patient-id").value          = data.patient_id;
    if (data.modality)            $("modality").value            = data.modality;
    if (data.body_part)           $("body-part").value           = data.body_part;
    if (data.referring_physician) $("referring-physician").value = data.referring_physician;
    updatePatientSummary({ collapse: true });
    await refreshPriorReports();
    setStatus("Patient details populated from FHIR RIS.", "success");
  } catch (err) {
    setStatus(`FHIR lookup failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

// ---------------------------------------------------------------------------
// HL7 worklist — pending orders from the RIS inbox
// ---------------------------------------------------------------------------
let _worklistOrders = [];
let _worklistFilter = "";
let _activeWorklistOrderId = "";
const _KNOWN_MODALITIES = new Set(["CT", "MR", "US", "XR"]);

function _waitingTime(receivedAt) {
  if (!receivedAt) return "";
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - receivedAt));
  if (secs < 60)     return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60)     return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)      return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function _worklistLabel(order) {
  const bits = [];
  if (order.modality || order.procedure) {
    bits.push(order.modality ? `${order.modality}${order.body_part ? " " + order.body_part : ""}` : order.procedure);
  }
  if (order.patient_name) bits.push(order.patient_name);
  if (order.accession)    bits.push(`#${order.accession}`);
  const wait = _waitingTime(order.received_at);
  if (wait) bits.push(wait);
  return bits.join(" · ") || order.order_id || "(unnamed order)";
}

function _worklistMatchesFilter(order) {
  if (!_worklistFilter) return true;
  const mod = (order.modality || "").toUpperCase();
  if (_worklistFilter === "OTHER") return !_KNOWN_MODALITIES.has(mod);
  return mod === _worklistFilter;
}

function _renderWorklistOptions() {
  const select = $("worklist-select");
  const count = $("worklist-count");
  if (!select) return;
  const prev = select.value;
  const filtered = _worklistOrders.filter(_worklistMatchesFilter);
  select.innerHTML = '<option value="">Select a pending order…</option>';
  for (const order of filtered) {
    const opt = document.createElement("option");
    opt.value = order.order_id;
    opt.textContent = _worklistLabel(order);
    select.appendChild(opt);
  }
  // Preserve selection across filter changes when the order is still visible.
  if (prev && filtered.some((o) => o.order_id === prev)) {
    select.value = prev;
  } else {
    select.value = "";
    $("btn-worklist-archive").disabled = true;
  }
  const total = _worklistOrders.length;
  if (count) {
    if (_worklistFilter && filtered.length !== total) {
      count.textContent = `(${filtered.length} / ${total})`;
    } else {
      count.textContent = total ? `(${total})` : "(empty)";
    }
  }
}

async function refreshWorklist() {
  const panel = $("worklist-panel");
  const select = $("worklist-select");
  if (!panel || !select) return;
  try {
    const resp = await fetch("/api/hl7/worklist");
    if (!resp.ok) {
      panel.style.display = "none";
      return;
    }
    const data = await resp.json();
    _worklistOrders = data.orders || [];
    // Hide the panel entirely when no inbox is configured and no orders exist.
    if (!data.enabled && _worklistOrders.length === 0) {
      panel.style.display = "none";
      return;
    }
    panel.style.display = "";
    _renderWorklistOptions();
  } catch (err) {
    console.warn("Worklist refresh failed:", err);
  }
}

function _setWorklistFilter(modality) {
  _worklistFilter = modality || "";
  document.querySelectorAll(".worklist-chip").forEach((chip) => {
    chip.classList.toggle("active", (chip.dataset.modality || "") === _worklistFilter);
  });
  _renderWorklistOptions();
}

function applyWorklistOrder() {
  const select = $("worklist-select");
  const archiveBtn = $("btn-worklist-archive");
  if (!select) return;
  const id = select.value;
  archiveBtn.disabled = !id;
  if (!id) {
    _activeWorklistOrderId = "";
    return;
  }
  if (id === _activeWorklistOrderId) return;
  const order = _worklistOrders.find((o) => o.order_id === id);
  if (!order) return;

  if (fbState.isRecording) {
    select.value = _activeWorklistOrderId;
    archiveBtn.disabled = !_activeWorklistOrderId;
    setStatus("Stop report refinement before switching orders.", "error");
    return;
  }

  // Changing the selected order is a case transition, not a partial form fill.
  // Keeping non-empty fields here can silently combine patient A's demographics
  // with patient B's accession. Protect unfinished work, then reset atomically.
  const hasTranscript = !!($("transcription")?.value || "").trim();
  const hasReport = !!($("report-raw")?.value || "").trim();
  if (state.isRecording || hasTranscript || hasReport) {
    const protectedWork = state.isRecording
      || (hasReport && !state.reportCopied && !_signedReportId)
      || (hasTranscript && !hasReport);
    if (protectedWork) {
      const ok = window.confirm(
        "This case has unfinished work.\n\n" +
        "Switch orders and discard it?"
      );
      if (!ok) {
        select.value = _activeWorklistOrderId;
        archiveBtn.disabled = !_activeWorklistOrderId;
        return;
      }
    }
    nextCase({ force: true });
    // nextCase clears the dropdown as part of its atomic reset.
    select.value = id;
    archiveBtn.disabled = false;
  }

  const setField = (elId, val) => {
    const el = $(elId);
    if (el) el.value = val || "";
  };
  setField("patient-name",         order.patient_name);
  setField("patient-dob",          order.patient_dob);
  setField("patient-id",           order.patient_id);
  setField("accession",            order.accession);
  setField("modality",             order.modality);
  setField("body-part",            order.body_part || order.procedure);
  setField("referring-physician",  order.referring_physician);
  _activeWorklistOrderId = id;
  const auditBtn = $("btn-audit");
  if (auditBtn) auditBtn.disabled = !(order.accession || "").trim();
  updatePatientSummary({ collapse: true });
  refreshPriorReports();
  setStatus(`Loaded order ${order.accession || order.order_id}`, "active");
}

async function archiveWorklistOrder() {
  const select = $("worklist-select");
  if (!select || !select.value) return;
  const id = select.value;
  try {
    const resp = await fetch(`/api/hl7/worklist/${encodeURIComponent(id)}/archive`, { method: "POST" });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    setStatus(`Order ${id} archived.`, "active");
    await refreshWorklist();
  } catch (err) {
    setStatus(`Archive failed: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Indication screenshot transcription (browser-only OCR)
// ---------------------------------------------------------------------------
const INDICATION_MAX_IMAGE_BYTES = 8 * 1024 * 1024;
let _indicationWorkerPromise = null;
let _indicationBusy = false;

function _setIndicationStatus(message, type = "") {
  const status = $("indication-status");
  if (!status) return;
  status.textContent = message;
  status.className = type ? `field-note ${type}` : "field-note";
}

function _cleanIndicationText(value) {
  let text = String(value || "")
    .replace(/\r\n?/g, "\n")
    .replace(/([A-Za-z])-\s*\n\s*(?=[a-z])/g, "$1")
    .replace(/\s*\n+\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  text = text.replace(
    /^(?:clinical\s+(?:details|history|indication)|indication|history|reason\s+for\s+(?:exam|examination)|request\s+details)\s*[:\-]\s*/i,
    ""
  );
  return text.trim();
}

function _renderIndicationBusy(isBusy) {
  _indicationBusy = isBusy;
  const zone = $("worksheet-drop-zone");
  if (zone) zone.classList.toggle("is-busy", isBusy);
  _renderWorksheetPreviews();
}

async function _getIndicationWorker() {
  if (!_indicationWorkerPromise) {
    if (!window.Tesseract || !window.Tesseract.createWorker) {
      throw new Error("Local text recognition did not load. Refresh and try again.");
    }
    _indicationWorkerPromise = window.Tesseract.createWorker("eng", 1, {
      workerPath: "/static/vendor/tesseract/worker.min.js",
      langPath: "/static/vendor/tesseract/",
      corePath: "/static/vendor/tesseract/tesseract-core-lstm.wasm.js",
      logger: (message) => {
        if (!_indicationBusy || message.status !== "recognizing text") return;
        const progress = Math.max(0, Math.min(100, Math.round((message.progress || 0) * 100)));
        _setIndicationStatus(`Reading screenshot locally… ${progress}%`);
      },
    });
  }
  return _indicationWorkerPromise;
}

async function transcribeIndicationImages(files) {
  if (_indicationBusy) return;
  const images = Array.from(files || []);
  if (!images.length) return;
  const result = $("indication-result");
  const text = $("indication-text");
  const copy = $("btn-indication-copy");
  if (!result || !text || !copy) return;

  for (const file of images) {
    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      result.hidden = false;
      _setIndicationStatus("Use PNG, JPEG or WebP screenshots.", "error");
      return;
    }
    if (file.size > INDICATION_MAX_IMAGE_BYTES) {
      result.hidden = false;
      _setIndicationStatus("Each screenshot must be 8 MB or smaller.", "error");
      return;
    }
  }

  result.hidden = false;
  text.value = "";
  copy.disabled = true;
  _renderIndicationBusy(true);
  _setIndicationStatus("Starting local text recognition…");
  try {
    const worker = await _getIndicationWorker();
    const extracted = [];
    for (let index = 0; index < images.length; index++) {
      if (images.length > 1) {
        _setIndicationStatus(`Reading screenshot ${index + 1} of ${images.length} locally…`);
      }
      const recognition = await worker.recognize(images[index]);
      const cleaned = _cleanIndicationText(recognition?.data?.text);
      if (cleaned) extracted.push(cleaned);
    }
    if (!extracted.length) {
      throw new Error("No text was found. Try a tighter or clearer screenshot.");
    }
    text.value = extracted.join(" ");
    copy.disabled = false;
    _setIndicationStatus("Ready — check the text, then copy it.", "success");
    text.focus();
    text.select();
  } catch (error) {
    _setIndicationStatus(error.message || "The screenshot could not be read.", "error");
  } finally {
    _renderIndicationBusy(false);
  }
}

function clearIndication() {
  const result = $("indication-result");
  const text = $("indication-text");
  if (text) text.value = "";
  if ($("btn-indication-copy")) $("btn-indication-copy").disabled = true;
  if (result) result.hidden = true;
  _setIndicationStatus("");
}

async function copyIndication() {
  const text = ($("indication-text")?.value || "").trim();
  if (!text) return;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const field = $("indication-text");
      field.focus();
      field.select();
      if (!document.execCommand("copy")) throw new Error("Copy was blocked");
    }
    _setIndicationStatus("Copied — paste into PowerScribe.", "success");
  } catch (_) {
    _setIndicationStatus("Copy was blocked. Select the text and press Ctrl/Cmd+C.", "error");
  }
}

// ---------------------------------------------------------------------------
// Sonographer worksheet screenshots
// ---------------------------------------------------------------------------
const WORKSHEET_MAX_IMAGES = 4;
const WORKSHEET_MAX_IMAGE_BYTES = 8 * 1024 * 1024;
let _worksheetImages = []; // [{file, url}]
let _worksheetBusy = false;

function _formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function _renderWorksheetPreviews() {
  const list = $("worksheet-preview-list");
  const zone = $("worksheet-drop-zone");
  if (!list || !zone) return;
  list.innerHTML = "";

  _worksheetImages.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "worksheet-preview";

    const image = document.createElement("img");
    image.src = item.url;
    image.alt = `Worksheet screenshot ${index + 1}`;

    const meta = document.createElement("div");
    meta.className = "worksheet-preview-meta";
    meta.textContent = `Screenshot ${index + 1} · ${_formatBytes(item.file.size)}`;

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "worksheet-preview-remove";
    remove.setAttribute("aria-label", `Remove worksheet screenshot ${index + 1}`);
    remove.textContent = "×";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      const [removed] = _worksheetImages.splice(index, 1);
      if (removed) URL.revokeObjectURL(removed.url);
      _renderWorksheetPreviews();
    });

    card.append(image, meta, remove);
    list.appendChild(card);
  });

  const hasImages = _worksheetImages.length > 0;
  zone.classList.toggle("has-images", hasImages);
  const busy = _worksheetBusy || _indicationBusy;
  $("btn-worksheet-clear").disabled = !hasImages || busy;
  $("btn-indication-transcribe").disabled = !hasImages || busy;
  $("btn-worksheet-generate").disabled = !hasImages || busy;
  $("btn-worksheet-one-pass").disabled = !hasImages || busy;
}

function clearWorksheetImages() {
  for (const item of _worksheetImages) URL.revokeObjectURL(item.url);
  _worksheetImages = [];
  const input = $("worksheet-file-input");
  if (input) input.value = "";
  clearIndication();
  _renderWorksheetPreviews();
}

function addWorksheetImages(files) {
  let added = 0;
  let rejected = "";
  for (const file of Array.from(files || [])) {
    if (!file || !["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
      rejected = "Use PNG, JPEG or WebP screenshots.";
      continue;
    }
    if (file.size > WORKSHEET_MAX_IMAGE_BYTES) {
      rejected = "Each worksheet screenshot must be 8 MB or smaller.";
      continue;
    }
    if (_worksheetImages.length >= WORKSHEET_MAX_IMAGES) {
      rejected = `Use no more than ${WORKSHEET_MAX_IMAGES} screenshots per worksheet.`;
      break;
    }
    _worksheetImages.push({ file, url: URL.createObjectURL(file) });
    added++;
  }
  if (added) clearIndication();
  _renderWorksheetPreviews();
  if (rejected) {
    setStatus(rejected, "error");
  } else if (added) {
    setStatus(
      `${_worksheetImages.length} screenshot${_worksheetImages.length === 1 ? "" : "s"} ready. Choose an action below.`,
      "active"
    );
  }
}

async function generateFromWorksheet(onePass = false) {
  if (_worksheetBusy || !_worksheetImages.length) return;
  if (state.isRecording) {
    setStatus("Stop the current recording before generating from a worksheet.", "error");
    return;
  }
  if (_signedReportId !== null) {
    setStatus(
      "This report is already signed. Start the next case before using a new worksheet.",
      "error"
    );
    return;
  }

  const existingReport = ($("report-raw")?.value || "").trim();
  if (existingReport && !state.reportCopied && !_signedReportId) {
    const ok = window.confirm(
      "This case already has an uncopied report.\n\n" +
      "Replace it with a report generated from the worksheet?"
    );
    if (!ok) return;
  }

  _worksheetBusy = true;
  _renderWorksheetPreviews();
  setUI("processing");
  setStatus("Reading worksheet tables and annotations…", "active");

  const form = new FormData();
  for (const item of _worksheetImages) form.append("images", item.file, item.file.name || "worksheet.png");
  const modality = ($("modality")?.value || "").trim();
  const bodyPart = ($("body-part")?.value || "").trim();
  if (modality) form.append("modality", modality);
  if (bodyPart) form.append("body_part", bodyPart);
  if (onePass) form.append("reasoning_effort", "low");

  try {
    const response = await fetch(onePass ? "/api/worksheet/draft" : "/api/worksheet/extract", {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || response.statusText);
    }
    const data = await response.json();
    const findings = String(data.findings || "").trim();
    if (!findings) throw new Error("No worksheet findings were returned.");

    const transcription = $("transcription");
    const existing = transcription.value.trim();
    const sourceBlock = `WORKSHEET SOURCE NOTES — blank/unmarked fields are unknown:\n${findings}`;
    transcription.value = existing ? `${existing}\n\n${sourceBlock}` : sourceBlock;
    const templateSelect = $("template-select");
    if (templateSelect && !templateSelect.value) {
      const worksheetOption = Array.from(templateSelect.options)
        .find((option) => option.value === "Ultrasound_Worksheet.txt");
      if (worksheetOption) templateSelect.value = worksheetOption.value;
    }
    state.sourceKind = existing ? "dictation" : "worksheet";
    state.sessionId = null;
    clearWorksheetImages();

    if (onePass) {
      const report = String(data.report || "").trim();
      if (!report) throw new Error("No worksheet report was returned.");
      $("report-raw").value = report;
      $("report-rendered").innerHTML = renderMarkdown(report);
      state.reportLlmOutput = report;
      state.reportCopied = false;
      _qaCheckedReport = "";
      _setReportEditMode(false);
      _signedReportId = null;
      if (typeof _setReportStatus === "function") _setReportStatus("preliminary");
      if (typeof _clearQaPanel === "function") _clearQaPanel();
      setUI("done");
      setStatus("One-step draft ready for review.", "success");
      runQaCheck({ quiet: true });
      refreshFollowupSuggestions(report);
      $("report-rendered").scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      setUI("transcribed");
      setStatus("Worksheet read. Generating the structured report…", "active");
      await formatReport();
    }
  } catch (error) {
    setUI(_inferUIMode());
    setStatus(`Worksheet error: ${error.message}`, "error");
  } finally {
    _worksheetBusy = false;
    _renderWorksheetPreviews();
  }
}

// ---------------------------------------------------------------------------
// Format — streaming SSE
// ---------------------------------------------------------------------------
const CT_CAP_TEMPLATE = "CT_CAP_Staging.txt";
const CT_CAP_LAYOUT_STORAGE_KEY = "radspeed.ctCapLayout";

function _selectedCtCapLayout() {
  return document.querySelector('input[name="ct-cap-layout"]:checked')?.value || "compare";
}

function _looksLikeCtCap(transcription) {
  const context = [
    transcription,
    $("modality")?.value || "",
    $("body-part")?.value || "",
  ].join(" ").toLowerCase().replace(/[–—]/g, "-");
  return /\bct\s+c[\s./-]*a[\s./-]*p(?:s)?\b/.test(context)
    || /\b(?:ct\s+)?(?:chest|thorax)\s*,?\s*(?:abdomen|abdominal)\s*(?:(?:and|&)\s*)?pelvis\b/.test(context);
}

function _updateCtCapLayoutVisibility() {
  const panel = $("ct-cap-layout-panel");
  if (!panel) return;
  panel.hidden = $("template-select")?.value !== CT_CAP_TEMPLATE;
}

function _initCtCapLayout() {
  let saved = "compare";
  try {
    const stored = localStorage.getItem(CT_CAP_LAYOUT_STORAGE_KEY);
    if (["components", "regions", "compare"].includes(stored)) saved = stored;
  } catch (_) {}
  const selected = document.querySelector(`input[name="ct-cap-layout"][value="${saved}"]`);
  if (selected) selected.checked = true;
  document.querySelectorAll('input[name="ct-cap-layout"]').forEach((input) => {
    input.addEventListener("change", () => {
      try { localStorage.setItem(CT_CAP_LAYOUT_STORAGE_KEY, input.value); } catch (_) {}
    });
  });
  _updateCtCapLayoutVisibility();
}

function _hideCtCapComparison({ clear = true } = {}) {
  const panel = $("ct-cap-comparison");
  if (panel) panel.hidden = true;
  document.querySelector("main")?.classList.remove("cap-comparison-open");
  if (!clear) return;
  state.ctCapDrafts = {};
  for (const layout of ["components", "regions"]) {
    const report = $(`ct-cap-${layout}-report`);
    const error = $(`ct-cap-${layout}-error`);
    const button = $(`btn-use-cap-${layout}`);
    if (report) report.innerHTML = "";
    if (error) { error.textContent = ""; error.hidden = true; }
    if (button) button.disabled = true;
  }
}

function _showCtCapComparison() {
  _hideCtCapComparison();
  const panel = $("ct-cap-comparison");
  if (panel) panel.hidden = false;
  document.querySelector("main")?.classList.add("cap-comparison-open");
}

function _formatRequestBody(transcription, overrides = {}) {
  return {
    transcription,
    template_name: $("template-select").value || null,
    session_id: state.sessionId || null,
    source_kind: state.sourceKind,
    patient_name:         $("patient-name").value.trim()        || null,
    patient_dob:          $("patient-dob").value.trim()         || null,
    patient_id:           $("patient-id").value.trim()          || null,
    accession:            $("accession").value.trim()            || null,
    modality:             $("modality").value.trim()             || null,
    body_part:            $("body-part").value.trim()            || null,
    referring_physician:  $("referring-physician").value.trim()  || null,
    radiologist:          $("radiologist").value.trim()          || null,
    ..._selectedPriorPayload(),
    ...overrides,
  };
}

async function _streamFormatRequest(body, { signal, onToken } = {}) {
  const resp = await fetch("/format/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let accumulated = "";
  let completed = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice(6).trim();
      if (!payload) continue;
      let msg;
      try { msg = JSON.parse(payload); } catch { continue; }
      if (msg.token) {
        accumulated += msg.token;
        if (onToken) onToken(accumulated);
      } else if (msg.done) {
        completed = { ...msg, report: msg.report || accumulated };
      } else if (msg.error) {
        throw new Error(msg.error);
      }
    }
  }
  if (!completed) throw new Error("Report generation ended before a completed draft was returned.");
  return completed;
}

function _activateGeneratedReport(report, statusText = "Report ready.") {
  $("report-raw").value = report;
  $("report-rendered").innerHTML = renderMarkdown(report);
  state.reportLlmOutput = report;
  state.reportCopied = false;
  _qaCheckedReport = "";
  _setReportEditMode(false);
  _signedReportId = null;
  if (typeof _setReportStatus === "function") _setReportStatus("preliminary");
  if (typeof _clearQaPanel === "function") _clearQaPanel();
  setUI("done");
  setStatus(statusText, "success");
  runQaCheck({ quiet: true });
  refreshFollowupSuggestions(report);
  $("report-rendered").scrollIntoView({ behavior: "smooth", block: "start" });
}

function useCtCapDraft(layout) {
  const report = state.ctCapDrafts[layout];
  if (!report) return;
  _hideCtCapComparison({ clear: false });
  _activateGeneratedReport(
    report,
    layout === "components"
      ? "Component-based CT CAP draft selected."
      : "Chest + Abdomen/Pelvis CT CAP draft selected.",
  );
}

async function _generateCtCapComparison(baseBody, myEpoch, ac) {
  _showCtCapComparison();
  $("report-raw").value = "";
  $("report-rendered").innerHTML = "";
  _qaCheckedReport = "";
  _setReportEditMode(false);
  setStatus("Generating both CT CAP layouts…", "active");

  const generate = async (layout) => {
    const reportEl = $(`ct-cap-${layout}-report`);
    const errorEl = $(`ct-cap-${layout}-error`);
    try {
      const result = await _streamFormatRequest(
        { ...baseBody, cap_layout: layout, comparison_draft: true },
        {
          signal: ac.signal,
          onToken: (text) => {
            if (state.caseEpoch === myEpoch) reportEl.innerHTML = renderMarkdown(text);
          },
        },
      );
      if (state.caseEpoch !== myEpoch) return false;
      state.ctCapDrafts[layout] = result.report;
      reportEl.innerHTML = renderMarkdown(result.report);
      $(`btn-use-cap-${layout}`).disabled = false;
      return true;
    } catch (error) {
      if (ac.signal.aborted || state.caseEpoch !== myEpoch) return false;
      errorEl.textContent = `This draft failed: ${error.message}`;
      errorEl.hidden = false;
      reportEl.innerHTML = "";
      return false;
    }
  };

  const outcomes = await Promise.all([generate("components"), generate("regions")]);
  if (ac.signal.aborted || state.caseEpoch !== myEpoch) return;
  const ready = outcomes.filter(Boolean).length;
  setUI("transcribed");
  if (ready === 2) {
    setStatus("Two CT CAP drafts are ready. Choose one to continue.", "success");
  } else if (ready === 1) {
    setStatus("One CT CAP draft is ready. The other draft failed.", "error");
  } else {
    setStatus("Both CT CAP drafts failed. Try again.", "error");
  }
  $("ct-cap-comparison").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function formatReport() {
  const transcription = $("transcription").value.trim();
  if (!transcription) {
    setStatus("No transcription to format.", "error");
    return;
  }

  const selectedTemplate = $("template-select").value;
  const autoDetectedCtCap = !selectedTemplate && _looksLikeCtCap(transcription);
  const isCtCap = selectedTemplate === CT_CAP_TEMPLATE || autoDetectedCtCap;
  // Auto-select uses comparison mode so an incorrect generic CT template can
  // never silently absorb a combined chest/abdomen/pelvis staging study.
  const capLayout = autoDetectedCtCap
    ? "compare"
    : (isCtCap ? _selectedCtCapLayout() : null);
  const body = _formatRequestBody(
    transcription,
    {
      ...(autoDetectedCtCap ? { template_name: CT_CAP_TEMPLATE } : {}),
      ...(capLayout && capLayout !== "compare" ? { cap_layout: capLayout } : {}),
    },
  );

  setUI("formatting");
  setStatus("Generating report…", "active");

  // Clear report area and start streaming into it.
  _hideCtCapComparison();
  $("report-raw").value = "";
  $("report-rendered").innerHTML = "";
  _qaCheckedReport = "";
  _setReportEditMode(false);

  // Tie this run to the current case; abort a prior in-flight format.
  const myEpoch = state.caseEpoch;
  if (state.formatAbort) { try { state.formatAbort.abort(); } catch (_) {} }
  const ac = new AbortController();
  state.formatAbort = ac;

  try {
    if (isCtCap && capLayout === "compare") {
      await _generateCtCapComparison(body, myEpoch, ac);
      return;
    }

    const result = await _streamFormatRequest(body, {
      signal: ac.signal,
      onToken: (text) => {
        if (state.caseEpoch !== myEpoch) return;
        $("report-raw").value = text;
        $("report-rendered").innerHTML = renderMarkdown(text);
      },
    });
    if (state.caseEpoch !== myEpoch) return;
    const fhirNote = result.fhir_saved ? " · FHIR R4 JSON saved." : "";
    _activateGeneratedReport(result.report, "Report ready." + fhirNote);
  } catch (err) {
    // Silent on an intentional abort (Next Case) or a superseded run.
    if (ac.signal.aborted || state.caseEpoch !== myEpoch) return;
    setUI("transcribed");
    setStatus(`Format error: ${err.message}`, "error");
  } finally {
    if (state.formatAbort === ac) state.formatAbort = null;
  }
}

async function generateImpressionOnly() {
  const report = ($("report-raw")?.value || "").trim();
  if (!report) {
    setStatus("Generate a report before refreshing its impression.", "error");
    return;
  }
  if (_signedReportId !== null) {
    setStatus("Amend the signed report before changing its impression.", "error");
    return;
  }

  const button = $("btn-impression");
  button.disabled = true;
  const originalLabel = button.innerHTML;
  button.textContent = "Generating…";
  setStatus("Generating the impression only with matching guidelines…", "active");

  try {
    const response = await fetch("/api/report/impression", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report,
        modality: $("modality")?.value.trim() || null,
        with_guidelines: true,
      }),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || response.statusText);
    }
    const data = await response.json();
    setReport(data.report);
    state.reportLlmOutput = data.report;
    _qaCheckedReport = "";
    _setReportStatus("preliminary");
    runQaCheck({ quiet: true });
    refreshFollowupSuggestions(data.report);
    setStatus("Impression updated. Findings and other sections were unchanged.", "success");
  } catch (error) {
    setStatus(`Impression error: ${error.message}`, "error");
  } finally {
    button.innerHTML = originalLabel;
    button.disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Report: render markdown + edit/preview toggle
// ---------------------------------------------------------------------------
let _reportEditMode = false;

// Render markdown to sanitized HTML. Report text is attacker-influenceable —
// HL7-derived patient names/procedures flow through the LLM prompt and templates
// echo demographics into the header, so raw HTML in the output (marked passes it
// through untouched) could otherwise inject script into report-rendered.innerHTML
// and exfiltrate the WS token / PHI. DOMPurify strips it; the same sanitized HTML
// is what lands on the clipboard.
function renderMarkdown(md) {
  const html = (window.marked ? marked.parse(md || "") : (md || ""));
  return (window.DOMPurify ? DOMPurify.sanitize(html) : html);
}

function setReport(markdown) {
  $("report-raw").value = markdown;
  $("report-rendered").innerHTML = renderMarkdown(markdown);
  if (_reportEditMode) _setReportEditMode(false);
  _markReportChanged();
  refreshFollowupSuggestions(markdown);
}

function _setReportEditMode(editing) {
  _reportEditMode = editing;
  $("report-rendered").style.display = editing ? "none" : "";
  $("report-raw").style.display      = editing ? "" : "none";
  $("btn-edit-toggle").textContent   = editing ? "✓ Done" : "✎ Edit";
  const hint = $("report-edit-hint");
  if (hint) hint.style.display = editing ? "" : "none";
}

function toggleReportEdit() {
  if (!_reportEditMode) {
    _setReportEditMode(true);
  } else {
    $("report-rendered").innerHTML = renderMarkdown($("report-raw").value);
    _setReportEditMode(false);
    refreshFollowupSuggestions($("report-raw").value);
  }
}

// ---------------------------------------------------------------------------
// Copy report — honors the user's paste_format preference
// ---------------------------------------------------------------------------
function _pasteFormat() {
  return document.body.dataset.pasteFormat || "rich";
}

function _clipboardInlineText(node) {
  if (node.nodeType === Node.TEXT_NODE) return node.nodeValue || "";
  if (node.nodeType !== Node.ELEMENT_NODE) return "";
  if (node.tagName === "BR") return "\n";
  if (node.tagName === "SCRIPT" || node.tagName === "STYLE") return "";
  return Array.from(node.childNodes).map(_clipboardInlineText).join("");
}

function _cleanClipboardChunk(text) {
  return String(text || "")
    .replace(/\r\n?/g, "\n")
    .replace(/\u00a0/g, " ")
    .split("\n")
    .map((line) => line.trim())
    .join("\n")
    .replace(/^\n+|\n+$/g, "")
    .replace(/\n{3,}/g, "\n\n");
}

function _isClipboardBoldOnlyBlock(node) {
  const children = Array.from(node.childNodes).filter((child) =>
    child.nodeType !== Node.TEXT_NODE || (child.nodeValue || "").trim()
  );
  return children.length === 1 &&
    children[0].nodeType === Node.ELEMENT_NODE &&
    (children[0].tagName === "STRONG" || children[0].tagName === "B");
}

function _clipboardBlockKind(node, text) {
  const trimmed = text.trim();
  if (_isClipboardBoldOnlyBlock(node)) {
    return trimmed.endsWith(":") ? "section" : "group";
  }
  if (!trimmed.includes("\n") && trimmed.endsWith(":")) return "section";
  if (!trimmed.includes("\n") &&
      /^[A-Z0-9][A-Z0-9 /&()+,.'’\-]{2,}$/.test(trimmed)) return "title";
  return "content";
}

function _clipboardListLines(list, depth = 0) {
  const ordered = list.tagName === "OL";
  const start = ordered ? (parseInt(list.getAttribute("start") || "1", 10) || 1) : 1;
  const lines = [];
  const items = Array.from(list.children).filter((child) => child.tagName === "LI");

  items.forEach((item, index) => {
    const directText = _cleanClipboardChunk(
      Array.from(item.childNodes)
        .filter((child) => !(child.nodeType === Node.ELEMENT_NODE &&
          (child.tagName === "OL" || child.tagName === "UL")))
        .map(_clipboardInlineText)
        .join("")
    );
    // ASCII hyphens survive Qscan/PowerScribe text fields more reliably than
    // typographic bullet glyphs. Long impression lists are numbered server-side.
    const prefix = ordered ? `${start + index}. ` : "- ";
    if (directText) lines.push(`${"  ".repeat(depth)}${prefix}${directText}`);

    Array.from(item.children)
      .filter((child) => child.tagName === "OL" || child.tagName === "UL")
      .forEach((nested) => lines.push(..._clipboardListLines(nested, depth + 1)));
  });
  return lines;
}

function _clipboardPlainText(html, markdown) {
  // Serialize Markdown blocks ourselves rather than reading innerText from a
  // styled offscreen element. Chromium expands block margins into repeated
  // newlines there, which PowerScribe then pastes as large gaps.
  const root = document.createElement("div");
  root.innerHTML = html;
  const chunks = [];

  const addChunk = (text, kind = "content") => {
    if (text.trim()) chunks.push({ text, kind });
  };

  Array.from(root.childNodes).forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = _cleanClipboardChunk(node.nodeValue);
      addChunk(text);
      return;
    }
    if (node.nodeType !== Node.ELEMENT_NODE) return;

    if (node.tagName === "OL" || node.tagName === "UL") {
      const lines = _clipboardListLines(node);
      if (lines.length) addChunk(lines.join("\n"));
      return;
    }
    if (node.tagName === "TABLE") {
      const rows = Array.from(node.querySelectorAll("tr")).map((row) =>
        Array.from(row.querySelectorAll(":scope > th, :scope > td"))
          .map((cell) => _cleanClipboardChunk(_clipboardInlineText(cell)))
          .join("\t")
      );
      if (rows.length) addChunk(rows.join("\n"));
      return;
    }
    if (node.tagName === "HR") return;

    const text = _cleanClipboardChunk(
      node.tagName === "PRE" ? node.textContent : _clipboardInlineText(node)
    );
    addChunk(text, _clipboardBlockKind(node, text));
  });

  if (!chunks.length) return _cleanClipboardChunk(markdown).trim();

  // Keep section and anatomical-group headings attached to their findings.
  // Within a named anatomical group, consecutive findings stay single-spaced;
  // a blank line remains between groups and between top-level sections.
  let inGroup = false;
  return chunks.reduce((plain, chunk, index) => {
    if (!plain) {
      inGroup = chunk.kind === "group";
      return chunk.text;
    }
    const previous = chunks[index - 1];
    let separator = "\n\n";
    if (previous.kind === "group") {
      separator = "\n";
    } else if (previous.kind === "section" && chunk.kind !== "section") {
      separator = "\n";
    } else if (chunk.kind !== "group" && chunk.kind !== "section" && inGroup) {
      separator = "\n";
    }
    if (chunk.kind === "section") inGroup = false;
    if (chunk.kind === "group") inGroup = true;
    return `${plain}${separator}${chunk.text}`;
  }, "").trim();
}

function _escapeClipboardHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function _isClipboardHeading(line) {
  const text = line.trim();
  if (!text || text.length > 100) return false;
  return text.endsWith(":") ||
    /^[A-Z0-9][A-Z0-9 /&()+,.'’\-]{2,}$/.test(text);
}

function _clipboardBoldLines(html) {
  const root = document.createElement("div");
  root.innerHTML = html;
  return new Set(Array.from(root.querySelectorAll("strong, b"))
    .map((node) => _cleanClipboardChunk(node.textContent).trim())
    .filter((text) => text && !text.includes("\n")));
}

function _clipboardSubheadingPrefix(line, explicitBold = new Set()) {
  const text = String(line || "").trimStart();
  if (!text || text.length > 500) return "";

  const explicit = Array.from(explicitBold)
    .filter((fragment) => fragment.endsWith(":") && text.startsWith(fragment) && text !== fragment)
    .sort((left, right) => right.length - left.length)[0];
  if (explicit) return explicit;

  const colon = text.indexOf(":");
  if (colon < 1 || colon > 60 || colon === text.length - 1) return "";
  const label = text.slice(0, colon + 1);
  const words = label.slice(0, -1).trim().split(/\s+/);
  if (words.length > 4 || /^(?:there|this|no|a|an|the)$/i.test(words[0])) return "";
  return /^[A-Za-z0-9][A-Za-z0-9 /&()+,.'’\-]*:$/.test(label) ? label : "";
}

function _clipboardBoldPrefixes(plain, html) {
  const explicitBold = _clipboardBoldLines(html);
  return Array.from(new Set(plain.split("\n")
    .map((line) => _clipboardSubheadingPrefix(line, explicitBold))
    .filter(Boolean)));
}

function _clipboardRtfBoldLines(plain, html) {
  const explicitBold = _clipboardBoldLines(html);
  const seen = new Set();
  return plain.split("\n")
    .map((line) => line.trim())
    .filter((text) => {
      const keep = text && !seen.has(text) &&
        (_isClipboardHeading(text) || explicitBold.has(text));
      if (keep) seen.add(text);
      return keep;
    });
}

function _clipboardRichHtml(plain, html) {
  // A line-break fragment pastes consistently into PowerScribe and still gives
  // Word/Outlook the original bold text. Paragraph/list block tags are avoided
  // because target editors commonly add their own spacing around those blocks.
  const boldLines = _clipboardBoldLines(html);
  const lines = plain.split("\n").map((line) => {
    const escaped = _escapeClipboardHtml(line);
    if (_isClipboardHeading(line) || boldLines.has(line.trim())) {
      return `<strong>${escaped}</strong>`;
    }
    const prefix = _clipboardSubheadingPrefix(line, boldLines);
    if (!prefix) return escaped;
    const start = line.indexOf(prefix);
    return `${_escapeClipboardHtml(line.slice(0, start))}` +
      `<strong>${_escapeClipboardHtml(prefix)}</strong>` +
      `${_escapeClipboardHtml(line.slice(start + prefix.length))}`;
  });
  return `<span>${lines.join("<br>")}</span>`;
}

// Fire-and-forget: diff the current report against the LLM output to detect
// impression-style or date-format drift, and show a style suggestion if found.
function _trackReportEdit(currentReport) {
  if (!state.reportLlmOutput || state.reportLlmOutput === currentReport) return;
  fetch("/api/track-report-edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      original_report: state.reportLlmOutput,
      edited_report: currentReport,
    }),
  })
    .then((r) => (r.ok ? r.json() : null))
    .then((data) => {
      if (data && data.suggest_setting) showStyleSuggest(data.suggest_setting);
    })
    .catch(() => {});
}

function _normaliseReportHeading(line) {
  return String(line || "")
    .trim()
    .replace(/^#{1,6}\s*/, "")
    .replace(/^(?:\*\*|__)/, "")
    .replace(/(?:\*\*|__)$/, "")
    .replace(/:\s*$/, "")
    .trim()
    .toLowerCase();
}

function _reportFromComparisonOrFindings(markdown) {
  const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  const comparisonHeadings = new Set([
    "comparison", "comparisons", "prior", "priors", "prior imaging",
  ]);
  let findingsIndex = -1;

  for (let index = 0; index < lines.length; index += 1) {
    const heading = _normaliseReportHeading(lines[index]);
    if (comparisonHeadings.has(heading)) {
      return { markdown: lines.slice(index).join("\n").trim(), startsAt: "Comparison" };
    }
    if (findingsIndex === -1 && heading === "findings") findingsIndex = index;
  }

  if (findingsIndex !== -1) {
    return { markdown: lines.slice(findingsIndex).join("\n").trim(), startsAt: "Findings" };
  }
  return { markdown: String(markdown || "").trim(), startsAt: "full report" };
}

function _reportWithoutExam(markdown) {
  const headings = new Set([
    "exam", "examination", "technique", "history", "clinical history",
    "clinical details", "clinical information", "clinical indication",
    "clinical question", "indication", "procedure", "comparison", "comparisons", "prior", "priors",
    "prior imaging", "findings", "impression", "conclusion", "opinion",
    "recommendation", "recommendations",
  ]);
  let inExam = false;
  return String(markdown || "").replace(/\r\n?/g, "\n").split("\n")
    .filter((line) => {
      // Accept heading-only and inline titles, including Markdown headings.
      const label = line.trim().replace(/^#{1,6}\s*/, "").replace(/\*\*|__/g, "");
      if (/^(?:exam|examination)\s*(?::|$)/i.test(label)) {
        inExam = true;
        return false;
      }
      const heading = _normaliseReportHeading(line);
      const inlineHeading = label.match(/^([^:]+):/);
      if (headings.has(heading) || (inlineHeading && headings.has(inlineHeading[1].trim().toLowerCase()))) {
        inExam = false;
      }
      return !inExam;
    }).join("\n").trim();
}

async function copyReport(options = {}) {
  const fullMarkdown = $("report-raw").value;
  const selection = options.fromComparison
    ? _reportFromComparisonOrFindings(fullMarkdown)
    : { markdown: fullMarkdown, startsAt: "full report" };
  const markdown = _reportWithoutExam(selection.markdown);
  if (!markdown.trim()) return;
  const fmt = _pasteFormat();
  // Always render fresh from the raw markdown source. In Edit mode the visible
  // #report-rendered div is not re-rendered until "Done" is clicked, so reading
  // its innerText/innerHTML here would silently copy the *pre-edit* report and
  // drop the radiologist's corrections.
  const html = renderMarkdown(markdown);
  // Copy scope must not remove sections from the visible or saved report.
  $("report-rendered").innerHTML = renderMarkdown(fullMarkdown);
  const plain = _clipboardPlainText(html, markdown);
  const richHtml = _clipboardRichHtml(plain, html);

  // Build the clipboard payload per the user's paste-format preference.
  //   rich     — html + plain-rendered fallback
  //   plain    — plain-rendered only (strips markdown markers)
  //   markdown — raw markdown source only
  const payload =
    fmt === "markdown" ? { plain: markdown } :
    fmt === "plain"    ? { plain } :
                         { plain, html: richHtml };
  const labelSuffix =
    fmt === "markdown" ? " (markdown)" :
    fmt === "plain"    ? " (plain)" :
                         "";
  const scopeSuffix = options.fromComparison ? ` from ${selection.startsAt}` : "";

  try {
    const nativeInvoke = window.__TAURI__?.core?.invoke;
    if (fmt === "rich" && nativeInvoke) {
      await nativeInvoke("cmd_copy_report_rtf", {
        body: {
          text: payload.plain,
          boldLines: _clipboardRtfBoldLines(plain, html),
          boldPrefixes: _clipboardBoldPrefixes(plain, html),
        },
      });
    } else if (payload.html && window.ClipboardItem && navigator.clipboard && navigator.clipboard.write) {
      const item = new ClipboardItem({
        "text/html":  new Blob([payload.html],  { type: "text/html" }),
        "text/plain": new Blob([payload.plain], { type: "text/plain" }),
      });
      await navigator.clipboard.write([item]);
    } else if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(payload.plain);
    } else {
      throw new Error("Clipboard API unavailable");
    }
    state.reportCopied = true;
    const nativeSuffix = fmt === "rich" && nativeInvoke ? " (PowerScribe rich text)" : labelSuffix;
    const browserWarning = fmt === "rich" && !nativeInvoke
      ? " PowerScribe may remove browser formatting; use the RadSpeed Windows app for native rich text."
      : "";
    setStatus(`Report copied${scopeSuffix} to clipboard${nativeSuffix}.${browserWarning} Press Alt+N for next case.`, "success");
    _trackReportEdit(fullMarkdown);
    return;
  } catch {
    // Fallback: use a contenteditable div + execCommand("copy").
    const div = document.createElement("div");
    div.contentEditable = "true";
    if (payload.html) {
      div.innerHTML = payload.html;
    } else {
      div.textContent = payload.plain;
    }
    div.style.position = "fixed";
    div.style.left = "-9999px";
    div.style.top = "0";
    div.style.opacity = "0";
    document.body.appendChild(div);
    const range = document.createRange();
    range.selectNodeContents(div);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    try {
      // execCommand returns false on failure instead of throwing — checking the
      // return value stops us reporting "copied" when nothing reached the
      // clipboard (which would let the radiologist paste stale content).
      const ok = document.execCommand("copy");
      if (ok) {
        state.reportCopied = true;
        const fallbackWarning = fmt === "rich"
          ? " Native rich copy was unavailable, so PowerScribe formatting may be removed."
          : "";
        setStatus(`Report copied${scopeSuffix} to clipboard${labelSuffix}.${fallbackWarning} Press Alt+N for next case.`, "success");
      } else {
        setStatus("Copy failed — select and copy manually.", "error");
      }
    } catch {
      setStatus("Copy failed — select and copy manually.", "error");
    }
    sel.removeAllRanges();
    div.remove();
  }
}

// ---------------------------------------------------------------------------
// Next Case — atomically reset the UI so a radiologist can burn through a list
// ---------------------------------------------------------------------------
function nextCase({ keepRadiologist = true, force = false } = {}) {
  if (fbState.isRecording || fbState.isProcessing) {
    setStatus("Finish report refinement before starting the next case.", "error");
    return;
  }
  // Guard against destroying an unsaved report. Alt+N is a single keystroke and
  // fires even while typing in the report field, so confirm when there is a
  // non-empty report that has been neither copied nor signed off.
  if (!force) {
    const rpt = $("report-raw").value.trim();
    if (rpt && !state.reportCopied && !_signedReportId) {
      const ok = window.confirm(
        "This case has a report that hasn't been copied or signed off.\n\n" +
        "Start the next case and discard it?"
      );
      if (!ok) return;
    }
  }

  // Invalidate in-flight work so a late token/transcript can't land in the new
  // case, and abort the format stream / stop any active recording.
  state.caseEpoch++;
  if (state.formatAbort) { try { state.formatAbort.abort(); } catch (_) {} state.formatAbort = null; }
  if (state.isRecording) {
    if (state.streamingSupported && state.streamingWs) {
      _cleanupStreaming();
    } else {
      state.isRecording = false;
      if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
        try { state.mediaRecorder.stop(); } catch (_) {}
      }
      if (state.stream) { RadSpeedMicrophone.stop(state.stream); state.stream = null; }
    }
    stopTimer();
    stopWaveform();
  }

  // Transcription + report
  $("transcription").value = "";
  $("report-raw").value = "";
  $("report-rendered").innerHTML = "";
  _hideCtCapComparison();
  state.reportCopied = false;
  state.reportLlmOutput = "";
  state.sourceKind = "dictation";
  clearWorksheetImages();

  // Patient context — preserve radiologist name so they don't retype it each case.
  const fields = [
    "patient-name", "patient-dob", "patient-id",
    "accession", "modality", "body-part", "referring-physician",
  ];
  if (!keepRadiologist) fields.push("radiologist");
  for (const id of fields) {
    const el = $(id);
    if (el) el.value = "";
  }
  updatePatientSummary();
  const patientDetails = $("patient-context-details");
  if (patientDetails) patientDetails.open = true;
  _resetPriors();
  const followupSuggest = $("followup-suggest-panel");
  if (followupSuggest) { followupSuggest.innerHTML = ""; followupSuggest.style.display = "none"; }
  for (const id of ["followup-patient", "followup-accession", "followup-due", "followup-recommendation", "score-target"]) {
    const el = $(id);
    if (el) el.value = "";
  }
  if ($("followup-modal")) $("followup-modal").style.display = "none";
  if ($("score-modal")) $("score-modal").style.display = "none";

  // Worklist — clear selection and disable archive button
  const wl = $("worklist-select");
  if (wl) wl.value = "";
  _activeWorklistOrderId = "";
  const arch = $("btn-worklist-archive");
  if (arch) arch.disabled = true;

  // Reset edit state so the user sees the rendered view next time
  if (typeof _setReportEditMode === "function") _setReportEditMode(false);

  // New dictation session
  state.sessionId = null;
  state.voiceEditTarget = null;
  if (fbState) fbState.selectedText = "";

  // Phase 1: clear sign-off state and QA flags
  _signedReportId = null;
  _qaCheckedReport = "";
  _setReportStatus("draft");
  _clearQaPanel();

  setUI("idle");
  setStatus("Ready for next case.", "active");

  // Focus the transcription textarea so the user can start immediately
  const t = $("transcription");
  if (t) t.focus();
}

// ---------------------------------------------------------------------------
// Radiologist-confirmed follow-up register
// ---------------------------------------------------------------------------
async function refreshFollowupSuggestions(reportText) {
  const panel = $("followup-suggest-panel");
  if (!panel) return;
  const epoch = state.caseEpoch;
  panel.innerHTML = "";
  panel.style.display = "none";
  if (!(reportText || "").trim()) return;
  try {
    const resp = await fetch("/api/followups/suggest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ report_text: reportText }),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    // A late response must never put patient A's recommendation banner into
    // patient B's newly reset case or an amended report.
    if (epoch !== state.caseEpoch || ($("report-raw")?.value || "") !== reportText) return;
    const suggestions = data.suggestions || [];
    if (!suggestions.length) return;
    const heading = document.createElement("strong");
    heading.textContent = "Explicit follow-up language detected — track if appropriate";
    panel.appendChild(heading);
    for (const suggestion of suggestions) {
      const row = document.createElement("div");
      row.className = "followup-suggestion";
      const text = document.createElement("span");
      text.textContent = suggestion.recommendation;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "btn btn-secondary btn-sm";
      button.textContent = "Track";
      button.addEventListener("click", () => openFollowupModal(suggestion.recommendation));
      row.append(text, button);
      panel.appendChild(row);
    }
    panel.style.display = "";
  } catch (_) { /* advisory feature only */ }
}

function closeFollowupModal() {
  $("followup-modal").style.display = "none";
}

async function openFollowupModal(recommendation = "") {
  $("followup-patient").value = ($("patient-name")?.value || "").trim();
  $("followup-accession").value = ($("accession")?.value || "").trim();
  if (recommendation) $("followup-recommendation").value = recommendation;
  $("followup-modal").style.display = "flex";
  await loadFollowups();
}

async function loadFollowups() {
  const list = $("followup-list");
  const count = $("followup-count");
  try {
    const resp = await fetch("/api/followups?status=open");
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "Follow-up register unavailable." }));
      if (list) list.textContent = err.detail || "Follow-up register unavailable.";
      if (count) count.style.display = "none";
      return;
    }
    const data = await resp.json();
    const items = data.followups || [];
    if (count) {
      count.textContent = String(items.length);
      count.style.display = items.length ? "" : "none";
    }
    if (!list) return;
    list.innerHTML = "";
    if (!items.length) {
      const empty = document.createElement("div");
      empty.className = "field-note";
      empty.textContent = "No outstanding follow-ups.";
      list.appendChild(empty);
      return;
    }
    const today = new Date().toISOString().slice(0, 10);
    for (const item of items) {
      const card = document.createElement("div");
      card.className = "followup-item" + (item.due_date && item.due_date < today ? " overdue" : "");
      const header = document.createElement("div");
      header.className = "followup-item-header";
      const patient = document.createElement("strong");
      patient.textContent = item.patient_name || item.patient_id || "Patient not recorded";
      const meta = document.createElement("span");
      meta.className = "field-note";
      meta.textContent = [item.accession ? `#${item.accession}` : "", item.due_date ? `Due ${item.due_date}` : "No due date"]
        .filter(Boolean).join(" · ");
      header.append(patient, meta);
      const recommendation = document.createElement("div");
      recommendation.className = "followup-item-recommendation";
      recommendation.textContent = item.recommendation;
      const actions = document.createElement("div");
      actions.className = "row";
      for (const [status, label] of [["completed", "Complete"], ["dismissed", "Dismiss"]]) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = status === "completed" ? "btn btn-success btn-sm" : "btn btn-ghost btn-sm";
        button.textContent = label;
        button.addEventListener("click", () => updateFollowupStatus(item.id, status));
        actions.appendChild(button);
      }
      card.append(header, recommendation, actions);
      list.appendChild(card);
    }
  } catch (err) {
    if (list) list.textContent = `Follow-up register unavailable: ${err.message}`;
  }
}

async function saveFollowup() {
  const recommendation = $("followup-recommendation").value.trim();
  if (!recommendation) {
    setStatus("Enter a follow-up recommendation first.", "error");
    return;
  }
  const button = $("followup-save");
  button.disabled = true;
  try {
    const resp = await fetch("/api/followups", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        recommendation,
        report_id: _signedReportId,
        patient_id: ($("patient-id")?.value || "").trim() || null,
        patient_name: $("followup-patient").value.trim() || null,
        accession: $("followup-accession").value.trim() || null,
        due_date: $("followup-due").value || null,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    $("followup-recommendation").value = "";
    $("followup-due").value = "";
    setStatus("Follow-up added to your register.", "success");
    await loadFollowups();
  } catch (err) {
    setStatus(`Could not track follow-up: ${err.message}`, "error");
  } finally {
    button.disabled = false;
  }
}

async function updateFollowupStatus(id, status) {
  try {
    const resp = await fetch(`/api/followups/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    await loadFollowups();
  } catch (err) {
    setStatus(`Could not update follow-up: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Radiologist-selected standardised assessment inserter
// ---------------------------------------------------------------------------
const _SCORE_SYSTEMS = {
  "BI-RADS": [
    ["BI-RADS 0", "Incomplete"], ["BI-RADS 1", "Negative"],
    ["BI-RADS 2", "Benign"], ["BI-RADS 3", "Probably benign"],
    ["BI-RADS 4", "Suspicious"], ["BI-RADS 4A", "Low suspicion for malignancy"],
    ["BI-RADS 4B", "Moderate suspicion for malignancy"],
    ["BI-RADS 4C", "High suspicion for malignancy"],
    ["BI-RADS 5", "Highly suggestive of malignancy"],
    ["BI-RADS 6", "Known biopsy-proven malignancy"],
  ],
  "PI-RADS v2.1": [
    ["PI-RADS 1", "Very low likelihood of clinically significant cancer"],
    ["PI-RADS 2", "Low likelihood of clinically significant cancer"],
    ["PI-RADS 3", "Intermediate likelihood of clinically significant cancer"],
    ["PI-RADS 4", "High likelihood of clinically significant cancer"],
    ["PI-RADS 5", "Very high likelihood of clinically significant cancer"],
  ],
  "LI-RADS CT/MRI v2018": [
    ["LR-NC", "Not categorizable"], ["LR-1", "Definitely benign"],
    ["LR-2", "Probably benign"], ["LR-3", "Intermediate probability of malignancy"],
    ["LR-4", "Probably HCC"], ["LR-5", "Definitely HCC"],
    ["LR-M", "Probably or definitely malignant, not necessarily HCC"],
    ["LR-TIV", "Tumor in vein"],
  ],
  "ACR TI-RADS": [
    ["TR1", "Benign"], ["TR2", "Not suspicious"], ["TR3", "Mildly suspicious"],
    ["TR4", "Moderately suspicious"], ["TR5", "Highly suspicious"],
  ],
};

function _updateScoreCategories() {
  const system = $("score-system").value;
  const category = $("score-category");
  category.innerHTML = "";
  for (const [code, label] of _SCORE_SYSTEMS[system] || []) {
    const option = document.createElement("option");
    option.value = code;
    option.dataset.label = label;
    option.textContent = `${code} — ${label}`;
    category.appendChild(option);
  }
  _updateScorePreview();
}

function _scoreLine() {
  const option = $("score-category").selectedOptions[0];
  if (!option) return "";
  const target = $("score-target").value.trim();
  return `${target ? target + ": " : ""}${option.value} — ${option.dataset.label}.`;
}

function _updateScorePreview() {
  $("score-preview").textContent = _scoreLine();
}

function openScoreModal() {
  const system = $("score-system");
  if (!system.options.length) {
    for (const name of Object.keys(_SCORE_SYSTEMS)) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      system.appendChild(option);
    }
  }
  _updateScoreCategories();
  $("score-modal").style.display = "flex";
}

function closeScoreModal() { $("score-modal").style.display = "none"; }

function insertScore() {
  const existing = ($("report-raw")?.value || "").trim();
  if (!existing) {
    setStatus("Generate or begin a report before inserting an assessment.", "error");
    return;
  }
  const line = _scoreLine();
  if (!line) return;
  const editingSignedReport = _signedReportId !== null;
  setReport(`${existing}\n\n**STANDARDISED ASSESSMENT:**\n${line}`);
  setUI("done");
  setStatus(
    editingSignedReport
      ? "Assessment inserted. Use Amend to save this change to the signed report."
      : "Radiologist-selected assessment inserted. Confirm it before sign-off.",
    "success"
  );
  closeScoreModal();
}

// ---------------------------------------------------------------------------
// Phase 1: sign-off, amendments, audit
// ---------------------------------------------------------------------------
let _signedReportId = null;

function _setReportStatus(status) {
  const badge = $("report-status-badge");
  if (!badge) return;
  badge.classList.remove("status-draft", "status-preliminary", "status-final", "status-amended");
  badge.classList.add("status-" + status);
  badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

function _patientPayload() {
  return {
    patient_name:        $("patient-name")?.value.trim()        || null,
    patient_dob:         $("patient-dob")?.value.trim()         || null,
    patient_id:          $("patient-id")?.value.trim()          || null,
    accession:           $("accession")?.value.trim()            || null,
    modality:            $("modality")?.value.trim()             || null,
    body_part:           $("body-part")?.value.trim()            || null,
    referring_physician: $("referring-physician")?.value.trim()  || null,
    radiologist:         $("radiologist")?.value.trim()          || null,
  };
}

async function signOffReport() {
  const text = ($("report-raw")?.value || "").trim();
  if (!text) {
    setStatus("Nothing to sign off.", "error");
    return;
  }
  // Recheck after any edit. The pass remains advisory, but surfacing its flags
  // here avoids a separate QA click and keeps sign-off clinically deliberate.
  if (_qaCheckedReport !== text) await runQaCheck({ quiet: true });
  const pendingQaFlags = document.querySelectorAll("#qa-panel .qa-flag").length;
  const qaNote = pendingQaFlags
    ? `QA has ${pendingQaFlags} unresolved flag${pendingQaFlags === 1 ? "" : "s"}.\n\n`
    : "";
  if (!confirm(
    qaNote + "Sign off this report as FINAL? Further edits will create a versioned amendment."
  )) return;

  const body = {
    report_text: text,
    template_name: $("template-select")?.value || null,
    ..._patientPayload(),
  };

  try {
    setStatus("Signing off…", "active");
    const resp = await fetch("/api/reports/sign-off", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    _signedReportId = data.report?.id || null;
    _setReportStatus("final");
    const exports = [
      data.hl7_saved ? "HL7" : null,
      data.sr_saved ? "DICOM SR" : null,
      data.fhir_saved ? "FHIR" : null,
    ].filter(Boolean).join(", ");
    setStatus(
      `Signed off (v${data.report?.version || 1})${exports ? " · exported: " + exports : ""}.`,
      "success"
    );
    setUI("done");
  } catch (err) {
    setStatus(`Sign-off failed: ${err.message}`, "error");
  }
}

function openAmendModal() {
  if (_signedReportId === null) {
    setStatus("Nothing to amend — sign the report off first.", "error");
    return;
  }
  $("amend-reason").value = "";
  $("amend-modal").style.display = "flex";
  setTimeout(() => $("amend-reason").focus(), 0);
}

function closeAmendModal() {
  $("amend-modal").style.display = "none";
}

async function confirmAmend() {
  const reason = ($("amend-reason")?.value || "").trim();
  const text = ($("report-raw")?.value || "").trim();
  if (!reason) {
    $("amend-reason").focus();
    return;
  }
  if (!text) {
    setStatus("Nothing to amend.", "error");
    return;
  }
  try {
    setStatus("Saving amendment…", "active");
    const resp = await fetch("/api/reports/amend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prior_report_id: _signedReportId,
        report_text: text,
        amendment_reason: reason,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    _signedReportId = data.report?.id || _signedReportId;
    _setReportStatus("amended");
    closeAmendModal();
    setStatus(`Amendment saved (v${data.report?.version || "?"}).`, "success");
  } catch (err) {
    setStatus(`Amendment failed: ${err.message}`, "error");
  }
}

async function openAuditModal() {
  const accession = ($("accession")?.value || "").trim();
  if (!accession) {
    setStatus("Enter an accession number to view its audit trail.", "error");
    return;
  }

  $("audit-modal-title").textContent = `Audit Trail — ${accession}`;
  const body = $("audit-modal-body");
  body.innerHTML = '<div style="color:var(--muted);font-size:13px;">Loading…</div>';
  $("audit-modal").style.display = "flex";

  try {
    const [reportsResp, eventsResp] = await Promise.all([
      fetch(`/api/reports?accession=${encodeURIComponent(accession)}`),
      fetch(`/api/audit-log?accession=${encodeURIComponent(accession)}&limit=200`),
    ]);
    const reportsData = reportsResp.ok ? await reportsResp.json() : { reports: [] };
    const eventsData = eventsResp.ok ? await eventsResp.json() : { events: [] };
    body.innerHTML = _renderAuditTrail(reportsData.reports || [], eventsData.events || []);
  } catch (err) {
    body.innerHTML = `<div style="color:var(--red);font-size:13px;">Failed to load: ${err.message}</div>`;
  }
}

function _renderAuditTrail(reports, events) {
  const _esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  let html = "";

  html += '<div class="audit-section-title">Report versions</div>';
  if (!reports.length) {
    html += '<div style="color:var(--muted);font-size:13px;">No signed versions yet.</div>';
  } else {
    for (const r of reports) {
      const status = (r.status || "").toLowerCase();
      html += `<div class="report-version">`
        + `<div class="report-version-header">`
        + `<span class="status-badge status-${_esc(status)}">${_esc(status || "?")}</span>`
        + ` <strong>v${_esc(r.version)}</strong>`
        + ` <span style="color:var(--muted);font-size:11px;">${_esc(r.signed_at || r.created_at || "")}</span>`
        + ` <span style="color:var(--muted);font-size:11px;margin-left:auto;">hash ${_esc((r.report_hash || "").slice(0, 12))}…</span>`
        + `</div>`;
      if (r.amendment_reason) {
        html += `<div class="report-version-reason">Reason: ${_esc(r.amendment_reason)}</div>`;
      }
      html += `</div>`;
    }
  }

  html += '<div class="audit-section-title">Audit events</div>';
  if (!events.length) {
    html += '<div style="color:var(--muted);font-size:13px;">No events recorded.</div>';
  } else {
    for (const e of events) {
      const meta = e.metadata ? JSON.stringify(e.metadata, null, 2) : "";
      html += `<div class="audit-event">`
        + `<div class="audit-event-header">`
        + `<span class="audit-event-type">${_esc(e.event_type)}</span>`
        + `<span class="audit-event-time">${_esc(e.created_at || "")}</span>`
        + (e.report_id ? `<span style="color:var(--muted);font-size:11px;">report #${_esc(e.report_id)}</span>` : "")
        + `</div>`;
      if (meta) {
        html += `<div class="audit-event-meta">${_esc(meta)}</div>`;
      }
      html += `</div>`;
    }
  }
  return html;
}

function closeAuditModal() {
  $("audit-modal").style.display = "none";
}

// ---------------------------------------------------------------------------
// Phase 2: NLP QA layer
// ---------------------------------------------------------------------------
function _clearQaPanel() {
  const p = $("qa-panel");
  if (!p) return;
  p.style.display = "none";
  p.innerHTML = "";
}

let _qaCheckedReport = "";

function _markReportChanged() {
  state.reportCopied = false;
  _qaCheckedReport = "";
  _clearQaPanel();
}

function _orderedSideFromBodyPart(bodyPart) {
  const text = String(bodyPart || "").toLowerCase();
  if (/\bbilat(?:eral)?\b/.test(text)) return "bilateral";
  if (/\bleft\b/.test(text)) return "left";
  if (/\bright\b/.test(text)) return "right";
  return null;
}

function _renderQaPanel(flags) {
  const p = $("qa-panel");
  if (!p) return;
  const _esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const icon = { error: "✕", warning: "!", info: "i" };

  let html = '<div class="qa-panel-header">';
  if (!flags.length) {
    html += '<span>QA</span></div><div class="qa-empty">✓ No flags raised. Always confirm clinically before signing off.</div>';
    p.innerHTML = html;
    p.style.display = "";
    return;
  }
  html += `<span>QA — ${flags.length} flag${flags.length === 1 ? "" : "s"}</span></div>`;
  for (const f of flags) {
    const sev = f.severity || "info";
    html += `<div class="qa-flag qa-flag-${_esc(sev)}">`
      + `<span class="qa-flag-icon ${_esc(sev)}">${icon[sev] || "i"}</span>`
      + `<div class="qa-flag-body">`
      + `<div class="qa-flag-message">${_esc(f.message)}</div>`
      + (f.location ? `<code class="qa-flag-location">${_esc(f.location)}</code>` : "")
      + `</div>`
      + `<button class="qa-flag-dismiss" title="Dismiss">×</button>`
      + `</div>`;
  }
  p.innerHTML = html;
  p.style.display = "";

  // Wire dismiss buttons
  p.querySelectorAll(".qa-flag-dismiss").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const flag = e.currentTarget.closest(".qa-flag");
      if (flag) flag.remove();
      const remaining = p.querySelectorAll(".qa-flag");
      const header = p.querySelector(".qa-panel-header span");
      if (header) header.textContent = `QA — ${remaining.length} flag${remaining.length === 1 ? "" : "s"}`;
      if (!remaining.length) _clearQaPanel();
    });
  });
}

async function runQaCheck({ quiet = false } = {}) {
  const text = ($("report-raw")?.value || "").trim();
  if (!text) {
    if (!quiet) setStatus("Nothing to check.", "error");
    return [];
  }
  // Best-effort: send the patient context we have. Infer ordered laterality
  // from body-part labels such as "Right Knee" so the deterministic check can
  // catch an opposite-side report without another field or click.
  const body = {
    report_text: text,
    source_text: ($("transcription")?.value || "").trim() || null,
    source_kind: state.sourceKind === "worksheet" ? "worksheet" : "dictation",
    accession: $("accession")?.value.trim() || null,
    body_part: $("body-part")?.value.trim() || null,
    ordered_side: _orderedSideFromBodyPart($("body-part")?.value),
  };
  try {
    if (!quiet) setStatus("Running QA checks…", "active");
    const resp = await fetch("/api/qa-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    // Ignore a stale response if the report changed while QA was running.
    if ((($("report-raw")?.value || "").trim()) !== text) return [];
    const flags = data.flags || [];
    _qaCheckedReport = text;
    if (flags.length === 0 && quiet) {
      _clearQaPanel();
    } else {
      _renderQaPanel(flags);
    }
    if (flags.length === 0) {
      if (!quiet) setStatus("QA: no flags raised.", "success");
    } else if (!quiet) {
      setStatus(`QA: ${flags.length} flag${flags.length === 1 ? "" : "s"} raised.`, "active");
    }
    return flags;
  } catch (err) {
    if (!quiet) setStatus(`QA check failed: ${err.message}`, "error");
    return null;
  }
}

// ---------------------------------------------------------------------------
// Canvas resize observer
// ---------------------------------------------------------------------------
function initCanvasResize() {
  const canvas = $("waveform");
  const ro = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    if (!state.isRecording) stopWaveform();
  });
  ro.observe(canvas);
  canvas.width = canvas.clientWidth;
  canvas.height = canvas.clientHeight;
  stopWaveform();
}

// ---------------------------------------------------------------------------
// Voice-feedback report refinement
// ---------------------------------------------------------------------------
const fbState = {
  isRecording: false,
  isProcessing: false,
  mediaRecorder: null,
  stream: null,
  chunks: [],
  selectedText: "",   // passage selected in report-rendered when Refine was clicked
};

function _captureReportSelection() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return "";
  const reportDiv = $("report-rendered");
  if (!reportDiv || !reportDiv.contains(sel.anchorNode)) return "";
  return sel.toString().trim();
}

function _resetFeedbackUI() {
  $("feedback-bar").style.display = "none";
  $("btn-refine-stop").disabled = false;
  fbState.selectedText = "";
  fbState.isRecording = false;
  fbState.isProcessing = false;
  // Re-evaluate Refine button based on current report state
  const hasReport = !!$("report-raw").value.trim();
  $("btn-refine").disabled = !hasReport;
  setUI(_inferUIMode());
}

async function startFeedback() {
  if (state.isRecording || fbState.isRecording || fbState.isProcessing) return;
  fbState.isProcessing = true;
  syncDesktopRefine();

  fbState.selectedText = _captureReportSelection();

  try {
    fbState.stream = await RadSpeedMicrophone.open();
  } catch (err) {
    _resetFeedbackUI();
    setStatus(`Microphone error: ${err.message}`, "error");
    return;
  }

  try {
    fbState.chunks = [];
    fbState.mediaRecorder = new MediaRecorder(fbState.stream);
    fbState.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) fbState.chunks.push(e.data);
    };
    fbState.mediaRecorder.onstop = _onFeedbackStop;
    fbState.mediaRecorder.start();
  } catch (err) {
    RadSpeedMicrophone.stop(fbState.stream);
    fbState.stream = null;
    _resetFeedbackUI();
    setStatus(`Could not start refinement: ${err.message}`, "error");
    return;
  }
  fbState.isRecording = true;
  fbState.isProcessing = false;
  syncDesktopRefine();
  setStatus("Listening for refinement. Press Stop refine when finished.", "active");

  // Show feedback bar, hide Refine button
  $("btn-refine").disabled = true;
  $("feedback-bar").style.display = "flex";
  $("feedback-status").textContent = fbState.selectedText
    ? `Listening… targeting "${fbState.selectedText.slice(0, 50).trimEnd()}${fbState.selectedText.length > 50 ? "…" : ""}"`
    : "Listening for feedback…";
}

function stopFeedback() {
  if (!fbState.isRecording || !fbState.mediaRecorder) return;
  $("feedback-status").textContent = "Processing…";
  $("btn-refine-stop").disabled = true;
  fbState.isProcessing = true;
  syncDesktopRefine();
  setStatus("Applying refinement…", "active");
  fbState.mediaRecorder.stop();
}

async function _onFeedbackStop() {
  if (fbState.stream) {
    RadSpeedMicrophone.stop(fbState.stream);
    fbState.stream = null;
  }
  fbState.isRecording = false;

  const blob = new Blob(fbState.chunks, { type: "audio/webm" });
  if (blob.size < MIN_SEGMENT_BYTES) {
    _resetFeedbackUI();
    setStatus("No speech detected — feedback unchanged.", "error");
    return;
  }

  // Transcribe the feedback dictation
  const fd = new FormData();
  fd.append("audio", blob, "feedback.webm");

  let feedbackText;
  try {
    const resp = await fetch("/transcribe", { method: "POST", body: fd });
    if (!resp.ok) throw new Error("Transcription failed");
    const data = await resp.json();
    feedbackText = data.transcription?.trim();
  } catch (err) {
    _resetFeedbackUI();
    setStatus(`Feedback transcription error: ${err.message}`, "error");
    return;
  }

  if (!feedbackText) {
    _resetFeedbackUI();
    setStatus("No speech detected — feedback unchanged.", "error");
    return;
  }

  // Apply the feedback to the report
  $("feedback-status").textContent = "Applying feedback…";
  try {
    const resp = await fetch("/format/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        report: $("report-raw").value,
        feedback: feedbackText,
        selected_text: fbState.selectedText || "",
        patient_name:        ($("patient-name")?.value || "").trim()        || null,
        patient_dob:         ($("patient-dob")?.value || "").trim()         || null,
        patient_id:          ($("patient-id")?.value || "").trim()          || null,
        accession:           ($("accession")?.value || "").trim()           || null,
        referring_physician: ($("referring-physician")?.value || "").trim() || null,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    setReport(data.report);
    _resetFeedbackUI();
    setStatus("Report updated.", "success");
    runQaCheck({ quiet: true });
  } catch (err) {
    _resetFeedbackUI();
    setStatus(`Feedback error: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Template editor modal
// ---------------------------------------------------------------------------
let _tmplData = [];         // [{name, is_custom}]
let _tmplCurrent = null;    // {name, is_custom}
let _tmplDirty = false;

async function tmplOpen() {
  $("tmpl-modal").style.display = "flex";
  await _tmplFetchList();
}

function tmplClose() {
  if (_tmplDirty && !confirm("You have unsaved changes. Close anyway?")) return;
  $("tmpl-modal").style.display = "none";
  _tmplDirty = false;
}

async function _tmplFetchList() {
  const r = await fetch("/templates");
  const data = await r.json();
  _tmplData = data.templates;   // [{name, is_custom}]
  _tmplRenderList();
}

function _tmplRenderList() {
  const q = $("tmpl-search").value.toLowerCase();
  const ul = $("tmpl-list");
  ul.innerHTML = "";
  _tmplData
    .filter(t => !q || t.name.toLowerCase().includes(q))
    .forEach(t => {
      const li = document.createElement("li");
      li.dataset.name = t.name;
      if (_tmplCurrent && t.name === _tmplCurrent.name) li.classList.add("tmpl-active");
      const label = t.name.replace(/_/g, " ").replace(/\.(txt|md)$/, "");
      li.innerHTML = `<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${label}</span>`
        + (t.is_custom ? `<span class="tmpl-badge">Custom</span>` : "");
      li.addEventListener("click", () => _tmplSelect(t.name));
      ul.appendChild(li);
    });
}

async function _tmplSelect(name) {
  if (_tmplDirty && !confirm("You have unsaved changes. Switch template anyway?")) return;
  _tmplDirty = false;

  document.querySelectorAll("#tmpl-list li").forEach(li =>
    li.classList.toggle("tmpl-active", li.dataset.name === name)
  );

  $("tmpl-status").textContent = "Loading…";
  const r = await fetch(`/api/templates/${encodeURIComponent(name)}`);
  if (!r.ok) { $("tmpl-status").textContent = "Failed to load."; return; }
  const data = await r.json();

  _tmplCurrent = { name, is_custom: data.is_custom, has_split: !!data.has_split };
  const titleEl = $("tmpl-editor-title");
  titleEl.textContent = name.replace(/_/g, " ").replace(/\.(txt|md)$/, "");
  titleEl.classList.remove("tmpl-dirty");

  // Populate the two sections. When a template predates the split format the
  // whole file lands in "Report structure" so nothing is lost; saving it
  // converts it to the new two-section format.
  if (data.has_split) {
    $("tmpl-content").value = data.structure || "";
    $("tmpl-content-ai").value = data.ai_instructions || "";
  } else {
    $("tmpl-content").value = data.content || "";
    $("tmpl-content-ai").value = "";
  }
  $("tmpl-tabs").style.display = "flex";
  _tmplSwitchTab("structure");
  $("tmpl-split-note").textContent = data.has_split
    ? ""
    : "Not split yet — save to separate AI instructions";

  $("tmpl-save").disabled = false;
  $("tmpl-duplicate").disabled = false;
  $("tmpl-restore").style.display = data.is_custom ? "inline-flex" : "none";
  $("tmpl-status").textContent = data.is_custom ? "Custom version" : "Bundled default";
}

function _tmplSwitchTab(pane) {
  document.querySelectorAll(".tmpl-tab").forEach(b =>
    b.classList.toggle("tmpl-tab-active", b.dataset.pane === pane)
  );
  $("tmpl-content").style.display = pane === "structure" ? "block" : "none";
  $("tmpl-content-ai").style.display = pane === "ai" ? "block" : "none";
  (pane === "structure" ? $("tmpl-content") : $("tmpl-content-ai")).focus();
}

function _tmplMarkDirty() {
  if (!_tmplCurrent) return;
  _tmplDirty = true;
  $("tmpl-editor-title").classList.add("tmpl-dirty");
}

async function tmplSave() {
  if (!_tmplCurrent) return;
  $("tmpl-status").textContent = "Saving…";
  $("tmpl-save").disabled = true;
  const r = await fetch(`/api/templates/${encodeURIComponent(_tmplCurrent.name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      structure: $("tmpl-content").value,
      ai_instructions: $("tmpl-content-ai").value,
    }),
  });
  $("tmpl-save").disabled = false;
  if (r.ok) {
    _tmplDirty = false;
    _tmplCurrent.is_custom = true;
    _tmplCurrent.has_split = true;
    $("tmpl-split-note").textContent = "";
    $("tmpl-editor-title").classList.remove("tmpl-dirty");
    $("tmpl-restore").style.display = "inline-flex";
    $("tmpl-status").textContent = "Saved ✓  (Custom version)";
    await _tmplFetchList();
  } else {
    $("tmpl-status").textContent = "Save failed.";
  }
}

async function tmplRestore() {
  if (!_tmplCurrent) return;
  const label = _tmplCurrent.name.replace(/_/g, " ").replace(/\.(txt|md)$/, "");
  if (!confirm(`Restore "${label}" to the bundled default? Your customisation will be deleted.`)) return;
  const r = await fetch(`/api/templates/${encodeURIComponent(_tmplCurrent.name)}`, { method: "DELETE" });
  if (r.ok) {
    _tmplDirty = false;
    await _tmplSelect(_tmplCurrent.name);
    await _tmplFetchList();
  }
}

async function tmplDuplicate() {
  if (!_tmplCurrent) return;
  const base = _tmplCurrent.name.replace(/\.(txt|md)$/, "");
  const raw = prompt("New template filename (no extension):", `${base}_copy`);
  if (!raw || !raw.trim()) return;
  const filename = raw.trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_\-. ]/g, "") + ".txt";
  const r = await fetch(`/api/templates/${encodeURIComponent(filename)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      structure: $("tmpl-content").value,
      ai_instructions: $("tmpl-content-ai").value,
    }),
  });
  if (r.ok) {
    await _tmplFetchList();
    await _tmplSelect(filename);
    _tmplSyncDropdown();
  }
}

async function tmplNew() {
  const raw = prompt("New template filename (no extension):", "My_Template");
  if (!raw || !raw.trim()) return;
  const filename = raw.trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_\-. ]/g, "") + ".txt";
  const r = await fetch(`/api/templates/${encodeURIComponent(filename)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ structure: "", ai_instructions: "" }),
  });
  if (r.ok) {
    await _tmplFetchList();
    await _tmplSelect(filename);
    _tmplSyncDropdown();
  }
}

// ---------------------------------------------------------------------------
// Template upload
// ---------------------------------------------------------------------------
const _TMPL_WRAP_HEADER = `### THIS IS CONTINUATION OF SYSTEM PROMPT ###
### THIS IS THE SAMPLE TEMPLATE TO BE USED FOR THE TRANSCRIPT ###

---

### Clinical Details
**Instructions:**
- **If provided:** 'The patient presented with [describe symptoms or reason for the examination], with a medical history notable for [mention relevant medical conditions].'
  - **Exclude** name, age, gender, and IDs anywhere in the report, even if provided in the transcript.
- **If not provided:** 'No clinical details provided.'

### Comparison:
**Instructions:**
- **If provided:** Mention any prior imaging studies and briefly note significant changes or findings compared to previous examinations.
- **If not provided:** 'No prior imaging scans/reports available for comparison.'

### Findings:
**Instructions:**
- Describe pathological structures specifically. Normal structures within the same anatomical group may be combined into a single statement — do not force a separate bullet for every structure when all are normal.
- For any structure the radiologist did not comment on, use standard normal radiological terminology.
- Follow the structure below.

`;

const _TMPL_WRAP_FOOTER = `

### Impression:
**Instructions:**
- Concise bullet-point summary of key positive findings.
- Omit normal findings from the impression unless clinically significant.

---
`;

let _tmplUploadFile = null;   // File object

function _tmplSanitizeFilename(raw) {
  if (!raw) return "";
  let name = raw.trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_\-. ]/g, "");
  if (!/\.(txt|md)$/i.test(name)) name += ".txt";
  return name;
}

function _tmplUploadValidate() {
  const nameInput = $("tmpl-upload-name");
  const confirmBtn = $("tmpl-upload-confirm");
  const warn = $("tmpl-upload-warn");
  const raw = nameInput.value.trim();
  if (!_tmplUploadFile || !raw) {
    confirmBtn.disabled = true;
    warn.style.display = "none";
    return;
  }
  if (!/^[A-Za-z0-9_\-. ]+\.(txt|md)$/i.test(raw)) {
    confirmBtn.disabled = true;
    warn.textContent = "⚠ Filename must contain only letters, numbers, spaces, dots, dashes, underscores, and end in .txt or .md";
    warn.style.display = "block";
    return;
  }
  const clash = _tmplData.find(t => t.name.toLowerCase() === raw.toLowerCase());
  if (clash) {
    warn.textContent = clash.is_custom
      ? `⚠ "${raw}" already exists as a custom template — uploading will overwrite it.`
      : `⚠ "${raw}" matches a bundled template — uploading will create a custom version that overrides it.`;
    warn.style.display = "block";
  } else {
    warn.style.display = "none";
  }
  confirmBtn.disabled = false;
}

function _tmplUploadOpen(file) {
  _tmplUploadFile = file;
  $("tmpl-upload-source").textContent = `Source: ${file.name} (${Math.round(file.size / 1024 * 10) / 10} KB)`;
  $("tmpl-upload-name").value = _tmplSanitizeFilename(file.name);
  $("tmpl-upload-mode-asis").checked = true;
  $("tmpl-upload-warn").style.display = "none";
  $("tmpl-upload-modal").style.display = "flex";
  _tmplUploadValidate();
  setTimeout(() => $("tmpl-upload-name").focus(), 0);
}

function _tmplUploadClose() {
  $("tmpl-upload-modal").style.display = "none";
  $("tmpl-upload-input").value = "";
  _tmplUploadFile = null;
}

async function _tmplUploadConfirm() {
  if (!_tmplUploadFile) return;
  const filename = _tmplSanitizeFilename($("tmpl-upload-name").value);
  if (!filename) return;

  const mode = document.querySelector('input[name="tmpl_upload_mode"]:checked').value;
  let content;
  try {
    content = await _tmplUploadFile.text();
  } catch (_) {
    $("tmpl-upload-warn").textContent = "⚠ Failed to read file.";
    $("tmpl-upload-warn").style.display = "block";
    return;
  }
  if (mode === "wrap") {
    content = _TMPL_WRAP_HEADER + content.trimEnd() + _TMPL_WRAP_FOOTER;
  }

  const clash = _tmplData.find(t => t.name.toLowerCase() === filename.toLowerCase());
  if (clash && clash.is_custom && !confirm(`"${filename}" already exists. Overwrite?`)) return;

  const confirmBtn = $("tmpl-upload-confirm");
  confirmBtn.disabled = true;
  const r = await fetch(`/api/templates/${encodeURIComponent(filename)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  });
  if (r.ok) {
    _tmplUploadClose();
    await _tmplFetchList();
    await _tmplSelect(filename);
    _tmplSyncDropdown();
    $("tmpl-status").textContent = "Uploaded ✓";
  } else {
    $("tmpl-upload-warn").textContent = "⚠ Upload failed.";
    $("tmpl-upload-warn").style.display = "block";
    confirmBtn.disabled = false;
  }
}

function _tmplSyncDropdown() {
  fetch("/templates")
    .then(r => r.json())
    .then(data => {
      const sel = $("template-select");
      const current = sel.value;
      sel.innerHTML = '<option value="">Auto-select (AI chooses)</option>';
      data.templates.forEach(t => {
        const opt = document.createElement("option");
        opt.value = t.name;
        opt.textContent = t.name.replace(/_/g, " ").replace(/\.(txt|md)$/, "");
        if (t.name === current) opt.selected = true;
        sel.appendChild(opt);
      });
      _updateCtCapLayoutVisibility();
    });
}

document.addEventListener("DOMContentLoaded", async () => {
  initDesktopOverlay();
  initCanvasResize();
  setUI("idle");
  setStatus("Press Record to start dictating.");
  _initCtCapLayout();
  $("template-select")?.addEventListener("change", _updateCtCapLayoutVisibility);

  // Editing the report after a copy re-arms the unsaved-work guard.
  const _reportRaw = $("report-raw");
  if (_reportRaw) _reportRaw.addEventListener("input", _markReportChanged);

  // Warn before navigating away from an uncopied, unsigned report or a
  // non-empty transcription — closing/refreshing the tab otherwise loses it.
  window.addEventListener("beforeunload", (e) => {
    const rpt = ($("report-raw")?.value || "").trim();
    const tx  = ($("transcription")?.value || "").trim();
    const unsaved = (rpt && !state.reportCopied && !_signedReportId) || (tx && !rpt);
    if (unsaved) {
      e.preventDefault();
      e.returnValue = "";
      return "";
    }
  });

  // Track the last known selection in the transcription/report textareas.
  // _pendingSelection is set on every non-zero selection event and cleared only
  // when consumed by _grabVoiceEdit or when focus leaves for a non-Record target.
  let _pendingSelection = null;
  // Timestamp of the last pointerdown on the Record button. Used to suppress
  // _pendingSelection clearing when blur is triggered by clicking Record — on
  // Safari/iOS, buttons don't receive focus so relatedTarget is null, causing
  // the blur handler to incorrectly wipe the selection before we can use it.
  let _recordPointerdownAt = 0;

  ["transcription", "report-raw"].forEach(id => {
    const el = $(id);
    if (!el) return;
    const save = () => {
      const s = el.selectionStart, e = el.selectionEnd;
      if (s !== e) {
        _pendingSelection = { elId: id, start: s, end: e, selectedText: el.value.slice(s, e) };
      }
    };
    el.addEventListener("mouseup", save);
    el.addEventListener("select", save);
    el.addEventListener("keyup", save);
    el.addEventListener("touchend", save);  // iOS: fires after selection gesture
    // Clear _pendingSelection when focus leaves, UNLESS the user just pressed
    // the Record button (within 500 ms). On Safari/iOS, buttons don't get focus
    // so relatedTarget is null even when Record was clicked — the timestamp check
    // prevents us from wiping the selection we need for voice-edit.
    el.addEventListener("blur", (evt) => {
      const toRecord = evt.relatedTarget && evt.relatedTarget.id === "btn-record";
      const justPressedRecord = Date.now() - _recordPointerdownAt < 500;
      if (!toRecord && !justPressedRecord) {
        _pendingSelection = null;
      }
    });
  });

  // Global selectionchange tracker — most reliable across browsers/platforms.
  // Fires on every selection mutation; we snapshot non-zero selections on the
  // two editable textareas. Never clears _pendingSelection on collapse, since
  // a collapse during a Record tap (e.g. Safari/iOS button tap) must not wipe
  // the last real selection before _grabVoiceEdit consumes it.
  document.addEventListener("selectionchange", () => {
    const active = document.activeElement;
    if (!active || (active.id !== "transcription" && active.id !== "report-raw")) return;
    const s = active.selectionStart, e = active.selectionEnd;
    if (s != null && e != null && s !== e) {
      const sel = { elId: active.id, start: s, end: e, selectedText: active.value.slice(s, e) };
      _pendingSelection = sel;
      // If already recording (user editing transcript mid-session), track as
      // a pending voice-edit so the next VAD cut does a splice not an append.
      if (state.isRecording && !state.streamingWs) {
        state.pendingVoiceEditSelection = sel;
      }
    } else if (s === e) {
      // Collapsed — clear the mid-recording selection so clicking elsewhere
      // during a session doesn't accidentally voice-edit the wrong span.
      state.pendingVoiceEditSelection = null;
    }
  });

  // Capture the textarea selection at pointerdown time.
  // Three-tier fallback to handle cross-browser selection clearing on blur:
  //   1. Read selectionStart/End directly from #transcription (most browsers
  //      preserve these even after blur).
  //   2. _pendingSelection saved by mouseup/select/keyup on the textarea.
  //      The blur handler won't clear this if the Record button was just pressed
  //      (timestamp guard handles Safari/iOS where buttons don't receive focus).
  //   3. _pendingSelection for report-raw editing.
  const _grabVoiceEdit = () => {
    _recordPointerdownAt = Date.now(); // timestamp for blur handler guard

    // Tier 0: selection in the rendered report div — survives button click.
    const domSel = window.getSelection();
    if (domSel && !domSel.isCollapsed) {
      const reportDiv = $("report-rendered");
      if (reportDiv && reportDiv.contains(domSel.anchorNode)) {
        const selectedText = domSel.toString().trim();
        if (selectedText) {
          state.voiceEditTarget = { elId: "report-rendered", selectedText };
          console.log("[voice-edit] captured rendered (tier0):", JSON.stringify(selectedText));
          return;
        }
      }
    }

    // Tier 1: direct read from either editable textarea.
    for (const id of ["transcription", "report-raw"]) {
      const el = $(id);
      if (el && el.selectionStart !== el.selectionEnd) {
        const s = el.selectionStart, e = el.selectionEnd;
        state.voiceEditTarget = { elId: id, start: s, end: e, selectedText: el.value.slice(s, e) };
        _pendingSelection = null;
        console.log("[voice-edit] captured (tier1):", id, s, e, JSON.stringify(el.value.slice(s, e)));
        return;
      }
    }
    // Tier 2 & 3: _pendingSelection (saved by mouseup/select/keyup/touchend/selectionchange)
    if (_pendingSelection) {
      state.voiceEditTarget = _pendingSelection;
      console.log("[voice-edit] captured (tier2):",
        _pendingSelection.elId, _pendingSelection.start, _pendingSelection.end,
        JSON.stringify(_pendingSelection.selectedText));
      _pendingSelection = null;
    } else {
      console.log("[voice-edit] no selection captured on pointerdown");
    }
    // If no selection is detected on this pointerdown, DO NOT clear an existing
    // voiceEditTarget: a rapid double-tap or repeat pointerdown during a voice-
    // edit session must not cancel the pending edit. The target is cleared only
    // by submitAudioSegment (after splice) or startStreamingRecording (consume).
  };

  // Live cursor tracking during streaming: when the user clicks or moves the
  // cursor inside #transcription while streaming is active, immediately resplit
  // the text so the next spoken words appear at the new cursor position.
  // This gives PowerScribe-style behaviour — no button presses needed.
  document.addEventListener("selectionchange", () => {
    if (_suppressStreamingSelChange) return;
    if (!state.streamingWs || !state.isRecording) return;
    const tx = $("transcription");
    if (document.activeElement !== tx) return;

    const displayPos = tx.selectionStart;
    const displayEnd = tx.selectionEnd;
    if (displayPos === state.streamingAnchorPos &&
        displayEnd === state.streamingAnchorEnd) return; // nothing changed

    // "Stable text" = what the textarea contains minus any trailing interim word.
    // For mid-text insertion we suppress interim, so stableText === tx.value.
    // For end-of-text, interim is at the tail — strip it.
    const interimSuffix = (!state.streamingAfter && state.interimText)
      ? (state.confirmedText ? " " : "") + state.interimText
      : "";
    const stableLen  = tx.value.length - interimSuffix.length;
    const newPos     = Math.min(displayPos, stableLen);
    const newEnd     = Math.min(displayEnd, stableLen);

    // Resplit: everything up to cursor becomes "before", everything after becomes "after".
    // Confirmed text is now baked into whichever side it falls on.
    const stableText = tx.value.slice(0, stableLen);
    state.streamingBefore = stableText.slice(0, newPos);
    state.streamingAfter  = stableText.slice(newEnd);
    // Store selected text so _updateStreamingDisplay can keep it highlighted
    state.streamingSelectedText = (newPos !== newEnd)
      ? stableText.slice(newPos, newEnd)
      : "";
    state.streamingBakedServerText = state.streamingServerConfirmedText;
    state.streamingCursorRepositioned = true;
    state.confirmedText   = "";
    state.interimText     = "";

    _suppressStreamingSelChange = true;
    // Only strip trailing interim — don't erase the selected text.
    if (interimSuffix) {
      tx.value = tx.value.slice(0, stableLen);
    }
    tx.selectionStart = newPos;
    tx.selectionEnd   = newEnd;
    state.streamingAnchorPos = newPos;
    state.streamingAnchorEnd = newEnd;
    setTimeout(() => { _suppressStreamingSelChange = false; }, 0);
  });
  // Use pointerdown (handles both mouse and touch, fires earliest in event chain).
  $("btn-record").addEventListener("pointerdown", _grabVoiceEdit, { passive: true });
  $("btn-record").addEventListener("click", onRecordClick);
  $("btn-stop").addEventListener("click", stopRecording);
  $("btn-format").addEventListener("click", formatReport);
  $("btn-use-cap-components")?.addEventListener("click", () => useCtCapDraft("components"));
  $("btn-use-cap-regions")?.addEventListener("click", () => useCtCapDraft("regions"));
  if ($("btn-impression")) $("btn-impression").addEventListener("click", generateImpressionOnly);
  $("btn-copy").addEventListener("click", copyReport);
  if ($("btn-copy-from-comparison")) {
    $("btn-copy-from-comparison").addEventListener(
      "click",
      () => copyReport({ fromComparison: true }),
    );
  }
  $("btn-edit-toggle").addEventListener("click", toggleReportEdit);
  $("btn-lookup").addEventListener("click", lookupPatient);
  if ($("btn-next-case")) $("btn-next-case").addEventListener("click", () => nextCase());

  // Shared worksheet / indication screenshot paste, choose and drag-and-drop.
  const worksheetZone = $("worksheet-drop-zone");
  const worksheetInput = $("worksheet-file-input");
  if (worksheetZone && worksheetInput) {
    worksheetZone.addEventListener("click", () => worksheetInput.click());
    worksheetZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        worksheetInput.click();
      }
    });
    worksheetInput.addEventListener("change", () => {
      addWorksheetImages(worksheetInput.files);
      worksheetInput.value = "";
    });
    $("btn-worksheet-choose").addEventListener("click", () => worksheetInput.click());
    $("btn-worksheet-clear").addEventListener("click", clearWorksheetImages);
    $("btn-indication-transcribe").addEventListener(
      "click",
      () => transcribeIndicationImages(_worksheetImages.map((item) => item.file)),
    );
    $("btn-worksheet-generate").addEventListener("click", () => generateFromWorksheet(false));
    $("btn-worksheet-one-pass").addEventListener("click", () => generateFromWorksheet(true));
    $("btn-indication-copy").addEventListener("click", copyIndication);
    $("indication-text").addEventListener("input", () => {
      $("btn-indication-copy").disabled = !$("indication-text").value.trim();
    });

    for (const eventName of ["dragenter", "dragover"]) {
      worksheetZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        worksheetZone.classList.add("drag-over");
      });
    }
    for (const eventName of ["dragleave", "drop"]) {
      worksheetZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        worksheetZone.classList.remove("drag-over");
      });
    }
    worksheetZone.addEventListener("drop", (event) => {
      addWorksheetImages(event.dataTransfer?.files || []);
    });
  }

  document.addEventListener("paste", (event) => {
    const files = Array.from(event.clipboardData?.items || [])
      .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
      .map((item) => item.getAsFile())
      .filter(Boolean);
    if (!files.length) return;
    event.preventDefault();
    addWorksheetImages(files);
    worksheetZone?.scrollIntoView({ behavior: "smooth", block: "center" });
  });

  // Keep the active patient visible when the detailed form is collapsed.
  let _priorDebounce = null;
  for (const id of ["patient-name", "patient-dob", "patient-id", "accession", "modality", "body-part"]) {
    const field = $(id);
    if (!field) continue;
    field.addEventListener("input", () => {
      updatePatientSummary();
      if (id === "patient-id" || id === "accession") {
        clearTimeout(_priorDebounce);
        _priorDebounce = setTimeout(refreshPriorReports, 350);
      }
    });
  }
  if ($("prior-select")) $("prior-select").addEventListener("change", onPriorSelected);

  // Follow-up register and standardised assessment inserter.
  if ($("btn-followups")) $("btn-followups").addEventListener("click", () => openFollowupModal());
  if ($("followup-modal-close")) $("followup-modal-close").addEventListener("click", closeFollowupModal);
  if ($("followup-modal")) $("followup-modal").addEventListener("click", (e) => {
    if (e.target === $("followup-modal")) closeFollowupModal();
  });
  if ($("followup-save")) $("followup-save").addEventListener("click", saveFollowup);
  if ($("btn-scores")) $("btn-scores").addEventListener("click", openScoreModal);
  if ($("score-modal-close")) $("score-modal-close").addEventListener("click", closeScoreModal);
  if ($("score-modal")) $("score-modal").addEventListener("click", (e) => {
    if (e.target === $("score-modal")) closeScoreModal();
  });
  if ($("score-system")) $("score-system").addEventListener("change", _updateScoreCategories);
  if ($("score-category")) $("score-category").addEventListener("change", _updateScorePreview);
  if ($("score-target")) $("score-target").addEventListener("input", _updateScorePreview);
  if ($("score-insert")) $("score-insert").addEventListener("click", insertScore);
  updatePatientSummary();
  if (document.body.dataset.oauthMode === "true") loadFollowups();

  // Phase 1 / 2: sign-off, amendments, audit, QA
  if ($("btn-qa")) $("btn-qa").addEventListener("click", runQaCheck);
  if ($("btn-sign-off")) $("btn-sign-off").addEventListener("click", signOffReport);
  if ($("btn-amend")) $("btn-amend").addEventListener("click", openAmendModal);
  if ($("btn-audit")) $("btn-audit").addEventListener("click", openAuditModal);
  if ($("audit-modal-close")) $("audit-modal-close").addEventListener("click", closeAuditModal);
  if ($("audit-modal")) $("audit-modal").addEventListener("click", (e) => {
    if (e.target === $("audit-modal")) closeAuditModal();
  });
  if ($("amend-modal-close")) $("amend-modal-close").addEventListener("click", closeAmendModal);
  if ($("amend-cancel")) $("amend-cancel").addEventListener("click", closeAmendModal);
  if ($("amend-confirm")) $("amend-confirm").addEventListener("click", confirmAmend);
  if ($("amend-modal")) $("amend-modal").addEventListener("click", (e) => {
    if (e.target === $("amend-modal")) closeAmendModal();
  });
  // Re-evaluate Audit button enabled state when accession changes
  if ($("accession")) $("accession").addEventListener("input", () => {
    const btn = $("btn-audit");
    if (btn) btn.disabled = !($("accession").value || "").trim();
  });
  if ($("body-part")) $("body-part").addEventListener("input", () => {
    _qaCheckedReport = "";
    _clearQaPanel();
  });

  // HL7 worklist
  if ($("worklist-select")) {
    $("worklist-select").addEventListener("change", applyWorklistOrder);
    $("btn-worklist-refresh").addEventListener("click", refreshWorklist);
    $("btn-worklist-archive").addEventListener("click", archiveWorklistOrder);
    document.querySelectorAll(".worklist-chip").forEach((chip) => {
      chip.addEventListener("click", () => _setWorklistFilter(chip.dataset.modality || ""));
    });
    refreshWorklist();
    // Refresh waiting-time labels every 30s without re-fetching the inbox.
    setInterval(() => {
      if ($("worklist-panel") && $("worklist-panel").style.display !== "none") {
        _renderWorklistOptions();
      }
    }, 30000);
  }

  // Voice feedback
  $("btn-refine").addEventListener("click", startFeedback);
  $("btn-refine-stop").addEventListener("click", stopFeedback);

  // Vocabulary suggestion banner
  if ($("vocab-suggest-accept")) {
    $("vocab-suggest-accept").addEventListener("click", acceptVocabSuggest);
    $("vocab-suggest-dismiss").addEventListener("click", hideVocabSuggest);
  }

  // Style-setting suggestion banner
  if ($("style-suggest-apply")) {
    $("style-suggest-apply").addEventListener("click", applyStyleSuggest);
    $("style-suggest-dismiss").addEventListener("click", dismissStyleSuggest);
  }

  // Keyboard-edit detection on the transcript textarea.
  // Fires ~1.5 s after the user pauses typing (debounced input event).
  // Blur acts as a fallback flush so nothing is missed when focus leaves.
  // The backend word-diff isolates the user's replacement from any
  // STT-appended words, so this works during active recording too.
  {
    const tx = $("transcription");
    let _txSnapshot = "";
    let _txDebounceTimer = null;

    async function _checkEditSuggestion(snapshot, current) {
      if (!snapshot || current === snapshot) return;
      try {
        const resp = await fetch("/api/check-edit-suggestion", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ old_text: snapshot, new_text: current }),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.suggest_vocab) showVocabSuggest(data.suggest_vocab);
        if (data.suggest_setting) showStyleSuggest(data.suggest_setting);
      } catch (_) { /* non-critical */ }
    }

    tx.addEventListener("focus", () => { _txSnapshot = tx.value; });

    tx.addEventListener("input", () => {
      // Typed or pasted text is no longer a worksheet-only source. Use the
      // stricter dictation rule even when the user started from worksheet OCR.
      state.sourceKind = "dictation";
      // Pasted or typed dictation should be format-ready just like speech-to-
      // text output. Previously the Re-format button stayed disabled unless a
      // recording session had changed the UI mode first.
      if (!state.isRecording && !($("report-raw")?.value || "").trim()) {
        setUI(tx.value.trim() ? "transcribed" : "idle");
      }
      clearTimeout(_txDebounceTimer);
      _txDebounceTimer = setTimeout(async () => {
        const current = tx.value;
        const snapshot = _txSnapshot;
        if (!snapshot || current === snapshot) return;
        _txSnapshot = current; // slide window so next pause diffs from here
        await _checkEditSuggestion(snapshot, current);
      }, 1500);
    });

    tx.addEventListener("blur", async () => {
      clearTimeout(_txDebounceTimer);
      const current = tx.value;
      const snapshot = _txSnapshot;
      _txSnapshot = "";
      await _checkEditSuggestion(snapshot, current);
    });
  }

  // Template editor
  $("btn-template-edit").addEventListener("click", tmplOpen);
  $("tmpl-close").addEventListener("click", tmplClose);
  $("tmpl-save").addEventListener("click", tmplSave);
  $("tmpl-restore").addEventListener("click", tmplRestore);
  $("tmpl-duplicate").addEventListener("click", tmplDuplicate);
  $("tmpl-new").addEventListener("click", tmplNew);
  $("tmpl-search").addEventListener("input", () => _tmplRenderList());
  $("tmpl-content").addEventListener("input", _tmplMarkDirty);
  $("tmpl-content-ai").addEventListener("input", _tmplMarkDirty);
  document.querySelectorAll(".tmpl-tab").forEach(b =>
    b.addEventListener("click", () => _tmplSwitchTab(b.dataset.pane))
  );
  $("tmpl-modal").addEventListener("click", (e) => { if (e.target === $("tmpl-modal")) tmplClose(); });

  // Upload wiring
  $("tmpl-upload").addEventListener("click", () => $("tmpl-upload-input").click());
  $("tmpl-upload-input").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) _tmplUploadOpen(file);
  });
  $("tmpl-upload-close").addEventListener("click", _tmplUploadClose);
  $("tmpl-upload-cancel").addEventListener("click", _tmplUploadClose);
  $("tmpl-upload-confirm").addEventListener("click", _tmplUploadConfirm);
  $("tmpl-upload-name").addEventListener("input", _tmplUploadValidate);
  $("tmpl-upload-modal").addEventListener("click", (e) => {
    if (e.target === $("tmpl-upload-modal")) _tmplUploadClose();
  });

  document.addEventListener("keydown", (e) => {
    if ($("tmpl-upload-modal").style.display !== "none") {
      if (e.key === "Escape") _tmplUploadClose();
      return;
    }
    if ($("tmpl-modal").style.display !== "none") {
      if (e.key === "Escape") tmplClose();
      if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); tmplSave(); }
      return;
    }
    if (e.repeat) return;
    const commandKey = e.ctrlKey || e.metaKey;

    // Ctrl/Cmd+Shift+C — copy the current report. This shortcut has long been
    // advertised in the button tooltip; wire it for a truly keyboard-first loop.
    if (commandKey && e.shiftKey && e.code === "KeyC") {
      if (!($("report-raw")?.value || "").trim()) return;
      e.preventDefault();
      copyReport();
      return;
    }

    // Ctrl/Cmd+Enter — generate or regenerate from pasted/edited dictation.
    if (commandKey && !e.shiftKey && e.key === "Enter") {
      if (state.isRecording || !($("transcription")?.value || "").trim()) return;
      e.preventDefault();
      formatReport();
      return;
    }

    // Alt/Option+R toggles Record/Pause/Resume; Alt/Option+S stops.
    if (e.altKey && !e.ctrlKey && !e.metaKey && e.code === "KeyR") {
      e.preventDefault();
      onRecordClick();
      return;
    }
    if (e.altKey && !e.ctrlKey && !e.metaKey && e.code === "KeyS") {
      if (!state.isRecording) return;
      e.preventDefault();
      stopRecording();
      return;
    }

    // Alt/Option+N — Next Case. Fires even while focused in the report or
    // transcription field, since the whole point is to transition quickly.
    if (e.altKey && !e.ctrlKey && !e.metaKey && e.code === "KeyN") {
      e.preventDefault();
      nextCase();
      return;
    }
  });

  // Check streaming STT capability
  try {
    const resp = await fetch("/api/capabilities");
    if (resp.ok) {
      const caps = await resp.json();
      state.streamingSupported = !!caps.streaming_stt;
      if (state.streamingSupported) {
        setStatus(
          `Press Record to start dictating. (Streaming STT: ${caps.provider || "enabled"})`,
        );
      }
    }
  } catch (_) {
    // Non-critical — fall back to segment mode silently
  }
});
