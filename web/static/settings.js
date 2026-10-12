"use strict";

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// Badge helpers
// ---------------------------------------------------------------------------
function makeBadge(configured) {
  const span = document.createElement("span");
  span.className = `key-badge ${configured ? "configured" : "missing"}`;
  span.textContent = configured ? "✓ Configured" : "✗ Not set";
  return span;
}

function setBadge(elementId, configured) {
  const el = $(elementId);
  if (!el) return;
  el.innerHTML = "";
  el.appendChild(makeBadge(configured));
}

// ---------------------------------------------------------------------------
// Provider selection UI
// ---------------------------------------------------------------------------
let _currentProvider = "auto";
let _keys = { transcription: false, text: false, deepgram: false, assemblyai: false };

function _updateProviderUI() {
  const val = _currentProvider;
  ["auto", "assemblyai", "deepgram", "groq"].forEach((id) => {
    const opt = $(`opt-${id}`);
    if (opt) opt.classList.toggle("selected", id === val);
  });

  // Show Groq fields only when it is selected explicitly.
  const groqFields = $("groq-fields");
  if (groqFields) groqFields.style.display = val === "groq" ? "" : "none";

  // Show warning if selected provider has no key
  const warn = $("warn-no-key");
  if (warn) {
    const needsKey = val === "deepgram" ? !_keys.deepgram
                   : val === "assemblyai" ? !_keys.assemblyai
                   : false;
    warn.style.display = needsKey ? "" : "none";
  }
}

document.querySelectorAll('input[name="streaming_stt_provider"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    _currentProvider = radio.value;
    _updateProviderUI();
  });
});

// ---------------------------------------------------------------------------
// Load settings from API
// ---------------------------------------------------------------------------
async function loadSettings() {
  try {
    const resp = await fetch("/api/settings");
    if (!resp.ok) throw new Error(resp.statusText);
    const data = await resp.json();

    _keys = data.keys;
    _currentProvider = data.streaming_stt_provider || "auto";

    // Set radio
    const radio = document.querySelector(
      `input[name="streaming_stt_provider"][value="${_currentProvider}"]`
    );
    if (radio) radio.checked = true;

    // Set text fields
    if ($("transcription_base_url")) $("transcription_base_url").value = data.transcription_base_url || "";
    if ($("transcription_model"))    $("transcription_model").value    = data.transcription_model    || "";
    if ($("text_base_url"))          $("text_base_url").value          = data.text_base_url          || "";
    if ($("text_model"))             $("text_model").value             = data.text_model             || "";
    if ($("text-effort"))            $("text-effort").textContent        = data.text_reasoning_effort || "model default";
    if ($("fhir_export_enabled"))    $("fhir_export_enabled").checked  = !!data.fhir_export_enabled;
    if ($("retain_deidentified_samples")) {
      $("retain_deidentified_samples").checked = !!data.retain_deidentified_samples;
      $("retain_deidentified_samples").disabled = !data.quality_retention_available;
    }

    // Shared provider/model settings are read-only for non-administrators.
    if (!data.can_manage_global_settings) {
      document.querySelectorAll("[data-global-setting]").forEach((el) => {
        el.disabled = true;
      });
      const note = $("global-settings-note");
      if (note) note.style.display = "";
    }

    // Reporting style
    const style = data.style || {};
    const styleFields = [
      ["style_spelling",             style.spelling],
      ["style_numerals",             style.numerals],
      ["style_measurement_unit",     style.measurement_unit],
      ["style_measurement_separator", style.measurement_separator],
      ["style_decimal_precision",    style.decimal_precision],
      ["style_laterality",           style.laterality],
      ["style_impression_style",     style.impression_style],
      ["style_negation_phrasing",    style.negation_phrasing],
      ["style_date_format",          style.date_format],
      ["style_paste_format",         style.paste_format],
    ];
    for (const [id, val] of styleFields) {
      const el = $(id);
      if (el && val != null) el.value = String(val);
    }

    // Render badges
    setBadge("badge-groq",       data.keys.transcription);
    setBadge("badge-deepgram",   data.keys.deepgram);
    setBadge("badge-assemblyai", data.keys.assemblyai);
    setBadge("badge-text",       data.keys.text);

    // Key status panel
    setBadge("key-transcription", data.keys.transcription);
    setBadge("key-text",          data.keys.text);
    setBadge("key-deepgram",      data.keys.deepgram);
    setBadge("key-assemblyai",    data.keys.assemblyai);

    _updateProviderUI();
  } catch (err) {
    showStatus(`Failed to load settings: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Save settings
// ---------------------------------------------------------------------------
async function saveSettings() {
  const body = {
    streaming_stt_provider: _currentProvider || "auto",
    transcription_base_url: ($("transcription_base_url") || {}).value || null,
    transcription_model:    ($("transcription_model")    || {}).value || null,
    text_base_url:          ($("text_base_url")          || {}).value || null,
    text_model:             ($("text_model")             || {}).value || null,
    fhir_export_enabled:    !!($("fhir_export_enabled")  || {}).checked,
    retain_deidentified_samples: !!($("retain_deidentified_samples") || {}).checked,
    style_spelling:              ($("style_spelling")              || {}).value || null,
    style_numerals:              ($("style_numerals")              || {}).value || null,
    style_measurement_unit:      ($("style_measurement_unit")      || {}).value || null,
    style_measurement_separator: ($("style_measurement_separator") || {}).value || null,
    style_decimal_precision:     (() => {
      const v = ($("style_decimal_precision") || {}).value;
      return v === "" || v == null ? null : parseInt(v, 10);
    })(),
    style_laterality:            ($("style_laterality")            || {}).value || null,
    style_impression_style:      ($("style_impression_style")      || {}).value || null,
    style_negation_phrasing:     ($("style_negation_phrasing")     || {}).value || null,
    style_date_format:           ($("style_date_format")           || {}).value || null,
    style_paste_format:          ($("style_paste_format")          || {}).value || null,
  };

  const btn = $("btn-save");
  btn.disabled = true;
  const msg = $("save-msg");
  msg.textContent = "Saving…";
  msg.style.color = "var(--muted)";

  try {
    const resp = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    msg.textContent = "✓ Saved";
    msg.style.color = "var(--green)";
    setTimeout(() => { msg.textContent = ""; }, 3000);
  } catch (err) {
    msg.textContent = `Error: ${err.message}`;
    msg.style.color = "var(--red)";
  } finally {
    btn.disabled = false;
  }
}

function showStatus(text, type) {
  const el = $("settings-status");
  if (!el) return;
  el.textContent = text;
  el.className = type || "";
  el.style.cssText = `padding:8px 12px;border-radius:6px;font-size:13px;margin-bottom:10px;
    border:1px solid ${type === "error" ? "var(--red)" : "var(--border)"};
    color:${type === "error" ? "var(--red)" : "var(--text)"};`;
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function setupMicrophoneSettings() {
  const filter = $("microphone-filter");
  const sensitivity = $("microphone-sensitivity");
  const button = $("microphone-test");
  const panel = $("microphone-test-panel");
  const status = $("microphone-test-status");
  const prefs = RadSpeedMicrophone.preferences();
  filter.checked = prefs.enabled;
  sensitivity.value = prefs.sensitivity;
  sensitivity.disabled = !filter.checked;
  let stream = null;
  let timer = null;
  let generation = 0;
  let controller = null;

  function stopTest() {
    generation++;
    controller?.abort();
    controller = null;
    clearTimeout(timer);
    RadSpeedMicrophone.stop(stream);
    stream = null;
    button.disabled = false;
    button.textContent = "Test microphone";
    $("microphone-level").value = 0;
    status.textContent = "Test stopped. No audio was saved or sent.";
  }

  function saveMicrophone() {
    stopTest();
    panel.hidden = true;
    sensitivity.disabled = !filter.checked;
    try {
      RadSpeedMicrophone.save({ enabled: filter.checked, sensitivity: sensitivity.value });
      $("microphone-save-status").textContent = "Saved for this device and browser. Changes apply to the next recording.";
    } catch (_) {
      $("microphone-save-status").textContent = "Could not save microphone settings. Allow browser storage, then try again.";
    }
  }
  filter.addEventListener("change", saveMicrophone);
  sensitivity.addEventListener("change", saveMicrophone);
  button.addEventListener("click", async () => {
    if (stream) { stopTest(); status.textContent = "Test stopped. No audio was saved or sent."; return; }
    const attempt = ++generation;
    controller = new AbortController();
    button.disabled = true;
    panel.hidden = false;
    status.textContent = "Opening microphone…";
    try {
      const opened = await RadSpeedMicrophone.open((level) => {
        if (attempt !== generation) return;
        $("microphone-level").value = level.rms;
        status.textContent = level.passing
          ? "Passing audio. Check that your voice passes and background speech does not."
          : "Filtering quieter audio. Speak normally to test your voice.";
      }, { signal: controller.signal });
      if (attempt !== generation) { RadSpeedMicrophone.stop(opened); return; }
      stream = opened;
      button.disabled = false;
      button.textContent = "Stop test";
      timer = setTimeout(() => { stopTest(); status.textContent = "Test finished. No audio was saved or sent."; }, 30000);
    } catch (error) {
      if (attempt !== generation) return;
      stopTest();
      status.textContent = error.message;
    }
  });
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopTest(); });
  window.addEventListener("pagehide", stopTest);
}

document.addEventListener("DOMContentLoaded", () => {
  setupMicrophoneSettings();
  loadSettings();
  $("btn-save").addEventListener("click", saveSettings);
});
