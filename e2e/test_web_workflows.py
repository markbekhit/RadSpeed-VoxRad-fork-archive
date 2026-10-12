"""Browser-level coverage for RadSpeed's highest-value web workflows."""
from __future__ import annotations

import io
import re
import time

import pytest
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, Page, expect


def _console_errors(page: Page) -> list[str]:
    errors: list[str] = []
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    return errors


def test_public_impressions_generation_and_validation(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/impressions")
    expect(
        page.get_by_role("heading", name="Radiology impression generator", exact=True)
    ).to_be_visible()
    expect(page.locator("#with-guidelines")).to_be_checked()

    page.locator("#btn-generate").click()
    expect(page.locator("#status")).to_have_text("Paste some findings first.")

    page.get_by_role("button", name="CT chest", exact=True).click()
    expect(page.locator("#findings")).to_have_value(re.compile("14 mm spiculated nodule"))
    expect(page.locator("#modality")).to_have_value("CT chest with contrast")
    expect(page.locator("#status")).to_contain_text("Synthetic example loaded")

    findings = (
        "CT chest with contrast. There is a 14 mm spiculated right upper lobe "
        "pulmonary nodule. No mediastinal lymphadenopathy or pleural effusion."
    )
    page.locator("#findings").fill(findings)
    page.locator("#modality").fill("CT chest with contrast")
    expect(page.locator("#findings-count")).to_have_text(f"{len(findings)} chars")

    page.locator("#btn-generate").click()
    expect(page.locator("#impression-output")).to_contain_text(
        "No acute cardiopulmonary abnormality", timeout=10_000
    )
    expect(page.locator("#btn-copy")).to_be_enabled()
    expect(page.locator("#status")).to_contain_text("Done")
    assert errors == []


def test_public_landing_page_keeps_impressions_and_sign_in_visible(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(base_url)
    expect(
        page.get_by_role("heading", name="Radiology reporting built around the way you dictate.")
    ).to_be_visible()
    expect(page.get_by_role("link", name="Try Impressions", exact=True).first).to_have_attribute(
        "href", "/impressions"
    )
    expect(page.get_by_role("link", name="Sign in", exact=True).first).to_have_attribute(
        "href", "/app"
    )
    expect(
        page.get_by_role("heading", name="Use the narrow tool that matches the task.")
    ).to_be_visible()
    expect(page.get_by_role("link", name=re.compile("Fleischner calculator"))).to_have_attribute(
        "href", "/fleischner-calculator"
    )
    assert errors == []


def test_public_powerscribe_companion_page_is_useful_and_responsive(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/powerscribe-companion")
    expect(
        page.get_by_role("heading", name="Keep PowerScribe open. Add a faster impression step.")
    ).to_be_visible()
    expect(page.get_by_role("link", name="Try the workflow", exact=True)).to_have_attribute(
        "href", "/impressions"
    )
    expect(page.get_by_text("Windows may show an “Unknown publisher” warning")).to_be_visible()

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    expect(page.get_by_role("heading", name="Three steps. No new reporting screen.")).to_be_visible()
    assert errors == []


def test_reporting_preferences_save_numerals_and_quality_opt_in(page: Page, base_url: str):
    saved: list[dict] = []

    def handle_settings(route):
        if route.request.method == "GET":
            route.fulfill(json={
                "keys": {
                    "transcription": False,
                    "text": False,
                    "deepgram": False,
                    "assemblyai": False,
                },
                "streaming_stt_provider": "auto",
                "can_manage_global_settings": False,
                "fhir_export_enabled": False,
                "retain_deidentified_samples": False,
                "quality_retention_available": True,
                "style": {"numerals": "roman"},
            })
        else:
            saved.append(route.request.post_data_json)
            route.fulfill(json={"ok": True})

    page.route("**/api/settings", handle_settings)
    page.goto(f"{base_url}/settings")
    quality = page.locator("#retain_deidentified_samples")
    expect(quality).to_be_enabled()
    expect(quality).not_to_be_checked()
    expect(page.get_by_text("Automated redaction can miss identifiers")).to_be_visible()
    quality.check()
    page.locator("#style_numerals").select_option("arabic")
    page.locator("#btn-save").click()
    page.wait_for_function("() => document.querySelector('#save-msg').textContent.includes('Saved')")
    assert saved[-1]["retain_deidentified_samples"] is True
    assert saved[-1]["style_numerals"] == "arabic"

    quality.uncheck()
    with page.expect_request("**/api/settings") as request_info:
        page.locator("#btn-save").click()
    assert request_info.value.post_data_json["retain_deidentified_samples"] is False


def test_authenticated_transcribe_to_streamed_report(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    expect(page.get_by_role("heading", name="RadSpeed")).to_be_visible()
    expect(page.locator("#template-select option")).not_to_have_count(1)

    page.locator("#template-select").select_option("CT_Chest.txt")
    # Exercise the same browser function used when MediaRecorder finishes.
    # The synthetic blob clears the minimum-size gate; mock mode then returns
    # canned transcription and automatically starts streamed formatting.
    page.evaluate("submitAudioSegment([new Uint8Array(13000)], true)")

    expect(page.locator("#transcription")).to_have_value(
        re.compile("CT chest with contrast"), timeout=10_000
    )
    expect(page.locator("#report-rendered")).to_contain_text(
        "No acute cardiopulmonary abnormality", timeout=15_000
    )
    expect(page.locator("#status")).to_contain_text("Report ready")
    expect(page.locator("#report-status-badge")).to_have_text("Preliminary")
    assert errors == []


def test_streaming_completion_uses_server_recovered_final_interim(page: Page, base_url: str):
    """A provider interim recovered during stop must not disappear in the UI."""
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    page.locator("#template-select").select_option("MRI_Knee.txt")
    page.locator("#transcription").fill("MRI knee.")

    page.evaluate(
        """() => {
          state.confirmedText = "There is a popliteus tendon tear";
          state.interimText = "complete arcuate ligament rupture";
          state.streamingBefore = "MRI knee.";
          state.streamingAfter = "";
          handleStreamingMessage({
            type: "session_complete",
            transcription: "There is a popliteus tendon tear and complete arcuate ligament rupture",
            session_id: "synthetic-stream-session"
          });
        }"""
    )

    expect(page.locator("#transcription")).to_have_value(
        re.compile("popliteus tendon tear and complete arcuate ligament rupture"),
        timeout=10_000,
    )
    expect(page.locator("#report-rendered")).to_contain_text(
        "No acute cardiopulmonary abnormality", timeout=15_000
    )
    assert errors == []


def test_streaming_completion_does_not_duplicate_finals_after_cursor_move(page: Page, base_url: str):
    page.goto(f"{base_url}/app")
    page.locator("#template-select").select_option("MRI_Knee.txt")
    page.evaluate(
        """() => {
          state.streamingBefore = "alpha";
          state.streamingAfter = "beta";
          state.streamingServerConfirmedText = "beta";
          state.streamingBakedServerText = "beta";
          state.streamingCursorRepositioned = true;
          state.confirmedText = "";
          handleStreamingMessage({
            type: "session_complete",
            transcription: "beta gamma",
            raw_transcription: "beta gamma",
            session_id: "synthetic-cursor-session"
          });
        }"""
    )

    expect(page.locator("#transcription")).to_have_value("alpha gamma beta")


def test_streaming_completion_without_server_transcript_uses_confirmed_text(page: Page, base_url: str):
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          state.streamingBefore = "";
          state.streamingAfter = "";
          state.confirmedText = "fallback confirmed phrase";
          handleStreamingMessage({type: "session_complete", session_id: "legacy-session"});
        }"""
    )

    expect(page.locator("#transcription")).to_have_value("fallback confirmed phrase")


def test_streaming_cursor_fallback_uses_only_unbaked_confirmed_text(page: Page, base_url: str):
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          state.streamingBefore = "inserted earlier alpha";
          state.streamingAfter = "tail";
          state.streamingServerConfirmedText = "alpha beta";
          state.streamingBakedServerText = "alpha";
          state.streamingCursorRepositioned = true;
          state.confirmedText = "beta";
          handleStreamingMessage({
            type: "session_complete",
            transcription: "corrected text that no longer matches raw",
            recovered_text: "gamma",
            session_id: "fallback-cursor-session"
          });
        }"""
    )
    expect(page.locator("#transcription")).to_have_value(
        "inserted earlier alpha beta gamma tail"
    )


def test_ct_cap_compare_mode_generates_both_layouts_then_uses_one(page: Page, base_url: str):
    errors = _console_errors(page)
    payloads = []
    page.on(
        "request",
        lambda request: payloads.append(request.post_data_json)
        if request.url.endswith("/format/stream") and request.method == "POST"
        else None,
    )
    page.goto(f"{base_url}/app")

    expect(page.locator('#template-select option[value="_CT_CAP_Staging_Regions.txt"]')).to_have_count(0)
    page.locator("#template-select").select_option("CT_CAP_Staging.txt")
    expect(page.locator("#ct-cap-layout-panel")).to_be_visible()
    page.locator("#template-select").select_option("")
    expect(page.locator("#ct-cap-layout-panel")).to_be_hidden()
    page.locator("#transcription").fill(
        "Staging CT CAP. No pulmonary or hepatic metastasis. No lymphadenopathy."
    )
    page.locator("#btn-format").click()

    expect(page.locator("#ct-cap-comparison")).to_be_visible()
    expect(page.locator("#ct-cap-components-report")).to_contain_text(
        "Lungs and Airways", timeout=15_000
    )
    expect(page.locator("#ct-cap-regions-report")).to_contain_text(
        "Abdomen and Pelvis", timeout=15_000
    )
    expect(page.locator("#btn-use-cap-components")).to_be_enabled()
    expect(page.locator("#btn-use-cap-regions")).to_be_enabled()

    comparison_payloads = [p for p in payloads if p.get("comparison_draft")]
    assert {p["cap_layout"] for p in comparison_payloads} == {"components", "regions"}
    assert all(p["template_name"] == "CT_CAP_Staging.txt" for p in comparison_payloads)

    page.locator("#btn-use-cap-regions").click()
    expect(page.locator("#ct-cap-comparison")).to_be_hidden()
    expect(page.locator("#report-rendered")).to_contain_text("Chest")
    expect(page.locator("#report-rendered")).to_contain_text("Abdomen and Pelvis")
    expect(page.locator("#status")).to_contain_text("selected")
    expect(page.locator("#report-status-badge")).to_have_text("Preliminary")
    assert errors == []


def test_desktop_overlay_keeps_reporting_controls_compact(page: Page, base_url: str):
    errors = _console_errors(page)
    page.set_viewport_size({"width": 520, "height": 300})
    page.goto(f"{base_url}/app?desktop=overlay")

    expect(page.locator("body")).to_have_class(re.compile("desktop-overlay"))
    expect(page.locator("#desktop-overlay-head")).to_be_visible()
    expect(page.locator("#btn-record")).to_be_visible()
    expect(page.locator("#btn-stop")).to_be_visible()
    expect(page.locator("#btn-overlay-copy")).to_be_visible()
    expect(page.locator("#btn-overlay-next")).to_be_visible()
    expect(page.locator("body > header")).to_be_hidden()
    assert page.evaluate("document.documentElement.scrollWidth") <= 520
    assert page.evaluate("document.documentElement.scrollHeight") <= 300

    page.evaluate('setUI("recording")')
    expect(page.locator("#btn-record")).to_contain_text("Pause")
    expect(page.locator("#btn-stop")).to_be_enabled()

    page.evaluate('setUI("done")')
    expect(page.locator("#btn-overlay-copy")).to_be_enabled()

    page.locator("#btn-desktop-expand").click()
    expect(page.locator("body")).not_to_have_class(re.compile("desktop-overlay"))
    expect(page.locator("body > header")).to_be_visible()
    expect(page.locator("#btn-desktop-collapse")).to_be_visible()
    assert errors == []


@pytest.mark.parametrize("desktop_signal", ["marker", "tauri-window"])
def test_desktop_overlay_survives_query_free_sign_in(page: Page, base_url: str, desktop_signal: str):
    errors = _console_errors(page)
    # Model WebView2's document-start scripts and the OAuth callback to /app.
    signal_script = (
        "window.__RADSPEED_DESKTOP_OVERLAY__ = true;" if desktop_signal == "marker"
        else "window.__TAURI__.window = {getCurrentWindow: () => ({label: 'app'})};"
    )
    page.add_init_script("""
        window.desktopCalls = [];
        window.__TAURI__ = {
            core: {invoke: async (command, args) => window.desktopCalls.push({command, args})}
        };
    """ + signal_script)
    page.set_viewport_size({"width": 420, "height": 280})
    page.goto(f"{base_url}/app")

    expect(page.locator("body")).to_have_class(re.compile("desktop-overlay"))
    expect(page.locator("body > header")).to_be_hidden()
    for width, height in ((420, 280), (520, 300), (700, 400)):
        page.set_viewport_size({"width": width, "height": height})
        for selector in ("#btn-record", "#btn-stop", "#btn-overlay-copy", "#btn-overlay-next", "#btn-overlay-refine", "#transcription"):
            expect(page.locator(selector)).to_be_in_viewport()
        assert page.evaluate("document.documentElement.scrollWidth") <= width
        assert page.evaluate("document.documentElement.scrollHeight") <= height
        # Two lines at the default size, then grow with the window height.
        assert page.locator("#transcription").bounding_box()["height"] == max(48, height - 252)

    page.locator("#btn-desktop-settings").click()
    page.locator("#btn-desktop-expand").click()
    expect(page.locator("body > header")).to_be_visible()
    page.locator("#btn-desktop-collapse").click()
    expect(page.locator("body > header")).to_be_hidden()
    assert page.evaluate("window.desktopCalls") == [
        {"command": "cmd_show_settings", "args": {}},
        {"command": "cmd_set_compact_mode", "args": {"compact": False}},
        {"command": "cmd_set_compact_mode", "args": {"compact": True}},
    ]
    page.reload()
    expect(page.locator("body")).to_have_class(re.compile("desktop-overlay"))
    assert errors == []


def test_browser_workstation_without_desktop_signal_stays_full_size(page: Page, base_url: str):
    page.goto(f"{base_url}/app")
    expect(page.locator("body")).not_to_have_class(re.compile("desktop-host"))
    expect(page.locator("body > header")).to_be_visible()
    expect(page.locator("#desktop-overlay-head")).to_be_hidden()


def test_compact_transcript_tail_and_voice_refinement(page: Page, base_url: str):
    errors = _console_errors(page)
    page.set_viewport_size({"width": 420, "height": 280})
    page.goto(f"{base_url}/app?desktop=overlay")
    page.route("**/transcribe", lambda route: route.fulfill(
        json={"transcription": "Use a shorter impression."}
    ))
    page.route("**/format/feedback", lambda route: route.fulfill(
        json={"report": "**IMPRESSION:**\nNo acute abnormality."}
    ))
    page.evaluate(r"""() => {
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: {getUserMedia: async () => ({getTracks: () => [{stop() {}}]})},
      });
      // This fixture tests refinement UI with a non-audio stream. Real audio
      // filtering and cleanup have separate synthetic microphone coverage.
      RadSpeedMicrophone.save({enabled: false, sensitivity: 'balanced'});
      window.MediaRecorder = class {
        start() {}
        stop() {
          this.ondataavailable({data: new Blob([new Uint8Array(13000)])});
          this.onstop();
        }
      };
      state.streamingBefore = '';
      state.streamingAfter = '';
      state.confirmedText = Array.from({length: 30}, (_, i) => `Synthetic line ${i}`).join('\n');
      _updateStreamingDisplay();
      state.isRecording = true;
      state.isPaused = true;
      setUI('paused');
    }""")
    expect(page.locator("#transcription")).to_have_value(re.compile("Synthetic line 29$"))
    assert page.locator("#transcription").evaluate(
        "el => el.scrollTop + el.clientHeight >= el.scrollHeight - 2"
    )
    expect(page.locator("#btn-overlay-refine")).to_be_disabled()
    page.evaluate(r"""() => {
      window.waveformReads = 0;
      window.AudioContext = class {
        createMediaStreamSource() { return {connect() {}}; }
        createAnalyser() {
          return {
            frequencyBinCount: 128,
            getByteTimeDomainData(data) {
              window.waveformReads++;
              data.forEach((_, i) => { data[i] = i % 2 ? 180 : 76; });
            },
          };
        }
        close() { return Promise.resolve(); }
      };
      window.waveIsFlat = () => {
        const canvas = document.getElementById('waveform');
        const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
        for (let y = 0; y < canvas.height; y++) {
          if (Math.abs(y - canvas.height / 2) <= 2) continue;
          for (let x = 0; x < canvas.width; x++) {
            if (pixels[(y * canvas.width + x) * 4] > 100) return false;
          }
        }
        return true;
      };
      startWaveform({});
    }""")
    assert page.evaluate("waveIsFlat()")
    assert page.evaluate("window.waveformReads") == 0
    page.locator("#btn-record").click()
    page.wait_for_function("window.waveformReads > 0 && !waveIsFlat()")
    page.locator("#btn-record").click()
    page.wait_for_function("waveIsFlat()")
    assert page.locator("#transcription").evaluate(
        "el => el.scrollTop + el.clientHeight >= el.scrollHeight - 2"
    )
    page.evaluate("stopWaveform()")
    page.evaluate(r"""() => {
      state.isRecording = false;
      state.isPaused = false;
      setReport('**IMPRESSION:**\n- No acute abnormality.');
      setUI('done');
    }""")
    page.locator("#btn-overlay-refine").click()
    expect(page.locator("#btn-overlay-refine")).to_have_text("■ Stop refine")
    expect(page.locator("#btn-record")).to_be_disabled()
    expect(page.locator("#btn-overlay-next")).to_be_disabled()
    with page.expect_request("**/format/feedback") as feedback_request:
        page.locator("#btn-overlay-refine").click()
    assert feedback_request.value.post_data_json["feedback"] == "Use a shorter impression."
    expect(page.locator("#status")).to_contain_text("Report updated")
    expect(page.locator("#btn-overlay-refine")).to_have_text("🎤 Refine")
    expect(page.locator("#btn-overlay-refine")).to_be_enabled()
    expect(page.locator("#btn-overlay-copy")).to_be_enabled()
    expect(page.locator("#btn-record")).to_be_enabled()
    expect(page.locator("#btn-overlay-next")).to_be_enabled()
    assert errors == []


def test_authenticated_impression_action_preserves_the_report(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    original = (
        "**EXAM:**\nSynthetic thyroid ultrasound.\n\n"
        "**FINDINGS:**\nSynthetic right thyroid nodule, 16 mm, ACR TI-RADS TR4.\n\n"
        "**IMPRESSION:**\n- Original impression."
    )
    page.evaluate(
        """(report) => {
          setReport(report);
          setUI("done");
        }""",
        original,
    )

    page.locator("#btn-impression").click()
    expect(page.locator("#status")).to_contain_text(
        "Findings and other sections were unchanged", timeout=10_000
    )
    expect(page.locator("#report-raw")).to_have_value(
        re.compile(r"Synthetic thyroid ultrasound[\s\S]*Synthetic right thyroid nodule[\s\S]*Mock guideline-aware impression")
    )
    expect(page.locator("#report-raw")).not_to_have_value(re.compile("Original impression"))
    assert errors == []


def test_shared_screenshot_area_transcribes_indication_locally_and_hides_patient_details(
    page: Page, base_url: str
):
    errors = _console_errors(page)
    external_requests: list[str] = []
    page.on(
        "request",
        lambda request: external_requests.append(request.url)
        if not request.url.startswith(base_url)
        else None,
    )
    page.route(
        "**/static/vendor/tesseract/tesseract.min.js*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/javascript",
            body="""
              window.Tesseract = {
                createWorker: async () => ({
                  recognize: async () => ({ data: { text:
                    "CLINICAL INDICATION:\\nFall onto outstretched hand. Radial-sided wrist pain."
                  }}),
                  terminate: async () => {},
                }),
              };
            """,
        ),
    )

    page.goto(f"{base_url}/app")
    expect(page.locator("#patient-context-details")).not_to_have_attribute("open", "")
    expect(page.locator("#indication-paste-zone")).to_have_count(0)
    expect(page.locator("#worksheet-drop-zone")).to_contain_text(
        "Paste a worksheet or indication screenshot here"
    )
    expect(
        page.locator("#btn-indication-transcribe + #btn-worksheet-generate")
    ).to_have_count(1)
    page.evaluate(
        """() => {
          window.__indicationClipboard = "";
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: { writeText: async (value) => { window.__indicationClipboard = value; } },
          });
          const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZJfQAAAAASUVORK5CYII=";
          const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
          const file = new File([bytes], "indication.png", { type: "image/png" });
          const data = new DataTransfer();
          data.items.add(file);
          document.dispatchEvent(new ClipboardEvent("paste", {
            clipboardData: data,
            bubbles: true,
            cancelable: true,
          }));
        }"""
    )

    expect(page.locator(".worksheet-preview")).to_have_count(1)
    expect(page.locator("#btn-indication-transcribe")).to_be_enabled()
    page.locator("#btn-indication-transcribe").click()
    expect(page.locator("#indication-text")).to_have_value(
        "Fall onto outstretched hand. Radial-sided wrist pain."
    )
    page.locator("#indication-text").fill(
        "Fall onto outstretched hand. Tender anatomical snuffbox."
    )
    page.locator("#btn-indication-copy").click()
    expect(page.locator("#indication-status")).to_contain_text("paste into PowerScribe")
    assert page.evaluate("window.__indicationClipboard") == (
        "Fall onto outstretched hand. Tender anatomical snuffbox."
    )
    assert external_requests == []
    assert errors == []


def test_authenticated_fracture_lab_is_reachable_from_radspeed(
    page: Page, base_url: str
):
    errors = _console_errors(page)
    # Production images live on the Fly volume rather than in the repository.
    # Stub image responses here so isolated browser QA can verify the viewer.
    page.route(
        "**/fracture-workbench/images/**",
        lambda route: route.fulfill(
            status=200,
            content_type="image/gif",
            body=(
                b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00"
                b"\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00"
                b"\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
            ),
        ),
    )

    page.goto(f"{base_url}/app")
    expect(page.get_by_role("link", name="Fracture Lab")).to_have_attribute(
        "href", "/fracture-workbench"
    )
    page.get_by_role("link", name="Fracture Lab").click()

    expect(page.get_by_role("heading", name="RadSpeed Fracture Lab")).to_be_visible()
    expect(page.locator(".notice")).to_contain_text("Experimental research interface")
    expect(page.get_by_role("button", name="All 1132")).to_be_visible()
    expect(page.locator("article.card")).to_have_count(1132)
    assert errors == []


def test_fracture_lab_runs_deidentification_engine_entirely_on_site(
    page: Page, base_url: str
):
    errors = _console_errors(page)
    external_requests: list[str] = []
    page.on(
        "request",
        lambda request: external_requests.append(request.url)
        if not request.url.startswith(base_url) and not request.url.startswith("blob:")
        else None,
    )
    page.route(
        "**/fracture-workbench/images/**",
        lambda route: route.fulfill(status=200, content_type="image/gif", body=b"GIF89a"),
    )

    screenshot = Image.new("RGB", (1200, 420), color="black")
    draw = ImageDraw.Draw(screenshot)
    font = ImageFont.load_default(size=58)
    draw.text((35, 35), "PATIENT NAME: EXAMPLE", fill="white", font=font)
    draw.text((35, 125), "MRN: 12345678", fill="white", font=font)
    screenshot_buffer = io.BytesIO()
    screenshot.save(screenshot_buffer, format="PNG")

    page.goto(f"{base_url}/fracture-workbench")
    page.locator("#fracture-file-input").set_input_files(
        {
            "name": "identifiable-synthetic.png",
            "mimeType": "image/png",
            "buffer": screenshot_buffer.getvalue(),
        }
    )

    expect(page.locator(".fracture-preview.privacy-ready")).to_have_count(1, timeout=30_000)
    expect(page.locator("#fracture-privacy-summary")).to_contain_text(
        re.compile(r"[1-9]\d* text areas? covered"), timeout=30_000
    )
    expect(page.locator("#fracture-analyse")).to_be_disabled()
    assert external_requests == []
    assert errors == []


def test_fracture_lab_analyses_uploaded_multiview_study(page: Page, base_url: str):
    errors = _console_errors(page)
    uploaded_payloads: list[bytes] = []

    page.route(
        "**/static/vendor/tesseract/tesseract.min.js*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/javascript",
            body="""
              window.Tesseract = {
                createWorker: async () => ({
                  recognize: async () => ({ data: { tsv:
                    "level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n" +
                    "5\\t1\\t1\\t1\\t1\\t1\\t4\\t8\\t24\\t10\\t96\\tPATIENT\\n" +
                    "5\\t1\\t1\\t1\\t1\\t2\\t30\\t8\\t16\\t10\\t96\\tNAME\\n" +
                    "5\\t1\\t1\\t1\\t1\\t3\\t48\\t8\\t40\\t10\\t96\\tSYNTHETIC\\n" +
                    "5\\t1\\t1\\t1\\t2\\t1\\t4\\t30\\t18\\t10\\t96\\tMRN\\n" +
                    "5\\t1\\t1\\t1\\t2\\t2\\t24\\t30\\t50\\t10\\t96\\tTEST12345\\n" +
                    "5\\t1\\t1\\t1\\t3\\t1\\t8\\t68\\t32\\t10\\t96\\tWRIST\\n" +
                    "5\\t1\\t1\\t1\\t4\\t1\\t80\\t80\\t8\\t8\\t96\\tR"
                  }}),
                  terminate: async () => {},
                }),
              };
            """,
        ),
    )
    page.route(
        "**/fracture-workbench/images/**",
        lambda route: route.fulfill(
            status=200,
            content_type="image/gif",
            body=(
                b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00"
                b"\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00"
                b"\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
            ),
        ),
    )
    synthetic_buffer = io.BytesIO()
    Image.new("L", (96, 96), color=110).save(synthetic_buffer, format="PNG")
    synthetic_xray = synthetic_buffer.getvalue()

    unconfirmed = page.request.post(
        f"{base_url}/api/fracture-analysis",
        multipart={
            "images": {
                "name": "synthetic-view.png",
                "mimeType": "image/png",
                "buffer": synthetic_xray,
            }
        },
    )
    assert unconfirmed.status == 400
    assert "confirm privacy" in unconfirmed.json()["detail"]

    def capture_analysis_request(request):
        if request.url.endswith("/api/fracture-analysis"):
            uploaded_payloads.append(request.post_data_buffer)

    page.on("request", capture_analysis_request)

    page.goto(f"{base_url}/fracture-workbench")
    expect(page.locator("#fracture-study-type")).to_have_count(0)
    page.locator("#fracture-file-input").set_input_files(
        [
            {
                "name": "synthetic-view.png",
                "mimeType": "image/png",
                "buffer": synthetic_xray,
            }
        ]
    )
    expect(page.locator(".fracture-preview")).to_have_count(1)
    expect(page.locator("#fracture-privacy-summary")).to_contain_text(
        "2 text areas covered", timeout=10_000
    )
    expect(page.locator("#fracture-analyse")).to_be_disabled()

    # Identifier-labelled lines are covered. Ordinary anatomical text and the
    # standard right-side marker are intentionally retained.
    pixels = page.locator(".fracture-preview-canvas").evaluate(
        """canvas => ({
          redacted: Array.from(canvas.getContext('2d').getImageData(10, 10, 1, 1).data),
          identifier: Array.from(canvas.getContext('2d').getImageData(10, 34, 1, 1).data),
          anatomy: Array.from(canvas.getContext('2d').getImageData(12, 72, 1, 1).data),
          marker: Array.from(canvas.getContext('2d').getImageData(82, 82, 1, 1).data),
        })"""
    )
    assert pixels["redacted"][:3] == [0, 0, 0]
    assert pixels["identifier"][:3] == [0, 0, 0]
    assert pixels["anatomy"][:3] == [110, 110, 110]
    assert pixels["marker"][:3] == [110, 110, 110]

    page.locator(".fracture-preview-canvas").scroll_into_view_if_needed()
    canvas_box = page.locator(".fracture-preview-canvas").bounding_box()
    assert canvas_box
    page.mouse.click(canvas_box["x"] + canvas_box["width"] * 0.10, canvas_box["y"] + canvas_box["height"] * 0.10)
    expect(page.locator("#fracture-privacy-summary")).to_contain_text("1 text area covered")
    assert page.locator(".fracture-preview-canvas").evaluate(
        "canvas => Array.from(canvas.getContext('2d').getImageData(10, 10, 1, 1).data).slice(0, 3)"
    ) == [110, 110, 110]

    page.mouse.move(canvas_box["x"] + canvas_box["width"] * 0.52, canvas_box["y"] + canvas_box["height"] * 0.52)
    page.mouse.down()
    page.mouse.move(canvas_box["x"] + canvas_box["width"] * 0.70, canvas_box["y"] + canvas_box["height"] * 0.70)
    page.mouse.up()
    expect(page.locator("#fracture-privacy-summary")).to_contain_text("2 text areas covered")
    assert page.locator(".fracture-preview-canvas").evaluate(
        "canvas => Array.from(canvas.getContext('2d').getImageData(56, 56, 1, 1).data).slice(0, 3)"
    ) == [0, 0, 0]
    page.get_by_role("button", name="Remove last blackout").click()
    expect(page.locator("#fracture-privacy-summary")).to_contain_text("1 text area covered")

    page.locator("#fracture-privacy-confirm").check()
    expect(page.locator("#fracture-analyse")).to_be_enabled()
    page.locator("#fracture-context").fill("Synthetic test context")
    page.locator("#fracture-analyse").click()

    expect(page.locator("#fracture-result")).to_be_visible(timeout=10_000)
    expect(page.locator("#fracture-result")).to_contain_text("Possible fracture")
    expect(page.locator("#fracture-result")).to_contain_text(
        "Independent open models"
    )
    expect(page.locator("#fracture-result")).to_contain_text(
        "Public-dataset fracture estimate: 42%"
    )
    expect(page.locator("#fracture-result")).to_contain_text("62% model confidence")
    expect(page.locator("#fracture-result svg rect")).to_have_count(1)
    expect(page.locator("#fracture-status")).to_contain_text("Review complete")
    assert uploaded_payloads
    assert b'deidentified-view-' in uploaded_payloads[-1]
    assert b'synthetic-view.png' not in uploaded_payloads[-1]
    assert b'privacy_confirmed' in uploaded_payloads[-1]
    assert b'study_type' not in uploaded_payloads[-1]
    assert errors == []


def test_pasted_worksheet_screenshot_generates_report_in_safe_source_mode(
    page: Page, base_url: str
):
    errors = _console_errors(page)
    format_payloads: list[dict] = []

    def capture_format_request(request):
        if request.url.endswith("/format/stream") and request.post_data_json:
            format_payloads.append(request.post_data_json)

    page.on("request", capture_format_request)
    page.goto(f"{base_url}/app")
    expect(page.locator("#worksheet-drop-zone")).to_contain_text(
        "Paste a worksheet or indication screenshot here"
    )

    page.evaluate(
        """() => {
          const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZJfQAAAAASUVORK5CYII=";
          const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
          const file = new File([bytes], "renal-worksheet.png", { type: "image/png" });
          const data = new DataTransfer();
          data.items.add(file);
          const zone = document.getElementById("worksheet-drop-zone");
          zone.focus();
          zone.dispatchEvent(new ClipboardEvent("paste", {
            clipboardData: data,
            bubbles: true,
            cancelable: true,
          }));
        }"""
    )

    expect(page.locator(".worksheet-preview")).to_have_count(1)
    expect(page.locator("#btn-worksheet-generate")).to_be_enabled()
    page.locator("#btn-worksheet-generate").click()

    expect(page.locator("#transcription")).to_have_value(
        re.compile("WORKSHEET SOURCE NOTES.*Left kidney", re.DOTALL),
        timeout=10_000,
    )
    expect(page.locator("#report-rendered")).to_contain_text(
        "No acute cardiopulmonary abnormality", timeout=15_000
    )
    expect(page.locator("#status")).to_contain_text("Report ready")
    expect(page.locator("#template-select")).to_have_value("Ultrasound_Worksheet.txt")
    assert format_payloads and format_payloads[-1]["source_kind"] == "worksheet"
    assert format_payloads[-1]["template_name"] == "Ultrasound_Worksheet.txt"

    first_source = page.locator("#transcription").input_value()
    page.evaluate("state.reportCopied = true")
    page.evaluate(
        """() => {
          const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZJfQAAAAASUVORK5CYII=";
          const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
          const file = new File([bytes], "second-worksheet.png", { type: "image/png" });
          const data = new DataTransfer();
          data.items.add(file);
          document.getElementById("worksheet-drop-zone").dispatchEvent(
            new ClipboardEvent("paste", {
              clipboardData: data,
              bubbles: true,
              cancelable: true,
            })
          );
        }"""
    )
    expect(page.locator("#btn-worksheet-generate")).to_be_enabled()
    page.locator("#btn-worksheet-generate").click()
    expect(page.locator("#status")).to_contain_text("Report ready", timeout=15_000)
    page.wait_for_function("() => document.querySelector('#transcription').value.split('WORKSHEET SOURCE NOTES').length >= 3")
    assert first_source in page.locator("#transcription").input_value()
    assert len(format_payloads) >= 2
    assert format_payloads[-1]["source_kind"] == "dictation"


    # Copy must preserve clinical line structure without exporting the browser's
    # paragraph/list spacing. PowerScribe prefers text/html when both clipboard
    # flavours are present, so assert both representations are compact.
    page.evaluate(
        """() => {
          setReport(
            "ULTRASOUND KIDNEYS\\n\\n" +
            "FINDINGS:\\nLeft kidney measures 10.2 cm.\\n\\n" +
            "CONCLUSION:\\n1. No hydronephrosis.\\n2. Simple renal cyst."
          );
          setUI("done");
          document.body.dataset.pasteFormat = "rich";
          window.__worksheetClipboard = {};
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: {
              write: async (items) => {
                for (const type of items[0].types) {
                  window.__worksheetClipboard[type] =
                    await (await items[0].getType(type)).text();
                }
              },
            },
          });
        }"""
    )
    page.locator("#btn-copy").click()
    page.wait_for_function(
        "() => Boolean(window.__worksheetClipboard['text/plain'])"
    )
    clipboard = page.evaluate("window.__worksheetClipboard")
    assert clipboard["text/plain"] == (
        "ULTRASOUND KIDNEYS\n\n"
        "FINDINGS:\nLeft kidney measures 10.2 cm.\n\n"
        "CONCLUSION:\n"
        "1. No hydronephrosis.\n2. Simple renal cyst."
    )
    assert "<p" not in clipboard["text/html"].lower()
    assert "<li" not in clipboard["text/html"].lower()
    assert "<br>" in clipboard["text/html"].lower()
    assert errors == []


def test_one_step_worksheet_draft_shows_notes_and_report(page, base_url):
    draft_requests = []
    page.on("request", lambda request: draft_requests.append(request.post_data_buffer or b"") if request.url.endswith("/api/worksheet/draft") else None)
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZJfQAAAAASUVORK5CYII=";
          const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
          const file = new File([bytes], "renal-worksheet.png", { type: "image/png" });
          const data = new DataTransfer();
          data.items.add(file);
          document.getElementById("worksheet-drop-zone").dispatchEvent(
            new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true })
          );
        }"""
    )
    page.evaluate("state.reportCopied = true")
    page.locator("#btn-worksheet-one-pass").click()
    expect(page.locator("#transcription")).to_have_value(re.compile("WORKSHEET SOURCE NOTES.*Left kidney", re.DOTALL))
    expect(page.locator("#report-rendered")).to_contain_text("Mild left pelvicaliectasis")
    expect(page.locator("#status")).to_contain_text("One-step draft ready")
    assert draft_requests and b'name="reasoning_effort"' in draft_requests[-1]
    assert b"low" in draft_requests[-1]
    assert page.evaluate("state.reportCopied") is False


def test_copy_keeps_section_heading_attached_to_its_text(page: Page, base_url: str):
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          setReport(
            "**EXAM:**\\n\\nMRI lumbar spine\\n\\n" +
            "**TECHNIQUE:**\\n\\nRoutine non-contrast protocol.\\n\\n" +
            "**FINDINGS:**\\n\\nNo acute abnormality.\\n\\n" +
            "**IMPRESSION:**\\n\\n- No acute abnormality."
          );
          setUI("done");
          document.body.dataset.pasteFormat = "rich";
          window.__sectionClipboard = {};
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: {
              write: async (items) => {
                for (const type of items[0].types) {
                  window.__sectionClipboard[type] =
                    await (await items[0].getType(type)).text();
                }
              },
            },
          });
        }"""
    )

    page.locator("#btn-copy").click()
    page.wait_for_function(
        "() => Boolean(window.__sectionClipboard['text/plain'])"
    )
    clipboard = page.evaluate("window.__sectionClipboard")
    assert clipboard["text/plain"] == (
        "TECHNIQUE:\nRoutine non-contrast protocol.\n\n"
        "FINDINGS:\nNo acute abnormality.\n\n"
        "IMPRESSION:\n- No acute abnormality."
    )
    assert "EXAM" not in clipboard["text/html"]
    assert "MRI lumbar spine" not in clipboard["text/html"]
    assert "<strong>TECHNIQUE:</strong><br>Routine non-contrast protocol.<br><br>" in clipboard["text/html"]
    expect(page.locator("#report-rendered")).to_contain_text("MRI lumbar spine")
    expect(page.locator("#report-raw")).to_have_value(re.compile("EXAM"))
    for paste_format in ("plain", "markdown"):
        page.evaluate("""(format) => {
          document.body.dataset.pasteFormat = format;
          navigator.clipboard.writeText = async text => { window.__plainCopy = text; };
        }""", paste_format)
        page.locator("#btn-copy").click()
        copied = page.evaluate("window.__plainCopy")
        assert "EXAM" not in copied and "MRI lumbar spine" not in copied
        assert "TECHNIQUE" in copied and "IMPRESSION" in copied
    for exam in ("EXAM:\nMRI knee", "**EXAM:** MRI knee", "### Examination:\nMRI knee"):
        assert page.evaluate("text => _reportWithoutExam(text)", exam + "\n\n**FINDINGS:**\nNo fracture.") == "**FINDINGS:**\nNo fracture."
    assert page.evaluate("text => _reportWithoutExam(text)", "**FINDINGS:**\nNo fracture.\n\n**IMPRESSION:**\n- No fracture.") == "**FINDINGS:**\nNo fracture.\n\n**IMPRESSION:**\n- No fracture."


def test_copy_preserves_knee_group_headings_without_gaps_between_findings(
    page: Page, base_url: str
):
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          setReport(
            "**FINDINGS:**\\n\\n" +
            "**Menisci**\\n\\n" +
            "**Medial meniscus:** Oblique undersurface tear.\\n\\n" +
            "**Lateral meniscus:** Small incomplete radial tear.\\n\\n" +
            "**Cruciate Ligaments**\\n\\n" +
            "**ACL and PCL:** Intact.\\n\\n" +
            "**IMPRESSION:**\\n\\n" +
            "1. Medial and lateral meniscal tears.\\n" +
            "2. Intact cruciate ligaments."
          );
          setUI("done");
          document.body.dataset.pasteFormat = "rich";
          window.__kneeClipboard = {};
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: {
              write: async (items) => {
                for (const type of items[0].types) {
                  window.__kneeClipboard[type] =
                    await (await items[0].getType(type)).text();
                }
              },
            },
          });
        }"""
    )

    page.locator("#btn-copy").click()
    page.wait_for_function("() => Boolean(window.__kneeClipboard['text/plain'])")
    clipboard = page.evaluate("window.__kneeClipboard")
    assert clipboard["text/plain"] == (
        "FINDINGS:\n"
        "Menisci\n"
        "Medial meniscus: Oblique undersurface tear.\n"
        "Lateral meniscus: Small incomplete radial tear.\n\n"
        "Cruciate Ligaments\n"
        "ACL and PCL: Intact.\n\n"
        "IMPRESSION:\n"
        "1. Medial and lateral meniscal tears.\n"
        "2. Intact cruciate ligaments."
    )
    assert "<strong>Menisci</strong><br>" in clipboard["text/html"]
    assert "<strong>Medial meniscus:</strong> Oblique undersurface tear." in clipboard["text/html"]
    assert "<strong>Lateral meniscus:</strong> Small incomplete radial tear." in clipboard["text/html"]
    assert "<strong>Cruciate Ligaments</strong><br>" in clipboard["text/html"]
    assert "<strong>ACL and PCL:</strong> Intact." in clipboard["text/html"]


def test_desktop_copy_uses_native_powerscribe_rtf_with_bold_headings(
    page: Page, base_url: str
):
    page.goto(f"{base_url}/app")
    assert page.evaluate("""() => _clipboardPlainText(
      '<p><strong>IMPRESSION:</strong></p><p>  First finding.<br>\u00a0 Second finding.</p>', ''
    )""") == "IMPRESSION:\nFirst finding.\nSecond finding."
    page.evaluate(
        """() => {
          setReport(
            "**EXAM:** MRI knee\\n\\n" +
            "FINDINGS:\\n\\n" +
            "**Menisci**\\n\\n" +
            "**Medial meniscus:** Oblique undersurface tear.\\n\\n" +
            "**IMPRESSION:**\\n\\n" +
            "1. Medial meniscal tear."
          );
          setUI("done");
          document.body.dataset.pasteFormat = "rich";
          window.__nativeCopy = null;
          window.__TAURI__ = {
            core: {
              invoke: async (command, args) => {
                window.__nativeCopy = { command, args };
              },
            },
          };
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: {
              write: async () => { throw new Error("browser clipboard must not run"); },
            },
          });
        }"""
    )

    page.locator("#btn-copy").click()
    page.wait_for_function("() => Boolean(window.__nativeCopy)")
    native_copy = page.evaluate("window.__nativeCopy")

    assert native_copy["command"] == "cmd_copy_report_rtf"
    assert native_copy["args"]["body"]["text"] == (
        "FINDINGS:\n"
        "Menisci\n"
        "Medial meniscus: Oblique undersurface tear.\n\n"
        "IMPRESSION:\n"
        "1. Medial meniscal tear."
    )
    assert native_copy["args"]["body"]["boldLines"] == [
        "FINDINGS:",
        "Menisci",
        "IMPRESSION:",
    ]
    assert native_copy["args"]["body"]["boldPrefixes"] == ["Medial meniscus:"]
    expect(page.locator("#status")).to_contain_text("PowerScribe rich text")


def test_keyboard_first_reporting_loop_and_automatic_qa(page: Page, base_url: str):
    errors = _console_errors(page)
    qa_requests: list[dict] = []
    page.on(
        "request",
        lambda request: qa_requests.append(request.post_data_json)
        if request.url.endswith("/api/qa-check")
        else None,
    )
    page.goto(f"{base_url}/app")

    expect(page.locator(".shortcut-strip")).to_contain_text("Ctrl/Cmd+Enter")
    page.locator("#transcription").fill(
        "CT chest with contrast. No focal pulmonary lesion or pleural effusion."
    )
    expect(page.locator("#btn-format")).to_be_enabled()
    page.keyboard.press("Control+Enter")
    expect(page.locator("#report-rendered")).to_contain_text(
        "No acute cardiopulmonary abnormality", timeout=15_000
    )
    page.wait_for_function("() => window.performance.now() > 0 && document.querySelector('#status').textContent !== 'Generating report…'")
    page.wait_for_timeout(200)
    assert qa_requests, "report generation should trigger deterministic QA automatically"
    assert qa_requests[-1]["source_text"].startswith("CT chest with contrast")

    page.evaluate(
        """() => {
          document.body.dataset.pasteFormat = "plain";
          window.__copiedReport = "";
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: { writeText: async (text) => { window.__copiedReport = text; } },
          });
        }"""
    )
    page.keyboard.press("Control+Shift+C")
    page.wait_for_function("() => window.__copiedReport.includes('No acute cardiopulmonary abnormality')")
    expect(page.locator("#status")).to_contain_text("Report copied")
    assert errors == []


def test_qa_infers_laterality_from_body_part(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    page.locator("#patient-context-details > summary").click()
    page.locator("#body-part").fill("Right knee")
    page.evaluate(
        """() => {
          setReport("**Findings:**\\nThe left meniscus is intact.");
          setUI("done");
        }"""
    )
    page.locator("#btn-qa").click()
    expect(page.locator("#qa-panel")).to_contain_text(
        "Order is for the RIGHT side", timeout=5_000
    )

    page.locator("#body-part").fill("Knee")
    page.locator("#transcription").fill("Complete rupture of the arcuate ligament.")
    page.evaluate(
        "setReport('**FINDINGS:**\\nPopliteus tendon tear.'); setUI('done');"
    )
    page.locator("#btn-qa").click()
    expect(page.locator("#qa-panel")).to_contain_text(
        "dictated finding may be missing", timeout=5_000
    )
    expect(page.locator("#qa-panel")).to_contain_text("arcuate ligament")
    assert errors == []


# Value: protects=worksheet QA hides ambiguous labels only when the report covers them;
# fails_when=the browser/API path regresses or the exception spreads to asserted findings;
# why_new=unit coverage does not prove the report screen sends and renders QA results;
# seam=none
def test_worksheet_qa_tones_down_labels_but_keeps_asserted_findings(
    page: Page, base_url: str
):
    errors = _console_errors(page)
    qa_requests: list[dict] = []
    page.on(
        "request",
        lambda request: qa_requests.append(request.post_data_json)
        if request.url.endswith("/api/qa-check")
        else None,
    )
    page.goto(f"{base_url}/app")
    report = (
        "**FINDINGS:**\nNo evidence of free fluid collection.\n\n"
        "**IMPRESSION:**\nNo evidence of ovarian or adnexal mass bilaterally."
    )
    page.locator("#transcription").fill(
        "FORM EXTRACT — blank fields are unknown:\n"
        "Free fluid collection seen\n"
        "Adnexal mass seen bilaterally"
    )
    page.evaluate(
        "report => { state.sourceKind = 'worksheet'; setReport(report); setUI('done'); }",
        report,
    )

    page.locator("#btn-qa").click()
    expect(page.locator("#qa-panel")).to_contain_text(
        "No flags raised", timeout=5_000
    )
    assert qa_requests[-1]["source_kind"] == "worksheet"

    page.evaluate("state.sourceKind = 'worksheet'")
    transcript = page.locator("#transcription")
    transcript.fill(f"{transcript.input_value()}\nAdditional dictated clinical text")
    with page.expect_request("**/api/qa-check") as edited_request:
        page.locator("#btn-qa").click()
    assert edited_request.value.post_data_json["source_kind"] == "dictation"

    page.locator("#transcription").fill(
        "WORKSHEET SOURCE NOTES — blank/unmarked fields are unknown:\n"
        "Pelvis / free fluid: Collection seen"
    )
    page.evaluate("state.sourceKind = 'worksheet'")
    with page.expect_request("**/api/qa-check") as structured_request:
        page.locator("#btn-qa").click()
    expect(page.locator("#qa-panel")).to_contain_text(
        "dictated finding may be missing", timeout=5_000
    )
    expect(page.locator("#qa-panel")).to_contain_text("Collection seen")
    assert structured_request.value.post_data_json["source_kind"] == "worksheet"

    page.evaluate(
        """() => {
          state.sourceKind = 'worksheet';
          Object.defineProperty(navigator, 'mediaDevices', {
            configurable: true,
            value: {getUserMedia: async () => { throw new Error('synthetic denial'); }},
          });
        }"""
    )
    page.locator("#btn-record").click()
    expect(page.locator("#status")).to_contain_text("Could not start microphone")
    with page.expect_request("**/api/qa-check") as request_info:
        page.locator("#btn-qa").click()
    expect(page.locator("#qa-panel")).to_contain_text(
        "dictated finding may be missing", timeout=5_000
    )
    assert request_info.value.post_data_json["source_kind"] == "dictation"
    assert errors == []


def test_rendered_impression_lists_are_compact(page: Page, base_url: str):
    page.goto(f"{base_url}/app")
    page.evaluate(
        "setReport('**IMPRESSION:**\\n1. First conclusion.\\n2. Second conclusion.\\n3. Third conclusion.');"
    )
    styles = page.locator("#report-rendered ol").evaluate(
        "el => ({margin: getComputedStyle(el).margin, paddingLeft: getComputedStyle(el).paddingLeft})"
    )
    item_styles = page.locator("#report-rendered li").first.evaluate(
        "el => ({marginTop: getComputedStyle(el).marginTop, marginBottom: getComputedStyle(el).marginBottom})"
    )
    assert styles["margin"].startswith("2px")
    assert float(styles["paddingLeft"].replace("px", "")) < 24
    assert item_styles == {"marginTop": "0px", "marginBottom": "0px"}

    page.evaluate("setReport('**IMPRESSION:**\\n- First.\\n- Second.');")
    unordered = page.locator("#report-rendered ul")
    expect(unordered).to_be_visible()
    assert unordered.evaluate("el => getComputedStyle(el).marginTop") == "2px"
    assert page.locator("#report-rendered li > p").count() == 0


def test_qscan_copy_starts_at_priors_and_keeps_numbered_conclusions(
    page: Page, base_url: str
):
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    expect(page.locator("#btn-copy-from-comparison")).to_contain_text(
        "Copy Comparison/Findings"
    )
    page.evaluate(
        """() => {
          setReport(
            "**Exam:**\\nMRI lumbar spine\\n\\n" +
            "**Technique:**\\nRoutine protocol.\\n\\n" +
            "**PRIORS:**\\nMRI 1/1/2025.\\n\\n" +
            "**Findings:**\\nNo fracture.\\n\\n" +
            "**Impression:**\\n1. First conclusion.\\n2. Second conclusion.\\n3. Third conclusion."
          );
          setUI("done");
          document.body.dataset.pasteFormat = "rich";
          window.__qscanClipboard = {};
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: {
              write: async (items) => {
                for (const type of items[0].types) {
                  window.__qscanClipboard[type] =
                    await (await items[0].getType(type)).text();
                }
              },
            },
          });
        }"""
    )

    page.locator("#btn-copy-from-comparison").click()
    page.wait_for_function("() => Boolean(window.__qscanClipboard['text/plain'])")
    clipboard = page.evaluate("window.__qscanClipboard")
    assert clipboard["text/plain"].startswith("PRIORS:\nMRI 1/1/2025.")
    assert "Exam:" not in clipboard["text/plain"]
    assert "Technique:" not in clipboard["text/plain"]
    assert "1. First conclusion." in clipboard["text/plain"]
    expect(page.locator("#status")).to_contain_text("copied from Comparison")
    assert errors == []


def test_qscan_copy_falls_back_to_findings_and_short_lists_use_hyphens(
    page: Page, base_url: str
):
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          setReport(
            "**Exam:**\\nCT chest\\n\\n" +
            "**Findings:**\\nNo acute abnormality.\\n\\n" +
            "**Conclusion:**\\n- No acute abnormality.\\n- No follow-up required."
          );
          setUI("done");
          document.body.dataset.pasteFormat = "plain";
          window.__qscanPlain = "";
          Object.defineProperty(navigator, "clipboard", {
            configurable: true,
            value: { writeText: async (text) => { window.__qscanPlain = text; } },
          });
        }"""
    )

    page.locator("#btn-copy-from-comparison").click()
    page.wait_for_function("() => Boolean(window.__qscanPlain)")
    copied = page.evaluate("window.__qscanPlain")
    assert copied.startswith("Findings:\nNo acute abnormality.")
    assert "Exam:" not in copied
    assert "- No acute abnormality.\n- No follow-up required." in copied


def test_worklist_switch_replaces_the_whole_case(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    page.locator("#patient-context-details > summary").click()
    response = page.request.post(
        f"{base_url}/api/worklist/push",
        headers={"X-VoxRad-Agent-Token": "synthetic-mwl-test-token"},
        data={
            "orders": [
                {
                    "patient_name": "Alice Example",
                    "patient_dob": "19600101",
                    "patient_id": "MRN-A",
                    "accession": "ACC-A",
                    "modality": "MR",
                    "body_part": "Left knee",
                },
                {
                    "patient_name": "Bob Example",
                    "patient_dob": "19700202",
                    "patient_id": "MRN-B",
                    "accession": "ACC-B",
                    "modality": "CT",
                    "body_part": "Chest",
                },
            ]
        },
    )
    assert response.ok
    assert response.json()["written"] == 2
    # The file-drop scanner deliberately ignores files still being written.
    time.sleep(1.1)
    page.locator("#btn-worklist-refresh").click()
    expect(page.locator("#worklist-select option")).to_have_count(3)

    page.locator("#worklist-select").select_option("mwl_ACC-A")
    expect(page.locator("#patient-name")).to_have_value("Alice Example")
    expect(page.locator("#body-part")).to_have_value("Left knee")
    expect(page.locator("#patient-summary")).to_contain_text("Alice Example")
    expect(page.locator("#patient-context-details")).not_to_have_attribute("open", "")

    page.locator("#transcription").fill("Unfinished dictation")
    page.locator("#patient-context-details > summary").click()
    page.once("dialog", lambda dialog: dialog.dismiss())
    page.locator("#worklist-select").select_option("mwl_ACC-B")
    expect(page.locator("#worklist-select")).to_have_value("mwl_ACC-A")
    expect(page.locator("#patient-name")).to_have_value("Alice Example")
    expect(page.locator("#transcription")).to_have_value("Unfinished dictation")
    page.locator("#transcription").fill("")

    # A copied/signed case is safe to advance. Switching should clear its text
    # and replace every patient field, never preserve Alice's populated values.
    page.evaluate(
        """() => {
          setReport("**Impression:**\\nNo acute abnormality.");
          setUI("done");
          state.reportCopied = true;
        }"""
    )
    page.locator("#worklist-select").select_option("mwl_ACC-B")
    expect(page.locator("#patient-name")).to_have_value("Bob Example")
    expect(page.locator("#patient-id")).to_have_value("MRN-B")
    expect(page.locator("#accession")).to_have_value("ACC-B")
    expect(page.locator("#body-part")).to_have_value("Chest")
    expect(page.locator("#report-raw")).to_have_value("")
    expect(page.locator("#transcription")).to_have_value("")
    assert errors == []


def test_followup_prompt_and_manual_score_insertion(page: Page, base_url: str):
    errors = _console_errors(page)
    page.goto(f"{base_url}/app")
    page.evaluate(
        """() => {
          setReport("**IMPRESSION:**\\nIndeterminate pulmonary nodule. Follow-up CT chest in 12 months is recommended.");
          setUI("done");
        }"""
    )
    expect(page.locator("#followup-suggest-panel")).to_contain_text(
        "Follow-up CT chest in 12 months", timeout=5_000
    )
    expect(page.locator("#followup-suggest-panel").get_by_role("button", name="Track")).to_be_visible()

    page.locator("#btn-scores").click()
    page.locator("#score-system").select_option("ACR TI-RADS")
    page.locator("#score-category").select_option("TR5")
    page.locator("#score-target").fill("Right thyroid nodule")
    expect(page.locator("#score-preview")).to_contain_text("TR5 — Highly suspicious")
    page.locator("#score-insert").click()
    expect(page.locator("#report-raw")).to_have_value(
        re.compile(r"Right thyroid nodule: TR5 — Highly suspicious")
    )
    assert errors == []


def test_mobile_impressions_has_no_horizontal_overflow(page: Page, base_url: str):
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(f"{base_url}/impressions")
    expect(page.locator("#findings")).to_be_visible()
    dimensions = page.evaluate(
        "() => ({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
    )
    assert dimensions["scrollWidth"] <= dimensions["clientWidth"]


def test_mobile_landing_has_no_horizontal_overflow(page: Page, base_url: str):
    page.set_viewport_size({"width": 375, "height": 812})
    page.goto(base_url)
    expect(
        page.get_by_role("heading", name="Radiology reporting built around the way you dictate.")
    ).to_be_visible()
    dimensions = page.evaluate(
        "() => ({scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth})"
    )
    assert dimensions["scrollWidth"] <= dimensions["clientWidth"]


def test_main_app_rejects_bad_basic_auth(browser: Browser, base_url: str):
    context = browser.new_context(
        http_credentials={"username": "voxrad", "password": "wrong-password"}
    )
    try:
        response = context.request.get(f"{base_url}/app")
        assert response.status == 401
        assert response.json()["detail"] == "Incorrect password"
    finally:
        context.close()
