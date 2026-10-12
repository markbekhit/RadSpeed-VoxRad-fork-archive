"""Microphone privacy, settings, and filtering with synthetic signals only."""
import pytest
from playwright.sync_api import expect


def test_microphone_preferences_are_local_and_persist(page, base_url):
    writes = []
    page.on("request", lambda request: writes.append(request.url) if request.method == "POST" else None)
    page.goto(f"{base_url}/settings")
    expect(page.locator("#microphone-filter")).to_be_checked()
    expect(page.locator("#microphone-sensitivity")).to_have_value("balanced")
    page.locator("#microphone-sensitivity").select_option("low")
    page.reload()
    expect(page.locator("#microphone-sensitivity")).to_have_value("low")
    page.locator("#microphone-filter").uncheck()
    expect(page.locator("#microphone-sensitivity")).to_be_disabled()
    page.goto(f"{base_url}/app?desktop=overlay")
    assert page.evaluate("RadSpeedMicrophone.preferences()") == {"enabled": False, "sensitivity": "low"}
    assert writes == []


def test_microphone_invalid_storage_keeps_safe_defaults(page, base_url):
    page.goto(f"{base_url}/settings")
    page.evaluate("localStorage.setItem('radspeed.microphone.v1', 'broken')")
    page.reload()
    expect(page.locator("#microphone-filter")).to_be_checked()
    expect(page.locator("#microphone-sensitivity")).to_have_value("balanced")


def test_microphone_gate_preserves_speech_and_closes_after_release(page, base_url):
    page.goto(f"{base_url}/settings")
    result = page.evaluate("""async () => {
      const source = await (await fetch('/static/microphone-worklet.js')).text();
      let Processor;
      const factory = new Function('AudioWorkletProcessor', 'registerProcessor', 'sampleRate', source);
      factory(class { constructor() { this.port = {postMessage() {}}; } },
        (_, value) => { Processor = value; }, 48000);
      function signal(processor, amplitude, frames = 1) {
        let energy = 0;
        for (let frame = 0; frame < frames; frame++) {
          const input = Float32Array.from({length: 128}, (_, i) => (i % 2 ? 1 : -1) * amplitude);
          const output = new Float32Array(128);
          processor.process([[input]], [[output]]);
          energy = output.reduce((sum, sample) => sum + sample * sample, 0) / 128;
        }
        return energy;
      }
      const gate = new Processor({processorOptions: {threshold: 0.006}});
      const background = signal(gate, 0.002, 20);
      const speech = signal(gate, 0.05, 2);
      const quietTail = signal(gate, 0.002, 30);
      const closed = signal(gate, 0.002, 150);
      const high = signal(new Processor({processorOptions: {threshold: 0.003}}), 0.01, 3);
      const low = signal(new Processor({processorOptions: {threshold: 0.018}}), 0.01, 3);
      const bypass = signal(new Processor({processorOptions: {threshold: 0}}), 0.002, 3);
      const missing = new Float32Array(128).fill(1);
      gate.process([[]], [[missing]]);
      return {background, speech, quietTail, closed, high, low, bypass,
        missing: missing.some(value => value !== 0)};
    }""")
    assert result["background"] == 0
    assert result["speech"] > 0.002
    assert result["quietTail"] > 0
    assert result["closed"] == 0
    assert result["high"] > 0
    assert result["low"] == 0
    assert result["bypass"] > 0
    assert result["missing"] is False


def _synthetic_microphone(page):
    page.evaluate("""() => {
      window.syntheticContext = new AudioContext();
      window.syntheticOscillator = syntheticContext.createOscillator();
      window.syntheticGain = syntheticContext.createGain();
      syntheticGain.gain.value = 0.001;
      window.syntheticDestination = syntheticContext.createMediaStreamDestination();
      syntheticOscillator.connect(syntheticGain).connect(syntheticDestination);
      syntheticOscillator.start();
      syntheticContext.resume();
      navigator.mediaDevices.getUserMedia = async constraints => {
        window.microphoneConstraints = constraints;
        return syntheticDestination.stream;
      };
    }""")


@pytest.mark.parametrize("stereo", [False, True])
def test_microphone_real_audio_graph_and_cleanup(page, base_url, stereo):
    page.goto(f"{base_url}/settings")
    _synthetic_microphone(page)
    if stereo:
        page.evaluate("""() => {
          // Simulate hardware that ignores channelCount:1 and delivers the
          // voice on the right channel only. The worklet must downmix it.
          syntheticGain.disconnect();
          window.syntheticMerger = syntheticContext.createChannelMerger(2);
          syntheticGain.connect(syntheticMerger, 0, 1);
          syntheticMerger.connect(syntheticDestination);
        }""")
    page.evaluate("""async () => {
      window.filteredStream = await RadSpeedMicrophone.open(level => { window.inputLevel = level; });
      window.outputContext = new AudioContext();
      window.outputAnalyser = outputContext.createAnalyser();
      outputContext.createMediaStreamSource(filteredStream).connect(outputAnalyser);
      await outputContext.resume();
      window.outputEnergy = () => {
        const data = new Float32Array(outputAnalyser.fftSize);
        outputAnalyser.getFloatTimeDomainData(data);
        return data.reduce((sum, x) => sum + x*x, 0) / data.length;
      };
    }""")
    page.wait_for_function("window.inputLevel && !inputLevel.passing")
    assert page.evaluate("outputEnergy()") == 0
    assert page.evaluate("microphoneConstraints.audio") == {
        "noiseSuppression": True, "echoCancellation": True, "autoGainControl": False, "channelCount": 1,
    }
    page.evaluate("syntheticGain.gain.value = 0.1")
    page.wait_for_function("inputLevel.passing && outputEnergy() > 0.001")
    page.evaluate("syntheticGain.gain.value = 0.001")
    page.wait_for_function("!inputLevel.passing && outputEnergy() === 0")
    page.evaluate("RadSpeedMicrophone.stop(filteredStream)")
    assert page.evaluate("syntheticDestination.stream.getTracks().every(t => t.readyState === 'ended')")
    assert page.evaluate("filteredStream.getTracks().every(t => t.readyState === 'ended')")
    page.evaluate("syntheticContext.close(); outputContext.close()")


def test_microphone_test_stays_local_and_stops(page, base_url):
    page.goto(f"{base_url}/settings")
    _synthetic_microphone(page)
    requests = []
    page.on("request", lambda request: requests.append(request.url) if request.method == "POST" else None)
    page.locator("#microphone-test").click()
    expect(page.locator("#microphone-test")).to_have_text("Stop test")
    expect(page.locator("#microphone-test-status")).to_contain_text("Filtering quieter audio")
    page.evaluate("syntheticGain.gain.value = 0.1")
    expect(page.locator("#microphone-test-status")).to_contain_text("Passing audio")
    page.locator("#microphone-test").click()
    expect(page.locator("#microphone-test-status")).to_contain_text("No audio was saved or sent")
    assert page.evaluate("syntheticDestination.stream.getTracks()[0].readyState") == "ended"
    assert requests == []
    page.evaluate("syntheticContext.close()")


def test_microphone_failure_stops_raw_capture_without_unfiltered_fallback(page, base_url):
    page.goto(f"{base_url}/settings")
    _synthetic_microphone(page)
    result = page.evaluate("""async () => {
      window.AudioContext = class {
        constructor() { this.audioWorklet = {addModule: async () => { throw new Error('synthetic failure'); }}; }
        async close() { window.failedContextClosed = true; }
      };
      try { await RadSpeedMicrophone.open(); return 'unexpected success'; }
      catch (error) { return error.message; }
    }""")
    assert "Background filtering could not start" in result
    assert page.evaluate("failedContextClosed")
    assert page.evaluate("syntheticDestination.stream.getTracks()[0].readyState") == "ended"
    page.evaluate("syntheticContext.close()")


def test_microphone_filter_off_uses_raw_stream(page, base_url):
    page.goto(f"{base_url}/settings")
    _synthetic_microphone(page)
    page.locator("#microphone-filter").uncheck()
    assert page.evaluate("""async () => {
      const stream = await RadSpeedMicrophone.open();
      const raw = stream === syntheticDestination.stream;
      RadSpeedMicrophone.stop(stream);
      return raw;
    }""")
    assert page.evaluate("microphoneConstraints.audio.noiseSuppression") is False
    assert page.evaluate("microphoneConstraints.audio.autoGainControl") is True
    page.evaluate("syntheticContext.close()")


def test_microphone_test_timeout_and_pending_cancellation(page, base_url):
    page.goto(f"{base_url}/settings")
    _synthetic_microphone(page)
    page.evaluate("""() => {
      const schedule = window.setTimeout;
      window.setTimeout = (callback, delay, ...args) => {
        if (delay === 30000) window.finishMicrophoneTest = callback;
        return schedule(callback, delay, ...args);
      };
    }""")
    page.locator("#microphone-test").click()
    expect(page.locator("#microphone-test")).to_have_text("Stop test")
    page.evaluate("finishMicrophoneTest()")
    expect(page.locator("#microphone-test-status")).to_contain_text("Test finished")
    assert page.evaluate("syntheticDestination.stream.getTracks()[0].readyState") == "ended"
    page.evaluate("syntheticContext.close()")

    # A permission prompt can resolve after settings change. That late stream
    # must close rather than starting a hidden test microphone.
    _synthetic_microphone(page)
    page.evaluate("() => { navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { window.resolveMicrophone = resolve; }); }")
    page.locator("#microphone-test").click()
    expect(page.locator("#microphone-test-status")).to_contain_text("Opening microphone")
    page.locator("#microphone-sensitivity").select_option("low")
    page.evaluate("resolveMicrophone(syntheticDestination.stream)")
    page.wait_for_function("syntheticDestination.stream.getTracks()[0].readyState === 'ended'")
    expect(page.locator("#microphone-test")).to_have_text("Test microphone")
    page.evaluate("syntheticContext.close()")


def test_microphone_settings_render_on_small_screens(page, base_url):
    page.goto(f"{base_url}/settings")
    page.set_viewport_size({"width": 390, "height": 844})
    expect(page.locator("#microphone-sensitivity")).to_be_visible()
    assert page.evaluate("document.documentElement.scrollWidth") <= 390
    page.screenshot(path="/tmp/radspeed-microphone-settings.png")


@pytest.mark.parametrize("mode", ["segment", "voice-edit", "refine", "streaming"])
def test_all_recording_modes_use_filtered_microphone(page, base_url, mode):
    page.goto(f"{base_url}/app")
    _synthetic_microphone(page)
    page.evaluate("""() => {
      window.microphoneOpens = 0;
      const open = RadSpeedMicrophone.open;
      RadSpeedMicrophone.open = async (...args) => { microphoneOpens++; return open(...args); };
      window.WebSocket = class {
        constructor() { this.readyState = 0; }
        close() {}
      };
    }""")
    if mode == "segment":
        page.evaluate("startSegmentRecording()")
        assert page.evaluate("state.isRecording")
    elif mode == "voice-edit":
        page.evaluate("state.voiceEditTarget = {elId: 'report-raw', selectedText: 'synthetic'}; startVoiceEditRecording()")
        assert page.evaluate("state.isRecording")
    elif mode == "refine":
        page.evaluate("setReport('**IMPRESSION:** No acute abnormality.'); startFeedback()")
        assert page.evaluate("fbState.isRecording")
    else:
        page.evaluate("startStreamingRecording()")
        assert page.evaluate("!!state.streamingWs")
    assert page.evaluate("microphoneOpens") == 1
    assert page.evaluate("microphoneConstraints.audio.noiseSuppression")
    page.evaluate("""() => {
      RadSpeedMicrophone.stop(state.stream);
      RadSpeedMicrophone.stop(fbState.stream);
      stopWaveform(); stopTimer(); syntheticContext.close();
    }""")
