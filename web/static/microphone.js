"use strict";

// Device-local controls: a shared hospital account must not change another
// workstation's microphone threshold. Read afresh at each recording start.
window.RadSpeedMicrophone = (() => {
  const KEY = "radspeed.microphone.v1";
  const LEVELS = { high: 0.003, balanced: 0.006, low: 0.018 };
  const sessions = new Map();

  function preferences() {
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
      return {
        enabled: saved.enabled !== false,
        sensitivity: Object.hasOwn(LEVELS, saved.sensitivity) ? saved.sensitivity : "balanced",
      };
    } catch (_) { return { enabled: true, sensitivity: "balanced" }; }
  }

  function save(value) {
    localStorage.setItem(KEY, JSON.stringify({
      enabled: value.enabled !== false,
      sensitivity: Object.hasOwn(LEVELS, value.sensitivity) ? value.sensitivity : "balanced",
    }));
  }

  function stop(stream) {
    if (!stream) return;
    const session = sessions.get(stream);
    sessions.delete(stream);
    stream.getTracks().forEach((track) => track.stop());
    if (session) {
      session.raw.getTracks().forEach((track) => track.stop());
      session.source.disconnect();
      session.node.disconnect();
      session.node.port.onmessage = null;
      session.context.close().catch(() => {});
    }
  }

  async function open(onLevel, { signal } = {}) {
    const prefs = preferences();
    const raw = await navigator.mediaDevices.getUserMedia({ audio: {
      noiseSuppression: prefs.enabled,
      echoCancellation: prefs.enabled,
      // Automatic gain can make distant dictation as loud as nearby speech.
      autoGainControl: !prefs.enabled,
      channelCount: 1,
    } });
    if (signal?.aborted) {
      raw.getTracks().forEach((track) => track.stop());
      throw new Error("Microphone test cancelled.");
    }
    if (!prefs.enabled && !onLevel) return raw;
    let context;
    const cancel = () => {
      raw.getTracks().forEach((track) => track.stop());
      if (context) context.close().catch(() => {});
    };
    signal?.addEventListener("abort", cancel, { once: true });
    try {
      context = new (window.AudioContext || window.webkitAudioContext)();
      const script = new URL(document.querySelector('script[src*="/static/microphone.js"]').src);
      const worklet = new URL("microphone-worklet.js", script);
      worklet.search = script.search;
      await context.audioWorklet.addModule(worklet);
      if (signal?.aborted) throw new Error("Microphone test cancelled.");
      const source = context.createMediaStreamSource(raw);
      const node = new AudioWorkletNode(context, "reporting-microphone", {
        outputChannelCount: [1],
        processorOptions: { threshold: prefs.enabled ? LEVELS[prefs.sensitivity] : 0 },
      });
      const destination = context.createMediaStreamDestination();
      node.port.onmessage = (event) => { if (onLevel) onLevel(event.data); };
      source.connect(node);
      node.connect(destination); // Never connect microphone audio to speakers.
      await context.resume();
      if (signal?.aborted) throw new Error("Microphone test cancelled.");
      const stream = destination.stream;
      sessions.set(stream, { raw, context, source, node, prefs });
      raw.getTracks().forEach((track) => track.addEventListener("ended", () => stop(stream), { once: true }));
      return stream;
    } catch (error) {
      raw.getTracks().forEach((track) => track.stop());
      if (context) await context.close().catch(() => {});
      // Do not silently send unfiltered audio when filtering was requested.
      throw new Error("Background filtering could not start. Try again, or turn it off in Settings. " + error.message);
    } finally {
      signal?.removeEventListener("abort", cancel);
    }
  }

  function speechThreshold(stream) {
    const prefs = sessions.get(stream)?.prefs;
    return prefs?.enabled ? LEVELS[prefs.sensitivity] : 0.015;
  }

  window.addEventListener("pagehide", () => { for (const stream of sessions.keys()) stop(stream); });
  return { preferences, save, open, stop, speechThreshold };
})();
