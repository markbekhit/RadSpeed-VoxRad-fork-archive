/* Level-based filtering, not speaker recognition. Browser noise suppression
 * runs first. A close microphone is still needed beside another speaker.
 * No look-ahead delay: stopping immediately must not lose buffered speech.
 */
class ReportingMicrophone extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.threshold = options.processorOptions.threshold;
    this.hold = 0;
    this.gain = 0;
    this.meterFrames = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!output) return true;
    let power = 0;
    if (input) for (const sample of input) power += sample * sample;
    const rms = input?.length ? Math.sqrt(power / input.length) : 0;
    if (this.threshold === 0 || rms >= this.threshold) {
      // Keep quiet consonants and short gaps within an utterance.
      this.hold = Math.round(sampleRate * 0.35);
    }
    const passing = this.hold > 0;
    const target = passing ? 1 : 0;
    const step = 1 / (sampleRate * 0.002); // 2 ms ramp avoids clicks.
    for (let i = 0; i < output.length; i++) {
      this.gain += Math.max(-step, Math.min(step, target - this.gain));
      output[i] = (input?.[i] || 0) * this.gain;
    }
    this.hold = Math.max(0, this.hold - output.length);
    this.meterFrames += output.length;
    if (this.meterFrames >= sampleRate / 10) {
      this.port.postMessage({ rms, passing, threshold: this.threshold });
      this.meterFrames = 0;
    }
    return true;
  }
}
registerProcessor("reporting-microphone", ReportingMicrophone);
