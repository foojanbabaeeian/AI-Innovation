// AudioWorkletProcessor that:
//   1. Downmixes incoming (possibly multichannel) tab audio to mono float32
//   2. Accumulates samples into fixed-size frames (~200ms)
//   3. Posts each frame as a Float32Array to the main thread via port.postMessage
//
// We don't resample here. The browser sample rate is sent to the backend in
// the StartMessage, and the backend handles resampling to the model's rate.

class DetectorProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const { frameSize = 4800 } = options?.processorOptions || {};
    this._frameSize = frameSize; // ~100ms @ 48kHz
    this._buffer = new Float32Array(frameSize);
    this._fill = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const frames = input[0].length;
    const channels = input.length;

    for (let i = 0; i < frames; i++) {
      let sum = 0;
      for (let c = 0; c < channels; c++) {
        const ch = input[c];
        if (ch) sum += ch[i];
      }
      this._buffer[this._fill++] = sum / channels;

      if (this._fill >= this._frameSize) {
        // Transfer a copy so we don't reuse the same backing buffer.
        const out = new Float32Array(this._buffer);
        this.port.postMessage(out, [out.buffer]);
        this._buffer = new Float32Array(this._frameSize);
        this._fill = 0;
      }
    }
    return true;
  }
}

registerProcessor("detector-processor", DetectorProcessor);
