/**
 * AudioWorklet processor for microphone capture.
 * Captures audio, resamples to 24kHz, converts to Int16 PCM,
 * and posts ArrayBuffer chunks to the main thread.
 *
 * Uses Float32Array ring buffer instead of plain Array for O(1) writes
 * and efficient chunked reads.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._targetRate = 24000;
    // Send ~20ms chunks (480 samples at 24kHz = 960 bytes)
    this._chunkSamples = 480;
    // Ring buffer: pre-allocate enough for ~200ms of audio at target rate
    this._ringSize = this._chunkSamples * 10;
    this._ring = new Float32Array(this._ringSize);
    this._writePos = 0;
    this._readPos = 0;
  }

  _available() {
    return (this._writePos - this._readPos + this._ringSize) % this._ringSize;
  }

  _push(value) {
    this._ring[this._writePos] = value;
    this._writePos = (this._writePos + 1) % this._ringSize;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array at context sampleRate

    // Resample to 24kHz if needed and push into ring buffer
    if (sampleRate === this._targetRate) {
      for (let i = 0; i < samples.length; i++) {
        this._push(samples[i]);
      }
    } else {
      const ratio = this._targetRate / sampleRate;
      const outLen = Math.floor(samples.length * ratio);
      for (let i = 0; i < outLen; i++) {
        const srcIdx = i / ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, samples.length - 1);
        const frac = srcIdx - idx0;
        this._push(samples[idx0] + (samples[idx1] - samples[idx0]) * frac);
      }
    }

    // Emit chunks from ring buffer
    while (this._available() >= this._chunkSamples) {
      const pcm16 = new Int16Array(this._chunkSamples);
      for (let i = 0; i < this._chunkSamples; i++) {
        const s = Math.max(-1, Math.min(1, this._ring[this._readPos]));
        pcm16[i] = s < 0 ? s * 32768 : s * 32767;
        this._readPos = (this._readPos + 1) % this._ringSize;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
