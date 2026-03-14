/**
 * AudioWorklet processor for audio playback.
 * Receives Int16 PCM 24kHz buffers from main thread,
 * resamples to context rate, and outputs to speakers.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = []; // Array of Float32Arrays
    this._queueOffset = 0;
    this._sourceRate = 24000;

    this.port.onmessage = (e) => {
      if (e.data === "clear") {
        this._queue = [];
        this._queueOffset = 0;
        return;
      }
      // e.data is ArrayBuffer of Int16 PCM at 24kHz
      const pcm16 = new Int16Array(e.data);
      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768;
      }

      // Resample from 24kHz to context sampleRate if needed
      if (sampleRate !== this._sourceRate) {
        const ratio = sampleRate / this._sourceRate;
        const outLen = Math.floor(float32.length * ratio);
        const resampled = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
          const srcIdx = i / ratio;
          const idx0 = Math.floor(srcIdx);
          const idx1 = Math.min(idx0 + 1, float32.length - 1);
          const frac = srcIdx - idx0;
          resampled[i] = float32[idx0] + (float32[idx1] - float32[idx0]) * frac;
        }
        this._queue.push(resampled);
      } else {
        this._queue.push(float32);
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) return true;

    const channel = output[0];
    let written = 0;

    while (written < channel.length && this._queue.length > 0) {
      const current = this._queue[0];
      const available = current.length - this._queueOffset;
      const needed = channel.length - written;
      const toCopy = Math.min(available, needed);

      for (let i = 0; i < toCopy; i++) {
        channel[written + i] = current[this._queueOffset + i];
      }

      written += toCopy;
      this._queueOffset += toCopy;

      if (this._queueOffset >= current.length) {
        this._queue.shift();
        this._queueOffset = 0;
      }
    }

    // Fill remaining with silence
    for (let i = written; i < channel.length; i++) {
      channel[i] = 0;
    }

    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
