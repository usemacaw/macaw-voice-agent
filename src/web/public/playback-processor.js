/**
 * AudioWorklet processor for audio playback.
 * Receives Int16 PCM 24kHz buffers from main thread,
 * resamples to context rate, and outputs to speakers.
 *
 * Anti-stutter: pre-buffers ~150ms before starting playback,
 * and allows up to 2s queue depth to cover gaps between TTS sentences.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._queue = []; // Array of Float32Arrays
    this._queueOffset = 0;
    this._sourceRate = 24000;
    // Max queue depth: ~2s of audio (covers gaps between TTS sentences)
    this._maxQueueSamples = Math.floor(sampleRate * 2.0);
    this._queueSamples = 0;
    // Pre-buffer: accumulate ~200ms before starting playback
    // This prevents stutter when first chunks arrive with gaps
    this._preBufferSamples = Math.floor(sampleRate * 0.2);
    this._isBuffering = true;
    this._hasStartedPlaying = false; // Only pre-buffer once per response

    this.port.onmessage = (e) => {
      if (e.data === "clear") {
        this._queue = [];
        this._queueOffset = 0;
        this._queueSamples = 0;
        this._isBuffering = true;
        this._hasStartedPlaying = false; // Reset on clear (new response)
        return;
      }
      // e.data is ArrayBuffer of Int16 PCM at 24kHz
      const pcm16 = new Int16Array(e.data);
      const float32 = new Float32Array(pcm16.length);
      for (let i = 0; i < pcm16.length; i++) {
        float32[i] = pcm16[i] / 32768;
      }

      let resampled;
      // Resample from 24kHz to context sampleRate if needed
      if (sampleRate !== this._sourceRate) {
        const ratio = sampleRate / this._sourceRate;
        const outLen = Math.floor(float32.length * ratio);
        resampled = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) {
          const srcIdx = i / ratio;
          const idx0 = Math.floor(srcIdx);
          const idx1 = Math.min(idx0 + 1, float32.length - 1);
          const frac = srcIdx - idx0;
          resampled[i] = float32[idx0] + (float32[idx1] - float32[idx0]) * frac;
        }
      } else {
        resampled = float32;
      }

      this._queue.push(resampled);
      this._queueSamples += resampled.length;

      // Check if pre-buffer threshold reached (only first time)
      if (this._isBuffering && this._queueSamples >= this._preBufferSamples) {
        this._isBuffering = false;
        this._hasStartedPlaying = true;
      }

      // Evict oldest chunks if queue exceeds max depth (~2s)
      while (this._queueSamples > this._maxQueueSamples && this._queue.length > 1) {
        const evicted = this._queue.shift();
        this._queueSamples -= evicted.length;
        this._queueOffset = 0;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || !output[0]) return true;

    const channel = output[0];

    // Pre-buffering: output silence until enough audio accumulated
    if (this._isBuffering) {
      for (let i = 0; i < channel.length; i++) {
        channel[i] = 0;
      }
      return true;
    }

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
      this._queueSamples -= toCopy;

      if (this._queueOffset >= current.length) {
        this._queue.shift();
        this._queueOffset = 0;
      }
    }

    // Fill remaining with silence (buffer underrun)
    for (let i = written; i < channel.length; i++) {
      channel[i] = 0;
    }

    // Do NOT re-enter pre-buffer mode after playback started.
    // Silence gaps between sentences are normal and brief (~90ms TTFA).
    // Re-buffering would add an extra 200ms delay on top of each gap.

    return true;
  }
}

registerProcessor("playback-processor", PlaybackProcessor);
