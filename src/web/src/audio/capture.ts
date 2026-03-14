/**
 * Microphone capture via AudioWorklet.
 * Outputs PCM16 24kHz chunks as ArrayBuffer.
 */
export class AudioCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private workletNode: AudioWorkletNode | null = null;

  async start(onChunk: (pcm16: ArrayBuffer) => void): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Try 24kHz directly; browser may choose closest supported rate
    this.context = new AudioContext({ sampleRate: 24000 });
    await this.context.audioWorklet.addModule("/capture-processor.js");

    const source = this.context.createMediaStreamSource(this.stream);
    this.workletNode = new AudioWorkletNode(this.context, "capture-processor");

    this.workletNode.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
      onChunk(e.data);
    };

    source.connect(this.workletNode);
    // Don't connect to destination — we don't want mic playback
  }

  stop(): void {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.context) {
      this.context.close();
      this.context = null;
    }
  }
}
