/**
 * Audio playback via AudioWorklet.
 * Receives base64-encoded PCM16 24kHz audio and plays it.
 */
export class AudioPlayback {
  private context: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;

  async start(): Promise<void> {
    this.context = new AudioContext({ sampleRate: 24000 });

    // Resume context immediately — required by Chrome/Safari autoplay policy.
    // start() is called from a user gesture (click), so resume will succeed.
    if (this.context.state === "suspended") {
      await this.context.resume();
    }

    await this.context.audioWorklet.addModule("/playback-processor.js");

    this.workletNode = new AudioWorkletNode(this.context, "playback-processor");
    this.workletNode.connect(this.context.destination);
  }

  enqueue(pcm16Base64: string): void {
    if (!this.workletNode) return;

    const binary = atob(pcm16Base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    this.workletNode.port.postMessage(bytes.buffer, [bytes.buffer]);
  }

  clear(): void {
    if (this.workletNode) {
      this.workletNode.port.postMessage("clear");
    }
  }

  /** Resume context after user gesture (autoplay policy). */
  async resume(): Promise<void> {
    if (this.context?.state === "suspended") {
      await this.context.resume();
    }
  }

  stop(): void {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.context) {
      this.context.close();
      this.context = null;
    }
  }
}
