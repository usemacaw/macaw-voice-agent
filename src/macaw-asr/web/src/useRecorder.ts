import { useState, useRef, useCallback } from 'react'

/**
 * Records audio from microphone and converts to WAV PCM16 (16kHz mono).
 * WAV format is universally supported by all ASR backends.
 */
export function useRecorder() {
  const [recording, setRecording] = useState(false)
  const [duration, setDuration] = useState(0)
  const ctxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const timerRef = useRef<number>(0)

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
    })
    const ctx = new AudioContext({ sampleRate: 16000 })
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)

    chunksRef.current = []
    processor.onaudioprocess = (e) => {
      const data = e.inputBuffer.getChannelData(0)
      chunksRef.current.push(new Float32Array(data))
    }

    source.connect(processor)
    processor.connect(ctx.destination)

    ctxRef.current = ctx
    streamRef.current = stream
    processorRef.current = processor
    setRecording(true)
    setDuration(0)

    const t0 = Date.now()
    timerRef.current = window.setInterval(() => {
      setDuration(Math.round((Date.now() - t0) / 1000))
    }, 500)
  }, [])

  const stop = useCallback((): File | null => {
    clearInterval(timerRef.current)
    processorRef.current?.disconnect()
    streamRef.current?.getTracks().forEach(t => t.stop())
    ctxRef.current?.close()
    setRecording(false)

    const chunks = chunksRef.current
    if (!chunks.length) return null

    // Merge chunks
    const totalLen = chunks.reduce((s, c) => s + c.length, 0)
    const pcm = new Float32Array(totalLen)
    let offset = 0
    for (const c of chunks) { pcm.set(c, offset); offset += c.length }

    // Convert to WAV
    const wav = encodeWAV(pcm, 16000)
    return new File([wav], 'recording.wav', { type: 'audio/wav' })
  }, [])

  return { recording, duration, start, stop }
}

function encodeWAV(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const numChannels = 1
  const bitsPerSample = 16
  const bytesPerSample = bitsPerSample / 8
  const dataLen = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataLen)
  const view = new DataView(buffer)

  // RIFF header
  writeStr(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataLen, true)
  writeStr(view, 8, 'WAVE')
  // fmt
  writeStr(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)  // PCM
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true)
  view.setUint16(32, numChannels * bytesPerSample, true)
  view.setUint16(34, bitsPerSample, true)
  // data
  writeStr(view, 36, 'data')
  view.setUint32(40, dataLen, true)

  // PCM samples
  let idx = 44
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
    idx += 2
  }

  return buffer
}

function writeStr(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
}
