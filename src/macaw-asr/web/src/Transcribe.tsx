import { useState, useRef, useCallback } from 'react'
import { transcribe, type TranscribeResult, type ModelInfo } from './api'
import { useRecorder } from './useRecorder'
import TimingBar from './TimingBar'

export default function Transcribe({ models }: { models: ModelInfo[] }) {
  const [model, setModel] = useState('whisper-1')
  const [lang, setLang] = useState('pt')
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<TranscribeResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)
  const { recording, duration, start: startRec, stop: stopRec } = useRecorder()

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }, [])

  const doTranscribe = async (audioFile: File | Blob) => {
    setLoading(true)
    setError('')
    setResult(null)
    const t0 = performance.now()
    try {
      const r = await transcribe(audioFile, model, lang)
      setElapsed(Math.round(performance.now() - t0))
      setResult(r)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRecord = async () => {
    if (recording) {
      const wav = stopRec()
      if (wav) {
        setFile(wav)
        await doTranscribe(wav)
      }
    } else {
      setResult(null)
      startRec()
    }
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: '1rem', height: '100%' }}>
      {/* Left: Controls */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        <div className="card">
          <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.3rem' }}>Model</label>
          <select value={model} onChange={e => setModel(e.target.value)}>
            <option value="whisper-1">whisper-1 (server default)</option>
            {models.map(m => <option key={m.id} value={m.id}>{m.id} ({m.parameters})</option>)}
          </select>
        </div>

        <div className="card">
          <label style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.3rem' }}>Language</label>
          <input value={lang} onChange={e => setLang(e.target.value)} placeholder="pt, en, es..." />
        </div>

        {/* Drop zone */}
        <div
          className="card"
          onDragOver={e => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            textAlign: 'center', cursor: 'pointer', padding: '1.5rem',
            borderStyle: 'dashed',
            borderColor: file ? 'var(--green)' : 'var(--border)',
            color: file ? 'var(--green)' : 'var(--text-muted)',
          }}
        >
          {file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : 'Drop audio file or click to select'}
          <input ref={fileRef} type="file" accept="audio/*" style={{ display: 'none' }}
            onChange={e => { if (e.target.files?.[0]) setFile(e.target.files[0]) }} />
        </div>

        {/* Record */}
        <button onClick={handleRecord} style={{
          padding: '0.7rem', fontSize: '0.85rem', width: '100%',
          background: recording ? 'var(--red)' : 'var(--bg-input)',
          color: recording ? '#fff' : 'var(--text)',
          border: `1px solid ${recording ? 'var(--red)' : 'var(--border)'}`,
          animation: recording ? 'pulse 1.5s infinite' : 'none',
        }}>
          {recording ? `Stop (${duration}s)` : 'Record'}
        </button>

        {/* Send */}
        <button className="btn-primary" disabled={loading || (!file && !recording)}
          onClick={() => file && doTranscribe(file)}
          style={{ padding: '0.7rem', fontSize: '0.85rem', width: '100%', opacity: loading ? 0.6 : 1 }}>
          {loading ? 'Transcribing...' : 'Transcribe'}
        </button>

        <style>{`@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}50%{box-shadow:0 0 0 10px rgba(239,68,68,0)}}`}</style>
      </div>

      {/* Right: Result */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
        {error && (
          <div className="card" style={{ borderColor: 'var(--red)', color: 'var(--red)' }}>{error}</div>
        )}

        {result && (
          <>
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Transcription</span>
                <span style={{ fontSize: '0.72rem', color: 'var(--green)', fontFamily: 'monospace' }}>{elapsed}ms</span>
              </div>
              <div style={{ fontSize: '1rem', lineHeight: 1.6, color: '#fff', minHeight: 40 }}>
                {result.text || <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>(empty)</span>}
              </div>
            </div>
            {result.timings && <TimingBar timings={result.timings} />}
          </>
        )}

        {!result && !error && (
          <div className="card" style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '3rem' }}>
            Upload or record audio, then click Transcribe.
          </div>
        )}
      </div>
    </div>
  )
}
