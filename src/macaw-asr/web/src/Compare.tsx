import { useState, useRef } from 'react'
import { transcribe, type TranscribeResult, type ModelInfo } from './api'
import { useRecorder } from './useRecorder'
import TimingBar from './TimingBar'

interface CompareResult {
  model: string
  result: TranscribeResult | null
  error: string
  elapsed: number
  loading: boolean
}

export default function Compare({ models }: { models: ModelInfo[] }) {
  const [selected, setSelected] = useState<string[]>([])
  const [lang, setLang] = useState('pt')
  const [file, setFile] = useState<File | null>(null)
  const [results, setResults] = useState<CompareResult[]>([])
  const [running, setRunning] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const { recording, duration, start: startRec, stop: stopRec } = useRecorder()

  const toggle = (id: string) => {
    setSelected(prev => prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id])
  }

  const run = async (audioFile: File | Blob) => {
    if (!selected.length) return
    setRunning(true)
    const initial: CompareResult[] = selected.map(m => ({
      model: m, result: null, error: '', elapsed: 0, loading: true,
    }))
    setResults(initial)

    // Run all in parallel
    const promises = selected.map(async (model, idx) => {
      const t0 = performance.now()
      try {
        const r = await transcribe(audioFile, model, lang)
        const elapsed = Math.round(performance.now() - t0)
        setResults(prev => prev.map((p, i) => i === idx ? { ...p, result: r, elapsed, loading: false } : p))
      } catch (e: any) {
        setResults(prev => prev.map((p, i) => i === idx ? { ...p, error: e.message, loading: false } : p))
      }
    })

    await Promise.all(promises)
    setRunning(false)
  }

  const handleRecord = async () => {
    if (recording) {
      const wav = stopRec()
      if (wav) { setFile(wav); await run(wav) }
    } else {
      setResults([])
      startRec()
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%' }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
        <div className="card" style={{ flex: '0 0 auto', padding: '0.5rem 0.8rem' }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Select models to compare:</span>
        </div>
        {models.map(m => (
          <button key={m.id} onClick={() => toggle(m.id)} style={{
            padding: '0.35rem 0.7rem', fontSize: '0.75rem',
            background: selected.includes(m.id) ? 'var(--accent)' : 'var(--bg-input)',
            color: selected.includes(m.id) ? '#fff' : 'var(--text-dim)',
            border: `1px solid ${selected.includes(m.id) ? 'var(--accent)' : 'var(--border)'}`,
          }}>
            {m.id}
          </button>
        ))}
      </div>

      <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
        <input value={lang} onChange={e => setLang(e.target.value)} placeholder="Language" style={{ width: 80 }} />
        <div className="card" onClick={() => fileRef.current?.click()} style={{
          flex: 1, textAlign: 'center', cursor: 'pointer', padding: '0.5rem',
          borderStyle: 'dashed', fontSize: '0.78rem',
          borderColor: file ? 'var(--green)' : 'var(--border)',
          color: file ? 'var(--green)' : 'var(--text-muted)',
        }}>
          {file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : 'Drop or select audio'}
          <input ref={fileRef} type="file" accept="audio/*" style={{ display: 'none' }}
            onChange={e => { if (e.target.files?.[0]) setFile(e.target.files[0]) }} />
        </div>
        <button onClick={handleRecord} style={{
          padding: '0.5rem 1rem', background: recording ? 'var(--red)' : 'var(--bg-input)',
          color: recording ? '#fff' : 'var(--text)', border: `1px solid ${recording ? 'var(--red)' : 'var(--border)'}`,
        }}>
          {recording ? `Stop (${duration}s)` : 'Record'}
        </button>
        <button className="btn-primary" disabled={running || !selected.length || (!file && !recording)}
          onClick={() => file && run(file)} style={{ opacity: running ? 0.6 : 1 }}>
          {running ? 'Running...' : `Compare ${selected.length} models`}
        </button>
      </div>

      {/* Results grid */}
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(selected.length || 1, 3)}, 1fr)`, gap: '0.6rem', flex: 1, overflow: 'auto' }}>
        {results.map((r, i) => (
          <div key={r.model} className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#fff' }}>{r.model}</span>
              {r.loading ? (
                <span style={{ fontSize: '0.72rem', color: 'var(--yellow)' }}>loading...</span>
              ) : r.error ? (
                <span style={{ fontSize: '0.72rem', color: 'var(--red)' }}>error</span>
              ) : (
                <span style={{ fontSize: '0.72rem', color: 'var(--green)', fontFamily: 'monospace' }}>{r.elapsed}ms</span>
              )}
            </div>

            {r.error && <div style={{ fontSize: '0.75rem', color: 'var(--red)' }}>{r.error}</div>}

            {r.result && (
              <>
                <div style={{ fontSize: '0.85rem', lineHeight: 1.5, color: 'var(--text)', minHeight: 50 }}>
                  {r.result.text || <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>(empty)</span>}
                </div>
                {r.result.timings && <TimingBar timings={r.result.timings} />}
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
