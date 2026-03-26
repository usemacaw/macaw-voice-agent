import { useState, useEffect } from 'react'
import { listModels, getHealth, type ModelInfo } from './api'
import Transcribe from './Transcribe'
import Compare from './Compare'
import Metrics from './Metrics'
import Models from './Models'

type Tab = 'transcribe' | 'compare' | 'metrics' | 'models'

export default function App() {
  const [tab, setTab] = useState<Tab>('transcribe')
  const [models, setModels] = useState<ModelInfo[]>([])
  const [online, setOnline] = useState(false)

  useEffect(() => {
    const check = async () => {
      const ok = await getHealth()
      setOnline(ok)
      if (ok) {
        const m = await listModels()
        setModels(m)
      }
    }
    check()
    const id = setInterval(check, 10000)
    return () => clearInterval(id)
  }, [])

  const tabs: { id: Tab; label: string }[] = [
    { id: 'transcribe', label: 'Transcribe' },
    { id: 'compare', label: 'Compare' },
    { id: 'metrics', label: 'Metrics' },
    { id: 'models', label: 'Models' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Header */}
      <header style={{
        padding: '0.6rem 1.2rem', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', gap: '1rem',
      }}>
        <h1 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#fff', letterSpacing: '-0.02em' }}>
          macaw-asr
        </h1>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: online ? 'var(--green)' : 'var(--red)',
        }} />
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {online ? `${models.length} models` : 'disconnected'}
        </span>
        <div style={{ flex: 1 }} />
        <nav style={{ display: 'flex', gap: '2px' }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              padding: '0.4rem 0.9rem', fontSize: '0.78rem',
              background: tab === t.id ? 'var(--accent)' : 'transparent',
              color: tab === t.id ? '#fff' : 'var(--text-dim)',
              border: tab === t.id ? 'none' : '1px solid var(--border)',
              borderRadius: 6,
            }}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Content */}
      <main style={{ flex: 1, overflow: 'auto', padding: '1rem 1.2rem' }}>
        {tab === 'transcribe' && <Transcribe models={models} />}
        {tab === 'compare' && <Compare models={models} />}
        {tab === 'metrics' && <Metrics />}
        {tab === 'models' && <Models models={models} onRefresh={async () => setModels(await listModels())} />}
      </main>
    </div>
  )
}
