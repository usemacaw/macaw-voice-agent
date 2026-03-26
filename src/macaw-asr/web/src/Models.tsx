import { useState } from 'react'
import { pullModel, deleteModel, getRunning, type ModelInfo } from './api'

export default function Models({ models, onRefresh }: { models: ModelInfo[]; onRefresh: () => void }) {
  const [pullId, setPullId] = useState('')
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState<any>(null)

  const doPull = async () => {
    if (!pullId.trim()) return
    setLoading(true)
    setStatus('')
    try {
      const r = await pullModel(pullId.trim())
      setStatus(`Pull: ${r.status}`)
      onRefresh()
    } catch (e: any) { setStatus(`Error: ${e.message}`) }
    finally { setLoading(false) }
  }

  const doDelete = async (model: string) => {
    if (!confirm(`Remove ${model}?`)) return
    try {
      await deleteModel(model)
      setStatus(`Removed: ${model}`)
      onRefresh()
    } catch (e: any) { setStatus(`Error: ${e.message}`) }
  }

  const loadRunning = async () => {
    try { setRunning(await getRunning()) } catch {}
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      {/* Pull */}
      <div className="card">
        <h3 style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
          Pull Model
        </h3>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <input value={pullId} onChange={e => setPullId(e.target.value)}
            placeholder="Model name (e.g. faster-whisper-small)" style={{ flex: 1 }} />
          <button className="btn-primary" onClick={doPull} disabled={loading}>
            {loading ? 'Pulling...' : 'Pull'}
          </button>
        </div>
        {status && <div style={{ fontSize: '0.75rem', marginTop: '0.4rem', color: status.startsWith('Error') ? 'var(--red)' : 'var(--green)' }}>{status}</div>}
      </div>

      {/* Running */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <h3 style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Running Models (GPU)</h3>
          <button className="btn-ghost" onClick={loadRunning} style={{ padding: '0.25rem 0.6rem', fontSize: '0.7rem' }}>Refresh</button>
        </div>
        {running ? (
          running.models?.length ? (
            running.models.map((m: any) => (
              <div key={m.model} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0', borderBottom: '1px solid var(--border)', fontSize: '0.8rem' }}>
                <span style={{ color: '#fff' }}>{m.name}</span>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{m.model}</span>
              </div>
            ))
          ) : <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No models loaded</div>
        ) : <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>Click Refresh to check</div>}
      </div>

      {/* Registry */}
      <div className="card">
        <h3 style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
          Available Models ({models.length})
        </h3>
        <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
              <th style={{ fontWeight: 400, padding: '0.3rem 0' }}>ID</th>
              <th style={{ fontWeight: 400 }}>Family</th>
              <th style={{ fontWeight: 400 }}>Params</th>
              <th style={{ fontWeight: 400 }}>HuggingFace ID</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {models.map(m => (
              <tr key={m.id} style={{ borderTop: '1px solid var(--border)' }}>
                <td style={{ padding: '0.4rem 0', color: '#fff', fontWeight: 500 }}>{m.id}</td>
                <td style={{ color: 'var(--text-dim)' }}>{m.family}</td>
                <td style={{ color: 'var(--text-dim)', fontFamily: 'monospace' }}>{m.parameters}</td>
                <td style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{m.model_id}</td>
                <td>
                  <button className="btn-ghost" onClick={() => doDelete(m.id)}
                    style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem', color: 'var(--red)' }}>
                    remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
