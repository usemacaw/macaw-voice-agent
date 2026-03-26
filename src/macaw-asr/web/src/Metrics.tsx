import { useState, useEffect, useRef } from 'react'
import { getMetrics, type MetricsData } from './api'

export default function Metrics() {
  const [data, setData] = useState<MetricsData | null>(null)
  const [auto, setAuto] = useState(false)
  const timerRef = useRef<number>(0)

  const load = async () => {
    try { setData(await getMetrics()) } catch {}
  }

  useEffect(() => {
    load()
    return () => clearInterval(timerRef.current)
  }, [])

  useEffect(() => {
    clearInterval(timerRef.current)
    if (auto) timerRef.current = window.setInterval(load, 2000)
    return () => clearInterval(timerRef.current)
  }, [auto])

  if (!data) return <div className="card" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>Loading metrics...</div>

  const { models_loaded, requests, system } = data

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      {/* Controls */}
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
        <button className="btn-primary" onClick={load}>Refresh</button>
        <button className={auto ? 'btn-danger' : 'btn-ghost'} onClick={() => setAuto(!auto)}>
          {auto ? 'Stop Auto' : 'Auto (2s)'}
        </button>
        <div style={{ flex: 1 }} />
        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          Uptime: {data.uptime_s}s | Total requests: {requests.total}
          {system.gpu_names && ` | GPU: ${system.gpu_names.join(', ')}`}
        </span>
      </div>

      {/* Loaded models */}
      <div className="card">
        <h3 style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.6rem' }}>
          Loaded Models
        </h3>
        {models_loaded.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No models loaded yet. Send a request to load one.</div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '0.5rem' }}>
            {models_loaded.map(m => (
              <div key={m.model_id} style={{ background: 'var(--bg)', padding: '0.6rem', borderRadius: 6 }}>
                <div style={{ fontWeight: 600, fontSize: '0.82rem', color: '#fff' }}>{m.model_name}</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                  {m.device}{m.replicas > 1 ? ` x${m.replicas}` : ''} | {m.request_count} requests
                </div>
                {m.startup_ms && (
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                    startup: {m.startup_ms.total_ms?.toFixed(0)}ms (load: {m.startup_ms.load_ms?.toFixed(0)}ms)
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Per-model latency */}
      {Object.entries(requests.by_model).map(([name, stats]) => (
        <div key={name} className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.6rem' }}>
            <h3 style={{ fontSize: '0.78rem', fontWeight: 600, color: '#fff' }}>{name}</h3>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              {stats.count} requests (window: {stats.window})
            </span>
          </div>
          <table style={{ width: '100%', fontSize: '0.75rem', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', textAlign: 'right' }}>
                <th style={{ textAlign: 'left', fontWeight: 400, padding: '0.3rem 0' }}>Stage</th>
                <th style={{ fontWeight: 400, padding: '0.3rem 0.5rem' }}>avg</th>
                <th style={{ fontWeight: 400, padding: '0.3rem 0.5rem' }}>p50</th>
                <th style={{ fontWeight: 400, padding: '0.3rem 0.5rem' }}>p95</th>
                <th style={{ fontWeight: 400, padding: '0.3rem 0.5rem' }}>p99</th>
                <th style={{ fontWeight: 400, padding: '0.3rem 0.5rem' }}>min</th>
                <th style={{ fontWeight: 400, padding: '0.3rem 0.5rem' }}>max</th>
                <th style={{ fontWeight: 400, width: 120 }}>distribution</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(stats.latency)
                .filter(([, v]) => v.avg > 0)
                .sort(([, a], [, b]) => b.avg - a.avg)
                .map(([key, v]) => {
                  const color = v.avg > 500 ? 'var(--red)' : v.avg > 100 ? 'var(--yellow)' : 'var(--green)'
                  const maxAll = Math.max(...Object.values(stats.latency).map(l => l.max), 1)
                  return (
                    <tr key={key} style={{ borderTop: '1px solid var(--border)' }}>
                      <td style={{ color: 'var(--text-dim)', padding: '0.35rem 0' }}>{key.replace('_ms', '')}</td>
                      <td style={{ textAlign: 'right', color, fontFamily: 'monospace', padding: '0.35rem 0.5rem' }}>{v.avg}</td>
                      <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-dim)', padding: '0.35rem 0.5rem' }}>{v.p50}</td>
                      <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-dim)', padding: '0.35rem 0.5rem' }}>{v.p95}</td>
                      <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-dim)', padding: '0.35rem 0.5rem' }}>{v.p99}</td>
                      <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-muted)', padding: '0.35rem 0.5rem' }}>{v.min}</td>
                      <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-muted)', padding: '0.35rem 0.5rem' }}>{v.max}</td>
                      <td style={{ padding: '0.35rem 0' }}>
                        <div style={{ height: 14, background: 'var(--bg)', borderRadius: 3, overflow: 'hidden' }}>
                          <div style={{
                            width: `${Math.max((v.avg / maxAll) * 100, 2)}%`,
                            height: '100%', background: color, borderRadius: 3,
                          }} />
                        </div>
                      </td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
      ))}

      {Object.keys(requests.by_model).length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
          No request data yet. Transcribe some audio first.
        </div>
      )}
    </div>
  )
}
