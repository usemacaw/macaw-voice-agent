/** Visual timing breakdown bar chart. */
export default function TimingBar({ timings }: { timings: Record<string, number> }) {
  const keys = ['preprocess_ms', 'prepare_ms', 'prefill_ms', 'decode_ms', 'inference_ms', 'e2e_ms']
  const present = keys.filter(k => timings[k] !== undefined && timings[k] > 0)
  if (!present.length) return null

  const maxVal = Math.max(...present.map(k => timings[k]), 1)

  const color = (v: number) => v > 500 ? 'var(--red)' : v > 100 ? 'var(--yellow)' : 'var(--accent)'

  return (
    <div className="card" style={{ marginTop: '0.6rem' }}>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        Timing Breakdown
      </div>
      {present.map(k => {
        const v = timings[k]
        const pct = Math.max((v / maxVal) * 100, 2)
        return (
          <div key={k} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.3rem' }}>
            <span style={{ width: 120, fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'right', flexShrink: 0 }}>
              {k.replace('_ms', '')}
            </span>
            <div style={{ flex: 1, height: 16, background: 'var(--bg)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%', background: color(v),
                borderRadius: 3, transition: 'width 0.3s',
              }} />
            </div>
            <span style={{
              width: 65, fontSize: '0.72rem', fontFamily: 'monospace', textAlign: 'right', flexShrink: 0,
              color: v > 500 ? 'var(--red)' : v > 100 ? 'var(--yellow)' : 'var(--green)',
            }}>
              {v.toFixed(1)}ms
            </span>
          </div>
        )
      })}
    </div>
  )
}
