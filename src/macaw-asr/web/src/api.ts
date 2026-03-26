const BASE = ''  // Uses Vite proxy

export interface TranscribeResult {
  text: string
  usage?: { type: string; seconds: number }
  timings?: Record<string, number>
}

export interface ModelInfo {
  id: string
  family: string
  parameters: string
  model_id: string
}

export interface MetricsData {
  uptime_s: number
  models_loaded: Array<{
    model_id: string
    model_name: string
    device: string
    replicas: number
    request_count: number
    startup_ms: Record<string, number>
  }>
  requests: {
    total: number
    by_model: Record<string, {
      count: number
      window: number
      latency: Record<string, { avg: number; min: number; max: number; p50: number; p95: number; p99: number }>
    }>
  }
  system: {
    version: string
    gpu_count?: number
    gpu_names?: string[]
  }
}

export async function transcribe(file: File | Blob, model: string, language: string, format = 'json'): Promise<TranscribeResult> {
  const fd = new FormData()
  fd.append('file', file, file instanceof File ? file.name : 'audio.wav')
  fd.append('model', model)
  fd.append('language', language)
  fd.append('response_format', format)
  const r = await fetch(`${BASE}/v1/audio/transcriptions`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`)
  return r.json()
}

export async function listModels(): Promise<ModelInfo[]> {
  const r = await fetch(`${BASE}/v1/models`)
  const d = await r.json()
  return d.data || []
}

export async function getMetrics(): Promise<MetricsData> {
  const r = await fetch(`${BASE}/api/metrics`)
  return r.json()
}

export async function getHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE}/`)
    return r.ok
  } catch { return false }
}

export async function pullModel(model: string): Promise<{ status: string }> {
  const r = await fetch(`${BASE}/api/pull`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, stream: false }),
  })
  return r.json()
}

export async function deleteModel(model: string): Promise<{ status: string }> {
  const r = await fetch(`${BASE}/api/delete`, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  })
  return r.json()
}

export async function getRunning(): Promise<any> {
  const r = await fetch(`${BASE}/api/ps`)
  return r.json()
}
