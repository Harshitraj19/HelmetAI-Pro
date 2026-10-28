import { useEffect, useRef, useState } from 'react';
import { Sparkles, Upload } from 'lucide-react';
import toast from 'react-hot-toast';
import { API } from '../lib/api';
import { KV } from './ImageDetect';

const MODES = [
  ['auto', 'Auto (scene-aware)'], ['night', 'Night: gamma + CLAHE'], ['fog', 'Fog: dark-channel dehaze'],
  ['low_contrast', 'Low contrast: CLAHE'], ['blurry', 'Blur: unsharp mask'], ['off', 'Off'],
];

export default function EnhanceLab() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState('auto');
  const [out, setOut] = useState(null);
  const [stats, setStats] = useState(null);
  const [busy, setBusy] = useState(false);
  const ref = useRef();
  const outRef = useRef(null);
  useEffect(() => () => { if (outRef.current) URL.revokeObjectURL(outRef.current); }, []);

  const run = async (f = file, m = mode) => {
    if (!f) return;
    const form = new FormData(); form.append('file', f);
    setBusy(true);
    try {
      const res = await fetch(`${API}/api/enhance/preview?mode=${m}`, { method: 'POST', body: form });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
      try { setStats(JSON.parse(res.headers.get('X-Result') || '{}')); } catch { setStats(null); }
      const url = URL.createObjectURL(await res.blob());
      if (outRef.current) URL.revokeObjectURL(outRef.current);
      outRef.current = url; setOut(url);
    } catch (e) { toast.error(e.message || 'Backend unreachable'); } finally { setBusy(false); }
  };

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Enhancement Lab</h1>
        <p className="page-subtitle">The Digital Image Processing core: each frame is analysed and routed to the right restoration operator before detection</p>
      </div>
      <div className="page-body">
        <div className="card" style={{ marginBottom: 16, display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <button className="btn btn-primary btn-sm" onClick={() => ref.current?.click()}><Upload size={13} /> {file ? file.name : 'Choose image'}</button>
          <input ref={ref} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files[0]; e.target.value = ''; if (!f) return; setFile(f); run(f, mode); }} />
          {MODES.map(([k, l]) => (
            <button key={k} className={`chip ${mode === k ? 'active' : ''}`} onClick={() => { setMode(k); run(file, k); }}>{l}</button>
          ))}
        </div>
        <div className="grid-3" style={{ alignItems: 'start' }}>
          <div className="card">
            <div className="card-title"><Sparkles size={16} /> Before · After {busy && <span className="loading-text" style={{ marginLeft: 'auto', fontSize: '0.75rem' }}>processing…</span>}</div>
            {out ? <img src={out} alt="before-after" className="result-image" /> :
              <div style={{ color: 'var(--text-muted)', padding: 40, textAlign: 'center' }}>Upload a night, foggy or blurry road image to compare.</div>}
          </div>
          <div className="card">
            <div className="card-title">Scene statistics</div>
            {stats ? <>
              <KV k="Scene" v={stats.scene} />
              <KV k="Operators" v={stats.ops?.length ? stats.ops.join(' → ') : 'none'} />
              <KV k="Mean luminance (L*)" v={stats.brightness} />
              <KV k="Contrast (σ L*)" v={stats.contrast} />
              <KV k="Dark-channel mean" v={stats.dark_channel} />
              <KV k="Sharpness (∇² var)" v={stats.sharpness} />
            </> : <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>–</div>}
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 14, lineHeight: 1.6 }}>
              Routing rules: L* &lt; 70 → night · dark channel &gt; 0.55 &amp; low σ → fog · σ &lt; 38 → low contrast · ∇² var &lt; 60 → blur.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
