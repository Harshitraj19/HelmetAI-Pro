import { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, Image, AlertCircle, CheckCircle2, X, FileText, Bike, Sun, User } from 'lucide-react';
import toast from 'react-hot-toast';
import { API, CHECK_MODES, VIOLATION_LABEL, SEVERITY_COLOR } from '../lib/api';

export default function ImageDetect() {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [mode, setMode] = useState('auto');
  const fileRef = useRef();
  const lastFile = useRef(null);
  const urls = useRef([]);              // blob URLs owned by this page
  const own = (u) => { urls.current.push(u); return u; };
  const freeAll = () => { urls.current.forEach((u) => URL.revokeObjectURL(u)); urls.current = []; };
  useEffect(() => freeAll, []);

  const handleFile = useCallback(async (file, checkMode) => {
    if (!file || !file.type.startsWith('image/')) { toast.error('Please upload a valid image file.'); return; }
    freeAll();
    lastFile.current = file;
    setPreview(own(URL.createObjectURL(file))); setResult(null); setLoading(true);
    try {
      const form = new FormData(); form.append('file', file);
      const res = await fetch(`${API}/api/detect/image?mode=${checkMode}`, { method: 'POST', body: form });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || `Server error ${res.status}`);
      let info = {};
      try { info = JSON.parse(res.headers.get('X-Result') || '{}'); } catch { /* header missing / cut by a proxy */ }
      const imageUrl = own(URL.createObjectURL(await res.blob()));
      setResult({ imageUrl, ...info });
      const site = info.mode === 'site';
      const vs = info.vehicles || [];
      const noHelmet = vs.reduce((a, v) => a + v.no_helmet, 0);
      const bad = vs.filter((v) => v.violations.length).length;
      if (!vs.length) toast(site ? 'No person found in this image' : 'No motorcycle found in this image', { icon: 'ℹ️' });
      else if (site) noHelmet ? toast.error(`⚠️ ${noHelmet} person(s) without a helmet`) : toast.success('✅ Everyone is wearing a helmet');
      else bad ? toast.error(`⚠️ ${bad} vehicle(s) violating`) : toast.success('✅ No violation found');
    } catch (err) {
      toast.error(err.message || 'Detection failed. Is the backend running?');
    } finally { setLoading(false); }
  }, []);

  const clear = () => { freeAll(); lastFile.current = null; setResult(null); setPreview(null); if (fileRef.current) fileRef.current.value = ''; };
  const changeMode = (m) => { setMode(m); if (lastFile.current && !loading) handleFile(lastFile.current, m); };
  const site = result?.mode === 'site';

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Image Detection</h1>
        <p className="page-subtitle">Adaptive enhancement → detection → rider association → plate OCR → evidence</p>
      </div>
      <div className="page-body">
        <div className="grid-2" style={{ alignItems: 'start' }}>
          <div>
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title"><Upload size={16} /> Upload image</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                {Object.entries(CHECK_MODES).map(([k, [label, hint]]) => (
                  <button key={k} className={`chip ${mode === k ? 'active' : ''}`} title={hint} onClick={() => changeMode(k)}>{label}</button>
                ))}
              </div>
              {!preview ? (
                <div className={`upload-zone${dragging ? ' dragging' : ''}`} onClick={() => fileRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
                  onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0], mode); }}>
                  <div className="upload-icon">📷</div>
                  <div className="upload-title">Drop an image here or click to browse</div>
                  <div className="upload-hint">JPG, PNG, WEBP · road scenes, or any photo of people in Helmet check mode</div>
                </div>
              ) : (
                <div style={{ position: 'relative' }}>
                  <img src={preview} alt="Preview" className="result-image" />
                  <button className="btn btn-secondary btn-sm" style={{ position: 'absolute', top: 8, right: 8 }} onClick={clear}><X size={14} /> Clear</button>
                </div>
              )}
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => { handleFile(e.target.files[0], mode); e.target.value = ''; }} />
            </div>
            {preview && (
              <button className="btn btn-primary" style={{ width: '100%' }} disabled={loading} onClick={() => fileRef.current?.click()}>
                {loading ? <span className="loading-text">Analysing…</span> : <><Upload size={16} /> Upload another</>}
              </button>
            )}
            {result?.scene && (
              <div className="card" style={{ marginTop: 16 }}>
                <div className="card-title"><Sun size={16} /> Scene analysis (DIP)</div>
                <KV k="Detected scene" v={result.scene.scene} />
                <KV k="Enhancement applied" v={result.scene.ops?.length ? result.scene.ops.join(' → ') : 'none needed'} />
                <KV k="Brightness / contrast" v={`${result.scene.brightness} / ${result.scene.contrast}`} />
                <KV k="Sharpness (Laplacian var.)" v={result.scene.sharpness} />
                <KV k="Latency" v={`${result.latency_ms} ms`} />
              </div>
            )}
          </div>

          <div>
            {loading && (
              <div className="card" style={{ textAlign: 'center', padding: '48px 24px' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: 12 }}>🔍</div>
                <div className="loading-text" style={{ fontSize: '1rem', fontWeight: 600 }}>Running the pipeline…</div>
              </div>
            )}
            {result && (
              <>
                <div className="card" style={{ marginBottom: 16 }}>
                  <div className="card-title"><Image size={16} /> Annotated result <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>bystanders auto-blurred</span></div>
                  <img src={result.imageUrl} alt="Detection result" className="result-image" />
                </div>
                <div className="card">
                  <div className="card-title">
                    {result.vehicles?.some((v) => v.violations.length)
                      ? <><AlertCircle size={16} color="#ef4444" /> {site ? 'People' : 'Vehicles'} — violations found</>
                      : <><CheckCircle2 size={16} color="#10b981" /> {site ? 'People' : 'Vehicles'} — all clear</>}
                    <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>{site ? 'helmet check' : 'traffic rules'}</span>
                  </div>
                  {(!result.vehicles || result.vehicles.length === 0) && (
                    <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                      {site ? 'No person detected.' : 'No motorcycle detected — try the Helmet check mode for photos without a bike.'} {result.detections?.length ? `${result.detections.length} other object(s) found.` : ''}
                    </div>
                  )}
                  <div className="detection-list">
                    {result.vehicles?.map((v) => {
                      const rec = result.violations?.find((r) => r.track_id === v.track_id);
                      return (
                        <div key={v.track_id} className={`detection-item ${v.violations.length ? 'violation' : 'safe'}`}>
                          <div>
                            <div className="detection-label">
                              {site
                                ? <><User size={13} style={{ display: 'inline', marginRight: 4 }} />Person #{v.track_id} · {v.helmets ? 'wearing a helmet' : 'no helmet'}</>
                                : <><Bike size={13} style={{ display: 'inline', marginRight: 4 }} />Motorcycle #{v.track_id} · {v.riders} rider(s), {v.helmets} helmet(s)</>}
                            </div>
                            <div className="detection-time">
                              {v.violations.length ? v.violations.map((x) => VIOLATION_LABEL[x] || x).join(' + ') : 'Compliant'}
                              {rec?.plate ? ` · plate ${rec.plate}` : ''}
                            </div>
                          </div>
                          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                            {v.violations.length
                              ? <span className="badge" style={{ background: (SEVERITY_COLOR[rec?.severity_level] || '#ef4444') + '22', color: SEVERITY_COLOR[rec?.severity_level] || '#ef4444' }}>severity {rec?.severity ?? '–'}</span>
                              : <span className="badge badge-success">Safe</span>}
                            {rec?.id && <a className="btn btn-secondary btn-sm" href={`${API}/api/violations/${rec.id}/challan.pdf`} target="_blank" rel="noreferrer"><FileText size={12} /> e-Challan</a>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            )}
            {!result && !loading && (
              <div className="card" style={{ textAlign: 'center', padding: '60px 24px' }}>
                <div style={{ fontSize: '3rem', marginBottom: 12, opacity: 0.3 }}>🛡️</div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Upload an image to see results here</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export const KV = ({ k, v }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', borderBottom: '1px solid var(--border)', padding: '8px 0' }}>
    <span style={{ color: 'var(--text-muted)' }}>{k}</span><span style={{ fontWeight: 500, textAlign: 'right' }}>{String(v)}</span>
  </div>
);
