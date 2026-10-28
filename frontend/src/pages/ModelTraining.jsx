import { useCallback, useEffect, useState } from 'react';
import { Brain, Play, RefreshCw, Server, BarChart3, Layers, Terminal } from 'lucide-react';
import toast from 'react-hot-toast';
import { getAdminToken, getJSON, postJSON, setAdminToken } from '../lib/api';

const STEPS = [
  ['Merge', 'Open Images (Siddhesh) + PPE construction sets → 5 unified classes, cleaned, 1,084 duplicates removed'],
  ['Pseudo-label R1', 'Teachers fill the missing classes: DIP helmet model → helmet/head, COCO → motorcycle'],
  ['Train R1', 'YOLO11 student on completed labels'],
  ['Pseudo-label R2', 'Student re-labels everything, adding licence plates'],
  ['Train final', 'Fine-tune from R1, export ONNX'],
  ['Evaluate', 'Per-class test AP + night/fog/blur robustness ablation'],
];

export default function ModelTraining() {
  const [status, setStatus] = useState({ status: 'idle', log: '' });
  const [model, setModel] = useState(null);
  const [profile, setProfile] = useState('cpu');
  const [token, setTokenState] = useState(getAdminToken);
  const setToken = (t) => { setTokenState(t); setAdminToken(t); };
  const [busy, setBusy] = useState(false);

  const poll = useCallback(() => getJSON('/api/train/status').then(setStatus).catch(() => {}), []);
  useEffect(() => {
    poll(); getJSON('/api/model').then(setModel).catch(() => {});
    const iv = setInterval(poll, 5000); return () => clearInterval(iv);
  }, [poll]);

  const hdr = token ? { 'X-Admin-Token': token } : {};
  const start = async () => {
    setBusy(true);
    try {
      const r = await getJSON('/api/train/start', { method: 'POST', headers: { 'Content-Type': 'application/json', ...hdr }, body: JSON.stringify({ profile }) });
      r.status === 'success' ? toast.success(r.message) : toast.error(r.message);
      poll();
    } catch (e) { toast.error(e.message); } finally { setBusy(false); }
  };
  const reload = async () => {
    try { const r = await getJSON('/api/model/reload', { method: 'POST', headers: hdr }); toast.success(`Loaded: ${r.loaded.join(', ')}`); getJSON('/api/model').then(setModel).catch(() => {}); }
    catch (e) { toast.error(e.message); }
  };
  const evaluate = async () => {
    setBusy(true);
    try { const r = await postJSON('/api/evaluate', {}, hdr); r.status === 'success' ? toast.success('Evaluation done') : toast.error(r.message?.slice(-200)); getJSON('/api/model').then(setModel); }
    catch (e) { toast.error(e.message); } finally { setBusy(false); }
  };

  const pc = model?.metrics?.per_class_eval?.per_class;
  const abl = model?.metrics?.per_class_eval?.ablation;

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Model & Training</h1>
        <p className="page-subtitle">Unified 5-class detector trained on every merged dataset with teacher-student label completion</p>
      </div>
      <div className="page-body">
        <div className="grid-2" style={{ alignItems: 'start' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card">
              <div className="card-title"><Layers size={16} /> Training recipe (training/run_all.sh)</div>
              {STEPS.map(([t, d], i) => (
                <div key={t} className="step"><span className="step-n">{i + 1}</span><div><b>{t}</b><div className="step-d">{d}</div></div></div>
              ))}
            </div>
            <div className="card">
              <div className="card-title"><Brain size={16} /> Run</div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                {['cpu', 'gpu', 'max'].map((p) => <button key={p} className={`chip ${profile === p ? 'active' : ''}`} onClick={() => setProfile(p)}>{p === 'cpu' ? 'CPU · YOLO11n 416px' : p === 'gpu' ? 'GPU · YOLO11s 640px' : 'GPU · YOLO11m (best)'}</button>)}
              </div>
              <input className="input" style={{ width: '100%', marginBottom: 10 }} placeholder="Admin token (if ADMIN_TOKEN is set)" value={token} onChange={(e) => setToken(e.target.value)} />
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn btn-primary" disabled={busy || status.status === 'running'} onClick={start}>
                  {status.status === 'running' ? <><RefreshCw size={15} className="spin" /> Training…</> : <><Play size={15} /> Start full training</>}
                </button>
                <button className="btn btn-secondary" onClick={reload}><RefreshCw size={15} /> Hot-reload model</button>
                <button className="btn btn-secondary" disabled={busy} onClick={evaluate}><BarChart3 size={15} /> Evaluate</button>
              </div>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 10 }}>For a full-strength model use the GPU notebook in training/ (Colab / Kaggle, ~1.5 h), then copy models/helmet_unified.onnx here and hot-reload.</div>
            </div>
            <div className="card">
              <div className="card-title"><Server size={16} /> Status: <span style={{ textTransform: 'capitalize', marginLeft: 6 }}>{status.status}</span></div>
              <pre className="log"><Terminal size={12} /> {status.log || 'No training log yet.'}</pre>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card">
              <div className="card-title"><BarChart3 size={16} /> Per-class test metrics</div>
              {!pc ? <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No evaluation yet — train a model or copy models/metrics.json.</div> : (
                <div className="table-wrap"><table>
                  <thead><tr><th>Class</th><th>Images</th><th>P</th><th>R</th><th>mAP50</th><th>mAP50-95</th></tr></thead>
                  <tbody>{Object.entries(pc).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td className="mono">{v.images}</td><td className="mono">{v.precision}</td><td className="mono">{v.recall}</td><td className="mono"><b>{v.mAP50}</b></td><td className="mono">{v.mAP50_95}</td></tr>
                  ))}</tbody>
                </table></div>
              )}
              {model?.metrics?.per_class_eval && <div style={{ marginTop: 10, fontSize: '0.85rem' }}>Mean mAP50 <b>{model.metrics.per_class_eval.mAP50}</b> · mAP50-95 <b>{model.metrics.per_class_eval.mAP50_95}</b></div>}
            </div>
            {abl && <div className="card">
              <div className="card-title">Robustness ablation (rider classes, mAP50)</div>
              <div className="table-wrap"><table>
                <thead><tr><th>Condition</th><th>Raw</th><th>+ adaptive enhancement</th></tr></thead>
                <tbody>{['night', 'fog', 'blur'].map((k) => <tr key={k}><td>{k}</td><td className="mono">{abl[k]}</td><td className="mono"><b>{abl[`${k}+enh`]}</b></td></tr>)}</tbody>
              </table></div>
            </div>}
            <div className="card">
              <div className="card-title">Loaded now</div>
              <div className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>{model?.loaded?.join(', ') || 'none'}</div>
              {model && <div style={{ fontSize: '0.78rem', marginTop: 8, color: model.plate_ocr_available ? '#10b981' : '#f59e0b' }}>
                Plate OCR: {model.plate_ocr_available ? 'Tesseract available' : 'Tesseract not installed - plates will not be read (install Tesseract-OCR or set TESSERACT_CMD)'}
              </div>}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
