import { useEffect, useState } from 'react';
import { Settings as Cog, Save } from 'lucide-react';
import toast from 'react-hot-toast';
import { getJSON, putJSON } from '../lib/api';

const TOGGLES = [
  ['privacy_blur', 'Privacy blur', 'Blur faces & plates of everyone except violators (DPDP-Act-style data minimisation)'],
  ['plate_ocr', 'Plate OCR', 'Multi-frame number-plate recognition for violators'],
  ['save_evidence', 'Save evidence', 'Write hash-chained evidence records + images'],
  ['show_enhanced', 'Show enhanced frame', 'Display the DIP-enhanced frame instead of the raw one'],
];

export default function Settings() {
  const [s, setS] = useState(null);
  const [model, setModel] = useState(null);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    getJSON('/api/settings').then(setS).catch((e) => toast.error(e.message));
    getJSON('/api/model').then(setModel).catch(() => {});
  }, []);
  const save = async () => {
    setSaving(true);
    try { setS(await putJSON('/api/settings', s)); toast.success('Settings applied live'); }
    catch (e) { toast.error(e.message); } finally { setSaving(false); }
  };
  if (!s) return <div className="page-body" style={{ paddingTop: 28 }}><span className="loading-text">Loading settings…</span></div>;

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-subtitle">Runtime configuration — applied immediately to new frames, no restart needed</p>
      </div>
      <div className="page-body">
        <div className="grid-2" style={{ alignItems: 'start' }}>
          <div className="card">
            <div className="card-title"><Cog size={16} /> Pipeline</div>
            <Field label="Operating mode" hint="Traffic: motorcycle rules · Site: workplace PPE (every head needs a helmet/hard-hat)">
              <select className="input" value={s.mode} onChange={(e) => setS({ ...s, mode: e.target.value })}>
                <option value="traffic">Traffic enforcement</option><option value="site">Construction / workplace</option>
              </select>
            </Field>
            <Field label="Enhancement" hint="auto = scene-aware routing; force a mode for testing">
              <select className="input" value={s.enhancement} onChange={(e) => setS({ ...s, enhancement: e.target.value })}>
                {['auto', 'off', 'night', 'fog', 'low_contrast', 'blurry'].map((m) => <option key={m}>{m}</option>)}
              </select>
            </Field>
            <Field label="Camera ID" hint="Printed on evidence and challans"><input className="input" value={s.camera_id} onChange={(e) => setS({ ...s, camera_id: e.target.value })} /></Field>
            <Field label={`Temporal smoothing window: ${s.smoothing_window} frames`} hint="Majority vote per tracked vehicle">
              <input type="range" min={1} max={15} value={s.smoothing_window} onChange={(e) => setS({ ...s, smoothing_window: Number(e.target.value) })} style={{ width: '100%' }} />
            </Field>
            <Field label={`Min. frames before a vehicle counts: ${s.min_track_hits}`} hint="Suppresses one-frame false alarms">
              <input type="range" min={1} max={10} value={s.min_track_hits} onChange={(e) => setS({ ...s, min_track_hits: Number(e.target.value) })} style={{ width: '100%' }} />
            </Field>
            {TOGGLES.map(([k, l, h]) => (
              <label key={k} className="toggle-row">
                <div><div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{l}</div><div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{h}</div></div>
                <input type="checkbox" checked={!!s[k]} onChange={(e) => setS({ ...s, [k]: e.target.checked })} />
              </label>
            ))}
            <button className="btn btn-primary" style={{ marginTop: 16, width: '100%' }} disabled={saving} onClick={save}><Save size={15} /> {saving ? 'Applying…' : 'Apply'}</button>
          </div>
          <div className="card">
            <div className="card-title">Loaded models</div>
            <pre className="mono" style={{ whiteSpace: 'pre-wrap', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{model ? JSON.stringify({ loaded: model.loaded, img_size: model.img_size, class_conf: model.class_conf }, null, 2) : '…'}</pre>
          </div>
        </div>
      </div>
    </>
  );
}

const Field = ({ label, hint, children }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 6 }}>{label}</div>
    {children}
    {hint && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 4 }}>{hint}</div>}
  </div>
);
