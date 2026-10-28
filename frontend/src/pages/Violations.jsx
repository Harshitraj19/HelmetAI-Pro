import { useCallback, useEffect, useState } from 'react';
import { ShieldAlert, FileText, CheckCircle, XCircle, Link2, Search, RefreshCw, Hash } from 'lucide-react';
import toast from 'react-hot-toast';
import { API, getJSON, postJSON, evidenceUrl, VIOLATION_LABEL, SEVERITY_COLOR } from '../lib/api';
import { useLive } from '../lib/live';

export default function Violations() {
  const [rows, setRows] = useState([]);
  const [filters, setFilters] = useState({ type: '', plate: '', status: '', min_severity: 0 });
  const [selected, setSelected] = useState(null);
  const [verify, setVerify] = useState(null);
  const [verifying, setVerifying] = useState(false);
  const { alerts } = useLive();

  const load = useCallback(() => {
    const q = new URLSearchParams(Object.entries(filters).filter(([, v]) => v !== '' && v !== 0));
    getJSON(`/api/violations?limit=200&${q}`).then(setRows).catch((e) => toast.error(e.message, { id: 'viol-load' }));
  }, [filters]);
  // typing a plate: wait for a pause instead of one request per keystroke
  useEffect(() => { const t = setTimeout(load, 300); return () => clearTimeout(t); }, [load, alerts.length]);
  // keep the detail panel in sync with refreshed rows
  useEffect(() => { setSelected((sel) => (sel ? rows.find((r) => r.id === sel.id) || sel : sel)); }, [rows]);

  const runVerify = async () => {
    setVerifying(true);
    try {
      const r = await getJSON('/api/evidence/verify?full=true');      // re-hash every image, no cache
      setVerify(r);
      r.valid ? toast.success(`Chain intact (${r.checked} records)`) : toast.error(`Tampering detected at record #${r.broken_id}`);
    } catch (e) { toast.error(e.message); } finally { setVerifying(false); }
  };
  const review = async (id, action) => {
    try {
      const r = await postJSON(`/api/violations/${id}/review`, { action });
      setRows((rs) => rs.map((x) => (x.id === id ? r : x))); setSelected(r);
      toast.success(`Violation #${id} ${action}`);
    } catch (e) { toast.error(e.message); }
  };

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Violations & e-Challan</h1>
        <p className="page-subtitle">Evidence records with SHA-256 hash chain · human review · auto-generated challans</p>
      </div>
      <div className="page-body">
        <div className="card" style={{ marginBottom: 16, display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <Search size={16} color="var(--text-muted)" />
          <input className="input" style={{ width: 170, maxWidth: '100%' }} placeholder="Plate e.g. MH 02" value={filters.plate} onChange={(e) => setFilters({ ...filters, plate: e.target.value })} />
          <select className="input" value={filters.type} onChange={(e) => setFilters({ ...filters, type: e.target.value })}>
            <option value="">All types</option>
            {Object.entries(VIOLATION_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <select className="input" value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })}>
            <option value="">Any status</option><option value="pending">Pending review</option><option value="confirmed">Confirmed</option><option value="rejected">Rejected</option>
          </select>
          <select className="input" value={filters.min_severity} onChange={(e) => setFilters({ ...filters, min_severity: Number(e.target.value) })}>
            <option value={0}>Any severity</option><option value={25}>≥ medium</option><option value={50}>≥ high</option><option value={75}>critical</option>
          </select>
          <button className="btn btn-secondary btn-sm" onClick={load}><RefreshCw size={13} /> Refresh</button>
          <button className="btn btn-primary btn-sm" style={{ marginLeft: 'auto' }} disabled={verifying} onClick={runVerify}><Link2 size={13} /> {verifying ? 'Verifying…' : 'Verify evidence chain'}</button>
        </div>
        {verify && (
          <div className={`chain-box ${verify.valid ? 'good' : 'bad'}`} style={{ marginBottom: 16 }}>
            {verify.valid ? <><CheckCircle size={14} /> All {verify.checked} records verified. Chain head: <span className="mono">{verify.head?.slice(0, 24)}…</span></>
              : <><XCircle size={14} /> Record #{verify.broken_id}: {verify.reason}</>}
          </div>
        )}

        <div className="grid-2" style={{ alignItems: 'start', gridTemplateColumns: '1.1fr 1fr' }}>
          <div className="card">
            <div className="card-title"><ShieldAlert size={16} /> {rows.length} record(s)</div>
            <div className="evidence-grid">
              {rows.length === 0 && <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No records match.</div>}
              {rows.map((v) => (
                <button key={v.id} className={`evidence-card ${selected?.id === v.id ? 'active' : ''}`} onClick={() => setSelected(v)}>
                  {v.bike_path ? <img src={evidenceUrl(v.bike_path)} alt="" /> : <div className="noimg">no image</div>}
                  <div className="ev-body">
                    <div className="ev-top"><b>#{v.id}</b><span className="badge" style={{ background: SEVERITY_COLOR[v.severity_level] + '22', color: SEVERITY_COLOR[v.severity_level] }}>{v.severity}</span></div>
                    <div className="ev-type">{v.violation_list.map((x) => VIOLATION_LABEL[x] || x).join(' + ')}</div>
                    <div className="mono ev-plate">{v.plate || '— plate n/a —'}</div>
                    <div className="ev-status" data-s={v.status}>{v.status}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="card" style={{ position: 'sticky', top: 16 }}>
            {!selected ? <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', textAlign: 'center', padding: 40 }}>Select a record to see its evidence</div> : (
              <>
                <div className="card-title">Violation #{selected.id} <span className="ev-status" data-s={selected.status} style={{ marginLeft: 'auto' }}>{selected.status}</span></div>
                {selected.frame_path && <img src={evidenceUrl(selected.frame_path)} alt="frame" className="result-image" style={{ marginBottom: 10 }} />}
                <div style={{ display: 'flex', gap: 10, marginBottom: 12 }}>
                  {selected.bike_path && <img src={evidenceUrl(selected.bike_path)} alt="bike" style={{ height: 120, borderRadius: 8 }} />}
                  {selected.plate_path && <img src={evidenceUrl(selected.plate_path)} alt="plate" style={{ height: 48, borderRadius: 6, alignSelf: 'flex-end', imageRendering: 'pixelated' }} />}
                </div>
                <Row k="Offence" v={selected.violation_list.map((x) => VIOLATION_LABEL[x] || x).join(' + ')} />
                <Row k="Plate" v={selected.plate ? `${selected.plate} (${Math.round((selected.plate_confidence || 0) * 100)}% conf.)` : 'not read'} />
                {selected.plate_reads?.length > 0 && <Row k="Per-frame OCR reads" v={selected.plate_reads.join(' · ')} mono />}
                <Row k="Riders / helmets" v={`${selected.riders} / ${selected.helmets}`} />
                <Row k="Severity" v={`${selected.severity} (${selected.severity_level})`} />
                <Row k="Scene" v={selected.scene} />
                <Row k="Camera / source" v={`${selected.camera_id} · ${selected.source}`} />
                <Row k="Time (UTC)" v={selected.created_at?.slice(0, 19).replace('T', ' ')} />
                <div style={{ marginTop: 12, fontSize: '0.72rem', color: 'var(--text-muted)' }}><Hash size={11} style={{ display: 'inline' }} /> record <span className="mono">{selected.record_hash?.slice(0, 32)}…</span><br />chain <span className="mono">{selected.chain_hash?.slice(0, 32)}…</span></div>
                <div style={{ display: 'flex', gap: 8, marginTop: 16, flexWrap: 'wrap' }}>
                  <a className="btn btn-primary btn-sm" href={`${API}/api/violations/${selected.id}/challan.pdf`} target="_blank" rel="noreferrer"><FileText size={13} /> e-Challan PDF</a>
                  <button className="btn btn-secondary btn-sm" disabled={selected.status === 'confirmed'} onClick={() => review(selected.id, 'confirmed')}><CheckCircle size={13} /> Confirm</button>
                  <button className="btn btn-danger btn-sm" disabled={selected.status === 'rejected'} onClick={() => review(selected.id, 'rejected')}><XCircle size={13} /> Reject (false positive)</button>
                  {selected.status !== 'pending' && <button className="btn btn-secondary btn-sm" onClick={() => review(selected.id, 'pending')}><RefreshCw size={13} /> Undo</button>}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

const Row = ({ k, v, mono }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, fontSize: '0.8rem', borderBottom: '1px solid var(--border)', padding: '7px 0' }}>
    <span style={{ color: 'var(--text-muted)' }}>{k}</span><span className={mono ? 'mono' : ''} style={{ textAlign: 'right' }}>{v ?? '–'}</span>
  </div>
);
