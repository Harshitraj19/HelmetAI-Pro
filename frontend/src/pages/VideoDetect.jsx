import { useState, useRef, useEffect, useCallback } from 'react';
import { Film, Upload, X, Play, BarChart3, Download, AlertTriangle, CheckCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { API, CHECK_MODES, newRunId, readMjpeg } from '../lib/api';
import { useLive } from '../lib/live';
import { KV } from './ImageDetect';

const EXTS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv'];

export default function VideoDetect() {
  const [file, setFile] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const [running, setRunning] = useState(false);
  const [live, setLive] = useState(null);
  const [summary, setSummary] = useState(null);
  const [mode, setMode] = useState('traffic');
  const [stride, setStride] = useState(1);
  const [phase, setPhase] = useState('idle');     // idle | uploading | processing
  const fileRef = useRef();
  const abortRef = useRef(null);
  const runRef = useRef(null);         // id of the run this page started
  const { subscribe } = useLive();

  useEffect(() => subscribe((m) => {
    if (m.run_id && m.run_id !== runRef.current) return;     // someone else's video
    if (m.type === 'video_upload_score') setLive(m);
    if (m.type === 'video_upload_done') {
      // FIX: the base read total_helmets while the server sent max_helmets -> summary was always undefined.
      setSummary(m); setRunning(false);
      if (m.error) toast.error(`Processing stopped: ${m.error}`);
      else toast.success(`Video processed: ${m.counters.frames} frames, ${m.counters.unique_motorcycles} motorcycles`);
    }
  }), [subscribe]);

  // keep only one frame blob alive; free the last one when leaving the page
  const frameRef = useRef(null);
  const showFrame = useCallback((url) => {
    if (frameRef.current) URL.revokeObjectURL(frameRef.current);
    frameRef.current = url; setFrameUrl(url);
  }, []);
  useEffect(() => () => {
    abortRef.current?.abort();
    if (frameRef.current) URL.revokeObjectURL(frameRef.current);
  }, []);

  const pick = useCallback((f) => {
    if (!f) return;
    if (!EXTS.includes(f.name.split('.').pop().toLowerCase())) { toast.error('Unsupported video format'); return; }
    setFile(f); setSummary(null); setLive(null); showFrame(null);
  }, [showFrame]);

  const start = async () => {
    if (!file) return;
    setRunning(true); setSummary(null); setLive(null); setPhase('uploading');
    const ctrl = new AbortController(); abortRef.current = ctrl;
    const runId = newRunId(); runRef.current = runId;
    try {
      const form = new FormData(); form.append('file', file);
      const res = await fetch(`${API}/api/detect/video?run_id=${runId}&mode=${mode}&stride=${stride}`, { method: 'POST', body: form, signal: ctrl.signal });
      setPhase('processing');
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || body.detail || `Server error ${res.status}`);
      }
      const n = await readMjpeg(res, showFrame, ctrl.signal);
      if (!n && !ctrl.signal.aborted) toast.error('No frames could be decoded from this video.');
    } catch (e) {
      if (e.name !== 'AbortError') toast.error(e.message || 'Upload failed. Is the backend running?');
    } finally {
      // the stream is over: don't rely on the WebSocket 'done' message alone (it may be missed)
      if (runRef.current === runId) { setRunning(false); setPhase('idle'); }
    }
  };
  const stop = () => { abortRef.current?.abort(); setRunning(false); setPhase('idle'); };
  const c = summary?.counters || live?.counters;

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Video Detection</h1>
        <p className="page-subtitle">Tracked, temporally-smoothed analysis — every rider counted once, one evidence record per violating vehicle</p>
      </div>
      <div className="page-body">
        <div className="grid-3" style={{ alignItems: 'start' }}>
          <div>
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="card-title"><Film size={16} /> {phase === 'uploading' ? 'Uploading…' : running ? 'Processing' : frameUrl ? 'Last processed frame' : 'Upload a video'}</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
                <select className="input" value={mode} disabled={running} onChange={(e) => setMode(e.target.value)} title={CHECK_MODES[mode][1]}>
                  <option value="traffic">Traffic (motorcycle riders)</option>
                  <option value="site">Helmet check (every person)</option>
                </select>
                <select className="input" value={stride} disabled={running} onChange={(e) => setStride(Number(e.target.value))}
                  title="Analysing fewer frames is faster; tracking still works at 2-3x">
                  <option value={1}>Every frame (most accurate)</option>
                  <option value={2}>Every 2nd frame (2x faster)</option>
                  <option value={3}>Every 3rd frame (3x faster)</option>
                </select>
              </div>
              {frameUrl ? <img src={frameUrl} alt="frame" className="result-image" /> : (
                <div className="upload-zone" onClick={() => fileRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); pick(e.dataTransfer.files[0]); }}>
                  <div className="upload-icon">🎬</div>
                  <div className="upload-title">{file ? file.name : 'Drop a video or click to browse'}</div>
                  <div className="upload-hint">{EXTS.join(', ').toUpperCase()} · processed frame-by-frame on the server</div>
                </div>
              )}
              <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => pick(e.target.files[0])} />
            </div>
            <div style={{ display: 'flex', gap: 10 }}>
              {!running
                ? <button className="btn btn-primary" style={{ flex: 1 }} disabled={!file} onClick={start}><Play size={16} /> Start analysis</button>
                : <button className="btn btn-danger" style={{ flex: 1 }} onClick={stop}><X size={16} /> Stop</button>}
              <button className="btn btn-secondary" disabled={running} onClick={() => fileRef.current?.click()}><Upload size={16} /> Choose file</button>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {file && <div className="card">
              <div className="card-title">📁 File</div>
              <KV k="Name" v={file.name} /><KV k="Size" v={`${(file.size / 1048576).toFixed(1)} MB`} />
              <KV k="Status" v={phase === 'uploading' ? '⬆️ uploading' : running ? '🔄 processing' : summary ? '✅ complete' : '📁 ready'} />
              {live && <KV k="Scene" v={live.scene} />}
            </div>}
            {c && <div className="card" style={{ background: 'rgba(59,130,246,0.05)', borderColor: 'rgba(59,130,246,0.2)' }}>
              <div className="card-title" style={{ color: '#3b82f6' }}><BarChart3 size={16} /> {summary ? 'Final results' : 'Live counters'}</div>
              <div className="mini-stats">
                <Mini v={c.unique_motorcycles} l={mode === 'site' ? 'people' : 'motorcycles'} col="#3b82f6" />
                <Mini v={c.unique_riders} l={mode === 'site' ? 'checked' : 'riders'} col="#8b5cf6" />
                <Mini v={c.riders_with_helmet} l="with helmet" col="#10b981" />
                <Mini v={c.violating_vehicles} l="violators" col="#ef4444" />
                <Mini v={c.frames} l="frames" col="#94a3b8" />
                <Mini v={c.fps} l="fps" col="#f59e0b" />
              </div>
              {c.compliance_rate != null && <div style={{ marginTop: 12, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Compliance: <b>{c.compliance_rate}%</b></div>}
            </div>}
            {summary && (
              <div className={`card ${c.violating_vehicles ? 'card-danger' : 'card-ok'}`}>
                <div className="card-title" style={{ color: c.violating_vehicles ? '#ef4444' : '#10b981' }}>
                  {c.violating_vehicles ? <AlertTriangle size={16} /> : <CheckCircle size={16} />} Verdict
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {c.violating_vehicles
                    ? `${c.violating_vehicles} of ${c.unique_motorcycles} motorcycles violated the rules; ${c.violations_recorded} evidence record(s) created — see Violations & e-Challan.`
                    : `All ${c.unique_riders} rider(s) on ${c.unique_motorcycles} motorcycle(s) were compliant.`}
                </div>
                {summary.output_video && <>
                  <video src={API + summary.output_video} controls muted playsInline className="result-image" style={{ marginTop: 12 }} />
                  <a className="btn btn-secondary btn-sm" style={{ marginTop: 10 }} href={API + summary.output_video} download><Download size={13} /> Download annotated MP4</a>
                </>}
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export const Mini = ({ v, l, col }) => (
  <div style={{ textAlign: 'center' }}>
    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: col }}>{v ?? '–'}</div>
    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '.5px' }}>{l}</div>
  </div>
);
