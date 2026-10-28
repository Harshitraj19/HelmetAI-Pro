import { useEffect, useRef, useState } from 'react';
import { Video, Square, Radio, AlertTriangle, Circle, ShieldCheck, ShieldX, UserX } from 'lucide-react';
import toast from 'react-hot-toast';
import { API, getJSON, newRunId, VIOLATION_LABEL } from '../lib/api';
import { useLive } from '../lib/live';
import { KV } from './ImageDetect';
import { Mini } from './VideoDetect';

const MODES = [
  ['auto', 'Auto', 'Webcam → helmet check on every person · videos / IP cameras → traffic rules'],
  ['site', 'Helmet check (every person)', 'Every visible person must wear a helmet - use this for a webcam demo'],
  ['traffic', 'Traffic (motorcycle riders)', 'Only people riding a detected motorcycle are checked; bystanders are blurred'],
];
const resolveMode = (mode, source) => (mode !== 'auto' ? mode : /^\d+$/.test(source.trim()) ? 'site' : 'traffic');

// verdict for the current frame, from the live per-frame counts
function verdict(score, mode) {
  if (!score) return null;
  const who = (n) => (mode === 'site' ? (n > 1 ? 'people' : 'person') : (n > 1 ? 'riders' : 'rider'));
  if (score.no_helmets > 0) return { kind: 'bad', text: `NO HELMET · ${score.no_helmets} ${who(score.no_helmets)} without a helmet` };
  if (score.helmets > 0) return { kind: 'good', text: `Helmet on · ${score.helmets} ${who(score.helmets)} wearing a helmet` };
  return { kind: 'none', text: mode === 'site' ? 'No person in view' : 'No motorcycle rider in view' };
}

const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACw=';     // 1x1 image: replaces the stream URL

// Ask the backend to stop a run and release its camera. keepalive/beacon still work while the tab closes.
async function stopOnServer(runId, unloading = false) {
  const url = `${API}/api/stream/stop?run_id=${encodeURIComponent(runId)}`;
  if (unloading && navigator.sendBeacon) return navigator.sendBeacon(url);
  try { return (await fetch(url, { method: 'POST', keepalive: true })).ok; } catch { return false; }
}

export default function LiveStream() {
  const [streaming, setStreaming] = useState(false);
  const [source, setSource] = useState('0');
  const [record, setRecord] = useState(false);
  const [samples, setSamples] = useState([]);
  const [score, setScore] = useState(null);
  const [ended, setEnded] = useState(null);       // final message of the last run (clip finished)
  const [mode, setMode] = useState('auto');
  const [runMode, setRunMode] = useState(null);    // the mode actually used by the running stream
  const imgRef = useRef();
  const runRef = useRef(null);
  const statusRef = useRef({ shown: null, pending: null, count: 0 });
  const { subscribe, alerts } = useLive();

  useEffect(() => { getJSON('/api/samples').then((s) => setSamples(s.filter((x) => /\.(mp4|avi|mov|mkv|webm)$/i.test(x)))).catch(() => {}); }, []);
  useEffect(() => subscribe((m) => {
    if (!runRef.current || (m.run_id && m.run_id !== runRef.current)) return;   // another viewer's stream
    if (m.type === 'video_score') {
      setScore(m);
      // pop-up when the verdict changes - only after it held for 2 updates (~1 s), so a single
      // missed detection doesn't flash a wrong message
      const v = verdict(m, m.mode)?.kind;
      const st = statusRef.current;
      if (v === st.pending) st.count += 1; else { st.pending = v; st.count = 1; }
      if (st.count === 2 && v !== st.shown) {
        st.shown = v;
        if (v === 'bad') toast.error(`No helmet detected! ${m.no_helmets} ${m.no_helmets > 1 ? 'people' : 'person'} without a helmet`, { id: 'live-verdict', duration: 3000 });
        else if (v === 'good') toast.success(`Helmet detected ✓ (${m.helmets})`, { id: 'live-verdict', duration: 2500 });
      }
    }
    if (m.type === 'stream_done') {
      setEnded(m);
      if (m.error) toast.error(`Stream stopped: ${m.error}`);
    }
  }), [subscribe]);
  // Leaving the page or closing the tab must release the camera. Closing the <img> connection alone
  // is not reliable (browsers may keep an MJPEG request open), so the server is also told to stop.
  useEffect(() => {
    const img = imgRef.current;
    const leave = () => { if (runRef.current) stopOnServer(runRef.current, true); };
    window.addEventListener('pagehide', leave);
    return () => {
      window.removeEventListener('pagehide', leave);
      leave();
      if (img) img.src = BLANK;
    };
  }, []);

  const start = () => {
    const runId = newRunId(); runRef.current = runId;
    const m = resolveMode(mode, source);
    statusRef.current = { shown: null, pending: null, count: 0 };
    setScore(null); setEnded(null); setStreaming(true); setRunMode(m);
    imgRef.current.src = `${API}/api/stream/video?source=${encodeURIComponent(source.trim())}&record=${record}&mode=${m}&run_id=${runId}`;
  };
  const stop = () => {
    const id = runRef.current;
    runRef.current = null;
    if (imgRef.current) imgRef.current.src = BLANK;      // drop the MJPEG connection
    setStreaming(false);
    if (id) stopOnServer(id).then((ok) => ok && toast.success('Stream stopped · camera released', { id: 'stream-stop' }));
  };
  const onError = async () => {
    if (!runRef.current) return;
    const src = imgRef.current?.src;
    runRef.current = null; setStreaming(false);
    // an <img> can't show the server's reason (bad source, no model ...). Ask again, but only read
    // the body if it is a JSON error - a working MJPEG stream is aborted at once, not consumed
    const ctrl = new AbortController();
    try {
      const r = await fetch(src.replace(/run_id=[^&]+/, 'run_id=probe'), { signal: ctrl.signal });
      if (!(r.headers.get('content-type') || '').includes('json')) { ctrl.abort(); toast.error('The stream was interrupted'); return; }
      toast.error((await r.json()).detail || 'Could not open the stream');
    } catch (e) { if (e.name !== 'AbortError') toast.error('Backend unreachable'); }
  };
  const c = (ended || score)?.counters;
  const site = (streaming && runMode ? runMode : resolveMode(mode, source)) === 'site';
  const v = streaming && !ended ? verdict(score, runMode) : null;

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Live Stream</h1>
        <p className="page-subtitle">Webcam, IP camera (RTSP), YouTube or a sample clip — tracked in real time with audible alarms</p>
      </div>
      <div className="page-body">
        <div className="card" style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <input className="input" style={{ flex: 1, minWidth: 0 }} value={source} disabled={streaming}
            onChange={(e) => setSource(e.target.value)} placeholder="0 = webcam · rtsp://user:pass@ip/stream · https://youtu.be/… · sample file" list="samples" />
          <datalist id="samples">{samples.map((s) => <option key={s} value={s} />)}</datalist>
          <select className="input" value={mode} disabled={streaming} onChange={(e) => setMode(e.target.value)}
            title={MODES.find((x) => x[0] === mode)[2]}>
            {MODES.map(([k, l]) => <option key={k} value={k}>{k === 'auto' ? `Auto (${resolveMode('auto', source) === 'site' ? 'helmet check' : 'traffic'})` : l}</option>)}
          </select>
          <label className="toggle-inline"><input type="checkbox" checked={record} disabled={streaming} onChange={(e) => setRecord(e.target.checked)} /> Record MP4</label>
          {!streaming
            ? <button className="btn btn-primary" onClick={start}><Video size={16} /> Start</button>
            : <button className="btn btn-danger" onClick={stop}><Square size={16} /> Stop</button>}
        </div>
        {samples.length > 0 && !streaming && (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
            {samples.map((s) => <button key={s} className="chip" onClick={() => setSource(s)}>{s}</button>)}
          </div>
        )}
        <div className="grid-3" style={{ alignItems: 'start' }}>
          <div className="stream-container">
            <img ref={imgRef} alt="MJPEG stream" className="stream-img" style={{ display: streaming ? 'block' : 'none' }} onError={onError} />
            {!streaming && <div className="stream-placeholder"><Video /><div style={{ marginTop: 8 }}>Stream is offline</div></div>}
            {streaming && !ended && <div className="stream-status"><div className="pulse-dot" /> LIVE {record && <Circle size={10} fill="#ef4444" color="#ef4444" style={{ marginLeft: 6 }} />}</div>}
            {streaming && ended && <div className="stream-status" style={{ background: 'rgba(15,22,35,0.85)' }}>■ ENDED · {ended.counters?.frames} frames</div>}
            {v && (
              <div className={`verdict verdict-${v.kind}`} role="status" aria-live="polite">
                {v.kind === 'bad' ? <ShieldX size={20} /> : v.kind === 'good' ? <ShieldCheck size={20} /> : <UserX size={20} />}
                {v.text}
              </div>
            )}
            {streaming && !ended && !site && score && !score.counters?.unique_motorcycles && score.frame > 40 && (
              <div className="verdict-hint">Traffic mode only checks riders on a motorcycle. Testing with a webcam? Stop and choose <b>Helmet check (every person)</b>.</div>
            )}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="card">
              <div className="card-title"><Radio size={16} /> Stream</div>
              <KV k="Status" v={streaming ? (ended ? '⏹ finished' : '🟢 live') : '🔴 offline'} />
              <KV k="Source" v={source} />
              <KV k="Mode" v={site ? 'Helmet check (every person)' : 'Traffic (motorcycle riders)'} />
              <KV k="Scene" v={score?.scene ?? '–'} />
              <KV k="Protocol" v="MJPEG over HTTP + WebSocket events" />
            </div>
            <div className="card" style={{ background: 'rgba(59,130,246,0.05)', borderColor: 'rgba(59,130,246,0.2)' }}>
              <div className="card-title" style={{ color: '#3b82f6' }}>📊 Session counters</div>
              <div className="mini-stats">
                <Mini v={streaming && !ended ? score?.helmets : '–'} l="helmets now" col="#10b981" />
                <Mini v={streaming && !ended ? score?.no_helmets : '–'} l="no helmet now" col="#ef4444" />
                <Mini v={c?.unique_motorcycles} l={site ? 'people seen' : 'bikes seen'} col="#3b82f6" />
                <Mini v={site ? c?.riders_with_helmet : c?.unique_riders} l={site ? 'with helmet' : 'riders'} col="#8b5cf6" />
                <Mini v={c?.violating_vehicles} l="violators" col="#ef4444" />
                <Mini v={c?.fps} l="fps" col="#f59e0b" />
              </div>
            </div>
            <div className="card" style={{ background: 'rgba(239,68,68,0.05)', borderColor: 'rgba(239,68,68,0.2)' }}>
              <div className="card-title" style={{ color: '#ef4444' }}><AlertTriangle size={16} /> Live violations</div>
              {alerts.length === 0 ? <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No violations yet.</div> :
                alerts.slice(0, 5).map((a) => (
                  <div key={a.id} className="violation-alert" style={{ marginBottom: 8 }}>
                    <span className="alert-icon">⚠️</span>
                    <div>
                      <div className="alert-title">{a.violation_list.map((x) => VIOLATION_LABEL[x] || x).join(' + ')}</div>
                      <div className="alert-desc">{a.violation_list.includes('NO_HEADGEAR') ? `person · ${a.camera_id}` : (a.plate || 'plate n/a')} · severity {a.severity}</div>
                      <div className="alert-time">{new Date(a.created_at + 'Z').toLocaleTimeString()}</div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
