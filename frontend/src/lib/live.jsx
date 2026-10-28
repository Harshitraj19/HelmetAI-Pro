// One shared WebSocket for the whole app (the base opened a new socket on every page),
// plus the audible alarm idea from the KrishArul26 repo, done with WebAudio (no sound file).
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { WS_URL, VIOLATION_LABEL } from './api';

const LiveCtx = createContext(null);
// eslint-disable-next-line react-refresh/only-export-components
export const useLive = () => useContext(LiveCtx);

function beep(severity = 'medium') {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const pattern = severity === 'critical' || severity === 'high' ? [880, 660, 880] : [740];
    pattern.forEach((f, i) => {
      const o = ctx.createOscillator(); const g = ctx.createGain();
      o.type = 'square'; o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, ctx.currentTime + i * 0.18);
      g.gain.exponentialRampToValueAtTime(0.15, ctx.currentTime + i * 0.18 + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + i * 0.18 + 0.15);
      o.connect(g).connect(ctx.destination);
      o.start(ctx.currentTime + i * 0.18); o.stop(ctx.currentTime + i * 0.18 + 0.16);
    });
    setTimeout(() => ctx.close(), 1200);
  } catch { /* audio blocked until first user gesture - ignore */ }
}

export function LiveProvider({ children }) {
  const [connected, setConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);        // recent violation records
  const [sound, setSound] = useState(() => { try { return localStorage.getItem('hai_sound') !== 'off'; } catch { return true; } });
  const listeners = useRef(new Set());
  const soundRef = useRef(sound);
  useEffect(() => { soundRef.current = sound; }, [sound]);

  useEffect(() => {
    let ws, ping, retry, stopped = false;
    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => { setConnected(true); ping = setInterval(() => ws.readyState === 1 && ws.send('ping'), 20000); };
      ws.onclose = () => { setConnected(false); clearInterval(ping); if (!stopped) retry = setTimeout(connect, 3000); };
      ws.onmessage = (e) => {
        let msg;
        try { msg = JSON.parse(e.data); } catch { return; }
        if (!msg || typeof msg !== 'object') return;
        listeners.current.forEach((fn) => { try { fn(msg); } catch (err) { console.error(err); } });
        if (msg.type === 'violation' && msg.record) {
          setAlerts((a) => [msg.record, ...a].slice(0, 30));
          const r = msg.record;
          const what = (r.violation_list || []).map((v) => VIOLATION_LABEL[v] || v).join(' + ');
          toast.error(`#${r.id} ${what}${r.plate ? ` · ${r.plate}` : ''}`, { id: `v${r.id}` });
          if (soundRef.current) beep(r.severity_level);
        }
        // per-frame '*_score' / '*_done' messages go only to subscribers (kept out of React state:
        // storing them here re-rendered the whole app twice a second while a video was running)
      };
    };
    connect();
    return () => { stopped = true; clearTimeout(retry); clearInterval(ping); ws && ws.close(); };
  }, []);

  const toggleSound = useCallback(() => setSound((s) => {
    try { localStorage.setItem('hai_sound', s ? 'off' : 'on'); } catch { /* storage blocked */ }
    return !s;
  }), []);
  // stable identity: pages subscribe in an effect keyed on it, so a new function every render
  // made them unsubscribe/resubscribe on every incoming message
  const subscribe = useCallback((fn) => { listeners.current.add(fn); return () => listeners.current.delete(fn); }, []);

  const value = useMemo(() => ({ connected, alerts, sound, toggleSound, subscribe }),
    [connected, alerts, sound, toggleSound, subscribe]);
  return <LiveCtx.Provider value={value}>{children}</LiveCtx.Provider>;
}
