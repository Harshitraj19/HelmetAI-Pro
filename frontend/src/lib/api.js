// Backend location is configurable (base project hard-coded http://localhost:8000 in every page).
// Set VITE_API_URL in frontend/.env, e.g. VITE_API_URL=http://192.168.1.20:8000
export const API = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
export const WS_URL = API.replace(/^http/, 'ws') + '/ws/notifications';

export async function getJSON(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export const putJSON = (path, body) =>
  getJSON(path, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export const postJSON = (path, body, headers = {}) =>
  getJSON(path, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) });

// Admin token (only needed when the backend sets ADMIN_TOKEN). Entered once on the Model page,
// remembered for this browser and sent by every protected call.
export const getAdminToken = () => { try { return localStorage.getItem('hai_admin_token') || ''; } catch { return ''; } };
export const setAdminToken = (t) => { try { localStorage.setItem('hai_admin_token', t); } catch { /* storage blocked */ } };
export const adminHeaders = () => { const t = getAdminToken(); return t ? { 'X-Admin-Token': t } : {}; };

// Each video / stream run gets an id that the backend echoes in its WebSocket messages,
// so a page only reacts to its own run when several are processed at once.
export const newRunId = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

export const evidenceUrl = (p) => (p ? `${API}/api/evidence/file/${p}` : null);

export const VIOLATION_LABEL = {
  NO_HELMET: 'No helmet',
  INVALID_HELMET: 'Hard-hat (invalid)',
  HELMET_NOT_WORN: 'Helmet carried, not worn',
  TRIPLE_RIDING: 'Triple riding',
  NO_HEADGEAR: 'No helmet (person check)',
};

// What gets checked. 'site' is the every-person helmet check (webcams, selfies, workplaces).
export const CHECK_MODES = {
  auto: ['Auto', 'Motorcycle in the picture → traffic rules, otherwise every person is checked'],
  traffic: ['Traffic', 'Only riders on a detected motorcycle are checked; bystanders are blurred'],
  site: ['Helmet check', 'Every visible person must wear a helmet'],
};

export const SEVERITY_COLOR = { critical: '#dc2626', high: '#ef4444', medium: '#f59e0b', low: '#10b981' };

// Reads a multipart/x-mixed-replace MJPEG body from a fetch() response and calls onFrame(blobUrl)
// for each JPEG (needed because an <img> cannot show a POST response).
// Only the newest frame is shown: when several JPEGs arrive in one chunk the older ones are
// skipped, and the scan resumes where it stopped instead of re-reading the whole buffer.
// Every frame URL except the last is revoked here (the last one stays on screen; the page revokes
// it when it shows something else). Returns the number of frames received.
export async function readMjpeg(res, onFrame, signal) {
  const reader = res.body.getReader();
  let buf = new Uint8Array(0);
  let scan = 0;          // index from which to continue searching
  let start = -1;        // index of the current frame's FFD8, -1 = not found yet
  let lastUrl = null;
  let frames = 0;
  const onAbort = () => reader.cancel().catch(() => {});
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done || signal?.aborted) break;
      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf); merged.set(value, buf.length);
      buf = merged;
      let newest = null;
      let cut = 0;
      for (let i = Math.max(0, scan); i < buf.length - 1; i++) {
        if (buf[i] !== 0xff) continue;
        const m = buf[i + 1];
        if (m === 0xd8 && start < 0) start = i;
        else if (m === 0xd9 && start >= 0) { newest = [start, i + 2]; cut = i + 2; start = -1; frames++; }
      }
      if (newest) {
        const url = URL.createObjectURL(new Blob([buf.subarray(newest[0], newest[1])], { type: 'image/jpeg' }));
        onFrame(url);
        if (lastUrl) URL.revokeObjectURL(lastUrl);
        lastUrl = url;
      }
      // keep only the unfinished frame (or the last byte, which may be the FF of the next FFD8);
      // the last kept byte pairs with the next chunk, so scanning resumes there
      const keepFrom = start >= 0 ? start : Math.max(cut, buf.length - 1);
      buf = buf.slice(keepFrom);
      if (start >= 0) start = 0;
      scan = buf.length - 1;
      if (buf.length > 16_000_000) { buf = new Uint8Array(0); start = -1; scan = 0; } // corrupt stream safety
    }
  } catch (e) {
    if (e.name !== 'AbortError' && !signal?.aborted) throw e;
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
  return frames;
}
