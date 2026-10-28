import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend, BarChart, Bar,
} from 'recharts';
import { Shield, TrendingUp, Bell, AlertTriangle, Repeat, ScanLine, CheckCircle } from 'lucide-react';
import { getJSON, VIOLATION_LABEL, SEVERITY_COLOR } from '../lib/api';
import { useLive } from '../lib/live';

const TT = { contentStyle: { background: '#0f1623', border: '1px solid rgba(99,179,237,0.15)', borderRadius: 10, fontSize: 12 }, labelStyle: { color: '#94a3b8' } };

function StatCard({ label, value, sub, icon, gradient }) {
  return (
    <div className="stat-card" style={{ '--gradient-bar': gradient }}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? '—'}</div>
      {sub && <div className="stat-sub">{sub}</div>}
      <div className="stat-icon">{icon}</div>
    </div>
  );
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [verify, setVerify] = useState(null);
  const { alerts } = useLive();

  const load = useCallback(() => {
    if (document.hidden) return;           // no polling in a background tab
    getJSON('/api/stats').then(setStats).catch(() => {});
    getJSON('/api/violations?limit=8').then(setRecent).catch(() => {});
    // quick check (cached hashes of unchanged images); the Violations page runs the full re-hash
    getJSON('/api/evidence/verify?full=false').then(setVerify).catch(() => {});
  }, []);
  useEffect(() => {
    load();
    const iv = setInterval(load, 10000);
    document.addEventListener('visibilitychange', load);
    return () => { clearInterval(iv); document.removeEventListener('visibilitychange', load); };
  }, [load]);
  useEffect(() => { if (alerts.length) load(); }, [alerts.length, load]);

  const pad = (h) => String(h).padStart(2, '0');
  const hourly = stats ? Array.from({ length: 24 }, (_, h) => ({
    hour: `${pad(h)}:00`, riders: stats.hourly_breakdown?.[pad(h)] ?? 0, violations: stats.violation_hourly?.[pad(h)] ?? 0,
  })) : [];
  const pie = stats ? [{ name: 'With Helmet', value: stats.with_helmet }, { name: 'Without Helmet', value: stats.without_helmet }] : [];
  const types = stats ? Object.entries(stats.violation_types || {}).map(([k, v]) => ({ name: VIOLATION_LABEL[k] || k, value: v })) : [];
  const sev = stats ? ['critical', 'high', 'medium', 'low'].map((k) => ({ name: k, value: stats.severity?.[k] || 0 })) : [];

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Analytics Dashboard</h1>
        <p className="page-subtitle">Unique riders, enforcement-grade violations and evidence integrity, in real time</p>
      </div>
      <div className="page-body">
        <div className="stats-grid">
          <StatCard label="Riders checked" value={stats ? (stats.with_helmet + stats.without_helmet).toLocaleString() : '…'} sub="Unique riders (tracked)" icon="🏍️" gradient="linear-gradient(90deg,#3b82f6,#6366f1)" />
          <StatCard label="Compliance rate" value={stats ? `${stats.compliance_rate}%` : '…'} sub="Riders wearing a helmet" icon="🛡️" gradient="linear-gradient(90deg,#10b981,#059669)" />
          <StatCard label="Violations" value={stats?.violations_total?.toLocaleString() ?? '…'} sub="Evidence records" icon="⚠️" gradient="linear-gradient(90deg,#ef4444,#dc2626)" />
          <StatCard label="Plates read" value={stats ? `${stats.plate_read_rate}%` : '…'} sub={`${stats?.plates_read ?? 0} vehicles identified`} icon="🔢" gradient="linear-gradient(90deg,#f59e0b,#d97706)" />
        </div>

        <div className="grid-3">
          <div className="card">
            <div className="card-title"><TrendingUp size={16} /> Riders vs violations (last 24 h)</div>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={hourly} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} /><stop offset="95%" stopColor="#3b82f6" stopOpacity={0} /></linearGradient>
                  <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ef4444" stopOpacity={0.35} /><stop offset="95%" stopColor="#ef4444" stopOpacity={0} /></linearGradient>
                </defs>
                <XAxis dataKey="hour" tick={{ fontSize: 10, fill: '#64748b' }} interval={3} />
                <YAxis tick={{ fontSize: 10, fill: '#64748b' }} allowDecimals={false} />
                <Tooltip {...TT} />
                <Area type="monotone" dataKey="riders" stroke="#3b82f6" fill="url(#g1)" strokeWidth={2} dot={false} />
                <Area type="monotone" dataKey="violations" stroke="#ef4444" fill="url(#g2)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="card-title"><Shield size={16} /> Helmet compliance</div>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={pie} cx="50%" cy="50%" innerRadius={60} outerRadius={90} dataKey="value" paddingAngle={4}>
                  {pie.map((_, i) => <Cell key={i} fill={['#10b981', '#ef4444'][i]} />)}
                </Pie>
                <Legend iconType="circle" iconSize={10} wrapperStyle={{ fontSize: 12, color: '#94a3b8' }} />
                <Tooltip {...TT} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="grid-3" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
          <div className="card">
            <div className="card-title"><AlertTriangle size={16} /> Violation types</div>
            {types.length === 0 ? <Empty text="No violations recorded yet" /> : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={types} layout="vertical" margin={{ left: 30 }}>
                  <XAxis type="number" hide allowDecimals={false} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} width={120} />
                  <Tooltip {...TT} />
                  <Bar dataKey="value" fill="#ef4444" radius={[0, 6, 6, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
          <div className="card">
            <div className="card-title"><ScanLine size={16} /> Severity mix</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={sev}>
                <XAxis dataKey="name" tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: '#64748b' }} />
                <Tooltip {...TT} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>{sev.map((s) => <Cell key={s.name} fill={SEVERITY_COLOR[s.name]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card">
            <div className="card-title"><Repeat size={16} /> Repeat offenders & integrity</div>
            {(stats?.repeat_offenders || []).length === 0 && <div style={{ color: 'var(--text-muted)', fontSize: '0.82rem', marginBottom: 12 }}>No plate seen violating twice yet.</div>}
            {(stats?.repeat_offenders || []).map((r) => (
              <div key={r.plate} className="detection-item violation"><div className="detection-label mono">{r.plate}</div><span className="badge badge-danger">{r.count}×</span></div>
            ))}
            <div className={`chain-box ${verify?.valid === false ? 'bad' : 'good'}`}>
              {verify == null ? 'Checking evidence chain…' : verify.valid
                ? <><CheckCircle size={14} /> Evidence chain intact · {verify.checked} records</>
                : <><AlertTriangle size={14} /> TAMPERING at record #{verify.broken_id}</>}
            </div>
          </div>
        </div>

        <div className="grid-2">
          <div className="card">
            <div className="card-title"><AlertTriangle size={16} /> Latest violations <Link to="/violations" style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--accent)' }}>open all →</Link></div>
            <div className="detection-list">
              {recent.length === 0 && <Empty text="No violations yet. Run an image, video or stream." />}
              {recent.map((v) => (
                <div key={v.id} className="detection-item violation">
                  <div>
                    <div className="detection-label">#{v.id} · {v.violation_list.map((x) => VIOLATION_LABEL[x] || x).join(' + ')}</div>
                    <div className="detection-time">{new Date(v.created_at + 'Z').toLocaleString()} · {v.camera_id} · {v.riders} rider(s)</div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                    <span className="mono" style={{ fontSize: '0.8rem' }}>{v.plate || 'plate n/a'}</span>
                    <span className="badge" style={{ background: SEVERITY_COLOR[v.severity_level] + '22', color: SEVERITY_COLOR[v.severity_level] }}>{v.severity_level} {v.severity}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="card">
            <div className="card-title"><Bell size={16} /> Live alerts</div>
            {alerts.length === 0 && <Empty text="🔕 Waiting for violations… start a stream or upload a video." />}
            {alerts.slice(0, 8).map((a) => (
              <div key={a.id} className="violation-alert">
                <span className="alert-icon">⚠️</span>
                <div>
                  <div className="alert-title">{a.violation_list.map((x) => VIOLATION_LABEL[x] || x).join(' + ')}</div>
                  <div className="alert-desc">{a.riders} rider(s) · {a.plate || 'plate not read'} · severity {a.severity}</div>
                  <div className="alert-time">{new Date(a.created_at + 'Z').toLocaleTimeString()} · {a.source}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

const Empty = ({ text }) => <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '24px 0' }}>{text}</div>;
