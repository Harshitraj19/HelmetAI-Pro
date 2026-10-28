import { useCallback, useEffect, useState } from 'react';
import { Database, RefreshCw, AlertTriangle, CheckCircle, Trash2 } from 'lucide-react';
import toast from 'react-hot-toast';

import { API, adminHeaders } from '../lib/api';

export default function History() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [limit, setLimit] = useState(100);
  const [filter, setFilter] = useState('all'); // 'all' | 'violation' | 'safe'

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/detections?limit=${limit}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = await res.json();
      setRecords(Array.isArray(d) ? d : []);
    } catch {
      setRecords([]);
      toast.error('Could not load history. Is the backend running?', { id: 'hist' });
    } finally {
      setLoading(false);
    }
  }, [limit]);
  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  async function handleClearHistory() {
    if (!window.confirm('Are you sure you want to clear ALL detection history? This cannot be undone.')) return;
    
    setLoading(true);
    try {
      const res = await fetch(`${API}/api/detections`, { method: 'DELETE', headers: adminHeaders() });
      if (res.status === 401) throw new Error('Admin token required - enter it on the Model & Training page');
      if (!res.ok) throw new Error('Failed to clear history');
      toast.success('History cleared successfully');
      setRecords([]);
    } catch (err) {
      toast.error(err.message || 'Error clearing history');
    } finally {
      setLoading(false);
    }
  }

  const filtered = records.filter(r => {
    if (filter === 'violation') return r.object_class?.toLowerCase().includes('without');
    if (filter === 'safe') return !r.object_class?.toLowerCase().includes('without');
    return true;
  });

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Detection History</h1>
        <p className="page-subtitle">All stored detection records from the database</p>
      </div>
      <div className="page-body">

        {/* Controls Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          {/* Filter Tabs */}
          <div style={{ display: 'flex', gap: 4, background: 'var(--bg-secondary)', padding: 4, borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)' }}>
            {['all', 'violation', 'safe'].map(f => (
              <button key={f} onClick={() => setFilter(f)}
                style={{
                  padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer',
                  fontSize: '0.78rem', fontWeight: 600, textTransform: 'capitalize', fontFamily: 'inherit',
                  background: filter === f ? 'var(--accent)' : 'transparent',
                  color: filter === f ? '#fff' : 'var(--text-secondary)',
                  transition: 'all 0.2s',
                }}
              >{f === 'violation' ? '⚠️ Violations' : f === 'safe' ? '✅ Safe' : '📋 All'}</button>
            ))}
          </div>

          {/* Limit selector */}
          <select
            value={limit}
            onChange={e => setLimit(Number(e.target.value))}
            style={{
              padding: '8px 12px', background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)', fontSize: '0.85rem',
              cursor: 'pointer', outline: 'none', fontFamily: 'inherit',
            }}
          >
            {[50, 100, 200, 500].map(n => <option key={n} value={n}>Last {n}</option>)}
          </select>

          <button className="btn btn-secondary btn-sm" onClick={fetchHistory} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
          </button>

          <button className="btn btn-danger btn-sm" onClick={handleClearHistory} disabled={loading || records.length === 0}>
            <Trash2 size={14} /> Clear detections
          </button>

          <div style={{ marginLeft: 'auto', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            Showing {filtered.length} of {records.length} records
          </div>
        </div>

        {/* Table */}
        <div className="card">
          <div className="card-title"><Database size={16} /> Records</div>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <div className="loading-text">Loading records…</div>
            </div>
          ) : filtered.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              No records found. Start detecting to populate the database.
            </div>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Timestamp</th>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>Source</th>
                    <th>Track</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((r) => {
                    const isViolation = r.object_class?.toLowerCase().includes('without');
                    return (
                      <tr key={r.id}>
                        <td className="mono" style={{ color: 'var(--text-muted)' }}>{r.id}</td>
                        <td className="mono">{r.timestamp ? new Date(r.timestamp + 'Z').toLocaleString() : '—'}</td>
                        <td style={{ fontWeight: 600, color: isViolation ? '#ef4444' : '#10b981' }}>{r.object_class}</td>
                        <td className="mono">{r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : '—'}</td>
                        <td style={{ color: 'var(--text-muted)', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {r.image_path || 'live stream'}
                        </td>
                        <td className="mono" style={{ color: 'var(--text-muted)' }}>{r.track_id ?? '–'}</td>
                        <td>
                          {isViolation
                            ? <span className="badge badge-danger"><AlertTriangle size={10} style={{ display: 'inline', marginRight: 3 }} />Violation</span>
                            : <span className="badge badge-success"><CheckCircle size={10} style={{ display: 'inline', marginRight: 3 }} />Safe</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

      </div>
    </>
  );
}
