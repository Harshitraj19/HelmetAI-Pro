import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import {
  LayoutDashboard, Upload, Video, Database, Film, Brain, ShieldAlert, Sparkles, Settings as SettingsIcon,
  Volume2, VolumeX, Wifi, WifiOff,
} from 'lucide-react';
import { LiveProvider, useLive } from './lib/live';

// one chunk per page: the charts library is only downloaded when the dashboard is opened
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ImageDetect = lazy(() => import('./pages/ImageDetect'));
const LiveStream = lazy(() => import('./pages/LiveStream'));
const VideoDetect = lazy(() => import('./pages/VideoDetect'));
const History = lazy(() => import('./pages/History'));
const ModelTraining = lazy(() => import('./pages/ModelTraining'));
const Violations = lazy(() => import('./pages/Violations'));
const EnhanceLab = lazy(() => import('./pages/EnhanceLab'));
const Settings = lazy(() => import('./pages/Settings'));

const NAV = [
  { to: '/', icon: <LayoutDashboard />, label: 'Dashboard' },
  { to: '/detect', icon: <Upload />, label: 'Image Detection', short: 'Image' },
  { to: '/video', icon: <Film />, label: 'Video Detection', short: 'Video' },
  { to: '/stream', icon: <Video />, label: 'Live Stream', short: 'Live' },
  { to: '/violations', icon: <ShieldAlert />, label: 'Violations & e-Challan', short: 'Violations' },
  { to: '/enhance', icon: <Sparkles />, label: 'Enhancement Lab', short: 'Enhance' },
  { to: '/training', icon: <Brain />, label: 'Model & Training', short: 'Model' },
  { to: '/history', icon: <Database />, label: 'History' },
  { to: '/settings', icon: <SettingsIcon />, label: 'Settings' },
];

function MobileNav() {
  const { connected } = useLive();
  return (
    <nav className="mobile-nav" aria-label="Main">
      <span className={`mobile-dot ${connected ? 'on' : ''}`} title={connected ? 'Live alerts connected' : 'Backend offline'} />
      {NAV.map((n) => (
        <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => `mobile-item${isActive ? ' active' : ''}`}>
          {n.icon}<span>{n.short || n.label}</span>
        </NavLink>
      ))}
    </nav>
  );
}

const PageFallback = () => <div className="page-body" style={{ paddingTop: 28 }}><span className="loading-text">Loading…</span></div>;

function Sidebar() {
  const { connected, sound, toggleSound } = useLive();
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="logo-icon">🛡️</div>
        <span>HelmetAI <span style={{ color: 'var(--accent)' }}>Pro</span></span>
      </div>
      <nav className="sidebar-nav">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.to === '/'} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
            {n.icon} {n.label}
          </NavLink>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
          {connected ? <Wifi size={12} color="#10b981" /> : <WifiOff size={12} color="#ef4444" />}
          <span>{connected ? 'Live alerts connected' : 'Backend offline'}</span>
        </div>
        <button className="btn btn-secondary btn-sm" onClick={toggleSound} style={{ width: '100%', justifyContent: 'center' }}>
          {sound ? <Volume2 size={13} /> : <VolumeX size={13} />} Alarm {sound ? 'on' : 'off'}
        </button>
        <div style={{ color: 'var(--text-muted)', marginTop: 8 }}>DIP Project · v3 Pro</div>
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <LiveProvider>
        <div className="app-shell">
          <Sidebar />
          <MobileNav />
          <main className="main-content">
            <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/detect" element={<ImageDetect />} />
              <Route path="/video" element={<VideoDetect />} />
              <Route path="/stream" element={<LiveStream />} />
              <Route path="/violations" element={<Violations />} />
              <Route path="/enhance" element={<EnhanceLab />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/history" element={<History />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Dashboard />} />
            </Routes>
            </Suspense>
          </main>
        </div>
        <Toaster position="top-right" toastOptions={{
          style: { background: 'var(--bg-card)', color: 'var(--text-primary)', border: '1px solid var(--border)', borderRadius: '12px', fontSize: '0.85rem' },
        }} />
      </LiveProvider>
    </BrowserRouter>
  );
}
