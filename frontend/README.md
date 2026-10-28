# HelmetAI-Pro — frontend

React 19 + Vite dashboard for the HelmetAI-Pro backend (see the project README one level up).

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production build in dist/
npm run lint
```

The backend URL defaults to `http://localhost:8000`; set `VITE_API_URL` in `frontend/.env` to point elsewhere.
Pages are code-split per route, so the charts library only loads with the dashboard.
