# lensml landing page

Single static HTML page deployed to **Vercel free tier** at
`lensml.vercel.app` (plan §8 D28). No build step, no framework, no
JavaScript runtime — just `index.html` + `vercel.json` so deploys
take ~5 seconds.

## Local preview

```bash
# anything that serves static files works:
python -m http.server -d landing 4000   # → http://localhost:4000
```

## Deploy to Vercel free tier

```bash
# one-time:
npm install -g vercel
vercel login

# from the repo root:
vercel deploy --cwd landing            # preview deploy → URL
vercel deploy --cwd landing --prod     # promote to lensml.vercel.app
```

The `vercel.json` pins the static-only setup (no Node runtime needed)
and adds light security headers + a 5-minute edge cache. Vercel's
free tier handles the traffic profile of a hackathon submission with
several orders of magnitude of headroom.

## Why static HTML over Next.js

The dashboard is already a Next.js app under `dashboard/`. Mixing the
landing page into the same project would mean dragging server runtime
into a page that's pure marketing content. Static HTML keeps the deploy
unit small (one file, ~6 KB gzipped), the Lighthouse score high, and
the cost provably zero.

## Design notes

- Color palette matches the dashboard (`--bg #0a0a0a`, `--accent
  #06b6d4`, neutral text). Designers comparing the landing page and
  the live dashboard see the same visual identity.
- No webfont download — system font stack only. First contentful paint
  is bounded by the HTML round-trip.
- Inline SVG favicon (data URI) — no extra request.
