# lensml dashboard

Next.js 15 App Router + React 19 dashboard for lensml.

## Stack (plan §7.4)

- Next.js 15 App Router + React 19
- Tailwind + shadcn/ui (judges recognize)
- uPlot for live charts (48KB, 60fps with thousands of points)
- react-diff-viewer-continued (the maintained fork)
- Native EventSource (auto-reconnect)
- Framer Motion for tool-card pop-ins
- Static export, deployed to Vercel (free)

## First-time setup

This directory is currently a placeholder structure. On Day 4 of the build:

```bash
cd dashboard
pnpm create next-app@15 . --typescript --tailwind --app --src-dir false
pnpm add uplot uplot-react react-diff-viewer-continued framer-motion cmdk
pnpm add -D @types/node
pnpx shadcn@latest init
```

## Layout (plan §7.1)

```
┌─ LIVE METRICS ──────┬─ AGENT REASONING ─┐
│ loss curve (uPlot)  │ tool-call cards   │
│ grad-L2 norm        │ hypothesis stream │
│ activation stats    │ confidence pill   │
│                     ├───────────────────┤
│                     │ DIFF PREVIEW      │
│                     │ [y]/[n]/[u] keys  │
└─────────────────────┴───────────────────┘
```

## Critical UX details (plan §7.3)

- Vertical line on the loss curve at every intervention point.
- Status pill at top: rotating verb ("Watching… Diagnosing… Patching… Resumed.").
- Keyboard shortcuts: `y` accept, `n` reject, `u` undo, `?` help, `j/k` nav, `⌘Enter` apply all.
- Reasoning trace streams token-by-token (16ms batch flushes).
- Demo replay mode: `lensml replay <jsonl>` runs deterministic playback. **Use this for the live demo.**
- Dark mode default.
- JA toggle (plan §10 Round 3 Tokyo): 3-second highlight during demo.

## Mobile (plan §7.2)

Single-column tabbed view: Metrics / Agent / Diff. Approve/Reject buttons
sticky to bottom. Test on iPhone Safari and 1080p projector.

## Build schedule

- Day 4: scaffold + uPlot live chart wired to SSE
- Day 12: diff viewer + approval flow
- Day 19: status pill, intervention markers, animations, JA toggle scaffold
