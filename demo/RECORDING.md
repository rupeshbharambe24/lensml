# 90-second Round 1 video — recording playbook (plan §8 D27)

The video lives on YouTube **unlisted** (free), linked from the
README. Three takes, pick the best, upload.

## Beat sheet (90 s budget)

| t       | beat | screen |
| ------- | ---- | ------ |
| 0–10s   | "Your training run is broken. lensml watches it like a debugger." | Hero card / title |
| 10–25s  | One-line install + import. Show `lensml.watch(model, optimizer)`. | Terminal (install.tape) |
| 25–45s  | Start the dashboard. Live loss curve and grad-L2 stream. | Browser + terminal split |
| 45–65s  | Pathology fires (red marker on chart, agent reasoning pops in). | Browser, dashboard pane |
| 65–80s  | `y` to approve the diff. Git commit. Trainer auto-resumes from checkpoint. | Browser + terminal |
| 80–90s  | "$0 to run. Open source. Try it." Repo URL on screen. | Title card |

## Capture setup

- **Browser**: Chrome at 1280 × 720, dashboard at `http://127.0.0.1:3000`.
- **Terminal**: 900 × 420, GruvboxDark theme (matches the .tape files).
- **Recorder**: OBS Studio (free) or QuickTime + Loom. Record both
  windows in a 1920 × 1080 canvas, side by side.
- **Audio**: voice-over recorded separately, mixed in DaVinci Resolve
  (free). Or skip audio and rely on captions — Round 1 judging is
  often muted.

## Driver script for the magic moment

The deterministic intervention demo runs entirely on the replay
scenario — no LLM, no GPU, no flakiness. Bring up the dashboard
server, the dashboard UI, then run:

```bash
lensml replay --scenario exploding_grad --speed 1
```

The dashboard will tick through ~110 s of synthetic training,
intervention markers appear at the spike, and the agent reasoning
pane animates in. For the 65 s "approve" beat, manually trigger
`/api/runs/demo-exploding-grad/decisions` flow with the included
test publisher (see `lensml/server/app.py::_test_publish` / a
canned approval POST, scripted in `demo/scripts/`).

## Takes plan

- Take 1: timing rehearsal, microphone levels, no upload.
- Take 2: the real one. Aim for 85–95 s total.
- Take 3: backup. Different camera angle for the talking-head intro
  if you record one.

Pick the best, edit ruthlessly. 90 s hard cap.

## Upload checklist

- [ ] Title: `lensml — an AI agent that supervises your ML training`
- [ ] Visibility: **Unlisted** (free). Public toggle flips on June 7.
- [ ] Description: 2-line synopsis + repo URL + "Built solo on free
      tiers in 30 days."
- [ ] Tags: `pytorch, ml, training, observability, agent, llm,
      hackathon, far-away`
- [ ] Add the YouTube link to the README's hero block (the slot is
      pre-marked with `<!-- VIDEO_LINK -->`).
- [ ] Add the link to the FAR AWAY hackathon submission form.

## Honest scope

Everything in this file is a checklist for the human, not automation.
The .tape files automate the *regenerable terminal recordings*; the
talking-head video and screen capture are still manual. That's
unavoidable — the credibility of "look how it feels" demands a real
person showing the dashboard reacting in real time.
