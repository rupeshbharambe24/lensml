# Pitch copy (paste-ready)

## One-line tagline

> The AI agent that supervises your ML training runs.

## 80-character version (for tweet bios, app stores)

> Open-source AI agent for PyTorch training: detect, diagnose, patch, resume.

## Short description (~280 characters — Unstop short field, X bio)

lensml watches your PyTorch / Lightning / HF Trainer run live, detects
pathologies (exploding gradients, dead ReLUs, missing normalize, lr
issues, …), proposes a libcst-level patch, and resumes from checkpoint
on your `y`. Free-tier friendly, $0 to run.

## Long description (~600 words — Unstop long field, README, docs home)

ML training breaks in many ways and most of those breaks are
embarrassingly diagnosable from the loss curve, the gradient norms,
and a glance at the source. The professional move is to babysit the
run — tail the logs, eyeball the chart, kill the process, edit the
script, restart, and lose half a day. lensml automates the whole
loop.

You add one line to your training script:

```python
import lensml
handle = lensml.watch(model, optimizer)   # also Lightning + HF callbacks
```

That installs forward hooks on activation modules, wraps
`optimizer.step` for pre-step gradient inspection, and streams
telemetry to a local FastAPI server at <2% overhead. The Next.js
dashboard plots loss + grad-L2 in real time and renders intervention
markers wherever a detector fires.

When something pathological happens — gradients explode, a layer of
ReLUs dies, the learning rate is high enough that loss oscillates,
inputs are unnormalized, BatchNorm stats are wrong because someone
forgot `model.eval()` — an LLM agent diagnoses the cause, picks a
libcst-backed fix template (8 of them in v1, all idempotent), and
shows you a unified diff inside the dashboard. You press **`y`** to
approve. lensml commits the patch to a session git branch,
checkpoints model + optimizer + scheduler + RNG state, and restarts
your training subprocess with `LENSML_RESUME=<ckpt_path>` — so the
next process picks up exactly where the last one left off, with the
fix in place.

The novelty claim is "integration, not invention." Each individual
piece (libcst, LiteLLM routing, ZMQ telemetry, SSE dashboards,
checkpoint-aware restart) already exists; lensml is the open-source
agent that wires them all into a workflow that actually works on a
Sunday afternoon, on commodity hardware, on a 30-day solo build, on
$0 of cloud spend (every LLM call routes through free-tier
providers — Groq, Cerebras, Gemini, OpenRouter, Mistral — with
LiteLLM fallback).

It also ships **`lensml-bench`**: 8 deterministic Tiny fixtures with
known pathologies and canonical patches, a subprocess-based grader
(plan §6.1's three-check protocol — runs / loss-converged /
metric-passed), a JSON-cached runner, and an LLM-only baseline arm
that runs the same fixtures through a single-shot diagnose-and-patch
prompt. The final report renders a confusion matrix, latency
histogram, and $0 cost breakdown side by side — that's the
credibility lynchpin for the "integration, not invention" claim.

For privacy-sensitive deployments, `lensml dashboard` carries a
toggle for local-only mode (Plan-C scaffold for v1: server endpoint
+ canned diagnoses for the 8 v1 detectors; Ollama-backed real path
in v0.2). For demos, `lensml replay --scenario exploding_grad` plays
back a deterministic recorded session through the same broker → SSE
→ dashboard pipeline as a live trainer, so the magic-moment demo
works even when Wi-Fi is fighting back.

Three integrations, eight detectors, eight libcst templates,
checkpoint resume, replay mode, bench infrastructure, dashboard with
EN/JA i18n + command palette + keyboard shortcuts, free-tier landing
page, GitHub Pages docs — built solo over 30 days, every dependency
on its free tier, total cost: $0. Open source under Apache-2.0;
future enterprise features in `/ee/` under a separate license.
