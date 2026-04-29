# lensml — final 30-day build plan

**Author:** Rupesh Bharambe
**Status:** v1 — locked
**Build window:** May 8 – June 7, 2026 (30 days, 6–8 hr/day)
**Submission:** FAR AWAY hackathon, Round 1 (Unstop), June 7
**Hardware:** RTX 3050 Ti laptop (4GB VRAM) + Colab T4 + Kaggle T4
**License:** Apache-2.0 core, EE folder for future paid features
**Repo visibility:** Private until June 7, public on submission day

---

## 0. The honest framing

This document is the spec you build from. It is not a sales document. Where things are uncertain, I name them. Where prior plans were wrong, I correct them. Where 30 days is not enough for an item, I cut it.

Three calibrations to start:

**The novelty claim is "integration, not invention."** TrainCheck (OSDI 2025) detects training pathologies. AutoTrainer (ICSE 2021, TDSC 2024) detects and repairs ~8 pathology classes — but in Keras only, with fixed rules, and operates on in-memory model graphs not source code. Cursor and Aider rewrite source code but are blind to running training processes. lensml's defensible novelty is the *combination*: live PyTorch/Lightning/HF instrumentation + LLM-driven diagnosis + libcst-level source patching + checkpoint-aware resume. Pitch it that way. Never claim "no one has done X" — claim "no maintained open-source tool integrates these into one workflow."

**The 4GB VRAM laptop is a real constraint.** Local development uses MNIST and CIFAR-10 only. All BERT/ResNet/GPT-2 benchmark runs go to Colab T4 (16GB) or Kaggle T4 (16GB × 2). The bench has been scoped to fit free quotas. This is not a fudge — see Section 6 for compute math.

**The three-framework scope is ambitious but doable in 30 days.** Raw PyTorch → Lightning → HF Trainer is a graceful tier where each builds on the previous. Cut order if Week 3 slips: drop HF Trainer integration to a documented "v0.2 roadmap item," ship Lightning + raw PyTorch fully. Don't cut Lightning — it's the easiest second integration and signals breadth.

---

## 1. What lensml is, in one paragraph

lensml is a Python package + local web dashboard that watches your PyTorch training run in real time. You add one line — `lensml.watch(model, optimizer, dataloader)` — to your training script. lensml installs hooks, streams metrics to a local FastAPI server, and runs an LLM agent that monitors the stream. When the agent detects a pathology (exploding gradients, dead ReLUs, missing normalization, model.eval() not called, etc.), it pauses training, diagnoses the cause, generates a code patch via libcst, and presents a diff in the dashboard. You press `y` to apply, `n` to reject, `u` to undo. On accept, lensml commits the patch to a git branch, checkpoints model + optimizer + RNG state, restarts your script, and resumes training from the checkpoint with the fixed code. Privacy mode keeps everything local: Ollama + a small open-source coder model, no data leaves your machine.

---

## 2. Verified vs unverified — what I'm confident about

I'll be specific about confidence. Anything below "high confidence" needs a Day 1 spike.

### High confidence (won't bet against in this plan)

- **PyTorch hooks.** `register_forward_hook` and `register_full_backward_hook` are stable in PyTorch 2.x. The deprecated `register_backward_hook` should be avoided. Reading `param.grad` after `loss.backward()` is the canonical, low-overhead way to get per-parameter gradient stats — preferred over backward hooks for this use case.
- **PyTorch Lightning callbacks** expose `on_train_batch_start/end`, `on_before_optimizer_step`, `on_after_backward`, `on_save_checkpoint`, `on_load_checkpoint`, etc. `Trainer.fit(ckpt_path=...)` resumes with full state. LightningDataModule + Trainer is the right mounting point.
- **HuggingFace TrainerCallback** is documented as read-only — callbacks cannot mutate the training loop. `TrainerControl.should_training_stop = True` is the supported escape hatch. lensml's pattern: callback signals stop → external orchestrator applies fix → restart Trainer with `resume_from_checkpoint`.
- **libcst** preserves comments and whitespace on round-trip. CSTTransformer + libcst.matchers is the right API. AddImportsVisitor handles imports cleanly. Gotchas exist for very new syntax but our target patterns (function calls, kwargs, simple statements) are rock solid.
- **SSE via sse-starlette** is the right transport for one-way server→browser streams of training metrics. Native EventSource auto-reconnects. WebSocket would be overkill.
- **Free LLM tiers exist** — Google AI Studio, Groq, Cerebras, OpenRouter, Mistral all have functional free tiers with tool use. Exact rate limits drift; verify on Day 1 of build.
- **AutoTrainer's actual scope** — Keras-only, fixed-rule, in-memory model mutation. Properly credit it; don't dismiss it.
- **TrainCheck** — invariant inference, detection only, no auto-repair.

### Medium confidence (verify on Day 1, have fallback)

- **Free tier exact numbers as of build start.** Gemini cut limits in Dec 2025; expect more cuts. Plan to use 3+ providers in rotation.
- **Ollama on 4GB VRAM.** Honest answer: 4GB will run a 4B-class model at low quality. Useful for tool-calling? Probably not reliably. Plan A: cloud-only with deterministic JSONL replay for offline demo. Plan B: Ollama on CPU with 32GB RAM running Qwen2.5-Coder-7B at slow but acceptable latency for the "privacy mode" demo screen.
- **Subprocess restart for HF Trainer.** Restarting HF Trainer with `resume_from_checkpoint=` is documented but the in-memory state handoff (RNG, dataloader sampler) has known edge cases. Test early.
- **Compute fits in free Colab/Kaggle.** Estimated math in Section 6 says yes; verify with one full bench run end of Week 3.

### Low confidence (treat as risks, see Section 11)

- **Round 1 reviewer attention to benchmark depth.** Hackathon Round 1 reviewers may not engage with a 48-bug benchmark. The video is what gets you to Round 2. The benchmark is what wins Round 3 in Tokyo.
- **Cross-provider tool-call schema portability.** LiteLLM normalizes most things, but some provider quirks slip through (Gemini description-sensitivity, Groq strictness). Build with one provider first; add fallbacks later.

---

## 3. Architecture

### 3.1 Process topology

```
┌─────────────────────────┐         ┌────────────────────────────┐
│   Training subprocess    │         │     FastAPI server          │
│   (user's train.py)      │         │     (lensml.server)         │
│                          │         │                              │
│   lensml.watch() hooks   │  ZMQ    │   - Aggregates metrics      │
│   - forward hooks        │ PUB/SUB │   - Runs LLM agent loop     │
│   - param.grad reads     │ ──────► │   - Manages git + checkpts  │
│   - optimizer.step wrap  │         │   - SQLite event log        │
│   - checkpoint trigger   │         │   - Holds approval state    │
│   - signal-driven exit   │ ◄───── │                              │
└─────────────────────────┘  signal └──────────┬──────────────────┘
                                                │ SSE + REST
                                                ▼
                                    ┌────────────────────────┐
                                    │   Browser dashboard    │
                                    │   (Next.js 15)         │
                                    │   - uPlot live charts  │
                                    │   - Agent trace pane   │
                                    │   - Diff viewer        │
                                    │   - Approve/Reject UI  │
                                    └────────────────────────┘
```

**Why three processes:**
- Trainer in its own subprocess keeps the user's training code clean and isolated. Restart-to-resume is straightforward.
- FastAPI server holds long-lived state (event log, agent context, git refs).
- Dashboard is a static Next.js export — connects via SSE to FastAPI, no Node server needed at runtime.

**Why ZMQ for trainer↔server:**
- pyzmq PUB/SUB is ~5 lines of code, survives either side restarting, no schema fights, microsecond latency.
- Alternative considered: AF_UNIX socket + msgpack. Slightly simpler dependency footprint but more boilerplate. Not worth the saving.
- Inproc fallback for testing: stdin/stdout JSONL when ZMQ is unavailable.

**Why SSE for server↔browser:**
- One-way stream of training ticks (10–50/sec) is exactly SSE's sweet spot. Native EventSource auto-reconnects on disconnect.
- WebSocket gains nothing here; loses native reconnection.
- Approval clicks go via REST POST (separate concern, doesn't need streaming).

### 3.2 Module boundaries

```
lensml/                      # Apache-2.0 core
├── core/                    # Instrumentation primitives
│   ├── hooks.py             # forward/grad hook installers
│   ├── telemetry.py         # metric extraction (loss, grad norms, etc.)
│   ├── transport.py         # ZMQ publisher
│   └── signals.py           # SIGUSR1 handler for pause/checkpoint
├── integrations/
│   ├── pytorch.py           # lensml.watch(model, optimizer, dataloader)
│   ├── lightning.py         # LensmlCallback for pl.Trainer
│   └── transformers.py      # LensmlTrainerCallback for HF Trainer
├── agent/
│   ├── loop.py              # ReAct loop (~80 lines)
│   ├── tools.py             # tool registry (12 tools)
│   ├── memory.py            # observation masking + diagnosis state
│   ├── confidence.py        # self-consistency at N=3
│   └── prompts/             # system prompts per task
├── fixes/
│   ├── base.py              # FixTemplate base class
│   ├── grad_clip.py         # AddGradientClipping
│   ├── lower_lr.py          # LowerLearningRate
│   ├── replace_activation.py
│   ├── add_normalize.py
│   ├── insert_eval_mode.py
│   ├── kaiming_init.py
│   ├── dataloader_workers.py
│   └── add_lr_scheduler.py
├── server/
│   ├── app.py               # FastAPI app
│   ├── sse.py               # /api/runs/{id}/metrics, /agent
│   ├── api.py               # REST endpoints (approve, reject, undo)
│   ├── git.py               # branch, commit, reset
│   ├── checkpoint.py        # save/load model+opt+rng+sampler
│   └── store.py             # SQLite event log
├── llm/
│   ├── client.py            # LiteLLM wrapper with failover
│   └── routing.py           # task → provider mapping
├── bench/                   # lensml-bench
│   ├── fixtures/            # 48 broken notebooks
│   ├── runner.py            # subprocess orchestrator
│   ├── grader.py            # per-bug deterministic checks
│   └── report.py            # markdown + json output
├── cli.py                   # lensml watch, lensml bench, lensml replay
└── __init__.py

dashboard/                   # Next.js 15 App Router
├── app/
├── components/
├── lib/
└── public/

ee/                          # Enterprise license, future paid features
└── README.md                # placeholder for now

tests/
docs/                        # mkdocs-material
.github/
  workflows/
  ISSUE_TEMPLATE/
README.md
LICENSE                      # Apache-2.0
LICENSE-EE
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
CHANGELOG.md
pyproject.toml               # uv-managed
```

### 3.3 State management

**SQLite event log** (`~/.lensml/runs/{run_id}/events.db`):
- `events` table: `(id, ts, run_id, kind, payload_json)` — every metric tick, every agent action, every approval. Append-only.
- `runs` table: `(id, started_at, ended_at, script_path, git_branch, status)`.
- `decisions` table: `(id, run_id, ts, fix_template, diff_text, approved, applied_at)`.

**Checkpoint storage** (`~/.lensml/runs/{run_id}/checkpoints/{step}/`):
- `model.pt` (state_dict)
- `optimizer.pt`
- `scheduler.pt` if present
- `rng.pt` (CPU + CUDA + numpy + python random)
- `sampler.pt` (DataLoader sampler state — best effort)
- `meta.json` (epoch, global_step, wall_time)

**Git integration:**
- On `lensml.watch()` start: ensure user is in a git repo (else create `.lensml-shadow` repo for the script's directory). Create branch `lensml/session-{ts}`.
- On approved fix apply: commit with message `lensml: {fix_template} (run {run_id} step {step})`.
- On reject: no-op.
- On undo: `git reset --hard HEAD~1`.
- On session end: branch persists; user can merge or discard manually.

---

## 4. Pathology catalogue — the 8 we ship in v1

Ranked by detect-clarity × fix-safety × demo-impact. Each has a detector class, a libcst fix template, and a fixture in lensml-bench.

| # | Pathology | Detection signal | Fix template | Demo-friendly? |
|---|---|---|---|---|
| 1 | **Exploding gradients → NaN** | Global grad-L2 > 10× rolling mean for ≥3 steps OR `loss.isnan()` | Insert `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `optimizer.step()` | ⭐⭐⭐⭐⭐ visceral, <10s |
| 2 | **LR too high (oscillating loss)** | Sign-flip rate of Δloss > 0.6 over 50 steps | Replace `lr=L` → `lr=L*0.3` in optimizer call | ⭐⭐⭐⭐⭐ before/after curve |
| 3 | **Dead ReLU on small MLP** | Forward hook: dead fraction > 0.5 sustained | Replace `nn.ReLU()` → `nn.LeakyReLU(0.01)` | ⭐⭐⭐⭐ animated meter |
| 4 | **Missing input normalization** | First-batch input mean ∉ [-3, 3] for image tensor | Insert `transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])` in transform pipeline | ⭐⭐⭐⭐ accuracy jumps |
| 5 | **DataLoader bottleneck** | GPU idle fraction > 30% sustained; `num_workers=0` confirmed | Set `num_workers=min(8, os.cpu_count())`, `pin_memory=True`, `persistent_workers=True` | ⭐⭐⭐ throughput 3× |
| 6 | **`model.eval()` not called at val** | At val time, walk modules; assert BN/Dropout `module.training==False`; if any True → flag | Insert `model.eval()` at val loop entry, `model.train()` after | ⭐⭐⭐⭐ val acc doubles |
| 7 | **Bad init (loss-at-init wrong)** | Step 0: K-class CE expected ≈ `log(K)`; observed > 2× expected | Add `model.apply(_init)` with Kaiming for ReLU layers | ⭐⭐⭐ Karpathy classic |
| 8 | **Vanishing gradients (deep sigmoid)** | Per-layer grad norm: deepest < 1e-7 while shallowest normal | Replace `nn.Sigmoid()` → `nn.GELU()` + insert `nn.LayerNorm` | ⭐⭐⭐⭐ bar chart drama |

**For demo video, use 1, 3, 4, 6** — they hit four distinct telemetry channels (grad norms, activations, input stats, module.training flag) and four distinct AST rewrites in under 60 seconds combined.

**Cut from v1 (move to v0.2 roadmap):**
- Label leakage (slow signal, dangerous to auto-fix)
- Train/val contamination
- Distributed sync bugs (no DDP support yet)
- Mixed-precision (subtle, low demo value)
- Tokenization bugs (NLP-specific, semantic)

---

## 5. Agent design

### 5.1 Loop architecture

Single-agent ReAct loop, hand-rolled in ~80 lines. No LangChain, no LangGraph, no Pydantic AI. Reasons: (1) every framework adds debug surface area, (2) solo + 30 days = control matters more than abstraction, (3) the loop is genuinely small.

Pseudocode:

```python
# lensml/agent/loop.py
async def agent_loop(run_id, llm_client, tool_registry, event_log):
    state = DiagnosisState()  # persistent Pydantic object
    history = []  # last N=20 (action, observation) pairs
    while not should_exit():
        anomaly = await wait_for_anomaly(run_id)
        if not anomaly:
            continue
        prompt = build_prompt(state, history, anomaly)
        for cycle in range(MAX_CYCLES):  # cap at 8 cycles per anomaly
            response = await llm_client.complete(prompt, tools=tool_registry)
            if response.tool_calls:
                for call in response.tool_calls:
                    obs = await tool_registry.execute(call)
                    history.append((call, obs))
                    history = compact(history)  # mask older obs
                prompt = update_prompt(state, history)
            elif response.has_diagnosis:
                state.update(response.diagnosis)
                if response.proposed_fix:
                    await present_fix_for_approval(response.proposed_fix)
                break
        await event_log.commit(state)
```

### 5.2 Tools (the diagnostic vocabulary)

12 tools, all read-only except `propose_fix`. Each returns summary stats (≤2KB), never raw tensors.

| Tool | Returns |
|---|---|
| `inspect_loss_curve(window, smooth)` | `{values, slope, plateau_score, nan_count, std}` |
| `inspect_gradient_flow(layer_pattern, last_n_steps)` | `{layer: {grad_norm_mean, max, dead_fraction}}` |
| `inspect_activations(layer, sample_size)` | `{mean, std, saturated_fraction, dead_fraction}` |
| `inspect_data_batch(n)` | `{shapes, dtypes, label_distribution, has_nan, value_range}` |
| `inspect_dataloader_throughput()` | `{batches_per_sec, gpu_idle_fraction, worker_count}` |
| `inspect_optimizer_state()` | `{type, lr, momentum_norm, second_moment_norm, step}` |
| `inspect_lr_schedule()` | `{current_lr, schedule_type, steps_remaining}` |
| `inspect_model_summary()` | `{param_count, trainable, layer_types, has_batchnorm, has_dropout}` |
| `inspect_train_val_gap(metric)` | `{train, val, gap, gap_trend}` |
| `inspect_memory()` | `{gpu_alloc_gb, peak, oom_risk}` |
| `read_training_script(start_line, end_line)` | source text |
| **`propose_fix(template, params, confidence, rationale)`** | emits ApprovalRequest, never applies |

`propose_fix` is the *only* write-side tool. It emits a structured request to the approval queue. The actual git+AST+restart sequence is handled by the server, not the agent.

### 5.3 Memory and context

Three layers (informed by JetBrains' Dec 2025 context-rot findings):

1. **Working memory:** last 20 (action, observation) pairs verbatim.
2. **Sliding observations:** older observations replaced with `[masked: tool_name@step_X]`. Agent re-calls if needed.
3. **Persistent state:** structured `DiagnosisState` object — `{current_hypotheses, confirmed, ruled_out, proposed_fixes}`. Always in system prompt prefix.

No vector DB. Bounded session. Simple wins.

### 5.4 Confidence calibration

Self-consistency at N=3, temperature 0.7. If ≥2/3 agree on the pathology label, MEDIUM confidence. 3/3 → HIGH. <2/3 → LOW, present as "I noticed X but I'm not sure" without a fix proposal.

Never apply fixes below MEDIUM without explicit user override.

### 5.5 LLM routing

Per-task provider mapping. Designed to fit free tiers across a 5-minute live demo without hitting limits.

| Task | Frequency | Primary | Secondary | Tertiary |
|---|---|---|---|---|
| Anomaly trigger classification | every 50 steps | Groq Llama 3.1 8B | Cerebras Llama 3.1 8B | Gemini 2.5 Flash-Lite |
| Pathology diagnosis | per anomaly (rare) | Gemini 2.5 Pro | Cerebras GPT-OSS 120B | OpenRouter DeepSeek :free |
| Code patch generation | per fix | OpenRouter Qwen3-Coder :free | Mistral Codestral | Gemini 2.5 Pro |
| Verification (does this fix make sense?) | per fix | Cerebras GPT-OSS 120B | Gemini 2.5 Pro | — |

LiteLLM handles failover. On any 429/503/rate-limit, retry with next provider. Log every routing decision to event log for post-hoc analysis.

**Quota math for live demo:**
- 5-minute demo with 3 anomalies and 3 fixes
- Anomaly classification: ~10 calls (light prompts) → well within Groq's 30 RPM
- Diagnosis: 3 calls (Gemini 2.5 Pro free is 5 RPM/100 RPD — fine)
- Patch generation: 3 calls (OpenRouter 20 RPM)
- Verification: 3 calls (Cerebras 30 RPM)

Total: ~19 LLM calls in 5 minutes across 4 providers. Comfortable.

### 5.6 Local privacy mode (the killer differentiator)

**Honest assessment for 4GB VRAM:** loading a tool-calling LLM into 4GB GPU is rough. Qwen2.5-Coder-7B-Instruct at Q4_K_M is ~4.5GB. It will not fit cleanly. Options:

- **Plan A (recommended):** Privacy mode runs Ollama on **CPU + 32GB system RAM** with Qwen2.5-Coder-7B-Instruct Q4. Latency ~5-15s per response. Acceptable for "watch your training while you sleep" — unacceptable for live demo.
- **Plan B:** Privacy mode toggle exists in UI but uses smaller 3B model (Qwen2.5-Coder-3B or Llama 3.2 3B). Faster but worse tool calling. Test on Day 22.
- **Plan C:** Privacy mode is documented + scaffolded but the live demo runs cloud-mode. Privacy mode shipped as "works, slower" with docs explaining the tradeoff.

Pick Plan C for Round 1. Privacy mode is *demonstrably present* (toggle works, model loads, one canned interaction works) but the speed-of-light demo uses cloud. Honest story for judges: "We support local-only mode for privacy-sensitive deployments. It's slower because we're optimizing for what runs on commodity hardware. Cloud mode is what you saw."

---

## 6. lensml-bench — the benchmark

### 6.1 Structure

**12 pathology classes × 4 examples = 48 broken notebooks.** Each example is one Python file (not a notebook — easier CI) with:

- `bug.py` — broken training script
- `fix.diff` — the canonical patch
- `grade.py` — deterministic checks: (a) does patched code run without error? (b) does loss converge below threshold T within N steps? (c) does test metric exceed threshold M?
- `meta.json` — pathology class, expected difficulty, host environment

The 12 classes are the 8 from Section 4 plus 4 stretch classes (label leakage, train/val contamination, mixed precision NaN, learning rate schedule mismatch) — included for evaluation breadth even if v1 doesn't auto-fix them. The agent can score "diagnosis correct, fix not yet supported" and we report this honestly.

### 6.2 Host environments and compute

| Environment | Model | Dataset | Per-run time on T4 | Examples |
|---|---|---|---|---|
| Tiny | Small MLP/CNN | MNIST | 30-90 sec | 16 |
| Small | ResNet-18 | CIFAR-10 (50k subset) | 3-8 min | 12 |
| Medium | DistilBERT | SST-2 (small split) | 8-20 min | 12 |
| Large | GPT-2 small | WikiText-2 (small) | 15-30 min | 8 |

**Compute budget verification:**
- Per fixture: original broken run + one fixed-run-after-patch ≈ 2× per-run time
- 48 fixtures × avg 8 min × 2 = ~13 GPU-hours total
- Plus baseline LLM-only runs (no agent) for credibility: another ~13 GPU-hours
- Plus one full re-run for final report: ~13 GPU-hours
- **Total: ~40 GPU-hours**

Free tier supply:
- Colab free: ~12 hours T4/day on a single account, throttled but usable
- Kaggle: 30 hours T4 per week (×2 GPUs available simultaneously)
- Math: Kaggle alone gives 30 hr/week × 4 weeks = 120 GPU-hours
- **Conclusion: we have ~3× headroom. Comfortable.** Allocate Week 4 to bench runs.

### 6.3 Baselines

Three baselines, each tested on the same 48 fixtures:

1. **LLM-only baseline** — single-shot prompt: "Here's a training script and its loss curve. Diagnose the issue and propose a fix." Test on Gemini 2.5 Pro, Groq Llama 3.3 70B, Cerebras GPT-OSS 120B, OpenRouter Qwen3-Coder. **This is the credibility lynchpin** — proves lensml's value over "just ask the LLM."
2. **Diagnosis-only ablation** — lensml without the AST patch step (just human-readable suggestion).
3. **Full lensml** — diagnosis + patch + apply.

### 6.4 Metrics

- **Top-1 diagnosis accuracy** — does the agent's primary hypothesis match the canonical bug class?
- **Top-3 diagnosis accuracy** — is the canonical class in the agent's top 3 hypotheses?
- **pass@1 fix rate** — does the first applied patch make `grade.py` pass?
- **pass@3 fix rate** — does at least one of three sampled patches pass?
- **Time-to-diagnose** — wall-clock from anomaly detection to diagnosis emission.
- **Tokens-to-diagnose** — total LLM tokens consumed per diagnosis.
- **Cost-to-diagnose** — $0 on free tiers (the killer slide).

### 6.5 Output

`bench/results/{date}/report.md` + `report.json`. Includes:
- Per-class confusion matrix (where the agent confuses pathologies)
- Latency histogram
- Cost breakdown (always $0 — make this a slide)
- "What doesn't work yet" section — honest about which classes lensml fails on. Japanese judges will respect this enormously.

---

## 7. Dashboard UX

### 7.1 Layout (desktop ≥ 1024px)

```
┌─────────────────────────────────────────────────────────────────┐
│  lensml  ▸  run_a8f3 ▸ train.py @ step 3,247        ⚙ EN/JA 🌙  │
├──────────────────────────┬──────────────────────────────────────┤
│                          │                                       │
│   LIVE METRICS           │    AGENT REASONING                   │
│   ┌──────────────────┐   │   ┌──────────────────────────────┐  │
│   │ loss curve       │   │   │ ◯ inspect_loss_curve         │  │
│   │ [uPlot, 60fps]   │   │   │   slope: -0.001 (plateau)    │  │
│   └──────────────────┘   │   │ ◯ inspect_gradient_flow      │  │
│   ┌──────────────────┐   │   │   layer.5: norm 47.2 (10×)   │  │
│   │ grad-L2 norm     │   │   │ ✦ Hypothesis: exploding grad │  │
│   └──────────────────┘   │   │   confidence: HIGH (3/3)     │  │
│   ┌──────────────────┐   │   │ ✦ Proposing fix...           │  │
│   │ activation stats │   │   └──────────────────────────────┘  │
│   └──────────────────┘   │                                       │
│                          │    DIFF PREVIEW                       │
│                          │   ┌──────────────────────────────┐  │
│                          │   │ + import torch.nn.utils      │  │
│                          │   │   loss.backward()            │  │
│                          │   │ + clip_grad_norm_(...)       │  │
│                          │   │   optimizer.step()           │  │
│                          │   │ [y]Accept [n]Reject [u]Undo  │  │
│                          │   └──────────────────────────────┘  │
└──────────────────────────┴──────────────────────────────────────┘
```

### 7.2 Mobile (<640px)

Single-column tabbed view: Metrics / Agent / Diff. Approve/Reject buttons sticky to bottom. Test on iPhone Safari and 1080p projector.

### 7.3 Critical UX details

- **Vertical line on loss curve at every intervention.** This is the visual hook. One glance shows "agent intervened here, here, here."
- **Status pill at top:** rotating verb ("Watching… Diagnosing… Patching… Resumed."). Pulsing dot. Demo money shot.
- **Keyboard shortcuts:** `y` accept, `n` reject, `u` undo, `?` help, `j/k` navigate hunks, `⌘Enter` apply all approved.
- **Reasoning trace streams token-by-token** (16ms batch flushes). Tool-call cards "pop" in with 150ms fade.
- **Demo replay mode:** `lensml replay <jsonl>` runs a deterministic pre-recorded session. **Use this for the live demo.** Live training during a live demo is a graveyard.
- **Dark mode default.** Every ML tool is dark. Signals familiarity.

### 7.4 Stack

- Next.js 15 App Router + React 19
- Tailwind + shadcn/ui (judges recognize)
- uPlot for live charts (48KB, 60fps with thousands of points)
- react-diff-viewer-continued (the maintained fork)
- Native EventSource (auto-reconnect)
- Framer Motion for tool-card pop-ins
- Static export, deployed to Vercel (free)

---

## 8. The 30-day calendar

Each day has: morning task, afternoon task, end-of-day deliverable, risk flag, fallback if it overflows. 6–8 hours/day, 6 days/week (one rest day per week).

### Week 1 — Foundation (May 8–14)

**Day 1 (Thu, May 8) — Setup & spike**
- Morning: pyproject.toml with uv, Python 3.11, install pyzmq, fastapi, sse-starlette, libcst, litellm. Initialize git, private GitHub repo.
- Afternoon: Smallest possible spike — train.py running MNIST, hand-write a forward hook that prints activation mean every step. Confirm <5% overhead.
- EOD: hooks-spike branch with working hook example.
- Risk: low. Fallback: none needed.

**Day 2 — Telemetry layer**
- Morning: `lensml.core.hooks` — install forward hooks on all leaf modules, capture activation mean/std/dead-fraction. Use `param.grad` post-backward for grad norms.
- Afternoon: `lensml.core.transport` — pyzmq PUB publisher. Trainer publishes JSONL events to ipc://lensml-{pid}.sock.
- EOD: train.py emits structured metric events to ZMQ.
- Risk: medium (ZMQ on different platforms). Fallback: stdin/stdout JSONL.

**Day 3 — FastAPI server skeleton**
- Morning: `lensml.server.app` — FastAPI with two SSE endpoints (`/api/runs/{id}/metrics`, `/api/runs/{id}/agent`). ZMQ SUB → SSE bridge.
- Afternoon: SQLite event log, `lensml.server.store`. Schema migration with raw SQL (no SQLAlchemy needed).
- EOD: `curl -N http://localhost:8000/api/runs/test/metrics` shows live event stream.
- Risk: medium (SSE buffering on some servers). Fallback: WebSocket.

**Day 4 — Dashboard scaffold**
- Morning: Next.js 15 app, Tailwind + shadcn init, three-pane layout shell.
- Afternoon: uPlot live loss chart wired to SSE endpoint. EventSource auto-reconnect.
- EOD: dashboard renders live loss curve from a running training script.
- Risk: medium (React 19 + Next.js 15 + uPlot integration). Fallback: vanilla HTML+JS dashboard if React fights.

**Day 5 — `lensml.watch()` API**
- Morning: One-line API. `lensml.watch(model, optimizer, dataloader)` installs everything. RemovableHandle teardown via context manager or explicit `lensml.unwatch()`.
- Afternoon: Monkey-patch `optimizer.step` for pre-step gradient inspection. Test on raw PyTorch + Lightning model + HF Trainer model (just to confirm hooks survive the wrapper).
- EOD: end-to-end raw PyTorch flow: script → hooks → ZMQ → server → SSE → dashboard chart.
- Risk: medium (Lightning/HF Trainer wrapping). Fallback: raw PyTorch only this week, integrations next week.

**Day 6 — First detector**
- Morning: `lensml.detectors.exploding_grad` — the canonical first one. Threshold: global grad-L2 > 10× rolling mean for 3 steps OR `loss.isnan()`.
- Afternoon: Detector emits structured `Anomaly` event to ZMQ. Server logs to event store. Dashboard shows "⚠ exploding gradient detected" toast.
- EOD: end-to-end anomaly detection pipeline working.
- Risk: low.

**Day 7 — Polish + rest**
- Morning: Clean up, docstrings, basic CI (uv + ruff + pytest matrix).
- Afternoon: REST.
- EOD: Week 1 complete. Foundation ready.

### Week 2 — Agent + AST + first end-to-end loop (May 15–21)

**Day 8 — LiteLLM client**
- Morning: `lensml.llm.client` — LiteLLM wrapper with provider failover. Config in `~/.lensml/config.toml` (provider keys, routing table).
- Afternoon: Hello-world tool call with Gemini 2.5 Pro free. Confirm tool schema works.
- EOD: agent can call one tool and get a structured response.
- Risk: medium (Gemini schema strictness). Fallback: pin to Groq if Gemini fights.

**Day 9 — Tool registry**
- Morning: All 12 tools as Pydantic schemas + Python functions reading from event store / live state.
- Afternoon: Tool execution plumbing — agent emits tool_call → server runs tool → returns result → loops.
- EOD: agent can chain 3-5 tool calls and produce a structured diagnosis.
- Risk: medium. Fallback: cut tools to top 6 if schema portability is brutal.

**Day 10 — Agent loop**
- Morning: ReAct loop in ~80 lines. Memory layers (working, sliding, persistent state).
- Afternoon: Self-consistency at N=3 for confidence scoring.
- EOD: full agent loop running on the exploding-grad scenario, emitting structured diagnosis with confidence.
- Risk: high (this is the hardest day of the agent build). Fallback: skip self-consistency for v1, ship single-shot.

**Day 11 — libcst transformers**
- Morning: `lensml.fixes.base` — FixTemplate ABC. `grad_clip.py` — AddGradientClipping transformer.
- Afternoon: 3 more transformers: lower_lr, replace_activation, add_normalize.
- EOD: 4 working AST transformers, each with a unit test on a known input.
- Risk: medium (libcst learning curve). Fallback: simpler text-based search/replace if libcst overflows.

**Day 12 — Approval flow + git integration**
- Morning: `propose_fix` tool emits ApprovalRequest. Server holds queue. Dashboard shows diff via react-diff-viewer-continued.
- Afternoon: Approve → libcst apply → git commit → trigger restart. Reject → discard. Undo → `git reset --hard HEAD~1`.
- EOD: end-to-end: detect bug → diagnose → diff in dashboard → user presses `y` → patched file committed to git branch.
- Risk: medium. Fallback: skip auto-restart, surface "patch applied, restart manually" message.

**Day 13 — Checkpoint + restart**
- Morning: `lensml.server.checkpoint` — save model/opt/scheduler/RNG/sampler. SIGUSR1 in trainer triggers checkpoint + clean exit.
- Afternoon: Subprocess supervisor in server — on apply, kill trainer, restart with `LENSML_RESUME=<ckpt_path>`. Trainer's `lensml.watch()` on init checks env var and loads checkpoint.
- EOD: full magic moment loop on raw PyTorch + exploding-grad fixture.
- Risk: high (checkpoint+restart has many edge cases — RNG, sampler state). Fallback: skip restart, just commit the patch and tell user to restart manually.

**Day 14 — Polish + rest**
- Morning: Bug fixes from week 2, write a 2-paragraph "what week 2 produced" log.
- Afternoon: REST.
- EOD: Week 2 complete. Magic moment works on raw PyTorch.

### Week 3 — Lightning + HF + 4 more detectors + UX polish (May 22–28)

**Day 15 — Lightning callback**
- Morning: `lensml.integrations.lightning.LensmlCallback`. Hooks: `on_train_batch_end`, `on_before_optimizer_step`, `on_save_checkpoint`, `on_load_checkpoint`.
- Afternoon: Test on a Lightning training script (CIFAR-10 + small CNN).
- EOD: Lightning integration works end-to-end, including approval and restart.
- Risk: medium. Fallback: callback-only, no auto-restart for Lightning.

**Day 16 — HF TrainerCallback**
- Morning: `lensml.integrations.transformers.LensmlTrainerCallback`. Hooks: `on_step_end`, `on_pre_optimizer_step` (transformers ≥ 4.40).
- Afternoon: External orchestrator pattern — callback sets `TrainerControl.should_training_stop = True`, server applies fix, restarts Trainer with `resume_from_checkpoint=`.
- EOD: HF Trainer integration works on DistilBERT/SST-2 (Colab T4).
- Risk: high (subprocess restart of HF Trainer with state continuity). Fallback: ship as "v0.2 — works in callback mode, restart is manual."

**Day 17 — Detectors 2-4**
- Morning: dead_relu, missing_normalize detectors.
- Afternoon: lr_too_high, model_eval_not_called detectors.
- EOD: 5 detectors total working.
- Risk: low.

**Day 18 — Detectors 5-8**
- Morning: dataloader_bottleneck, bad_init detectors.
- Afternoon: vanishing_gradients detector. Plus 4 more libcst fix templates.
- EOD: 8 detectors, 8 fix templates. Full v1 pathology coverage.
- Risk: low-medium (fixtures need to actually trigger detectors reliably).

**Day 19 — UX polish**
- Morning: Status pill, vertical lines on loss curve at intervention points, agent trace pop-in animations.
- Afternoon: Keyboard shortcuts, command palette (`cmdk`), JA toggle scaffold (just nav + buttons translated).
- EOD: dashboard feels production-grade.
- Risk: low.

**Day 20 — Replay mode**
- Morning: `lensml replay <session.jsonl>` — deterministic playback for demos.
- Afternoon: Record three canonical sessions: exploding-grad, dead-relu, missing-normalize. Save JSONL.
- EOD: bulletproof live demo path.
- Risk: low. **This is critical for hackathon — do not skip.**

**Day 21 — Polish + rest**
- Morning: Privacy mode toggle (Plan C — scaffolded, one canned interaction works).
- Afternoon: REST.
- EOD: Week 3 complete. All integrations + polish done.

### Week 4 — lensml-bench, video, submission package (May 29 – Jun 4)

**Day 22 — Bench fixtures**
- Morning: 12 fixtures (Tiny env: MNIST/MLP, all 12 pathology classes covered).
- Afternoon: 12 fixtures (Small env: CIFAR-10/ResNet-18).
- EOD: 24 fixtures ready.
- Risk: medium (fixtures need to actually demonstrate the pathology cleanly).

**Day 23 — Bench fixtures cont. + grader**
- Morning: 12 fixtures (Medium env: DistilBERT/SST-2).
- Afternoon: 12 fixtures (Large env: GPT-2 small/WikiText). Grader script.
- EOD: 48 fixtures + grader complete.
- Risk: medium. Fallback: cut Large env, ship 36 fixtures.

**Day 24 — Bench runner**
- Morning: `lensml bench run` orchestrator. Spawns subprocesses, runs grader, aggregates results.
- Afternoon: First end-to-end bench run on Tiny env (16 fixtures). Verify it works.
- EOD: bench infrastructure tested.
- Risk: medium.

**Day 25 — Bench runs (Colab/Kaggle)**
- Morning: Bench on Tiny + Small (Colab T4 session 1). Pass/fail counts logged.
- Afternoon: Bench on Medium (Kaggle T4 session 1). Plus baseline LLM-only runs in parallel.
- EOD: ~32 fixtures benched + baselines done.
- Risk: high (Colab/Kaggle disconnects). Fallback: re-run failed sessions.

**Day 26 — Bench runs continued + report**
- Morning: Large env bench (Kaggle T4 session 2). 16 fixtures.
- Afternoon: Generate `bench/results/2026-06-02/report.md` — confusion matrix, latency, cost ($0), failure analysis.
- EOD: Complete benchmark report.
- Risk: medium.

**Day 27 — Demo video**
- Morning: Record VHS demo tapes (3 GIFs: install, watch, intervene).
- Afternoon: Record 90-second Round 1 video. Multiple takes.
- EOD: video uploaded to YouTube unlisted + draft.
- Risk: medium (video always takes 2× expected).

**Day 28 — README + landing page**
- Morning: README with hero GIF, install command, comparison table, benchmark results table.
- Afternoon: Landing page on lensml.dev (Next.js export to Vercel). Domain registration if needed.
- EOD: public-ready repo + landing page.
- Risk: low-medium.

### Final stretch — Submission + launch (Jun 5–7)

**Day 29 (Fri, Jun 5) — Submission package**
- Morning: Fill Unstop submission form. Attach video, README link (still private), benchmark report.
- Afternoon: Final review. Test install on a fresh VM/container.
- EOD: submission ready to send.
- Risk: low.

**Day 30 (Sat, Jun 6) — Final polish + launch prep**
- Morning: Last-pass bug fixes. Verify all install paths work.
- Afternoon: Draft the launch tweet thread. Prepare HN Show post (don't post yet).
- EOD: everything queued for Sunday launch.
- Risk: low.

**Day 31 (Sun, Jun 7) — SUBMIT + GO PUBLIC**
- Morning: Submit on Unstop before noon IST.
- Afternoon: Toggle GitHub repo public. Push final commit. First tweet thread. Ship to LinkedIn. Hold HN/Reddit for after Round 2 results.
- EOD: SUBMITTED. PUBLIC. DONE.

### Cut points if you fall behind

- **End of Week 1:** if hooks/SSE not solid → defer dashboard polish, ship CLI-only for v1, dashboard in v0.2.
- **End of Week 2:** if magic moment not working → ship "diagnosis-only" mode (suggest patch as text, no auto-apply). Still wins on novelty.
- **End of Week 3:** if HF Trainer not working → drop it, document as v0.2. Lightning + raw PyTorch is enough.
- **End of Week 3:** if bench fixtures look hard → cut to 24 (Tiny + Small only).
- **End of Day 26:** if benchmark results are bad → publish methodology + partial results, frame as "lensml-bench v0.1 — preliminary results."

---

## 9. Round 1 submission package

### 9.1 90-second video script

| Time | Beat | Visual | Voiceover |
|---|---|---|---|
| 0–8s | Hook | Real-looking loss curve diving into NaN, red alert flash, blank chair at desk | "10 hours into training. Loss spiked. Gradient exploded. You're asleep." |
| 8–15s | Reveal | lensml logo for 1.5s, then dashboard | "Meet lensml." |
| 15–55s | Magic moment | Split screen: chart with vertical intervention line / agent reasoning streams "inspect_loss_curve... inspect_gradient_flow... Hypothesis: exploding gradients. Confidence HIGH." / diff appears below. Hand presses `y`. Loss curve resumes descending. | "lensml watches your training. Diagnoses why loss is misbehaving. Proposes a patch. One keystroke applies it via AST rewrite, and resumes from your last checkpoint." |
| 55–75s | Result | Side-by-side: original run dies / lensml run converges. "$340 GPU saved." counter. | "8 hours saved. $340 of GPU rescued. Run converged to 92% accuracy." |
| 75–88s | Outro | github.com/rupeshbharambe24/lensml + tagline | "lensml. Open source. Free LLM tiers. Supervises so you can sleep." |
| 88–90s | Logo | FAR AWAY hackathon logo | (silence) |

Voiceover: slow English, no US idioms (Round 3 judges will rewatch). Record with a decent mic — bad audio kills good video.

Two versions: 1080p60 master, 720p <25MB for slow upload portals.

### 9.2 README structure

```markdown
# lensml

> The AI agent that supervises your ML training runs.

[Hero GIF — 8 sec, the magic moment]

[badges: PyPI, Python versions, license, CI, downloads]

## What it does

One paragraph + bullet list.

## 30-second install

```bash
pip install lensml
echo "import lensml; lensml.watch(model, optimizer, dataloader)" >> train.py
lensml dashboard  # opens browser
```

## Why lensml?

[Comparison table: lensml vs W&B vs TensorBoard vs TrainCheck vs AutoTrainer]

## Architecture

[Mermaid diagram]

## Benchmarks

[Table: top-1 / top-3 / pass@1 / pass@3 / time / cost — including LLM-only baseline]

## Roadmap

- v0.2: HF Trainer auto-restart, distributed training (DDP/FSDP)
- v0.3: more pathology classes, custom detector API
- v1.0: enterprise features in `/ee/` (SSO, team dashboards, fine-tuned supervisor models)

## Contributing

## License

Apache-2.0 (core), see LICENSE-EE for enterprise features.
```

### 9.3 Unstop submission form fields

Likely fields (verify on submission day):
- Project name: lensml
- One-liner: An open-source AI agent that supervises ML training runs in real time.
- Long description: ~300 words, lead with the problem (silent training failures cost time and money), the solution (lensml watches + diagnoses + patches with approval), the technical novelty (live instrumentation + LLM agent + AST rewrites + checkpoint resume), and the benchmark stat.
- Demo video URL: YouTube unlisted link
- GitHub URL: rupeshbharambe24/lensml (toggle public morning of submission)
- Tech stack: Python, FastAPI, Next.js, libcst, LiteLLM, PyTorch, PyTorch Lightning, HuggingFace Transformers
- License: Apache-2.0
- Team size: 1

---

## 10. Pitch evolution across rounds

### Round 1 (June 7) — video + README judging
- The 90-second video carries everything. 80% of judging weight, my estimate.
- Benchmark report is the 20% — credibility for reviewers who read deeply.
- Stealth → public on submission day. First tweet 09:00 IST: "Just submitted lensml to FAR AWAY. 30 days, solo, open source. Here's what I built →"

### Round 2 (Delhi, ~mid-June) — 24-hour in-person + live pitch
- Reuse the video as opening. Then live demo on your laptop.
- **Bring the second laptop with cloned repo. Bring the deterministic JSONL replay loaded.** Demo what you rehearsed, not live training.
- 5-minute structure: 60s problem & you, 180s live demo, 45s metrics + roadmap, 15s ask. 6 slides max.
- Pre-bake answers for: "How does this differ from W&B?" / "How do you know the fix is correct?" / "Distributed training?" / "Privacy?"
- Bring the Round 2 magic move: build a *new* detector live (e.g., learning rate schedule mismatch). Even partial completion proves architecture extensibility. Differentiator from teams who can only demo what they pre-built.

### Round 3 (Tokyo, July) — international finals
- Reframe the pitch: drop the "Indian student" framing. Lead with the universal pain.
- 7–10 minute structure: 45s problem (cite TrainCheck and AutoTrainer — Japanese judges value related work), 30s solution, 3 min demo, 90s architecture deep-dive (one slide with pseudocode of the agent loop), 60s benchmark results from lensml-bench, 30s "what doesn't work yet" (limitations + future work — Japanese audiences respect humility), 15s ask.
- JA toggle in dashboard for 3 seconds during demo as a "we built this with Japan in mind" beat.
- Reference Sakana AI, Preferred Networks, RIKEN AIP by name. They're hiring exactly this skill stack.
- Bring printed business cards (meishi). Receive cards with two hands. Speak slowly. No US idioms.

---

## 11. Risk register

Top 10 risks, ranked by P(occurs) × impact, with concrete mitigations.

| # | Risk | P | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **HF Trainer subprocess restart with full state continuity is harder than expected** | high | medium | Day 16 spike. Fallback: ship HF as "callback mode only, restart is manual" — document, don't hide. |
| 2 | **libcst f-string / async / type-hint edge cases break a fix template on a real fixture** | medium | medium | Each transformer has unit tests (Day 11). Add fixtures that exercise these features. Fall back to text search/replace for the one edge case. |
| 3 | **Free LLM tier rate limits hit during a live demo** | medium | high | Use replay mode for live demos. 4 providers in failover. Quota math (Section 5.5) confirms comfort. |
| 4 | **4GB VRAM OOM during local dev** | medium | low | Local dev = MNIST/CIFAR only. Mid-scale benchmarks run on Colab/Kaggle. Already designed for this. |
| 5 | **Cross-provider tool-call schema portability** | medium | high | Day 8 spike with Gemini first. If broken: pin to one provider (Groq Llama 3.3 70B has reliable tool calling) for v1. |
| 6 | **Colab/Kaggle session disconnects during bench runs** | high | low | Idempotent grader. Per-fixture state checkpointing in bench runner. Re-run failed fixtures only. |
| 7 | **Dashboard polish takes 2× expected** | high | medium | Day 19 is dedicated to polish. Cut framer-motion if needed. Ugly but functional > pretty but broken. |
| 8 | **Demo video takes 4 hours instead of 8** | medium | low | Day 27 morning + Day 30 buffer. Worst case: ship a 60-second video with 3 demos instead of 4. |
| 9 | **Round 1 reviewer skims and misses benchmark depth** | high | low | Video carries the case. Benchmark is for Round 2/3 judges. Mention in 5 seconds of voiceover. |
| 10 | **Personal life intrudes (work, illness, family)** | medium | medium | One rest day per week built in. Cut points pre-defined for end of each week. |

### Graceful degradation cascade

If everything goes wrong, the absolute minimum that still wins Round 1:
1. Raw PyTorch only (no Lightning, no HF)
2. 4 detectors, not 8
3. Diagnosis-only (suggest patch as text, no auto-apply)
4. CLI dashboard (text TUI, not browser)
5. 16 bench fixtures (Tiny env only)
6. 60-second video, no Japanese toggle, no privacy mode

This minimum is *still novel* — no one else is shipping diagnostic agents over live training runs. You'd still be in the Round 1 top tier.

---

## 12. Open questions to resolve on Day 1

- [ ] **Free tier numbers as of May 8, 2026** — verify Gemini, Groq, Cerebras, OpenRouter, Mistral against current vendor docs. Build will start at the numbers verified on Day 1.
- [ ] **Domain availability** — `lensml.dev`, `.ai`, `.com`. If taken, alternates: `probeml`, `trainsight`, `scopeml`. Verify Day 1.
- [ ] **GitHub username** — `rupeshbharambe24/lensml` available? Personal repo is fine for stealth.
- [ ] **Latest Lightning, transformers, libcst, fastapi versions** — pin in `pyproject.toml` Day 1 and don't move.
- [ ] **uv vs pip+venv** — uv is faster but adds a dependency on uv being installed. For solo dev: uv. For published package consumers: pip works fine on the published wheel.

---

## 13. Public design doc (condensed) — what goes in DESIGN.md

(Below is the public-facing version. ~900 words. Goes on lensml.dev/design or in DESIGN.md.)

---

# lensml — design

**lensml is an open-source AI agent that supervises your ML training runs.** It hooks into your PyTorch, PyTorch Lightning, or HuggingFace Trainer training loop, watches gradients, activations, loss curves, and dataloader stats in real time, and uses an LLM agent to diagnose pathologies as they emerge. When it finds a fixable bug — exploding gradients, dead ReLUs, missing normalization, model.eval() not called, learning rate issues — it pauses training, generates a code patch via AST-level rewrite, presents the diff in a local browser dashboard, and on your approval applies the fix and resumes training from a checkpoint.

It is not another observability dashboard. W&B and MLflow show you metrics and let you draw conclusions. lensml is *active* — it watches, reasons, intervenes.

## Why now

Three things changed in 2024–2025. First, agentic coding tools (Cursor, Aider, Claude Code) crossed a reliability threshold for code rewriting. Second, long-context LLMs can ingest full training logs and reason about them. Third, the major commercial training trackers — W&B (acquired by CoreWeave, May 2025) and Neptune (acquired by OpenAI, late 2025) — left the neutral, open-source space. lensml occupies that space.

## What's different

The published prior work splits cleanly. **TrainCheck** (OSDI 2025) infers training invariants and detects silent errors — but stops at detection. **AutoTrainer** (ICSE 2021, TDSC 2024) detects and repairs ~8 pathology classes — but is Keras-only, uses fixed rules with no LLM, and operates on in-memory model graphs rather than source code. **Cursor and Aider** rewrite source code — but are blind to running training processes; they cannot pause a 10-hour run, patch it, and resume from checkpoint. **W&B/MLflow/Comet/ClearML/Aim** track and display — but never modify user code.

lensml is the integration of these capabilities into one workflow: live runtime instrumentation, LLM-driven diagnosis with confidence calibration, libcst-level source patching, approval-gated apply, and checkpoint-aware resume.

## Architecture

A user's training script runs in a subprocess with `lensml.watch(model, optimizer, dataloader)` installed. Forward hooks capture activation statistics, post-backward gradient reads capture per-parameter norms, and a monkey-patched `optimizer.step` captures pre-step state. Metrics stream over ZMQ PUB/SUB to a local FastAPI server, which forwards them via Server-Sent Events to a Next.js dashboard.

A single-agent ReAct loop runs in the server. When a detector flags an anomaly (e.g., gradient L2 exceeding 10× rolling mean), the agent receives a structured event and chooses from 12 read-only diagnostic tools (`inspect_loss_curve`, `inspect_gradient_flow`, etc.) to gather evidence. After 2–8 tool calls, it produces a structured diagnosis with self-consistency-calibrated confidence (N=3 sampling). On HIGH confidence, the agent calls `propose_fix` — emitting a structured fix template + parameters, never writing files directly.

The server takes the proposal, runs the corresponding libcst transformer to generate a diff, and displays it in the dashboard. On approval (one keystroke), the server commits the patch to a `lensml/session-{ts}` git branch, signals the trainer to checkpoint, kills the trainer subprocess, and restarts it with `LENSML_RESUME=<ckpt>`. The trainer's `lensml.watch()` checks the env var on init and restores model + optimizer + scheduler + RNG + sampler state.

## What we don't do (yet)

- **Distributed training** (DDP/FSDP/DeepSpeed). v1 is single-GPU. v0.2 roadmap.
- **Auto-fix for label leakage, train/val contamination, mixed precision NaN, tokenization bugs**. Detection-only for v1.
- **Closed-source training loops**. We hook PyTorch internals; if your trainer hides them, lensml can't see them.
- **Production model serving**. lensml is for development training, not deployed inference.

## Privacy

lensml supports a local-only mode: run an Ollama-hosted code model and lensml will route all LLM calls locally. No training code, gradients, or logs leave your machine. Cloud mode (free LLM tiers) is faster but sends abstracted telemetry — never raw model weights or user data.

## Benchmarks

lensml-bench is a 48-fixture suite across 12 pathology classes and 4 host environments (MNIST/MLP, CIFAR/ResNet-18, SST-2/DistilBERT, WikiText/GPT-2). Metrics: top-1 / top-3 diagnosis accuracy, pass@1 / pass@3 fix rate, time-to-diagnose, tokens-to-diagnose, cost. Baselines compare against single-shot LLM-only prompting on Gemini 2.5 Pro, Groq Llama 3.3 70B, Cerebras GPT-OSS 120B, and OpenRouter Qwen3-Coder. Full results published at lensml.dev/bench.

## License

Apache-2.0 for the core. Future enterprise features (SSO, team-shared dashboards, fine-tuned supervisor models) live in `/ee/` under a separate enterprise license. Trademark: USPTO ITU filed for "lensml" wordmark.

---

## 14. Day 1 setup checklist

Before you write a line of `lensml`, do this on Day 1 morning (~2 hours):

- [ ] `pyenv install 3.11.x` — pin Python version
- [ ] `uv init lensml/` — package skeleton
- [ ] Private GitHub repo `rupeshbharambe24/lensml`, push initial commit
- [ ] Verify domain `lensml.dev` available — register if so (~$15/yr)
- [ ] Verify `pip install lensml` returns nothing on PyPI (name available)
- [ ] Verify `github.com/lensml` org/user not taken
- [ ] Verify `x.com/lensml` handle not taken — register if available
- [ ] Sign up: Google AI Studio (free Gemini), Groq (free Llama), Cerebras (free GPT-OSS), OpenRouter (free models). Note current rate limits in `~/.lensml/notes/limits-2026-05-08.md`.
- [ ] `uv add fastapi sse-starlette pyzmq libcst litellm pydantic 'torch>=2.2'` etc.
- [ ] Create `~/.lensml/config.toml` template with provider keys (gitignored)
- [ ] `LICENSE` (Apache-2.0), `LICENSE-EE` (placeholder), `README.md` (placeholder), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, `CHANGELOG.md`
- [ ] `.github/workflows/ci.yml` — uv + ruff + pytest matrix on Python 3.11/3.12, Linux/macOS
- [ ] First commit. Begin Day 1 afternoon: hooks spike.

---

**End of plan.** Build it.
