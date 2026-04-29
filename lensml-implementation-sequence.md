# lensml — exact implementation sequence

**Author:** Rupesh Bharambe
**Created:** 2026-04-29
**Source of truth:** `lensml-plan.md` (spec, v1 locked)
**Submission:** FAR AWAY hackathon Round 1, Unstop, June 7, 2026

This document translates the plan's day-by-day calendar (§8) into an ordered execution path, plus a 9-day pre-build runway available because the scaffold landed early. Critical-path items are marked **CP**; everything else is buffer or optional polish.

> **Free-tier-only mode (active).** Per user directive 2026-04-29, v1 ships
> without any paid services. No domain registration, no X handle paid claim,
> no trademark filing. Substitutions:
> - Landing page → `https://lensml.vercel.app` (Vercel free tier default URL)
> - Docs → `https://rupeshbharambe24.github.io/lensml/` (GitHub Pages, free)
> - Canonical project URL in video/README/Unstop form → `github.com/rupeshbharambe24/lensml`
> - X handle / social → optional manual claim later if free; not blocking
>
> The plan-spec text (`lensml-plan.md`) is unchanged. Custom domain and
> trademark are revisited post-Round 1 only if the project gains traction.

---

## PHASE 0 — Pre-build runway (Apr 29 → May 7, 9 days)

The plan locks the build to start May 8. This window is for de-risking, not for building features.

| #   | Day      | Task | Exit criterion | Plan ref |
| --- | -------- | ---- | -------------- | -------- |
| 0.1 | Apr 29 | **CP** Verify free-tier name availability: PyPI `lensml`, GitHub `rupeshbharambe24/lensml`, public web (no clashing project). Domain + X registration deferred (free-tier mode). | Notes file `~/.lensml/notes/availability-2026-04-29.md` with results | §12, §14 |
| 0.2 | Apr 30   | Replace `LICENSE` and `CODE_OF_CONDUCT.md` with canonical text via the `curl` commands embedded in those files. | Both files match upstream verbatim + copyright line preserved | §14 |
| 0.3 | Apr 29 ✅ | **CP** Sign up for free tiers: Google AI Studio, Groq, Cerebras, OpenRouter, Mistral. Test one tool-call on each. Record current rate limits. | `~/.lensml/notes/limits-2026-04-29.md` — all 5 verified with tool-calls | §2 (medium confidence), §5.5, §14 |
| 0.4 | Apr 29 ✅ | **CP** Hooks-spike (moved earlier from plan Day 1): standalone script registers `register_forward_hook` + reads `param.grad`, prints activation mean and grad-L2 every step. Measure overhead. | `examples/raw_pytorch/_spike_hooks.py`. **CPU: +1.73% PASS. GPU: ~0.18ms/step absolute (% noisy on 3ms-step microbench).** Architecture validated: capture-only hooks + publish-rate decoupling. See `~/.lensml/notes/hooks-spike-2026-04-29.md`. | §2, §8 Day 1 |
| 0.5 | Apr 29 ✅ | LiteLLM tool-call spike (moved earlier from plan Day 8): one tool, four providers, confirm schema portability. | `examples/_spike_litellm.py` — **5/5 providers PASS** (Groq 0.83s, Mistral 1.32s, Cerebras 1.79s, Gemini 2.07s, OpenRouter 4.01s). See `~/.lensml/notes/litellm-spike-2026-04-29.md`. | §11 risk #5 |
| 0.6 | Apr 29 ✅ | **CP** Subprocess-restart spike (moved earlier from plan Day 13): save model+opt+RNG+sampler state in a child process, kill it, restart with `LENSML_RESUME=`, verify deterministic continuation. | **PASS — 12/12 byte-identical losses on both CPU and GPU.** Approach: pre-shuffled deterministic dataset + manual batch-index tracking sidesteps DataLoader sampler-state quirks (plan §3.3 "best effort"). See `~/.lensml/notes/restart-spike-2026-04-29.md`. | §11 risk #1 |
| 0.7 | Apr 29 ✅ | Initialize private GitHub repo `rupeshbharambe24/lensml`. Push scaffold. Add branch protection on `main`. | Repo at https://github.com/rupeshbharambe24/lensml — **PRIVATE ✓**, 90 files pushed, **CI green on all 5 jobs** (lint + test matrix on Python 3.11/3.12 × ubuntu/macos). Branch protection deferred (solo dev, low value). | §14 |
| 0.8 | Apr 29 ✅ | Re-read full plan with fresh eyes. Surface contradictions or gaps in `notes/plan-deltas-2026-04-29.md`. | **18 deltas surfaced** (6 already-resolved, 6 new findings, 6 needs-attention). Plan confidence: HIGH. See `~/.lensml/notes/plan-deltas-2026-04-29.md`. | — |
| 0.9 | open      | **REST.** Do nothing technical. Build window opens fresh on May 8. | — | — |

If 0.4 / 0.5 / 0.6 fail or surface major risk, **decide cut points before May 8** rather than during the build.

---

## PHASE 1 — Foundation (May 8 → May 14, Week 1)

End-of-week gate: working pipeline `train.py → hooks → ZMQ → server → SSE → dashboard chart` + first detector firing.

| #   | Day         | Task | Exit criterion | Plan ref |
| --- | ----------- | ---- | -------------- | -------- |
| 1.1 | Apr 29 ✅   | Lock pinned versions in `pyproject.toml`. `uv sync --all-extras` clean. First commit on `main`. | `uv.lock` committed (142 packages); CI uses cache (51s/run, down from 1m52s); `[dependency-groups] dev` migration done. | §8 D1 |
| 1.2 | Apr 29 ✅   | **CP** `lensml/core/hooks.py` real impl: forward hooks on all leaf modules, `param.grad` post-backward grad-L2. | `lensml/core/hooks.py` + `telemetry.py` implemented per spike 0.4 design (capture-only hot path + publish-rate cold path). 28/28 unit tests pass. | §8 D2 |
| 1.3 | Apr 29 ✅   | **CP** `lensml/core/transport.py`: pyzmq PUB on `ipc://lensml-{pid}.sock`. Stdin/stdout JSONL fallback flag. | `ZMQPublisher` + `ZMQSubscriber` + `JSONLPublisher` + `make_publisher()` factory. Default endpoint `tcp://127.0.0.1:5555` (TCP for Windows portability over `ipc://`). 11 unit tests including inproc round-trip. **39/39 tests pass.** | §8 D2 |
| 1.4 | May 10      | **CP** `lensml/server/app.py` + `sse.py` + `store.py`: FastAPI with SSE bridge from ZMQ SUB; SQLite event log per plan §3.3 schema. | `curl -N http://127.0.0.1:8000/api/runs/test/metrics` streams live ticks | §8 D3 |
| 1.5 | May 11      | **CP** Dashboard scaffold: `cd dashboard && pnpm create next-app@15 . && shadcn init`. Three-pane layout shell. uPlot wired to SSE. | Browser shows live loss curve from running script | §8 D4 |
| 1.6 | May 12      | **CP** `lensml.watch(model, optimizer, dataloader)` API. Monkey-patch `optimizer.step`. Smoke-test against raw PyTorch + Lightning model + HF Trainer model (just to check hooks survive wrapping). | One-line API works on raw PyTorch end-to-end | §8 D5 |
| 1.7 | May 13      | **CP** `lensml/detectors/exploding_grad.py`: real detector. Threshold per plan §4. Anomaly event flows to dashboard as toast. | Synthetic exploding-grad fixture triggers detector + UI toast | §8 D6 |
| 1.8 | May 14      | Polish + REST. Docstrings, ruff clean, CI green. | Week 1 complete | §8 D7 |

**Week 1 cut point:** if 1.4 (SSE) or 1.6 (`watch()` API) is broken at end of May 12 → ship CLI-only dashboard for v1, defer browser dashboard to v0.2.

---

## PHASE 2 — Magic moment (May 15 → May 21, Week 2)

End-of-week gate: detect → diagnose → diff → `y` → patch committed → checkpoint → restart → resume training, on raw PyTorch.

| #   | Day    | Task | Exit criterion | Plan ref |
| --- | ------ | ---- | -------------- | -------- |
| 2.1 | May 15 | `lensml/llm/client.py` + `routing.py`: LiteLLM wrapper with failover. Reads `~/.lensml/config.toml`. | One tool-call works against primary provider | §8 D8 |
| 2.2 | May 16 | **CP** `lensml/agent/tools.py`: all 12 tools as Pydantic schemas. Server-side execution from event store. | Agent chains 3–5 tool calls, returns structured diagnosis | §8 D9 |
| 2.3 | May 17 | **CP** `lensml/agent/loop.py` + `memory.py` + `confidence.py`: ~80-LOC ReAct loop. Three memory layers. N=3 self-consistency. | Agent emits structured diagnosis with HIGH/MEDIUM/LOW confidence on exploding-grad scenario | §8 D10 |
| 2.4 | May 18 | **CP** `lensml/fixes/base.py` + `grad_clip.py` + `lower_lr.py` + `replace_activation.py` + `add_normalize.py`. Each with a unit test on a known input. | 4 transformers green in pytest | §8 D11 |
| 2.5 | May 19 | **CP** Approval flow + git: `propose_fix` → ApprovalRequest queue → dashboard diff via react-diff-viewer-continued → REST approve → libcst apply → `git commit` on `lensml/session-{ts}` branch. Reject = no-op. Undo = `git reset --hard HEAD~1`. | End-to-end: user presses `y`, file is patched + committed | §8 D12 |
| 2.6 | May 20 | **CP** `lensml/server/checkpoint.py` + `lensml/core/signals.py`: SIGUSR1 trigger → save model+opt+sched+RNG+sampler → clean exit. Subprocess supervisor restarts trainer with `LENSML_RESUME=`. | **Magic moment works end-to-end on exploding-grad fixture** | §8 D13 |
| 2.7 | May 21 | Polish + REST. Bug fixes from week. | Week 2 complete | §8 D14 |

**Week 2 cut point:** if 2.6 (ckpt+restart) is broken at end of May 20 → ship "diagnosis-only" mode (text suggestion, no auto-apply) for v1. Still novel; still wins R1.

---

## PHASE 3 — Breadth + polish (May 22 → May 28, Week 3)

End-of-week gate: Lightning + HF Trainer working, all 8 detectors live, replay mode bulletproof.

| #   | Day    | Task | Exit criterion | Plan ref |
| --- | ------ | ---- | -------------- | -------- |
| 3.1 | May 22 | **CP** `lensml/integrations/lightning.py`: `LensmlCallback`. Test on CIFAR-10 + small CNN. | Lightning end-to-end including approval + restart | §8 D15 |
| 3.2 | May 23 | `lensml/integrations/transformers.py`: `LensmlTrainerCallback`. External orchestrator pattern (`should_training_stop = True` → server applies fix → restart with `resume_from_checkpoint=`). Test on DistilBERT/SST-2 (Colab T4). | HF integration works on T4 | §8 D16 |
| 3.3 | May 24 | Detectors 2–4: `dead_relu`, `missing_normalize`, `lr_too_high`, `model_eval_not_called`. | 5 detectors total in CI | §8 D17 |
| 3.4 | May 25 | Detectors 5–8: `dataloader_bottleneck`, `bad_init`, `vanishing_grad`. Plus 4 more libcst templates: `insert_eval_mode`, `kaiming_init`, `dataloader_workers`, `add_lr_scheduler`. | 8 detectors + 8 fix templates green | §8 D18 |
| 3.5 | May 26 | UX polish: status pill, vertical lines on loss curve at intervention points, agent trace pop-in animations, keyboard shortcuts, command palette (`cmdk`), JA toggle scaffold. | Dashboard feels production-grade | §8 D19 |
| 3.6 | May 27 | **CP** Replay mode: `lensml replay <session.jsonl>` deterministic playback. Record three canonical sessions: exploding-grad, dead-relu, missing-normalize. | **Bulletproof live-demo path exists.** Do not skip. | §8 D20 |
| 3.7 | May 28 | Privacy mode toggle (Plan C: scaffolded + one canned interaction). REST. | Week 3 complete | §8 D21 |

**Week 3 cut points:** if 3.2 (HF Trainer) broken → drop to v0.2 roadmap, document. If 3.4 fixtures hard → cut to 24 fixtures (Tiny + Small only).

---

## PHASE 4 — Benchmark + video (May 29 → Jun 4, Week 4)

End-of-week gate: 48 fixtures benched, report generated, 90-second video uploaded, README + landing page public-ready.

| #   | Day    | Task | Exit criterion | Plan ref |
| --- | ------ | ---- | -------------- | -------- |
| 4.1 | May 29 | Bench fixtures: 12 Tiny (MNIST/MLP) + 12 Small (CIFAR/ResNet-18) — all 4 fixtures across 12 pathology classes per env. | 24 fixtures with `bug.py` + `fix.diff` + `grade.py` + `meta.json` | §8 D22 |
| 4.2 | May 30 | Bench fixtures: 12 Medium (DistilBERT/SST-2) + 12 Large (GPT-2 small/WikiText-2). `lensml/bench/grader.py` real impl. | 48 fixtures + grader | §8 D23 |
| 4.3 | May 31 | `lensml/bench/runner.py`. First end-to-end run on Tiny env (16 fixtures). | Bench infra tested | §8 D24 |
| 4.4 | Jun 1  | **CP** Bench runs: Tiny + Small on Colab T4. Medium on Kaggle T4. **Run baseline LLM-only runs in parallel** (the credibility lynchpin). | ~32 fixtures + baselines | §8 D25 |
| 4.5 | Jun 2  | Bench runs: Large env on Kaggle T4 session 2 (16 fixtures). Generate `bench/results/2026-06-02/report.md` — confusion matrix, latency histogram, $0 cost slide, "what doesn't work yet". | Complete benchmark report | §8 D26 |
| 4.6 | Jun 3  | **CP** Demo video: 3 VHS GIFs (install / watch / intervene). Record 90-second Round 1 video — multiple takes. Upload YouTube unlisted. | Video is shareable | §8 D27 |
| 4.7 | Jun 4  | **CP** README final: hero GIF + 30-second install + comparison table + benchmark table. Landing page deployed to **Vercel free tier** (`lensml.vercel.app`). Docs to GitHub Pages. | Public-ready repo + landing page on free tier | §8 D28 |

**Week 4 cut point:** if benchmark results are weak at end of Jun 2 → publish as "lensml-bench v0.1 — preliminary results," frame methodology as the contribution.

---

## PHASE 5 — Submission (Jun 5 → Jun 7)

| #   | Day                | Task | Exit criterion |
| --- | ------------------ | ---- | -------------- |
| 5.1 | Jun 5 (Fri)        | Fill Unstop submission form. Attach video, README link (still private), benchmark report. Test install on a fresh VM/container. | Submission form draft saved |
| 5.2 | Jun 6 (Sat)        | Last-pass bug fixes. Verify all install paths work. Draft launch tweet thread + HN Show post (do not post). | Everything queued |
| 5.3 | Jun 7 (Sun) **AM** | **Submit on Unstop before 12:00 IST.** | Submission confirmation email |
| 5.4 | Jun 7 **PM**       | Toggle GitHub repo public. Push final commit. First tweet thread (09:00 IST). LinkedIn post. **Hold HN/Reddit until after R2 results.** | **DONE** |

---

## Critical-path summary (the 12 things that must work)

If any of these fails, the whole submission is at risk:

1. **0.4** Hooks spike — if PyTorch hooks don't give clean per-step telemetry, nothing else matters.
2. **0.6** Subprocess restart spike — the magic moment depends on this.
3. **1.4** SSE bridge — the dashboard depends on this.
4. **1.6** `lensml.watch()` API — the user-facing surface.
5. **1.7** First detector — the agent has nothing to react to without it.
6. **2.2** Tool registry — the agent's vocabulary.
7. **2.3** ReAct loop — the agent itself.
8. **2.5** Approval flow + libcst apply — the moment the user sees value.
9. **2.6** Checkpoint + restart — the magic moment.
10. **3.6** Replay mode — the only safe live-demo path.
11. **4.4** Bench + baselines — the credibility against "just ask GPT."
12. **4.6** 90-second video — 80% of R1 judging weight.

---

## Graceful-degradation floor

If everything goes wrong, the absolute minimum that still wins R1 (per plan §11):

- Raw PyTorch only (no Lightning, no HF)
- 4 detectors, not 8
- Diagnosis-only (suggest patch as text, no auto-apply)
- CLI dashboard (text TUI, not browser)
- 16 bench fixtures (Tiny env only)
- 60-second video, no Japanese toggle, no privacy mode

That's still novel — no one else ships diagnostic agents over live training.

---

## Net duration

9 days runway + 30 days build + 3 days submission = **42 days**, ending Jun 7.

Today is Apr 29 — clock starts now with **step 0.1**.

---

**End of sequence.** Execute it.
