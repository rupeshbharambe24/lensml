# Pre-baked answers (plan §11 risk #4 — judge questions)

Anticipated questions, drafted answers. Read before submission so
the on-call response is fast.

## "How is this different from Weights & Biases / MLflow / Neptune?"

W&B and friends are passive dashboards: they show metrics, they
don't act on them. lensml is the active agent — it detects, it
diagnoses, it patches the source, it restarts your training. The
two are complementary, not competitive: I'd happily ship W&B
alongside lensml in a serious lab. lensml's job is to take the
metrics that W&B already collects and *do something about them*
without a human in the loop.

## "How do you know the fix is correct?"

We don't always — and the design is honest about that. Three
defenses:

1. **Human in the loop.** Every patch is shown as a unified diff
   inside the dashboard. You press `y` to apply, `n` to reject,
   `u` to undo. No silent rewrites.
2. **Idempotent transformers.** Each libcst template is a no-op
   on its own output. Applying `AddGradientClipping` twice doesn't
   add two clips.
3. **Checkpoint + git.** Every patch lands on a session branch
   (`lensml/session-<ts>`); reverting is `git reset --hard HEAD~1`
   plus `git checkout main`. The fix never destroys the trainer's
   state — `LENSML_RESUME` makes resume bit-exact (verified in
   spike 0.6).

## "What about distributed training (DDP / FSDP)?"

v1 is single-process. v0.2 plan: hook installation runs on rank 0
only, telemetry aggregates from all ranks, and the supervisor
orchestrates a coordinated restart via `torchrun`'s built-in
re-spawn. Documented in the roadmap; not shipped today.

## "Privacy?"

There's a privacy-mode toggle in the dashboard header (and a REST
endpoint). v1 ships Plan-C: the toggle works, server-side state is
authoritative, and a canned-diagnosis endpoint returns structured
local-mode responses for every detector class with no network call.
The Ollama-backed real path is v0.2. The story is honest: "we
support local-only mode for privacy-sensitive deployments. It's
slower because we're optimizing for what runs on commodity hardware.
Cloud mode is what you saw in the demo."

## "Why an LLM agent for diagnosis instead of rules?"

Two reasons. (a) The detectors are rule-based — they run server-side
on the metric stream, no LLM needed for the *detection* path. The
LLM is only used to *explain* the firing detector + assemble the
right `libcst` template + bind it to the user's source. (b) The
prose layer makes lensml useful even when the bug is novel: the
agent says "I haven't seen this exact pattern, here's what's
suspicious about layer 12's grads" rather than crashing.

## "Why is the bench so small?"

8 Tiny fixtures in v1 because each is fully reproducible on a CPU
in seconds. The Medium/Large envs (DistilBERT/SST-2,
GPT-2/WikiText-2) require real GPU compute and are out of scope
for a 30-day solo build. The infrastructure (`BenchRunner`,
`BaselineRunner`, `Report`) handles 48 fixtures the same way it
handles 8 — adding the heavier envs is fixture work, not platform
work.

## "What's the cost story?"

$0. Every LLM provider used (Groq, Cerebras, Gemini 2.5 Pro,
OpenRouter, Mistral) is on its free tier with rate limits that fit
the agent loop's expected traffic (~19 calls per 5-minute
diagnosis cycle, well under any free-tier ceiling). The dashboard
runs on Vercel free; docs run on GitHub Pages free. The whole
30-day build, including baselines, ran on $0 of cloud spend.

## "What doesn't work yet?"

Honest list, also in `docs/roadmap.md`:

- Distributed training (DDP/FSDP)
- Real Ollama in privacy mode (today: scaffold + canned)
- Medium / Large bench envs
- 4 stretch detectors are detection-only, no fix templates yet
- HF Trainer auto-restart (callback works; subprocess hand-off
  documented but not productionized)

The bench's "what doesn't work yet" section in the rendered report
keeps this list automatically up to date per fixture.

## "Why not just use Cursor?"

Cursor is a great IDE-side code editor. It doesn't see your *running*
training process — it can't watch grad-L2 spike in real time and
intervene. lensml is the supervisor that sits between your trainer
and your editor. (Once it diagnoses and you press `y`, the patched
file shows up in your IDE — Cursor is welcome to live on top of it.)
