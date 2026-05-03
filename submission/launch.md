# Launch posts — drafts (do NOT post until cleared)

Plan §8 D30 / step 5.4: post the tweet thread on Sunday June 7 at
09:00 IST. Hold HN/Reddit until after Round 2 results. LinkedIn post
with the tweet thread is fine same-day.

## Tweet thread (X / Twitter)

> 🧵 **lensml — the AI agent that supervises your ML training run.**
>
> 1/ Your training script breaks. You catch it on the loss curve
> three hours late. You eyeball the chart, kill the run, edit the
> code, restart, lose half a day. lensml automates the loop.

> 2/ One line:
>
> ```python
> import lensml
> lensml.watch(model, optimizer)
> ```
>
> Forward hooks + optimizer.step interception stream loss + grad-L2
> + activation stats to a local dashboard at <2% overhead.

> 3/ When a pathology fires (8 detectors in v1: exploding gradient,
> dead ReLU, lr too high, missing normalize, model.eval not called,
> bad init, vanishing grad, dataloader bottleneck) an LLM agent
> proposes a libcst-level patch.
>
> [intervene.gif]

> 4/ You press `y`. lensml commits the patch to a session git branch,
> checkpoints model + optimizer + RNG state, and restarts your script
> from the checkpoint with the fix in place. You didn't lose a step.

> 5/ Three integrations: raw PyTorch, PyTorch Lightning, HuggingFace
> Trainer. Eight libcst-backed fix templates, all idempotent.

> 6/ Free-tier friendly. LLM calls fan out across Groq, Cerebras,
> Gemini, OpenRouter, Mistral via LiteLLM with automatic fallback.
> $0 to run. Privacy mode toggle scaffolded for fully-local
> deployments.

> 7/ Plus `lensml-bench`: deterministic fixture grader, runner with
> resume cache, and LLM-only baseline arm so the comparison is
> falsifiable. The credibility story is "integration, not invention."

> 8/ Built solo over 30 days for the FAR AWAY hackathon. Every
> dependency on its free tier. Total cost: $0.
>
> 🔗 https://github.com/rupeshbharambe24/lensml
> 🌐 https://lensml.vercel.app
> 📺 [YouTube link]

## LinkedIn post (~1500 chars)

> I shipped lensml today — an open-source AI agent that supervises
> your ML training runs in real time. It hooks into PyTorch /
> Lightning / HuggingFace Trainer, detects pathologies (exploding
> gradients, dead ReLUs, missing normalize, model.eval forgotten,
> learning-rate issues, …), proposes a code patch via libcst, and
> resumes training from checkpoint on your approval.
>
> The novelty isn't any single piece — TrainCheck (OSDI '25) detects,
> AutoTrainer (ICSE '21) repairs in-memory Keras, Cursor and Aider
> rewrite source code, observability platforms (W&B, MLflow, Neptune)
> show metrics. lensml is the OSS agent that wires them into one
> workflow.
>
> Built solo over 30 days for the FAR AWAY hackathon, on free tiers
> only — every LLM call routes through Groq / Cerebras / Gemini /
> OpenRouter / Mistral free tiers via LiteLLM with automatic fallback.
> Total cost: $0.
>
> Repo + docs + landing page in the comments.
>
> #MachineLearning #PyTorch #AIAgents #OpenSource

## Show HN draft (HOLD until after R2 — do NOT post Round 1 weekend)

> Show HN: lensml — an AI agent that supervises your PyTorch training run
>
> Hi HN, I built lensml as a 30-day solo project. It watches your
> PyTorch / Lightning / HF Trainer training in real time, detects
> common bugs (exploding gradients, dead ReLUs, missing normalize, …),
> proposes libcst-level patches, and resumes from checkpoint with the
> fix applied — all on free LLM tiers ($0 to run).
>
> The interesting bits: …
>
> — pre-step gradient capture via wrapping `optimizer.step`
> — libcst transformers that produce idempotent diffs (apply twice = no-op)
> — checkpoint contract: model + optimizer + scheduler + RNG +
>   sampler-position, restart subprocess with `LENSML_RESUME=`
> — `lensml-bench`: deterministic fixtures + LLM-only baseline arm
>   for honest evaluation
>
> Honest about gaps: distributed training, real Ollama integration,
> Medium/Large bench envs all in v0.2. Apache-2.0 core; enterprise
> scaffolding under `/ee/`.
>
> Code: https://github.com/rupeshbharambe24/lensml
> Docs: https://rupeshbharambe24.github.io/lensml/
> Landing: https://lensml.vercel.app
>
> Curious what would make this useful in your workflow.
