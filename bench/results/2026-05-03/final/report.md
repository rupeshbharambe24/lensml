# lensml-bench v0.1 — May 03 results

_2026-05-03 14:50:06 — 8 fixtures, 117.5s_


## Headline

- **Overall pass:** 3/8  (37.5%)
- **Fix passes grader:** 7/8  (87.5%)
- **Bug correctly fails:** 4/8


## lensml-full vs LLM-only baselines

Plan §6.3: the LLM-only baseline is the credibility lynchpin. If lensml-full doesn't beat single-shot LLM diagnosis, the integration story doesn't hold up. Numbers below are over the same fixture set.

| arm | n | diagnosis acc | fix-pass rate | overall pass |
| --- | --- | --- | --- | --- |
| **lensml-full** | 8 | n/a (unique) | 87.5% | 37.5% |
| llm-only · groq | 8 | 50.0% | 0.0% | 0/8 |
| llm-only · gemini | 8 | 75.0% | 0.0% | 0/8 |


## Confusion matrix (LLM-only baseline diagnoses)

Each row is a fixture's *actual* pathology; each column is the single-shot LLM diagnosis. Diagonal cells (correct) are bolded.

### baseline · groq

| actual ↓ / predicted → | (unparsed) | bad_init | dead_relu | lr_too_high | model_eval_not_called | vanishing_grad | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bad_init | 0 | **1** | 0 | 0 | 0 | 0 | 1 |
| dead_relu | 0 | 0 | **1** | 0 | 0 | 0 | 1 |
| exploding_grad | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| lr_schedule_mismatch | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| lr_too_high | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| missing_normalize | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| model_eval_not_called | 0 | 0 | 0 | 0 | **1** | 0 | 1 |
| vanishing_grad | 0 | 0 | 0 | 0 | 0 | **1** | 1 |


### baseline · gemini

| actual ↓ / predicted → | (unparsed) | bad_init | dead_relu | exploding_grad | lr_too_high | model_eval_not_called | vanishing_grad | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bad_init | 0 | **1** | 0 | 0 | 0 | 0 | 0 | 1 |
| dead_relu | 0 | 0 | **1** | 0 | 0 | 0 | 0 | 1 |
| exploding_grad | 0 | 0 | 0 | **1** | 0 | 0 | 0 | 1 |
| lr_schedule_mismatch | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| lr_too_high | 0 | 0 | 0 | 0 | **1** | 0 | 0 | 1 |
| missing_normalize | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| model_eval_not_called | 0 | 0 | 0 | 0 | 0 | **1** | 0 | 1 |
| vanishing_grad | 0 | 0 | 0 | 0 | 0 | 0 | **1** | 1 |


## Latency histogram

```
lensml-full:
  [  0.0–  5.5s]  (0)
  [  5.5– 11.1s]  (0)
  [ 11.1– 16.6s]  (0)
  [ 16.6– 22.1s]  (0)
  [ 22.1– 27.7s] ####### (7)
  [ 27.7– 33.2s] # (1)
baseline·groq:
  [  0.0–  5.5s]  (0)
  [  5.5– 11.1s] # (1)
  [ 11.1– 16.6s] ### (3)
  [ 16.6– 22.1s] ## (2)
  [ 22.1– 27.7s] ## (2)
  [ 27.7– 33.2s]  (0)
baseline·gemini:
  [  0.0–  5.5s]  (0)
  [  5.5– 11.1s] ## (2)
  [ 11.1– 16.6s] ### (3)
  [ 16.6– 22.1s] # (1)
  [ 22.1– 27.7s] # (1)
  [ 27.7– 33.2s] # (1)
```


## Cost breakdown

| arm | tier | cost |
| --- | --- | --- |
| lensml-full | free | **$0.00** |
| baseline · groq | free | **$0.00** |
| baseline · gemini | free | **$0.00** |

_Every provider used (Groq, Cerebras, Gemini, OpenRouter, Mistral) is on its free tier. The whole 30-day build, including baselines, ran on $0._


## Per-pathology

| pathology | n | overall pass | fix-pass rate |
| --- | --- | --- | --- |
| bad_init | 1 | 0/1 | 1/1 |
| dead_relu | 1 | 0/1 | 0/1 |
| exploding_grad | 1 | 1/1 | 1/1 |
| lr_schedule_mismatch | 1 | 0/1 | 1/1 |
| lr_too_high | 1 | 0/1 | 1/1 |
| missing_normalize | 1 | 1/1 | 1/1 |
| model_eval_not_called | 1 | 0/1 | 1/1 |
| vanishing_grad | 1 | 1/1 | 1/1 |


## Per-fixture

| fixture | pathology | env | bug fails | fix passes | overall | elapsed |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny/bad_init/0` | bad_init | tiny | ✗ | ✓ | ✗ | 24.4s |
| `tiny/dead_relu/0` | dead_relu | tiny | ✓ | ✗ | ✗ | 22.6s |
| `tiny/exploding_grad/0` | exploding_grad | tiny | ✓ | ✓ | ✓ | 22.3s |
| `tiny/lr_schedule_mismatch/0` | lr_schedule_mismatch | tiny | ✗ | ✓ | ✗ | 22.6s |
| `tiny/lr_too_high/0` | lr_too_high | tiny | ✗ | ✓ | ✗ | 25.4s |
| `tiny/missing_normalize/0` | missing_normalize | tiny | ✓ | ✓ | ✓ | 23.8s |
| `tiny/model_eval_not_called/0` | model_eval_not_called | tiny | ✗ | ✓ | ✗ | 22.6s |
| `tiny/vanishing_grad/0` | vanishing_grad | tiny | ✓ | ✓ | ✓ | 30.6s |


## What doesn't work yet

- **`tiny/bad_init/0`** (bad_init)
  - bug.py unexpectedly passed grader: (passed all checks — bug isn't a bug?)
- **`tiny/dead_relu/0`** (dead_relu)
  - patched.py failed grader: (passed all checks but classified failed)
- **`tiny/lr_schedule_mismatch/0`** (lr_schedule_mismatch)
  - bug.py unexpectedly passed grader: (passed all checks — bug isn't a bug?)
- **`tiny/lr_too_high/0`** (lr_too_high)
  - bug.py unexpectedly passed grader: (passed all checks — bug isn't a bug?)
- **`tiny/model_eval_not_called/0`** (model_eval_not_called)
  - bug.py unexpectedly passed grader: (passed all checks — bug isn't a bug?)
