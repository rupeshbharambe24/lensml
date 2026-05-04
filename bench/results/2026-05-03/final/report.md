# lensml-bench v0.1 — 8/8 fixtures, 5 LLM baselines (May 3 2026)

_2026-05-04 11:30:09 — 8 fixtures, 87.0s_


## Headline

- **Overall pass:** 8/8  (100.0%)
- **Fix passes grader:** 8/8  (100.0%)
- **Bug correctly fails:** 8/8


## lensml-full vs LLM-only baselines

Plan §6.3: the LLM-only baseline is the credibility lynchpin. If lensml-full doesn't beat single-shot LLM diagnosis, the integration story doesn't hold up. Numbers below are over the same fixture set.

| arm | n | diagnosis acc | fix-pass rate | overall pass |
| --- | --- | --- | --- | --- |
| **lensml-full** | 8 | n/a (unique) | 100.0% | 100.0% |
| llm-only · groq | 8 | 37.5% | 0.0% | 0/8 |
| llm-only · gemini | 8 | 75.0% | 0.0% | 0/8 |
| llm-only · cerebras | 8 | 62.5% | 0.0% | 0/8 |
| llm-only · openrouter | 8 | 25.0% | 0.0% | 0/8 |
| llm-only · mistral | 8 | 37.5% | 0.0% | 0/8 |


## Confusion matrix (LLM-only baseline diagnoses)

Each row is a fixture's *actual* pathology; each column is the single-shot LLM diagnosis. Diagonal cells (correct) are bolded.

### baseline · groq

| actual ↓ / predicted → | (unparsed) | bad_init | exploding_grad | lr_too_high | model_eval_not_called | total |
| --- | --- | --- | --- | --- | --- | --- |
| bad_init | 0 | **1** | 0 | 0 | 0 | 1 |
| dead_relu | 1 | 0 | 0 | 0 | 0 | 1 |
| exploding_grad | 0 | 0 | **1** | 0 | 0 | 1 |
| lr_schedule_mismatch | 0 | 0 | 0 | 1 | 0 | 1 |
| lr_too_high | 1 | 0 | 0 | 0 | 0 | 1 |
| missing_normalize | 1 | 0 | 0 | 0 | 0 | 1 |
| model_eval_not_called | 0 | 0 | 0 | 0 | **1** | 1 |
| vanishing_grad | 1 | 0 | 0 | 0 | 0 | 1 |


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


### baseline · cerebras

| actual ↓ / predicted → | (unparsed) | bad_init | exploding_grad | lr_too_high | model_eval_not_called | vanishing_grad | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bad_init | 0 | **1** | 0 | 0 | 0 | 0 | 1 |
| dead_relu | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| exploding_grad | 0 | 0 | **1** | 0 | 0 | 0 | 1 |
| lr_schedule_mismatch | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| lr_too_high | 0 | 0 | 0 | **1** | 0 | 0 | 1 |
| missing_normalize | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| model_eval_not_called | 0 | 0 | 0 | 0 | **1** | 0 | 1 |
| vanishing_grad | 0 | 0 | 0 | 0 | 0 | **1** | 1 |


### baseline · openrouter

| actual ↓ / predicted → | (unparsed) | bad_init | lr_too_high | model_eval_not_called | total |
| --- | --- | --- | --- | --- | --- |
| bad_init | 0 | **1** | 0 | 0 | 1 |
| dead_relu | 1 | 0 | 0 | 0 | 1 |
| exploding_grad | 1 | 0 | 0 | 0 | 1 |
| lr_schedule_mismatch | 0 | 0 | 1 | 0 | 1 |
| lr_too_high | 1 | 0 | 0 | 0 | 1 |
| missing_normalize | 1 | 0 | 0 | 0 | 1 |
| model_eval_not_called | 0 | 0 | 0 | **1** | 1 |
| vanishing_grad | 1 | 0 | 0 | 0 | 1 |


### baseline · mistral

| actual ↓ / predicted → | (unparsed) | bad_init | dead_relu | lr_too_high | model_eval_not_called | total |
| --- | --- | --- | --- | --- | --- | --- |
| bad_init | 0 | **1** | 0 | 0 | 0 | 1 |
| dead_relu | 0 | 0 | **1** | 0 | 0 | 1 |
| exploding_grad | 1 | 0 | 0 | 0 | 0 | 1 |
| lr_schedule_mismatch | 0 | 0 | 0 | 1 | 0 | 1 |
| lr_too_high | 1 | 0 | 0 | 0 | 0 | 1 |
| missing_normalize | 1 | 0 | 0 | 0 | 0 | 1 |
| model_eval_not_called | 0 | 0 | 0 | 0 | **1** | 1 |
| vanishing_grad | 1 | 0 | 0 | 0 | 0 | 1 |


## Latency histogram

```
lensml-full:
  [  0.0–  8.4s]  (0)
  [  8.4– 16.9s] ######## (8)
  [ 16.9– 25.3s]  (0)
  [ 25.3– 33.7s]  (0)
  [ 33.7– 42.2s]  (0)
  [ 42.2– 50.6s]  (0)
baseline·groq:
  [  0.0–  8.4s]  (0)
  [  8.4– 16.9s] ## (2)
  [ 16.9– 25.3s] ### (3)
  [ 25.3– 33.7s] ## (2)
  [ 33.7– 42.2s] # (1)
  [ 42.2– 50.6s]  (0)
baseline·gemini:
  [  0.0–  8.4s]  (0)
  [  8.4– 16.9s] ## (2)
  [ 16.9– 25.3s] # (1)
  [ 25.3– 33.7s] ## (2)
  [ 33.7– 42.2s]  (0)
  [ 42.2– 50.6s] ### (3)
baseline·cerebras:
  [  0.0–  8.4s]  (0)
  [  8.4– 16.9s] ## (2)
  [ 16.9– 25.3s] #### (4)
  [ 25.3– 33.7s] ## (2)
  [ 33.7– 42.2s]  (0)
  [ 42.2– 50.6s]  (0)
baseline·openrouter:
  [  0.0–  8.4s]  (0)
  [  8.4– 16.9s] ### (3)
  [ 16.9– 25.3s] ### (3)
  [ 25.3– 33.7s] # (1)
  [ 33.7– 42.2s] # (1)
  [ 42.2– 50.6s]  (0)
baseline·mistral:
  [  0.0–  8.4s]  (0)
  [  8.4– 16.9s] #### (4)
  [ 16.9– 25.3s] ## (2)
  [ 25.3– 33.7s] # (1)
  [ 33.7– 42.2s] # (1)
  [ 42.2– 50.6s]  (0)
```


## Cost breakdown

| arm | tier | cost |
| --- | --- | --- |
| lensml-full | free | **$0.00** |
| baseline · groq | free | **$0.00** |
| baseline · gemini | free | **$0.00** |
| baseline · cerebras | free | **$0.00** |
| baseline · openrouter | free | **$0.00** |
| baseline · mistral | free | **$0.00** |

_Every provider used (Groq, Cerebras, Gemini, OpenRouter, Mistral) is on its free tier. The whole 30-day build, including baselines, ran on $0._


## Per-pathology

| pathology | n | overall pass | fix-pass rate |
| --- | --- | --- | --- |
| bad_init | 1 | 1/1 | 1/1 |
| dead_relu | 1 | 1/1 | 1/1 |
| exploding_grad | 1 | 1/1 | 1/1 |
| lr_schedule_mismatch | 1 | 1/1 | 1/1 |
| lr_too_high | 1 | 1/1 | 1/1 |
| missing_normalize | 1 | 1/1 | 1/1 |
| model_eval_not_called | 1 | 1/1 | 1/1 |
| vanishing_grad | 1 | 1/1 | 1/1 |


## Per-fixture

| fixture | pathology | env | bug fails | fix passes | overall | elapsed |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny/bad_init/0` | bad_init | tiny | ✓ | ✓ | ✓ | 10.1s |
| `tiny/dead_relu/0` | dead_relu | tiny | ✓ | ✓ | ✓ | 10.3s |
| `tiny/exploding_grad/0` | exploding_grad | tiny | ✓ | ✓ | ✓ | 11.1s |
| `tiny/lr_schedule_mismatch/0` | lr_schedule_mismatch | tiny | ✓ | ✓ | ✓ | 10.5s |
| `tiny/lr_too_high/0` | lr_too_high | tiny | ✓ | ✓ | ✓ | 10.9s |
| `tiny/missing_normalize/0` | missing_normalize | tiny | ✓ | ✓ | ✓ | 11.4s |
| `tiny/model_eval_not_called/0` | model_eval_not_called | tiny | ✓ | ✓ | ✓ | 10.8s |
| `tiny/vanishing_grad/0` | vanishing_grad | tiny | ✓ | ✓ | ✓ | 12.0s |


## What doesn't work yet

_All fixtures pass cleanly._
