# lensml-bench report

_2026-05-03 14:43:23 — 8 fixtures, 117.5s_


## Headline

- **Overall pass:** 3/8  (37.5%)
- **Fix passes grader:** 7/8  (87.5%)
- **Bug correctly fails:** 4/8


## Latency histogram

```
lensml-full:
  [  0.0–  5.1s]  (0)
  [  5.1– 10.2s]  (0)
  [ 10.2– 15.3s]  (0)
  [ 15.3– 20.4s]  (0)
  [ 20.4– 25.5s] ####### (7)
  [ 25.5– 30.6s] # (1)
```


## Cost breakdown

| arm | tier | cost |
| --- | --- | --- |
| lensml-full | free | **$0.00** |

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
