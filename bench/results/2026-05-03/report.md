# lensml-bench report

_2026-05-04 11:12:44 — 8 fixtures, 87.0s_


## Headline

- **Overall pass:** 8/8  (100.0%)
- **Fix passes grader:** 8/8  (100.0%)
- **Bug correctly fails:** 8/8


## Latency histogram

```
lensml-full:
  [  0.0–  2.0s]  (0)
  [  2.0–  4.0s]  (0)
  [  4.0–  6.0s]  (0)
  [  6.0–  8.0s]  (0)
  [  8.0– 10.0s]  (0)
  [ 10.0– 12.0s] ######## (8)
```


## Cost breakdown

| arm | tier | cost |
| --- | --- | --- |
| lensml-full | free | **$0.00** |

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
