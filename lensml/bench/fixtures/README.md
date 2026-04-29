# lensml-bench fixtures

Per plan §6.1: **12 pathology classes × 4 host environments = 48 fixtures**.

## Directory layout

Each fixture is a directory with four files:

```
fixtures/
  <env>/<class>/<index>/
    bug.py        broken training script
    fix.diff      canonical patch (the ground truth)
    grade.py      deterministic checks (run / converge / metric threshold)
    meta.json     {pathology_class, expected_difficulty, host_env}
```

## Host environments (plan §6.2)

| Environment | Model           | Dataset                    | Per-run on T4 | Fixture count |
|-------------|-----------------|----------------------------|---------------|---------------|
| `tiny`      | small MLP / CNN | MNIST                      | 30–90 sec     | 16            |
| `small`     | ResNet-18       | CIFAR-10 (50k subset)      | 3–8 min       | 12            |
| `medium`    | DistilBERT      | SST-2 (small split)        | 8–20 min      | 12            |
| `large`     | GPT-2 small     | WikiText-2 (small)         | 15–30 min     | 8             |

## Pathology classes (plan §6.1)

The 12 classes are the 8 v1 pathologies (plan §4) plus 4 stretch classes:

1. exploding_grad
2. lr_too_high
3. dead_relu
4. missing_normalize
5. dataloader_bottleneck
6. model_eval_not_called
7. bad_init
8. vanishing_grad
9. *(stretch)* label_leakage
10. *(stretch)* train_val_contamination
11. *(stretch)* mixed_precision_nan
12. *(stretch)* lr_schedule_mismatch

For stretch classes, the agent can score "diagnosis correct, fix not yet
supported" — and the report will flag this honestly.

## Build schedule (plan §8 Days 22–23)

- Day 22 morning: 12 `tiny/` fixtures
- Day 22 afternoon: 12 `small/` fixtures
- Day 23 morning: 12 `medium/` fixtures
- Day 23 afternoon: 12 `large/` fixtures + grader

If schedule slips: cut the `large/` env (plan §8 cut points).
