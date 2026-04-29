# lensml examples

Minimal training scripts that demonstrate `lensml.watch()` integration.

## Planned examples (built alongside detectors, plan §8)

```
examples/
  raw_pytorch/
    mnist_mlp.py           Day 5  — first end-to-end with hooks
    cifar_cnn.py           Day 6  — first detector (exploding_grad)
  lightning/
    cifar_resnet18.py      Day 15 — Lightning callback
  transformers/
    distilbert_sst2.py     Day 16 — HF Trainer callback
  demo/
    exploding_grad.py      Day 20 — replay-recorded session
    dead_relu.py           Day 20
    missing_normalize.py   Day 20
```

Each example is < 80 lines and runs on CPU or 4GB GPU (per plan §0:
"Local development uses MNIST and CIFAR-10 only").
