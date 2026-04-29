"""lensml.integrations — framework adapters.

Per plan §3.2:
    pytorch.py      `lensml.watch(model, optimizer, dataloader)` — raw PyTorch
    lightning.py    LensmlCallback for `pytorch_lightning.Trainer`
    transformers.py LensmlTrainerCallback for `transformers.Trainer`

Per plan §0: cut order if Week 3 slips is to drop HF Trainer integration to
v0.2; never cut Lightning.
"""
