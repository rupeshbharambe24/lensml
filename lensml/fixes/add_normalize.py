"""AddNormalize — fix for missing input normalization.

Inserts `transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`
into the transform pipeline (defaults are ImageNet stats; plan §4 row 4).

Implemented Day 11.
"""

from __future__ import annotations
