"""DataLoaderWorkers — fix for DataLoader bottleneck.

Sets `num_workers=min(8, os.cpu_count())`, `pin_memory=True`,
`persistent_workers=True` on the user's DataLoader call site.

Plan §4 row 5. Implemented Day 18.
"""

from __future__ import annotations
