"""LiteLLM wrapper with provider failover.

Plan §5.5:
    - On any 429 / 503 / rate-limit, retry with the next provider.
    - Log every routing decision to event log for post-hoc analysis.
    - Config in `~/.lensml/config.toml` (provider keys, routing table).

Plan §11 risk #5: cross-provider tool-call schema portability. Day 8 spike
with Gemini first; if broken, pin to Groq Llama 3.3 70B for v1.
"""

from __future__ import annotations

# Implemented Day 8.
