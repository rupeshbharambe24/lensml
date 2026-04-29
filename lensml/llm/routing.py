"""Per-task provider routing table.

Plan §5.5 quota math for a 5-minute live demo with 3 anomalies + 3 fixes:
    Anomaly classification: ~10 calls   (Groq 30 RPM — comfortable)
    Diagnosis:               3 calls   (Gemini 2.5 Pro free 5 RPM / 100 RPD)
    Patch generation:        3 calls   (OpenRouter 20 RPM)
    Verification:            3 calls   (Cerebras 30 RPM)
Total: ~19 LLM calls in 5 minutes across 4 providers.
"""

from __future__ import annotations

# Implemented Day 8.
