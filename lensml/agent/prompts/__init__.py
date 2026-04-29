"""System prompts per agent task.

Planned files (created during Day 8–10 build):
    system_diagnosis.md   primary diagnosis prompt
    system_anomaly.md     anomaly classification (lightweight)
    system_patch.md       code patch generation
    system_verify.md      verification ("does this fix make sense?")

Per plan §5.5, different tasks route to different LLM providers; the prompts
are tuned per task, not per provider.
"""
