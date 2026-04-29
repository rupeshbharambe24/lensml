# Security policy

## Supported versions

Until v0.1 ships (June 7, 2026), there are no supported versions. Treat all
pre-release builds as experimental.

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ (after release) |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, email software@biopanscientific.com with:

- A description of the vulnerability
- Steps to reproduce
- Affected version(s)
- Your assessment of impact

You will receive an acknowledgement within 72 hours and a remediation timeline
within 7 days.

## Scope

Security-relevant areas:

- LLM provider key handling (`~/.lensml/config.toml`)
- libcst-applied patches (must not introduce code injection vectors)
- Subprocess restart and signal handling (must not leak handles)
- Local FastAPI server (default bind is `127.0.0.1`; do not expose externally)
- SQLite event log (no PII expected, but verify before sharing)
