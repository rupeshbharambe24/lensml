# Submission packet — FAR AWAY hackathon (Round 1)

This directory holds the form-ready content for the Unstop submission
on June 7, 2026 (plan §8 D29–D30 → steps 5.1–5.4).

## Files

| file | what to do with it |
| ---- | ------------------ |
| `pitch.md` | Paste the short / long descriptions verbatim into the Unstop form. |
| `links.md` | Repo URL, landing page, docs, video link, benchmark report — all in one place. |
| `launch.md` | Draft tweet thread + Show HN post. **Do not post until Round 1 results** unless the plan changes. |
| `qa.md` | Pre-baked answers to anticipated judge questions (W&B differentiation, distributed training, privacy, etc.). |
| `submission_checklist.md` | The Sunday-morning checklist for the actual submit. |

## Submit-day flow (June 7)

```
07:00  pull main, run scripts/smoke_install.sh, run pytest -q
08:00  fill Unstop form using submission/pitch.md + submission/links.md
09:00  hold — final read-through
11:30  click Submit (well before 12:00 IST cutoff)
12:00  IST submission deadline
PM     toggle GitHub repo to public; push final commit; post tweet thread (linked-out, not yet HN)
```

## Reproducible install path (the question reviewers ask)

```bash
docker build -t lensml-smoke .
docker run --rm lensml-smoke
```

The smoke image runs `scripts/smoke_install.sh` which validates every
public submodule imports, the CLI is wired, all 8 fix templates are
registered, all 8 packaged tiny fixtures load, and all 3 replay
scenarios generate. If a fresh container can't pass that, neither
can a reviewer's laptop.

Locally: `uv run bash scripts/smoke_install.sh` does the same thing
without Docker.
