# Submission-day checklist (June 7, 2026)

The Sunday-morning sequence. Nothing here is risky if you start at
07:00 IST. The deadline is **12:00 IST** so there's a 5-hour buffer.

## T-5 hours (07:00 IST)

- [ ] `git pull origin main`
- [ ] `uv sync --all-extras`
- [ ] `uv run pytest -q` — must be all green
- [ ] `uv run bash scripts/smoke_install.sh` — must finish "All smoke checks passed"
- [ ] (Optional) `docker build -t lensml-smoke . && docker run --rm lensml-smoke`
- [ ] `cd dashboard && pnpm install && pnpm exec next build` — Next build green
- [ ] `uv run mkdocs build --strict` — docs build green

## T-4 hours (08:00 IST)

- [ ] Open `submission/links.md`. Fill every `TBD`:
    - [ ] YouTube unlisted URL for the 90-second video
    - [ ] Final benchmark report path
- [ ] Open Unstop submission form in browser
- [ ] Paste short description from `submission/pitch.md` ("Short description")
- [ ] Paste long description from `submission/pitch.md` ("Long description")
- [ ] Paste links from `submission/links.md`
- [ ] Attach: video, README screenshot, benchmark `report.md`
- [ ] **Save draft on Unstop** (do not submit yet)

## T-3 hours (09:00 IST)

- [ ] Final read-through of the saved draft
- [ ] Sanity-check every link opens cleanly in an incognito window
- [ ] Triple-check Apache-2.0 vs LICENSE-EE wording matches the form

## T-30 minutes (11:30 IST)

- [ ] Click **Submit** on Unstop
- [ ] Save the confirmation email
- [ ] Tell two friends (the witness move)

## T+ (post-submit, June 7 PM)

- [ ] Toggle GitHub repo to **public** (Settings → Danger zone → Change visibility)
- [ ] Push final commit if any post-deadline cleanup happened
- [ ] Verify CI green on the public repo
- [ ] Verify `lensml.vercel.app` and `rupeshbharambe24.github.io/lensml` resolve
- [ ] Post the tweet thread from `submission/launch.md` at 09:00 IST … wait, that's morning. Do it in the evening if you missed the morning slot.
- [ ] Cross-post to LinkedIn
- [ ] **Hold** Show HN / Reddit until after R2 results

## If anything breaks

- Tests red? → revert to last green main, submit that
- Submission form fails? → screenshot, email Unstop support, save proof of attempt before deadline
- GitHub Pages doesn't deploy? → docs work locally; submit anyway, fix the workflow Monday
- Vercel landing 502? → submit with just the GitHub link; landing is decoration

The important artifact is the **public GitHub repo + the video + the
benchmark report**. Everything else is polish.
