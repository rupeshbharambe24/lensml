#!/usr/bin/env bash
# Regenerate the 3 README hero GIFs (plan §8 D27).
#
# Prereq: install vhs (https://github.com/charmbracelet/vhs).
# macOS:  brew install vhs
# Linux:  see VHS README for the latest install snippet.
#
# Outputs land in demo/out/. They are gitignored — the README links
# point at relative paths so any developer can regenerate them.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p demo/out

if ! command -v vhs >/dev/null 2>&1; then
    echo "vhs not found on PATH. Install it: https://github.com/charmbracelet/vhs"
    exit 2
fi

echo "Recording install.tape …"
vhs demo/install.tape

echo "Recording watch.tape (requires `lensml dashboard` to be running)…"
vhs demo/watch.tape

echo "Recording intervene.tape (requires `lensml dashboard` to be running)…"
vhs demo/intervene.tape

echo "Done. GIFs in demo/out/."
