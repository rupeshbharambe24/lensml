# VHS — lensml intervene demo, the magic moment (plan §8 D27).
# Render: `vhs demo/intervene.tape` → demo/out/intervene.gif
#
# Beat: replay the canned exploding_grad scenario through the
# dashboard server. No GPU, no LLM — fully deterministic. The
# screen recording for the actual demo composes this terminal
# alongside a browser at http://127.0.0.1:3000 showing the dashboard
# react in real time (intervention markers, agent reasoning pop-in,
# diff approval).
# Target length: ~16 s.

Output demo/out/intervene.gif

Set Shell "bash"
Set FontSize 14
Set Width  900
Set Height 420
Set Theme "GruvboxDark"
Set Padding 20
Set TypingSpeed 30ms

Type "# 1. Dashboard server already running at :8000."
Sleep 200ms
Enter
Type "# 2. Replay the canned exploding_grad scenario through it:"
Sleep 300ms
Enter
Sleep 600ms

Type "lensml replay --scenario exploding_grad --speed 2"
Sleep 400ms
Enter
Sleep 9000ms

Type "# Dashboard shows: vertical intervention marker + agent diagnosis."
Sleep 200ms
Enter
Sleep 1500ms
