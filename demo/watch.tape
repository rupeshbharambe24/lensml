# VHS — lensml watch demo (plan §8 D27).
# Render: `vhs demo/watch.tape` → demo/out/watch.gif
#
# Beat: a tiny training script with `lensml.watch(...)` printing live step
# lines. The dashboard runs separately for the screen capture; this GIF
# is just the terminal-side view.
# Target length: ~14 s.

Output demo/out/watch.gif

Set Shell "bash"
Set FontSize 14
Set Width  900
Set Height 420
Set Theme "GruvboxDark"
Set Padding 20
Set TypingSpeed 30ms

Type "# 1. Start the lensml dashboard server in another tab:"
Sleep 200ms
Enter
Type "#    lensml dashboard --db lensml.sqlite"
Sleep 400ms
Enter
Sleep 800ms

Type "python demo/scripts/demo_train.py"
Sleep 400ms
Enter
Sleep 6500ms
