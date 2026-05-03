# VHS — lensml install demo (plan §8 D27).
# Render: `vhs demo/install.tape` → demo/out/install.gif
#
# Beat: pip install → `import lensml` → `lensml --help` to show subcommands.
# Target length: ~12 s.

Output demo/out/install.gif

Set Shell "bash"
Set FontSize 14
Set Width  900
Set Height 380
Set Theme "GruvboxDark"
Set Padding 20
Set TypingSpeed 30ms

Type "pip install lensml"
Sleep 400ms
Enter
Sleep 1500ms

Type "python -c 'import lensml; print(lensml.__name__)'"
Sleep 400ms
Enter
Sleep 800ms

Type "lensml --help"
Sleep 400ms
Enter
Sleep 1800ms
