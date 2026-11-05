import sys

if len(sys.argv) > 1:
    from .cli import main
else:
    from .gui.app import main
main()
