import os
import sys


def main():
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QIcon
    from .. import APP_NAME
    from .theme import STYLE, dark_palette
    from .main_window import MainWindow
    from PySide6.QtCore import QEvent

    class App(QApplication):
        """Receives files dropped on the Dock icon / 'Open With' in Finder."""
        pending = []
        window = None

        def event(self, e):
            if e.type() == QEvent.FileOpen:
                f = e.file()
                if self.window is not None:
                    self.window._add_files([f])
                else:
                    self.pending.append(f)
                return True
            return super().event(e)

    app = App(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setStyleSheet(STYLE)
    icon = os.path.join(os.path.dirname(__file__), "..", "resources", "icon.png")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    w = MainWindow()
    app.window = w
    w.show()
    files = [a for a in sys.argv[1:] if os.path.isfile(a)] + app.pending
    if files:
        w._add_files(files)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
