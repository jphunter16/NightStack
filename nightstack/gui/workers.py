"""Run long jobs off the UI thread, with progress and cancel."""
import traceback

from PySide6.QtCore import QObject, QRunnable, Signal

from ..pipeline import Cancelled


class WorkerSignals(QObject):
    progress = Signal(float, str)
    finished = Signal(object)
    failed = Signal(str, str)


class Worker(QRunnable):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()
        self.cancelled = False
        self.setAutoDelete(False)

    def run(self):
        def progress(p, msg):
            if self.cancelled:
                raise Cancelled()
            self.signals.progress.emit(float(p), str(msg))
        try:
            result = self.fn(progress)
        except Cancelled:
            self.signals.failed.emit("Cancelled", "")
        except MemoryError:
            self.signals.failed.emit("Ran out of memory. Try 'Fast mode' or fewer photos.", traceback.format_exc())
        except Exception as e:  # noqa: BLE001
            self.signals.failed.emit(str(e) or e.__class__.__name__, traceback.format_exc())
        else:
            self.signals.finished.emit(result)
