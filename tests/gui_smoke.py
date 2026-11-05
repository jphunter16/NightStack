"""Headless GUI walk-through: loads photos, edits the mask, aligns, stacks, exports, screenshots each step."""
import glob
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from nightstack.gui.main_window import MainWindow  # noqa: E402
from nightstack.gui.theme import STYLE, dark_palette  # noqa: E402
from nightstack.gui.viewer import Tool  # noqa: E402
from tests_util import ensure_data  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(out, exist_ok=True)
app = QApplication(sys.argv)
app.setStyle("Fusion")
app.setPalette(dark_palette())
app.setStyleSheet(STYLE)
QMessageBox.exec = lambda self: 0  # never block on dialogs
w = MainWindow()
w.resize(1500, 920)
w.show()


def wait():
    while w.worker is not None:
        QTest.qWait(50)
    QTest.qWait(150)


def shot(name):
    QTest.qWait(200)
    w.grab().save(os.path.join(out, name))
    print("saved", name)


shot("0_welcome.png")
files = sorted(glob.glob(os.path.join(ensure_data(), "frame_*.tif")))
w._add_files(files)
w._add_files(files[:2], "foregrounds")   # exercise the separate-foreground path
w.refresh_calibration()
wait()
shot("1_photos.png")
w.steps.setCurrentRow(1)
shot("2_calibration.png")
w.prepare()
wait()
assert w.project.prepared and w.stack.currentIndex() == 2
shot("3_sky.png")
# Simulate erasing dots on the ground and adding a horizon point via the viewer signals
n0 = int(w.project.star_display_points()[1].sum())
w.set_tool(Tool.ERASE)
w.viewer.stroke.emit(300.0, 200.0, None, True)
w.viewer.stroke.emit(500.0, 260.0, (300.0, 200.0), False)
w.viewer.strokeEnded.emit()
n1 = int(w.project.star_display_points()[1].sum())
print("stars before/after erase:", n0, n1)
assert n1 < n0
w.undo()
assert int(w.project.star_display_points()[1].sum()) == n0, "undo failed"
# Real mouse drag with the eraser in the viewport
w.set_tool(Tool.ERASE)
vp = w.viewer.viewport()
p1 = w.viewer.mapFromScene(QPointF(1200, 300))
p2 = w.viewer.mapFromScene(QPointF(1400, 350))
QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, p1)
QTest.mouseMove(vp, p2)
QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, p2)
print("after mouse erase:", int(w.project.star_display_points()[1].sum()))
w.undo()
w.set_tool(Tool.HORIZON)
shot("3b_sky_horizon_tool.png")
w.steps.setCurrentRow(3)
wait()
assert w.project.aligned_ok
w.align_table.selectRow(2)
shot("4_align.png")
w.steps.setCurrentRow(4)
wait()
assert w.project.stacked
shot("5_stack.png")
w.ground_method.setCurrentIndex(w.ground_method.findData("best"))
assert w.ground_pick.isVisible() and w.ground_pick.count() > 1
w.do_stack()
wait()
assert w.project.ground_choice is not None
print("best foreground frame:", w.project.ground_choice[0])
shot("5c_best_frame.png")
w.show_what.setCurrentIndex(3)
shot("5b_count_map.png")
w.show_what.setCurrentIndex(0)
w.steps.setCurrentRow(5)
QTest.qWait(300)
w.gradient.setChecked(True)
w.stars_boost.set(0.5)
QTest.qWait(400)
shot("6_finish.png")
w.tone.setCurrentIndex(1)
QTest.qWait(400)
shot("6b_finish_asinh.png")
fr = w.project.export_frames(os.path.join(out, "timelapse"), w.finish_params())
assert len(fr) == len(files)
path = os.path.join(out, "gui_export.tif")
w.project.export(path, w.finish_params())
assert os.path.exists(path)
w.close()
print("GUI SMOKE TEST PASSED")
