"""NightStack main window: a guided, six-step workflow."""
from __future__ import annotations

import os
import re
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QAbstractSpinBox, QDoubleSpinBox, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..imageio import ALL_EXTS, DEMOSAIC, RAW_EXTS, is_supported, load_preview, resize_max
from ..pipeline import Project
from ..postprocess import FinishParams, apply_finish
from ..skymask import stretch_for_display
from ..stack import BLEND_MODES, GROUND_METHODS, SKY_METHODS
from .viewer import ImageViewer, Tool
from .workers import Worker

STEPS = ["1   Photos", "2   Calibration", "3   Sky & Foreground", "4   Align Stars", "5   Stack", "6   Finish & Save"]
FILTER = "Images (" + " ".join(f"*{e} *{e.upper()}" for e in sorted(ALL_EXTS)) + ")"


# --------------------------------------------------------------------------- small helpers
def title(text):
    l = QLabel(text)
    l.setObjectName("Title")
    return l


def hint(text):
    l = QLabel(text)
    l.setObjectName("Hint")
    l.setWordWrap(True)
    return l


def primary(text):
    b = QPushButton(text)
    b.setObjectName("Primary")
    b.setCursor(Qt.PointingHandCursor)
    return b


class LabeledSlider(QWidget):
    """Slider plus a box you can type an exact value into.

    `fmt` describes how the value is shown, e.g. "{:+.2f} EV", "{:.0%}", "{:.0f}px".
    The value is kept as a float, so typed values aren't rounded to the slider's steps.
    """
    changed = Signal(float)

    def __init__(self, lo, hi, value, step=0.01, fmt="{:.2f}", parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.step, self.fmt = lo, hi, step, fmt
        m = re.match(r"\{:(\+?)\.(\d)([f%])\}(.*)", fmt)
        plus, dec, kind, suffix = (m.group(1), int(m.group(2)), m.group(3), m.group(4)) if m else ("", 2, "f", "")
        self.scale = 100.0 if kind == "%" else 1.0
        if kind == "%":
            suffix = "%" + suffix
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) / step)))
        self.slider.setMinimumHeight(22)
        self.spin = QDoubleSpinBox()
        self.spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.spin.setKeyboardTracking(False)          # apply on Return / leaving the box
        self.spin.setDecimals(dec)
        self.spin.setRange(lo * self.scale, hi * self.scale)
        self.spin.setSingleStep(step * self.scale)    # arrow keys / scroll still nudge
        self.spin.setSuffix(suffix)
        if plus:
            self.spin.setPrefix("")
        self.spin.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.spin.setFixedWidth(84 if len(suffix) > 2 else 70)
        self.spin.setFixedHeight(26)
        self.spin.setToolTip("Type a value and press Return")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._v = float(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self._sync()

    def _sync(self):
        for w, v in ((self.slider, int(round((self._v - self.lo) / self.step))), (self.spin, self._v * self.scale)):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)

    def _from_slider(self, pos):
        self._v = self.lo + pos * self.step
        self._sync()
        self.changed.emit(self._v)

    def _from_spin(self, shown):
        self._v = float(np.clip(shown / self.scale, self.lo, self.hi))
        self._sync()
        self.changed.emit(self._v)

    def value(self):
        return self._v

    def set(self, v):
        self._v = float(np.clip(v, self.lo, self.hi))
        self._sync()
        self.changed.emit(self._v)


def combo(options: dict, current: str) -> QComboBox:
    c = QComboBox()
    for k, v in options.items():
        c.addItem(v, k)
    c.setCurrentIndex(list(options).index(current))
    shrink(c)
    return c


def shrink(c: QComboBox) -> QComboBox:
    """Let combo boxes narrow to the panel; the full text shows in the drop-down."""
    c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    c.setMinimumContentsLength(12)
    c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return c


class TooltipFilter(QObject):
    """Hides hover descriptions app-wide when they're turned off in Preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = True

    def eventFilter(self, obj, ev):
        if not self.enabled and ev.type() == QEvent.ToolTip:
            return True
        return False


# --------------------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1440, 900)
        self.setAcceptDrops(True)
        self.project = Project()
        self.qsettings = QSettings("NightStack", "NightStack")
        self.pool = QThreadPool.globalInstance()
        self.worker: Optional[Worker] = None
        self.undo_stack = []
        self.preview_cache = {}
        self.finish_timer = QTimer(self)
        self.finish_timer.setSingleShot(True)
        self.finish_timer.setInterval(120)
        self.finish_timer.timeout.connect(self.update_finish_preview)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar ------------------------------------------------------------
        side = QFrame()
        side.setObjectName("Sidebar")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        logo = QLabel(f"<span style='font-size:20px;font-weight:700;color:white'>✦ {APP_NAME}</span><br>"
                      f"<span style='color:#7d869a;font-size:11px'>Starry landscape stacker</span>")
        logo.setContentsMargins(14, 14, 8, 14)
        sl.addWidget(logo)
        self.steps = QListWidget()
        self.steps.setObjectName("Steps")
        for s in STEPS:
            self.steps.addItem(QListWidgetItem(s))
        self.steps.currentRowChanged.connect(self.go_step)
        sl.addWidget(self.steps, 1)
        self.info = hint("")
        self.info.setContentsMargins(12, 0, 8, 6)
        sl.addWidget(self.info)
        gear = QPushButton("⚙  Preferences")
        gear.setToolTip("App settings, your default processing options, and a list of every setting (⌘,)")
        gear.clicked.connect(self.open_preferences)
        sl.addWidget(gear)
        root.addWidget(side)

        # Viewer ---------------------------------------------------------------
        mid = QWidget()
        ml = QVBoxLayout(mid)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.view_caption = QLabel("")
        self.view_caption.setObjectName("Hint")
        bar.addWidget(self.view_caption, 1)
        self.zoom_label = QLabel("")
        self.zoom_label.setObjectName("Hint")
        bar.addWidget(self.zoom_label)
        for text, fn in (("Fit", lambda: self.viewer.fit()), ("100%", self.zoom_100),
                         ("−", lambda: self.viewer.zoom_by(1 / 1.4)), ("+", lambda: self.viewer.zoom_by(1.4))):
            b = QPushButton(text)
            b.setFixedWidth(62 if len(text) > 1 else 34)
            b.clicked.connect(fn)
            bar.addWidget(b)
        ml.addLayout(bar)
        self.viewer = ImageViewer()
        self.viewer.zoomChanged.connect(self._zoom_changed)
        ml.addWidget(self.viewer, 1)
        root.addWidget(mid, 1)

        # Panels ----------------------------------------------------------------
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setFixedWidth(396)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        pl.addWidget(self.stack)
        root.addWidget(panel)

        for build in (self._build_photos, self._build_calibration, self._build_sky, self._build_align,
                      self._build_stack, self._build_finish):
            self.stack.addWidget(self._pin_footer(build()))

        # Status bar -------------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setFixedWidth(260)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(self.cancel_task)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().showMessage("Welcome! Start by adding your night sky photos.")

        self._build_menu()
        self._shortcuts()
        self.steps.setCurrentRow(0)
        self.refresh_steps()
        self.show_welcome()
        self._update_worker = None
        self.setup_help_and_prefs()
        QTimer.singleShot(4000, self.auto_check_updates)

    @staticmethod
    def _pin_footer(page: QWidget) -> QWidget:
        """Settings scroll; the page's last widget (its main button) stays pinned at the bottom."""
        lay = page.layout()
        n = lay.count()
        stretches = [lay.stretch(i) for i in range(n)]
        items = [lay.takeAt(0) for _ in range(n)]
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(16, 16, 16, 8)
        def move(item, target, stretch=0):
            # addWidget/addLayout re-parent the widgets; a bare addItem would leave them behind.
            if item.widget() is not None:
                target.addWidget(item.widget(), stretch)
            elif item.layout() is not None:
                target.addLayout(item.layout(), stretch)
            else:
                target.addItem(item)
                target.setStretch(target.count() - 1, stretch)

        for it, st in zip(items[:-1], stretches[:-1]):
            move(it, cl, st)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setWidget(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(sc, 1)
        foot = QWidget()
        fl = QVBoxLayout(foot)
        fl.setContentsMargins(16, 8, 16, 14)
        move(items[-1], fl)
        lay.addWidget(foot)
        return page

    # ======================================================================= menu & shortcuts
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        a = QAction("Add Photos…", self, shortcut=QKeySequence.Open, triggered=self.add_photos)
        m.addAction(a)
        m.addAction(QAction("Save Image…", self, shortcut=QKeySequence.Save, triggered=self.export))
        m.addSeparator()
        m.addAction(QAction("Start Over", self, triggered=self.start_over))
        prefs = QAction("Preferences…", self, shortcut=QKeySequence.Preferences, triggered=self.open_preferences)
        prefs.setMenuRole(QAction.PreferencesRole)     # macOS: NightStack menu → Preferences… (⌘,)
        m.addAction(prefs)
        e = self.menuBar().addMenu("&Edit")
        e.addAction(QAction("Undo Mask Edit", self, shortcut=QKeySequence.Undo, triggered=self.undo))
        v = self.menuBar().addMenu("&View")
        v.addAction(QAction("Fit to Window", self, shortcut="Ctrl+0", triggered=lambda: self.viewer.fit()))
        v.addAction(QAction("Actual Pixels", self, shortcut="Ctrl+1", triggered=self.zoom_100))
        v.addAction(QAction("Zoom In", self, shortcut=QKeySequence.ZoomIn, triggered=lambda: self.viewer.zoom_by(1.4)))
        v.addAction(QAction("Zoom Out", self, shortcut=QKeySequence.ZoomOut,
                            triggered=lambda: self.viewer.zoom_by(1 / 1.4)))
        h = self.menuBar().addMenu("&Help")
        h.addAction(QAction("Quick Guide", self, triggered=self.show_guide))
        h.addSeparator()
        h.addAction(QAction("Check for Updates…", self, triggered=lambda: self.check_updates(silent=False)))
        self.auto_update_action = QAction("Check for Updates Automatically", self, checkable=True)
        self.auto_update_action.setChecked(self.qsettings.value("auto_update", True, type=bool))
        self.auto_update_action.toggled.connect(lambda on: self.qsettings.setValue("auto_update", on))
        h.addAction(self.auto_update_action)
        h.addAction(QAction(f"About {APP_NAME}", self, triggered=self.about))

    def _shortcuts(self):
        def tool_key(key, tool):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda: self.set_tool(tool) if self.stack.currentIndex() == 2 else None)
        tool_key("E", Tool.ERASE)
        tool_key("S", Tool.SKY)
        tool_key("A", Tool.ADD_STAR)
        tool_key("H", Tool.HORIZON)
        tool_key("P", Tool.PAN)
        QShortcut(QKeySequence("]"), self, activated=lambda: self.brush.set(min(400, self.brush.value() * 1.25)))
        QShortcut(QKeySequence("["), self, activated=lambda: self.brush.set(max(3, self.brush.value() / 1.25)))
        QShortcut(QKeySequence("M"), self, activated=lambda: self.show_mask.toggle())

    # ======================================================================= step 1: photos
    def _build_photos(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 16, 16, 16)
        l.addWidget(title("Your night photos"))
        l.addWidget(hint("Add a series of exposures shot from a tripod (RAW is best). 10–30 frames of "
                         "the same scene work great. Shoot them back-to-back with identical settings."))
        self.drop = QLabel("Drop photos here\nor click “Add Photos…”")
        self.drop.setObjectName("Drop")
        self.drop.setAlignment(Qt.AlignCenter)
        self.drop.setMinimumHeight(90)
        l.addWidget(self.drop)
        row = QHBoxLayout()
        b = primary("Add Photos…")
        b.clicked.connect(self.add_photos)
        row.addWidget(b)
        b2 = QPushButton("Remove")
        b2.clicked.connect(self.remove_selected)
        row.addWidget(b2)
        b3 = QPushButton("Clear")
        b3.clicked.connect(self.clear_photos)
        row.addWidget(b3)
        l.addLayout(row)
        self.photo_list = QListWidget()
        self.photo_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.photo_list.currentRowChanged.connect(self.preview_photo)
        self.photo_list.setMinimumHeight(220)
        l.addWidget(self.photo_list, 1)
        rb = QPushButton("★  Use selected as reference frame")
        rb.setToolTip("The reference frame sets the final star positions. The middle of the sequence is "
                      "chosen automatically - it keeps stars sharpest across the frame.")
        rb.clicked.connect(self.set_ref_from_list)
        l.addWidget(rb)
        g = QGroupBox("Separate foreground photos (optional)")
        gl = QVBoxLayout(g)
        gl.addWidget(hint("A blue-hour shot or a long, low-ISO exposure of the landscape from the same "
                          "tripod position gives a much cleaner foreground (Sequator-style HDR blend)."))
        row = QHBoxLayout()
        self.fg_label = QLabel("None")
        self.fg_label.setObjectName("Stat")
        row.addWidget(self.fg_label, 1)
        fa = QPushButton("Add…")
        fa.clicked.connect(lambda: self.add_cal("foregrounds"))
        fc = QPushButton("Clear")
        fc.clicked.connect(lambda: self.clear_cal("foregrounds"))
        row.addWidget(fa)
        row.addWidget(fc)
        gl.addLayout(row)
        l.addWidget(g)
        self.fast_mode = QCheckBox("Fast mode (half resolution, great for a quick test)")
        l.addWidget(self.fast_mode)
        self.photo_count = QLabel("")
        self.photo_count.setObjectName("Stat")
        l.addWidget(self.photo_count)
        nxt = primary("Continue  →")
        nxt.clicked.connect(lambda: self.steps.setCurrentRow(1))
        self.photos_next = nxt
        l.addWidget(nxt)
        return w

    def add_photos(self):
        start = self.qsettings.value("last_dir", os.path.expanduser("~/Pictures"))
        files, _ = QFileDialog.getOpenFileNames(self, "Add night photos", start, FILTER)
        self._add_files(files)

    def _add_files(self, files, kind="lights"):
        files = [f for f in files if is_supported(f)]
        if not files:
            return
        self.qsettings.setValue("last_dir", os.path.dirname(files[0]))
        target = getattr(self.project, kind)
        for f in sorted(files):
            if f not in target:
                target.append(f)
        self.project.invalidate("prepare")
        self.project.settings.ref_index = None
        self.refresh_photo_list()
        self.refresh_calibration()
        if kind == "lights" and self.photo_list.currentRow() < 0 and self.project.lights:
            self.photo_list.setCurrentRow(len(self.project.lights) // 2)

    def refresh_photo_list(self):
        self.photo_list.blockSignals(True)
        cur = self.photo_list.currentRow()
        self.photo_list.clear()
        n = len(self.project.lights)
        ref = self.project.settings.ref_index if self.project.settings.ref_index is not None else n // 2
        for i, f in enumerate(self.project.lights):
            star = "★ " if i == ref else "    "
            self.photo_list.addItem(f"{star}{i + 1:>3}.  {os.path.basename(f)}")
        self.photo_list.setCurrentRow(min(cur, n - 1))
        self.photo_list.blockSignals(False)
        raws = sum(os.path.splitext(f)[1].lower() in RAW_EXTS for f in self.project.lights)
        self.photo_count.setText(f"{n} photos" + (f"  ({raws} RAW)" if raws else "") +
                                 ("" if n == 0 else "   ★ = reference"))
        if n and n < 4:
            self.photo_count.setText(self.photo_count.text() + "\nTip: 8+ photos give much cleaner results.")
        self.refresh_steps()

    def remove_selected(self):
        rows = sorted({i.row() for i in self.photo_list.selectedIndexes()}, reverse=True)
        for r in rows:
            self.project.lights.pop(r)
        self.project.invalidate("prepare")
        self.project.settings.ref_index = None
        self.refresh_photo_list()

    def clear_photos(self):
        self.project.lights.clear()
        self.project.invalidate("prepare")
        self.refresh_photo_list()
        self.show_welcome()

    def set_ref_from_list(self):
        r = self.photo_list.currentRow()
        if r >= 0:
            self.project.settings.ref_index = r
            self.project.invalidate("prepare")
            self.refresh_photo_list()

    def preview_photo(self, row):
        if row < 0 or row >= len(self.project.lights):
            return
        path = self.project.lights[row]
        self.view_caption.setText(os.path.basename(path))
        if path in self.preview_cache:
            self._show_linear(self.preview_cache[path])
            return

        def job(progress):
            progress(0.3, f"Opening {os.path.basename(path)}")
            return path, load_preview(path, 1800)

        def done(res):
            p, img = res
            self.preview_cache[p] = img
            if len(self.preview_cache) > 12:
                self.preview_cache.pop(next(iter(self.preview_cache)))
            if self.photo_list.currentRow() >= 0 and self.project.lights[self.photo_list.currentRow()] == p:
                self._show_linear(img)
        self.run_task(job, done, quiet=True)

    def _show_linear(self, img):
        self.viewer.clear_overlays()
        self.viewer.edit_enabled = False
        self.viewer.set_tool(Tool.PAN)
        self.viewer.set_image(stretch_for_display(img))

    # ======================================================================= step 2: calibration
    def _build_calibration(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 16, 16, 16)
        l.addWidget(title("Calibration (optional)"))
        l.addWidget(hint("Calibration frames remove sensor noise patterns and vignetting. "
                         "Skip this step if you don't have them - NightStack still removes hot pixels "
                         "automatically."))
        self.cal_labels = {}
        w.setMinimumWidth(0)
        info = {
            "darks": ("Dark frames", "Same exposure/ISO/temperature as your photos, lens cap on."),
            "flats": ("Flat frames", "Evenly lit white surface (e.g. a tablet screen), same lens & aperture."),
            "biases": ("Bias frames", "Fastest shutter speed, lens cap on, same ISO."),
        }
        for kind, (name, tip) in info.items():
            g = QGroupBox(name)
            gl = QVBoxLayout(g)
            gl.addWidget(hint(tip))
            row = QHBoxLayout()
            lab = QLabel("None")
            lab.setObjectName("Stat")
            self.cal_labels[kind] = lab
            row.addWidget(lab, 1)
            add = QPushButton("Add…")
            add.clicked.connect(lambda _=False, k=kind: self.add_cal(k))
            row.addWidget(add)
            clr = QPushButton("Clear")
            clr.clicked.connect(lambda _=False, k=kind: self.clear_cal(k))
            row.addWidget(clr)
            gl.addLayout(row)
            l.addWidget(g)
        g = QGroupBox("RAW conversion")
        gf = QFormLayout(g)
        self.demosaic = QComboBox()
        for k, v in DEMOSAIC.items():
            self.demosaic.addItem(v, k)
        gf.addRow("Demosaic", shrink(self.demosaic))
        gf.addRow(hint("When your photos and calibration frames are all RAW, darks and flats are applied to "
                       "the raw sensor data before demosaicing (like Siril) for the cleanest result."))
        l.addWidget(g)
        self.hot_cb = QCheckBox("Automatically remove hot && cold pixels")
        self.hot_cb.setChecked(True)
        self.hot_cb.setToolTip("Finds pixels that stay bright in the same spot on every frame (stars move, "
                               "hot pixels don't) and repairs them.")
        l.addWidget(self.hot_cb)
        l.addStretch(1)
        self.prepare_btn = primary("Load photos  →")
        self.prepare_btn.clicked.connect(self.prepare)
        l.addWidget(self.prepare_btn)
        return w

    def add_cal(self, kind):
        start = self.qsettings.value("last_dir", os.path.expanduser("~/Pictures"))
        title = {"foregrounds": "Add foreground photos"}.get(kind, f"Add {kind}")
        files, _ = QFileDialog.getOpenFileNames(self, title, start, FILTER)
        self._add_files(files, kind)

    def clear_cal(self, kind):
        getattr(self.project, kind).clear()
        self.project.invalidate("prepare")
        self.refresh_calibration()

    def refresh_calibration(self):
        for k, lab in list(self.cal_labels.items()) + [("foregrounds", self.fg_label)]:
            n = len(getattr(self.project, k))
            lab.setText(f"{n} frame{'s' if n != 1 else ''}" if n else "None")

    def prepare(self):
        if not self.project.lights:
            self.steps.setCurrentRow(0)
            return
        s = self.project.settings
        s.half_size = self.fast_mode.isChecked()
        s.hot_pixels = self.hot_cb.isChecked()
        s.demosaic = self.demosaic.currentData()

        def job(progress):
            self.project.prepare(progress)
            self.project.auto_mask(refine=True, progress=lambda p, m: progress(0.9 + 0.1 * p, m))

        def done(_):
            self.undo_stack.clear()
            self.refresh_steps()
            self.steps.setCurrentRow(2)
            msg = f"Loaded {len(self.project.lights)} photos"
            if self.project.hot_count:
                msg += f", repaired {self.project.hot_count:,} hot pixels"
            if self.project.cfa_calibrated:
                msg += ", calibrated on RAW sensor data"
            self.statusBar().showMessage(msg + ". Check the sky mask, then align.")
        self.run_task(job, done)

    # ======================================================================= step 3: sky & foreground
    def _build_sky(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 16, 16, 16)
        l.addWidget(title("Sky & foreground"))
        l.addWidget(hint("NightStack found the horizon automatically. The <b style='color:#ffb070'>orange</b> "
                         "area is foreground (kept still); <b style='color:#ff6060'>red dots</b> are the stars "
                         "used to align the sky. Erase any red dots over the ground, and add sky or horizon "
                         "points where needed."))
        grid = QGridLayout()
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        tools = [(Tool.ERASE, "⌫  Erase dots", "E", "Brush over the ground: removes red dots and marks it "
                                                     "as foreground"),
                 (Tool.SKY, "✚  Paint sky", "S", "Brush to mark an area as sky (restores dots)"),
                 (Tool.ADD_STAR, "•  Add star", "A", "Click a star to add an alignment dot (snaps to the star)"),
                 (Tool.HORIZON, "⌇  Horizon points", "H", "Click to add green points along the horizon, drag to "
                                                          "move, right-click (or ⌥-click) to delete"),
                 (Tool.PAN, "✋  Move / zoom", "P", "Drag to move. Pinch or ⌘-scroll to zoom. Hold Space with "
                                                   "any tool to pan.")]
        for i, (tool, text, key, tip) in enumerate(tools):
            b = QToolButton()
            b.setText(f"{text}   ({key})")
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, t=tool: self.set_tool(t))
            self.tool_group.addButton(b)
            self.tool_buttons[tool] = b
            grid.addWidget(b, i // 2, i % 2)
        l.addLayout(grid)
        f = QFormLayout()
        self.brush = LabeledSlider(3, 400, 40, 1, "{:.0f}px")
        self.brush.changed.connect(lambda: self.viewer.set_brush(self.brush.value()))
        f.addRow("Brush size  [ ]", self.brush)
        self.opacity = LabeledSlider(0, 1, 0.8, 0.05, "{:.0%}")
        self.opacity.changed.connect(self.refresh_mask_view)
        f.addRow("Overlay", self.opacity)
        l.addLayout(f)
        self.show_mask = QCheckBox("Show foreground tint  (M)")
        self.show_mask.setChecked(True)
        self.show_mask.toggled.connect(self.refresh_mask_view)
        l.addWidget(self.show_mask)
        self.erase_marks = QCheckBox("Eraser also marks the area as foreground")
        self.erase_marks.setChecked(True)
        l.addWidget(self.erase_marks)
        self.no_fg = QCheckBox("No foreground (whole picture is sky)")
        self.no_fg.toggled.connect(self.toggle_no_fg)
        l.addWidget(self.no_fg)
        row = QHBoxLayout()
        b1 = QPushButton("Auto-detect again")
        b1.clicked.connect(self.redetect)
        b2 = QPushButton("✨ Refine edges")
        b2.setToolTip("Snap the mask to fine details like tree branches and rocks")
        b2.clicked.connect(self.refine_edges)
        row.addWidget(b1)
        row.addWidget(b2)
        l.addLayout(row)
        row = QHBoxLayout()
        b3 = QPushButton("Undo  (⌘Z)")
        b3.clicked.connect(self.undo)
        b4 = QPushButton("Reset my edits")
        b4.clicked.connect(self.reset_edits)
        row.addWidget(b3)
        row.addWidget(b4)
        l.addLayout(row)
        self.sky_stat = QLabel("")
        self.sky_stat.setObjectName("Stat")
        self.sky_stat.setWordWrap(True)
        l.addWidget(self.sky_stat)
        l.addStretch(1)
        nxt = primary("Align stars  →")
        nxt.clicked.connect(lambda: self.steps.setCurrentRow(3))
        l.addWidget(nxt)

        self.viewer.stroke.connect(self.on_stroke)
        self.viewer.strokeEnded.connect(self.on_stroke_end)
        self.viewer.starClicked.connect(self.on_add_star)
        self.viewer.horizonEditStarted.connect(self.push_undo)
        self.viewer.horizonEdited.connect(self.on_horizon_edit)
        return w

    def enter_sky(self):
        if not self.project.prepared:
            return
        self.viewer.edit_enabled = True
        self.viewer.set_image(self.project.display_image())
        self.no_fg.blockSignals(True)
        self.no_fg.setChecked(self.project.mask_model.no_foreground)
        self.no_fg.blockSignals(False)
        self.set_tool(self.viewer.tool if self.viewer.tool != Tool.PAN else Tool.ERASE)
        self.viewer.set_brush(self.brush.value())
        self.view_caption.setText(f"Reference frame: {os.path.basename(self.project.lights[self.project.ref_idx])}")
        self.refresh_mask_view()

    def set_tool(self, tool):
        self.viewer.set_tool(tool)
        b = self.tool_buttons.get(tool)
        if b:
            b.setChecked(True)

    def refresh_mask_view(self, *_):
        if not self.project.prepared or self.stack.currentIndex() != 2:
            return
        mm = self.project.mask_model
        self.viewer.set_overlay(mm.mask() if self.show_mask.isChecked() else None, self.opacity.value())
        pts, shown = self.project.star_display_points()
        self.viewer.set_stars(pts[shown])
        self.viewer.set_horizon(mm.horizon if not mm.no_foreground else [])
        n = int(shown.sum())
        sky = (mm.mask() > 127).mean()
        txt = f"{n} alignment stars  •  {sky:.0%} sky"
        if n < 30:
            txt += "\n⚠ Few stars - paint more sky or add stars by hand."
        self.sky_stat.setText(txt)

    def push_undo(self):
        mm = self.project.mask_model
        self.undo_stack.append((mm.base.copy(), mm.edits.copy(), list(mm.horizon), mm.no_foreground,
                                self.project.stars.pts.copy(), self.project.stars.flux.copy(),
                                self.project.stars.removed.copy()))
        if len(self.undo_stack) > 15:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack or not self.project.prepared:
            return
        mm = self.project.mask_model
        (mm.base, mm.edits, mm.horizon, mm.no_foreground, self.project.stars.pts, self.project.stars.flux,
         self.project.stars.removed) = self.undo_stack.pop()
        self.project.invalidate("mask")
        self.refresh_mask_view()
        self.refresh_steps()

    def on_stroke(self, x, y, last, first):
        if not self.project.prepared:
            return
        if first:
            self.push_undo()
        r = self.brush.value()
        mm = self.project.mask_model
        if self.viewer.tool == Tool.ERASE:
            self.project.erase_stars(x, y, r)
            if self.erase_marks.isChecked():
                mm.paint(x, y, r, -1, last)
        elif self.viewer.tool == Tool.SKY:
            mm.paint(x, y, r, +1, last)
            self.project.restore_stars(x, y, r)
        self.refresh_mask_view()

    def on_stroke_end(self):
        self.project.invalidate("mask")
        self.refresh_steps()

    def on_add_star(self, x, y):
        self.push_undo()
        sx, sy = self.project.add_star(x, y)
        mm = self.project.mask_model
        if mm.mask()[int(min(sy, mm.h - 1)), int(min(sx, mm.w - 1))] <= 127:
            mm.paint(sx, sy, max(4, self.brush.value() / 4), +1)
        self.project.invalidate("mask")
        self.refresh_mask_view()

    def on_horizon_edit(self, pts):
        mm = self.project.mask_model
        mm.horizon = list(pts)
        mm.rebuild_from_horizon()
        self.project.invalidate("mask")
        self.refresh_mask_view()

    def toggle_no_fg(self, on):
        self.push_undo()
        self.project.mask_model.no_foreground = on
        self.project.invalidate("mask")
        self.refresh_mask_view()

    def reset_edits(self):
        self.push_undo()
        self.project.mask_model.clear_edits()
        self.project.stars.removed[:] = False
        self.project.invalidate("mask")
        self.refresh_mask_view()

    def redetect(self):
        self.push_undo()
        self.run_task(lambda p: self.project.auto_mask(True, p), lambda _: self.enter_sky())

    def refine_edges(self):
        self.push_undo()
        self.run_task(lambda p: self.project.refine_mask(p), lambda _: self.refresh_mask_view())

    # ======================================================================= step 4: align
    def _build_align(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 16, 16, 16)
        l.addWidget(title("Align the stars"))
        l.addWidget(hint("NightStack tracks every red-dot star through the sequence and models the sky's "
                         "rotation, so stars line up across the whole frame while the foreground stays put."))
        f = QFormLayout()
        self.align_model = QComboBox()
        self.align_model.addItem("Wide-angle lens correction (recommended)", True)
        self.align_model.addItem("Standard (rigid sky rotation)", False)
        self.align_model.setToolTip("Lens correction adds a smooth local refinement that removes residual "
                                    "misalignment from lens distortion in the corners.")
        f.addRow("Model", shrink(self.align_model))
        self.interp = QComboBox()
        self.interp.addItem("Lanczos (sharpest)", "lanczos")
        self.interp.addItem("Bicubic", "cubic")
        self.interp.addItem("Bilinear (fastest)", "linear")
        f.addRow("Resampling", shrink(self.interp))
        self.auto_reject = QCheckBox("Skip bad frames automatically (blur, clouds, bumps)")
        self.auto_reject.setChecked(True)
        self.auto_reject.setToolTip("Measures star sharpness (FWHM), star shape, star count and sky brightness "
                                    "for every frame and unticks the outliers. You can tick them back on.")
        f.addRow(self.auto_reject)
        l.addLayout(f)
        self.align_btn = primary("Align stars")
        self.align_btn.clicked.connect(self.align)
        l.addWidget(self.align_btn)
        self.align_table = QTableWidget(0, 6)
        self.align_table.setHorizontalHeaderLabels(["Use", "Frame", "Stars", "Accuracy", "Sharpness", "Weight"])
        self.align_table.verticalHeader().hide()
        hdr = self.align_table.horizontalHeader()
        for c in range(6):
            hdr.setSectionResizeMode(c, QHeaderView.Stretch if c == 1 else QHeaderView.ResizeToContents)
        self.align_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.align_table.setTextElideMode(Qt.ElideMiddle)
        self.align_table.setWordWrap(False)
        self.align_table.setHorizontalHeaderLabels(["", "Frame", "Stars", "Error", "FWHM", "Wt"])
        for c, tip in enumerate(["Use this frame", "File name", "Stars matched", "Alignment error (pixels)",
                                 "Star size - smaller is sharper", "Weight in the stack"]):
            self.align_table.horizontalHeaderItem(c).setToolTip(tip)
        self.align_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.align_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.align_table.setMinimumHeight(260)
        self.align_table.currentCellChanged.connect(lambda r, *_: self.preview_aligned(r))
        self.align_table.itemChanged.connect(self.on_use_toggled)
        l.addWidget(self.align_table, 1)
        self.align_stat = hint("Click a row to preview that frame aligned to the reference.")
        l.addWidget(self.align_stat)
        nxt = primary("Stack  →")
        nxt.clicked.connect(lambda: self.steps.setCurrentRow(4))
        self.align_next = nxt
        l.addWidget(nxt)
        return w

    def enter_align(self):
        if self.project.prepared:
            self.viewer.edit_enabled = False
            self.viewer.set_tool(Tool.PAN)
            self.viewer.clear_overlays()
            if not self.project.aligned_ok:
                self.viewer.set_image(self.project.display_image())
                self.align_table.setRowCount(0)

    def align(self):
        s = self.project.settings
        s.align_local = self.align_model.currentData()
        s.interp = self.interp.currentData()
        s.auto_reject = self.auto_reject.isChecked()
        s.weighting = self.weighting.currentData()
        self.run_task(lambda p: self.project.align(p), lambda _: self.after_align())

    def after_align(self):
        res = self.project.align_results
        t = self.align_table
        t.blockSignals(True)
        t.setRowCount(len(res))
        for i, r in enumerate(res):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            use.setCheckState(Qt.Checked if r.enabled else Qt.Unchecked)
            t.setItem(i, 0, use)
            name = os.path.basename(self.project.lights[i]) + ("  ★" if i == self.project.ref_idx else "")
            t.setItem(i, 1, QTableWidgetItem(name))
            t.setItem(i, 2, QTableWidgetItem(str(r.n_inliers) if r.ok else "–"))
            if i == self.project.ref_idx:
                acc = "ref"
            elif r.ok:
                acc = f"{r.rms:.2f}"
            else:
                acc = "✗ failed"
            it = QTableWidgetItem(acc)
            it.setToolTip(r.message)
            if not r.ok:
                it.setForeground(Qt.red)
            t.setItem(i, 3, it)
            sh = QTableWidgetItem(f"{r.fwhm:.1f}" if r.fwhm else "–")
            sh.setToolTip(f"Star FWHM {r.fwhm:.2f}px, roundness {r.roundness:.2f}, sky level {r.background:.4f}")
            t.setItem(i, 4, sh)
            t.setItem(i, 5, QTableWidgetItem(f"{r.weight:.2f}" if r.ok else "–"))
            if not r.enabled and r.ok:
                for c in range(1, 6):
                    t.item(i, c).setForeground(Qt.darkYellow)
                    t.item(i, c).setToolTip(r.message)
        t.blockSignals(False)
        ok = sum(r.ok for r in res)
        bad = len(res) - ok
        skipped = [r for r in res if r.ok and not r.enabled]
        self.align_stat.setText(f"{ok} of {len(res)} frames aligned." +
                                (f" {bad} frame(s) couldn't be aligned (clouds?) - they'll still help the "
                                 "foreground." if bad else " Click a row to preview.") +
                                (f"\n{len(skipped)} frame(s) auto-skipped (yellow) - hover for the reason, "
                                 "tick to include anyway." if skipped else ""))
        self.refresh_steps()
        self.statusBar().showMessage("Alignment done. Continue to Stack.")
        t.selectRow(self.project.ref_idx)

    def on_use_toggled(self, item):
        if item.column() == 0 and self.project.align_results:
            self.project.align_results[item.row()].enabled = item.checkState() == Qt.Checked
            self.project.stacked = False
            self.refresh_steps()

    def preview_aligned(self, row):
        if row < 0 or not self.project.aligned_ok:
            return
        r = self.project.align_results[row]
        src = self.project.aligned if r.ok else self.project.orig
        img = resize_max(src.read(row), self.project.DISPLAY_MAX)
        self.viewer.set_image(stretch_for_display(img), keep_view=True)
        self.view_caption.setText(f"Frame {row + 1} {'aligned to reference' if r.ok else '(not aligned)'} - "
                                  "stars should stay put as you click through the list")

    # ======================================================================= step 5: stack
    def _build_stack(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 16, 16, 16)
        l.addWidget(title("Stack"))
        l.addWidget(hint("The sky is stacked from the aligned frames, the foreground from the originals, then "
                         "the two are blended at the horizon."))
        s = self.project.settings
        mode = QHBoxLayout()
        self.mode_land = QRadioButton("Starry landscape")
        self.mode_trails = QRadioButton("Star trails")
        self.mode_land.setChecked(True)
        mode.addWidget(self.mode_land)
        mode.addWidget(self.mode_trails)
        l.addLayout(mode)
        g = QGroupBox("Noise reduction")
        f = QFormLayout(g)
        self.sky_method = combo(SKY_METHODS, s.sky_method)
        f.addRow("Sky", self.sky_method)
        self.ground_method = combo(GROUND_METHODS, s.ground_method)
        self.ground_method.setToolTip("Best single frame keeps moving trees, grass or water crisp instead of "
                                      "averaging them into a blur (more noise - try Foreground smoothing in step 6).")
        f.addRow("Foreground", self.ground_method)
        self.ground_pick = QComboBox()
        self.ground_pick.setToolTip("Auto picks the frame closest to the consensus of all frames (no headlights, "
                                    "people or bumps) that is also sharp and low-noise.")
        shrink(self.ground_pick)
        self.ground_pick_label = QLabel("Use frame")
        f.addRow(self.ground_pick_label, self.ground_pick)
        self.ground_info = QLabel("")
        self.ground_info.setObjectName("Stat")
        self.ground_info.setWordWrap(True)
        f.addRow(self.ground_info)
        self.ground_method.currentIndexChanged.connect(self.refresh_ground_pick)
        self.kappa = LabeledSlider(1.5, 4.0, s.kappa, 0.1, "{:.1f}σ")
        self.kappa.setToolTip("Lower = more aggressive rejection of planes/satellites (sigma methods)")
        f.addRow("Clip strength", self.kappa)
        self.normalize = QCheckBox("Normalize frames (match sky brightness)")
        self.normalize.setChecked(True)
        self.normalize.setToolTip("Siril-style background + scale normalisation: keeps rejection working when "
                                  "skyglow, moonlight or haze changes during the sequence.")
        f.addRow(self.normalize)
        self.weighting = QComboBox()
        for k, t in (("quality", "Quality (sharp, dark-sky frames count more)"), ("noise", "Noise"),
                     ("fwhm", "Star sharpness"), ("none", "Equal")):
            self.weighting.addItem(t, k)
        f.addRow("Frame weights", shrink(self.weighting))
        self.small_stack_cb = QCheckBox("Small-stack protection (<10 frames)")
        self.small_stack_cb.setChecked(True)
        f.addRow(self.small_stack_cb)
        self.match_fg = QCheckBox("Match foreground photos to night brightness")
        self.match_fg.setChecked(True)
        f.addRow(self.match_fg)
        l.addWidget(g)
        g2 = QGroupBox("Horizon blending")
        f2 = QFormLayout(g2)
        self.blend_mode = combo(BLEND_MODES, s.blend_mode)
        self.blend_mode.setToolTip(
            "Minimum horizon noise: just above the horizon fewer aligned frames contain sky (the rest "
            "show ground there), which makes that band noisy. NightStack never lets ground leak into the "
            "sky, and smoothly leans on the all-frames stack where aligned data is thin, so the horizon "
            "is as clean as the rest of the sky.")
        f2.addRow("Mode", self.blend_mode)
        self.horizon_strength = LabeledSlider(0, 1, s.horizon_strength, 0.05, "{:.0%}")
        f2.addRow("Noise fill", self.horizon_strength)
        self.feather = LabeledSlider(0, 40, s.feather, 1, "{:.0f}px")
        f2.addRow("Edge softness", self.feather)
        self.margin = QSpinBox()
        self.margin.setRange(0, 30)
        self.margin.setValue(s.horizon_margin)
        self.margin.setSuffix(" px")
        self.margin.setToolTip("Ignore sky pixels this close to the horizon in each frame - prevents "
                               "a glow or dark halo along the skyline.")
        f2.addRow("Horizon guard", self.margin)
        l.addWidget(g2)
        self.stack_btn = primary("Stack!")
        self.stack_btn.clicked.connect(self.do_stack)
        l.addWidget(self.stack_btn)
        g3 = QGroupBox("Show")
        v = QVBoxLayout(g3)
        self.show_what = QComboBox()
        for k, t in (("result", "Final result"), ("sky", "Sky stack only"), ("ground", "Foreground stack only"),
                     ("count", "Frames used per pixel (heat map)"), ("single", "Single frame (before)")):
            self.show_what.addItem(t, k)
        self.show_what.currentIndexChanged.connect(self.show_stack_view)
        v.addWidget(shrink(self.show_what))
        l.addWidget(g3)
        l.addStretch(1)
        nxt = primary("Finish && save  →")
        nxt.clicked.connect(lambda: self.steps.setCurrentRow(5))
        self.stack_next = nxt
        l.addWidget(nxt)
        return w

    def refresh_ground_pick(self, *_):
        """Fill the frame chooser (night frames, or separate foreground photos if any)."""
        best = self.ground_method.currentData() == "best"
        self.ground_pick.setVisible(best)
        self.ground_pick_label.setVisible(best)
        self.ground_info.setVisible(best and bool(self.ground_info.text()))
        pr = self.project
        names = pr.foregrounds if pr.foregrounds else pr.lights
        cur = self.ground_pick.currentData()
        self.ground_pick.blockSignals(True)
        self.ground_pick.clear()
        self.ground_pick.addItem("Auto (sharpest, cleanest)", None)
        for i, n in enumerate(names):
            if not pr.foregrounds and pr.align_results and not pr.align_results[i].enabled:
                continue
            self.ground_pick.addItem(f"{i + 1}.  {os.path.basename(n)}", i)
        k = self.ground_pick.findData(cur)
        self.ground_pick.setCurrentIndex(max(0, k))
        self.ground_pick.blockSignals(False)

    def enter_stack(self):
        self.refresh_ground_pick()
        self.viewer.edit_enabled = False
        self.viewer.set_tool(Tool.PAN)
        self.viewer.clear_overlays()
        if self.project.stacked:
            self.show_stack_view()

    def do_stack(self):
        s = self.project.settings
        s.mode = "trails" if self.mode_trails.isChecked() else "landscape"
        if s.mode == "landscape" and not self.project.aligned_ok:
            QMessageBox.information(self, APP_NAME, "Please align the stars first (step 4).")
            self.steps.setCurrentRow(3)
            return
        s.sky_method = self.sky_method.currentData()
        s.ground_method = self.ground_method.currentData()
        s.kappa = self.kappa.value()
        s.blend_mode = self.blend_mode.currentData()
        s.horizon_strength = self.horizon_strength.value()
        s.feather = self.feather.value()
        s.horizon_margin = self.margin.value()
        s.ground_frame = self.ground_pick.currentData() if s.ground_method == "best" else None
        s.normalize = self.normalize.isChecked()
        s.auto_small_stack = self.small_stack_cb.isChecked()
        s.match_foreground = self.match_fg.isChecked()
        if s.weighting != self.weighting.currentData() and self.project.align_results:
            s.weighting = self.weighting.currentData()
            self.project._grade_frames_weights_only()

        def done(_):
            self.preview_src = None
            self.full_preview = False
            self.show_what.setCurrentIndex(0)
            self.show_stack_view()
            self.refresh_steps()
            ch = self.project.ground_choice
            self.ground_info.setText(f"Foreground from: {ch[0]}" if ch else "")
            self.ground_info.setVisible(bool(ch))
            notes = self.project.method_notes
            self.ground_info.setText("\n".join(([f"Foreground from: {ch[0]}"] if ch else []) + notes))
            self.ground_info.setVisible(bool(ch or notes))
            self.statusBar().showMessage("Stack complete! Fine-tune and save in step 6." +
                                         (f"  Foreground: {ch[0]}" if ch else "") +
                                         (f"  ({notes[0]})" if notes else ""))
        self.run_task(lambda p: self.project.stack(p), done)

    def show_stack_view(self):
        if not self.project.stacked:
            return
        what = self.show_what.currentData()
        pr = self.project
        fp = self.finish_params()
        fp.remove_gradient = False
        fp.sky_denoise = fp.ground_denoise = 0
        m = cv2.resize(pr.final_mask, None, fx=1, fy=1)
        if what == "count":
            c = resize_max(pr.count, self.preview_size()) / max(1, pr.n_sky)
            heat = cv2.applyColorMap((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            self.viewer.set_image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), keep_view=True)
            self.view_caption.setText(f"Frames contributing sky data (red = all {pr.n_sky}, blue = few). "
                                      "The thin band above the horizon is what Minimum Horizon Noise fixes.")
            return
        src = {"result": pr.result, "sky": pr.sky_stack, "ground": pr.ground_stack,
               "single": pr.orig.read(pr.ref_idx) if what == "single" else None}[what]
        img = resize_max(src, self.preview_size())
        ms = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.viewer.set_image(apply_finish(img, ms, fp), keep_view=True)
        self.view_caption.setText(self.show_what.currentText())

    # ======================================================================= step 6: finish
    def _build_finish(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(16, 16, 16, 16)
        l.addWidget(title("Finish & save"))
        l.addWidget(hint("Quick adjustments with a live preview. For full editing, save a 16-bit TIFF and "
                         "open it in Lightroom or Photoshop."))
        g = QGroupBox("Tone")
        f = QFormLayout(g)
        self.exposure = LabeledSlider(-5, 10, 0, 0.05, "{:+.2f} EV")
        f.addRow("Brightness", self.exposure)
        auto = QPushButton("Auto")
        auto.clicked.connect(lambda: self.exposure.set(self.project.suggest_exposure(self.black.value())))
        f.addRow("", auto)
        self.black = LabeledSlider(0, 1, 0.0, 0.01, "{:.0%}")
        self.black.setToolTip("How much of the sky's background glow to remove (darker, punchier sky)")
        f.addRow("Black point", self.black)
        self.contrast = LabeledSlider(-0.5, 0.8, 0.0, 0.02, "{:+.2f}")
        f.addRow("Contrast", self.contrast)
        self.saturation = LabeledSlider(0, 2, 1.0, 0.01, "{:.2f}")
        f.addRow("Saturation", self.saturation)
        self.tone = QComboBox()
        self.tone.addItem("Natural (photographic)", "natural")
        self.tone.addItem("Astro (asinh, keeps star colour)", "asinh")
        f.addRow("Tone curve", shrink(self.tone))
        self.asinh = LabeledSlider(1, 300, 30, 1, "{:.0f}")
        self.asinh.setToolTip("Stretch strength for the Astro tone curve")
        f.addRow("Astro stretch", self.asinh)
        l.addWidget(g)
        g2 = QGroupBox("Clean-up")
        f2 = QFormLayout(g2)
        self.gradient = QCheckBox("Remove light-pollution gradient")
        f2.addRow(self.gradient)
        self.neutral = QCheckBox("Neutral sky colour (remove colour cast)")
        f2.addRow(self.neutral)
        self.highlights = QCheckBox("Protect bright stars && lights")
        f2.addRow(self.highlights)
        self.grad_deg = QComboBox()
        for d, t in ((1, "Simple (linear)"), (2, "Normal"), (3, "Complex"), (4, "Very complex")):
            self.grad_deg.addItem(t, d)
        self.grad_deg.setCurrentIndex(1)
        f2.addRow("Gradient shape", shrink(self.grad_deg))
        self.stars_boost = LabeledSlider(0, 1, 0, 0.05, "{:.0%}")
        self.stars_boost.setToolTip("Brightens stars without touching the Milky Way glow or the ground")
        f2.addRow("Enhance stars", self.stars_boost)
        self.sky_dn = LabeledSlider(0, 1, 0, 0.05, "{:.0%}")
        f2.addRow("Extra sky smoothing", self.sky_dn)
        self.gnd_dn = LabeledSlider(0, 1, 0, 0.05, "{:.0%}")
        f2.addRow("Foreground smoothing", self.gnd_dn)
        l.addWidget(g2)
        for wdg in (self.exposure, self.black, self.contrast, self.saturation, self.sky_dn, self.gnd_dn,
                    self.asinh, self.stars_boost):
            wdg.changed.connect(lambda *_: self.finish_timer.start())
        for c in (self.tone, self.grad_deg):
            c.currentIndexChanged.connect(lambda *_: self.finish_timer.start())
        for cb in (self.gradient, self.neutral, self.highlights):
            cb.toggled.connect(lambda *_: self.finish_timer.start())
        self.before_btn = QPushButton("Hold to compare with a single frame")
        self.before_btn.pressed.connect(self.show_before)
        self.before_btn.released.connect(self.update_finish_preview)
        l.addWidget(self.before_btn)
        g3 = QGroupBox("Save")
        f3 = QFormLayout(g3)
        self.fmt = QComboBox()
        for label, ext, bits in (("TIFF 16-bit (best for editing)", ".tif", 16), ("JPEG (sharing)", ".jpg", 8),
                                 ("PNG 16-bit", ".png", 16), ("TIFF 32-bit float", ".tif", 32),
                                 ("FITS 32-bit (Siril / PixInsight)", ".fits", 32)):
            self.fmt.addItem(label, (ext, bits))
        f3.addRow("Format", shrink(self.fmt))
        self.linear = QCheckBox("Linear data (no brightness/tone applied)")
        self.linear.setToolTip("For further processing in Siril, PixInsight or Sequator-style workflows")
        f3.addRow(self.linear)
        self.layers = QCheckBox("Also save sky, foreground && mask layers")
        self.layers.setToolTip("Separate files for blending by hand in Photoshop")
        f3.addRow(self.layers)
        l.addWidget(g3)
        l.addStretch(1)
        row = QVBoxLayout()
        reset = QPushButton("Reset to baseline")
        reset.setToolTip("No adjustments: the stacked image exactly as shot (best for Lightroom)")
        reset.clicked.connect(self.reset_finish)
        row.addWidget(reset)
        self.full_btn = QPushButton("🔍  Render 100% full preview")
        self.full_btn.setToolTip("Renders the full-resolution image so you can inspect stars at 1:1. "
                                 "Moving a slider switches back to the fast preview.")
        self.full_btn.clicked.connect(lambda: self.render_full_preview())
        row.addWidget(self.full_btn)
        l.addLayout(row)
        self.frames_btn = QPushButton("Export aligned frames for a time-lapse…")
        self.frames_btn.setToolTip("Saves every frame with the sky aligned and your finishing applied")
        self.frames_btn.clicked.connect(self.export_frames)
        l.addWidget(self.frames_btn)
        self.save_btn = primary("Save Image…")
        self.save_btn.clicked.connect(self.export)
        l.addWidget(self.save_btn)
        return w

    def finish_params(self) -> FinishParams:
        return FinishParams(exposure=self.exposure.value(), black_point=self.black.value(),
                            saturation=self.saturation.value(), contrast=self.contrast.value(),
                            remove_gradient=self.gradient.isChecked(), neutralize=self.neutral.isChecked(),
                            sky_denoise=self.sky_dn.value(), ground_denoise=self.gnd_dn.value(),
                            protect_highlights=self.highlights.isChecked(), linear_output=self.linear.isChecked(),
                            gradient_degree=self.grad_deg.currentData(), enhance_stars=self.stars_boost.value(),
                            tone=self.tone.currentData(), asinh_strength=self.asinh.value())

    def enter_finish(self):
        self.full_preview = False
        self.viewer.edit_enabled = False
        self.viewer.set_tool(Tool.PAN)
        self.viewer.clear_overlays()
        if self.project.stacked:
            self.update_finish_preview()

    def update_finish_preview(self):
        if not self.project.stacked or self.stack.currentIndex() != 5:
            return
        img, m = self.project.preview_inputs(self.preview_size())
        out = apply_finish(img, m, self.finish_params())
        if getattr(self, "full_preview", False):
            self.full_preview = False
            self.swap_image(out)
            self.view_caption.setText("Fast preview - click “Render 100% full preview” to inspect stars at 1:1")
        else:
            self.viewer.set_image(out, keep_view=True)
            self.view_caption.setText("Fast preview (the saved image is full resolution)")

    def swap_image(self, img):
        """Replace the displayed image with one of a different size, keeping the same visible area."""
        v = self.viewer
        old_w = max(1.0, v.image_rect().width())
        c = v.mapToScene(v.viewport().rect().center())
        rel = (c.x() / old_w, c.y() / max(1.0, v.image_rect().height()))
        z = v.transform().m11()
        v.set_image(img, keep_view=True)
        new_w = v.image_rect().width()
        v.resetTransform()
        v.scale(z * old_w / new_w, z * old_w / new_w)
        v.centerOn(rel[0] * new_w, rel[1] * v.image_rect().height())
        v.zoomChanged.emit(v.transform().m11())

    def render_full_preview(self, zoom_to_100: bool = True):
        """Full-resolution render on demand, then show it at true 1:1 pixels."""
        if not self.project.stacked:
            return
        fp = self.finish_params()
        pr = self.project

        def job(progress):
            progress(0.2, "Rendering the full-resolution preview…")
            return apply_finish(pr.result, pr.final_mask, fp)

        def done(img):
            if self.stack.currentIndex() != 5:
                return
            self.swap_image(img)
            self.full_preview = True
            if zoom_to_100:
                v = self.viewer
                c = v.mapToScene(v.viewport().rect().center())
                v.resetTransform()
                ratio = v.devicePixelRatioF() or 1.0   # Retina: one image pixel per screen pixel
                v.scale(1 / ratio, 1 / ratio)
                v.centerOn(c)
                v.zoomChanged.emit(v.transform().m11())
            h, w = img.shape[:2]
            self.view_caption.setText(f"100% full resolution ({w}×{h}) - drag to look around; "
                                      "moving a slider returns to the fast preview")
            self.statusBar().showMessage("Full-resolution preview ready")
        self.run_task(job, done)

    def reset_finish(self):
        """Baseline: no tone changes or clean-up - the stack exactly as shot."""
        for w, v in ((self.exposure, 0.0), (self.black, 0.0), (self.contrast, 0.0), (self.saturation, 1.0),
                     (self.asinh, 30.0), (self.stars_boost, 0.0), (self.sky_dn, 0.0), (self.gnd_dn, 0.0)):
            w.set(v)
        for cb in (self.gradient, self.neutral, self.highlights, self.linear):
            cb.setChecked(False)
        self.tone.setCurrentIndex(0)
        self.grad_deg.setCurrentIndex(1)

    def show_before(self):
        if not self.project.stacked:
            return
        img = resize_max(self.project.orig.read(self.project.ref_idx), self.preview_size())
        m = cv2.resize(self.project.final_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.viewer.set_image(apply_finish(img, m, self.finish_params()), keep_view=True)
        self.view_caption.setText("Single frame (before stacking)")

    def export(self):
        if not self.project.stacked:
            QMessageBox.information(self, APP_NAME, "Stack your photos first.")
            return
        ext, bits = self.fmt.currentData()
        base = os.path.splitext(os.path.basename(self.project.lights[self.project.ref_idx]))[0]
        start = os.path.join(self.qsettings.value("save_dir", os.path.dirname(self.project.lights[0])),
                             f"{base}_NightStack{ext}")
        path, _ = QFileDialog.getSaveFileName(self, "Save image", start, f"*{ext}")
        if not path:
            return
        if not path.lower().endswith(ext):
            path += ext
        self.qsettings.setValue("save_dir", os.path.dirname(path))
        fp = self.finish_params()

        def done(files):
            box = QMessageBox(self)
            box.setWindowTitle(APP_NAME)
            box.setText("Saved!\n\n" + "\n".join(os.path.basename(f) for f in files))
            show = box.addButton("Show in Finder", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Ok)
            box.exec()
            if box.clickedButton() == show:
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))
        self.run_task(lambda p: self.project.export(path, fp, bits=bits, layers=self.layers.isChecked(),
                                                    progress=p), done)

    def export_frames(self):
        if not self.project.stacked:
            QMessageBox.information(self, APP_NAME, "Stack your photos first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the frames",
                                                  self.qsettings.value("save_dir", os.path.expanduser("~")))
        if not folder:
            return
        fp = self.finish_params()
        fp.linear_output = False
        self.run_task(lambda p: self.project.export_frames(folder, fp, ".jpg", progress=p),
                      lambda files: self.statusBar().showMessage(f"Saved {len(files)} frames to {folder}"))

    # ======================================================================= navigation
    def available(self, step: int) -> bool:
        pr = self.project
        return {0: True, 1: bool(pr.lights), 2: pr.prepared, 3: pr.prepared,
                4: pr.prepared and (pr.aligned_ok or self.mode_trails.isChecked()), 5: pr.stacked}[step]

    def refresh_steps(self):
        for i in range(self.steps.count()):
            it = self.steps.item(i)
            ok = self.available(i)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable if ok else Qt.NoItemFlags)
            done = {1: self.project.prepared, 2: self.project.aligned_ok, 3: self.project.aligned_ok,
                    4: self.project.stacked}.get(i, False)
            it.setText(STEPS[i] + ("   ✓" if done else ""))
        self.photos_next.setEnabled(bool(self.project.lights))
        self.prepare_btn.setText("Reload photos  →" if self.project.prepared else "Load photos  →")
        n = len(self.project.lights)
        self.info.setText(f"{n} photo{'s' if n != 1 else ''}" if n else "")

    def go_step(self, i):
        if i < 0:
            return
        if not self.available(i):
            self.steps.blockSignals(True)
            self.steps.setCurrentRow(self.stack.currentIndex())
            self.steps.blockSignals(False)
            return
        if i == 2 and not self.project.prepared:
            return
        self.stack.setCurrentIndex(i)
        if i == 0:
            self.viewer.edit_enabled = False
            self.viewer.set_tool(Tool.PAN)
            self.preview_photo(self.photo_list.currentRow())
        elif i == 1:
            self.viewer.edit_enabled = False
            self.viewer.set_tool(Tool.PAN)
        elif i == 2:
            self.enter_sky()
        elif i == 3:
            self.enter_align()
            if self.project.prepared and not self.project.aligned_ok:
                if self.pref("auto_align"):
                    self.align()
                else:
                    self.statusBar().showMessage("Choose your options, then click “Align stars”.")
        elif i == 4:
            self.enter_stack()
            if not self.project.stacked:
                if self.pref("auto_stack"):
                    self.do_stack()
                else:
                    self.statusBar().showMessage("Choose your options, then click “Stack!”.")
        elif i == 5:
            self.enter_finish()

    # ======================================================================= tasks
    def run_task(self, fn: Callable, on_done: Callable, quiet: bool = False):
        if self.worker is not None:
            if quiet:
                return
            QMessageBox.information(self, APP_NAME, "Please wait for the current task to finish.")
            return
        w = Worker(fn)
        self.worker = w
        if not quiet:
            self.progress.setValue(0)
            self.progress.show()
            self.cancel_btn.show()
            self.stack.setEnabled(False)
            self.steps.setEnabled(False)
            self.viewer.edit_enabled = False

        def finish():
            self.worker = None
            self.progress.hide()
            self.cancel_btn.hide()
            self.stack.setEnabled(True)
            self.steps.setEnabled(True)
            self.viewer.edit_enabled = self.stack.currentIndex() == 2 and self.project.prepared
            self.viewer.set_tool(self.viewer.tool)

        def ok(res):
            finish()
            on_done(res)

        def bad(msg, tb):
            finish()
            self.refresh_steps()
            if msg == "Cancelled":
                self.statusBar().showMessage("Cancelled.")
                return
            if not quiet:
                box = QMessageBox(QMessageBox.Warning, APP_NAME, msg, QMessageBox.Ok, self)
                if tb:
                    box.setDetailedText(tb)
                box.exec()
            self.statusBar().showMessage(msg)

        def prog(p, m):
            if not quiet:
                self.progress.setValue(int(p * 1000))
                self.statusBar().showMessage(m)
        w.signals.progress.connect(prog)
        w.signals.finished.connect(ok)
        w.signals.failed.connect(bad)
        self.pool.start(w)

    def cancel_task(self):
        if self.worker:
            self.worker.cancelled = True
            self.statusBar().showMessage("Cancelling…")

    # ======================================================================= preferences & help
    def pref(self, key):
        from .preferences import GENERAL
        if key == "auto_update":
            return self.qsettings.value("auto_update", True, type=bool)
        return self.qsettings.value(f"pref/{key}", GENERAL[key][1], type=bool)

    def preview_size(self) -> int:
        return int(self.qsettings.value("pref/preview_size", 1600))

    def setup_help_and_prefs(self):
        from PySide6.QtWidgets import QApplication, QFormLayout as _QF
        from .help_text import HELP
        from .preferences import PROCESSING, SAVING, set_widget_value, widget_value
        # Hover descriptions on every setting (and on its label in forms).
        for attr, (name, where, desc) in HELP.items():
            w = getattr(self, attr, None)
            if w is None:
                continue
            w.setToolTip(desc)
            for child in (getattr(w, "slider", None), getattr(w, "spin", None)):
                if child is not None:
                    child.setToolTip(desc)
        for form in self.findChildren(_QF):
            for r in range(form.rowCount()):
                li, fi = form.itemAt(r, _QF.LabelRole), form.itemAt(r, _QF.FieldRole)
                if li and fi and li.widget() and fi.widget() and fi.widget().toolTip():
                    li.widget().setToolTip(fi.widget().toolTip())
        # Remember factory values, then apply the user's saved defaults.
        self.factory_defaults = {a: widget_value(getattr(self, a)) for a in PROCESSING + SAVING}
        for a in PROCESSING + SAVING:
            v = self.qsettings.value(f"pref/{a}")
            if v is not None:
                set_widget_value(getattr(self, a), v)
        self.tooltip_filter = TooltipFilter(self)
        QApplication.instance().installEventFilter(self.tooltip_filter)
        self.apply_general_prefs()

    def apply_general_prefs(self):
        self.tooltip_filter.enabled = self.pref("tooltips")
        self.auto_update_action.blockSignals(True)
        self.auto_update_action.setChecked(self.pref("auto_update"))
        self.auto_update_action.blockSignals(False)
        if self.stack.currentIndex() == 5:
            self.update_finish_preview()

    def open_preferences(self):
        from .preferences import PreferencesDialog
        PreferencesDialog(self).exec()

    # ======================================================================= updates
    def auto_check_updates(self):
        import time
        if not self.qsettings.value("auto_update", True, type=bool):
            return
        last = float(self.qsettings.value("last_update_check", 0) or 0)
        if time.time() - last < 20 * 3600:   # at most about once a day
            return
        self.check_updates(silent=True)

    def check_updates(self, silent: bool = True):
        """Ask GitHub for the latest release (in the background, never blocks the app)."""
        import time
        from .. import updater
        if self._update_worker is not None:
            return
        w = Worker(lambda progress: updater.check_latest())
        self._update_worker = w

        def done(info):
            self._update_worker = None
            self.qsettings.setValue("last_update_check", time.time())
            if not info or not info.get("version"):
                if not silent:
                    QMessageBox.information(self, APP_NAME, "No releases are published yet.")
                return
            skipped = self.qsettings.value("skip_version", "")
            if not updater.is_newer(info["version"]):
                if not silent:
                    QMessageBox.information(self, APP_NAME, f"You have the latest version ({__version__}).")
                return
            if silent and skipped == info["version"]:
                return
            self.offer_update(info)

        def failed(msg, tb):
            self._update_worker = None
            if not silent:
                QMessageBox.warning(self, APP_NAME, f"Couldn't check for updates:\n{msg}")
        w.signals.finished.connect(done)
        w.signals.failed.connect(failed)
        self.pool.start(w)

    def offer_update(self, info):
        from .. import updater
        ok, reason = updater.can_self_update()
        box = QMessageBox(self)
        box.setWindowTitle("Update available")
        box.setIcon(QMessageBox.Information)
        box.setText(f"<b>NightStack {info['version']}</b> is available. You have {__version__}.")
        notes = info.get("notes") or ""
        box.setInformativeText((notes[:600] + ("…" if len(notes) > 600 else "")) if notes else "")
        if ok and info.get("url"):
            now = box.addButton("Update Now", QMessageBox.AcceptRole)
        else:
            now = box.addButton("Open Download Page", QMessageBox.AcceptRole)
            if reason:
                box.setDetailedText(reason)
        box.addButton("Later", QMessageBox.RejectRole)
        skip = box.addButton("Skip This Version", QMessageBox.DestructiveRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == skip:
            self.qsettings.setValue("skip_version", info["version"])
        elif clicked == now:
            if ok and info.get("url"):
                self.install_update(info)
            else:
                QDesktopServices.openUrl(QUrl(info.get("page") or updater.RELEASES_PAGE))

    def install_update(self, info):
        from PySide6.QtWidgets import QApplication
        from .. import updater
        if self.project.stacked and QMessageBox.question(
                self, APP_NAME, "NightStack will restart to finish updating. Your current stack will be "
                "closed (save your image first if you need it). Continue?") != QMessageBox.Yes:
            return

        def job(progress):
            work = updater.new_workdir()
            z = updater.download(info["url"], os.path.join(work, updater.ASSET_NAME), progress, info.get("size", 0))
            progress(0.97, "Installing update…")
            updater.install_and_relaunch(z, workdir=work)
            return True

        def done(_):
            self.statusBar().showMessage("Restarting NightStack…")
            self.close()
            QApplication.quit()
        self.run_task(job, done)

    # ======================================================================= misc
    def zoom_100(self):
        if self.stack.currentIndex() == 5 and self.project.stacked:
            if not getattr(self, "full_preview", False):
                self.render_full_preview()   # 100% only means something on the full-size image
                return
            ratio = self.viewer.devicePixelRatioF() or 1.0
            self.viewer.zoom_100(ratio)
            return
        self.viewer.zoom_100(self.project.display_scale if self.stack.currentIndex() in (2, 3) else 1.0)

    def _zoom_changed(self, z):
        self.zoom_label.setText(f"{z * 100:.0f}%  ")

    def show_welcome(self):
        img = np.zeros((900, 1400, 3), np.float32)
        rng = np.random.default_rng(3)
        for _ in range(600):
            x, y = rng.integers(0, 1400), rng.integers(0, 620)
            b = float(rng.random() ** 3)
            cv2.circle(img, (int(x), int(y)), 1 if b < 0.6 else 2, (0.6 + 0.4 * b,) * 3, -1, cv2.LINE_AA)
        xs = np.arange(1400)
        hor = (700 + 40 * np.sin(xs / 140) + 25 * np.sin(xs / 37)).astype(int)
        for x in range(1400):
            img[hor[x]:, x] = (0.05, 0.06, 0.08)
        img[:620] += np.linspace(0.02, 0.10, 620)[:, None, None] * np.array([0.5, 0.6, 1.0])
        img = (np.clip(cv2.GaussianBlur(img, (0, 0), 0.7), 0, 1) * 255).astype(np.uint8)
        cv2.putText(img, "Drop your night sky photos anywhere", (330, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (215, 225, 255), 2, cv2.LINE_AA)
        self.viewer.clear_overlays()
        self.viewer.set_image(img)
        self.view_caption.setText("")

    def start_over(self):
        self.project.cleanup()
        self.project = Project()
        self.undo_stack.clear()
        self.refresh_photo_list()
        self.refresh_calibration()
        self.steps.setCurrentRow(0)
        self.show_welcome()

    def show_guide(self):
        QMessageBox.information(self, "Quick guide", (
            "1. Photos – add 10-30 tripod shots of the same scene.\n"
            "2. Calibration – optional darks / flats / bias.\n"
            "3. Sky & Foreground – the sky is found automatically.\n"
            "    • E: erase red dots over the ground\n"
            "    • S: paint sky   • A: add a star   • H: horizon points\n"
            "    • Space+drag: pan   • Pinch / ⌘-scroll: zoom   • [ ]: brush size\n"
            "4. Align – stars are tracked and the sky rotation modelled.\n"
            "5. Stack – Sigma-clip sky + Median foreground + Minimum horizon noise\n"
            "    is the best all-round choice.\n"
            "6. Finish – adjust and save a 16-bit TIFF."))

    def about(self):
        QMessageBox.about(self, APP_NAME, f"<b>{APP_NAME} {__version__}</b><br>Friendly astro-landscape stacking."
                                          "<br><br>Built with Python, NumPy, OpenCV, LibRaw and Qt.")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = []
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isdir(p):
                files += [os.path.join(p, f) for f in sorted(os.listdir(p))]
            else:
                files.append(p)
        kind = "lights"
        if self.stack.currentIndex() == 1:
            kind = "darks"  # dropping on the calibration page: assume darks (most common)
        self._add_files(files, kind)
        if kind == "lights":
            self.steps.setCurrentRow(0)

    def closeEvent(self, e):
        if self.worker:
            self.worker.cancelled = True
            self.pool.waitForDone(3000)
        self.project.cleanup()
        super().closeEvent(e)
