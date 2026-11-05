"""NightStack main window: a guided, six-step workflow."""
from __future__ import annotations

import os
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QThreadPool, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..imageio import ALL_EXTS, RAW_EXTS, is_supported, load_preview, resize_max
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
    """Slider with a live value label; maps an int slider to a float range."""

    def __init__(self, lo, hi, value, step=0.01, fmt="{:.2f}", parent=None):
        super().__init__(parent)
        self.lo, self.step, self.fmt = lo, step, fmt
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(round((hi - lo) / step)))
        self.label = QLabel()
        self.label.setMinimumWidth(44)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._upd)
        self.set(value)

    def _upd(self):
        self.label.setText(self.fmt.format(self.value()))

    def value(self):
        return self.lo + self.slider.value() * self.step

    def set(self, v):
        self.slider.setValue(int(round((v - self.lo) / self.step)))
        self._upd()

    @property
    def changed(self):
        return self.slider.valueChanged


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
        self.info.setContentsMargins(12, 0, 8, 12)
        sl.addWidget(self.info)
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
        panel.setFixedWidth(370)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self.stack)
        pl.addWidget(scroll)
        root.addWidget(panel)

        self.stack.addWidget(self._build_photos())
        self.stack.addWidget(self._build_calibration())
        self.stack.addWidget(self._build_sky())
        self.stack.addWidget(self._build_align())
        self.stack.addWidget(self._build_stack())
        self.stack.addWidget(self._build_finish())

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

    # ======================================================================= menu & shortcuts
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        a = QAction("Add Photos…", self, shortcut=QKeySequence.Open, triggered=self.add_photos)
        m.addAction(a)
        m.addAction(QAction("Save Image…", self, shortcut=QKeySequence.Save, triggered=self.export))
        m.addSeparator()
        m.addAction(QAction("Start Over", self, triggered=self.start_over))
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
        files, _ = QFileDialog.getOpenFileNames(self, f"Add {kind}", start, FILTER)
        self._add_files(files, kind)

    def clear_cal(self, kind):
        getattr(self.project, kind).clear()
        self.project.invalidate("prepare")
        self.refresh_calibration()

    def refresh_calibration(self):
        for k, lab in self.cal_labels.items():
            n = len(getattr(self.project, k))
            lab.setText(f"{n} frame{'s' if n != 1 else ''}" if n else "None")

    def prepare(self):
        if not self.project.lights:
            self.steps.setCurrentRow(0)
            return
        s = self.project.settings
        s.half_size = self.fast_mode.isChecked()
        s.hot_pixels = self.hot_cb.isChecked()

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
        l.addLayout(f)
        self.align_btn = primary("Align stars")
        self.align_btn.clicked.connect(self.align)
        l.addWidget(self.align_btn)
        self.align_table = QTableWidget(0, 4)
        self.align_table.setHorizontalHeaderLabels(["Use", "Frame", "Stars", "Accuracy"])
        self.align_table.verticalHeader().hide()
        self.align_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
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
                acc = "reference"
            elif r.ok:
                acc = f"{r.rms:.2f} px  ✓"
            else:
                acc = "✗ " + r.message
            it = QTableWidgetItem(acc)
            it.setToolTip(r.message)
            if not r.ok:
                it.setForeground(Qt.red)
            t.setItem(i, 3, it)
        t.blockSignals(False)
        ok = sum(r.ok for r in res)
        bad = len(res) - ok
        self.align_stat.setText(f"{ok} of {len(res)} frames aligned." +
                                (f" {bad} frame(s) couldn't be aligned (clouds?) - they'll still help the "
                                 "foreground." if bad else " Click a row to preview."))
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
        f.addRow("Foreground", self.ground_method)
        self.kappa = LabeledSlider(1.5, 4.0, s.kappa, 0.1, "{:.1f}σ")
        self.kappa.setToolTip("Lower = more aggressive rejection of planes/satellites (sigma methods)")
        f.addRow("Clip strength", self.kappa)
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

    def enter_stack(self):
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

        def done(_):
            self.preview_src = None
            fp = self.finish_params()
            ev = self.project.suggest_exposure(fp.black_point)
            self.exposure.set(ev)
            self.show_what.setCurrentIndex(0)
            self.show_stack_view()
            self.refresh_steps()
            self.statusBar().showMessage("Stack complete! Fine-tune and save in step 6.")
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
            c = resize_max(pr.count, 1600) / max(1, pr.n_sky)
            heat = cv2.applyColorMap((np.clip(c, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            self.viewer.set_image(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), keep_view=True)
            self.view_caption.setText(f"Frames contributing sky data (red = all {pr.n_sky}, blue = few). "
                                      "The thin band above the horizon is what Minimum Horizon Noise fixes.")
            return
        src = {"result": pr.result, "sky": pr.sky_stack, "ground": pr.ground_stack,
               "single": pr.orig.read(pr.ref_idx) if what == "single" else None}[what]
        img = resize_max(src, 1600)
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
        self.exposure = LabeledSlider(-3, 10, 0, 0.05, "{:+.2f} EV")
        f.addRow("Brightness", self.exposure)
        auto = QPushButton("Auto")
        auto.clicked.connect(lambda: self.exposure.set(self.project.suggest_exposure(self.black.value())))
        f.addRow("", auto)
        self.black = LabeledSlider(0, 1, 0.5, 0.02, "{:.0%}")
        self.black.setToolTip("How much of the sky's background glow to remove (darker, punchier sky)")
        f.addRow("Black point", self.black)
        self.contrast = LabeledSlider(-0.5, 0.8, 0.0, 0.02, "{:+.2f}")
        f.addRow("Contrast", self.contrast)
        self.saturation = LabeledSlider(0, 2, 1.15, 0.05, "{:.2f}")
        f.addRow("Saturation", self.saturation)
        l.addWidget(g)
        g2 = QGroupBox("Clean-up")
        f2 = QFormLayout(g2)
        self.gradient = QCheckBox("Remove light-pollution gradient")
        f2.addRow(self.gradient)
        self.neutral = QCheckBox("Neutral sky colour (remove colour cast)")
        self.neutral.setChecked(True)
        f2.addRow(self.neutral)
        self.highlights = QCheckBox("Protect bright stars && lights")
        self.highlights.setChecked(True)
        f2.addRow(self.highlights)
        self.sky_dn = LabeledSlider(0, 1, 0, 0.05, "{:.0%}")
        f2.addRow("Extra sky smoothing", self.sky_dn)
        self.gnd_dn = LabeledSlider(0, 1, 0, 0.05, "{:.0%}")
        f2.addRow("Foreground smoothing", self.gnd_dn)
        l.addWidget(g2)
        for wdg in (self.exposure, self.black, self.contrast, self.saturation, self.sky_dn, self.gnd_dn):
            wdg.changed.connect(self.finish_timer.start)
        for cb in (self.gradient, self.neutral, self.highlights):
            cb.toggled.connect(self.finish_timer.start)
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
        self.save_btn = primary("Save Image…")
        self.save_btn.clicked.connect(self.export)
        l.addWidget(self.save_btn)
        return w

    def finish_params(self) -> FinishParams:
        return FinishParams(exposure=self.exposure.value(), black_point=self.black.value(),
                            saturation=self.saturation.value(), contrast=self.contrast.value(),
                            remove_gradient=self.gradient.isChecked(), neutralize=self.neutral.isChecked(),
                            sky_denoise=self.sky_dn.value(), ground_denoise=self.gnd_dn.value(),
                            protect_highlights=self.highlights.isChecked(), linear_output=self.linear.isChecked())

    def enter_finish(self):
        self.viewer.edit_enabled = False
        self.viewer.set_tool(Tool.PAN)
        self.viewer.clear_overlays()
        if self.project.stacked:
            self.update_finish_preview()

    def update_finish_preview(self):
        if not self.project.stacked or self.stack.currentIndex() != 5:
            return
        img, m = self.project.preview_inputs(1600)
        self.viewer.set_image(apply_finish(img, m, self.finish_params()), keep_view=True)
        self.view_caption.setText("Preview (final image is saved at full resolution)")

    def show_before(self):
        if not self.project.stacked:
            return
        img = resize_max(self.project.orig.read(self.project.ref_idx), 1600)
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
                self.align()
        elif i == 4:
            self.enter_stack()
            if not self.project.stacked:
                self.do_stack()
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

    # ======================================================================= misc
    def zoom_100(self):
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
