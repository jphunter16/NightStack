"""Preferences window (⌘,): app behaviour, default processing settings and a settings reference."""
from __future__ import annotations

import os
import shutil

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QPushButton, QRadioButton, QScrollArea, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..imageio import default_cache_root
from .help_text import HELP

# App behaviour (not tied to a panel widget): key -> (label, default, description)
GENERAL = {
    "tooltips": ("Show descriptions when hovering over settings", True,
                 "Turn off once you know the app, to hide the pop-up help."),
    "auto_update": ("Check for updates automatically", True,
                    "Checks GitHub about once a day when NightStack starts. Help → Check for Updates works "
                    "either way."),
    "auto_align": ("Start aligning automatically when opening step 4", False,
                   "Off: nothing happens until you click “Align stars”."),
    "auto_stack": ("Start stacking automatically when opening step 5", False,
                   "Off: nothing happens until you click “Stack!”."),
}
PREVIEW_SIZES = [(1200, "Small (fastest sliders)"), (1600, "Medium (default)"), (2400, "Large (sharper preview)")]

# Panel widgets whose value is remembered as your default.
PROCESSING = ["fast_mode", "hot_cb", "demosaic", "align_model", "interp", "auto_reject", "sky_method",
              "ground_method", "kappa", "small_stack_cb", "normalize", "weighting", "match_fg", "blend_mode",
              "horizon_strength", "feather", "margin"]
SAVING = ["fmt", "layers"]


def widget_value(w):
    if isinstance(w, (QCheckBox, QRadioButton)):
        return bool(w.isChecked())
    if isinstance(w, QComboBox):
        return int(w.currentIndex())
    if isinstance(w, QSpinBox):
        return int(w.value())
    return float(w.value())             # LabeledSlider


def set_widget_value(w, v):
    if v is None:
        return
    if isinstance(w, (QCheckBox, QRadioButton)):
        w.setChecked(v in (True, "true", "1", 1))
    elif isinstance(w, QComboBox):
        i = int(v)
        if 0 <= i < w.count():
            w.setCurrentIndex(i)
    elif isinstance(w, QSpinBox):
        w.setValue(int(float(v)))
    else:
        w.set(float(v))


def clone_editor(w):
    """A fresh editor of the same kind as a panel widget."""
    from .main_window import LabeledSlider
    if isinstance(w, (QCheckBox, QRadioButton)):
        return QCheckBox()
    if isinstance(w, QComboBox):
        c = QComboBox()
        for i in range(w.count()):
            c.addItem(w.itemText(i))
        return c
    if isinstance(w, QSpinBox):
        s = QSpinBox()
        s.setRange(w.minimum(), w.maximum())
        s.setSuffix(w.suffix())
        return s
    if isinstance(w, LabeledSlider):
        return LabeledSlider(w.lo, w.hi, w.value(), w.step, w.fmt)
    raise TypeError(type(w))


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


class PreferencesDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Preferences")
        self.resize(640, 560)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)

        # General ----------------------------------------------------------------
        g = QWidget()
        gl = QVBoxLayout(g)
        self.general = {}
        for key, (label, default, desc) in GENERAL.items():
            cb = QCheckBox(label)
            cb.setChecked(win.pref(key))
            cb.setToolTip(desc)
            gl.addWidget(cb)
            note = QLabel(desc)
            note.setObjectName("Hint")
            note.setWordWrap(True)
            note.setContentsMargins(26, 0, 0, 6)
            gl.addWidget(note)
            self.general[key] = cb
        f = QFormLayout()
        self.preview = QComboBox()
        for size, label in PREVIEW_SIZES:
            self.preview.addItem(label, size)
        self.preview.setCurrentIndex(max(0, self.preview.findData(win.preview_size())))
        self.preview.setToolTip("Size of the live preview in steps 5-6. Smaller = faster sliders. "
                                "Use “Render 100% full preview” to inspect full resolution.")
        f.addRow("Live preview size", self.preview)
        gl.addLayout(f)
        gl.addStretch(1)
        tabs.addTab(g, "General")

        # Processing defaults / Saving ------------------------------------------------
        self.editors = {}
        tabs.addTab(self._defaults_tab(PROCESSING, "These are used every time NightStack opens. You can still "
                                                   "change them for a single project in each step."), "Processing defaults")
        tabs.addTab(self._defaults_tab(SAVING, "Defaults for step 6. Tone and clean-up always start at the "
                                               "baseline (no changes) so you can edit in Lightroom."), "Saving")

        # All settings reference -------------------------------------------------------
        r = QWidget()
        rl = QVBoxLayout(r)
        rl.addWidget(QLabel("Every setting in NightStack, where to find it and what it does."))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search settings…")
        rl.addWidget(self.search)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Setting", "Where", "What it does"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setWordWrap(True)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)
        rows = sorted(HELP.values(), key=lambda t: (t[1], t[0]))
        self.table.setRowCount(len(rows))
        for i, (name, where, desc) in enumerate(rows):
            for c, txt in enumerate((name, where, desc)):
                it = QTableWidgetItem(txt)
                it.setToolTip(desc)
                self.table.setItem(i, c, it)
        self.table.resizeRowsToContents()
        self.search.textChanged.connect(self._filter)
        rl.addWidget(self.table, 1)
        tabs.addTab(r, "All settings")

        # Storage ------------------------------------------------------------------------
        st = QWidget()
        sl = QVBoxLayout(st)
        root = default_cache_root()
        sl.addWidget(QLabel("While you work, frames are kept in a temporary cache (deleted when you quit)."))
        self.cache_label = QLabel()
        self.cache_label.setWordWrap(True)
        self.cache_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sl.addWidget(self.cache_label)
        row = QHBoxLayout()
        b = QPushButton("Show in Finder")
        b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(root)))
        row.addWidget(b)
        b = QPushButton("Clear leftover cache files")
        b.setToolTip("Deletes cache folders left behind by a crash. The current project is kept.")
        b.clicked.connect(self._clear_cache)
        row.addWidget(b)
        row.addStretch(1)
        sl.addLayout(row)
        sl.addStretch(1)
        self._update_cache_label()
        tabs.addTab(st, "Storage")

        # Buttons --------------------------------------------------------------------------
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)
        lay.addWidget(bb)

    def _defaults_tab(self, attrs, note):
        page = QWidget()
        pl = QVBoxLayout(page)
        n = QLabel(note)
        n.setWordWrap(True)
        n.setObjectName("Hint")
        pl.addWidget(n)
        form = QFormLayout()
        for attr in attrs:
            w = getattr(self.win, attr)
            ed = clone_editor(w)
            set_widget_value(ed, widget_value(w))
            name, _, desc = HELP.get(attr, (attr, "", ""))
            label = QLabel(name)
            label.setToolTip(desc)
            ed.setToolTip(desc)
            form.addRow(label, ed)
            self.editors[attr] = ed
        inner = QWidget()
        inner.setLayout(form)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        pl.addWidget(sc, 1)
        return page

    def _filter(self, text):
        t = text.lower().strip()
        for i in range(self.table.rowCount()):
            row = " ".join(self.table.item(i, c).text().lower() for c in range(3))
            self.table.setRowHidden(i, bool(t) and t not in row)

    def _update_cache_label(self):
        root = default_cache_root()
        self.cache_label.setText(f"Cache folder: {root}\nCurrently using {dir_size(root) / 1e9:.2f} GB")

    def _clear_cache(self):
        root = default_cache_root()
        keep = self.win.project.cache_dir
        for d in os.listdir(root):
            p = os.path.join(root, d)
            if os.path.isdir(p) and (not keep or os.path.abspath(p) != os.path.abspath(keep)):
                shutil.rmtree(p, ignore_errors=True)
        self._update_cache_label()

    def restore_defaults(self):
        for key, (_, default, _) in GENERAL.items():
            self.general[key].setChecked(default)
        self.preview.setCurrentIndex(self.preview.findData(1600))
        for attr, ed in self.editors.items():
            set_widget_value(ed, self.win.factory_defaults.get(attr))

    def accept(self):
        qs = self.win.qsettings
        for key, cb in self.general.items():
            qs.setValue(f"pref/{key}", cb.isChecked())
        qs.setValue("auto_update", self.general["auto_update"].isChecked())
        qs.setValue("pref/preview_size", self.preview.currentData())
        for attr, ed in self.editors.items():
            v = widget_value(ed)
            qs.setValue(f"pref/{attr}", v)
            set_widget_value(getattr(self.win, attr), v)   # apply now as well
        self.win.apply_general_prefs()
        super().accept()
