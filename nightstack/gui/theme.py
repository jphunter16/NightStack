"""Dark, astro-friendly look (dark UI keeps your night vision and lets the photo shine)."""
from PySide6.QtGui import QColor, QPalette

ACCENT = "#5b8cff"
STYLE = f"""
QWidget {{ background: #15171c; color: #e6e8ee; font-size: 13px; }}
QFrame#Sidebar {{ background: #101216; border-right: 1px solid #262a33; }}
QFrame#Panel {{ background: #181b21; border-left: 1px solid #262a33; }}
QLabel#Title {{ font-size: 19px; font-weight: 600; color: #ffffff; padding: 2px 0 2px 0; }}
QLabel#Hint {{ color: #9aa3b5; font-size: 12px; }}
QLabel#Stat {{ color: #c7d2ff; font-size: 12px; }}
QLabel#Drop {{ border: 2px dashed #3a4150; border-radius: 12px; color: #9aa3b5; padding: 18px;
              font-size: 14px; background: #1b1f27; }}
QGroupBox {{ border: 1px solid #2a2f3a; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #aab4c8; }}
QPushButton {{ background: #262b35; border: 1px solid #343a47; border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ background: #2f3542; }}
QPushButton:disabled {{ color: #5d6472; background: #1d2027; }}
QPushButton#Primary {{ background: {ACCENT}; border: none; color: white; font-weight: 600; padding: 9px 14px;
                      font-size: 14px; }}
QPushButton#Primary:hover {{ background: #6f9bff; }}
QPushButton#Primary:disabled {{ background: #2a3350; color: #7f8aa8; }}
QToolButton {{ background: #222731; border: 1px solid #323846; border-radius: 8px; padding: 6px; }}
QToolButton:checked {{ background: #2d3c66; border: 1px solid {ACCENT}; }}
QToolButton:hover {{ background: #2a303c; }}
QListWidget#Steps {{ background: transparent; border: none; outline: 0; }}
QListWidget#Steps::item {{ padding: 11px 10px; margin: 2px 6px; border-radius: 8px; color: #c9cfdb; }}
QListWidget#Steps::item:selected {{ background: #243256; color: white; }}
QListWidget#Steps::item:disabled {{ color: #4d5361; }}
QListWidget, QTableWidget {{ background: #12151a; border: 1px solid #2a2f3a; border-radius: 6px; }}
QHeaderView::section {{ background: #1d2129; color: #aab4c8; border: none; padding: 4px; }}
QDoubleSpinBox {{ background: #222731; border: 1px solid #343a47; border-radius: 6px; padding-right: 4px; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; width: 0px; height: 0px;
                                                         border: none; }}
QDoubleSpinBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox, QSpinBox {{ background: #222731; border: 1px solid #343a47; border-radius: 6px;
                                      padding: 4px 6px; }}
QSlider {{ min-height: 22px; background: transparent; }}
QSlider::groove:horizontal {{ height: 4px; background: #2c3240; border-radius: 2px; margin: 0 7px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; border: 2px solid #c9d6ff; width: 12px; height: 12px;
                             margin: -6px -7px; border-radius: 8px; }}
QSlider::handle:horizontal:hover {{ background: #7aa2ff; }}
QSlider::sub-page:horizontal {{ background: #3e5cab; border-radius: 2px; }}
QProgressBar {{ background: #1d2129; border: 1px solid #2a2f3a; border-radius: 5px; text-align: center;
               height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
QScrollArea {{ border: none; }}
QStatusBar {{ background: #101216; color: #9aa3b5; }}
QGraphicsView {{ background: #0b0c0f; border: none; }}
QToolTip {{ background: #222731; color: #e6e8ee; border: 1px solid #3a4150; }}
"""


def dark_palette() -> QPalette:
    p = QPalette()
    for role, c in ((QPalette.Window, "#15171c"), (QPalette.WindowText, "#e6e8ee"), (QPalette.Base, "#12151a"),
                    (QPalette.AlternateBase, "#1b1f27"), (QPalette.Text, "#e6e8ee"), (QPalette.Button, "#262b35"),
                    (QPalette.ButtonText, "#e6e8ee"), (QPalette.Highlight, ACCENT),
                    (QPalette.HighlightedText, "#ffffff"), (QPalette.ToolTipBase, "#222731"),
                    (QPalette.ToolTipText, "#e6e8ee")):
        p.setColor(role, QColor(c))
    return p
