"""
app/ui/loading_window.py
========================
Loading window styled to match the main application.
Shown during initial database bootstrap instead of a splash screen.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
    QFrame,
    QHBoxLayout,
)

_BG       = "#1A2332"
_SIDEBAR  = "#1C2833"
_ACCENT   = "#1ABC9C"
_TEXT     = "#ECF0F1"
_SUBTEXT  = "#7F8C8D"
_BORDER   = "#2E4057"


class LoadingWindow(QWidget):
    """
    Standalone loading window updated by ``update_progress``.
    The caller closes it once bootstrapping finishes.
    """

    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("InvestPortfolio Optimizer — Ініціалізація")
        self.setFixedSize(520, 360)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self._build_ui()
        self._center_on_screen()

        # Animate dots in the title while the worker is running.
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_dots)
        self._dot_timer.start(500)

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {_BG};
                color: {_TEXT};
                font-family: Arial, sans-serif;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(72)
        header.setStyleSheet(f"background-color: {_SIDEBAR}; border-bottom: 1px solid {_BORDER};")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(28, 0, 28, 0)

        title_block = QVBoxLayout()
        title_block.setSpacing(2)

        app_name = QLabel("Portfolio")
        app_name.setStyleSheet("color: white; font-size: 18px; font-weight: 800; letter-spacing: 1px; background: transparent;")

        app_sub = QLabel("OPTIMIZER")
        app_sub.setStyleSheet(f"color: {_ACCENT}; font-size: 9px; letter-spacing: 2.5px; background: transparent;")

        title_block.addStretch()
        title_block.addWidget(app_name)
        title_block.addWidget(app_sub)
        title_block.addStretch()
        h_layout.addLayout(title_block)
        h_layout.addStretch()

        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet(f"color: {_ACCENT}; font-size: 14px; background: transparent;")
        h_layout.addWidget(self._status_dot)

        root.addWidget(header)

        # ── Body ──────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background-color: {_BG};")
        b_layout = QVBoxLayout(body)
        b_layout.setContentsMargins(36, 32, 36, 36)
        b_layout.setSpacing(0)

        self._main_label = QLabel("Ініціалізація бази даних")
        self._main_label.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #ECF0F1; letter-spacing: 0.5px;"
        )
        b_layout.addWidget(self._main_label)

        b_layout.addSpacing(8)

        self._step_label = QLabel("Підготовка до завантаження...")
        self._step_label.setStyleSheet(
            f"font-size: 11px; color: {_SUBTEXT}; min-height: 28px;"
        )
        self._step_label.setWordWrap(True)
        b_layout.addWidget(self._step_label)

        b_layout.addSpacing(28)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {_BORDER};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {_ACCENT};
                border-radius: 3px;
            }}
        """)
        b_layout.addWidget(self._bar)

        b_layout.addSpacing(10)

        self._pct_label = QLabel("0%")
        self._pct_label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {_ACCENT};")
        self._pct_label.setAlignment(Qt.AlignRight)
        b_layout.addWidget(self._pct_label)

        b_layout.addStretch()

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {_BORDER}; border: none;")
        line.setFixedHeight(1)
        b_layout.addWidget(line)

        b_layout.addSpacing(14)

        hint = QLabel("Це відбувається лише при першому запуску. Надалі програма стартує миттєво.")
        hint.setStyleSheet(f"font-size: 10px; color: {_BORDER}; color: #3D566E;")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        b_layout.addWidget(hint)

        root.addWidget(body, 1)

    # ── Public API ──────────────────────────────────────────────────────────

    def update_progress(self, percent: int, message: str) -> None:
        """Update progress from ``DataLoaderWorker`` signals."""
        self._bar.setValue(percent)
        self._pct_label.setText(f"{percent}%")
        self._step_label.setText(message)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _animate_dots(self) -> None:
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self._main_label.setText(f"Ініціалізація бази даних{dots}")

    def _center_on_screen(self) -> None:
        from PySide6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().availableGeometry()
        x = (screen.width()  - self.width())  // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def closeEvent(self, event):
        self._dot_timer.stop()
        self.closed.emit()
        super().closeEvent(event)
