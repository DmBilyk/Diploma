"""
app/ui/loading_window.py
========================
Вікно завантаження, стилізоване під головне вікно програми.
Показується замість splash screen під час первинного завантаження БД.
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

_BG       = "#1A2332"   # фон — трохи темніший за sidebar
_SIDEBAR  = "#1C2833"   # колір sidebar головного вікна
_ACCENT   = "#1ABC9C"   # той самий акцент
_TEXT     = "#ECF0F1"
_SUBTEXT  = "#7F8C8D"
_BORDER   = "#2E4057"


class LoadingWindow(QWidget):
    """
    Окреме вікно завантаження.
    Викликати: window.update_progress(percent, message)
    Після завершення — вікно само закривається через close().
    """

    # Якщо захочете підписатися з main.py
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("InvestPortfolio Optimizer — Ініціалізація")
        self.setFixedSize(520, 360)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self._build_ui()
        self._center_on_screen()

        # Анімація крапок у заголовку ("Завантаження.", "..", "...")
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_dots)
        self._dot_timer.start(500)

    # ── Побудова UI ────────────────────────────────────────────────────────

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

        # ── Шапка (стиль sidebar) ─────────────────────────────────────
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

        # ── Тіло ──────────────────────────────────────────────────────
        body = QWidget()
        body.setStyleSheet(f"background-color: {_BG};")
        b_layout = QVBoxLayout(body)
        b_layout.setContentsMargins(36, 32, 36, 36)
        b_layout.setSpacing(0)

        # Анімований заголовок
        self._main_label = QLabel("Ініціалізація бази даних")
        self._main_label.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #ECF0F1; letter-spacing: 0.5px;"
        )
        b_layout.addWidget(self._main_label)

        b_layout.addSpacing(8)

        # Поточний крок
        self._step_label = QLabel("Підготовка до завантаження...")
        self._step_label.setStyleSheet(
            f"font-size: 11px; color: {_SUBTEXT}; min-height: 28px;"
        )
        self._step_label.setWordWrap(True)
        b_layout.addWidget(self._step_label)

        b_layout.addSpacing(28)

        # Прогрес-бар у стилі акценту
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

        # Відсоток
        self._pct_label = QLabel("0%")
        self._pct_label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {_ACCENT};")
        self._pct_label.setAlignment(Qt.AlignRight)
        b_layout.addWidget(self._pct_label)

        b_layout.addStretch()

        # Роздільник
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {_BORDER}; border: none;")
        line.setFixedHeight(1)
        b_layout.addWidget(line)

        b_layout.addSpacing(14)

        # Підказка
        hint = QLabel("Це відбувається лише при першому запуску. Надалі програма стартує миттєво.")
        hint.setStyleSheet(f"font-size: 10px; color: {_BORDER}; color: #3D566E;")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignCenter)
        b_layout.addWidget(hint)

        root.addWidget(body, 1)

    # ── Публічний API ──────────────────────────────────────────────────────

    def update_progress(self, percent: int, message: str) -> None:
        """Викликається з DataLoaderWorker через сигнал."""
        self._bar.setValue(percent)
        self._pct_label.setText(f"{percent}%")
        self._step_label.setText(message)

    # ── Службові методи ────────────────────────────────────────────────────

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