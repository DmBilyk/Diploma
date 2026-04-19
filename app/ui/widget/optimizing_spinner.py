from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QConicalGradient, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

# ── Design tokens ─────────────────────────────────────────────────────────────
_BG       = "#0B0F19"
_ACCENT   = "#F59E0B"
_TEXT_PRI = "#F9FAFB"
_TEXT_SEC = "#6B7280"


class OptimizingSpinner(QWidget):
    """
    Loading / progress widget for long-running operations.
    Renders an amber arc spinner with breathing opacity and a two-line status label.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._accent_color = QColor(_ACCENT)
        self._angle         = 0
        self._opacity       = 1.0
        self._anim_step     = 0

        self._rotation_timer = QTimer(self)
        self._rotation_timer.timeout.connect(self._rotate)
        self._rotation_timer.setInterval(16)   # ~60 fps

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.setInterval(50)

        self._setup_ui()
        self.hide()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        # Spacer that acts as the canvas for paintEvent arc
        self._spinner_spacer = QWidget()
        self._spinner_spacer.setFixedSize(52, 52)
        layout.addWidget(self._spinner_spacer, 0, Qt.AlignCenter)

        # Primary message
        self.label = QLabel("Processing...")
        font = QFont()
        font.setPointSize(11)
        font.setWeight(QFont.DemiBold)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"color: {_TEXT_PRI};")
        layout.addWidget(self.label)

        # Secondary detail / progress
        self.detail_label = QLabel("")
        detail_font = QFont()
        detail_font.setPointSize(9)
        self.detail_label.setFont(detail_font)
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setStyleSheet(f"color: {_TEXT_SEC};")
        layout.addWidget(self.detail_label)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, message: str | None = None) -> None:
        if message:
            self.label.setText(message)
        self.detail_label.setText("")
        self.show()
        self._rotation_timer.start()
        self._pulse_timer.start()
        self._anim_step = 0

    def stop(self) -> None:
        self._rotation_timer.stop()
        self._pulse_timer.stop()
        self.hide()

    def update_progress(self, percent: int, detail_text: str = "") -> None:
        if detail_text:
            self.detail_label.setText(f"{detail_text}  ({percent}%)")
        else:
            self.detail_label.setText(f"{percent}%")

    # ── Animation ─────────────────────────────────────────────────────────────

    def _rotate(self) -> None:
        self._angle = (self._angle + 4) % 360
        self.update()

    def _pulse(self) -> None:
        self._anim_step += 1
        self._opacity = 0.75 + 0.25 * math.sin(self._anim_step * 0.10)

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        if not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        geom   = self._spinner_spacer.geometry()
        center = QPointF(geom.center().x(), geom.center().y())
        radius = min(geom.width(), geom.height()) / 2
        pw     = 3.5

        # Faint background ring
        bg_color = QColor(_ACCENT)
        bg_color.setAlpha(20)
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawEllipse(center, radius - pw / 2, radius - pw / 2)

        # Rotating gradient arc
        painter.save()
        painter.translate(center)
        painter.rotate(self._angle)

        gradient = QConicalGradient(QPointF(0, 0), 0)
        main_color = QColor(_ACCENT)
        main_color.setAlphaF(self._opacity)
        gradient.setColorAt(0.0, main_color)
        gradient.setColorAt(0.65, QColor(0, 0, 0, 0))
        gradient.setColorAt(1.0, main_color)

        arc_pen = QPen(QBrush(gradient), pw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(arc_pen)

        arc_rect = QRectF(
            -radius + pw, -radius + pw,
            (radius - pw) * 2, (radius - pw) * 2,
        )
        painter.drawArc(arc_rect, 0 * 16, 270 * 16)
        painter.restore()