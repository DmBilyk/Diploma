from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ── Design tokens ────────────────────────────────────────────────────────────
_BG      = "#0B0F19"
_SURFACE = "#111827"
_BORDER  = "#1F2937"
_ACCENT  = "#F59E0B"
_TEXT_PRI = "#F9FAFB"
_TEXT_SEC = "#6B7280"
_POSITIVE = "#10B981"

PERIODS: list[tuple[str, Optional[int]]] = [
    ("1M",  30),
    ("6M",  180),
    ("1Y",  365),
    ("5Y",  365 * 5),
    ("MAX", None),
]


# ── Chart theme ───────────────────────────────────────────────────────────────

@dataclass
class ChartTheme:
    fig_bg:            str
    ax_bg:             str
    text:              str
    grid:              str
    line:              str = _ACCENT
    positive_fill:     str = field(init=False)
    annotation_bg:     str = field(init=False)
    annotation_border: str = field(init=False)

    def __post_init__(self) -> None:
        self.positive_fill    = _POSITIVE
        self.annotation_bg    = "#1C2333"
        self.annotation_border = _BORDER


def _chart_theme() -> ChartTheme:
    """Fixed dark fintech theme — independent of system palette."""
    return ChartTheme(
        fig_bg=_BG,
        ax_bg=_BG,
        text=_TEXT_PRI,
        grid=_BORDER,
        line=_ACCENT,
    )


# ── Canvas ────────────────────────────────────────────────────────────────────

class _Canvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        fig = Figure(figsize=(7, 4), dpi=100)
        fig.patch.set_facecolor(_BG)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background-color: {_BG};")


# ── Renderer ──────────────────────────────────────────────────────────────────

class _ChartRenderer:
    """Draws the price line and manages hover annotation."""

    _HOVER_TOLERANCE_FACTOR: float = 2.0

    def __init__(self, figure: Figure, canvas: FigureCanvas) -> None:
        self._fig    = figure
        self._canvas = canvas
        self._ax     = figure.add_subplot(111)
        self._annotation = None
        self._x_num: Optional[np.ndarray] = None
        self._y:     Optional[np.ndarray] = None
        self._dates  = None

        # Paint axes dark immediately so there is no white flash before first render
        self._ax.set_facecolor(_BG)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_visible(False)
        canvas.draw()

    def render(self, ticker: str, data: pd.DataFrame) -> None:
        theme = _chart_theme()
        self._annotation = None
        self._cache_series(data)
        self._draw(ticker, data, theme)

    def handle_hover(self, event) -> None:
        if event.inaxes != self._ax or self._x_num is None:
            self._clear_annotation()
            return

        distances = np.abs(self._x_num - event.xdata)
        idx = int(np.argmin(distances))
        span      = self._x_num[-1] - self._x_num[0]
        tolerance = span / len(self._x_num) * self._HOVER_TOLERANCE_FACTOR
        if distances[idx] > tolerance:
            self._clear_annotation()
            return
        self._update_annotation(idx)

    # ── Private ───────────────────────────────────────────────────────────────

    def _cache_series(self, data: pd.DataFrame) -> None:
        self._dates = data["date"]
        self._x_num = mdates.date2num(data["date"])
        self._y     = data["adj_close"].values

    def _draw(self, ticker: str, data: pd.DataFrame, t: ChartTheme) -> None:
        ax = self._ax
        ax.cla()

        self._fig.patch.set_facecolor(t.fig_bg)
        ax.set_facecolor(t.ax_bg)

        # Determine trend colour
        first_price = data["adj_close"].iloc[0]
        last_price  = data["adj_close"].iloc[-1]
        line_color  = _POSITIVE if last_price >= first_price else "#F43F5E"

        ax.plot(
            data["date"], data["adj_close"],
            color=line_color, linewidth=1.5,
        )
        ax.fill_between(
            data["date"], data["adj_close"],
            alpha=0.07, color=line_color,
        )

        # Performance annotation (top-right)
        pct = (last_price / first_price - 1.0) * 100
        pct_color = _POSITIVE if pct >= 0 else "#F43F5E"
        sign = "+" if pct >= 0 else ""
        ax.annotate(
            f"{sign}{pct:.2f}%",
            xy=(1.0, 1.02),
            xycoords="axes fraction",
            ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            color=pct_color,
        )

        ax.set_title(
            f"{ticker}    ${last_price:,.2f}",
            fontsize=11, fontweight="700", pad=10,
            color=t.text, loc="left",
        )

        # Axes styling — Bloomberg-style: price on right
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", colors=t.text, labelsize=8, length=0)
        ax.tick_params(axis="x", colors=_TEXT_SEC, labelsize=8, length=0)
        ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}")

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

        ax.margins(x=0.01, y=0.20)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_edgecolor(_BORDER)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.grid(True, axis="y", linestyle="-", linewidth=0.4,
                alpha=0.4, color=t.grid)
        ax.grid(False, axis="x")

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    def _update_annotation(self, idx: int) -> None:
        self._clear_annotation(redraw=False)
        date_str = pd.Timestamp(self._dates.iloc[idx]).strftime("%d %b %Y")
        price    = self._y[idx]
        label    = f"{date_str}\n${price:,.2f}"

        ax_frac  = (self._x_num[idx] - self._x_num[0]) / (self._x_num[-1] - self._x_num[0])
        x_offset = -80 if ax_frac > 0.75 else 12

        self._annotation = self._ax.annotate(
            label,
            xy=(self._x_num[idx], price),
            xytext=(x_offset, 12),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="#1C2333",
                ec=_BORDER,
                alpha=0.95,
                linewidth=0.8,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=_TEXT_SEC,
                connectionstyle="arc3,rad=0.15",
                lw=0.8,
            ),
            fontsize=8.5,
            color=_TEXT_PRI,
        )
        self._canvas.draw_idle()

    def _clear_annotation(self, redraw: bool = True) -> None:
        if self._annotation:
            self._annotation.remove()
            self._annotation = None
            if redraw:
                self._canvas.draw_idle()


# ── Public widget ─────────────────────────────────────────────────────────────

class StockChartWidget(QWidget):
    """
    Self-contained Qt widget: header with period toggles + price chart.

    Usage
    -----
    >>> widget = StockChartWidget(parent)
    >>> widget.load(ticker="AAPL", data=df)   # df: columns ['date', 'adj_close']
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_data: Optional[pd.DataFrame] = None
        self._ticker: str = ""
        self._active_period: str = "1Y"
        self._build_ui()

    def load(self, ticker: str, data: pd.DataFrame) -> None:
        self._ticker    = ticker
        self._full_data = data.copy()
        self._render_current_period()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        canvas = _Canvas(self)
        self._renderer = _ChartRenderer(canvas.figure, canvas)
        canvas.mpl_connect("motion_notify_event", self._renderer.handle_hover)
        root.addWidget(canvas)

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("chartHeader")
        bar.setFixedHeight(44)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(8)

        layout.addStretch()

        self._period_buttons: dict[str, QPushButton] = {}
        for label, _ in PERIODS:
            btn = QPushButton(label)
            btn.setObjectName("periodBtn")
            btn.setCheckable(True)
            btn.setFixedSize(36, 26)
            btn.clicked.connect(lambda _, l=label: self._on_period_clicked(l))
            self._period_buttons[label] = btn
            layout.addWidget(btn)

        self._period_buttons["1Y"].setChecked(True)

        bar.setStyleSheet(f"""
            QWidget#chartHeader {{
                background-color: {_SURFACE};
                border-bottom: 1px solid {_BORDER};
            }}
            QPushButton#periodBtn {{
                background: transparent;
                border: none;
                border-radius: 3px;
                font-size: 10px;
                font-weight: 600;
                color: {_TEXT_SEC};
                letter-spacing: 0.3px;
            }}
            QPushButton#periodBtn:hover {{
                background-color: rgba(255,255,255,0.05);
                color: {_TEXT_PRI};
            }}
            QPushButton#periodBtn:checked {{
                background-color: rgba(245,158,11,0.15);
                color: {_ACCENT};
            }}
        """)
        return bar

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_period_clicked(self, label: str) -> None:
        self._active_period = label
        for lbl, btn in self._period_buttons.items():
            btn.setChecked(lbl == label)
        self._render_current_period()

    def _render_current_period(self) -> None:
        if self._full_data is None or self._full_data.empty:
            return

        days = dict(PERIODS).get(self._active_period)
        if days is None:
            data = self._full_data
        else:
            last_date = pd.to_datetime(self._full_data["date"]).max()
            cutoff    = last_date - pd.Timedelta(days=days)
            data      = self._full_data[pd.to_datetime(self._full_data["date"]) >= cutoff]

        if data.empty:
            data = self._full_data

        self._renderer.render(self._ticker, data)