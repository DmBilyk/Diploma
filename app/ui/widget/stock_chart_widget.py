from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (                        # pylint: disable=no-name-in-module
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Константи
# ═══════════════════════════════════════════════════════════════════════════════

PERIODS: list[tuple[str, Optional[int]]] = [
    ("1M",  30),
    ("6M",  180),
    ("1Y",  365),
    ("5Y",  365 * 5),
    ("MAX", None),
]

_ACCENT = "#1ABC9C"


# ═══════════════════════════════════════════════════════════════════════════════
# Конфігурація
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChartTheme:
    """Кольори графіка залежно від теми."""

    fig_bg: str
    ax_bg: str
    text: str
    grid: str
    line: str = "#2980B9"
    annotation_bg: str = field(init=False)
    annotation_border: str = field(init=False)

    def __post_init__(self) -> None:
        dark = self.fig_bg != "white"
        self.annotation_bg = "#2A2A2A" if dark else "white"
        self.annotation_border = "#555555" if dark else "#AAAAAA"


def _detect_theme() -> ChartTheme:
    """Повертає ChartTheme відповідно до поточної палітри Qt."""
    palette = QApplication.instance().palette()
    bg = palette.color(QPalette.Window)
    text = palette.color(QPalette.WindowText)

    dark = bg.lightness() < 128

    return ChartTheme(
        fig_bg=bg.name(),  # Беремо точний системний колір фону вікна
        ax_bg="#222222" if dark else "#F7F7F7",
        text=text.name(),  # Беремо точний системний колір тексту
        grid="#3A3A3A" if dark else "#DDDDDD"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Внутрішнє полотно
# ═══════════════════════════════════════════════════════════════════════════════

class _Canvas(FigureCanvas):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        fig = Figure(figsize=(7, 4), dpi=100)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ═══════════════════════════════════════════════════════════════════════════════
# Рендерер
# ═══════════════════════════════════════════════════════════════════════════════

class _ChartRenderer:
    """
    Малює лінію цін та управляє hover-анотацією.

    Не залежить від Qt-віджетів — лише matplotlib.
    """

    _HOVER_TOLERANCE_FACTOR: float = 2.0

    def __init__(self, figure: Figure, canvas: FigureCanvas) -> None:
        self._fig = figure
        self._canvas = canvas
        self._ax = figure.add_subplot(111)
        self._annotation = None
        self._x_num: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._dates = None

    # ── Публічний API ────────────────────────────────────────────────────────

    def render(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Відображає графік для вказаного тікера.

        Parameters
        ----------
        ticker : str
            Символ активу.
        data : pd.DataFrame
            Колонки: 'date' (datetime-like), 'adj_close' (float).
        """
        theme = _detect_theme()
        self._annotation = None
        self._cache_series(data)
        self._apply_theme(theme)
        self._draw(ticker, data, theme)

    def handle_hover(self, event) -> None:
        """Обробляє motion_notify_event від FigureCanvas."""
        if event.inaxes != self._ax or self._x_num is None:
            self._clear_annotation()
            return

        distances = np.abs(self._x_num - event.xdata)
        idx = int(np.argmin(distances))

        span = self._x_num[-1] - self._x_num[0]
        tolerance = span / len(self._x_num) * self._HOVER_TOLERANCE_FACTOR
        if distances[idx] > tolerance:
            self._clear_annotation()
            return

        self._update_annotation(idx)

    # ── Приватні методи ──────────────────────────────────────────────────────

    def _cache_series(self, data: pd.DataFrame) -> None:
        self._dates = data["date"]
        self._x_num = mdates.date2num(data["date"])
        self._y = data["adj_close"].values

    def _apply_theme(self, t: ChartTheme) -> None:
        ax = self._ax
        self._fig.patch.set_facecolor(t.fig_bg)
        ax.set_facecolor(t.ax_bg)
        for spine in ax.spines.values():
            spine.set_edgecolor(t.grid)
        for item in (ax.xaxis.label, ax.yaxis.label, ax.title):
            item.set_color(t.text)
        ax.tick_params(colors=t.text, which="both")

    def _draw(self, ticker: str, data: pd.DataFrame, t: ChartTheme) -> None:
        ax = self._ax
        ax.cla()
        self._apply_theme(t)

        ax.plot(
            data["date"], data["adj_close"],
            color=t.line, linewidth=1.5, label="Adj Close",
        )
        ax.fill_between(data["date"], data["adj_close"], alpha=0.06, color=t.line)

        ax.set_title(f"{ticker}  ·  Price History", fontsize=11, fontweight="bold", pad=10, color=t.text)


        ax.margins(x=0, y=0.25)


        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_edgecolor(t.grid)


        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Price (USD)", fontsize=9, labelpad=6, color=t.text)


        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))

        # ──────────────────────────────────────────────

        ax.legend(loc="upper left", framealpha=0.0, labelcolor=t.text, fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color=t.grid)
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

        self._fig.tight_layout(pad=1.5)
        self._canvas.draw()

    def _update_annotation(self, idx: int) -> None:
        theme = _detect_theme()
        self._clear_annotation(redraw=False)

        date_str = pd.Timestamp(self._dates.iloc[idx]).strftime("%d %b %Y")
        price = self._y[idx]
        label = f"{date_str}\n${price:,.2f}"

        ax_frac = (self._x_num[idx] - self._x_num[0]) / (self._x_num[-1] - self._x_num[0])
        x_offset = -80 if ax_frac > 0.75 else 12

        self._annotation = self._ax.annotate(
            label,
            xy=(self._x_num[idx], price),
            xytext=(x_offset, 12),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.45",
                fc=theme.annotation_bg,
                ec=theme.annotation_border,
                alpha=0.92,
                linewidth=0.8,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=theme.annotation_border,
                connectionstyle="arc3,rad=0.15",
                lw=0.8,
            ),
            fontsize=8.5,
            color=theme.text,
        )
        self._canvas.draw_idle()

    def _clear_annotation(self, redraw: bool = True) -> None:
        if self._annotation:
            self._annotation.remove()
            self._annotation = None
            if redraw:
                self._canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# Публічний віджет
# ═══════════════════════════════════════════════════════════════════════════════

class StockChartWidget(QWidget):
    """
    Самодостатній Qt-віджет: заголовок + перемикачі діапазону + графік.

    Використання
    ------------
    >>> widget = StockChartWidget(parent)
    >>> widget.load(ticker="AAPL", data=df)   # df: columns ['date', 'adj_close']
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_data: Optional[pd.DataFrame] = None
        self._ticker: str = ""
        self._active_period: str = "1Y"

        self._build_ui()

    # ── Публічний API ────────────────────────────────────────────────────────

    def load(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Завантажує нові дані та перемальовує графік.

        Parameters
        ----------
        ticker : str
            Символ акції (відображається в заголовку).
        data : pd.DataFrame
            Повна доступна історія. Колонки: 'date', 'adj_close'.
        """
        self._ticker = ticker
        self._full_data = data.copy()
        self._ticker_label.setText(ticker)
        self._render_current_period()

    # ── Побудова UI ──────────────────────────────────────────────────────────

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
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(8)

        self._ticker_label = QLabel("—")
        self._ticker_label.setObjectName("tickerLabel")

        layout.addWidget(self._ticker_label)
        layout.addStretch()

        self._period_buttons: dict[str, QPushButton] = {}
        for label, _ in PERIODS:
            btn = QPushButton(label)
            btn.setObjectName("periodBtn")
            btn.setCheckable(True)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _, l=label: self._on_period_clicked(l))
            self._period_buttons[label] = btn
            layout.addWidget(btn)

        self._period_buttons["1Y"].setChecked(True)

        bar.setStyleSheet(f"""
                    QWidget#chartHeader {{
                        background: transparent;
                        border-bottom: 1px solid rgba(128,128,128,0.2);
                    }}
                    QLabel#tickerLabel {{
                        font-size: 16px;
                        font-weight: 700;
                        letter-spacing: 1px;
                        color: palette(windowText); /* Адаптивний колір тексту */
                    }}
                    QPushButton#periodBtn {{
                        background: transparent;
                        border: none;
                        border-radius: 5px;
                        font-size: 11px;
                        padding: 4px 10px;
                        color: palette(windowText); /* Адаптивний колір тексту */
                    }}
                    QPushButton#periodBtn:hover {{
                        background-color: rgba(128, 128, 128, 0.15); /* Універсальний напівпрозорий ховер */
                    }}
                    QPushButton#periodBtn:checked {{
                        background-color: {_ACCENT};
                        color: white;
                        font-weight: bold;
                    }}
                """)
        return bar

    # ── Слоти та логіка ──────────────────────────────────────────────────────

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
            cutoff = last_date - pd.Timedelta(days=days)
            data = self._full_data[pd.to_datetime(self._full_data["date"]) >= cutoff]

        if data.empty:
            data = self._full_data   # fallback: показуємо все

        self._renderer.render(self._ticker, data)