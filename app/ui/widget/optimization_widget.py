"""
app/ui/widget/optimization_widget.py
======================================

Сторінка «Portfolio Optimization» у темному стилі (Fintech Terminal).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from pypfopt import expected_returns as _exp_ret, risk_models as _risk_models
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QDate, QThread, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.algorithms.hybrid_evo_optimizer import OptimizationResult
from app.core.core import PortfolioCore
from app.ui.widget.optimizing_spinner import OptimizingSpinner

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS (Fintech terminal palette)
# ══════════════════════════════════════════════════════════════════════════════

_BG       = "#0B0F19"
_SURFACE  = "#111827"
_SURFACE2 = "#1A2235"
_BORDER   = "#1F2937"
_BORDER2  = "#2D3748"
_ACCENT   = "#F59E0B"
_TEXT_PRI = "#F9FAFB"
_TEXT_SEC = "#6B7280"
_TEXT_MUT = "#374151"
_SUCCESS  = "#10B981"
_DANGER   = "#F43F5E"

_SERIES_COLORS = [
    "#F59E0B", "#10B981", "#60A5FA", "#A78BFA", "#F472B6", "#34D399"
]

# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB КАНВАС
# ══════════════════════════════════════════════════════════════════════════════

class _ThemeAwareCanvas(FigureCanvas):
    def __init__(self, fig: Figure, parent: QWidget | None = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure.patch.set_facecolor(_BG)
        self.setStyleSheet(f"background-color: {_BG};")


# ══════════════════════════════════════════════════════════════════════════════
#  ПАНЕЛЬ КОНТРОЛЮ
# ══════════════════════════════════════════════════════════════════════════════

class _ControlPanel(QScrollArea):
    run_requested = Signal(dict)

    def __init__(self, core: PortfolioCore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._core = core
        self.setMinimumWidth(260)
        self.setMaximumWidth(320)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {_SURFACE}; border: none; }}
            QWidget#inner {{ background: {_SURFACE}; }}
            QLabel {{
                color: {_TEXT_PRI};
                font-size: 11px;
                background: transparent;
            }}
            QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox {{
                background: {_BG};
                color: {_TEXT_PRI};
                border: 1px solid {_BORDER};
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 11px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QComboBox:focus {{
                border-color: {_ACCENT};
            }}
            QComboBox::drop-down {{ border: none; width: 18px; }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid {_TEXT_SEC};
                margin-right: 6px;
            }}
            QScrollBar:vertical {{
                border: none; width: 4px; background: {_BG};
            }}
            QScrollBar::handle:vertical {{
                background: {_BORDER2}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        inner = QWidget()
        inner.setObjectName("inner")
        self.setWidget(inner)
        self.setWidgetResizable(True)
        self._build(inner)

    def _build(self, parent: QWidget) -> None:
        lyt = QVBoxLayout(parent)
        lyt.setContentsMargins(12, 16, 12, 16)
        lyt.setSpacing(16)

        lyt.addWidget(self._section_label("ALGORITHMS"))
        self._combo_algo = QComboBox()
        self._combo_algo.addItem("Hybrid Evo (GA + SLSQP)", "evo")
        try:
            plugins = self._core.get_plugins()
            for p_name in plugins:
                self._combo_algo.addItem(f"Plugin: {p_name}", f"plugin:{p_name}")
        except Exception as e:
            logger.warning(f"Failed to load plugins: {e}")
        lyt.addWidget(self._combo_algo)

        lyt.addWidget(self._section_label("DATE RANGE"))
        lyt.addLayout(self._build_dates())

        lyt.addWidget(self._section_label("PARAMETERS"))
        lyt.addLayout(self._build_params())

        lyt.addSpacing(4)
        lyt.addWidget(self._build_run_button())
        lyt.addStretch()

    def _build_dates(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self._field_label("Train Start"), 0, 0)
        grid.addWidget(self._field_label("Train End"),   0, 1)

        self._train_start = self._date_edit("2015-01-01")
        self._train_end   = self._date_edit("2023-12-31")
        grid.addWidget(self._train_start, 1, 0)
        grid.addWidget(self._train_end,   1, 1)
        return grid

    def _build_params(self) -> QGridLayout:
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(0, 0, 0, 0)

        grid.addWidget(self._field_label("Population"),  0, 0)
        grid.addWidget(self._field_label("Generations"), 0, 1)
        self._pop_size = self._spin(50, 500, 150, step=50)
        self._n_gen    = self._spin(50, 500, 150, step=50)
        grid.addWidget(self._pop_size, 1, 0)
        grid.addWidget(self._n_gen,    1, 1)

        grid.addWidget(self._field_label("Max Assets (K)"), 2, 0)
        grid.addWidget(self._field_label("Risk-Free Rate"), 2, 1)
        self._max_k = self._spin(5, 50, 15)
        self._rfr   = self._dspin(0.0, 10.0, 2.0, decimals=1, suffix=" %")
        grid.addWidget(self._max_k, 3, 0)
        grid.addWidget(self._rfr,   3, 1)

        return grid

    def _build_run_button(self) -> QPushButton:
        self.btn_run = QPushButton("▶  Run Optimization")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                background-color: {_ACCENT};
                color: #0B0F19;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
                padding: 10px;
            }}
            QPushButton:hover   {{ background-color: #FBBF24; }}
            QPushButton:pressed {{ background-color: #D97706; }}
            QPushButton:disabled {{
                background-color: rgba(245,158,11,0.20);
                color: rgba(11,15,25,0.4);
            }}
        """)
        self.btn_run.clicked.connect(self._on_run)
        return self.btn_run

    def _on_run(self) -> None:
        def qs(qd: QDate) -> str:
            return qd.toString("yyyy-MM-dd")

        self.run_requested.emit(dict(
            method          = self._combo_algo.currentData(),
            start_date      = qs(self._train_start.date()),
            end_date        = qs(self._train_end.date()),
            pop_size        = self._pop_size.value(),
            n_generations   = self._n_gen.value(),
            max_cardinality = self._max_k.value(),
            risk_free_rate  = self._rfr.value() / 100.0,
        ))

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {_TEXT_SEC}; font-size: 9px; font-weight: 700;
            letter-spacing: 1.5px; padding-bottom: 2px; background: transparent;
        """)
        return lbl

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 10px; background: transparent;")
        return lbl

    @staticmethod
    def _date_edit(iso: str) -> QDateEdit:
        w = QDateEdit()
        w.setCalendarPopup(True)
        w.setDate(QDate.fromString(iso, "yyyy-MM-dd"))
        w.setDisplayFormat("yyyy-MM-dd")
        return w

    @staticmethod
    def _spin(lo: int, hi: int, val: int, step: int = 1) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setSingleStep(step)
        return w

    @staticmethod
    def _dspin(lo: float, hi: float, val: float,
               decimals: int = 2, suffix: str = "",
               step: float | None = None) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setDecimals(decimals)
        if suffix:
            w.setSuffix(suffix)
        if step is not None:
            w.setSingleStep(step)
        return w


# ══════════════════════════════════════════════════════════════════════════════
#  КАРТКИ МЕТРИК
# ══════════════════════════════════════════════════════════════════════════════

class _MetricCard(QFrame):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{ background: {_SURFACE}; border: 1px solid {_BORDER}; border-radius: 6px; }}
            QLabel {{ border: none; background: transparent; }}
        """)
        lyt = QVBoxLayout(self)
        lyt.setContentsMargins(16, 12, 16, 12)
        lyt.setSpacing(4)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 10px; letter-spacing: 0.5px; font-weight: 600;")

        self._val = QLabel("—")
        self._val.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 18px; font-weight: 700;")

        lyt.addWidget(self._lbl)
        lyt.addWidget(self._val)

    def set_value(self, text: str, color: str = "") -> None:
        self._val.setText(text)
        if color:
            self._val.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: 700;")


class _MetricsBadges(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lyt = QHBoxLayout(self)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(12)

        self._sharpe  = _MetricCard("Sharpe (ex-ante)")
        self._n_gen   = _MetricCard("Generations")
        self._ret     = _MetricCard("Expected Return")
        self._risk    = _MetricCard("Risk (σ)")
        self._assets  = _MetricCard("Assets Count")

        for card in (self._sharpe, self._n_gen, self._ret, self._risk, self._assets):
            lyt.addWidget(card, 1)

    def populate(self, result: OptimizationResult) -> None:
        sharpe_color = _SUCCESS if result.sharpe_ratio >= 1.0 else (
            _ACCENT if result.sharpe_ratio >= 0.5 else _DANGER
        )
        ret_color = _SUCCESS if result.expected_return > 0 else _DANGER

        self._sharpe.set_value(f"{result.sharpe_ratio:.2f}", sharpe_color)

        # Show generation count for Hybrid Evo; "Plugin" for external algorithms
        if result.n_generations > 0:
            self._n_gen.set_value(str(result.n_generations), _TEXT_PRI)
        else:
            self._n_gen.set_value("Plugin", _TEXT_SEC)

        self._ret.set_value(f"{result.expected_return:.1%}", ret_color)
        self._risk.set_value(f"{result.portfolio_risk:.1%}")
        self._assets.set_value(str(len(result.selected_assets)))


# ══════════════════════════════════════════════════════════════════════════════
#  DONUT-ГРАФІК ВАГИ
# ══════════════════════════════════════════════════════════════════════════════

class _WeightsDonut(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fig    = Figure(figsize=(5, 4), dpi=100)
        self._canvas = _ThemeAwareCanvas(self._fig, self)
        lyt = QVBoxLayout(self)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.addWidget(self._canvas)

        # hover state
        self._ax:        object | None = None
        self._wedges:    list          = []
        self._labels_d:  list          = []
        self._values_d:  list          = []
        self._annot:     object | None = None

        self._canvas.mpl_connect("motion_notify_event", self._on_hover)

    # ── hover handler ────────────────────────────────────────────────────
    def _on_hover(self, event) -> None:
        if not self._ax or not self._wedges or event.inaxes != self._ax:
            if self._annot and self._annot.get_visible():
                self._annot.set_visible(False)
                self._canvas.draw_idle()
            return

        hit = False
        total = sum(self._values_d) or 1.0
        for i, wedge in enumerate(self._wedges):
            if wedge.contains(event)[0]:
                label = self._labels_d[i]
                pct   = self._values_d[i] / total * 100
                self._annot.set_text(f"  {label}\n  {pct:.2f}%  ")
                self._annot.xy = (event.xdata, event.ydata)
                self._annot.set_visible(True)
                hit = True
                break

        if not hit and self._annot and self._annot.get_visible():
            self._annot.set_visible(False)

        self._canvas.draw_idle()

    # ── render ───────────────────────────────────────────────────────────
    def render(self, weights: dict[str, float]) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor("none")

        sorted_w = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))
        top15    = dict(list(sorted_w.items())[:15])

        others = sum(v for k, v in sorted_w.items() if k not in top15)
        if others > 1e-4:
            top15["Others"] = others

        labels = list(top15.keys())
        values = list(top15.values())
        colors = _SERIES_COLORS * (len(labels) // len(_SERIES_COLORS) + 1)

        wedges, _, autotexts = ax.pie(
            values,
            labels=None,
            colors=colors[:len(labels)],
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            pctdistance=0.78,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": _BG, "linewidth": 1.5},
        )

        for at in autotexts:
            at.set_fontsize(8)
            at.set_color(_TEXT_PRI)
            at.set_fontweight("bold")

        # raise wedges to top so contains() works reliably
        for w in wedges:
            w.set_zorder(3)

        ax.legend(
            wedges, labels,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize=9,
            labelcolor=_TEXT_PRI,
            framealpha=0.2,
            facecolor=_BG,
            edgecolor=_BORDER,
        )
        ax.set_title(
            "Portfolio Allocation",
            color=_TEXT_PRI, fontsize=12, fontweight="bold", pad=14, loc="left",
        )

        # persistent hover annotation (hidden until mouse enters a wedge)
        self._annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(16, 16),
            textcoords="offset points",
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor=_SURFACE2,
                edgecolor=_ACCENT,
                linewidth=1.2,
                alpha=0.95,
            ),
            color=_TEXT_PRI,
            fontsize=10,
            fontweight="bold",
            visible=False,
            zorder=10,
        )

        # cache for hover handler
        self._wedges   = wedges
        self._labels_d = labels
        self._values_d = values

        self._fig.tight_layout(pad=1.2)
        self._canvas.draw()


# ══════════════════════════════════════════════════════════════════════════════
#  ТАБЛИЦЯ АКТИВІВ
# ══════════════════════════════════════════════════════════════════════════════

class _AssetsTable(QTableWidget):
    _HEADERS = ["#", "Ticker", "Weight", "Exp. Return (μ)"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self._HEADERS), parent)
        self.setHorizontalHeaderLabels(self._HEADERS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {_SURFACE};
                color: {_TEXT_PRI};
                border: 1px solid {_BORDER};
                border-radius: 6px;
                font-size: 11px;
                gridline-color: {_BORDER};
                selection-background-color: rgba(245,158,11,0.10);
                selection-color: {_TEXT_PRI};
            }}
            QTableWidget::item {{ padding: 5px 10px; border: none; }}
            QTableWidget::item:alternate {{ background-color: rgba(255,255,255,0.02); }}
            QHeaderView::section {{
                background-color: {_BG};
                color: {_TEXT_SEC};
                border: none;
                border-bottom: 1px solid {_BORDER};
                padding: 5px 10px;
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }}
        """)

    def populate(
        self,
        weights: dict[str, float],
        mu_per_asset: dict[str, float] | None = None,
    ) -> None:
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        self.setRowCount(len(sorted_items))

        for row, (ticker, weight) in enumerate(sorted_items):
            rank_item = QTableWidgetItem(str(row + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)

            ticker_item = QTableWidgetItem(ticker)
            font = QFont()
            font.setBold(True)
            ticker_item.setFont(font)
            color_idx = row % len(_SERIES_COLORS)
            ticker_item.setForeground(QColor(_SERIES_COLORS[color_idx]))

            weight_item = QTableWidgetItem(f"{weight:.2%}")
            weight_item.setTextAlignment(Qt.AlignCenter)

            # Expected annual return per asset (if available from mu calculation)
            mu_val = (mu_per_asset or {}).get(ticker, None)
            if mu_val is not None:
                mu_color = _SUCCESS if mu_val >= 0 else _DANGER
                mu_text  = f"{'▲' if mu_val >= 0 else '▼'} {abs(mu_val):.1%} p.a."
            else:
                mu_color = _TEXT_MUT
                mu_text  = "—"
            mu_item = QTableWidgetItem(mu_text)
            mu_item.setTextAlignment(Qt.AlignCenter)
            mu_item.setForeground(QColor(mu_color))

            for col, item in enumerate((rank_item, ticker_item, weight_item, mu_item)):
                self.setItem(row, col, item)


# ══════════════════════════════════════════════════════════════════════════════
#  ДАШБОРД РЕЗУЛЬТАТІВ
# ══════════════════════════════════════════════════════════════════════════════

class _ResultDashboard(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background: transparent; border: none;")

        inner = QWidget()
        scroll.setWidget(inner)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        lyt = QVBoxLayout(inner)
        lyt.setContentsMargins(20, 16, 20, 20)
        lyt.setSpacing(16)

        self._badges = _MetricsBadges()
        lyt.addWidget(self._badges)

        charts_row = QHBoxLayout()
        charts_row.setSpacing(16)

        self._donut = _WeightsDonut()
        self._donut.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        charts_row.addWidget(self._donut, 3)

        self._table = _AssetsTable()
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        charts_row.addWidget(self._table, 2)

        lyt.addLayout(charts_row)

    def render(
        self,
        result: OptimizationResult,
        mu_per_asset: dict[str, float] | None = None,
    ) -> None:
        self._badges.populate(result)
        self._donut.render(result.weights)
        self._table.populate(result.weights, mu_per_asset=mu_per_asset)


# ══════════════════════════════════════════════════════════════════════════════
#  EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════

class _EmptyState(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        lyt = QVBoxLayout(self)
        lyt.setAlignment(Qt.AlignCenter)
        lyt.setSpacing(8)

        title = QLabel("No Optimization Run Yet")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 18px; font-weight: 700; background: transparent;")

        subtitle = QLabel("Configure parameters on the left and click Run Optimization")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 12px; padding-top: 6px; background: transparent;")

        for w in (title, subtitle):
            lyt.addWidget(w)


# ══════════════════════════════════════════════════════════════════════════════
#  ФОНОВИЙ WORKER
# ══════════════════════════════════════════════════════════════════════════════

class _OptimizationWorker(QThread):
    progress_updated = Signal(int, str)
    finished         = Signal(object)
    error            = Signal(str)

    def __init__(self, core: PortfolioCore, params: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._core         = core
        self._params       = params
        self.mu_per_asset: dict[str, float] = {}   # filled during run()

    def run(self) -> None:
        try:
            method = self._params.pop("method", "evo")

            if method == "evo":
                self.progress_updated.emit(10, "Loading market data...")
                self.progress_updated.emit(20, "Initializing population...")
                result: OptimizationResult = self._core.run_optimization(**self._params)

                # Calculate per-asset μ for the table column
                self.progress_updated.emit(92, "Computing per-asset returns...")
                try:
                    prices = self._core.get_price_history(
                        tickers=list(result.weights.keys()),
                        start_date=self._params.get("start_date"),
                        end_date=self._params.get("end_date"),
                    )
                    if not prices.empty and len(prices) > 10:
                        mu = _exp_ret.mean_historical_return(prices, frequency=52)
                        self.mu_per_asset = {
                            t: float(mu[t]) for t in mu.index if t in result.weights
                        }
                except Exception as _e:
                    logger.warning("mu_per_asset (evo) failed: %s", _e)

            elif method.startswith("plugin:"):
                plugin_name = method.split(":")[1]
                self.progress_updated.emit(10, f"Running plugin {plugin_name}...")

                start_date      = self._params.get("start_date")
                end_date        = self._params.get("end_date")
                risk_free_rate  = self._params.get("risk_free_rate", 0.02)

                weights = self._core.run_plugin_optimization(
                    plugin_name=plugin_name,
                    start_date=start_date,
                    end_date=end_date,
                    config=self._params,
                )

                # ── Calculate ex-ante metrics via PyPortfolioOpt ──────────
                self.progress_updated.emit(70, "Calculating ex-ante metrics...")
                sharpe_ratio    = 0.0
                expected_return = 0.0
                portfolio_risk  = 0.0
                try:
                    prices = self._core.get_price_history(
                        tickers=list(weights.keys()),
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if not prices.empty and len(prices) > 10:
                        mu = _exp_ret.mean_historical_return(prices, frequency=52)
                        S  = _risk_models.CovarianceShrinkage(
                            prices, frequency=52
                        ).ledoit_wolf()
                        tickers_ord = [t for t in mu.index if t in weights]
                        w_vec = np.array([weights.get(t, 0.0) for t in tickers_ord])
                        mu_vec = mu[tickers_ord].values
                        S_sub  = S.loc[tickers_ord, tickers_ord].values
                        expected_return = float(w_vec @ mu_vec)
                        variance        = float(w_vec @ S_sub @ w_vec)
                        portfolio_risk  = float(np.sqrt(max(variance, 0.0)))
                        if portfolio_risk > 1e-12:
                            sharpe_ratio = (
                                (expected_return - risk_free_rate) / portfolio_risk
                            )
                        # Save per-asset μ for the table
                        self.mu_per_asset = {
                            t: float(mu[t]) for t in tickers_ord
                        }
                except Exception as _e:
                    logger.warning("Plugin metric calculation failed: %s", _e)
                # ─────────────────────────────────────────────────────────

                result = OptimizationResult(
                    weights=weights,
                    selected_assets=list(weights.keys()),
                    sharpe_ratio=sharpe_ratio,
                    expected_return=expected_return,
                    portfolio_risk=portfolio_risk,
                    n_generations=0,
                    history=[],
                )
            else:
                raise ValueError(f"Unknown method '{method}'")

            self.progress_updated.emit(100, "Done!")
            self.finished.emit(result)
        except Exception as exc:
            logger.exception("OptimizationWorker failed")
            self.error.emit(str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  SPINNER VIEW
# ══════════════════════════════════════════════════════════════════════════════

class _SpinnerView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        lyt = QVBoxLayout(self)
        lyt.setAlignment(Qt.AlignCenter)
        self._spinner = OptimizingSpinner(self)
        lyt.addWidget(self._spinner, 0, Qt.AlignCenter)
        self._spinner.show()

    def start(self, message: str = "") -> None:
        self._spinner.start(message)

    def update_progress(self, percent: int, msg: str) -> None:
        self._spinner.update_progress(percent, msg)

    def stop(self) -> None:
        self._spinner.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  ГОЛОВНИЙ WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class OptimizationWidget(QWidget):
    _IDX_EMPTY   = 0
    _IDX_SPINNER = 1
    _IDX_RESULTS = 2

    def __init__(self, core: PortfolioCore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        self._core   = core
        self._worker: Optional[_OptimizationWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())
        root.addWidget(self._hline())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._ctrl = _ControlPanel(core=self._core)
        self._ctrl.run_requested.connect(self._on_run_requested)
        body.addWidget(self._ctrl)
        body.addWidget(self._vline())

        self._stack = QStackedWidget()
        self._empty_state   = _EmptyState()
        self._spinner_view  = _SpinnerView()
        self._dashboard     = _ResultDashboard()

        self._stack.addWidget(self._empty_state)   # 0
        self._stack.addWidget(self._spinner_view)  # 1
        self._stack.addWidget(self._dashboard)     # 2

        body.addWidget(self._stack, 1)

        container = QWidget()
        container.setLayout(body)
        root.addWidget(container, 1)

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"background-color: {_SURFACE}; border-bottom: 1px solid {_BORDER};")
        lyt = QHBoxLayout(bar)
        lyt.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Portfolio Optimization")
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 14px; font-weight: 700; letter-spacing: 0.3px;")
        lyt.addWidget(title)
        lyt.addStretch()

        self._status_badge = QLabel("Idle")
        self._status_badge.setStyleSheet(f"""
            color: {_TEXT_SEC};
            border: 1px solid {_BORDER}; border-radius: 10px;
            padding: 3px 12px; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;
        """)
        lyt.addWidget(self._status_badge)
        return bar

    def _on_run_requested(self, params: dict) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._ctrl.btn_run.setEnabled(False)
        self._stack.setCurrentIndex(self._IDX_SPINNER)
        self._spinner_view.start("Running optimization...")
        self._set_status("Running", _ACCENT)

        self._worker = _OptimizationWorker(self._core, params)
        self._worker.progress_updated.connect(self._spinner_view.update_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: OptimizationResult) -> None:
        self._spinner_view.stop()
        self._ctrl.btn_run.setEnabled(True)
        mu = getattr(self._worker, "mu_per_asset", {})
        self._dashboard.render(result, mu_per_asset=mu or None)
        self._stack.setCurrentIndex(self._IDX_RESULTS)
        n = len(result.selected_assets)
        self._set_status(f"Complete  ({n} assets)", _SUCCESS)

    def _on_error(self, msg: str) -> None:
        self._spinner_view.stop()
        self._ctrl.btn_run.setEnabled(True)
        self._stack.setCurrentIndex(self._IDX_EMPTY)
        self._set_status("Error", _DANGER)
        logger.error("Optimization failed: %s", msg)
        QMessageBox.critical(self, "Optimization Error", f"Failed to run:\n\n{msg}")

    def _set_status(self, text: str, color: str) -> None:
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(f"""
            color: {color};
            border: 1px solid {color}; border-radius: 10px;
            padding: 3px 12px; font-size: 11px; font-weight: 600;
        """)

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet(f"background: {_BORDER}; border: none;")
        return f

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet(f"background: {_BORDER}; border: none;")
        return f