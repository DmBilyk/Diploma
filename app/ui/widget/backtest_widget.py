"""
app/ui/widgets/backtest_widget.py
=================================

Інтерактивний бектест-дашборд для InvestPortfolio Optimizer.
Підтримує порівняння КІЛЬКОХ алгоритмів одночасно:
  • Hybrid Evo (GA + SLSQP)
  • Markowitz (Mean-Variance)
  • Плагіни (BaseOptimizer)
  • Ринковий бенчмарк (Equal-Weight)

Структура сторінки
──────────────────
┌─────────────────────────────────────────────────────────────────────┐
│  TopBar — «Comparison Backtest» + статус-badge                      │
├──────────────────┬──────────────────────────────────────────────────┤
│  ControlPanel    │  ResultsStack                                     │
│  (340 px fixed)  │   ├─ EmptyState  (до запуску)                    │
│  ─ Algorithms    │   ├─ SpinnerView (під час запуску)               │
│  ─ Market        │   └─ ComparisonDashboard (після запуску)         │
│  ─ Dates         │       ├─ MetricTable (рядки = алгоритми)         │
│  ─ Parameters    │       ├─ PortfolioValueChart (всі серії)         │
│  ─ [Run btn]     │       ├─ DrawdownChart                           │
│                  │       └─ WeightsChart (для обраного алгоритму)   │
└──────────────────┴──────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, List

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from app.core.core import PortfolioCore
from app.ui.widget.optimizing_spinner import OptimizingSpinner
from app.ui.workers import BacktestWorker

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  (Fintech terminal palette)
# ══════════════════════════════════════════════════════════════════════════════

_BG      = "#0B0F19"
_SURFACE = "#111827"
_BORDER  = "#1F2937"
_ACCENT  = "#F59E0B"   # amber — primary action / highlight
_TEXT_PRI = "#F9FAFB"
_TEXT_SEC = "#6B7280"
_SUCCESS  = "#10B981"
_DANGER   = "#F43F5E"
_BG1      = _BG        # alias used in legacy references

# Chart series colours — distinct, readable on dark background
_SERIES_COLORS = [
    "#F59E0B",  # amber
    "#10B981",  # emerald
    "#60A5FA",  # blue
    "#A78BFA",  # violet
    "#F472B6",  # rose
    "#34D399",  # teal
]

# Keep compat aliases for equal-weight benchmark
_ORANGE = "#94A3B8"  # muted slate for benchmark line

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
#  ПАНЕЛЬ КОНТРОЛЮ  (ліва колонка)
# ══════════════════════════════════════════════════════════════════════════════

class _ControlPanel(QScrollArea):
    """
    Ліва панель з вибором алгоритмів і параметрами.
    Signal: run_requested(dict)  — словник з усіма параметрами запуску
    """
    run_requested = Signal(dict)

    def __init__(self, core: PortfolioCore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._core = core
        self.setFixedWidth(300)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {_SURFACE}; border: none; }}
            QWidget {{ background: {_SURFACE}; color: {_TEXT_PRI}; }}
            QLabel {{
                color: {_TEXT_PRI};
                font-size: 11px;
            }}
            QGroupBox {{
                border: 1px solid {_BORDER};
                border-radius: 4px;
                margin-top: 4px;
                padding-top: 4px;
            }}
            QCheckBox {{ color: {_TEXT_PRI}; font-size: 11px; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {_BORDER};
                border-radius: 3px;
                background: {_BG};
            }}
            QCheckBox::indicator:checked {{
                background: {_ACCENT};
                border-color: {_ACCENT};
            }}
            QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox {{
                background: {_BG};
                color: {_TEXT_PRI};
                border: 1px solid {_BORDER};
                border-radius: 3px;
                padding: 3px 6px;
                font-size: 11px;
            }}
            QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus {{
                border-color: {_ACCENT};
            }}
            QComboBox::drop-down {{ border: none; }}
            QScrollBar:vertical {{
                border: none; width: 4px; background: {_BG};
            }}
            QScrollBar::handle:vertical {{
                background: {_BORDER}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

        self._inner = QWidget()

        self.setWidget(self._inner)
        self.setWidgetResizable(True)
        self._build_inner()

    # ── Побудова ─────────────────────────────────────────────────────────────

    def _build_inner(self) -> None:
        lyt = QVBoxLayout(self._inner)
        lyt.setContentsMargins(14, 16, 14, 16)
        lyt.setSpacing(14)

        # ── 1. Algorithms ──────────────────────────────────────────────────
        lyt.addWidget(self._section_label("ALGORITHMS"))

        algo_box = QGroupBox()

        algo_lyt = QVBoxLayout(algo_box)
        algo_lyt.setSpacing(6)
        algo_lyt.setContentsMargins(10, 8, 10, 8)

        self._cb_hybrid    = self._make_checkbox("Hybrid Evo  (GA + SLSQP)",      _SERIES_COLORS[0], checked=True)
        self._cb_markowitz = self._make_checkbox("Markowitz  (Mean-Variance)",     _SERIES_COLORS[1], checked=True)

        # Плагіни — динамічний список
        # Отримуємо доступні плагіни через ядро
        algo_lyt.addWidget(self._cb_hybrid)
        algo_lyt.addWidget(self._cb_markowitz)

        # Отримуємо доступні плагіни через ядро
        try:
            plugins_dict = self._core.get_plugins()
            plugin_names = list(plugins_dict.keys())
        except Exception as e:
            logger.warning(f"Не вдалося завантажити плагіни: {e}")
            plugin_names = []

        if plugin_names:
            # Розділювач
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: palette(mid); margin-top: 4px; margin-bottom: 4px;")
            algo_lyt.addWidget(sep)

            # Горизонтальний контейнер для плагіна
            plugin_lyt = QHBoxLayout()
            plugin_lyt.setContentsMargins(0, 0, 0, 0)
            plugin_lyt.setSpacing(8)

            # Чекбокс активації плагіна
            self._cb_plugin = self._make_checkbox("Plugin:", _SERIES_COLORS[3], checked=False)
            plugin_lyt.addWidget(self._cb_plugin)

            # Випадаючий список (розгорткове меню)
            self._combo_plugin = QComboBox()
            self._combo_plugin.addItems(plugin_names)

            self._combo_plugin.setEnabled(False)  # Вимкнений за замовчуванням

            # Активуємо список лише коли стоїть галочка
            self._cb_plugin.toggled.connect(self._combo_plugin.setEnabled)

            plugin_lyt.addWidget(self._combo_plugin, 1)
            algo_lyt.addLayout(plugin_lyt)
        else:
            self._cb_plugin = None
            self._combo_plugin = None

        lyt.addWidget(algo_box)

        # ── 2. Benchmark ───────────────────────────────────────────────────
        lyt.addWidget(self._section_label("BENCHMARK"))

        bench_box = QGroupBox()

        bench_lyt = QVBoxLayout(bench_box)
        bench_lyt.setSpacing(6)
        bench_lyt.setContentsMargins(10, 8, 10, 8)

        self._cb_equal_weight = self._make_checkbox("Equal-Weight", _ORANGE, checked=True)

        bench_lyt.addWidget(self._cb_equal_weight)
        lyt.addWidget(bench_box)

        # ── 3. Date range ──────────────────────────────────────────────────
        lyt.addWidget(self._section_label("DATE RANGE"))

        date_grid = QGridLayout()
        date_grid.setSpacing(6)
        date_grid.setContentsMargins(0, 0, 0, 0)

        date_grid.addWidget(self._field_label("Train Start"), 0, 0)
        date_grid.addWidget(self._field_label("Train End"),   0, 1)

        self._train_start = self._make_date("2015-01-01")
        self._train_end   = self._make_date("2019-12-31")
        date_grid.addWidget(self._train_start, 1, 0)
        date_grid.addWidget(self._train_end,   1, 1)

        date_grid.addWidget(self._field_label("Backtest Start"), 2, 0)
        date_grid.addWidget(self._field_label("Backtest End"),   2, 1)

        self._bt_start = self._make_date("2020-01-01")
        self._bt_end   = self._make_date("2025-01-01")
        date_grid.addWidget(self._bt_start, 3, 0)
        date_grid.addWidget(self._bt_end,   3, 1)

        lyt.addLayout(date_grid)

        # ── 4. Algorithm parameters ────────────────────────────────────────
        lyt.addWidget(self._section_label("PARAMETERS"))

        params_grid = QGridLayout()
        params_grid.setSpacing(6)
        params_grid.setContentsMargins(0, 0, 0, 0)

        params_grid.addWidget(self._field_label("Population"),    0, 0)
        params_grid.addWidget(self._field_label("Generations"),   0, 1)
        self._pop_size = self._make_spin(50, 500, 150, step=50)
        self._n_gen    = self._make_spin(50, 500, 150, step=50)
        params_grid.addWidget(self._pop_size, 1, 0)
        params_grid.addWidget(self._n_gen,    1, 1)

        params_grid.addWidget(self._field_label("Max Assets (K)"), 2, 0)
        params_grid.addWidget(self._field_label("Risk-Free Rate"), 2, 1)
        self._max_k = self._make_spin(5, 50, 15)
        self._rfr   = self._make_dspin(0.0, 10.0, 2.0, decimals=1, suffix=" %")
        params_grid.addWidget(self._max_k, 3, 0)
        params_grid.addWidget(self._rfr,   3, 1)

        params_grid.addWidget(self._field_label("Capital ($)"),       4, 0)
        params_grid.addWidget(self._field_label("Rebalance (weeks)"), 4, 1)
        self._capital   = self._make_dspin(10_000, 10_000_000, 100_000,
                                           decimals=0, step=10_000)
        self._rebalance = self._make_spin(1, 52, 4)
        params_grid.addWidget(self._capital,   5, 0)
        params_grid.addWidget(self._rebalance, 5, 1)

        lyt.addLayout(params_grid)

        # ── 5. Run button ──────────────────────────────────────────────────
        lyt.addSpacing(8)
        self.btn_run = QPushButton("Run Comparison")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet(f"""
            QPushButton {{
                background-color: {_ACCENT};
                color: #0B0F19;
                border: none;
                border-radius: 4px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.5px;
                padding: 8px;
            }}
            QPushButton:hover   {{ background-color: #FBBF24; }}
            QPushButton:pressed {{ background-color: #D97706; }}
            QPushButton:disabled {{
                background-color: rgba(245,158,11,0.25);
                color: rgba(11,15,25,0.5);
            }}
        """)
        self.btn_run.clicked.connect(self._on_run)
        lyt.addWidget(self.btn_run)

        lyt.addStretch()

    # ── Слот ─────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        def qs(qd: QDate) -> str:
            return qd.toString("yyyy-MM-dd")

        algorithms: List[str] = []
        if self._cb_hybrid.isChecked():
            algorithms.append("hybrid_evo")
        if self._cb_markowitz.isChecked():
            algorithms.append("markowitz")
        if getattr(self, "_cb_plugin", None) and self._cb_plugin.isChecked():
            selected_plugin = self._combo_plugin.currentText()
            algorithms.append(f"plugin:{selected_plugin}")

        benchmarks: List[str] = []
        if self._cb_equal_weight.isChecked():
            benchmarks.append("equal_weight")

        if not algorithms and not benchmarks:
            return  # Нічого не вибрано

        params = dict(
            algorithms      = algorithms,
            benchmarks      = benchmarks,
            train_start     = qs(self._train_start.date()),
            train_end       = qs(self._train_end.date()),
            backtest_start  = qs(self._bt_start.date()),
            backtest_end    = qs(self._bt_end.date()),
            # Для сумісності з BacktestWorker (він очікує ці ключі)
            method          = algorithms[0].replace("hybrid_evo", "evo") if algorithms else "evo",
            start_date      = qs(self._train_start.date()),
            end_date        = qs(self._bt_end.date()),
            pop_size        = self._pop_size.value(),
            n_generations   = self._n_gen.value(),
            max_cardinality = self._max_k.value(),
            risk_free_rate  = self._rfr.value() / 100.0,
            initial_capital = self._capital.value(),
            rebalance_every = self._rebalance.value(),
        )
        self.run_requested.emit(params)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"""
            color: {_TEXT_SEC};
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 1.2px;
            padding-bottom: 2px;
        """)
        return lbl

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 10px;")
        return lbl

    @staticmethod
    def _make_checkbox(text: str, color: str, checked: bool = True) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setStyleSheet(f"""
            QCheckBox {{ color: {color}; font-size: 11px; font-weight: 600; spacing: 6px; }}
            QCheckBox::indicator {{
                width: 14px; height: 14px;
                border: 1px solid {_BORDER};
                border-radius: 3px;
                background: {_BG};
            }}
            QCheckBox::indicator:checked {{
                background: {color};
                border-color: {color};
            }}
        """)
        return cb

    def _make_date(self, iso: str) -> QDateEdit:
        w = QDateEdit()
        w.setCalendarPopup(True)
        w.setDate(QDate.fromString(iso, "yyyy-MM-dd"))
        w.setDisplayFormat("yyyy-MM-dd")

        return w

    @staticmethod
    def _make_spin(lo: int, hi: int, val: int, step: int = 1) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setSingleStep(step)

        return w

    @staticmethod
    def _make_dspin(lo: float, hi: float, val: float,
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
#  ТАБЛИЦЯ МЕТРИК  (рядки = алгоритми)
# ══════════════════════════════════════════════════════════════════════════════

class _MetricsTable(QTableWidget):
    """Компактна таблиця порівняння метрик усіх алгоритмів."""

    _COLUMNS = ["Algorithm", "Start ($)", "End ($)", "Return",
                "CAGR", "Volatility", "Max Drawdown", "Sharpe"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, len(self._COLUMNS), parent)
        self.setHorizontalHeaderLabels(self._COLUMNS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setFixedHeight(150)
        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {_SURFACE};
                color: {_TEXT_PRI};
                border: 1px solid {_BORDER};
                border-radius: 4px;
                font-size: 11px;
                gridline-color: {_BORDER};
                selection-background-color: rgba(245,158,11,0.12);
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
                font-weight: 600;
                letter-spacing: 0.8px;
                text-transform: uppercase;
            }}
        """)



    def populate(self, rows: list[dict]) -> None:
        """rows: список словників з ключами _COLUMNS (англ. назви → вже форматовані значення)."""
        self.setRowCount(len(rows))
        for row_idx, row_data in enumerate(rows):
            color = row_data.get("_color", "#888888")
            values = [
                row_data.get("name", "—"),
                row_data.get("start", "—"),
                row_data.get("finish", "—"),
                row_data.get("total_return", "—"),
                row_data.get("cagr", "—"),
                row_data.get("volatility", "—"),
                row_data.get("max_drawdown", "—"),
                row_data.get("sharpe", "—"),
            ]
            for col_idx, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col_idx == 0:
                    # Назва алгоритму — кольорова
                    item.setForeground(QColor(color))
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
                self.setItem(row_idx, col_idx, item)


# ══════════════════════════════════════════════════════════════════════════════
#  ПОРІВНЯЛЬНИЙ ДАШБОРД
# ══════════════════════════════════════════════════════════════════════════════

class _ComparisonDashboard(QWidget):
    """Рендерить таблицю метрик + три matplotlib-графіки для N алгоритмів."""

    # Алгоритм, ваги якого показуються в WeightsChart
    _selected_algo_idx: int = 0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        self._series: list[dict] = []  # [{name, color, equity, weights}, ...]
        self._build_ui()

    # ── Побудова ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 14, 20, 14)
        root.setSpacing(12)

        # Таблиця
        self._table = _MetricsTable()
        root.addWidget(self._table)

        # Верхній графік — вартість
        self._fig_main = Figure(figsize=(9, 3.4), dpi=100)
        self._canvas_main = _ThemeAwareCanvas(self._fig_main)
        root.addWidget(self._canvas_main, 3)

        # Нижній рядок — просадка + ваги (з перемикачем алгоритму)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        self._fig_dd = Figure(figsize=(4.5, 2.6), dpi=100)
        self._canvas_dd = _ThemeAwareCanvas(self._fig_dd)
        bottom_row.addWidget(self._canvas_dd, 1)

        # Права панель: перемикач алгоритму + графік ваг
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        self._weights_selector = QComboBox()

        self._weights_selector.currentIndexChanged.connect(self._on_weights_algo_changed)
        right_col.addWidget(self._weights_selector)

        self._fig_w = Figure(figsize=(4.5, 2.4), dpi=100)
        self._canvas_w = _ThemeAwareCanvas(self._fig_w)
        right_col.addWidget(self._canvas_w)

        bottom_row.addLayout(right_col, 1)
        root.addLayout(bottom_row, 2)

    # ── Публічний API ─────────────────────────────────────────────────────────

    def render(self, series_list: list[dict]) -> None:
        """
        series_list: [{
            name: str,
            color: str,
            equity: pd.Series,
            weights: dict[str, float],  # {} для бенчмарків без ваг
            metrics: dict               # total_return, cagr, volatility, max_dd, sharpe
        }]
        """
        self._series = series_list

        # Таблиця
        table_rows = []
        for s in series_list:
            m = s.get("metrics", {})
            table_rows.append({
                "_color":      s["color"],
                "name":        s["name"],
                "start":       f"${s['equity'].iloc[0]:,.0f}",
                "finish":      f"${s['equity'].iloc[-1]:,.0f}",
                "total_return": f"{m.get('total_return', 0):+.2%}",
                "cagr":        f"{m.get('cagr', 0):.2%}",
                "volatility":  f"{m.get('volatility', 0):.2%}",
                "max_drawdown": f"{m.get('max_dd', 0):.2%}",
                "sharpe":      f"{m.get('sharpe', 0):.2f}",
            })
        self._table.populate(table_rows)

        # Перемикач ваг — лише ті алгоритми, що мають weights
        self._weights_selector.blockSignals(True)
        self._weights_selector.clear()
        for s in series_list:
            if s.get("weights"):
                self._weights_selector.addItem(s["name"])
        self._weights_selector.blockSignals(False)
        self._selected_algo_idx = 0

        # Графіки
        self._draw_equity()
        self._draw_drawdown()
        self._draw_weights(self._get_selected_weights_series())

    # ── Слоти ─────────────────────────────────────────────────────────────────

    def _on_weights_algo_changed(self, idx: int) -> None:
        self._selected_algo_idx = idx
        self._draw_weights(self._get_selected_weights_series())

    def _get_selected_weights_series(self) -> dict | None:
        name = self._weights_selector.currentText()
        for s in self._series:
            if s["name"] == name and s.get("weights"):
                return s
        return None

    # ── Графік 1: Вартість ────────────────────────────────────────────────────

    def _get_theme_colors(self):
        # Fixed fintech dark theme — always the same regardless of OS palette
        return _TEXT_PRI, _BORDER, _BG

    def _draw_equity(self) -> None:
        text_color, grid_color, bg_color = self._get_theme_colors()
        fig = self._fig_main
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, text_color, grid_color, bg_color)

        for s in self._series:
            eq = s["equity"]
            lw = 2.4 if not s.get("is_benchmark") else 1.6
            ls = "--" if s.get("is_benchmark") else "-"
            ax.plot(eq.index, eq / 1_000, color=s["color"],
                    lw=lw, ls=ls, label=s["name"], zorder=3)
            ax.annotate(
                f"  ${eq.iloc[-1] / 1_000:,.1f}K",
                xy=(eq.index[-1], eq.iloc[-1] / 1_000),
                color=s["color"], fontsize=9, fontweight="bold", va="center",
            )

        ax.set_title("Portfolio Value  ($K)", color=text_color, fontsize=10,
                     fontweight="600", pad=10, loc="left")
        ax.yaxis.set_major_formatter(lambda x, _: f"${x:,.0f}K")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=9))
        ax.tick_params(axis="x", rotation=0)
        ax.margins(x=0.01, y=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.8, color=grid_color)
        ax.legend(
            loc="upper left", framealpha=0.3,
            fontsize=9, labelcolor="linecolor",
            facecolor=bg_color, edgecolor=grid_color,
        )

        fig.tight_layout(pad=1.2)
        self._canvas_main.draw()

    # ── Графік 2: Просадка ────────────────────────────────────────────────────

    def _draw_drawdown(self) -> None:
        text_color, grid_color, bg_color = self._get_theme_colors()
        fig = self._fig_dd
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, text_color, grid_color, bg_color)

        for s in self._series:
            if s.get("is_benchmark"):
                continue  # Бенчмарк не показуємо в просадці
            eq = s["equity"]
            dd = (eq / eq.cummax() - 1) * 100
            ax.plot(dd.index, dd, color=s["color"], lw=1.5, label=s["name"])
            ax.fill_between(dd.index, dd, 0, color=s["color"], alpha=0.12)

        ax.axhline(0, color=grid_color, lw=0.7, linestyle="--")
        ax.set_title("Drawdown (%)", color=text_color, fontsize=10,
                     fontweight="600", pad=8, loc="left")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        ax.margins(x=0.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.8, color=grid_color)
        ax.legend(loc="lower left", framealpha=0.3, fontsize=8,
                  facecolor=bg_color, edgecolor=grid_color)

        fig.tight_layout(pad=1.0)
        self._canvas_dd.draw()

    # ── Графік 3: Ваги ────────────────────────────────────────────────────────

    def _draw_weights(self, series: dict | None) -> None:
        text_color, grid_color, bg_color = self._get_theme_colors()
        fig = self._fig_w
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, text_color, grid_color, bg_color)

        if not series or not series.get("weights"):
            ax.text(0.5, 0.5, "No weight data available",
                    ha="center", va="center", color=grid_color, fontsize=10,
                    transform=ax.transAxes)
            fig.tight_layout(pad=1.0)
            self._canvas_w.draw()
            return

        color = series["color"]
        weights = series["weights"]
        top = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:15])
        labels = list(top.keys())
        values = [v * 100 for v in top.values()]

        n = len(labels)
        bars = ax.barh(labels, values, color=color,
                       alpha=0.85, edgecolor=bg_color, linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.4,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center", ha="left", fontsize=7.5, color=text_color,
            )

        ax.invert_yaxis()
        ax.set_title(
            f"Weights: {series['name']}  (Top {min(n, 15)})",
            color=text_color, fontsize=10, fontweight="600", pad=8, loc="left"
        )
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.1)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.8, color=grid_color)

        fig.tight_layout(pad=1.0)
        self._canvas_w.draw()

    # ── Хелпер ───────────────────────────────────────────────────────────────

    @staticmethod
    def _style_ax(ax, text_color, grid_color, bg_color) -> None:
        ax.tick_params(colors=text_color, which="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.set_facecolor("none")


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

        title = QLabel("No Analysis Run Yet")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 18px; font-weight: 700; letter-spacing: 0.3px;")

        subtitle = QLabel("Select algorithms on the left and click  Run Comparison")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 12px; padding-top: 6px;")

        hint = QLabel("Supports:  Hybrid Evo  ·  Markowitz  ·  Plugins  ·  Equal-Weight Benchmark")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {_BORDER}; font-size: 10px; padding-top: 4px;")

        for w in (title, subtitle, hint):
            lyt.addWidget(w)


# ══════════════════════════════════════════════════════════════════════════════
#  SPINNER VIEW
# ══════════════════════════════════════════════════════════════════════════════

class _SpinnerView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        lyt = QVBoxLayout(self)
        lyt.setAlignment(Qt.AlignCenter)
        self._spinner = OptimizingSpinner(self)
        lyt.addWidget(self._spinner, 0, Qt.AlignCenter)
        self._spinner.show()

    def start(self, message: str = "") -> None:
        self._spinner.start(message)
        self._spinner.show()

    def update_progress(self, percent: int, msg: str) -> None:
        self._spinner.update_progress(percent, msg)

    def stop(self) -> None:
        self._spinner.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  ФОНОВИЙ МУЛЬТИ-WORKER
# ══════════════════════════════════════════════════════════════════════════════

class _MultiBacktestWorker(QThread):
    """
    Запускає N алгоритмів послідовно, емітує серіалізований список серій.
    Signals:
        progress_updated(int, str)
        finished(list[dict])  — список series_dict для _ComparisonDashboard.render()
        error(str)
    """
    progress_updated = Signal(int, str)
    finished         = Signal(list)
    error            = Signal(str)

    def __init__(
        self,
        core: PortfolioCore,
        params: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._core   = core
        self._params = params

    def run(self) -> None:
        try:
            series_list = self._execute()
            self.finished.emit(series_list)
        except Exception as exc:
            logger.exception("MultiBacktestWorker failed")
            self.error.emit(str(exc))

    def _execute(self) -> list[dict]:
        from pypfopt import risk_models, expected_returns
        from pypfopt.efficient_frontier import EfficientFrontier
        from app.algorithms.hybrid_evo_optimizer import HybridEvoOptimizer
        from app.data.repository import PortfolioRepository
        from app.backtesting.backtest_engine import BacktestEngine, PortfolioSpec

        p = self._params
        algorithms  = p["algorithms"]
        benchmarks  = p["benchmarks"]
        train_start = p["train_start"]
        train_end   = p["train_end"]
        bt_start    = p["backtest_start"]
        bt_end      = p["backtest_end"]
        rfr         = p["risk_free_rate"]
        capital     = p["initial_capital"]
        pop_size    = p["pop_size"]
        n_gen       = p["n_generations"]
        max_k       = p["max_cardinality"]
        rebalance   = p["rebalance_every"]

        total_steps = len(algorithms) + len(benchmarks)
        step = 0

        # ── Завантажуємо всі дані одним запитом (train + test разом) ───────────
        repo = PortfolioRepository()
        tickers = repo.get_all_tickers()
        df = repo.get_price_history(tickers)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        train_raw = df[(df.index >= train_start) & (df.index <= train_end)]
        test_raw  = df[(df.index >= bt_start)    & (df.index <= bt_end)]

        # Тікер валідний лише якщо має дані в ОБОХ вікнах:
        #   • train: ≥50% непустих рядків (для оптимізації)
        #   • test:  має хоча б один непустий рядок (потрібна ціна відкриття)
        min_train = int(len(train_raw) * 0.5)
        valid_in_train = train_raw.notna().sum()
        valid_in_train = valid_in_train[valid_in_train >= min_train].index

        test_first_valid = test_raw[valid_in_train].apply(
            lambda col: col.first_valid_index() is not None
        )
        valid_tickers = test_first_valid[test_first_valid].index.tolist()

        if not valid_tickers:
            raise RuntimeError(
                "No tickers have data in both the training and backtest windows. "
                "Please check your date ranges."
            )

        train = train_raw[valid_tickers].dropna()

        # ── BacktestEngine отримує вже відфільтровані дані через mock-repo ─────
        # Передаємо готовий DataFrame напряму, щоб engine не робив зайвий
        # запит до БД і не натикався на тікери без test-даних.
        class _InMemoryRepo:
            """Мінімальний адаптер: повертає вже завантажений DataFrame."""
            def __init__(self, prices_df):
                self._df = prices_df

            def get_price_history(self, tickers, start_date=None, end_date=None):
                cols = [t for t in tickers if t in self._df.columns]
                sub = self._df[cols]
                if start_date:
                    sub = sub[sub.index >= start_date]
                if end_date:
                    sub = sub[sub.index <= end_date]
                return sub

        test_df = test_raw[valid_tickers].copy()
        engine = BacktestEngine(
            start_date=bt_start,
            end_date=bt_end,
            initial_capital=capital,
            risk_free_rate=rfr,
            rebalance_every=rebalance,
            repo=_InMemoryRepo(test_df),   # ← дані вже в пам'яті, без зайвого DB-запиту
        )

        def _metrics_to_dict(m) -> dict:
            """Конвертує BacktestMetrics → dict у форматі, який очікує _ComparisonDashboard."""
            return dict(
                total_return=m.total_return,
                cagr=m.cagr,
                volatility=m.annualised_volatility,
                max_dd=m.max_drawdown,
                sharpe=m.sharpe_ratio,
            )

        # Збираємо PortfolioSpec-и для всіх алгоритмів, щоб передати в engine.run()
        # одним викликом і уникнути повторного завантаження цін.
        specs: list[PortfolioSpec] = []
        spec_meta: list[dict] = []   # зберігає color, is_benchmark, weights для відображення

        # ── Hybrid Evo ──────────────────────────────────────────────────────
        if "hybrid_evo" in algorithms:
            color = _SERIES_COLORS[len(specs) % len(_SERIES_COLORS)]
            step += 1
            self.progress_updated.emit(int(step / total_steps * 90),
                                       "Hybrid Evo: optimising...")
            opt = HybridEvoOptimizer(
                pop_size=pop_size, n_generations=n_gen,
                max_cardinality=max_k, risk_free_rate=rfr, seed=42,
            )
            res = opt.run(tickers=valid_tickers, end_date=train_end)
            specs.append(PortfolioSpec(name="Hybrid Evo", weights=res.weights))
            spec_meta.append(dict(color=color, is_benchmark=False,
                                  weights=res.weights))

        # ── Markowitz ───────────────────────────────────────────────────────
        if "markowitz" in algorithms:
            color = _SERIES_COLORS[len(specs) % len(_SERIES_COLORS)]
            step += 1
            self.progress_updated.emit(int(step / total_steps * 90),
                                       "Markowitz: optimising...")
            mu = expected_returns.mean_historical_return(train, frequency=52)
            S  = risk_models.CovarianceShrinkage(train, frequency=52).ledoit_wolf()
            ef = EfficientFrontier(mu, S, solver="SCS")
            ef.max_sharpe(risk_free_rate=rfr)
            w_m = {k: v for k, v in ef.clean_weights().items() if v > 0.001}
            specs.append(PortfolioSpec(name="Markowitz", weights=w_m))
            spec_meta.append(dict(color=color, is_benchmark=False, weights=w_m))

        # ── Плагіни ─────────────────────────────────────────────────────────
        for pa in [a for a in algorithms if a.startswith("plugin:")]:
            plugin_name = pa.removeprefix("plugin:")
            color = _SERIES_COLORS[len(specs) % len(_SERIES_COLORS)]
            step += 1
            self.progress_updated.emit(
                int(step / total_steps * 90),
                f"Plugin '{plugin_name}': optimising...",
            )
            try:
                w_p = self._core.run_plugin_optimization(
                    plugin_name=plugin_name,
                    tickers=valid_tickers,
                    start_date=train_start,
                    end_date=train_end,
                    config=dict(risk_free_rate=rfr, max_cardinality=max_k),
                )
                specs.append(PortfolioSpec(name=plugin_name, weights=w_p))
                spec_meta.append(dict(color=color, is_benchmark=False, weights=w_p))
            except Exception as exc:
                logger.warning("Plugin %s failed: %s", plugin_name, exc)

        # ── Бенчмарк Equal-Weight ────────────────────────────────────────────
        # Додаємо як звичайний PortfolioSpec — engine рахує його нарівні з іншими.
        if "equal_weight" in benchmarks:
            step += 1
            self.progress_updated.emit(int(step / total_steps * 90),
                                       "Equal-Weight benchmark...")
            w_eq = {t: 1.0 / len(valid_tickers) for t in valid_tickers}
            specs.append(PortfolioSpec(name="Equal-Weight", weights=w_eq))
            spec_meta.append(dict(color=_ORANGE, is_benchmark=True, weights={}))

        if not specs:
            self.progress_updated.emit(100, "Done")
            return []

        # ── Запуск BacktestEngine одним викликом ─────────────────────────────
        self.progress_updated.emit(92, "Simulating backtest with rebalancing...")
        report = engine.run(specs)

        # ── Збірка series_list для _ComparisonDashboard ──────────────────────
        series_list: list[dict] = []
        for result, meta in zip(report.results, spec_meta):
            series_list.append(dict(
                name=result.spec.name,
                color=meta["color"],
                equity=result.portfolio_values,
                weights=meta["weights"],
                metrics=_metrics_to_dict(result.metrics),
                is_benchmark=meta["is_benchmark"],
            ))

        self.progress_updated.emit(100, "Done")
        return series_list


# ══════════════════════════════════════════════════════════════════════════════
#  ГОЛОВНИЙ COMPARISON BACKTEST WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class BacktestWidget(QWidget):
    """
    Сторінка «Comparison Backtest» — порівняння кількох алгоритмів.

    Parameters
    ----------
    core : PortfolioCore
    """

    _IDX_EMPTY   = 0
    _IDX_SPINNER = 1
    _IDX_RESULTS = 2

    def __init__(self, core: PortfolioCore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        self._core   = core
        self._worker: _MultiBacktestWorker | None = None
        self._build_ui()

    # ── Побудова ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_topbar())
        root.addWidget(self._hline())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._ctrl = _ControlPanel(self._core)
        self._ctrl.run_requested.connect(self._on_run_requested)
        body.addWidget(self._ctrl)
        body.addWidget(self._vline())

        self._result_stack = QStackedWidget()


        self._empty_view   = _EmptyState()
        self._spinner_view = _SpinnerView()
        self._dashboard    = _ComparisonDashboard()

        self._result_stack.addWidget(self._empty_view)    # 0
        self._result_stack.addWidget(self._spinner_view)  # 1
        self._result_stack.addWidget(self._dashboard)     # 2

        body.addWidget(self._result_stack, 1)

        container = QWidget()
        container.setLayout(body)
        root.addWidget(container, 1)

    def _build_topbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(52)
        bar.setStyleSheet(f"background-color: {_SURFACE}; border-bottom: 1px solid {_BORDER};")

        lyt = QHBoxLayout(bar)
        lyt.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Comparison Backtest")
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 14px; font-weight: 700; letter-spacing: 0.3px;")
        lyt.addWidget(title)
        lyt.addStretch()

        self._status_badge = QLabel("Idle")
        self._status_badge.setStyleSheet(f"""
            color: {_TEXT_SEC};
            border: 1px solid {_BORDER};
            border-radius: 10px;
            padding: 3px 12px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
        """)
        lyt.addWidget(self._status_badge)
        return bar

    @staticmethod
    def _hline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.HLine)
        f.setFixedHeight(1)
        f.setStyleSheet("background: palette(mid); border: none;")
        return f

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet("background: palette(mid); border: none;")
        return f

    # ── Слоти ─────────────────────────────────────────────────────────────────

    def _on_run_requested(self, params: dict) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._ctrl.btn_run.setEnabled(False)
        self._result_stack.setCurrentIndex(self._IDX_SPINNER)
        self._spinner_view.start("Running comparison analysis...")
        self._set_status("Running", _ACCENT)

        self._worker = _MultiBacktestWorker(self._core, params)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, percent: int, msg: str) -> None:
        self._spinner_view.update_progress(percent, msg)

    def _on_finished(self, series_list: list) -> None:
        self._spinner_view.stop()
        self._ctrl.btn_run.setEnabled(True)
        if series_list:
            self._dashboard.render(series_list)
            self._result_stack.setCurrentIndex(self._IDX_RESULTS)
            self._set_status(f"Complete  ({len(series_list)} series)", _SUCCESS)
        else:
            self._result_stack.setCurrentIndex(self._IDX_EMPTY)
            self._set_status("No results", _TEXT_SEC)

    def _on_error(self, msg: str) -> None:
        from PySide6.QtWidgets import QMessageBox
        self._spinner_view.stop()
        self._ctrl.btn_run.setEnabled(True)
        self._result_stack.setCurrentIndex(self._IDX_EMPTY)
        self._set_status("Error", _DANGER)
        logger.error("Comparison backtest error: %s", msg)
        QMessageBox.critical(self, "Backtest Error",
                             f"The comparison run failed:\n\n{msg}")

    def _set_status(self, text: str, color: str) -> None:
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(f"""
            color: {color};
            border: 1px solid {color}; border-radius: 10px;
            padding: 3px 12px; font-size: 11px; font-weight: 600;
        """)