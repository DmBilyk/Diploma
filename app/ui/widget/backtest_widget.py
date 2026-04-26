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
│  (min 260 px)    │   ├─ EmptyState  (до запуску)                    │
│  ─ Algorithms    │   ├─ SpinnerView (під час запуску)               │
│  ─ Plugins       │   └─ ComparisonDashboard (після запуску)         │
│  ─ Benchmark     │       ├─ MetricTable (рядки = алгоритми)         │
│  ─ Dates         │       ├─ PortfolioValueChart (всі серії)         │
│  ─ Parameters    │       ├─ DrawdownChart                           │
│  ─ [Run btn]     │       └─ WeightsChart (для обраного алгоритму)   │
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

from PySide6.QtCore import Qt, QThread, Signal, QDate, QSize
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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

_BG       = "#0B0F19"
_SURFACE  = "#111827"
_SURFACE2 = "#1A2235"
_BORDER   = "#1F2937"
_BORDER2  = "#2D3748"
_ACCENT   = "#F59E0B"   # amber — primary action / highlight
_TEXT_PRI = "#F9FAFB"
_TEXT_SEC = "#6B7280"
_TEXT_MUT = "#374151"
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
#  TOGGLE CARD  — QFrame з нормальним layout (без paintEvent)
# ══════════════════════════════════════════════════════════════════════════════

class _ToggleCard(QFrame):
    """
    Клікабельна картка-перемикач для вибору алгоритму.
    QFrame + QHBoxLayout з QLabel — без жодного paintEvent.
    """
    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        subtitle: str,
        color: str,
        checked: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toggleCard")
        self._title   = title  # plain string — read by _PluginSelector.selected_plugins()
        self._color   = color
        self._checked = checked
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(54)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # ── Layout ────────────────────────────────────────────────────────────
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)

        # Текстовий блок (назва + підзаголовок)
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        text_col.setContentsMargins(0, 0, 0, 0)

        self._title_lbl = QLabel(title)
        self._sub_lbl   = QLabel(subtitle)
        self._title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._sub_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        text_col.addWidget(self._title_lbl)
        text_col.addWidget(self._sub_lbl)
        row.addLayout(text_col, 1)

        # Галочка праворуч
        self._check_lbl = QLabel("✓")
        self._check_lbl.setFixedSize(18, 18)
        self._check_lbl.setAlignment(Qt.AlignCenter)
        row.addWidget(self._check_lbl)

        self._refresh_style()

    # ── Стан ──────────────────────────────────────────────────────────────────

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, val: bool) -> None:
        self._checked = val
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._checked = not self._checked
            self._refresh_style()
            self.toggled.emit(self._checked)
        super().mousePressEvent(event)

    # ── Стилі ─────────────────────────────────────────────────────────────────

    def _refresh_style(self) -> None:
        on  = self._checked
        rgb = self._hex_to_rgb(self._color)
        bg      = f"rgba({rgb}, 0.10)" if on else _SURFACE
        border  = self._color          if on else _BORDER

        # Стиль рамки — через objectName щоб не лізти в дочірні віджети
        self.setStyleSheet(f"""
            QFrame#toggleCard {{
                background-color: {bg};
                border: 1px solid {border};
                border-left: 3px solid {self._color};
                border-radius: 6px;
            }}
            QFrame#toggleCard:hover {{
                background-color: rgba({rgb}, 0.16);
                border-color: {self._color};
            }}
            QLabel {{ background: transparent; border: none; }}
        """)

        title_color = _TEXT_PRI if on else _TEXT_SEC
        sub_color   = _TEXT_SEC if on else _TEXT_MUT
        chk_color   = self._color if on else "transparent"

        self._title_lbl.setStyleSheet(
            f"color: {title_color}; font-size: 11px; font-weight: 700;"
        )
        self._sub_lbl.setStyleSheet(
            f"color: {sub_color}; font-size: 9px;"
        )
        self._check_lbl.setStyleSheet(
            f"color: {chk_color}; font-size: 12px; font-weight: 700;"
        )

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"


# ══════════════════════════════════════════════════════════════════════════════
#  PLUGIN SELECTOR  — окремий блок для плагінів
# ══════════════════════════════════════════════════════════════════════════════

class _PluginSelector(QWidget):
    """
    Блок вибору плагіну: список плагінів як toggle-картки.
    Якщо плагінів немає — показує плейсхолдер.
    """

    def __init__(self, plugin_names: list[str], color: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color   = color
        self._cards: list[_ToggleCard] = []
        lyt = QVBoxLayout(self)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(4)

        if not plugin_names:
            placeholder = QLabel("No plugins found in app/plugins/")
            placeholder.setStyleSheet(f"""
                color: {_TEXT_MUT};
                font-size: 10px;
                font-style: italic;
                border: 1px dashed {_BORDER};
                border-radius: 6px;
                padding: 10px 14px;
            """)
            lyt.addWidget(placeholder)
        else:
            for name in plugin_names:
                card = _ToggleCard(
                    title=name,
                    subtitle="Custom plugin · BaseOptimizer",
                    color=color,
                    checked=False,
                )
                self._cards.append(card)
                lyt.addWidget(card)

    def selected_plugins(self) -> list[str]:
        return [c._title for c in self._cards if c.isChecked()]


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
            QGroupBox {{
                border: none;
                margin-top: 0px;
                padding-top: 0px;
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

        self._inner = QWidget()
        self._inner.setObjectName("inner")
        self.setWidget(self._inner)
        self.setWidgetResizable(True)
        self._build_inner()

    # ── Побудова ─────────────────────────────────────────────────────────────

    def _build_inner(self) -> None:
        lyt = QVBoxLayout(self._inner)
        lyt.setContentsMargins(12, 16, 12, 16)
        lyt.setSpacing(16)

        # ── 1. Алгоритми ──────────────────────────────────────────────────
        lyt.addWidget(self._section_label("ALGORITHMS"))

        self._card_hybrid = _ToggleCard(
            title="Hybrid Evo",
            subtitle="GA + SLSQP + Local Search",
            color=_SERIES_COLORS[0],
            checked=True,
        )
        self._card_markowitz = _ToggleCard(
            title="Markowitz",
            subtitle="Mean-Variance (MPT)",
            color=_SERIES_COLORS[1],
            checked=True,
        )
        lyt.addWidget(self._card_hybrid)
        lyt.addWidget(self._card_markowitz)

        # ── 2. Плагіни ────────────────────────────────────────────────────
        lyt.addWidget(self._section_label("PLUGINS"))

        try:
            plugins_dict  = self._core.get_plugins()
            plugin_names  = list(plugins_dict.keys())
        except Exception as exc:
            logger.warning("Could not load plugins: %s", exc)
            plugin_names = []

        self._plugin_selector = _PluginSelector(plugin_names, _SERIES_COLORS[3])
        lyt.addWidget(self._plugin_selector)

        # ── 3. Бенчмарк ───────────────────────────────────────────────────
        lyt.addWidget(self._section_label("BENCHMARK"))

        self._card_ew = _ToggleCard(
            title="Equal-Weight",
            subtitle="Naive 1/N diversification",
            color=_ORANGE,
            checked=True,
        )
        lyt.addWidget(self._card_ew)

        # ── 4. Діапазон дат ───────────────────────────────────────────────
        lyt.addWidget(self._section_label("DATE RANGE"))

        date_grid = QGridLayout()
        date_grid.setSpacing(6)
        date_grid.setContentsMargins(0, 0, 0, 0)
        date_grid.setColumnStretch(0, 1)
        date_grid.setColumnStretch(1, 1)

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

        # ── 5. Параметри ─────────────────────────────────────────────────
        lyt.addWidget(self._section_label("PARAMETERS"))

        params_grid = QGridLayout()
        params_grid.setSpacing(6)
        params_grid.setContentsMargins(0, 0, 0, 0)
        params_grid.setColumnStretch(0, 1)
        params_grid.setColumnStretch(1, 1)

        params_grid.addWidget(self._field_label("Population"),   0, 0)
        params_grid.addWidget(self._field_label("Generations"),  0, 1)
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

        params_grid.addWidget(self._field_label("Capital ($)"),        4, 0)
        params_grid.addWidget(self._field_label("Rebalance (weeks)"),  4, 1)
        self._capital   = self._make_dspin(10_000, 10_000_000, 100_000,
                                           decimals=0, step=10_000)
        self._rebalance = self._make_spin(1, 52, 4)
        params_grid.addWidget(self._capital,   5, 0)
        params_grid.addWidget(self._rebalance, 5, 1)

        lyt.addLayout(params_grid)

        # ── 6. Кнопка Run ─────────────────────────────────────────────────
        lyt.addSpacing(4)
        self.btn_run = QPushButton("▶  Run Comparison")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
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
        lyt.addWidget(self.btn_run)
        lyt.addStretch()

    # ── Слот ─────────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        def qs(qd: QDate) -> str:
            return qd.toString("yyyy-MM-dd")

        algorithms: List[str] = []
        if self._card_hybrid.isChecked():
            algorithms.append("hybrid_evo")
        if self._card_markowitz.isChecked():
            algorithms.append("markowitz")
        for plugin_name in self._plugin_selector.selected_plugins():
            algorithms.append(f"plugin:{plugin_name}")

        benchmarks: List[str] = []
        if self._card_ew.isChecked():
            benchmarks.append("equal_weight")

        if not algorithms and not benchmarks:
            return

        params = dict(
            algorithms      = algorithms,
            benchmarks      = benchmarks,
            train_start     = qs(self._train_start.date()),
            train_end       = qs(self._train_end.date()),
            backtest_start  = qs(self._bt_start.date()),
            backtest_end    = qs(self._bt_end.date()),
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
            font-weight: 700;
            letter-spacing: 1.5px;
            padding-bottom: 2px;
            background: transparent;
        """)
        return lbl

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 10px; background: transparent;")
        return lbl

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
                "CAGR", "Volatility", "Max Drawdown", "Sharpe", "Sortino",
                "Calmar", "Info Ratio", "VaR 95%", "CVaR 95%",
                "Win Rate", "Turnover", "# Holdings"]

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
        # Мінімальна висота замість фіксованої — масштабується разом з вікном
        self.setMinimumHeight(80)
        self.setMaximumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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

    def populate(self, rows: list[dict]) -> None:
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
                row_data.get("sortino", "—"),
                row_data.get("calmar", "—"),
                row_data.get("info_ratio", "—"),
                row_data.get("var_95", "—"),
                row_data.get("cvar_95", "—"),
                row_data.get("win_rate", "—"),
                row_data.get("turnover", "—"),
                row_data.get("n_holdings", "—"),
            ]
            for col_idx, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                if col_idx == 0:
                    item.setForeground(QColor(color))
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
                self.setItem(row_idx, col_idx, item)
        # Підганяємо висоту таблиці під кількість рядків
        row_h = self.rowHeight(0) if rows else 28
        header_h = self.horizontalHeader().height()
        needed = header_h + row_h * len(rows) + 4
        self.setMaximumHeight(max(80, min(needed, 200)))


# ══════════════════════════════════════════════════════════════════════════════
#  ПОРІВНЯЛЬНИЙ ДАШБОРД
# ══════════════════════════════════════════════════════════════════════════════

class _ComparisonDashboard(QWidget):
    """Рендерить таблицю метрик + три matplotlib-графіки для N алгоритмів."""

    _selected_algo_idx: int = 0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        self._series: list[dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Таблиця
        self._table = _MetricsTable()
        root.addWidget(self._table)

        # Верхній графік — вартість
        self._fig_main = Figure(dpi=100)
        self._canvas_main = _ThemeAwareCanvas(self._fig_main)
        self._canvas_main.setMinimumHeight(180)
        root.addWidget(self._canvas_main, 4)

        # ── Нижній рядок: Drawdown (ліво) + Weights (право) ──────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        # Лівий блок — Drawdown
        self._fig_dd = Figure(dpi=100)
        self._canvas_dd = _ThemeAwareCanvas(self._fig_dd)
        self._canvas_dd.setMinimumHeight(150)
        bottom_row.addWidget(self._canvas_dd, 1)

        # Правий блок — заголовок з selector + canvas weights
        # Структура: QWidget > QVBoxLayout > [header_row, canvas]
        right_container = QWidget()
        right_container.setStyleSheet(f"background: {_BG};")
        right_vbox = QVBoxLayout(right_container)
        right_vbox.setContentsMargins(0, 0, 0, 0)
        right_vbox.setSpacing(4)

        # Рядок заголовку: "Weights:" label + combobox
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        weights_lbl = QLabel("Weights:")
        weights_lbl.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 10px; background: transparent;")
        weights_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header_row.addWidget(weights_lbl)

        self._weights_selector = QComboBox()
        self._weights_selector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._weights_selector.setMaximumHeight(26)
        self._weights_selector.setStyleSheet(f"""
            QComboBox {{
                background: {_SURFACE};
                color: {_TEXT_PRI};
                border: 1px solid {_BORDER};
                border-radius: 4px;
                padding: 2px 6px;
                font-size: 11px;
            }}
            QComboBox:focus {{ border-color: {_ACCENT}; }}
            QComboBox::drop-down {{ border: none; width: 16px; }}
        """)
        self._weights_selector.currentIndexChanged.connect(self._on_weights_algo_changed)
        header_row.addWidget(self._weights_selector, 1)

        right_vbox.addLayout(header_row)

        # Canvas weights — займає весь простір що залишився
        self._fig_w = Figure(dpi=100)
        self._canvas_w = _ThemeAwareCanvas(self._fig_w)
        self._canvas_w.setMinimumHeight(130)
        right_vbox.addWidget(self._canvas_w, 1)

        bottom_row.addWidget(right_container, 1)
        root.addLayout(bottom_row, 3)

    # ── Публічний API ─────────────────────────────────────────────────────────

    def render(self, series_list: list[dict]) -> None:
        self._series = series_list

        def _fmt_pct(v, sign: bool = False) -> str:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                return f"{v:+.2%}" if sign else f"{v:.2%}"
            except (TypeError, ValueError):
                return "—"

        def _fmt_num(v, places: int = 2) -> str:
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return "—"
                return f"{v:.{places}f}"
            except (TypeError, ValueError):
                return "—"

        table_rows = []
        for s in series_list:
            m = s.get("metrics", {})
            table_rows.append({
                "_color":       s["color"],
                "name":         s["name"],
                "start":        f"${s['equity'].iloc[0]:,.0f}",
                "finish":       f"${s['equity'].iloc[-1]:,.0f}",
                "total_return": _fmt_pct(m.get("total_return"), sign=True),
                "cagr":         _fmt_pct(m.get("cagr")),
                "volatility":   _fmt_pct(m.get("volatility")),
                "max_drawdown": _fmt_pct(m.get("max_dd")),
                "sharpe":       _fmt_num(m.get("sharpe")),
                "sortino":      _fmt_num(m.get("sortino")),
                "calmar":       _fmt_num(m.get("calmar")),
                "info_ratio":   _fmt_num(m.get("info_ratio")),
                "var_95":       _fmt_pct(m.get("var_95")),
                "cvar_95":      _fmt_pct(m.get("cvar_95")),
                "win_rate":     _fmt_pct(m.get("win_rate")),
                "turnover":     _fmt_pct(m.get("turnover")),
                "n_holdings":   str(int(m.get("n_holdings") or 0)),
            })
        self._table.populate(table_rows)

        self._weights_selector.blockSignals(True)
        self._weights_selector.clear()
        for s in series_list:
            if s.get("weights"):
                self._weights_selector.addItem(s["name"])
        self._weights_selector.blockSignals(False)
        self._selected_algo_idx = 0

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
        ax.legend(loc="upper left", framealpha=0.3, fontsize=9,
                  labelcolor="linecolor", facecolor=bg_color, edgecolor=grid_color)
        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.12)
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
                continue
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
        try:
            fig.tight_layout(pad=0.8)
        except Exception:
            fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
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
            try:
                fig.tight_layout(pad=0.8)
            except Exception:
                pass
            self._canvas_w.draw()
            return

        color   = series["color"]
        weights = series["weights"]
        top     = dict(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:15])
        labels  = list(top.keys())
        values  = [v * 100 for v in top.values()]

        n    = len(labels)
        bars = ax.barh(labels, values, color=color,
                       alpha=0.85, edgecolor=bg_color, linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.4,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center", ha="left", fontsize=7.5, color=text_color)

        ax.invert_yaxis()
        # Заголовок прибрано — він тепер у header_row зверху (QLabel + QComboBox)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.12)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.8, color=grid_color)
        try:
            fig.tight_layout(pad=0.6)
        except Exception:
            fig.subplots_adjust(left=0.30, right=0.88, top=0.96, bottom=0.08)
        self._canvas_w.draw()

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
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 18px; font-weight: 700;"
                            f" letter-spacing: 0.3px; background: transparent;")

        subtitle = QLabel("Select algorithms on the left and click  Run Comparison")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {_TEXT_SEC}; font-size: 12px; padding-top: 6px;"
                               f" background: transparent;")

        hint = QLabel("Supports:  Hybrid Evo  ·  Markowitz  ·  Plugins  ·  Equal-Weight Benchmark")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet(f"color: {_BORDER2}; font-size: 10px; padding-top: 4px;"
                           f" background: transparent;")

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
        finished(list[dict])
        error(str)
    """
    progress_updated = Signal(int, str)
    finished         = Signal(list)
    error            = Signal(str)

    def __init__(self, core: PortfolioCore, params: dict,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._core   = core
        self._params = params
        # Phase 7: keep the full BacktestReport so the UI can export it.
        self.report = None

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

        repo    = PortfolioRepository()
        tickers = repo.get_all_tickers()
        df      = repo.get_price_history(tickers)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        train_raw = df[(df.index >= train_start) & (df.index <= train_end)]
        test_raw  = df[(df.index >= bt_start)    & (df.index <= bt_end)]

        min_train      = int(len(train_raw) * 0.5)
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

        class _InMemoryRepo:
            def __init__(self, prices_df):
                self._df = prices_df

            def get_price_history(self, tickers, start_date=None, end_date=None):
                cols = [t for t in tickers if t in self._df.columns]
                sub  = self._df[cols]
                if start_date:
                    sub = sub[sub.index >= start_date]
                if end_date:
                    sub = sub[sub.index <= end_date]
                return sub

        test_df = test_raw[valid_tickers].copy()
        engine  = BacktestEngine(
            start_date=bt_start, end_date=bt_end,
            initial_capital=capital, risk_free_rate=rfr,
            rebalance_every=rebalance,
            repo=_InMemoryRepo(test_df),
        )

        def _metrics_to_dict(m) -> dict:
            # Phase 1 fields are exposed alongside the legacy ones.  Defaults
            # via getattr keep this safe if an older BacktestMetrics ever
            # flows through (e.g. a deserialised report from a prior build).
            return dict(
                total_return=m.total_return,
                cagr=m.cagr,
                volatility=m.annualised_volatility,
                max_dd=m.max_drawdown,
                sharpe=m.sharpe_ratio,
                sortino=m.sortino_ratio,
                calmar=getattr(m, "calmar_ratio", float("nan")),
                info_ratio=getattr(m, "information_ratio", float("nan")),
                var_95=getattr(m, "var_95", float("nan")),
                cvar_95=getattr(m, "cvar_95", float("nan")),
                win_rate=getattr(m, "win_rate", float("nan")),
                turnover=getattr(m, "turnover", float("nan")),
                n_holdings=getattr(m, "avg_n_holdings", 0),
            )

        specs: list[PortfolioSpec] = []
        spec_meta: list[dict]      = []

        # ── Hybrid Evo ──────────────────────────────────────────────────────
        if "hybrid_evo" in algorithms:
            color = _SERIES_COLORS[len(specs) % len(_SERIES_COLORS)]
            step += 1
            self.progress_updated.emit(int(step / total_steps * 90),
                                       "Hybrid Evo: optimising...")
            opt = HybridEvoOptimizer(
                pop_size=pop_size, n_generations=n_gen,
                max_cardinality=max_k, risk_free_rate=rfr, seed=42,
                mu_shrinkage=True,            # James-Stein на μ — ↓ оцінкового шуму
                penalty_concentration=0.1,    # м'який anti-Herfindahl
            )
            # ВАЖЛИВО: передаємо start_date, інакше Hybrid Evo тренується на
            # всій історії БД (~30 років), а Markowitz — лише на [train_start,
            # train_end].  Це робило порівняння нечесним.
            res = opt.run(
                tickers=valid_tickers,
                start_date=train_start,
                end_date=train_end,
            )
            specs.append(PortfolioSpec(name="Hybrid Evo", weights=res.weights))
            spec_meta.append(dict(color=color, is_benchmark=False, weights=res.weights))

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
            # Apply the same cardinality constraint as Hybrid Evo so the
            # comparison is like-for-like (same K assets to choose from).
            # Without this, Markowitz silently runs unconstrained while
            # Hybrid Evo is capped at K, which is not a fair contest.
            if len(w_m) > max_k:
                top_k = dict(sorted(w_m.items(), key=lambda kv: kv[1], reverse=True)[:max_k])
                _s = sum(top_k.values())
                w_m = {t: v / _s for t, v in top_k.items()} if _s > 1e-9 else top_k
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

        # ── Equal-Weight benchmark ───────────────────────────────────────────
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

        self.progress_updated.emit(92, "Simulating backtest with rebalancing...")
        report = engine.run(specs)
        # Phase 7: expose the full report so BacktestWidget can export it.
        self.report = report

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
    """Сторінка «Comparison Backtest» — порівняння кількох алгоритмів."""

    _IDX_EMPTY   = 0
    _IDX_SPINNER = 1
    _IDX_RESULTS = 2

    def __init__(self, core: PortfolioCore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {_BG};")
        self._core   = core
        self._worker: _MultiBacktestWorker | None = None
        # Phase 7: latest BacktestReport for export
        self._last_report = None
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
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 14px; font-weight: 700;"
                            f" letter-spacing: 0.3px;")
        lyt.addWidget(title)
        lyt.addStretch()

        # ── Phase 7: export buttons (disabled until a report exists) ──
        export_btn_style = (
            f"QPushButton {{"
            f"  color: {_TEXT_PRI};"
            f"  background-color: transparent;"
            f"  border: 1px solid {_BORDER};"
            f"  border-radius: 4px;"
            f"  padding: 4px 12px;"
            f"  font-size: 11px;"
            f"  font-weight: 600;"
            f"}}"
            f"QPushButton:hover:enabled {{ border-color: {_ACCENT}; color: {_ACCENT}; }}"
            f"QPushButton:disabled {{ color: {_TEXT_SEC}; border-color: {_BORDER}; }}"
        )

        self._btn_export_json = QPushButton("Export JSON")
        self._btn_export_json.setEnabled(False)
        self._btn_export_json.setStyleSheet(export_btn_style)
        self._btn_export_json.setToolTip(
            "Save the full backtest report as a JSON file (round-trippable)."
        )
        self._btn_export_json.clicked.connect(self._on_export_json)
        lyt.addWidget(self._btn_export_json)

        self._btn_export_html = QPushButton("Export HTML")
        self._btn_export_html.setEnabled(False)
        self._btn_export_html.setStyleSheet(export_btn_style)
        self._btn_export_html.setToolTip(
            "Save a self-contained HTML report with embedded charts."
        )
        self._btn_export_html.clicked.connect(self._on_export_html)
        lyt.addWidget(self._btn_export_html)

        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.setStyleSheet(export_btn_style)
        self._btn_export_csv.setToolTip(
            "Save per-portfolio equity curves and a metrics-summary CSV."
        )
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        lyt.addWidget(self._btn_export_csv)

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
        f.setStyleSheet(f"background: {_BORDER}; border: none;")
        return f

    @staticmethod
    def _vline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setFixedWidth(1)
        f.setStyleSheet(f"background: {_BORDER}; border: none;")
        return f

    # ── Слоти ─────────────────────────────────────────────────────────────────

    def _on_run_requested(self, params: dict) -> None:
        if self._worker and self._worker.isRunning():
            return

        self._ctrl.btn_run.setEnabled(False)
        # Clear stale export state — the previous report is no longer "current".
        self._last_report = None
        self._btn_export_json.setEnabled(False)
        self._btn_export_html.setEnabled(False)
        self._btn_export_csv.setEnabled(False)
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
        # Phase 7: capture the full BacktestReport for export.
        self._last_report = getattr(self._worker, "report", None)
        has_export = self._last_report is not None and bool(series_list)
        self._btn_export_json.setEnabled(has_export)
        self._btn_export_html.setEnabled(has_export)
        self._btn_export_csv.setEnabled(has_export)
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

    # ── Phase 7: export slots ────────────────────────────────────────────────

    def _on_export_json(self) -> None:
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export backtest report (JSON)",
            "backtest_report.json", "JSON files (*.json)",
        )
        if not path:
            return
        try:
            self._last_report.save_json(path)
        except Exception as exc:
            logger.exception("JSON export failed")
            QMessageBox.critical(self, "Export failed", f"Could not write JSON:\n\n{exc}")
            return
        QMessageBox.information(self, "Export complete",
                                f"Report saved to:\n{path}")

    def _on_export_html(self) -> None:
        if self._last_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export backtest report (HTML)",
            "backtest_report.html", "HTML files (*.html)",
        )
        if not path:
            return
        try:
            self._last_report.save_html(path)
        except Exception as exc:
            logger.exception("HTML export failed")
            QMessageBox.critical(self, "Export failed", f"Could not write HTML:\n\n{exc}")
            return
        QMessageBox.information(self, "Export complete",
                                f"Report saved to:\n{path}")

    def _on_export_csv(self) -> None:
        if self._last_report is None:
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Choose folder for CSV export",
        )
        if not directory:
            return
        try:
            written = self._last_report.save_csv(directory)
        except Exception as exc:
            logger.exception("CSV export failed")
            QMessageBox.critical(self, "Export failed", f"Could not write CSV:\n\n{exc}")
            return
        QMessageBox.information(
            self, "Export complete",
            f"Wrote {len(written)} files to:\n{directory}",
        )

    def _set_status(self, text: str, color: str) -> None:
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(f"""
            color: {color};
            border: 1px solid {color};
            border-radius: 10px;
            padding: 3px 12px;
            font-size: 11px;
            font-weight: 600;
        """)
