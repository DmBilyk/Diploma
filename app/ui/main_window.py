from __future__ import annotations
"""
app/ui/main_window.py — Головне вікно програми InvestPortfolio Optimizer.

Структура:
- Фіксований темний sidebar зі навігаційними кнопками
- QStackedWidget:
    0 — Market Data Explorer (таблиця тікерів + StockChartWidget)
    1 — Optimization  (spinner-placeholder)
    2 — Backtesting   (BacktestWidget)
"""

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from app.data.repository import PortfolioRepository
from app.ui.widget.stock_chart_widget import StockChartWidget
from app.ui.widget.optimizing_spinner import OptimizingSpinner
from app.ui.widget.backtest_widget import BacktestWidget
from app.ui.widget.optimization_widget import OptimizationWidget
from app.core.core import PortfolioCore
from app.ui.workers import DataSyncWorker

logger = logging.getLogger(__name__)

# ── Design tokens ────────────────────────────────────────────────────────────
_BG        = "#0B0F19"
_SURFACE   = "#111827"
_BORDER    = "#1F2937"
_ACCENT    = "#F59E0B"
_TEXT_PRI  = "#F9FAFB"
_TEXT_SEC  = "#6B7280"
_POSITIVE  = "#10B981"
_NEGATIVE  = "#F43F5E"


# ═══════════════════════════════════════════════════════════════════════════════
# Головне вікно
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Головне вікно програми.

    Layout: [Sidebar 200px] | [QStackedWidget — решта ширини]
    """

    _SIDEBAR_W   = 200
    _WINDOW_SIZE = (1280, 800)

    def __init__(self) -> None:
        super().__init__()

        self.core  = PortfolioCore()
        self._repo = self.core.repo

        self.setWindowTitle("InvestPortfolio Optimizer")
        self.resize(*self._WINDOW_SIZE)
        self._apply_global_style()
        self._build_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # Global stylesheet
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_global_style(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {_BG};
                color: {_TEXT_PRI};
                font-family: "SF Pro Display", "Inter", "Segoe UI", sans-serif;
            }}
            QWidget {{
                color: {_TEXT_PRI};
                font-family: "SF Pro Display", "Inter", "Segoe UI", sans-serif;
            }}
            QTableWidget {{
                background-color: {_SURFACE};
                border: none;
                outline: none;
                font-size: 12px;
                gridline-color: {_BORDER};
                selection-background-color: rgba(245,158,11,0.15);
                selection-color: {_TEXT_PRI};
            }}
            QTableWidget::item {{
                padding: 6px 12px;
                border: none;
                color: {_TEXT_PRI};
            }}
            QTableWidget::item:alternate {{
                background-color: rgba(255,255,255,0.02);
            }}
            QTableWidget::item:selected {{
                background-color: rgba(245,158,11,0.18);
                color: {_TEXT_PRI};
            }}
            QHeaderView::section {{
                background-color: {_SURFACE};
                color: {_TEXT_SEC};
                padding: 7px 12px;
                border: none;
                border-bottom: 1px solid {_BORDER};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 0.8px;
                text-transform: uppercase;
            }}
            QScrollBar:vertical {{
                border: none;
                background: {_SURFACE};
                width: 5px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {_BORDER};
                border-radius: 2px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {{ height: 0; }}
            QMessageBox {{
                background-color: {_SURFACE};
                color: {_TEXT_PRI};
            }}
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout
    # ─────────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_sidebar())

        self._pages = QStackedWidget()
        self._pages.addWidget(self._build_market_page())
        self._pages.addWidget(OptimizationWidget(core=self.core))
        self._pages.addWidget(BacktestWidget(core=self.core))
        layout.addWidget(self._pages)

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setFixedWidth(self._SIDEBAR_W)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: {_SURFACE};
                border: none;
                border-right: 1px solid {_BORDER};
            }}
            QPushButton {{
                background: transparent;
                color: {_TEXT_SEC};
                text-align: left;
                padding: 11px 18px;
                border: none;
                border-left: 2px solid transparent;
                font-size: 12px;
                font-weight: 500;
                letter-spacing: 0.3px;
                border-radius: 0;
            }}
            QPushButton:hover {{
                background-color: rgba(255,255,255,0.04);
                color: {_TEXT_PRI};
            }}
            QPushButton:checked {{
                background-color: rgba(245,158,11,0.08);
                color: {_ACCENT};
                font-weight: 600;
                border-left: 2px solid {_ACCENT};
                padding-left: 16px;
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo block
        logo = QWidget()
        logo.setFixedHeight(68)
        logo.setStyleSheet(f"background-color: {_BG}; border-bottom: 1px solid {_BORDER};")
        logo_layout = QVBoxLayout(logo)
        logo_layout.setContentsMargins(18, 0, 0, 0)
        logo_layout.setSpacing(1)

        name = QLabel("INVEST")
        name.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 15px; font-weight: 800; letter-spacing: 2.5px; border: none;")
        sub = QLabel("PORTFOLIO  OPTIMIZER")
        sub.setStyleSheet(f"color: {_ACCENT}; font-size: 7.5px; letter-spacing: 2px; font-weight: 600; border: none;")

        logo_layout.addStretch()
        logo_layout.addWidget(name)
        logo_layout.addWidget(sub)
        logo_layout.addStretch()
        layout.addWidget(logo)

        layout.addSpacing(8)

        # Nav section label
        nav_lbl = QLabel("NAVIGATION")
        nav_lbl.setStyleSheet(f"""
            color: {_TEXT_SEC};
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 1.2px;
            padding: 0 18px 6px 18px;
            border: none;
        """)
        layout.addWidget(nav_lbl)

        self._btn_data = self._nav_btn("Market Data")
        self._btn_algo = self._nav_btn("Optimization")
        self._btn_back = self._nav_btn("Backtesting")

        self._btn_data.clicked.connect(lambda: self._switch(0))
        self._btn_algo.clicked.connect(lambda: self._switch(1))
        self._btn_back.clicked.connect(lambda: self._switch(2))

        for btn in (self._btn_data, self._btn_algo, self._btn_back):
            layout.addWidget(btn)

        layout.addStretch()

        ver = QLabel("v0.1 Alpha  ·  © 2026")
        ver.setStyleSheet(f"color: {_BORDER}; font-size: 9px; padding: 0 18px 14px 18px; border: none;")
        ver.setAlignment(Qt.AlignLeft)
        layout.addWidget(ver)

        self._btn_data.setChecked(True)
        return sidebar

    @staticmethod
    def _nav_btn(text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setFixedHeight(40)
        return btn

    # ─────────────────────────────────────────────────────────────────────────
    # Pages
    # ─────────────────────────────────────────────────────────────────────────

    def _build_market_page(self) -> QWidget:
        """Market Data Explorer: ticker table + StockChartWidget + Sync button."""
        page = QWidget()
        page.setStyleSheet(f"background-color: {_BG};")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(f"""
            background-color: {_SURFACE};
            border-bottom: 1px solid {_BORDER};
        """)
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Market Data")
        title.setStyleSheet(f"color: {_TEXT_PRI}; font-size: 14px; font-weight: 700; letter-spacing: 0.3px;")
        tb_layout.addWidget(title)
        tb_layout.addStretch()

        self.btn_sync = QPushButton("Sync Data")
        self.btn_sync.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {_ACCENT};
                border: 1px solid rgba(245,158,11,0.4);
                border-radius: 4px;
                padding: 5px 14px;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
            }}
            QPushButton:hover {{
                background-color: rgba(245,158,11,0.1);
                border-color: {_ACCENT};
            }}
            QPushButton:pressed {{
                background-color: rgba(245,158,11,0.2);
            }}
        """)
        self.btn_sync.clicked.connect(self._on_sync_clicked)
        tb_layout.addWidget(self.btn_sync)

        layout.addWidget(topbar)

        # Content stack (data view / spinner)
        self.market_stack = QStackedWidget()
        self.market_stack.setStyleSheet(f"background-color: {_BG};")

        # 1. Data view
        data_view = QWidget()
        data_view.setStyleSheet(f"background-color: {_BG};")
        content = QHBoxLayout(data_view)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Ticker", "Index"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setFixedWidth(260)
        self._table.setAlternatingRowColors(True)
        self._table.itemClicked.connect(self._on_ticker_selected)
        self._table.setStyleSheet(f"background-color: {_SURFACE}; border-right: 1px solid {_BORDER};")
        content.addWidget(self._table)

        self._stock_widget = StockChartWidget()
        self._stock_widget.setStyleSheet(f"background-color: {_BG};")
        content.addWidget(self._stock_widget, 1)

        self.market_stack.addWidget(data_view)

        # 2. Spinner
        self.market_spinner = OptimizingSpinner()
        self.market_stack.addWidget(self.market_spinner)

        layout.addWidget(self.market_stack, 1)

        self._load_tickers()
        return page

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────────────────────────

    def _switch(self, index: int) -> None:
        self._pages.setCurrentIndex(index)
        for i, btn in enumerate((self._btn_data, self._btn_algo, self._btn_back)):
            btn.setChecked(i == index)

    # ─────────────────────────────────────────────────────────────────────────
    # Data
    # ─────────────────────────────────────────────────────────────────────────

    def _load_tickers(self) -> None:
        tickers = sorted(self._repo.get_all_tickers())
        self._table.setRowCount(len(tickers))
        self._table.setSortingEnabled(False)
        for row, ticker in enumerate(tickers):
            self._table.setItem(row, 0, QTableWidgetItem(ticker))
            self._table.setItem(row, 1, QTableWidgetItem("S&P 500"))
        self._table.setSortingEnabled(True)

        # Auto-select the first ticker so the chart is never blank on startup
        if tickers:
            self._table.selectRow(0)
            quotes = self._repo.get_quotes(tickers[0])
            if not quotes.empty:
                self._stock_widget.load(ticker=tickers[0], data=quotes)

    def _on_ticker_selected(self, item: QTableWidgetItem) -> None:
        ticker = self._table.item(item.row(), 0).text()
        quotes = self._repo.get_quotes(ticker)
        if not quotes.empty:
            self._stock_widget.load(ticker=ticker, data=quotes)
        else:
            logger.warning("No data for ticker: %s", ticker)

    # ─────────────────────────────────────────────────────────────────────────
    # Sync
    # ─────────────────────────────────────────────────────────────────────────

    def _on_sync_clicked(self) -> None:
        self.market_stack.setCurrentIndex(1)
        self.market_spinner.start("Connecting to market data feed...")

        self._worker = DataSyncWorker(self.core)
        self._worker.progress_updated.connect(self.market_spinner.update_progress)
        self._worker.finished.connect(self._on_sync_finished)
        self._worker.error.connect(self._on_sync_error)
        self._worker.start()

    def _on_sync_finished(self, result: dict) -> None:
        self.market_spinner.stop()
        self.market_stack.setCurrentIndex(0)
        self._load_tickers()

        QMessageBox.information(
            self,
            "Sync Complete",
            f"Updated {result.get('assets_updated', 0)} assets successfully."
        )

    def _on_sync_error(self, err_msg: str) -> None:
        self.market_spinner.stop()
        self.market_stack.setCurrentIndex(0)
        logger.error("Data sync failed: %s", err_msg)
        QMessageBox.critical(self, "Sync Error", f"Market data sync failed:\n\n{err_msg}")