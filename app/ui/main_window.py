from __future__ import annotations
"""
app/ui/main_window.py — Головне вікно програми InvestPortfolio Optimizer.

Структура:
- Фіксований темний sidebar зі навігаційними кнопками
- QStackedWidget:
    0 — Market Data Explorer (таблиця тікерів + StockChartWidget)
    1 — Optimization  (плейсхолдер)
    2 — Backtesting   (плейсхолдер)
"""

import logging



from PySide6.QtCore import Qt, QThread, Signal                        # pylint: disable=no-name-in-module
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
from app.core.core import PortfolioCore
from app.ui.workers import DataSyncWorker

logger = logging.getLogger(__name__)

_ACCENT = "#1ABC9C"

# ═══════════════════════════════════════════════════════════════════════════════
# Головне вікно
# ═══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """
    Головне вікно програми.

    Layout: [Sidebar 220px] | [QStackedWidget — решта ширини]
    """

    _SIDEBAR_W = 220
    _WINDOW_SIZE = (1280, 800)

    def __init__(self) -> None:
        super().__init__()

        self.core = PortfolioCore()
        self._repo = self.core.repo

        self.setWindowTitle("InvestPortfolio Optimizer")
        self.resize(*self._WINDOW_SIZE)
        self._apply_global_style()
        self._build_layout()

    # ─────────────────────────────────────────────────────────────────────────
    # Глобальні стилі
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_global_style(self) -> None:
        """
        Мінімальний загальний stylesheet.
        Не перебиває системну палітру Qt — лише точкові уточнення.
        """
        self.setStyleSheet(f"""
            QTableWidget {{
                border: none;
                outline: none;
                font-size: 12px;
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid rgba(128,128,128,0.15);
            }}
            QTableWidget::item:selected {{
                background-color: {_ACCENT};
                color: white;
            }}
            QHeaderView::section {{
                background-color: #34495E;
                color: #BDC3C7;
                padding: 7px 10px;
                border: none;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 0.5px;
                text-transform: uppercase;
            }}
            QScrollBar:vertical {{
                border: none;
                width: 6px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(128,128,128,0.4);
                border-radius: 3px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # Побудова макету
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
        self._pages.addWidget(self._build_optimization_page())
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
                background-color: #1C2833;
                border: none;
            }}
            QPushButton {{
                background: transparent;
                color: #7F8C8D;
                text-align: left;
                padding: 13px 20px;
                border: none;
                font-size: 13px;
                border-radius: 0;
            }}
            QPushButton:hover {{
                background-color: #273444;
                color: #ECF0F1;
            }}
            QPushButton:checked {{
                background-color: transparent;
                color: {_ACCENT};
                font-weight: bold;
                border-left: 3px solid {_ACCENT};
                padding-left: 17px;
            }}
        """)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Логотип / назва
        logo = QWidget()
        logo.setFixedHeight(72)
        logo.setStyleSheet("background-color: #161D27;")
        logo_layout = QVBoxLayout(logo)
        logo_layout.setContentsMargins(20, 0, 0, 0)
        logo_layout.setSpacing(2)

        name = QLabel("Portfolio")
        name.setStyleSheet("color: white; font-size: 17px; font-weight: 800; letter-spacing: 1px;")
        sub = QLabel("OPTIMIZER")
        sub.setStyleSheet(f"color: {_ACCENT}; font-size: 9px; letter-spacing: 2.5px;")

        logo_layout.addStretch()
        logo_layout.addWidget(name)
        logo_layout.addWidget(sub)
        logo_layout.addStretch()
        layout.addWidget(logo)

        layout.addWidget(self._hline())
        layout.addSpacing(8)

        self._btn_data = self._nav_btn("Market Data", "📊")
        self._btn_algo = self._nav_btn("Optimization", "🧮")
        self._btn_back = self._nav_btn("Backtesting", "📈")

        self._btn_data.clicked.connect(lambda: self._switch(0))
        self._btn_algo.clicked.connect(lambda: self._switch(1))
        self._btn_back.clicked.connect(lambda: self._switch(2))

        for btn in (self._btn_data, self._btn_algo, self._btn_back):
            layout.addWidget(btn)

        layout.addStretch()

        ver = QLabel("v0.1 Alpha  ·  © 2026")
        ver.setStyleSheet("color: #2E4057; font-size: 10px; padding: 0 20px 14px 20px;")
        ver.setAlignment(Qt.AlignLeft)
        layout.addWidget(ver)

        self._btn_data.setChecked(True)
        return sidebar

    @staticmethod
    def _nav_btn(text: str, icon: str) -> QPushButton:
        btn = QPushButton(f"  {icon}  {text}")
        btn.setCheckable(True)
        btn.setFixedHeight(44)
        return btn

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #2E4057; border: none;")
        return line

    # ─────────────────────────────────────────────────────────────────────────
    # Сторінки
    # ─────────────────────────────────────────────────────────────────────────

    def _build_market_page(self) -> QWidget:
        """Market Data Explorer: таблиця тікерів + StockChartWidget + Кнопка Sync."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Верхня панель ──
        topbar = QWidget()
        topbar.setFixedHeight(56)
        topbar.setStyleSheet("border-bottom: 1px solid rgba(128,128,128,0.2);")
        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Market Data")
        title.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 0.5px;")
        tb_layout.addWidget(title)
        tb_layout.addStretch()

        # Кнопка оновлення даних
        self.btn_sync = QPushButton("🔄 Синхронізувати дані")
        self.btn_sync.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {_ACCENT};
                border: 1px solid {_ACCENT};
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {_ACCENT};
                color: white;
            }}
        """)
        self.btn_sync.clicked.connect(self._on_sync_clicked)
        tb_layout.addWidget(self.btn_sync)

        layout.addWidget(topbar)

        # ── Стек контенту (Дані АБО Спінер) ──
        self.market_stack = QStackedWidget()

        # 1. Віджет з даними (Таблиця + Графік)
        data_view = QWidget()
        content = QHBoxLayout(data_view)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Ticker", "Sector"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setShowGrid(False)
        self._table.setFixedWidth(280)
        self._table.setAlternatingRowColors(True)
        self._table.itemClicked.connect(self._on_ticker_selected)
        content.addWidget(self._table)

        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setFixedWidth(1)
        vline.setStyleSheet("background-color: rgba(128,128,128,0.2); border: none;")
        content.addWidget(vline)

        self._stock_widget = StockChartWidget()
        content.addWidget(self._stock_widget, 1)

        self.market_stack.addWidget(data_view)

        # 2. Віджет спінера (ховається за замовчуванням)
        self.market_spinner = OptimizingSpinner()
        self.market_stack.addWidget(self.market_spinner)

        layout.addWidget(self.market_stack, 1)

        self._load_tickers()
        return page

    @staticmethod
    def _build_placeholder(text: str) -> QWidget:
        page = QWidget()
        lyt = QVBoxLayout(page)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "font-size: 28px; font-weight: 700; color: palette(mid); letter-spacing: 1px;"
        )
        lyt.addWidget(lbl)
        return page

    def _build_optimization_page(self) -> QWidget:
        """Сторінка оптимізації (поки що показує вічний красивий спінер)."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # Створюємо наш гарний спінер
        self.optimization_spinner = OptimizingSpinner(page)
        layout.addWidget(self.optimization_spinner)

        # Одразу запускаємо його з крутим текстом
        self.optimization_spinner.start(
            "Налаштування середовища...\nЗапуск еволюційних алгоритмів та LSTM"
        )

        return page
    # ─────────────────────────────────────────────────────────────────────────
    # Навігація
    # ─────────────────────────────────────────────────────────────────────────

    def _switch(self, index: int) -> None:
        self._pages.setCurrentIndex(index)
        for i, btn in enumerate((self._btn_data, self._btn_algo, self._btn_back)):
            btn.setChecked(i == index)

    # ─────────────────────────────────────────────────────────────────────────
    # Дані
    # ─────────────────────────────────────────────────────────────────────────

    def _load_tickers(self) -> None:
        """Завантажує список тікерів з БД у таблицю."""
        tickers = sorted(self._repo.get_all_tickers())
        self._table.setRowCount(len(tickers))
        self._table.setSortingEnabled(False)
        for row, ticker in enumerate(tickers):
            self._table.setItem(row, 0, QTableWidgetItem(ticker))
            self._table.setItem(row, 1, QTableWidgetItem("S&P 500"))
        self._table.setSortingEnabled(True)

    def _on_ticker_selected(self, item: QTableWidgetItem) -> None:
        """Обробляє клік по рядку таблиці: завантажує та відображає графік."""
        ticker = self._table.item(item.row(), 0).text()
        quotes = self._repo.get_quotes(ticker)
        if not quotes.empty:
            self._stock_widget.load(ticker=ticker, data=quotes)
        else:
            logger.warning("Немає даних для тікера: %s", ticker)

    # ─────────────────────────────────────────────────────────────────────────
    # Синхронізація (Фонові задачі)
    # ─────────────────────────────────────────────────────────────────────────

    def _on_sync_clicked(self) -> None:
        """Запускає фонове оновлення бази даних."""
        # Показуємо спінер
        self.market_stack.setCurrentIndex(1)
        self.market_spinner.start("З'єднання з сервером...")

        # Запускаємо Worker
        self._worker = DataSyncWorker(self.core)
        self._worker.progress_updated.connect(self.market_spinner.update_progress)
        self._worker.finished.connect(self._on_sync_finished)
        self._worker.error.connect(self._on_sync_error)
        self._worker.start()

    def _on_sync_finished(self, result: dict) -> None:
        """Викликається, коли завантаження успішно завершено."""
        self.market_spinner.stop()
        self.market_stack.setCurrentIndex(0)  # Повертаємо таблицю та графік
        self._load_tickers()  # Оновлюємо таблицю новими даними

        QMessageBox.information(
            self,
            "Синхронізація успішна",
            f"Оновлено активів: {result.get('assets_updated', 0)}"
        )

    def _on_sync_error(self, err_msg: str) -> None:
        """Обробка помилок під час завантаження."""
        self.market_spinner.stop()
        self.market_stack.setCurrentIndex(0)
        logger.error("Data sync failed: %s", err_msg)
        QMessageBox.critical(self, "Помилка синхронізації", f"Сталася помилка:\n{err_msg}")