import pandas as pd
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QStackedWidget, QLabel, QFrame,
                               QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from app.data.repository import PortfolioRepository


class MplCanvas(FigureCanvas):
    """Полотно для малювання графіків Matplotlib"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        # Встановлюємо білий фон для самого графіка, щоб він був контрастним
        # у будь-якій темі (інакше чорні літери на темному фоні не видно)
        fig.patch.set_facecolor('white')

        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.repo = PortfolioRepository()
        self.setWindowTitle("InvestPortfolio Optimizer (Diploma)")
        self.resize(1280, 800)

        # Головний макет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Створюємо Sidebar та Pages
        self.sidebar = self.create_sidebar()
        self.main_layout.addWidget(self.sidebar)

        self.pages = QStackedWidget()
        self.main_layout.addWidget(self.pages)

        # Створення сторінок
        self.init_data_page()

        # Додавання сторінок у стек
        self.pages.addWidget(self.page_data)
        self.pages.addWidget(self.create_placeholder_page("Optimization Algorithms"))
        self.pages.addWidget(self.create_placeholder_page("Backtesting Results"))

    def create_sidebar(self):
        sidebar = QFrame()
        # Sidebar залишаємо темним завжди - це акцент дизайну
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2C3E50;
                color: white;
                border: none;
            }
            QPushButton {
                background-color: transparent;
                color: #BDC3C7;
                text-align: left;
                padding: 15px;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #34495E;
                color: white;
            }
            QPushButton:checked {
                background-color: #1ABC9C;
                color: white;
                font-weight: bold;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 20px;
                padding: 20px;
            }
        """)
        sidebar.setFixedWidth(250)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)

        # Заголовок
        title = QLabel("Portfolio\nOptimizer AI")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(20)

        # Кнопки навігації
        self.btn_data = QPushButton("📊 Market Data")
        self.btn_algo = QPushButton("🧮 Optimization")
        self.btn_backtest = QPushButton("📈 Backtesting")

        self.btn_data.setCheckable(True)
        self.btn_algo.setCheckable(True)
        self.btn_backtest.setCheckable(True)

        # Логіка перемикання
        self.btn_data.clicked.connect(lambda: self.switch_page(0))
        self.btn_algo.clicked.connect(lambda: self.switch_page(1))
        self.btn_backtest.clicked.connect(lambda: self.switch_page(2))

        layout.addWidget(self.btn_data)
        layout.addWidget(self.btn_algo)
        layout.addWidget(self.btn_backtest)

        layout.addStretch()

        # Підвал
        footer = QLabel("v0.1 Alpha\n© 2026 Diploma")
        footer.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)
        layout.addSpacing(10)

        # Активуємо першу кнопку
        self.btn_data.setChecked(True)

        return sidebar

    def create_placeholder_page(self, text):
        """Тимчасова заглушка для сторінок"""
        page = QWidget()
        # Прибрали жорсткий колір фону, тепер він системний
        layout = QVBoxLayout(page)

        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        # Колір тексту не задаємо жорстко, щоб він адаптувався під тему
        label.setStyleSheet("font-size: 32px; font-weight: bold;")

        layout.addWidget(label)
        return page

    def init_data_page(self):
        """Ініціалізація сторінки Market Data Explorer"""
        self.page_data = QWidget()
        # Прибрали background-color, щоб використовувати системний (темний/світлий)

        layout = QVBoxLayout(self.page_data)

        header = QLabel("Market Data Explorer")
        # Прибрали колір тексту, щоб він був системним (білим на темному, чорним на світлому)
        header.setStyleSheet("font-size: 24px; font-weight: bold; margin: 10px;")
        layout.addWidget(header)

        content = QHBoxLayout()

        # 1. Таблиця активів
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Ticker", "Sector"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemClicked.connect(self.on_asset_selected)

        # Стилізація таблиці: мінімалізм, щоб не ламати темну тему
        self.table.setStyleSheet("""
            QHeaderView::section { background-color: #34495E; color: white; padding: 5px; }
        """)
        content.addWidget(self.table, 1)

        # 2. Графік
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        content.addWidget(self.canvas, 2)

        layout.addLayout(content)
        self.load_assets_to_table()

    def switch_page(self, index):
        self.pages.setCurrentIndex(index)
        self.btn_data.setChecked(index == 0)
        self.btn_algo.setChecked(index == 1)
        self.btn_backtest.setChecked(index == 2)

    def load_assets_to_table(self):
        """Завантаження списку тікерів з БД у таблицю"""
        tickers = self.repo.get_all_tickers()

        # Сортуємо для зручності
        tickers.sort()

        self.table.setRowCount(len(tickers))
        self.table.setSortingEnabled(False)

        for i, ticker in enumerate(tickers):
            self.table.setItem(i, 0, QTableWidgetItem(ticker))
            self.table.setItem(i, 1, QTableWidgetItem("S&P 500"))

        self.table.setSortingEnabled(True)

    def on_asset_selected(self, item):
        """Обробка вибору активу користувачем"""
        row = item.row()
        ticker = self.table.item(row, 0).text()

        # Отримуємо історичні дані
        quotes = self.repo.get_quotes(ticker)
        if not quotes.empty:
            self.update_chart(ticker, quotes)

    def update_chart(self, ticker, data):
        """Візуалізація графіка"""
        self.canvas.axes.cla()  # Очистити старий графік

        # Малюємо Adj Close
        self.canvas.axes.plot(data['date'], data['adj_close'], label='Adj Close', color='#2980B9', linewidth=1.5)

        self.canvas.axes.set_title(f"Price History: {ticker}", fontsize=12, fontweight='bold')
        self.canvas.axes.legend(loc='upper left')
        self.canvas.axes.grid(True, linestyle='--', alpha=0.5)

        # Форматування осі X
        self.canvas.axes.tick_params(axis='x', rotation=45)

        self.canvas.figure.tight_layout()
        self.canvas.draw()