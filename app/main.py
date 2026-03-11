import sys
import os

# Додаємо кореневу папку в шлях, щоб Python бачив модулі app.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow
from app.data.models import init_db


def main():
    # 1. Ініціалізація бази даних (перевірка структури)
    init_db()

    # 2. Запуск GUI
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Сучасний стиль для Windows/Mac

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()