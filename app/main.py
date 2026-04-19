import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication, QMessageBox

from app.ui.main_window import MainWindow
from app.ui.loading_window import LoadingWindow
from app.ui.workers import DataLoaderWorker


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── 1. Показуємо вікно завантаження ───────────────────────────────────
    loading = LoadingWindow()
    loading.show()

    # Тримаємо посилання, щоб GC не знищив вікно після on_finished
    main_window: list[MainWindow] = []

    # ── 2. Колбеки воркера ────────────────────────────────────────────────

    def on_progress(percent: int, message: str):
        loading.update_progress(percent, message)

    def on_finished(_loaded: bool):
        loading.update_progress(100, "Запуск інтерфейсу користувача...")

        window = MainWindow()
        main_window.append(window)
        window.show()

        loading.close()
        worker.deleteLater()

    def on_error(message: str):
        loading.close()
        QMessageBox.critical(
            None,
            "Помилка ініціалізації",
            f"Не вдалося завантажити дані:\n\n{message}\n\n"
            "Перевірте з'єднання з інтернетом і запустіть застосунок знову.",
        )
        app.quit()

    # ── 3. Запускаємо воркер ──────────────────────────────────────────────
    worker = DataLoaderWorker()
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()