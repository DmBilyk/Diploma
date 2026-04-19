import logging
from PySide6.QtCore import QThread, Signal
from app.core.core import PortfolioCore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Воркер для синхронізації ринкових даних
# ══════════════════════════════════════════════════════════════════════════════
class DataSyncWorker(QThread):
    progress_updated = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, core_instance: PortfolioCore):
        super().__init__()
        self._core = core_instance

    def run(self):
        """Цей код виконується у фоновому потоці."""
        try:
            def callback(percent: int, msg: str):
                self.progress_updated.emit(percent, msg)

            result = self._core.sync_market_data(progress_callback=callback)
            self.finished.emit(result)
        except Exception as e:
            logger.exception("DataSyncWorker failed")
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Воркер для запуску оптимізації та бектесту
# ══════════════════════════════════════════════════════════════════════════════
class BacktestWorker(QThread):
    progress_updated = Signal(int, str)
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, core_instance: PortfolioCore, params: dict):
        super().__init__()
        self._core = core_instance
        self._params = params

    def run(self):
        try:
            self.progress_updated.emit(5, "Завантаження цінових даних...")
            self.progress_updated.emit(20, "Ініціалізація популяції...")
            result, report = self._core.run_and_backtest(**self._params)
            self.progress_updated.emit(100, "Готово!")
            self.finished.emit(result, report)
        except Exception as exc:
            logger.exception("BacktestWorker failed")
            self.error.emit(str(exc))



"""
==================
QThread-воркер для завантаження даних у фоновому потоці.
UI залишається живим, прогрес передається через сигнали.
"""
class DataLoaderWorker(QThread):
    """
    Запускає ensure_database_populated у фоновому потоці.
    Передає прогрес у головний потік через Qt-сигнали.
    """

    # Сигнал: (відсоток 0-100, повідомлення)
    progress = Signal(int, str)
    # Сигнал: завантаження завершено (True = були нові дані, False = БД вже заповнена)
    finished = Signal(bool)
    # Сигнал: помилка під час завантаження
    error = Signal(str)

    def run(self):
        """Виконується в окремому потоці."""
        try:
            from app.data.db_bootstrap import ensure_database_populated

            def on_progress(percent: int, message: str):
                # emit() безпечний між потоками — Qt поставить у чергу
                self.progress.emit(percent, message)

            result = ensure_database_populated(progress_callback=on_progress)
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))