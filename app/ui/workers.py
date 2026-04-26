import logging
from PySide6.QtCore import QThread, Signal
from app.core.core import PortfolioCore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Market-data sync worker
# ══════════════════════════════════════════════════════════════════════════════
class DataSyncWorker(QThread):
    progress_updated = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, core_instance: PortfolioCore):
        super().__init__()
        self._core = core_instance

    def run(self):
        """Run market-data synchronisation in a background thread."""
        try:
            def callback(percent: int, msg: str):
                self.progress_updated.emit(percent, msg)

            result = self._core.sync_market_data(progress_callback=callback)
            self.finished.emit(result)
        except Exception as e:
            logger.exception("DataSyncWorker failed")
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Optimisation and backtest worker
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
QThread worker for background database loading.
The UI remains responsive while progress is delivered through Qt signals.
"""
class DataLoaderWorker(QThread):
    """
    Run ``ensure_database_populated`` outside the UI thread and forward
    progress through Qt signals.
    """

    progress = Signal(int, str)
    finished = Signal(bool)
    error = Signal(str)

    def run(self):
        """Execute the bootstrap check in a worker thread."""
        try:
            from app.data.db_bootstrap import ensure_database_populated

            def on_progress(percent: int, message: str):
                # Qt queues signal delivery safely across threads.
                self.progress.emit(percent, message)

            result = ensure_database_populated(progress_callback=on_progress)
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))
