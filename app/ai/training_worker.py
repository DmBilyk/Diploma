"""
training_worker.py
==================
QThread wrapper, який запускає навчання в ізольованому процесі (QProcess).
Перехоплює stdout для оновлення прогрес-бару в GUI.
"""

import logging
import os
from typing import List, Optional
from PySide6.QtCore import QThread, Signal, QProcess

from app.ai.lstm_model import PortfolioLSTMModel

logger = logging.getLogger(__name__)

class TrainingWorker(QThread):
    progress = Signal(int, float, float)      # epoch, train_loss, val_loss
    finished = Signal(object, list)            # PortfolioLSTMModel, tickers
    error = Signal(str)

    def __init__(
        self,
        repo_placeholder=None, # Більше не передаємо сюди реальний repo, щоб уникнути конфліктів
        tickers: Optional[List[str]] = None,
        seq_length: int = 52,
        epochs: int = 50,
        batch_size: int = 32,
        parent=None,
    ):
        super().__init__(parent)
        self.tickers = tickers or []
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_path = os.path.join(os.getcwd(), "temp_portfolio_model.keras")

    def run(self) -> None:
        process = QProcess()
        # Об'єднуємо вивід помилок і звичайний вивід в один потік
        process.setProcessChannelMode(QProcess.MergedChannels)

        # Формуємо команду запуску
        # Використовуємо той самий інтерпретатор Python, в якому запущено програму
        import sys
        python_exe = sys.executable

        args = [
            "app/ai/ai_runner.py",
            "--save_path", self.save_path,
            "--epochs", str(self.epochs),
            "--batch_size", str(self.batch_size),
            "--seq_length", str(self.seq_length)
        ]

        if self.tickers:
            args.append("--tickers")
            args.extend(self.tickers)

        process.start(python_exe, args)

        used_tickers = []
        process_failed = False
        error_msg = "Unknown error occurred during training."

        # Читаємо вивід з процесу в реальному часі
        while process.waitForReadyRead(-1):
            output = process.readAllStandardOutput().data().decode('utf-8').strip()

            for line in output.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if line.startswith("PROGRESS|"):
                    parts = line.split("|")
                    if len(parts) == 4:
                        self.progress.emit(int(parts[1]), float(parts[2]), float(parts[3]))

                elif line.startswith("DONE|"):
                    used_tickers = line.split("|")[1].split(",")

                elif line.startswith("ERROR|"):
                    error_msg = line.split("|")[1]
                    process_failed = True

        process.waitForFinished(-1)

        # Перевіряємо статус завершення
        if process.exitStatus() == QProcess.NormalExit and process.exitCode() == 0 and not process_failed:
            try:
                # Навчання пройшло успішно. Тепер безпечно завантажуємо готову модель в основний потік
                loaded_model = PortfolioLSTMModel.load(self.save_path)
                self.finished.emit(loaded_model, used_tickers)
            except Exception as e:
                self.error.emit(f"Failed to load saved model: {str(e)}")
        else:
            self.error.emit(f"Training process failed: {error_msg}")