import os
import argparse
import sys

# Глушимо всі конфлікти потоків до імпорту TF
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from app.ai.lstm_model import PortfolioLSTMModel
from app.data.repository import PortfolioRepository
from app.ai.create_dataset import prepare_lstm_data


class IPC_Callback(tf.keras.callbacks.Callback):
    """Відправляє прогрес у stdout для QProcess"""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Формат: PROGRESS|epoch|train_loss|val_loss
        print(f"PROGRESS|{epoch + 1}|{logs.get('loss', 0):.6f}|{logs.get('val_loss', 0):.6f}", flush=True)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs='*', help="Список тікерів (опціонально)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_length", type=int, default=52)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    try:
        repo = PortfolioRepository()
        dataset, used_tickers = prepare_lstm_data(
            repo=repo,
            tickers=args.tickers,
            seq_length=args.seq_length
        )

        X, y = dataset.X.numpy(), dataset.y.numpy()
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        model = PortfolioLSTMModel(seq_length=args.seq_length, num_assets=X.shape[2])

        model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[IPC_Callback()]
        )

        model.save(args.save_path)

        # Сигнал успішного завершення з переліком використаних активів
        print(f"DONE|{','.join(used_tickers)}", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"ERROR|{str(e)}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    run()