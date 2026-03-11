import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Optional

from app.data.repository import PortfolioRepository

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class PortfolioLSTMDataset:
    """
    Lightweight NumPy-based dataset (replaces PyTorch Dataset).
    Зберігає X та y як float32 NumPy масиви.
    Має атрибути .X та .y з методом .numpy() для сумісності з predictor.py
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]

    # Методи-заглушки для сумісності з кодом, що викликає .numpy()
    class _NumpyWrapper:
        def __init__(self, arr):
            self._arr = arr

        def numpy(self):
            return self._arr

        def __getitem__(self, idx):
            # Повертає wrapper щоб X[-1:].numpy() працювало
            return PortfolioLSTMDataset._NumpyWrapper(self._arr[idx])

        def __len__(self):
            return len(self._arr)

    @property
    def X(self):
        return self._X_wrapper

    @X.setter
    def X(self, arr):
        self._X_arr = arr
        self._X_wrapper = self._NumpyWrapper(arr)

    @property
    def y(self):
        return self._y_wrapper

    @y.setter
    def y(self, arr):
        self._y_arr = arr
        self._y_wrapper = self._NumpyWrapper(arr)


def prepare_lstm_data(
        repo: PortfolioRepository,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        seq_length: int = 52,
        horizon: int = 1
) -> Tuple[PortfolioLSTMDataset, List[str]]:
    """
    Завантажує дані з БД, рахує логарифмічні дохідності та створює вікна для LSTM.
    """
    if tickers is None:
        tickers = repo.get_all_tickers()

    logger.info("Завантаження даних з бази...")
    prices_df = repo.get_price_history(tickers, start_date=start_date, end_date=end_date)

    if prices_df.empty:
        raise ValueError("Немає даних у базі за цей період.")

    # 1. Відкидаємо активи з забагато пропусків
    threshold = int(len(prices_df) * 0.8)
    valid_tickers = prices_df.dropna(axis=1, thresh=threshold).columns.tolist()
    prices_df = prices_df[valid_tickers].ffill().dropna()

    logger.info(f"Відібрано {len(valid_tickers)} активів після очищення.")

    # 2. Логарифмічні дохідності
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()

    # 3. Ковзні вікна
    data_values = log_returns.values
    num_samples = len(data_values) - seq_length - horizon + 1

    X_list, y_list = [], []

    logger.info(f"Створення ковзних вікон (seq_length={seq_length}, horizon={horizon})...")
    for i in range(num_samples):
        X_window = data_values[i: i + seq_length, :]
        y_target = data_values[i + seq_length + horizon - 1, :]
        X_list.append(X_window)
        y_list.append(y_target)

    X_arr = np.array(X_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32)

    logger.info(f"Створено масиви: X: {X_arr.shape}, y: {y_arr.shape}")

    dataset = PortfolioLSTMDataset(X_arr, y_arr)
    return dataset, valid_tickers


if __name__ == "__main__":
    repo = PortfolioRepository()

    dataset, used_tickers = prepare_lstm_data(
        repo=repo,
        start_date="2010-01-01",
        end_date="2019-12-31",
        seq_length=26
    )

    X_batch = dataset.X.numpy()[:32]
    y_batch = dataset.y.numpy()[:32]
    print(f"\n✅ Dataset працює!")
    print(f"Розмірність X батчу: {X_batch.shape}  -> (Batch_Size, Sequence_Length, Num_Assets)")
    print(f"Розмірність y батчу: {y_batch.shape}  -> (Batch_Size, Num_Assets)")