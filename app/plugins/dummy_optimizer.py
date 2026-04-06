import pandas as pd
from typing import Dict
from app.plugins.base_optimizer import BaseOptimizer


class EqualWeightOptimizer(BaseOptimizer):
    """Тестовий плагін: розподіляє капітал рівними частинами."""

    def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
        tickers = prices_df.columns
        n = len(tickers)
        if n == 0:
            return {}
        return {ticker: 1.0 / n for ticker in tickers}