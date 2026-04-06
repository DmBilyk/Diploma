import pandas as pd
import numpy as np
from typing import Dict
from app.plugins.base_optimizer import BaseOptimizer


class InverseVolatilityOptimizer(BaseOptimizer):
    """
    Плагін: Inverse Volatility (Зворотна волатильність).
    Стратегія, яка розподіляє капітал обернено пропорційно до ризику активу.
    Найбільш стабільні активи отримують найбільшу вагу.
    """

    def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
        # 1. Зчитуємо ліміт активів з UI (за замовчуванням 15)
        max_k = config_dict.get("max_cardinality", 15)

        # 2. Рахуємо дохідність кожного періоду
        returns = prices_df.pct_change().dropna()

        # 3. Рахуємо стандартне відхилення (волатильність) кожного активу
        volatilities = returns.std()

        # Захист від ділення на нуль (якщо актив має нульову волатильність)
        volatilities = volatilities.replace(0, np.nan).dropna()

        # 4. Рахуємо зворотну волатильність (1 / ризик)
        inv_vol = 1.0 / volatilities

        # 5. Відбираємо Топ-K найбільш стабільних активів (найбільше значення inv_vol)
        top_inv_vol = inv_vol.nlargest(max_k)

        # 6. Нормалізуємо ваги так, щоб їхня сума дорівнювала 1.0 (100%)
        weights = top_inv_vol / top_inv_vol.sum()

        # Повертаємо словник у форматі {ticker: weight}
        return weights.to_dict()