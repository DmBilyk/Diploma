from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class BaseOptimizer(ABC):
    """Abstract base class for plugin optimisation algorithms."""

    @abstractmethod
    def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
        """Run the optimisation and return portfolio weights.

        Parameters
        ----------
        prices_df : pd.DataFrame
            Historical price matrix (rows = dates, columns = tickers).
        config_dict : dict
            Algorithm configuration (e.g. ``max_cardinality``).

        Returns
        -------
        Dict[str, float]
            Mapping ``{ticker: weight}`` whose values sum to 1.0.
        """
        pass
