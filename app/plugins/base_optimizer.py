from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

class BaseOptimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    Each plugin algorithm must inherit from this class and implement the optimize method.
    """
    
    @abstractmethod
    def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
        """
        Executes the optimization process based on the given prices and configuration.
        
        Args:
            prices_df (pd.DataFrame): Historical prices data.
            config_dict (dict): Optimization configuration parameters.
            
        Returns:
            Dict[str, float]: A dictionary containing optimal weights for each asset in a single standard.
        """
        pass
