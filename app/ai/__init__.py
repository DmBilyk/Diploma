"""
app/ai
======

PPO-based reinforcement-learning portfolio optimiser.

Public surface
--------------
``PortfolioEnv``          – Gymnasium environment
``PPOPortfolioTrainer``   – training pipeline
``PPOInference``          – load model + extract weights
``load_and_prepare``      – data loading & splitting
"""

from app.ai.environment import PortfolioEnv
from app.ai.inference import PPOInference
from app.ai.trainer import PPOPortfolioTrainer
from app.ai.data_prep import load_and_prepare, DataSplit

__all__ = [
    "PortfolioEnv",
    "PPOPortfolioTrainer",
    "PPOInference",
    "load_and_prepare",
    "DataSplit",
]