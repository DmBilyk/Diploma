"""
app/ai
======

Public API for the PPO portfolio optimiser package.

Public surface
--------------
``PortfolioEnv``          – Gymnasium-compatible environment
``PPOPortfolioTrainer``   – PPO training workflow
``PPOInference``          – model loading and weight extraction
``load_and_prepare``      – market-data cleaning and splitting
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
