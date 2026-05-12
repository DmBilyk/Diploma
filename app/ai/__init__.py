"""
app/ai
======

Public API for the PPO portfolio optimiser package.
Requires the AI extras: pip install -r requirements-ai.txt

Public surface
--------------
``PortfolioEnv``          – Gymnasium-compatible environment
``PPOPortfolioTrainer``   – PPO training workflow
``PPOInference``          – model loading and weight extraction
``load_and_prepare``      – market-data cleaning and splitting
"""


def __getattr__(name: str):
    if name == "PortfolioEnv":
        from app.ai.environment import PortfolioEnv
        return PortfolioEnv
    if name == "PPOPortfolioTrainer":
        from app.ai.trainer import PPOPortfolioTrainer
        return PPOPortfolioTrainer
    if name == "PPOInference":
        from app.ai.inference import PPOInference
        return PPOInference
    if name in ("load_and_prepare", "DataSplit"):
        from app.ai import data_prep
        return getattr(data_prep, name)
    raise AttributeError(f"module 'app.ai' has no attribute {name!r}")


__all__ = [
    "PortfolioEnv",
    "PPOPortfolioTrainer",
    "PPOInference",
    "load_and_prepare",
    "DataSplit",
]
