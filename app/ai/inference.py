"""
app/ai/inference.py
===================

Inference utilities for turning a trained PPO policy into portfolio weights.

The main entry point is :class:`PPOInference`. It loads a saved model, runs it
over an out-of-sample return matrix, and returns ``Dict[str, float]`` weights
compatible with ``BacktestEngine`` and ``PortfolioCore``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from app.ai.environment import PortfolioEnv

logger = logging.getLogger(__name__)


class PPOInference:
    """Run a trained PPO model and extract a clean allocation.

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.zip`` file produced by :class:`~app.ai.trainer.PPOPortfolioTrainer`.
    env_kwargs : dict | None
        Extra kwargs forwarded to :class:`~app.ai.environment.PortfolioEnv`
        and expected to match the training environment.
    """

    def __init__(
        self,
        model_path: str | Path,
        env_kwargs: Optional[dict] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.env_kwargs = env_kwargs or {}
        self._model: Optional[PPO] = None

    # ────────────────────────────────────────────────────────────────────
    #  Loading
    # ────────────────────────────────────────────────────────────────────

    def load(self) -> "PPOInference":
        """Load model weights into memory before inference."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"PPO model file not found: {self.model_path}. "
                "Train the model first or check the path."
            )
        self._model = PPO.load(str(self.model_path))
        logger.info("PPO model loaded from %s", self.model_path)
        return self

    # ────────────────────────────────────────────────────────────────────
    #  Inference helpers
    # ────────────────────────────────────────────────────────────────────

    def run_episode(
        self,
        returns_df: pd.DataFrame,
        initial_balance: float = 100_000.0,
    ) -> Tuple[List[float], List[np.ndarray]]:
        """Run the policy across one complete inference episode.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns for the inference period, usually out-of-sample data.
        initial_balance : float

        Returns
        -------
        portfolio_values : list[float]
            Portfolio value at each step. Length is len(returns_df) + 1, and
            the first entry is
            ``initial_balance``).
        weight_history : list[np.ndarray]
            Target weights chosen at each step.
        """
        if self._model is None:
            self.load()

        env = PortfolioEnv(returns_df, initial_balance=initial_balance, **self.env_kwargs)
        obs, _ = env.reset()
        done = False
        weight_history: List[np.ndarray] = []

        while not done:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            weight_history.append(info["weights"])

        return env.history, weight_history

    def get_final_weights(
        self,
        returns_df: pd.DataFrame,
        top_n: Optional[int] = None,
        min_weight: float = 0.01,
    ) -> Dict[str, float]:
        """Return the agent's terminal allocation.

        The result is suitable for constructing a ``PortfolioSpec`` for
        ``BacktestEngine``.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Return history used to derive the allocation.
        top_n : int | None
            Keep only the ``top_n`` highest-weight assets.  ``None`` keeps all
            assets that pass ``min_weight``.
        min_weight : float
            Assets below this threshold are removed before normalisation.

        Returns
        -------
        Dict[str, float]
            ``{ticker: weight}``, normalised to sum to 1.0.
        """
        _, weight_history = self.run_episode(returns_df)
        # Smooth the terminal allocation with the most recent policy choices.
        tail = min(20, len(weight_history))
        mean_weights = np.mean(weight_history[-tail:], axis=0)

        tickers = list(returns_df.columns)
        weights = dict(zip(tickers, mean_weights.tolist()))

        # Remove tiny positions that would not be useful in a static portfolio.
        weights = {t: w for t, w in weights.items() if w >= min_weight}

        # Enforce an optional cardinality limit.
        if top_n is not None and len(weights) > top_n:
            sorted_w = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
            weights = dict(sorted_w[:top_n])

        # Normalise after pruning so the allocation remains fully invested.
        total = sum(weights.values())
        if total < 1e-8:
            raise RuntimeError("All weights are zero after pruning – check your data.")
        weights = {t: w / total for t, w in weights.items()}

        logger.info(
            "Final weights extracted – %d assets, top=%s min_weight=%.3f",
            len(weights), top_n, min_weight,
        )
        return weights

    def get_average_weights(
        self,
        returns_df: pd.DataFrame,
        min_weight: float = 0.01,
    ) -> Dict[str, float]:
        """Return weights averaged across the whole episode.

        This is useful when the policy changes slowly and a representative
        allocation is preferable to the final step alone.
        """
        _, weight_history = self.run_episode(returns_df)
        mean_weights = np.mean(weight_history, axis=0)
        tickers = list(returns_df.columns)
        weights = {t: float(w) for t, w in zip(tickers, mean_weights)}
        weights = {t: w for t, w in weights.items() if w >= min_weight}
        total = sum(weights.values())
        if total < 1e-8:
            raise RuntimeError("All average weights are zero after pruning – check your data.")
        return {t: w / total for t, w in weights.items()}
