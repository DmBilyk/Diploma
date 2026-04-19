"""
app/ai/environment.py
=====================

Gymnasium-compatible trading environment for portfolio optimisation with PPO.

Design decisions
----------------
* **Rolling window observation** – the agent sees the last ``window`` days of
  returns + current weights, giving it temporal context without LSTM overhead.
* **Multi-objective reward** – log-return penalised by drawdown and turnover,
  producing smoother, more robust policies than raw P&L.
* **Symmetric action space** – raw logits in [-1, 1] converted via softmax,
  so PPO's Gaussian policy can explore freely without hitting hard boundaries.
* **Proper episode termination** – no silent off-by-one resets.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """Continuous-action portfolio environment.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily (or weekly) returns matrix – shape (T, N).
    initial_balance : float
        Starting portfolio value in dollars.
    transaction_cost : float
        One-way proportional cost applied to turnover.
    window : int
        Number of past time-steps included in each observation.
    reward_drawdown_penalty : float
        Coefficient for the running drawdown penalty added to log-return reward.
    reward_turnover_penalty : float
        Extra proportional cost on turnover beyond actual fees (encourages
        the agent to *not* churn the portfolio).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        returns_df: pd.DataFrame,
        initial_balance: float = 100_000.0,
        transaction_cost: float = 0.001,
        window: int = 20,
        reward_drawdown_penalty: float = 0.1,
        reward_turnover_penalty: float = 0.002,
    ) -> None:
        super().__init__()

        self.returns_arr = returns_df.values.astype(np.float32)
        self.dates = returns_df.index
        self.tickers: List[str] = list(returns_df.columns)
        self.n_assets = self.returns_arr.shape[1]
        self.n_steps = self.returns_arr.shape[0]

        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.window = int(window)
        self.dd_penalty = float(reward_drawdown_penalty)
        self.to_penalty = float(reward_turnover_penalty)

        # ── Spaces ──────────────────────────────────────────────────────
        # Actions: raw logits, converted to weights via softmax internally.
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(self.n_assets,), dtype=np.float32
        )

        # Observations: (window × returns) + current weights + portfolio stats
        #   window × n_assets : scaled returns history
        #   n_assets           : current weights
        #   3                  : [log(value/initial), running_sharpe, current_drawdown]
        obs_dim = self.window * self.n_assets + self.n_assets + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ── Episode state (initialised in reset) ────────────────────────
        self.current_step: int = 0
        self.portfolio_value: float = self.initial_balance
        self.peak_value: float = self.initial_balance
        self.current_weights: np.ndarray = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.history: List[float] = []
        self._return_history: List[float] = []  # for running Sharpe

    # ────────────────────────────────────────────────────────────────────
    #  Core API
    # ────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.current_step = 0
        self.portfolio_value = self.initial_balance
        self.peak_value = self.initial_balance
        self.current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.history = [self.portfolio_value]
        self._return_history = []

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Convert logits → valid portfolio weights (softmax)
        target_weights = self._softmax(action)

        # 2. Costs
        turnover = float(np.sum(np.abs(target_weights - self.current_weights))) / 2.0
        total_cost = turnover * (self.transaction_cost + self.to_penalty)

        # 3. Returns for this step
        step_returns = self.returns_arr[self.current_step]
        gross_return = float(np.dot(target_weights, step_returns))
        net_return = gross_return - total_cost

        # 4. Update value
        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.history.append(self.portfolio_value)
        self._return_history.append(net_return)

        # 5. Reward = log-return − drawdown penalty
        log_ret = float(np.log1p(max(net_return, -0.99)))
        drawdown = (self.portfolio_value - self.peak_value) / (self.peak_value + 1e-12)
        reward = log_ret * 1000.0 + self.dd_penalty * drawdown * 1000.0

        # 6. Advance
        self.current_weights = target_weights
        self.current_step += 1

        terminated = self.current_step >= self.n_steps
        truncated = False

        if terminated:
            # Don't advance further; obs is still valid for info purposes
            self.current_step -= 1
            obs = self._get_obs()
            self.current_step += 1
        else:
            obs = self._get_obs()

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": target_weights.copy(),
            "turnover": turnover,
            "net_return": net_return,
        }
        return obs, reward, terminated, truncated, info

    # ────────────────────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        # Rolling returns window (zero-padded at start)
        start = max(0, self.current_step - self.window)
        window_data = self.returns_arr[start : self.current_step]  # (≤window, N)
        if len(window_data) < self.window:
            pad = np.zeros((self.window - len(window_data), self.n_assets), dtype=np.float32)
            window_data = np.vstack([pad, window_data])
        scaled_returns = window_data.flatten() * 100.0  # scale for NN gradients

        # Portfolio stats
        log_value = float(np.log(self.portfolio_value / self.initial_balance + 1e-12))
        current_dd = (self.portfolio_value - self.peak_value) / (self.peak_value + 1e-12)

        if len(self._return_history) >= 5:
            recent = np.array(self._return_history[-52:], dtype=np.float32)
            running_sharpe = float(
                recent.mean() / (recent.std() + 1e-8) * np.sqrt(252)
            )
        else:
            running_sharpe = 0.0

        portfolio_stats = np.array(
            [log_value, running_sharpe, current_dd], dtype=np.float32
        )

        return np.concatenate([scaled_returns, self.current_weights, portfolio_stats])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(x - x.max())
        return (e / e.sum()).astype(np.float32)

    def get_weights_dict(self) -> Dict[str, float]:
        """Return current weights as {ticker: weight} dict."""
        return {t: float(w) for t, w in zip(self.tickers, self.current_weights)}