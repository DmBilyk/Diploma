"""
app/ai/trainer.py
=================

PPO training pipeline for the portfolio environment.

Features
--------
* **Curriculum training** – warm up on a random sub-window before using the
  full dataset, reducing early overfitting to one market regime.
* **Custom callbacks** – record realised portfolio metrics next to PPO losses.
* **Model persistence** – save checkpoints and the final model for reuse.
* **Explicit configuration** – expose training knobs as keyword arguments with
  stable defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from app.ai.environment import PortfolioEnv

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════


class PortfolioMetricsCallback(BaseCallback):
    """Log realised portfolio metrics to TensorBoard after each episode."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_values: List[float] = []

    def _on_step(self) -> bool:
        # DummyVecEnv exposes per-step environment details through infos.
        infos = self.locals.get("infos", [])
        for info in infos:
            if "portfolio_value" in info:
                self._episode_values.append(info["portfolio_value"])

        # Record episode-level metrics once the rollout has finished.
        dones = self.locals.get("dones", [])
        if any(dones) and len(self._episode_values) > 1:
            values = np.array(self._episode_values)
            returns = np.diff(values) / values[:-1]

            total_return = (values[-1] / values[0]) - 1.0
            max_dd = float(((values / np.maximum.accumulate(values)) - 1.0).min())

            if len(returns) > 1 and returns.std() > 1e-10:
                sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
            else:
                sharpe = 0.0

            self.logger.record("portfolio/total_return", total_return)
            self.logger.record("portfolio/max_drawdown", max_dd)
            self.logger.record("portfolio/sharpe_ratio", sharpe)
            self.logger.record("portfolio/final_value", values[-1])
            self._episode_values = []

        return True


# ═══════════════════════════════════════════════════════════════════════════
#  PPO CONFIG DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PPO_KWARGS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,           # Keeps exploration active during training.
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINER
# ═══════════════════════════════════════════════════════════════════════════


class PPOPortfolioTrainer:
    """Train a PPO agent on a return matrix.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Training return matrix only; test data must stay out-of-sample.
    model_dir : str | Path
        Directory where checkpoints and the final model are saved.
    env_kwargs : dict | None
        Extra kwargs forwarded to :class:`PortfolioEnv`.
    ppo_kwargs : dict | None
        Overrides for PPO hyperparameters (merged into ``DEFAULT_PPO_KWARGS``).
    n_envs : int
        Number of vectorised environments. The wrapper is kept even when this
        is one, because Stable-Baselines expects VecEnv-compatible input.
    tensorboard_log : str | None
        TensorBoard log directory.  ``None`` → no logging.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        returns_df: pd.DataFrame,
        model_dir: str | Path = "models/ppo_portfolio",
        env_kwargs: Optional[Dict[str, Any]] = None,
        ppo_kwargs: Optional[Dict[str, Any]] = None,
        n_envs: int = 1,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.returns_df = returns_df
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.env_kwargs: Dict[str, Any] = env_kwargs or {}
        self.ppo_kwargs: Dict[str, Any] = {**DEFAULT_PPO_KWARGS, **(ppo_kwargs or {})}
        self.n_envs = n_envs
        self.tensorboard_log = tensorboard_log
        self.seed = seed

        self._model: Optional[PPO] = None
        self._vec_env: Optional[DummyVecEnv] = None
        # Reuse one RNG so curriculum windows vary while staying reproducible.
        self._rng = np.random.default_rng(seed)

    # ────────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 500_000,
        checkpoint_every: int = 50_000,
        eval_episodes: int = 5,
        curriculum: bool = True,
        curriculum_warmup_steps: int = 100_000,
        resume_from: Optional[str | Path] = None,
    ) -> PPO:
        """Run the complete training pipeline.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps across all phases.
        checkpoint_every : int
            Save a checkpoint every N steps.
        eval_episodes : int
            Number of episodes used by the eval callback.
        curriculum : bool
            When True, warm up on a shorter random slice before switching to
            the full dataset.
        curriculum_warmup_steps : int
            Number of steps assigned to the warm-up phase.
        resume_from : str | Path | None
            Path to an existing ``.zip`` model to resume training from.
        """
        logger.info(
            "Starting PPO training | assets=%d  steps=%d  timesteps=%d",
            self.returns_df.shape[1],
            self.returns_df.shape[0],
            total_timesteps,
        )

        if curriculum and curriculum_warmup_steps < total_timesteps:
            logger.info("Phase 1/2 – curriculum warm-up (%d steps)", curriculum_warmup_steps)
            self._run_phase(
                returns_df=self._random_subwindow(min_frac=0.3, max_frac=0.6),
                timesteps=curriculum_warmup_steps,
                checkpoint_every=checkpoint_every,
                phase_tag="warmup",
                resume_from=resume_from,
            )
            remaining = total_timesteps - curriculum_warmup_steps
            logger.info("Phase 2/2 – full dataset (%d steps)", remaining)
            self._run_phase(
                returns_df=self.returns_df,
                timesteps=remaining,
                checkpoint_every=checkpoint_every,
                phase_tag="full",
                resume_from=None,  # The phase 1 model is already in memory.
            )
        else:
            self._run_phase(
                returns_df=self.returns_df,
                timesteps=total_timesteps,
                checkpoint_every=checkpoint_every,
                phase_tag="full",
                resume_from=resume_from,
            )

        # Persist the final policy after all phases complete.
        final_path = self.model_dir / "final_model"
        self._model.save(str(final_path))
        logger.info("Final model saved → %s.zip", final_path)

        return self._model

    def load(self, path: str | Path) -> PPO:
        """Load a saved model for inference or continued training."""
        env = self._make_vec_env(self.returns_df)
        self._model = PPO.load(str(path), env=env)
        logger.info("Model loaded from %s", path)
        return self._model

    @property
    def model(self) -> Optional[PPO]:
        return self._model

    # ────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ────────────────────────────────────────────────────────────────────

    def _run_phase(
        self,
        returns_df: pd.DataFrame,
        timesteps: int,
        checkpoint_every: int,
        phase_tag: str,
        resume_from: Optional[str | Path],
    ) -> None:
        vec_env = self._make_vec_env(returns_df)

        checkpoint_cb = CheckpointCallback(
            save_freq=max(checkpoint_every // self.n_envs, 1),
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix=f"ppo_{phase_tag}",
            verbose=0,
        )
        metrics_cb = PortfolioMetricsCallback(verbose=0)
        callbacks = CallbackList([checkpoint_cb, metrics_cb])

        if self._model is None:
            if resume_from is not None and Path(str(resume_from)).exists():
                logger.info("Resuming from %s", resume_from)
                self._model = PPO.load(str(resume_from), env=vec_env)
            else:
                self._model = PPO(
                    policy="MlpPolicy",
                    env=vec_env,
                    verbose=1,
                    tensorboard_log=self.tensorboard_log,
                    seed=self.seed,
                    **self.ppo_kwargs,
                )
        else:
            # Continue the same policy on the next training environment.
            self._model.set_env(vec_env)

        self._model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,  # Preserve the global step counter.
            progress_bar=True,
        )

    def _make_vec_env(self, returns_df: pd.DataFrame) -> DummyVecEnv:
        env_kwargs = self.env_kwargs

        def _make() -> PortfolioEnv:
            return PortfolioEnv(returns_df, **env_kwargs)

        return DummyVecEnv([_make] * self.n_envs)

    def _random_subwindow(self, min_frac: float, max_frac: float) -> pd.DataFrame:
        """Return a random contiguous slice for curriculum warm-up."""
        n = len(self.returns_df)
        length = int(n * self._rng.uniform(min_frac, max_frac))
        start = int(self._rng.integers(0, n - length))
        return self.returns_df.iloc[start : start + length]
