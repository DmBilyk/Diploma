# tests/ai/test_ml_components.py
"""
Unit tests for app/ai ML components.
Uses mock objects to avoid real model training, filesystem I/O, and DB calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_ASSETS = 4
N_STEPS = 60
WINDOW = 5


def _make_returns(n_steps: int = N_STEPS, n_assets: int = N_ASSETS) -> pd.DataFrame:
    """Synthetic returns matrix – small, deterministic, never hits real data."""
    rng = np.random.default_rng(0)
    data = rng.uniform(-0.02, 0.02, size=(n_steps, n_assets))
    tickers = [f"T{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_steps, freq="B")
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def returns_df() -> pd.DataFrame:
    return _make_returns()


# ─────────────────────────────────────────────────────────────────────────────
#  PortfolioEnv
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioEnv:
    """Tests for the Gymnasium trading environment."""

    from app.ai.environment import PortfolioEnv  # local import keeps test isolation

    def _env(self, returns_df: pd.DataFrame, **kwargs) -> "PortfolioEnv":
        from app.ai.environment import PortfolioEnv
        return PortfolioEnv(returns_df, window=WINDOW, **kwargs)

    def test_observation_space_shape_matches_formula(self, returns_df):
        # obs_dim = window * n_assets + n_assets + 3
        env = self._env(returns_df)
        expected_dim = WINDOW * N_ASSETS + N_ASSETS + 3
        assert env.observation_space.shape == (expected_dim,)

    def test_action_space_shape_equals_n_assets(self, returns_df):
        env = self._env(returns_df)
        assert env.action_space.shape == (N_ASSETS,)

    def test_reset_returns_correct_obs_shape_and_empty_info(self, returns_df):
        env = self._env(returns_df)
        obs, info = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_reset_restores_initial_balance(self, returns_df):
        env = self._env(returns_df, initial_balance=50_000.0)
        env.reset()
        assert env.portfolio_value == pytest.approx(50_000.0)

    def test_reset_sets_equal_initial_weights(self, returns_df):
        env = self._env(returns_df)
        env.reset()
        np.testing.assert_allclose(env.current_weights, np.full(N_ASSETS, 1 / N_ASSETS))

    def test_step_returns_five_tuple(self, returns_df):
        # Gymnasium API: (obs, reward, terminated, truncated, info)
        env = self._env(returns_df)
        env.reset()
        result = env.step(np.zeros(N_ASSETS, dtype=np.float32))
        assert len(result) == 5

    def test_step_obs_shape_consistent_with_observation_space(self, returns_df):
        env = self._env(returns_df)
        env.reset()
        obs, *_ = env.step(np.zeros(N_ASSETS, dtype=np.float32))
        assert obs.shape == env.observation_space.shape

    def test_step_info_contains_required_keys(self, returns_df):
        env = self._env(returns_df)
        env.reset()
        _, _, _, _, info = env.step(np.zeros(N_ASSETS, dtype=np.float32))
        for key in ("portfolio_value", "weights", "turnover", "net_return"):
            assert key in info

    def test_step_weights_sum_to_one(self, returns_df):
        # Softmax output must be a valid probability simplex
        env = self._env(returns_df)
        env.reset()
        action = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert sum(info["weights"]) == pytest.approx(1.0, abs=1e-5)

    def test_episode_terminates_after_all_steps(self, returns_df):
        env = self._env(returns_df)
        env.reset()
        terminated = False
        for _ in range(N_STEPS):
            _, _, terminated, _, _ = env.step(np.zeros(N_ASSETS, dtype=np.float32))
        assert terminated is True

    def test_transaction_cost_reduces_portfolio_value(self, returns_df):
        # With zero-return data, turnover cost must reduce portfolio value
        zero_returns = pd.DataFrame(
            np.zeros((N_STEPS, N_ASSETS)),
            columns=[f"T{i}" for i in range(N_ASSETS)],
        )
        env_no_cost = self._env(zero_returns, transaction_cost=0.0, reward_turnover_penalty=0.0)
        env_with_cost = self._env(zero_returns, transaction_cost=0.01, reward_turnover_penalty=0.0)

        action = np.array([2.0, -2.0, 1.0, -1.0], dtype=np.float32)  # forces turnover

        env_no_cost.reset()
        env_no_cost.step(action)

        env_with_cost.reset()
        env_with_cost.step(action)

        assert env_with_cost.portfolio_value < env_no_cost.portfolio_value

    def test_softmax_output_sums_to_one(self):
        from app.ai.environment import PortfolioEnv
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = PortfolioEnv._softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)
        assert (result > 0).all()

    def test_softmax_is_numerically_stable_with_large_inputs(self):
        from app.ai.environment import PortfolioEnv
        x = np.array([1000.0, 1001.0, 999.0, 1002.0])
        result = PortfolioEnv._softmax(x)
        assert np.isfinite(result).all()
        assert result.sum() == pytest.approx(1.0, abs=1e-6)

    def test_get_weights_dict_maps_tickers_to_weights(self, returns_df):
        env = self._env(returns_df)
        env.reset()
        d = env.get_weights_dict()
        assert set(d.keys()) == set(returns_df.columns)
        assert sum(d.values()) == pytest.approx(1.0, abs=1e-5)

    def test_observation_zero_padded_at_episode_start(self, returns_df):
        # Before window steps have elapsed, leading rows must be zero
        env = self._env(returns_df)
        obs, _ = env.reset()  # step=0, no history yet
        window_part = obs[: WINDOW * N_ASSETS]
        # All elements should be zero because no returns have been observed
        np.testing.assert_array_equal(window_part, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  PortfolioMetricsCallback  (fixed _callback_with_locals)
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioMetricsCallback:
    """Tests for the TensorBoard metrics callback."""

    def _callback_with_locals(self, infos, dones):
        from app.ai.trainer import PortfolioMetricsCallback
        cb = PortfolioMetricsCallback(verbose=0)
        # logger is a read-only property on BaseCallback that delegates to
        # self.model.logger — setting model to a MagicMock makes logger
        # automatically return a trackable MagicMock without touching the property.
        cb.model = MagicMock()
        cb.locals = {"infos": infos, "dones": dones}
        return cb

    def test_continues_without_episode_end(self):
        # No logging should happen mid-episode
        cb = self._callback_with_locals(
            infos=[{"portfolio_value": 101_000}],
            dones=[False],
        )
        result = cb._on_step()
        cb.logger.record.assert_not_called()
        assert result is True

    def test_logs_four_metrics_at_episode_end(self):
        # At done=True with sufficient history, exactly 4 metrics must be logged
        cb = self._callback_with_locals(infos=[], dones=[False])
        cb._episode_values = [100_000, 101_000, 102_000, 103_000]
        cb.locals = {"infos": [{"portfolio_value": 104_000}], "dones": [True]}
        cb._on_step()
        logged_keys = {call.args[0] for call in cb.logger.record.call_args_list}
        assert logged_keys == {
            "portfolio/total_return",
            "portfolio/max_drawdown",
            "portfolio/sharpe_ratio",
            "portfolio/final_value",
        }

    def test_episode_values_cleared_after_logging(self):
        cb = self._callback_with_locals(infos=[], dones=[False])
        cb._episode_values = [100_000, 101_000, 102_000]
        cb.locals = {"infos": [], "dones": [True]}
        cb._on_step()
        assert cb._episode_values == []

    def test_no_log_when_only_one_value_recorded(self):
        # Sharpe / return need at least 2 data points; single value → skip
        cb = self._callback_with_locals(infos=[], dones=[False])
        cb._episode_values = [100_000]
        cb.locals = {"infos": [], "dones": [True]}
        cb._on_step()
        cb.logger.record.assert_not_called()

    def test_zero_std_returns_produces_sharpe_zero(self):
        # Constant portfolio value → std ≈ 0 → Sharpe must default to 0.0
        cb = self._callback_with_locals(infos=[], dones=[False])
        cb._episode_values = [100_000] * 10
        cb.locals = {"infos": [], "dones": [True]}
        cb._on_step()
        sharpe_call = next(
            c for c in cb.logger.record.call_args_list
            if c.args[0] == "portfolio/sharpe_ratio"
        )
        assert sharpe_call.args[1] == pytest.approx(0.0)

# ─────────────────────────────────────────────────────────────────────────────
#  PPOPortfolioTrainer
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOPortfolioTrainer:
    """Tests for trainer init, curriculum logic, and model persistence."""

    @patch("app.ai.trainer.PPO")
    @patch("app.ai.trainer.DummyVecEnv")
    def test_train_without_curriculum_creates_single_ppo_instance(
        self, mock_vec_env_cls, mock_ppo_cls, returns_df, tmp_path
    ):
        # curriculum=False → only one _run_phase → one PPO() construction
        from app.ai.trainer import PPOPortfolioTrainer

        mock_model = MagicMock()
        mock_ppo_cls.return_value = mock_model
        mock_vec_env_cls.return_value = MagicMock()

        trainer = PPOPortfolioTrainer(returns_df, model_dir=tmp_path, seed=0)
        trainer.train(total_timesteps=100, curriculum=False)

        mock_ppo_cls.assert_called_once()

    @patch("app.ai.trainer.PPO")
    @patch("app.ai.trainer.DummyVecEnv")
    def test_train_with_curriculum_calls_learn_twice(
        self, mock_vec_env_cls, mock_ppo_cls, returns_df, tmp_path
    ):
        # curriculum=True → warm-up phase + full phase → model.learn called twice
        from app.ai.trainer import PPOPortfolioTrainer

        mock_model = MagicMock()
        mock_ppo_cls.return_value = mock_model
        mock_vec_env_cls.return_value = MagicMock()

        trainer = PPOPortfolioTrainer(returns_df, model_dir=tmp_path, seed=0)
        trainer.train(
            total_timesteps=200,
            curriculum=True,
            curriculum_warmup_steps=100,
        )

        assert mock_model.learn.call_count == 2

    @patch("app.ai.trainer.PPO")
    @patch("app.ai.trainer.DummyVecEnv")
    def test_train_saves_final_model(
        self, mock_vec_env_cls, mock_ppo_cls, returns_df, tmp_path
    ):
        from app.ai.trainer import PPOPortfolioTrainer

        mock_model = MagicMock()
        mock_ppo_cls.return_value = mock_model
        mock_vec_env_cls.return_value = MagicMock()

        trainer = PPOPortfolioTrainer(returns_df, model_dir=tmp_path, seed=0)
        trainer.train(total_timesteps=100, curriculum=False)

        mock_model.save.assert_called_once()
        saved_path: str = mock_model.save.call_args.args[0]
        assert "final_model" in saved_path

    @patch("app.ai.trainer.PPO")
    @patch("app.ai.trainer.DummyVecEnv")
    def test_load_wraps_ppo_load_and_returns_model(
        self, mock_vec_env_cls, mock_ppo_cls, returns_df, tmp_path
    ):
        # load() must delegate to PPO.load() and cache result in self._model
        from app.ai.trainer import PPOPortfolioTrainer

        mock_loaded = MagicMock()
        mock_ppo_cls.load.return_value = mock_loaded
        mock_vec_env_cls.return_value = MagicMock()

        trainer = PPOPortfolioTrainer(returns_df, model_dir=tmp_path, seed=0)
        model = trainer.load("some/path.zip")

        mock_ppo_cls.load.assert_called_once()
        assert model is mock_loaded
        assert trainer.model is mock_loaded

    def test_random_subwindow_within_bounds(self, returns_df, tmp_path):
        # Subwindow slice must lie inside the original DataFrame
        from app.ai.trainer import PPOPortfolioTrainer

        trainer = PPOPortfolioTrainer(returns_df, model_dir=tmp_path, seed=7)
        sub = trainer._random_subwindow(min_frac=0.3, max_frac=0.6)

        assert len(sub) >= int(N_STEPS * 0.3)
        assert len(sub) <= int(N_STEPS * 0.6) + 1
        assert sub.index[0] in returns_df.index
        assert sub.index[-1] in returns_df.index

    def test_ppo_kwargs_override_merges_with_defaults(self, returns_df, tmp_path):
        from app.ai.trainer import PPOPortfolioTrainer, DEFAULT_PPO_KWARGS

        trainer = PPOPortfolioTrainer(
            returns_df,
            model_dir=tmp_path,
            ppo_kwargs={"learning_rate": 1e-3},
        )
        # Override applied
        assert trainer.ppo_kwargs["learning_rate"] == 1e-3
        # Non-overridden default preserved
        assert trainer.ppo_kwargs["gamma"] == DEFAULT_PPO_KWARGS["gamma"]

    @patch("app.ai.trainer.PPO")
    @patch("app.ai.trainer.DummyVecEnv")
    def test_resume_from_loads_existing_checkpoint(
        self, mock_vec_env_cls, mock_ppo_cls, returns_df, tmp_path
    ):
        # When resume_from points to an existing file, PPO.load() is called instead of PPO()
        from app.ai.trainer import PPOPortfolioTrainer

        fake_zip = tmp_path / "checkpoint.zip"
        fake_zip.touch()

        mock_model = MagicMock()
        mock_ppo_cls.load.return_value = mock_model
        mock_vec_env_cls.return_value = MagicMock()

        trainer = PPOPortfolioTrainer(returns_df, model_dir=tmp_path, seed=0)
        trainer.train(total_timesteps=100, curriculum=False, resume_from=fake_zip)

        mock_ppo_cls.load.assert_called_once()
        mock_ppo_cls.assert_not_called()  # constructor must NOT be invoked


# ─────────────────────────────────────────────────────────────────────────────
#  PPOInference
# ─────────────────────────────────────────────────────────────────────────────

class TestPPOInference:
    """Tests for the inference wrapper: loading, episode execution, weight extraction."""

    def _make_inference(self, tmp_path):
        from app.ai.inference import PPOInference
        fake_zip = tmp_path / "model.zip"
        fake_zip.touch()
        return PPOInference(fake_zip)

    @patch("app.ai.inference.PPO")
    def test_load_delegates_to_ppo_load(self, mock_ppo_cls, tmp_path):
        mock_ppo_cls.load.return_value = MagicMock()
        inf = self._make_inference(tmp_path)
        inf.load()
        mock_ppo_cls.load.assert_called_once()
        assert inf._model is not None

    @patch("app.ai.inference.PPO")
    def test_run_episode_calls_load_lazily_when_model_none(self, mock_ppo_cls, returns_df, tmp_path):
        # If _model is None, run_episode must call load() automatically
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.zeros(N_ASSETS, dtype=np.float32), None
        )
        mock_ppo_cls.load.return_value = mock_model

        inf = self._make_inference(tmp_path)
        assert inf._model is None
        inf.run_episode(returns_df)

        mock_ppo_cls.load.assert_called_once()

    @patch("app.ai.inference.PPO")
    def test_run_episode_returns_correct_history_length(self, mock_ppo_cls, returns_df, tmp_path):
        # history has N_STEPS+1 entries (initial_balance + one per step)
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.zeros(N_ASSETS, dtype=np.float32), None
        )
        mock_ppo_cls.load.return_value = mock_model

        inf = self._make_inference(tmp_path)
        inf.load()
        values, weights = inf.run_episode(returns_df)

        assert len(values) == N_STEPS + 1
        assert len(weights) == N_STEPS

    @patch("app.ai.inference.PPO")
    def test_get_final_weights_sums_to_one(self, mock_ppo_cls, returns_df, tmp_path):
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.zeros(N_ASSETS, dtype=np.float32), None
        )
        mock_ppo_cls.load.return_value = mock_model

        inf = self._make_inference(tmp_path)
        inf.load()
        weights = inf.get_final_weights(returns_df)

        assert abs(sum(weights.values()) - 1.0) < 1e-5

    @patch("app.ai.inference.PPO")
    def test_get_final_weights_respects_min_weight_threshold(self, mock_ppo_cls, returns_df, tmp_path):
        # Skewed action → softmax ≈ [0.64, 0.24, 0.09, 0.03].
        # min_weight=0.05 prunes the last asset so only 3 survive.
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32), None
        )
        mock_ppo_cls.load.return_value = mock_model

        inf = self._make_inference(tmp_path)
        inf.load()
        weights = inf.get_final_weights(returns_df, min_weight=0.05)

        assert all(w >= 0.05 for w in weights.values())
        assert len(weights) < N_ASSETS  # at least one asset was pruned

    @patch("app.ai.inference.PPO")
    def test_get_final_weights_respects_top_n(self, mock_ppo_cls, returns_df, tmp_path):
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.zeros(N_ASSETS, dtype=np.float32), None
        )
        mock_ppo_cls.load.return_value = mock_model

        inf = self._make_inference(tmp_path)
        inf.load()
        weights = inf.get_final_weights(returns_df, top_n=2, min_weight=0.0)

        assert len(weights) <= 2

    @patch("app.ai.inference.PPO")
    def test_get_average_weights_keys_match_tickers(self, mock_ppo_cls, returns_df, tmp_path):
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            np.zeros(N_ASSETS, dtype=np.float32), None
        )
        mock_ppo_cls.load.return_value = mock_model

        inf = self._make_inference(tmp_path)
        inf.load()
        weights = inf.get_average_weights(returns_df, min_weight=0.0)

        assert set(weights.keys()).issubset(set(returns_df.columns))


# ─────────────────────────────────────────────────────────────────────────────
#  load_and_prepare (data_prep)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndPrepare:
    """Tests for the data pipeline: cleaning, splitting, and outlier clipping."""

    def _make_repo(self, prices: pd.DataFrame) -> MagicMock:
        repo = MagicMock()
        repo.get_all_tickers.return_value = list(prices.columns)
        repo.get_price_history.return_value = prices
        return repo

    def _make_prices(self, n_rows: int = 300, n_assets: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        data = 100 + np.cumsum(rng.normal(0, 1, (n_rows, n_assets)), axis=0)
        tickers = [f"A{i}" for i in range(n_assets)]
        dates = pd.date_range("2015-01-01", periods=n_rows, freq="B")
        return pd.DataFrame(data, index=dates, columns=tickers)

    def test_returns_datasplit_with_correct_shapes(self):
        from app.ai.data_prep import load_and_prepare

        prices = self._make_prices()
        repo = self._make_repo(prices)
        split = load_and_prepare(repo, train_frac=0.70, val_frac=0.15)

        total = len(split.train) + len(split.val) + len(split.test)
        # Total rows = prices - 1 (pct_change drops first row)
        assert total == len(prices) - 1

    def test_train_val_test_do_not_overlap(self):
        from app.ai.data_prep import load_and_prepare

        prices = self._make_prices()
        repo = self._make_repo(prices)
        split = load_and_prepare(repo, train_frac=0.70, val_frac=0.15)

        # Chronological order: no index from val/test appears in train
        assert not split.train.index.isin(split.val.index).any()
        assert not split.train.index.isin(split.test.index).any()
        assert not split.val.index.isin(split.test.index).any()

    def test_returns_clipped_within_outlier_bounds(self):
        from app.ai.data_prep import load_and_prepare

        prices = self._make_prices()
        repo = self._make_repo(prices)
        clip = 0.05
        split = load_and_prepare(repo, outlier_clip=clip)

        for df in (split.train, split.val, split.test):
            assert df.values.max() <= clip + 1e-9
            assert df.values.min() >= -clip - 1e-9

    def test_low_coverage_tickers_dropped(self):
        from app.ai.data_prep import load_and_prepare

        prices = self._make_prices(n_assets=4)
        # Introduce 60% NaN in column A3 → below default 95% threshold
        prices["A3"].iloc[: int(len(prices) * 0.60)] = np.nan
        repo = self._make_repo(prices)
        split = load_and_prepare(repo, min_coverage=0.95)

        assert "A3" not in split.tickers

    def test_max_assets_limits_output_columns(self):
        from app.ai.data_prep import load_and_prepare

        prices = self._make_prices(n_assets=10)
        repo = self._make_repo(prices)
        split = load_and_prepare(repo, max_assets=3)

        for df in (split.train, split.val, split.test):
            assert df.shape[1] <= 3

    def test_raises_when_insufficient_rows_after_cleaning(self):
        from app.ai.data_prep import load_and_prepare

        # Only 10 rows of prices → after pct_change and possible NaN drop,
        # way below the 100-row minimum
        prices = self._make_prices(n_rows=10)
        repo = self._make_repo(prices)

        with pytest.raises(ValueError, match="Only .* rows remain"):
            load_and_prepare(repo)

    def test_explicit_ticker_list_respected(self):
        from app.ai.data_prep import load_and_prepare

        prices = self._make_prices(n_assets=5)
        repo = self._make_repo(prices)
        # Restrict to a known subset
        subset = ["A0", "A1"]
        repo.get_price_history.return_value = prices[subset]

        split = load_and_prepare(repo, tickers=subset)
        assert set(split.tickers) == set(subset)