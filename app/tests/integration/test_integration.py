"""
test_integration.py — end-to-end integration tests for the main scenarios.

These tests exercise multiple subsystems together (repository ↔ optimiser ↔
backtester ↔ plugins ↔ facade) using a temporary SQLite database seeded with
synthetic price data. They avoid heavy dependencies (PPO / LSTM / Yahoo
Finance) and run fast.

Scenarios covered:
  1. Repository round-trip       — assets + quotes saved and read back.
  2. Plugin discovery + run      — built-in plugins via PluginManager.
  3. Backtest engine (synthetic) — PortfolioSpec → metrics.
  4. Hybrid evolutionary run     — small population/generations on temp DB.
  5. Facade run_and_backtest     — full optimise → backtest pipeline.
  6. Facade plugin pipeline      — run_and_backtest with method='plugin'.
  7. Experiment persistence      — save_experiment writes a row.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from app.backtesting.backtest_engine import (
    BacktestEngine,
    PortfolioSpec,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data helpers
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]
N_ROWS = 250  # ~5 years of weekly data


def _make_synthetic_prices(
    tickers: List[str] = TICKERS,
    n_rows: int = N_ROWS,
    start: str = "2018-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Produce realistic-looking price series via geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_rows, freq="W-MON")
    data = {}
    for i, t in enumerate(tickers):
        # Each ticker gets a different drift + vol so the optimiser sees
        # meaningful differences between assets.
        drift = 0.0005 + 0.0008 * i
        vol = 0.02 + 0.005 * i
        rets = rng.normal(drift, vol, n_rows)
        prices = 100.0 * np.exp(np.cumsum(rets))
        data[t] = prices
    return pd.DataFrame(data, index=dates)


def _prices_to_quote_df(series: pd.Series) -> pd.DataFrame:
    """Convert a single-ticker price series into the OHLCV frame the
    repository's ``save_quotes_bulk`` expects."""
    return pd.DataFrame(
        {
            "Open": series.values,
            "High": series.values * 1.01,
            "Low": series.values * 0.99,
            "Close": series.values,
            "Adj Close": series.values,
            "Volume": np.full(len(series), 1_000_000, dtype=np.int64),
        },
        index=series.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures: temp SQLite DB, seeded repository, PortfolioCore
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Spin up a fresh SQLite DB for the test and reset the global engine.

    The data layer caches an engine in module globals; we override the
    DB_PATH and clear those globals so ``init_db()`` re-creates everything
    against the temp file.
    """
    from app.data import models as models_mod

    db_file = tmp_path / "portfolio_test.db"
    monkeypatch.setattr(models_mod, "DB_FILE", db_file)
    monkeypatch.setattr(models_mod, "DB_PATH", f"sqlite:///{db_file}")
    monkeypatch.setattr(models_mod, "_engine", None)
    monkeypatch.setattr(models_mod, "_Session", None)

    yield db_file

    # Reset globals after the test so the cached engine doesn't leak
    # into the next test (each gets its own temp file).
    monkeypatch.setattr(models_mod, "_engine", None)
    monkeypatch.setattr(models_mod, "_Session", None)


@pytest.fixture
def seeded_repo(temp_db):
    """Repository pointing at a temp DB pre-loaded with synthetic data."""
    from app.data.repository import PortfolioRepository

    prices = _make_synthetic_prices()
    repo = PortfolioRepository()
    for ticker in prices.columns:
        repo.add_asset(ticker, name=ticker, sector="Test")
        repo.save_quotes_bulk(ticker, _prices_to_quote_df(prices[ticker]))
    return repo


@pytest.fixture
def core(seeded_repo):
    """PortfolioCore wired to the seeded temp repository."""
    from app.core.core import PortfolioCore

    return PortfolioCore(repo=seeded_repo)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Repository round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestRepositoryRoundTrip:
    def test_tickers_persisted(self, seeded_repo):
        stored = sorted(seeded_repo.get_all_tickers())
        assert stored == sorted(TICKERS)

    def test_assets_metadata(self, seeded_repo):
        assets = seeded_repo.get_all_assets()
        assert len(assets) == len(TICKERS)
        assert all(a["sector"] == "Test" for a in assets)

    def test_price_history_pivot(self, seeded_repo):
        df = seeded_repo.get_price_history(TICKERS)
        assert not df.empty
        assert sorted(df.columns) == sorted(TICKERS)
        assert len(df) == N_ROWS
        # All adj_close values strictly positive.
        assert (df.values > 0).all()

    def test_price_history_date_filter(self, seeded_repo):
        df_full = seeded_repo.get_price_history(TICKERS)
        cutoff = df_full.index[100].date().isoformat()
        df_filtered = seeded_repo.get_price_history(TICKERS, start_date=cutoff)
        assert len(df_filtered) < len(df_full)
        assert df_filtered.index[0] >= pd.Timestamp(cutoff)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Plugin discovery + execution via the facade
# ─────────────────────────────────────────────────────────────────────────────


class TestPluginPipeline:
    def test_plugins_discovered(self, core):
        plugins = core.get_plugins()
        # The repo ships with at least these two built-in plugins.
        assert "InverseVolatilityOptimizer" in plugins
        assert "EqualWeightOptimizer" in plugins

    def test_equal_weight_plugin(self, core):
        weights = core.run_plugin_optimization(
            plugin_name="EqualWeightOptimizer",
            tickers=TICKERS,
        )
        assert set(weights.keys()) == set(TICKERS)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # All weights equal.
        unique_vals = {round(v, 8) for v in weights.values()}
        assert len(unique_vals) == 1

    def test_inverse_volatility_plugin(self, core):
        weights = core.run_plugin_optimization(
            plugin_name="InverseVolatilityOptimizer",
            tickers=TICKERS,
            config={"max_cardinality": 3},
        )
        assert 0 < len(weights) <= 3
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # All weights non-negative.
        assert all(w >= 0 for w in weights.values())

    def test_unknown_plugin_raises(self, core):
        with pytest.raises(KeyError):
            core.run_plugin_optimization(plugin_name="DoesNotExist")


# ─────────────────────────────────────────────────────────────────────────────
#  3. Backtest engine (synthetic, no DB)
# ─────────────────────────────────────────────────────────────────────────────


class TestBacktestEngine:
    def test_synthetic_backtest_metrics(self):
        prices = _make_synthetic_prices(tickers=["A", "B"], n_rows=60)
        # Inject prices via a fake repo to avoid DB.
        class _FakeRepo:
            def get_price_history(self, tickers, start_date=None, end_date=None):
                return prices[tickers]

        engine = BacktestEngine(
            start_date=str(prices.index[0].date()),
            end_date=str(prices.index[-1].date()),
            initial_capital=100_000.0,
            repo=_FakeRepo(),
            rebalance_every=4,
        )
        spec = PortfolioSpec(name="50/50", weights={"A": 0.5, "B": 0.5})
        report = engine.run([spec])

        assert len(report.results) == 1
        result = report.results[0]
        m = result.metrics

        # Sanity: every numerical metric is finite.
        for field in (
            "total_return", "cagr", "annualised_volatility",
            "max_drawdown", "sharpe_ratio",
        ):
            v = getattr(m, field)
            assert np.isfinite(v), f"{field} should be finite, got {v}"

        assert m.start_value == pytest.approx(100_000.0, rel=1e-6)
        assert m.max_drawdown <= 0.0
        # Equal-weight benchmark attached.
        assert result.benchmark is not None


# ─────────────────────────────────────────────────────────────────────────────
#  4. Hybrid evolutionary optimisation against temp DB
# ─────────────────────────────────────────────────────────────────────────────


class TestEvolutionaryOptimization:
    def test_run_optimization_returns_valid_weights(self, core):
        result = core.run_optimization(
            tickers=TICKERS,
            pop_size=20,
            n_generations=5,
            max_cardinality=3,
            seed=123,
        )
        assert result.weights, "Optimizer returned empty weights"
        # Weights sum to ≈1 and are non-negative.
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=0.05)
        assert all(w >= -1e-9 for w in result.weights.values())
        # Cardinality constraint respected.
        active = [w for w in result.weights.values() if w > 1e-6]
        assert len(active) <= 3
        # All chosen tickers come from the seeded universe.
        assert set(result.weights.keys()).issubset(set(TICKERS))


# ─────────────────────────────────────────────────────────────────────────────
#  5. Facade compound flow: optimise → backtest
# ─────────────────────────────────────────────────────────────────────────────


class TestRunAndBacktest:
    def test_evo_pipeline(self, core, seeded_repo):
        prices = seeded_repo.get_price_history(TICKERS)
        train_end = prices.index[150].date().isoformat()
        end_date = prices.index[-1].date().isoformat()
        start_date = prices.index[0].date().isoformat()

        opt_result, report = core.run_and_backtest(
            method="evo",
            tickers=TICKERS,
            start_date=start_date,
            train_end=train_end,
            end_date=end_date,
            pop_size=20,
            n_generations=5,
            max_cardinality=3,
            seed=7,
            initial_capital=100_000.0,
            rebalance_every=4,
        )

        assert opt_result.weights
        assert len(report.results) == 1
        bt = report.results[0]
        assert bt.spec.name.startswith("EVO")
        assert np.isfinite(bt.metrics.total_return)
        assert bt.metrics.start_value == pytest.approx(100_000.0, rel=1e-6)

    def test_plugin_pipeline(self, core, seeded_repo):
        prices = seeded_repo.get_price_history(TICKERS)
        train_end = prices.index[150].date().isoformat()
        end_date = prices.index[-1].date().isoformat()
        start_date = prices.index[0].date().isoformat()

        opt_result, report = core.run_and_backtest(
            method="plugin",
            plugin_name="EqualWeightOptimizer",
            tickers=TICKERS,
            start_date=start_date,
            train_end=train_end,
            end_date=end_date,
            initial_capital=50_000.0,
            rebalance_every=4,
        )

        assert opt_result.weights
        assert sum(opt_result.weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert report.results[0].metrics.start_value == pytest.approx(
            50_000.0, rel=1e-6
        )

    def test_missing_dates_raises(self, core):
        # Neither train_end nor backtest_start/end provided → must reject.
        with pytest.raises(ValueError):
            core.run_and_backtest(
                method="evo",
                tickers=TICKERS,
                start_date="2018-01-01",
                end_date="2022-01-01",
                pop_size=10,
                n_generations=2,
            )

    def test_unknown_method_raises(self, core):
        with pytest.raises(ValueError):
            core.run_and_backtest(
                method="bogus",
                tickers=TICKERS,
                start_date="2018-01-01",
                train_end="2020-01-01",
                end_date="2022-01-01",
            )


# ─────────────────────────────────────────────────────────────────────────────
#  6. Experiment persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestExperimentPersistence:
    def test_save_experiment_writes_row(self, core, temp_db):
        from sqlalchemy import select
        from app.data.models import Experiment, init_db

        core.save_experiment(
            name="integration-run",
            algorithm="evo",
            parameters={"pop_size": 20, "n_generations": 5},
            metrics={"sharpe_ratio": 1.42, "total_return": 0.18},
        )

        Session = init_db()
        with Session() as session:
            rows = session.execute(select(Experiment)).scalars().all()
            assert len(rows) == 1
            assert rows[0].name == "integration-run"
            assert rows[0].algorithm == "evo"
            assert rows[0].parameters["pop_size"] == 20
            assert rows[0].metrics["sharpe_ratio"] == 1.42
