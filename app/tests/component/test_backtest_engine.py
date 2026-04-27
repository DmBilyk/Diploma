"""
Unit tests for app/backtesting/backtest_engine.py

Covers the most important scenarios:
  - Dataclasses (PortfolioSpec, BacktestMetrics, BacktestResult, BacktestReport)
  - BacktestEngine.__init__  (validation of initial_capital, rebalance_every)
  - _validate_spec           (empty, missing tickers, negatives, bad weight sums)
  - _simulate                (buy-and-hold formula, rebalancing behaviour,
                              invalid first-row prices)
  - _compute_metrics         (formula correctness on a known series,
                              drawdown, too-few-points guard)
  - _load_prices             (forward-fill of gaps, drop of tickers without
                              opening price, empty-frame error)
  - run                      (end-to-end with a mocked repository,
                              per-portfolio EW benchmark, deprecated global
                              benchmark, empty portfolios list)
"""
from __future__ import annotations

import unittest
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestMetrics,
    BacktestReport,
    BacktestResult,
    PortfolioSpec,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_price_frame(
    tickers,
    n_obs: int = 60,
    start: str = "2020-01-06",  # a Monday
    freq: str = "W-MON",
    seed: int = 0,
    drift: float = 0.002,
    vol: float = 0.02,
) -> pd.DataFrame:
    """Build a clean weekly price matrix (no NaNs) for the given tickers."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=vol, size=(n_obs, len(tickers)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range(start=start, periods=n_obs, freq=freq)
    return pd.DataFrame(prices, index=idx, columns=list(tickers))


def make_repo(prices_df: pd.DataFrame) -> MagicMock:
    """Return a MagicMock that mimics PortfolioRepository.get_price_history."""
    repo = MagicMock()
    repo.get_price_history.return_value = prices_df.copy()
    return repo


# ═════════════════════════════════════════════════════════════════════════════
# 1. Dataclasses
# ═════════════════════════════════════════════════════════════════════════════
class TestDataclasses(unittest.TestCase):

    def test_portfolio_spec_is_frozen(self):
        spec = PortfolioSpec(name="P", weights={"AAPL": 1.0})
        with self.assertRaises(Exception):  # FrozenInstanceError
            spec.name = "other"

    def test_backtest_metrics_is_frozen(self):
        m = BacktestMetrics(
            total_return=0.1, cagr=0.05, annualised_volatility=0.15,
            max_drawdown=-0.2, sharpe_ratio=0.5, sortino_ratio=0.7,
            start_value=100.0, end_value=110.0,
        )
        with self.assertRaises(Exception):
            m.cagr = 0.99


# ═════════════════════════════════════════════════════════════════════════════
# 2. BacktestEngine.__init__ — input validation
# ═════════════════════════════════════════════════════════════════════════════
class TestEngineInit(unittest.TestCase):

    def test_negative_capital_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine("2020-01-01", "2021-01-01", initial_capital=-1)

    def test_zero_capital_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine("2020-01-01", "2021-01-01", initial_capital=0)

    def test_zero_rebalance_every_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine(
                "2020-01-01", "2021-01-01",
                initial_capital=1000, rebalance_every=0,
            )

    def test_negative_rebalance_every_rejected(self):
        with self.assertRaises(ValueError):
            BacktestEngine(
                "2020-01-01", "2021-01-01",
                initial_capital=1000, rebalance_every=-5,
            )

    def test_valid_init_stores_state(self):
        engine = BacktestEngine(
            "2020-01-01", "2021-01-01",
            initial_capital=50_000, risk_free_rate=0.03,
            rebalance_every=4,
        )
        # Use public behaviour — pass the engine to a minimal run later
        # but here, just assert no exception was raised.
        self.assertIsInstance(engine, BacktestEngine)


# ═════════════════════════════════════════════════════════════════════════════
# 3. _validate_spec
# ═════════════════════════════════════════════════════════════════════════════
class TestValidateSpec(unittest.TestCase):

    def setUp(self):
        self.available = pd.Index(["AAPL", "MSFT", "GOOG"])

    def test_empty_weights_rejected(self):
        spec = PortfolioSpec(name="empty", weights={})
        with self.assertRaises(ValueError):
            BacktestEngine._validate_spec(spec, self.available)

    def test_unknown_ticker_rejected(self):
        spec = PortfolioSpec(name="bad", weights={"AAPL": 0.5, "XYZ": 0.5})
        with self.assertRaisesRegex(ValueError, r"XYZ"):
            BacktestEngine._validate_spec(spec, self.available)

    def test_negative_weight_rejected(self):
        spec = PortfolioSpec(name="short", weights={"AAPL": 1.5, "MSFT": -0.5})
        with self.assertRaisesRegex(ValueError, r"negative"):
            BacktestEngine._validate_spec(spec, self.available)

    def test_weights_sum_far_from_one_rejected(self):
        spec = PortfolioSpec(name="half", weights={"AAPL": 0.3, "MSFT": 0.3})
        with self.assertRaisesRegex(ValueError, r"weights sum"):
            BacktestEngine._validate_spec(spec, self.available)

    def test_weights_within_tolerance_accepted(self):
        # Sum = 1.005 → inside ±0.01 tolerance
        spec = PortfolioSpec(name="ok", weights={"AAPL": 0.5, "MSFT": 0.505})
        # Should not raise
        BacktestEngine._validate_spec(spec, self.available)


# ═════════════════════════════════════════════════════════════════════════════
# 4. _simulate
# ═════════════════════════════════════════════════════════════════════════════
class TestSimulate(unittest.TestCase):

    def test_buy_and_hold_matches_manual_formula(self):
        # Deterministic, hand-checkable: two assets, two time steps.
        idx = pd.date_range("2020-01-06", periods=3, freq="W-MON")
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 120.0],
             "B": [50.0, 55.0, 45.0]},
            index=idx,
        )
        weights = {"A": 0.6, "B": 0.4}
        capital = 10_000.0

        values = BacktestEngine._simulate(weights, prices, capital)

        # Manual calculation
        shares_a = capital * 0.6 / 100.0   # 60
        shares_b = capital * 0.4 / 50.0    # 80
        expected = np.array([
            shares_a * 100.0 + shares_b * 50.0,
            shares_a * 110.0 + shares_b * 55.0,
            shares_a * 120.0 + shares_b * 45.0,
        ])
        np.testing.assert_allclose(values.values, expected)
        # Initial value equals starting capital
        self.assertAlmostEqual(values.iloc[0], capital, places=6)
        # Index is preserved
        self.assertTrue(values.index.equals(idx))

    def test_buy_and_hold_constant_prices_stays_constant(self):
        prices = pd.DataFrame(
            {"A": [100.0] * 10, "B": [50.0] * 10},
            index=pd.date_range("2020-01-06", periods=10, freq="W-MON"),
        )
        values = BacktestEngine._simulate(
            {"A": 0.5, "B": 0.5}, prices, 1000.0,
        )
        np.testing.assert_allclose(values.values, 1000.0)

    def test_simulate_raises_on_invalid_opening_price(self):
        prices = pd.DataFrame(
            {"A": [np.nan, 110.0, 120.0],
             "B": [50.0, 55.0, 45.0]},
            index=pd.date_range("2020-01-06", periods=3, freq="W-MON"),
        )
        with self.assertRaisesRegex(ValueError, r"opening price"):
            BacktestEngine._simulate({"A": 0.5, "B": 0.5}, prices, 1000.0)

    def test_simulate_raises_on_zero_opening_price(self):
        prices = pd.DataFrame(
            {"A": [0.0, 110.0, 120.0],
             "B": [50.0, 55.0, 45.0]},
            index=pd.date_range("2020-01-06", periods=3, freq="W-MON"),
        )
        with self.assertRaisesRegex(ValueError, r"opening price"):
            BacktestEngine._simulate({"A": 0.5, "B": 0.5}, prices, 1000.0)

    def test_rebalancing_matches_buy_and_hold_when_prices_constant(self):
        # If prices don't move, rebalancing shouldn't change anything.
        prices = pd.DataFrame(
            {"A": [100.0] * 20, "B": [50.0] * 20},
            index=pd.date_range("2020-01-06", periods=20, freq="W-MON"),
        )
        v_hold = BacktestEngine._simulate(
            {"A": 0.5, "B": 0.5}, prices, 1000.0, rebalance_every=None,
        )
        v_rebal = BacktestEngine._simulate(
            {"A": 0.5, "B": 0.5}, prices, 1000.0, rebalance_every=4,
        )
        np.testing.assert_allclose(v_hold.values, v_rebal.values)

    def test_rebalancing_starts_at_initial_capital(self):
        prices = make_price_frame(["A", "B", "C"], n_obs=20, seed=3)
        values = BacktestEngine._simulate(
            {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, prices, 5000.0,
            rebalance_every=4,
        )
        self.assertAlmostEqual(values.iloc[0], 5000.0, places=6)
        self.assertEqual(len(values), len(prices))


# ═════════════════════════════════════════════════════════════════════════════
# 5. _compute_metrics
# ═════════════════════════════════════════════════════════════════════════════
class TestComputeMetrics(unittest.TestCase):

    def _engine(self, rf: float = 0.0) -> BacktestEngine:
        return BacktestEngine(
            "2020-01-06", "2021-01-06",
            initial_capital=1000.0, risk_free_rate=rf,
        )

    def test_too_few_points_raises(self):
        # _MIN_SERIES_LENGTH = 4 → length 3 must fail.
        values = pd.Series(
            [1000.0, 1010.0, 1020.0],
            index=pd.date_range("2020-01-06", periods=3, freq="W-MON"),
        )
        with self.assertRaisesRegex(ValueError, r"Too few"):
            self._engine()._compute_metrics(values)

    def test_total_return_and_endpoints(self):
        values = pd.Series(
            [1000.0, 1050.0, 1100.0, 1200.0, 1300.0],
            index=pd.date_range("2020-01-06", periods=5, freq="W-MON"),
        )
        m = self._engine()._compute_metrics(values)
        self.assertAlmostEqual(m.start_value, 1000.0)
        self.assertAlmostEqual(m.end_value, 1300.0)
        self.assertAlmostEqual(m.total_return, 0.3, places=10)

    def test_max_drawdown_matches_manual(self):
        # Peak = 1200, trough after = 900  →  drawdown = 900/1200 − 1 = -0.25
        values = pd.Series(
            [1000.0, 1200.0, 1100.0, 900.0, 1000.0, 1050.0],
            index=pd.date_range("2020-01-06", periods=6, freq="W-MON"),
        )
        m = self._engine()._compute_metrics(values)
        self.assertAlmostEqual(m.max_drawdown, -0.25, places=10)

    def test_constant_series_has_zero_return_and_drawdown(self):
        values = pd.Series(
            [1000.0] * 10,
            index=pd.date_range("2020-01-06", periods=10, freq="W-MON"),
        )
        m = self._engine()._compute_metrics(values)
        self.assertAlmostEqual(m.total_return, 0.0)
        self.assertAlmostEqual(m.cagr, 0.0)
        self.assertAlmostEqual(m.annualised_volatility, 0.0)
        self.assertAlmostEqual(m.max_drawdown, 0.0)

    def test_sortino_is_nan_when_no_downside(self):
        # Strictly increasing → no negative excess returns (rf=0).
        values = pd.Series(
            [100.0 * (1.01 ** i) for i in range(10)],
            index=pd.date_range("2020-01-06", periods=10, freq="W-MON"),
        )
        m = self._engine(rf=0.0)._compute_metrics(values)
        self.assertTrue(np.isnan(m.sortino_ratio))


# ═════════════════════════════════════════════════════════════════════════════
# 6. _load_prices
# ═════════════════════════════════════════════════════════════════════════════
class TestLoadPrices(unittest.TestCase):

    def test_forward_fill_gaps(self):
        idx = pd.date_range("2020-01-06", periods=5, freq="W-MON")
        df = pd.DataFrame(
            {"A": [100.0, np.nan, 110.0, np.nan, 120.0],
             "B": [50.0, 55.0, np.nan, 60.0, 65.0]},
            index=idx,
        )
        repo = make_repo(df)
        engine = BacktestEngine("2020-01-06", "2020-02-06", repo=repo)
        result = engine._load_prices(["A", "B"])

        # No NaNs remain
        self.assertFalse(result.isna().any().any())
        # Filled values equal the last valid observation
        self.assertEqual(result.loc[idx[1], "A"], 100.0)
        self.assertEqual(result.loc[idx[2], "B"], 55.0)

    def test_ticker_without_opening_price_is_dropped(self):
        idx = pd.date_range("2020-01-06", periods=5, freq="W-MON")
        df = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0, 103.0, 104.0],
             "B": [np.nan, np.nan, 50.0, 51.0, 52.0]},
            index=idx,
        )
        repo = make_repo(df)
        engine = BacktestEngine("2020-01-06", "2020-02-06", repo=repo)
        result = engine._load_prices(["A", "B"])

        # B had no opening price → must be dropped
        self.assertIn("A", result.columns)
        self.assertNotIn("B", result.columns)

    def test_empty_frame_raises(self):
        repo = make_repo(pd.DataFrame())
        engine = BacktestEngine("2020-01-06", "2020-02-06", repo=repo)
        with self.assertRaises(RuntimeError):
            engine._load_prices(["A"])

    def test_index_is_sorted_and_datetime(self):
        # Deliberately out of order
        idx = pd.to_datetime(["2020-02-03", "2020-01-06", "2020-01-20"])
        df = pd.DataFrame({"A": [110.0, 100.0, 105.0]}, index=idx)
        repo = make_repo(df)
        engine = BacktestEngine("2020-01-06", "2020-02-06", repo=repo)
        result = engine._load_prices(["A"])

        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertIsInstance(result.index, pd.DatetimeIndex)


# ═════════════════════════════════════════════════════════════════════════════
# 7. run — end-to-end with mocked repository
# ═════════════════════════════════════════════════════════════════════════════
class TestRunEndToEnd(unittest.TestCase):

    def setUp(self):
        self.prices = make_price_frame(["AAPL", "MSFT", "GOOG"], n_obs=60, seed=42)
        self.repo = make_repo(self.prices)
        self.engine = BacktestEngine(
            start_date="2020-01-06",
            end_date="2021-03-01",
            initial_capital=100_000.0,
            risk_free_rate=0.02,
            repo=self.repo,
        )

    def test_empty_portfolio_list_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.run([])

    def test_run_returns_report_with_expected_structure(self):
        spec = PortfolioSpec(
            name="Two-Asset",
            weights={"AAPL": 0.6, "MSFT": 0.4},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            report = self.engine.run([spec])

        self.assertIsInstance(report, BacktestReport)
        self.assertEqual(len(report.results), 1)
        self.assertEqual(report.start_date, "2020-01-06")
        self.assertEqual(report.end_date, "2021-03-01")
        self.assertAlmostEqual(report.initial_capital, 100_000.0)

        r = report.results[0]
        self.assertIsInstance(r, BacktestResult)
        self.assertIsInstance(r.metrics, BacktestMetrics)
        # Portfolio starts at initial capital
        self.assertAlmostEqual(r.portfolio_values.iloc[0], 100_000.0, places=4)
        # Metrics are finite on a clean price series
        self.assertTrue(np.isfinite(r.metrics.total_return))
        self.assertTrue(np.isfinite(r.metrics.cagr))
        self.assertTrue(np.isfinite(r.metrics.annualised_volatility))

    def test_per_portfolio_ew_benchmark_present(self):
        spec = PortfolioSpec(
            name="Two-Asset",
            weights={"AAPL": 0.6, "MSFT": 0.4},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            report = self.engine.run([spec])

        bench = report.results[0].benchmark
        self.assertIsInstance(bench, BacktestResult)
        # EW benchmark uses exactly the same tickers as the portfolio
        self.assertEqual(set(bench.spec.weights.keys()), {"AAPL", "MSFT"})
        # With 2 tickers, each weight must be 0.5
        for w in bench.spec.weights.values():
            self.assertAlmostEqual(w, 0.5)
        # Benchmark also starts at initial capital
        self.assertAlmostEqual(bench.portfolio_values.iloc[0], 100_000.0, places=4)

    def test_report_global_benchmark_is_none(self):
        """Global benchmark was removed; per-portfolio benchmarks live on
        ``BacktestResult.benchmark`` and ``BacktestResult.benchmarks``.

        ``engine.run`` MUST NOT populate ``BacktestReport.benchmark`` and
        MUST NOT emit any DeprecationWarning during a normal run.
        """
        spec = PortfolioSpec(name="P", weights={"AAPL": 1.0})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = self.engine.run([spec])

        # New contract: global benchmark slot is unused.
        self.assertIsNone(report.benchmark)

        # Per-portfolio benchmark IS populated on the result (Phase 5).
        self.assertIsInstance(report.results[0].benchmark, BacktestResult)
        self.assertGreaterEqual(len(report.results[0].benchmarks), 1)

        # Normal `run()` must be quiet — deprecation noise would clutter
        # callers' logs and contradict the documented "stable" status of
        # the per-portfolio benchmark API.
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(len(dep_warnings), 0,
                         f"Unexpected DeprecationWarning(s): "
                         f"{[str(w.message) for w in dep_warnings]}")

    def test_legacy_build_benchmark_still_emits_deprecation_warning(self):
        """``_build_benchmark`` is kept for backward compatibility but must
        emit a DeprecationWarning when called directly, so any external
        legacy code paths continue to be flagged."""
        prices = self.engine._load_prices(["AAPL", "MSFT"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bench = self.engine._build_benchmark(prices)

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertGreaterEqual(len(dep_warnings), 1)
        self.assertIsInstance(bench, BacktestResult)

    def test_repository_called_with_union_of_tickers(self):
        p1 = PortfolioSpec(name="A", weights={"AAPL": 1.0})
        p2 = PortfolioSpec(name="B", weights={"MSFT": 0.5, "GOOG": 0.5})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.engine.run([p1, p2])

        # Repo must be asked for the union of tickers, sorted.
        self.repo.get_price_history.assert_called_once()
        called_tickers = self.repo.get_price_history.call_args[0][0]
        self.assertEqual(sorted(called_tickers), ["AAPL", "GOOG", "MSFT"])


if __name__ == "__main__":
    unittest.main(verbosity=2)