"""
test_walk_forward.py
====================

Tests for :meth:`BacktestEngine.run_walk_forward` (Phase 3).

Uses a mock repository that returns deterministic synthetic prices so the
window math is fully under test control.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.backtesting.backtest_engine import (
    BacktestEngine,
    PortfolioSpec,
    WalkForwardReport,
    WalkForwardWindow,
)


def _synthetic_prices(n_periods: int = 260, tickers=("AAPL", "MSFT", "GOOG")) -> pd.DataFrame:
    """Deterministic random-walk price matrix indexed by weekly dates."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2018-01-07", periods=n_periods, freq="W")
    out = {}
    for i, t in enumerate(tickers):
        rets = rng.normal(0.0008 + i * 0.0002, 0.02, size=n_periods)
        out[t] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(out, index=idx)


def _engine(prices: pd.DataFrame) -> BacktestEngine:
    repo = MagicMock()
    repo.get_price_history.return_value = prices
    return BacktestEngine(
        start_date=str(prices.index[0].date()),
        end_date=str(prices.index[-1].date()),
        initial_capital=100_000.0,
        risk_free_rate=0.02,
        repo=repo,
    )


def _equal_weight_factory(train_prices: pd.DataFrame):
    """Reference spec_factory: equal weight across the universe seen in train."""
    cols = list(train_prices.columns)
    w = 1.0 / len(cols)
    return [PortfolioSpec(name="EqualWeight", weights={t: w for t in cols})]


def _two_algo_factory(train_prices: pd.DataFrame):
    """Two algorithms: equal-weight and inverse-volatility on training data."""
    cols = list(train_prices.columns)
    rets = train_prices.pct_change().dropna()
    inv_vol = 1.0 / (rets.std() + 1e-12)
    inv_vol = inv_vol / inv_vol.sum()
    return [
        PortfolioSpec(name="EqualWeight", weights={t: 1.0 / len(cols) for t in cols}),
        PortfolioSpec(name="InverseVol", weights={t: float(inv_vol[t]) for t in cols}),
    ]


# ═══════════════════════════════════════════════════════════════════════════
class TestWindowMath(unittest.TestCase):

    def test_rolling_window_count(self):
        # 260 obs, train=104, test=52, step=52  → windows at cursors:
        # 0, 52, 104 → train ranges: [0,104), [52,156), [104,208)
        # test ranges: [104,156), [156,208), [208,260) → 3 windows
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _equal_weight_factory,
            universe=list(prices.columns),
            train_periods=104,
            test_periods=52,
            step=52,
        )
        self.assertEqual(len(rep.windows), 3)

    def test_default_step_equals_test_periods(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep_default = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        rep_explicit = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52, step=52,
        )
        self.assertEqual(len(rep_default.windows), len(rep_explicit.windows))

    def test_train_test_disjoint(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        for w in rep.windows:
            self.assertLess(w.train_end, w.test_start,
                            "Train must end strictly before test starts (no leakage)")

    def test_expanding_window_train_grows(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52, expanding=True,
        )
        # All training windows start at the same point
        first_starts = {w.train_start for w in rep.windows}
        self.assertEqual(len(first_starts), 1)
        # And training length grows monotonically
        train_lengths = [(w.train_end - w.train_start).days for w in rep.windows]
        self.assertEqual(train_lengths, sorted(train_lengths))


# ═══════════════════════════════════════════════════════════════════════════
class TestSpecFactoryContract(unittest.TestCase):

    def test_factory_called_once_per_window_with_train_slice(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        seen_train_lengths = []

        def factory(train_prices):
            seen_train_lengths.append(len(train_prices))
            return _equal_weight_factory(train_prices)

        rep = engine.run_walk_forward(
            factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        # Each call gets exactly train_periods rows
        self.assertEqual(len(seen_train_lengths), len(rep.windows))
        self.assertTrue(all(n == 104 for n in seen_train_lengths))

    def test_factory_failure_skips_window(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        call = {"n": 0}

        def factory(train_prices):
            call["n"] += 1
            if call["n"] == 2:
                raise RuntimeError("simulated algorithm failure")
            return _equal_weight_factory(train_prices)

        rep = engine.run_walk_forward(
            factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        # 3 expected windows, one was skipped → 2 succeeded
        self.assertEqual(len(rep.windows), 2)

    def test_all_windows_failing_raises(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)

        def bad_factory(_):
            raise RuntimeError("always fail")

        with self.assertRaises(RuntimeError):
            engine.run_walk_forward(
                bad_factory, universe=list(prices.columns),
                train_periods=104, test_periods=52,
            )


# ═══════════════════════════════════════════════════════════════════════════
class TestAggregation(unittest.TestCase):

    def test_aggregation_produces_one_entry_per_algorithm(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _two_algo_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        self.assertEqual(set(rep.algorithm_names()), {"EqualWeight", "InverseVol"})

    def test_aggregation_summary_stats_present(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _two_algo_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        summary = rep.aggregated["EqualWeight"]["sharpe_ratio"]
        for k in ("mean", "median", "std", "min", "max", "count", "values"):
            self.assertIn(k, summary)
        self.assertEqual(summary["count"], len(rep.windows))
        self.assertEqual(len(summary["values"]), len(rep.windows))

    def test_mean_matches_manual_calc(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        manual = np.mean([
            r.results[0].metrics.cagr for r in rep.window_reports
        ])
        self.assertAlmostEqual(rep.aggregated["EqualWeight"]["cagr"]["mean"], manual)


# ═══════════════════════════════════════════════════════════════════════════
class TestValidation(unittest.TestCase):

    def test_train_too_small_rejected(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        with self.assertRaises(ValueError):
            engine.run_walk_forward(
                _equal_weight_factory, universe=list(prices.columns),
                train_periods=2, test_periods=52,
            )

    def test_test_too_small_rejected(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        with self.assertRaises(ValueError):
            engine.run_walk_forward(
                _equal_weight_factory, universe=list(prices.columns),
                train_periods=104, test_periods=1,
            )

    def test_empty_universe_rejected(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        with self.assertRaises(ValueError):
            engine.run_walk_forward(
                _equal_weight_factory, universe=[],
                train_periods=104, test_periods=52,
            )

    def test_zero_step_rejected(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        with self.assertRaises(ValueError):
            engine.run_walk_forward(
                _equal_weight_factory, universe=list(prices.columns),
                train_periods=104, test_periods=52, step=0,
            )

    def test_insufficient_data_rejected(self):
        prices = _synthetic_prices(80)  # too short for 104+52
        engine = _engine(prices)
        with self.assertRaises(ValueError):
            engine.run_walk_forward(
                _equal_weight_factory, universe=list(prices.columns),
                train_periods=104, test_periods=52,
            )


# ═══════════════════════════════════════════════════════════════════════════
class TestReportShape(unittest.TestCase):

    def test_reports_returned_in_order(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        for prev, nxt in zip(rep.windows, rep.windows[1:]):
            self.assertLess(prev.test_start, nxt.test_start)

    def test_report_dates_match_window(self):
        prices = _synthetic_prices(260)
        engine = _engine(prices)
        rep = engine.run_walk_forward(
            _equal_weight_factory, universe=list(prices.columns),
            train_periods=104, test_periods=52,
        )
        for window, report in zip(rep.windows, rep.window_reports):
            self.assertEqual(report.start_date, str(window.test_start.date()))
            self.assertEqual(report.end_date, str(window.test_end.date()))


if __name__ == "__main__":
    unittest.main()
