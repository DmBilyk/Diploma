"""
test_robustness.py
==================

Tests for :meth:`BacktestEngine.run_with_seeds` (Phase 4).

Each test uses a deterministic mock repository and a synthetic
``spec_factory`` whose output depends on the seed in a controlled way,
so seed → metric mappings can be asserted exactly.
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
    RobustnessReport,
)


def _synthetic_prices(n_periods: int = 156, tickers=("AAPL", "MSFT", "GOOG")) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2018-01-07", periods=n_periods, freq="W")
    return pd.DataFrame(
        {t: 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.02, size=n_periods))
         for t in tickers},
        index=idx,
    )


def _engine(prices: pd.DataFrame) -> BacktestEngine:
    repo = MagicMock()
    repo.get_price_history.return_value = prices
    return BacktestEngine(
        start_date=str(prices.index[0].date()),
        end_date=str(prices.index[-1].date()),
        initial_capital=100_000.0,
        repo=repo,
    )


def _stochastic_factory(prices: pd.DataFrame):
    """Create a factory that produces seed-dependent random weights."""
    cols = list(prices.columns)

    def factory(seed: int):
        rng = np.random.default_rng(seed)
        w = rng.dirichlet(np.ones(len(cols)))
        weights = {t: float(w[i]) for i, t in enumerate(cols)}
        return [PortfolioSpec(name="StochasticAlgo", weights=weights)]
    return factory


# ═══════════════════════════════════════════════════════════════════════════
class TestSeedDispatch(unittest.TestCase):

    def test_one_report_per_seed(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        seeds = [1, 2, 3, 4, 5]
        rep = engine.run_with_seeds(_stochastic_factory(prices), seeds)
        self.assertEqual(len(rep.seeds), len(seeds))
        self.assertEqual(rep.seeds, seeds)
        self.assertEqual(len(rep.seed_reports), len(seeds))

    def test_factory_receives_int_seed(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        seen = []

        def factory(seed):
            seen.append(seed)
            return _stochastic_factory(prices)(seed)

        engine.run_with_seeds(factory, [10, 20, 30])
        self.assertEqual(seen, [10, 20, 30])
        self.assertTrue(all(isinstance(s, int) for s in seen))

    def test_different_seeds_yield_different_metrics(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        rep = engine.run_with_seeds(_stochastic_factory(prices), [1, 2, 3, 4, 5])
        sharpes = [r.results[0].metrics.sharpe_ratio for r in rep.seed_reports]
        self.assertGreater(len(set(round(s, 6) for s in sharpes)), 1,
                           "Different seeds must produce different portfolios")


# ═══════════════════════════════════════════════════════════════════════════
class TestFailureHandling(unittest.TestCase):

    def test_seed_failure_skipped(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        good = _stochastic_factory(prices)

        def factory(seed):
            if seed == 2:
                raise RuntimeError("simulated failure")
            return good(seed)

        rep = engine.run_with_seeds(factory, [1, 2, 3])
        self.assertEqual(rep.seeds, [1, 3])

    def test_all_seeds_failing_raises(self):
        prices = _synthetic_prices()
        engine = _engine(prices)

        def bad(_):
            raise RuntimeError("never works")

        with self.assertRaises(RuntimeError):
            engine.run_with_seeds(bad, [1, 2, 3])

    def test_empty_seed_list_rejected(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        with self.assertRaises(ValueError):
            engine.run_with_seeds(_stochastic_factory(prices), [])

    def test_factory_returning_no_specs_skipped(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        good = _stochastic_factory(prices)

        def factory(seed):
            if seed == 5:
                return []
            return good(seed)

        rep = engine.run_with_seeds(factory, [1, 5, 9])
        self.assertEqual(rep.seeds, [1, 9])


# ═══════════════════════════════════════════════════════════════════════════
class TestAggregation(unittest.TestCase):

    def test_aggregation_keys_present(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        rep = engine.run_with_seeds(_stochastic_factory(prices), [1, 2, 3, 4, 5])
        summary = rep.aggregated["StochasticAlgo"]["sharpe_ratio"]
        for k in ("mean", "median", "std",
                  "p05", "p25", "p75", "p95",
                  "min", "max", "count", "values"):
            self.assertIn(k, summary)
        self.assertEqual(summary["count"], 5)
        self.assertEqual(len(summary["values"]), 5)

    def test_percentiles_monotonic(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        rep = engine.run_with_seeds(
            _stochastic_factory(prices),
            list(range(1, 11)),
        )
        s = rep.aggregated["StochasticAlgo"]["cagr"]
        self.assertLessEqual(s["min"], s["p05"])
        self.assertLessEqual(s["p05"], s["p25"])
        self.assertLessEqual(s["p25"], s["median"])
        self.assertLessEqual(s["median"], s["p75"])
        self.assertLessEqual(s["p75"], s["p95"])
        self.assertLessEqual(s["p95"], s["max"])

    def test_mean_matches_manual(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        rep = engine.run_with_seeds(_stochastic_factory(prices), [1, 2, 3])
        manual = float(np.mean([
            r.results[0].metrics.cagr for r in rep.seed_reports
        ]))
        self.assertAlmostEqual(
            rep.aggregated["StochasticAlgo"]["cagr"]["mean"],
            manual,
            places=12,
        )

    def test_std_zero_for_single_seed(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        rep = engine.run_with_seeds(_stochastic_factory(prices), [42])
        s = rep.aggregated["StochasticAlgo"]["sharpe_ratio"]
        self.assertEqual(s["count"], 1)
        self.assertEqual(s["std"], 0.0)
        self.assertEqual(s["min"], s["max"])

    def test_multi_algorithm_aggregation(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        cols = list(prices.columns)

        def factory(seed):
            rng = np.random.default_rng(seed)
            w = rng.dirichlet(np.ones(len(cols)))
            algo_a = PortfolioSpec(name="A", weights={t: float(w[i]) for i, t in enumerate(cols)})
            algo_b = PortfolioSpec(name="B", weights={t: 1.0 / len(cols) for t in cols})
            return [algo_a, algo_b]

        rep = engine.run_with_seeds(factory, [1, 2, 3, 4])
        self.assertEqual(set(rep.algorithm_names()), {"A", "B"})
        # Algorithm B is deterministic → std of CAGR ≈ 0
        b_std = rep.aggregated["B"]["cagr"]["std"]
        self.assertAlmostEqual(b_std, 0.0, places=10)
        # Algorithm A is stochastic → std > 0
        a_std = rep.aggregated["A"]["cagr"]["std"]
        self.assertGreater(a_std, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
class TestReproducibility(unittest.TestCase):

    def test_same_seed_yields_same_result(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        factory = _stochastic_factory(prices)
        rep1 = engine.run_with_seeds(factory, [99])
        rep2 = engine.run_with_seeds(factory, [99])
        self.assertAlmostEqual(
            rep1.seed_reports[0].results[0].metrics.cagr,
            rep2.seed_reports[0].results[0].metrics.cagr,
            places=12,
        )


if __name__ == "__main__":
    unittest.main()
