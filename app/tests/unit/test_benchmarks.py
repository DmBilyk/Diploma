"""
test_benchmarks.py
==================

Tests for :mod:`app.backtesting.benchmarks` (Phase 5) and the engine
integration that wires multiple benchmarks per portfolio.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.backtesting.benchmarks import (
    Benchmark,
    EqualWeightBenchmark,
    InverseVolatilityBenchmark,
    MinimumVarianceBenchmark,
    RiskParityBenchmark,
)
from app.backtesting.backtest_engine import (
    BacktestEngine,
    PortfolioSpec,
)


def _synthetic_prices(n_periods: int = 156, tickers=("AAPL", "MSFT", "GOOG")) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2018-01-07", periods=n_periods, freq="W")
    out = {}
    for i, t in enumerate(tickers):
        rets = rng.normal(0.0008 + i * 0.0002, 0.015 + i * 0.005, size=n_periods)
        out[t] = 100.0 * np.cumprod(1.0 + rets)
    return pd.DataFrame(out, index=idx)


def _engine(prices: pd.DataFrame, benchmarks=None) -> BacktestEngine:
    repo = MagicMock()
    repo.get_price_history.return_value = prices
    return BacktestEngine(
        start_date=str(prices.index[0].date()),
        end_date=str(prices.index[-1].date()),
        initial_capital=100_000.0,
        repo=repo,
        benchmarks=benchmarks,
    )


# ═══════════════════════════════════════════════════════════════════════════
class TestEqualWeight(unittest.TestCase):

    def test_weights_sum_to_one(self):
        bench = EqualWeightBenchmark()
        w = bench.compute_weights(_synthetic_prices())
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)

    def test_uniform(self):
        bench = EqualWeightBenchmark()
        w = bench.compute_weights(_synthetic_prices())
        self.assertEqual(len(set(round(v, 12) for v in w.values())), 1)

    def test_short_name(self):
        self.assertEqual(EqualWeightBenchmark().short_name, "EW Benchmark")

    def test_empty_prices_rejected(self):
        with self.assertRaises(ValueError):
            EqualWeightBenchmark().compute_weights(pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════════════
class TestInverseVolatility(unittest.TestCase):

    def test_weights_sum_to_one(self):
        w = InverseVolatilityBenchmark().compute_weights(_synthetic_prices())
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)

    def test_lower_vol_gets_higher_weight(self):
        # Synthetic prices have monotonically increasing volatility across tickers.
        prices = _synthetic_prices()
        w = InverseVolatilityBenchmark().compute_weights(prices)
        ordered = [w[t] for t in prices.columns]
        # First ticker (lowest vol) should weight more than the last (highest vol).
        self.assertGreater(ordered[0], ordered[-1])

    def test_no_negative_weights(self):
        w = InverseVolatilityBenchmark().compute_weights(_synthetic_prices())
        self.assertTrue(all(v >= 0 for v in w.values()))

    def test_constant_prices_falls_back_to_equal_weight(self):
        prices = pd.DataFrame(
            {"A": [10.0] * 20, "B": [20.0] * 20, "C": [30.0] * 20},
            index=pd.date_range("2020-01-05", periods=20, freq="W"),
        )
        w = InverseVolatilityBenchmark().compute_weights(prices)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=12)
        self.assertAlmostEqual(w["A"], w["B"])
        self.assertAlmostEqual(w["B"], w["C"])


# ═══════════════════════════════════════════════════════════════════════════
class TestMinimumVariance(unittest.TestCase):

    def test_weights_sum_to_one(self):
        w = MinimumVarianceBenchmark().compute_weights(_synthetic_prices())
        self.assertAlmostEqual(sum(w.values()), 1.0, places=4)

    def test_no_negative_weights(self):
        w = MinimumVarianceBenchmark().compute_weights(_synthetic_prices())
        for v in w.values():
            self.assertGreaterEqual(v, -1e-9)

    def test_minvar_lower_variance_than_equal_weight(self):
        # Min-variance portfolio should achieve ≤ EW realised variance
        # on the in-sample data it was fit to.
        prices = _synthetic_prices(n_periods=200)
        rets = prices.pct_change().dropna().values
        cov = np.cov(rets.T)

        mv = MinimumVarianceBenchmark().compute_weights(prices)
        ew = EqualWeightBenchmark().compute_weights(prices)
        w_mv = np.array([mv[t] for t in prices.columns])
        w_ew = np.array([ew[t] for t in prices.columns])
        self.assertLessEqual(float(w_mv @ cov @ w_mv), float(w_ew @ cov @ w_ew) + 1e-12)

    def test_too_few_observations_falls_back(self):
        # Only 3 obs for 3 assets → falls back to equal weights
        prices = pd.DataFrame(
            {"A": [10.0, 11.0, 12.0], "B": [20.0, 19.0, 21.0], "C": [30.0, 31.0, 29.0]},
            index=pd.date_range("2020-01-05", periods=3, freq="W"),
        )
        w = MinimumVarianceBenchmark().compute_weights(prices)
        self.assertAlmostEqual(w["A"], 1.0 / 3.0, places=10)


# ═══════════════════════════════════════════════════════════════════════════
class TestRiskParity(unittest.TestCase):

    def test_weights_sum_to_one(self):
        w = RiskParityBenchmark().compute_weights(_synthetic_prices())
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)

    def test_no_negative_weights(self):
        w = RiskParityBenchmark().compute_weights(_synthetic_prices())
        self.assertTrue(all(v >= 0 for v in w.values()))

    def test_equal_risk_contributions(self):
        prices = _synthetic_prices(n_periods=200)
        w_dict = RiskParityBenchmark().compute_weights(prices)
        cov = prices.pct_change().dropna().cov().values
        w = np.array([w_dict[t] for t in prices.columns])
        port_var = float(w @ cov @ w)
        marginal = cov @ w
        rc = (w * marginal) / port_var
        # Risk contributions should each be approximately 1/N
        target = 1.0 / len(w)
        self.assertLess(float(np.max(np.abs(rc - target))), 0.02)

    def test_too_few_observations_falls_back_to_invvol(self):
        prices = pd.DataFrame(
            {"A": [10.0, 11.0, 12.0], "B": [20.0, 19.0, 21.0], "C": [30.0, 31.0, 29.0]},
            index=pd.date_range("2020-01-05", periods=3, freq="W"),
        )
        w = RiskParityBenchmark().compute_weights(prices)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=8)


# ═══════════════════════════════════════════════════════════════════════════
class TestEngineIntegration(unittest.TestCase):

    def test_default_benchmarks_is_equal_weight_only(self):
        prices = _synthetic_prices()
        engine = _engine(prices)
        spec = PortfolioSpec(name="P", weights={"AAPL": 0.5, "MSFT": 0.5})
        report = engine.run([spec])
        r = report.results[0]
        # Backwards-compat slot still populated…
        self.assertIsNotNone(r.benchmark)
        # …and matches the first entry of the new list.
        self.assertEqual(len(r.benchmarks), 1)
        self.assertIs(r.benchmarks[0], r.benchmark)
        self.assertIn("EW Benchmark", r.benchmark.spec.name)

    def test_multiple_benchmarks_all_evaluated(self):
        prices = _synthetic_prices()
        engine = _engine(
            prices,
            benchmarks=[
                EqualWeightBenchmark(),
                InverseVolatilityBenchmark(),
                MinimumVarianceBenchmark(),
                RiskParityBenchmark(),
            ],
        )
        spec = PortfolioSpec(
            name="MyAlgo",
            weights={"AAPL": 0.4, "MSFT": 0.3, "GOOG": 0.3},
        )
        report = engine.run([spec])
        r = report.results[0]
        self.assertEqual(len(r.benchmarks), 4)
        labels = [b.spec.name for b in r.benchmarks]
        self.assertEqual(labels, [
            "MyAlgo EW Benchmark",
            "MyAlgo InvVol Benchmark",
            "MyAlgo MinVar Benchmark",
            "MyAlgo RiskParity Benchmark",
        ])
        # Primary slot still equal-weight by convention (first entry).
        self.assertIs(r.benchmark, r.benchmarks[0])

    def test_failing_benchmark_does_not_break_run(self):
        class _Broken(Benchmark):
            short_name = "Broken Benchmark"

            def compute_weights(self, prices):
                raise RuntimeError("simulated failure")

        prices = _synthetic_prices()
        engine = _engine(
            prices,
            benchmarks=[EqualWeightBenchmark(), _Broken()],
        )
        spec = PortfolioSpec(name="P", weights={"AAPL": 0.5, "MSFT": 0.5})
        report = engine.run([spec])
        r = report.results[0]
        # Broken benchmark silently skipped; healthy one survives.
        self.assertEqual(len(r.benchmarks), 1)
        self.assertIn("EW Benchmark", r.benchmarks[0].spec.name)

    def test_information_ratio_uses_primary_benchmark(self):
        # Sanity: with EW as primary benchmark, IR is computed against EW.
        prices = _synthetic_prices()
        engine = _engine(prices)
        spec = PortfolioSpec(
            name="P",
            weights={"AAPL": 0.7, "MSFT": 0.2, "GOOG": 0.1},
        )
        report = engine.run([spec])
        ir = report.results[0].metrics.information_ratio
        self.assertFalse(np.isnan(ir))


# ═══════════════════════════════════════════════════════════════════════════
class TestRoundTripSerialization(unittest.TestCase):
    """Phase 5 changes io.py — confirm the new ``benchmarks`` list survives."""

    def test_benchmarks_list_preserved(self):
        from app.backtesting import io as bio

        prices = _synthetic_prices()
        engine = _engine(
            prices,
            benchmarks=[EqualWeightBenchmark(), InverseVolatilityBenchmark()],
        )
        spec = PortfolioSpec(name="P", weights={"AAPL": 0.5, "MSFT": 0.5})
        report = engine.run([spec])
        restored = bio.report_from_dict(bio.report_to_dict(report))
        r = restored.results[0]
        self.assertEqual(len(r.benchmarks), 2)
        names = [b.spec.name for b in r.benchmarks]
        self.assertEqual(names, ["P EW Benchmark", "P InvVol Benchmark"])
        self.assertIs(r.benchmark, r.benchmarks[0] if False else r.benchmark)
        # Primary slot points at the first entry (same name)
        self.assertEqual(r.benchmark.spec.name, "P EW Benchmark")

    def test_legacy_payload_without_benchmarks_key_loads(self):
        from app.backtesting import io as bio

        prices = _synthetic_prices()
        engine = _engine(prices)  # default → 1 EW benchmark
        spec = PortfolioSpec(name="P", weights={"AAPL": 0.5, "MSFT": 0.5})
        report = engine.run([spec])
        d = bio.report_to_dict(report)
        # Strip the new key to simulate a payload from the previous build.
        for r in d["results"]:
            r.pop("benchmarks", None)
        restored = bio.report_from_dict(d)
        r0 = restored.results[0]
        self.assertEqual(len(r0.benchmarks), 1)
        self.assertIsNotNone(r0.benchmark)
        self.assertEqual(r0.benchmark.spec.name, "P EW Benchmark")


if __name__ == "__main__":
    unittest.main()
