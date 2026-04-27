"""
test_statistics.py
==================

Tests for :mod:`app.backtesting.statistics` (Phase 6).

Where possible, expected values are derived analytically from the test
data so the failure message points directly at a single broken formula.
"""

from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.backtesting import statistics as stats
from app.backtesting.backtest_engine import (
    BacktestEngine,
    PortfolioSpec,
)


def _series(values, freq: str = "W") -> pd.Series:
    idx = pd.date_range("2020-01-05", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


def _normal_returns(mean: float, sd: float, n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(rng.normal(mean, sd, size=n))


# ═══════════════════════════════════════════════════════════════════════════
class TestPearsonCorr(unittest.TestCase):

    def test_identical_series_corr_one(self):
        r = _series([0.01, 0.02, -0.01, 0.03, 0.0])
        self.assertAlmostEqual(stats.pearson_corr(r, r.copy()), 1.0)

    def test_anticorrelated_minus_one(self):
        r = _series([0.01, 0.02, -0.01, 0.03, 0.0])
        self.assertAlmostEqual(stats.pearson_corr(r, -r), -1.0)

    def test_constant_series_returns_nan(self):
        r = _series([0.0, 0.0, 0.0, 0.0])
        s = _series([0.01, 0.02, 0.03, 0.04])
        self.assertTrue(math.isnan(stats.pearson_corr(r, s)))


# ═══════════════════════════════════════════════════════════════════════════
class TestJKM(unittest.TestCase):

    def test_identical_series_zero_diff_high_p(self):
        r = _normal_returns(0.001, 0.02, 200, seed=1)
        out = stats.jobson_korkie_memmel(r, r.copy())
        self.assertAlmostEqual(out["sharpe_diff"], 0.0)
        self.assertAlmostEqual(out["z"], 0.0)
        self.assertAlmostEqual(out["p_value"], 1.0)
        self.assertAlmostEqual(out["correlation"], 1.0)

    def test_clearly_better_series_low_p(self):
        better = _normal_returns(0.005, 0.02, 500, seed=2)
        worse = _normal_returns(-0.001, 0.02, 500, seed=3)
        out = stats.jobson_korkie_memmel(better, worse)
        self.assertGreater(out["sharpe_diff"], 0.0)
        self.assertLess(out["p_value"], 0.05)

    def test_p_value_is_two_sided(self):
        # If A is worse than B, p should still be small (two-sided)
        worse = _normal_returns(-0.005, 0.02, 500, seed=4)
        better = _normal_returns(0.005, 0.02, 500, seed=5)
        out_a = stats.jobson_korkie_memmel(worse, better)
        out_b = stats.jobson_korkie_memmel(better, worse)
        self.assertAlmostEqual(out_a["p_value"], out_b["p_value"], places=12)
        # Diff and z reverse sign
        self.assertAlmostEqual(out_a["z"], -out_b["z"], places=12)

    def test_too_few_observations(self):
        out = stats.jobson_korkie_memmel(_series([0.01, 0.02]), _series([0.01, 0.02]))
        self.assertTrue(math.isnan(out["z"]))
        self.assertTrue(math.isnan(out["p_value"]))

    def test_annualisation_only_affects_reported_sharpes(self):
        a = _normal_returns(0.002, 0.02, 200, seed=6)
        b = _normal_returns(0.001, 0.02, 200, seed=7)
        out52 = stats.jobson_korkie_memmel(a, b, ann_factor=52.0)
        out252 = stats.jobson_korkie_memmel(a, b, ann_factor=252.0)
        # z and p are scale-invariant
        self.assertAlmostEqual(out52["z"], out252["z"], places=12)
        self.assertAlmostEqual(out52["p_value"], out252["p_value"], places=12)
        # Annualised Sharpe scales by sqrt(ratio)
        ratio = math.sqrt(252.0 / 52.0)
        self.assertAlmostEqual(out252["sharpe_a"], out52["sharpe_a"] * ratio, places=10)


# ═══════════════════════════════════════════════════════════════════════════
class TestBootstrapCI(unittest.TestCase):

    def test_estimate_matches_metric_on_sample(self):
        rets = _normal_returns(0.001, 0.02, 100, seed=8)
        out = stats.bootstrap_ci(np.mean, rets, n_boot=200, seed=42)
        self.assertAlmostEqual(out["estimate"], float(rets.mean()))

    def test_lower_below_upper(self):
        rets = _normal_returns(0.001, 0.02, 200, seed=9)
        out = stats.bootstrap_ci(np.mean, rets, n_boot=500, seed=42)
        self.assertLess(out["lower"], out["upper"])

    def test_ci_brackets_estimate(self):
        rets = _normal_returns(0.001, 0.02, 300, seed=10)
        out = stats.bootstrap_ci(np.mean, rets, n_boot=500, ci=0.95, seed=42)
        # Bootstrap mean should be near the original; original is usually
        # inside the percentile interval for this sample size.
        self.assertLessEqual(out["lower"], out["estimate"])
        self.assertGreaterEqual(out["upper"], out["estimate"])

    def test_seed_reproducibility(self):
        rets = _normal_returns(0.001, 0.02, 100, seed=11)
        a = stats.bootstrap_ci(np.mean, rets, n_boot=300, seed=123)
        b = stats.bootstrap_ci(np.mean, rets, n_boot=300, seed=123)
        self.assertAlmostEqual(a["lower"], b["lower"], places=12)
        self.assertAlmostEqual(a["upper"], b["upper"], places=12)

    def test_invalid_ci_rejected(self):
        rets = _series([0.01, 0.02, 0.03])
        with self.assertRaises(ValueError):
            stats.bootstrap_ci(np.mean, rets, n_boot=100, ci=1.5)

    def test_invalid_n_boot_rejected(self):
        rets = _series([0.01, 0.02, 0.03])
        with self.assertRaises(ValueError):
            stats.bootstrap_ci(np.mean, rets, n_boot=0)

    def test_too_short_returns_nan(self):
        rets = _series([0.01])
        out = stats.bootstrap_ci(np.mean, rets, n_boot=100)
        self.assertTrue(math.isnan(out["lower"]))


# ═══════════════════════════════════════════════════════════════════════════
class TestPairedReturnsTest(unittest.TestCase):

    def test_identical_series_p_one(self):
        r = _normal_returns(0.001, 0.02, 100, seed=12)
        out = stats.paired_returns_test(r, r.copy())
        self.assertAlmostEqual(out["mean_diff"], 0.0)
        self.assertEqual(out["p_value"], 1.0)
        self.assertAlmostEqual(out["std_diff"], 0.0)

    def test_constant_offset_zero_p(self):
        r = _normal_returns(0.001, 0.02, 100, seed=13)
        out = stats.paired_returns_test(r + 0.005, r.copy())
        self.assertAlmostEqual(out["mean_diff"], 0.005, places=12)
        # std_diff = 0 → p = 0 by our convention (mean is non-zero)
        self.assertEqual(out["p_value"], 0.0)

    def test_clear_difference_low_p(self):
        a = _normal_returns(0.005, 0.02, 500, seed=14)
        b = _normal_returns(0.0, 0.02, 500, seed=15)
        out = stats.paired_returns_test(a, b)
        self.assertGreater(out["mean_diff"], 0.0)
        self.assertLess(out["p_value"], 0.05)

    def test_no_difference_high_p(self):
        a = _normal_returns(0.001, 0.02, 500, seed=16)
        b = _normal_returns(0.001, 0.02, 500, seed=17)
        out = stats.paired_returns_test(a, b)
        self.assertGreater(out["p_value"], 0.05)

    def test_too_few_observations(self):
        out = stats.paired_returns_test(_series([0.01]), _series([0.02]))
        self.assertTrue(math.isnan(out["t"]))
        self.assertTrue(math.isnan(out["p_value"]))


# ═══════════════════════════════════════════════════════════════════════════
class TestCompareResults(unittest.TestCase):
    """Integration: compare two real BacktestResult objects via the engine."""

    def setUp(self):
        rng = np.random.default_rng(42)
        n = 156
        idx = pd.date_range("2018-01-07", periods=n, freq="W")
        self.prices = pd.DataFrame({
            "AAPL": 100.0 * np.cumprod(1.0 + rng.normal(0.0015, 0.02, n)),
            "MSFT": 100.0 * np.cumprod(1.0 + rng.normal(0.0010, 0.018, n)),
            "GOOG": 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.022, n)),
        }, index=idx)
        repo = MagicMock()
        repo.get_price_history.return_value = self.prices
        self.engine = BacktestEngine(
            start_date=str(idx[0].date()),
            end_date=str(idx[-1].date()),
            initial_capital=100_000.0,
            repo=repo,
        )

    def test_compare_returns_full_dict(self):
        report = self.engine.run([
            PortfolioSpec(name="A", weights={"AAPL": 1.0}),
            PortfolioSpec(name="B", weights={"GOOG": 1.0}),
        ])
        cmp = report.compare("A", "B", n_boot=200, seed=7)
        self.assertEqual(cmp["a"], "A")
        self.assertEqual(cmp["b"], "B")
        self.assertIn("jkm", cmp)
        self.assertIn("paired", cmp)
        self.assertIn("ann_mean_diff_bootstrap", cmp)
        for k in ("sharpe_a", "sharpe_b", "sharpe_diff", "z", "p_value"):
            self.assertIn(k, cmp["jkm"])
        self.assertIn("lower", cmp["ann_mean_diff_bootstrap"])
        self.assertIn("upper", cmp["ann_mean_diff_bootstrap"])

    def test_compare_unknown_name_raises(self):
        report = self.engine.run([PortfolioSpec(name="A", weights={"AAPL": 1.0})])
        with self.assertRaises(KeyError):
            report.compare("A", "Z")
        with self.assertRaises(KeyError):
            report.compare("Q", "A")

    def test_compare_all_pairs_count(self):
        report = self.engine.run([
            PortfolioSpec(name="A", weights={"AAPL": 1.0}),
            PortfolioSpec(name="B", weights={"GOOG": 1.0}),
            PortfolioSpec(name="C", weights={"MSFT": 1.0}),
        ])
        pairs = report.compare_all_pairs(n_boot=100, seed=7)
        self.assertEqual(len(pairs), 3)  # C(3,2) = 3
        # Keys are sorted alphabetically — never (B, A) or (C, A) reversed.
        for (a, b) in pairs:
            self.assertLess(a, b)


if __name__ == "__main__":
    unittest.main()
