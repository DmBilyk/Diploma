"""
test_metrics.py
===============

Unit tests for :mod:`app.backtesting.metrics`.

Each test uses a small synthetic series with a hand-computable answer so
the test failure messages directly point at a single broken formula.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from app.backtesting import metrics as m


def _series(values, freq: str = "W"):
    idx = pd.date_range("2020-01-05", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
#  Risk-adjusted ratios
# ═══════════════════════════════════════════════════════════════════════════
class TestCalmarRatio(unittest.TestCase):

    def test_basic(self):
        self.assertAlmostEqual(m.calmar_ratio(0.20, -0.10), 2.0)

    def test_handles_positive_drawdown_argument(self):
        # API accepts either sign convention for max_drawdown
        self.assertAlmostEqual(m.calmar_ratio(0.20, 0.10), 2.0)

    def test_zero_drawdown_returns_nan(self):
        self.assertTrue(math.isnan(m.calmar_ratio(0.10, 0.0)))


class TestInformationRatio(unittest.TestCase):

    def test_zero_active_returns_nan(self):
        r = _series([0.01, 0.02, -0.005, 0.0, 0.015])
        self.assertTrue(math.isnan(m.information_ratio(r, r.copy(), ann_factor=52.0)))

    def test_constant_outperformance_yields_inf_or_large(self):
        # Constant active return → std=0 → defined as nan
        r = _series([0.02] * 10)
        b = _series([0.01] * 10)
        self.assertTrue(math.isnan(m.information_ratio(r, b, ann_factor=52.0)))

    def test_known_value(self):
        # active = [0.01, -0.01, 0.02, -0.02]
        # mean = 0, std = √((0.01²+0.01²+0.02²+0.02²)/3) ≈ 0.01826
        # IR = 0 / std · √52 = 0
        r = _series([0.03, 0.00, 0.04, 0.00])
        b = _series([0.02, 0.01, 0.02, 0.02])
        self.assertAlmostEqual(m.information_ratio(r, b, ann_factor=52.0), 0.0, places=10)


class TestTrackingError(unittest.TestCase):

    def test_identical_series_zero_te(self):
        r = _series([0.01, 0.02, 0.0, -0.01, 0.03])
        self.assertAlmostEqual(m.tracking_error(r, r.copy(), ann_factor=52.0), 0.0)

    def test_known_value(self):
        # active = [0.01, 0.01], std (ddof=1) = 0.0 → returns 0
        r = _series([0.02, 0.03])
        b = _series([0.01, 0.02])
        self.assertAlmostEqual(m.tracking_error(r, b, ann_factor=52.0), 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Tail-risk
# ═══════════════════════════════════════════════════════════════════════════
class TestVaR(unittest.TestCase):

    def test_var_returns_positive_loss(self):
        # Worst 5 % of [-10..-1, 0..89] / 100 → quantile at 5 % ≈ -0.05
        rets = _series([x / 100.0 for x in range(-10, 90)])
        v = m.historical_var(rets, level=0.95)
        self.assertGreater(v, 0.0)
        self.assertAlmostEqual(v, 0.0505, places=3)

    def test_var_zero_when_no_losses(self):
        rets = _series([0.01, 0.02, 0.03, 0.04, 0.05])
        self.assertEqual(m.historical_var(rets, level=0.95), 0.0)

    def test_invalid_level(self):
        rets = _series([0.01, 0.02])
        with self.assertRaises(ValueError):
            m.historical_var(rets, level=1.5)


class TestCVaR(unittest.TestCase):

    def test_cvar_geq_var(self):
        rng = np.random.default_rng(0)
        rets = _series(rng.normal(0.001, 0.02, size=200))
        v = m.historical_var(rets, level=0.95)
        c = m.historical_cvar(rets, level=0.95)
        self.assertGreaterEqual(c, v)

    def test_cvar_known_value(self):
        # Tail at 90 % cutoff over 10 ordered returns: bottom 1
        rets = _series([-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10])
        c = m.historical_cvar(rets, level=0.90)
        self.assertAlmostEqual(c, 0.10, places=2)


# ═══════════════════════════════════════════════════════════════════════════
#  Drawdown / dispersion
# ═══════════════════════════════════════════════════════════════════════════
class TestUlcerIndex(unittest.TestCase):

    def test_no_drawdown_zero(self):
        v = _series([100, 101, 102, 103, 104])
        self.assertAlmostEqual(m.ulcer_index(v), 0.0)

    def test_known_value(self):
        # 100 → 80 → 100: dd_pct = [0, -20, 0] → RMS = √(400/3) ≈ 11.547
        v = _series([100, 80, 100])
        self.assertAlmostEqual(m.ulcer_index(v), math.sqrt(400.0 / 3.0), places=6)


class TestDownsideDeviation(unittest.TestCase):

    def test_no_downside_returns_nan(self):
        r = _series([0.01, 0.02, 0.03])
        self.assertTrue(math.isnan(m.downside_deviation(r, mar=0.0)))

    def test_annualisation(self):
        r = _series([-0.01, -0.02, 0.05, -0.03])
        a = m.downside_deviation(r, mar=0.0, ann_factor=1.0)
        b = m.downside_deviation(r, mar=0.0, ann_factor=52.0)
        self.assertAlmostEqual(b, a * math.sqrt(52.0), places=10)


# ═══════════════════════════════════════════════════════════════════════════
#  Distribution shape
# ═══════════════════════════════════════════════════════════════════════════
class TestPeriodReturns(unittest.TestCase):

    def setUp(self):
        self.r = _series([0.02, -0.01, 0.05, -0.03, 0.0, 0.04])

    def test_best(self):
        self.assertAlmostEqual(m.best_period_return(self.r), 0.05)

    def test_worst(self):
        self.assertAlmostEqual(m.worst_period_return(self.r), -0.03)

    def test_win_rate(self):
        # 3 strictly positive out of 6
        self.assertAlmostEqual(m.win_rate(self.r), 0.5)

    def test_empty_series(self):
        empty = pd.Series([], dtype=float)
        self.assertTrue(math.isnan(m.best_period_return(empty)))
        self.assertTrue(math.isnan(m.worst_period_return(empty)))
        self.assertTrue(math.isnan(m.win_rate(empty)))


# ═══════════════════════════════════════════════════════════════════════════
#  Portfolio composition
# ═══════════════════════════════════════════════════════════════════════════
class TestCountHoldings(unittest.TestCase):

    def test_threshold(self):
        w = {"A": 0.5, "B": 0.0, "C": 1e-9, "D": 0.5}
        self.assertEqual(m.count_holdings(w), 2)

    def test_all_held(self):
        w = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        self.assertEqual(m.count_holdings(w), 4)


class TestTurnover(unittest.TestCase):

    def test_buy_and_hold_zero(self):
        prices = pd.DataFrame(
            {"A": [10, 11, 12], "B": [20, 19, 22]},
            index=pd.date_range("2020-01-05", periods=3, freq="W"),
        )
        self.assertEqual(m.turnover({"A": 0.5, "B": 0.5}, prices, None), 0.0)

    def test_constant_prices_zero_turnover(self):
        # Prices flat → drift = 0 → no rebalance trades
        prices = pd.DataFrame(
            {"A": [10] * 6, "B": [20] * 6},
            index=pd.date_range("2020-01-05", periods=6, freq="W"),
        )
        self.assertAlmostEqual(
            m.turnover({"A": 0.5, "B": 0.5}, prices, rebalance_every=2),
            0.0,
        )

    def test_known_drift_one_rebalance(self):
        # Two assets, equal weights, A doubles while B halves between t0→t2.
        # At t2 (rebalance): drifted weights ∝ (1·2, 1·0.5) = (2, 0.5) → (0.8, 0.2)
        # |Δw| = (0.3, 0.3); one-way turnover = 0.3
        prices = pd.DataFrame(
            {"A": [10.0, 15.0, 20.0], "B": [20.0, 15.0, 10.0]},
            index=pd.date_range("2020-01-05", periods=3, freq="W"),
        )
        t = m.turnover({"A": 0.5, "B": 0.5}, prices, rebalance_every=2)
        self.assertAlmostEqual(t, 0.3, places=10)

    def test_invalid_rebalance_every(self):
        prices = pd.DataFrame({"A": [1.0, 1.0]}, index=pd.date_range("2020-01-05", periods=2, freq="W"))
        with self.assertRaises(ValueError):
            m.turnover({"A": 1.0}, prices, rebalance_every=0)


if __name__ == "__main__":
    unittest.main()
