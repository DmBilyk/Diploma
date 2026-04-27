"""
test_io.py
==========

Tests for :mod:`app.backtesting.io` — JSON round-trip, CSV export, HTML report.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from app.backtesting import io as bio
from app.backtesting.backtest_engine import (
    BacktestMetrics,
    BacktestReport,
    BacktestResult,
    PortfolioSpec,
)


def _series(n: int = 12, start_value: float = 100_000.0) -> pd.Series:
    rng = np.random.default_rng(42)
    rets = rng.normal(0.001, 0.01, size=n - 1)
    vals = [start_value]
    for r in rets:
        vals.append(vals[-1] * (1.0 + float(r)))
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    return pd.Series(vals, index=idx, name="portfolio_value")


def _metrics(**overrides) -> BacktestMetrics:
    base = dict(
        total_return=0.12, cagr=0.058, annualised_volatility=0.14,
        max_drawdown=-0.07, sharpe_ratio=0.85, sortino_ratio=1.10,
        start_value=100_000.0, end_value=112_000.0,
        calmar_ratio=0.83, information_ratio=0.42, tracking_error=0.04,
        var_95=0.025, cvar_95=0.038, ulcer_index=2.7,
        downside_deviation=0.09, best_period_return=0.05,
        worst_period_return=-0.04, win_rate=0.58, turnover=0.12,
        avg_n_holdings=8,
    )
    base.update(overrides)
    return BacktestMetrics(**base)


def _make_report(n_portfolios: int = 2) -> BacktestReport:
    results = []
    for i in range(n_portfolios):
        spec = PortfolioSpec(name=f"Strategy {i + 1}", weights={"AAPL": 0.6, "MSFT": 0.4})
        bench_spec = PortfolioSpec(
            name=f"Strategy {i + 1} EW Benchmark",
            weights={"AAPL": 0.5, "MSFT": 0.5},
        )
        bench = BacktestResult(
            spec=bench_spec,
            metrics=_metrics(sharpe_ratio=0.5),
            portfolio_values=_series(start_value=100_000.0),
        )
        results.append(BacktestResult(
            spec=spec,
            metrics=_metrics(sharpe_ratio=0.7 + i * 0.1),
            portfolio_values=_series(start_value=100_000.0),
            benchmark=bench,
        ))
    return BacktestReport(
        results=results,
        benchmark=None,
        start_date="2020-01-01",
        end_date="2020-04-01",
        initial_capital=100_000.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
class TestRoundTrip(unittest.TestCase):

    def test_dict_roundtrip_preserves_metrics(self):
        report = _make_report()
        d = bio.report_to_dict(report)
        restored = bio.report_from_dict(d)

        self.assertEqual(len(restored.results), len(report.results))
        self.assertEqual(restored.start_date, report.start_date)
        self.assertEqual(restored.end_date, report.end_date)
        self.assertEqual(restored.initial_capital, report.initial_capital)

        for orig, got in zip(report.results, restored.results):
            self.assertEqual(orig.spec.name, got.spec.name)
            self.assertEqual(orig.spec.weights, got.spec.weights)
            self.assertAlmostEqual(orig.metrics.sharpe_ratio, got.metrics.sharpe_ratio)
            self.assertAlmostEqual(orig.metrics.calmar_ratio, got.metrics.calmar_ratio)
            self.assertEqual(orig.metrics.avg_n_holdings, got.metrics.avg_n_holdings)
            pd.testing.assert_series_equal(
                orig.portfolio_values.reset_index(drop=True),
                got.portfolio_values.reset_index(drop=True),
                check_names=False,
            )

    def test_nan_preserved_as_nan(self):
        report = _make_report(1)
        # Wipe one metric to NaN to confirm it survives the round-trip
        report.results[0] = BacktestResult(
            spec=report.results[0].spec,
            metrics=_metrics(information_ratio=float("nan"), tracking_error=float("nan")),
            portfolio_values=report.results[0].portfolio_values,
            benchmark=report.results[0].benchmark,
        )
        restored = bio.report_from_dict(bio.report_to_dict(report))
        self.assertTrue(math.isnan(restored.results[0].metrics.information_ratio))
        self.assertTrue(math.isnan(restored.results[0].metrics.tracking_error))

    def test_benchmark_is_round_tripped(self):
        report = _make_report(1)
        restored = bio.report_from_dict(bio.report_to_dict(report))
        self.assertIsNotNone(restored.results[0].benchmark)
        self.assertEqual(
            restored.results[0].benchmark.spec.name,
            report.results[0].benchmark.spec.name,
        )

    def test_schema_version_mismatch_rejected(self):
        d = bio.report_to_dict(_make_report(1))
        d["schema_version"] = 99
        with self.assertRaises(ValueError):
            bio.report_from_dict(d)


# ═══════════════════════════════════════════════════════════════════════════
class TestJsonPersistence(unittest.TestCase):

    def test_json_save_and_load(self):
        report = _make_report(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            bio.save_json(report, path)
            self.assertTrue(os.path.exists(path))

            # Plain JSON load → must be valid JSON without NaN tokens
            with open(path, "r", encoding="utf-8") as fh:
                raw = fh.read()
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)
            json.loads(raw)  # must parse cleanly

            restored = bio.load_json(path)
            self.assertEqual(len(restored.results), 2)
            self.assertEqual(restored.results[0].spec.name, "Strategy 1")

    def test_method_on_report_dataclass(self):
        report = _make_report(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "via_method.json")
            report.save_json(path)
            restored = BacktestReport.load_json(path)
            self.assertEqual(restored.results[0].spec.name, "Strategy 1")


# ═══════════════════════════════════════════════════════════════════════════
class TestCsvExport(unittest.TestCase):

    def test_csv_files_created(self):
        report = _make_report(2)
        with tempfile.TemporaryDirectory() as tmp:
            written = bio.save_csv(report, tmp)
            self.assertIn("summary.csv", written)
            for r in report.results:
                key = f"equity_{r.spec.name.replace(' ', '_')}.csv"
                self.assertIn(key, written)
                self.assertTrue(os.path.exists(written[key]))

    def test_summary_csv_contains_all_metrics_columns(self):
        report = _make_report(1)
        with tempfile.TemporaryDirectory() as tmp:
            bio.save_csv(report, tmp)
            df = pd.read_csv(os.path.join(tmp, "summary.csv"))
            for col in ["portfolio", "kind", "sharpe_ratio", "calmar_ratio",
                        "var_95", "cvar_95", "turnover", "avg_n_holdings"]:
                self.assertIn(col, df.columns)
            # one strategy row + one benchmark row
            self.assertEqual(len(df), 2)
            self.assertEqual(set(df["kind"]), {"strategy", "benchmark"})

    def test_equity_csv_round_trip(self):
        report = _make_report(1)
        with tempfile.TemporaryDirectory() as tmp:
            bio.save_csv(report, tmp)
            equity_file = os.path.join(tmp, "equity_Strategy_1.csv")
            df = pd.read_csv(equity_file, index_col="date", parse_dates=True)
            self.assertIn("Strategy 1", df.columns)
            self.assertEqual(len(df), len(report.results[0].portfolio_values))


# ═══════════════════════════════════════════════════════════════════════════
class TestHtmlReport(unittest.TestCase):

    def test_html_file_is_self_contained(self):
        report = _make_report(2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            bio.save_html(report, path)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as fh:
                html = fh.read()
            # Charts embedded as base64 PNG, not external files
            self.assertIn("data:image/png;base64,", html)
            # Metrics table includes portfolio name + at least one new metric
            self.assertIn("Strategy 1", html)
            self.assertIn("Sharpe", html)
            # No unresolved template placeholders
            self.assertNotIn("{equity_png}", html)
            self.assertNotIn("{table}", html)


if __name__ == "__main__":
    unittest.main()
