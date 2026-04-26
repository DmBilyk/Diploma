"""
backtest_engine.py
==================

Universal retrospective backtesting engine.

Accepts ready-made portfolios (weight dictionaries from any optimiser),
a date range, and a budget.  Simulates historical performance and
computes standard financial metrics.

Usage
-----
>>> from app.backtesting.backtest_engine import BacktestEngine, PortfolioSpec
>>> spec = PortfolioSpec(name="My Strategy", weights={"AAPL": 0.4, "MSFT": 0.6})
>>> engine = BacktestEngine(
...     start_date="2020-01-01",
...     end_date="2025-01-01",
...     initial_capital=100_000,
... )
>>> report = engine.run([spec])
>>> print(report.results[0].metrics)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from app.backtesting import metrics as _metrics
from app.backtesting.benchmarks import Benchmark, EqualWeightBenchmark
from app.data.repository import PortfolioRepository

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PortfolioSpec:
    """Input descriptor for a single portfolio.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. "Markowitz", "Hybrid Evo").
    weights : Dict[str, float]
        Mapping *ticker → weight*.  Weights should sum to ≈ 1.0.
        Only tickers with weight > 0 need to be present.
    """

    name: str
    weights: Dict[str, float]


@dataclass(frozen=True)
class BacktestMetrics:
    """Aggregated financial metrics for a backtest run.

    All rates are expressed as decimals (0.12 = 12 %).
    """

    total_return: float
    cagr: float
    annualised_volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    start_value: float
    end_value: float

    # ── Extended metrics (Phase 1) ────────────────────────────────────
    # All default to NaN / 0 so old constructors and serialised reports
    # remain valid.  Benchmark-dependent metrics are only populated when
    # a benchmark return series is supplied to `_compute_metrics`.
    calmar_ratio: float = float("nan")
    information_ratio: float = float("nan")
    tracking_error: float = float("nan")
    var_95: float = float("nan")
    cvar_95: float = float("nan")
    ulcer_index: float = float("nan")
    downside_deviation: float = float("nan")
    best_period_return: float = float("nan")
    worst_period_return: float = float("nan")
    win_rate: float = float("nan")
    turnover: float = float("nan")
    avg_n_holdings: int = 0


@dataclass
class BacktestResult:
    """Result for a single portfolio.

    ``benchmark`` is preserved for backward compatibility — it always points
    to the *first* entry of ``benchmarks`` (typically the equal-weight
    benchmark, when default engine config is used).  New code should
    iterate ``benchmarks`` to access the full set.
    """

    spec: PortfolioSpec
    metrics: BacktestMetrics
    portfolio_values: pd.Series  # indexed by datetime
    benchmark: Optional["BacktestResult"] = None  # legacy: == benchmarks[0]
    benchmarks: List["BacktestResult"] = field(default_factory=list)


@dataclass
class BacktestReport:
    """Aggregated report containing all portfolio results + benchmark."""

    results: List[BacktestResult]
    benchmark: Optional[BacktestResult]
    start_date: str
    end_date: str
    initial_capital: float

    # ── Persistence / export (Phase 2) ────────────────────────────────
    # Implementations live in app.backtesting.io to keep this dataclass
    # minimal.  Imported lazily to avoid a circular import.

    def to_dict(self) -> dict:
        from app.backtesting import io as _io
        return _io.report_to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BacktestReport":
        from app.backtesting import io as _io
        return _io.report_from_dict(d)

    def save_json(self, path: str) -> None:
        from app.backtesting import io as _io
        _io.save_json(self, path)

    @classmethod
    def load_json(cls, path: str) -> "BacktestReport":
        from app.backtesting import io as _io
        return _io.load_json(path)

    def save_csv(self, directory: str) -> dict:
        from app.backtesting import io as _io
        return _io.save_csv(self, directory)

    def save_html(self, path: str) -> None:
        from app.backtesting import io as _io
        _io.save_html(self, path)

    # ── Statistical comparison (Phase 6) ──────────────────────────────

    def compare(
        self,
        name_a: str,
        name_b: str,
        ann_factor: float = 52.0,
        n_boot: int = 1000,
        ci: float = 0.95,
        seed: Optional[int] = None,
    ) -> dict:
        """Run JKM + paired t + bootstrap CI on two named results.

        Looks up portfolios by ``spec.name`` from :attr:`results` (only the
        strategies — benchmarks are not addressable here; pass them as a
        full :class:`BacktestResult` to :func:`statistics.compare_results`
        directly if needed).
        """
        from app.backtesting import statistics as _stats
        by_name = {r.spec.name: r for r in self.results}
        if name_a not in by_name:
            raise KeyError(f"No result named {name_a!r}")
        if name_b not in by_name:
            raise KeyError(f"No result named {name_b!r}")
        return _stats.compare_results(
            by_name[name_a], by_name[name_b],
            ann_factor=ann_factor, n_boot=n_boot, ci=ci, seed=seed,
        )

    def compare_all_pairs(
        self,
        ann_factor: float = 52.0,
        n_boot: int = 1000,
        ci: float = 0.95,
        seed: Optional[int] = None,
    ) -> dict:
        """Pairwise comparison between every strategy in :attr:`results`."""
        from app.backtesting import statistics as _stats
        return _stats.compare_all_pairs(
            self.results,
            ann_factor=ann_factor, n_boot=n_boot, ci=ci, seed=seed,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WalkForwardWindow:
    """One in-sample / out-of-sample window in a walk-forward run."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class WalkForwardReport:
    """Output of :meth:`BacktestEngine.run_walk_forward`.

    Attributes
    ----------
    windows
        Successfully evaluated windows in chronological order.
    window_reports
        One :class:`BacktestReport` per window (1-to-1 with ``windows``).
    aggregated
        ``{algo_name: {metric_name: {"mean", "median", "std", "min", "max",
        "count", "values"}}}``.  Built by :func:`_aggregate_walk_forward`.
    """

    windows: List["WalkForwardWindow"]
    window_reports: List["BacktestReport"]
    aggregated: Dict[str, Dict[str, Dict[str, float]]]

    def algorithm_names(self) -> List[str]:
        return list(self.aggregated.keys())


# ═══════════════════════════════════════════════════════════════════════════
#  ROBUSTNESS / MULTI-SEED (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RobustnessReport:
    """Output of :meth:`BacktestEngine.run_with_seeds`.

    Attributes
    ----------
    seeds
        Seeds used for the successful runs (in order).
    seed_reports
        One :class:`BacktestReport` per seed (1-to-1 with ``seeds``).
    aggregated
        ``{algo_name: {metric_name: {"mean", "median", "std", "min", "max",
        "p05", "p25", "p75", "p95", "count", "values"}}}``.
        Built by :func:`_aggregate_robustness`.
    """

    seeds: List[int]
    seed_reports: List["BacktestReport"]
    aggregated: Dict[str, Dict[str, Dict[str, float]]]

    def algorithm_names(self) -> List[str]:
        return list(self.aggregated.keys())


def _aggregate_robustness(
    seed_reports: List["BacktestReport"],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Per-algorithm summary stats across seeds, including box-plot percentiles."""
    import math

    by_algo: Dict[str, Dict[str, List[float]]] = {}
    for report in seed_reports:
        for result in report.results:
            algo = result.spec.name
            algo_bucket = by_algo.setdefault(algo, {})
            for metric in _AGGREGATABLE_METRICS:
                v = getattr(result.metrics, metric)
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                algo_bucket.setdefault(metric, []).append(float(v))

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for algo, metric_map in by_algo.items():
        algo_summary: Dict[str, Dict[str, float]] = {}
        for metric, vals in metric_map.items():
            arr = np.array(vals, dtype=np.float64)
            algo_summary[metric] = {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "p05":    float(np.percentile(arr, 5)),
                "p25":    float(np.percentile(arr, 25)),
                "p75":    float(np.percentile(arr, 75)),
                "p95":    float(np.percentile(arr, 95)),
                "count":  int(len(arr)),
                "values": vals,
            }
        aggregated[algo] = algo_summary
    return aggregated


# Metrics aggregated across windows.  Excludes start_value / end_value
# because they're absolute capital snapshots — a "mean end_value" across
# windows is meaningless.
_AGGREGATABLE_METRICS = (
    "total_return", "cagr", "annualised_volatility", "max_drawdown",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "information_ratio", "tracking_error",
    "var_95", "cvar_95", "ulcer_index", "downside_deviation",
    "best_period_return", "worst_period_return", "win_rate",
    "turnover", "avg_n_holdings",
)


def _aggregate_walk_forward(
    window_reports: List["BacktestReport"],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Collapse per-window metrics into per-algorithm summary statistics."""
    import math

    by_algo: Dict[str, Dict[str, List[float]]] = {}
    for report in window_reports:
        for result in report.results:
            algo = result.spec.name
            algo_bucket = by_algo.setdefault(algo, {})
            for metric in _AGGREGATABLE_METRICS:
                v = getattr(result.metrics, metric)
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                algo_bucket.setdefault(metric, []).append(float(v))

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for algo, metric_map in by_algo.items():
        algo_summary: Dict[str, Dict[str, float]] = {}
        for metric, vals in metric_map.items():
            arr = np.array(vals, dtype=np.float64)
            algo_summary[metric] = {
                "mean":   float(arr.mean()),
                "median": float(np.median(arr)),
                "std":    float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "count":  int(len(arr)),
                "values": vals,
            }
        aggregated[algo] = algo_summary
    return aggregated


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_MIN_SERIES_LENGTH = 4          # minimum observations to compute metrics
_ANNUALISATION_FACTOR = 52      # weekly data → annual
_EPSILON = 1e-12                # guard against division by zero
_WEIGHT_SUM_TOLERANCE = 0.01    # acceptable deviation of Σw from 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════════════════


class BacktestEngine:
    """Universal retrospective backtesting engine.

    Parameters
    ----------
    start_date : str
        ISO-format start of the test window (inclusive).
    end_date : str
        ISO-format end of the test window (inclusive).
    initial_capital : float
        Budget in dollars invested at the start.
    risk_free_rate : float
        Annualised risk-free rate used for Sharpe / Sortino.
    repo : PortfolioRepository | None
        Data source.  Created lazily if not provided.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.02,
        repo: Optional[PortfolioRepository] = None,
        rebalance_every: Optional[int] = None,
        benchmarks: Optional[Sequence[Benchmark]] = None,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if rebalance_every is not None and rebalance_every < 1:
            raise ValueError("rebalance_every must be a positive integer")

        try:
            _t_start = pd.Timestamp(start_date)
            _t_end   = pd.Timestamp(end_date)
        except Exception as exc:
            raise ValueError(
                f"Invalid date format: start='{start_date}', end='{end_date}'"
            ) from exc
        if _t_start >= _t_end:
            raise ValueError(
                f"start_date ({start_date}) must be strictly before "
                f"end_date ({end_date})"
            )

        self._start = start_date
        self._end = end_date
        self._capital = float(initial_capital)
        self._rf = float(risk_free_rate)
        self._repo = repo
        self._rebalance_every = rebalance_every
        # Default preserves legacy behaviour: one per-portfolio EW benchmark.
        self._benchmarks: List[Benchmark] = (
            list(benchmarks) if benchmarks else [EqualWeightBenchmark()]
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(self, portfolios: List[PortfolioSpec]) -> BacktestReport:
        """Execute the backtest for every portfolio in *portfolios*.

        Returns a :class:`BacktestReport` containing per-portfolio metrics,
        value series, per-portfolio equal-weight benchmarks, and a
        (deprecated) global equal-weight benchmark.
        """
        if not portfolios:
            raise ValueError("At least one PortfolioSpec is required")

        # Collect every ticker mentioned by any portfolio.
        all_tickers = sorted(
            {t for spec in portfolios for t in spec.weights}
        )

        prices = self._load_prices(all_tickers)
        logger.info(
            "Loaded prices: %d tickers, %d observations (%s → %s)",
            prices.shape[1],
            prices.shape[0],
            prices.index[0].date(),
            prices.index[-1].date(),
        )

        return self._run_with_prices(portfolios, prices, self._start, self._end)

    def _run_with_prices(
        self,
        portfolios: List[PortfolioSpec],
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> BacktestReport:
        """Backtest a list of portfolios against an already-loaded price matrix.

        Used by :meth:`run` (whole-period) and :meth:`run_walk_forward`
        (per-window) so the simulation logic is not duplicated.
        """
        results: List[BacktestResult] = []
        for spec in portfolios:
            self._validate_spec(spec, prices.columns)
            values = self._simulate(
                spec.weights, prices, self._capital, self._rebalance_every,
            )

            # Per-portfolio benchmarks — one per configured Benchmark.
            # All benchmarks see the same asset universe as the strategy
            # under test, so comparisons remain like-for-like.
            p_tickers = list(spec.weights.keys())
            spec_prices = prices[p_tickers]
            bench_results: List[BacktestResult] = []
            for bench in self._benchmarks:
                try:
                    b_weights = bench.compute_weights(spec_prices)
                except Exception as exc:
                    logger.warning(
                        "Benchmark %s failed for portfolio %r — skipped: %s",
                        bench.short_name, spec.name, exc,
                    )
                    continue
                if not b_weights:
                    continue
                b_vals = self._simulate(
                    b_weights, prices, self._capital, self._rebalance_every,
                )
                b_metrics = self._compute_metrics(
                    b_vals, weights=b_weights, prices=prices,
                )
                bench_results.append(BacktestResult(
                    spec=PortfolioSpec(
                        name=f"{spec.name} {bench.short_name}",
                        weights=b_weights,
                    ),
                    metrics=b_metrics,
                    portfolio_values=b_vals,
                ))

            # Primary benchmark (legacy slot) = first successful benchmark,
            # which under default config is equal-weight — matches the
            # historical contract and keeps existing tests / serialisers happy.
            primary = bench_results[0] if bench_results else None
            primary_vals = primary.portfolio_values if primary is not None else None

            metrics = self._compute_metrics(
                values,
                weights=spec.weights,
                prices=prices,
                benchmark_values=primary_vals,
            )

            results.append(BacktestResult(
                spec=spec,
                metrics=metrics,
                portfolio_values=values,
                benchmark=primary,
                benchmarks=bench_results,
            ))

        return BacktestReport(
            results=results,
            benchmark=None,   # global EW benchmark removed; use BacktestResult.benchmark
            start_date=start_date,
            end_date=end_date,
            initial_capital=self._capital,
        )

    # ------------------------------------------------------------------
    #  Walk-forward (Phase 3)
    # ------------------------------------------------------------------

    def run_walk_forward(
        self,
        spec_factory: Callable[[pd.DataFrame], List[PortfolioSpec]],
        universe: Sequence[str],
        train_periods: int,
        test_periods: int,
        step: Optional[int] = None,
        expanding: bool = False,
    ) -> "WalkForwardReport":
        """Rolling / expanding walk-forward backtest.

        Parameters
        ----------
        spec_factory
            Callable invoked once per window with the **training-slice price
            DataFrame**.  Must return one or more :class:`PortfolioSpec`
            objects (one per algorithm being compared).  Use the same names
            across windows so per-algorithm aggregation works.
        universe
            Tickers loaded once for the whole evaluation window.  Each
            window slices this matrix in time only — the asset universe
            is constant.
        train_periods, test_periods
            Length of the in-sample and out-of-sample slices, measured in
            **price-index rows** (so weekly data → 52 ≈ 1 year).
        step
            Rows to advance between windows.  Defaults to ``test_periods``
            (non-overlapping test windows).
        expanding
            If True, every window's training slice starts from row 0 and
            grows; if False (default), it rolls forward by *step* rows.

        Returns
        -------
        WalkForwardReport
            Per-window :class:`BacktestReport` objects, the window
            definitions, and per-algorithm aggregated metrics.

        Notes
        -----
        * The engine's ``start_date`` / ``end_date`` define the *outer*
          range from which windows are carved; nothing is loaded outside it.
        * Look-ahead bias is structurally prevented: ``spec_factory`` only
          ever sees the training slice; the backtest runs against the
          disjoint test slice.
        """
        if train_periods < _MIN_SERIES_LENGTH:
            raise ValueError(f"train_periods must be ≥ {_MIN_SERIES_LENGTH}")
        if test_periods < _MIN_SERIES_LENGTH:
            raise ValueError(f"test_periods must be ≥ {_MIN_SERIES_LENGTH}")
        if not universe:
            raise ValueError("universe must contain at least one ticker")
        if step is None:
            step = test_periods
        if step < 1:
            raise ValueError("step must be a positive integer")

        full_prices = self._load_prices(sorted(set(universe)))
        n_obs = len(full_prices)
        if n_obs < train_periods + test_periods:
            raise ValueError(
                f"Not enough observations ({n_obs}) for "
                f"train_periods={train_periods} + test_periods={test_periods}"
            )

        windows: List[WalkForwardWindow] = []
        window_reports: List[BacktestReport] = []

        cursor = 0
        while cursor + train_periods + test_periods <= n_obs:
            train_lo = 0 if expanding else cursor
            train_hi = cursor + train_periods
            test_lo = train_hi
            test_hi = test_lo + test_periods

            train_slice = full_prices.iloc[train_lo:train_hi]
            test_slice = full_prices.iloc[test_lo:test_hi]

            window = WalkForwardWindow(
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
            )

            try:
                specs = spec_factory(train_slice)
            except Exception as exc:
                logger.warning(
                    "Walk-forward window %s skipped — spec_factory raised: %s",
                    window, exc,
                )
                cursor += step
                continue

            if not specs:
                logger.warning(
                    "Walk-forward window %s skipped — spec_factory returned no portfolios",
                    window,
                )
                cursor += step
                continue

            try:
                report = self._run_with_prices(
                    list(specs),
                    test_slice,
                    start_date=str(window.test_start.date()),
                    end_date=str(window.test_end.date()),
                )
            except Exception as exc:
                logger.warning(
                    "Walk-forward window %s skipped — backtest raised: %s",
                    window, exc,
                )
                cursor += step
                continue

            windows.append(window)
            window_reports.append(report)
            cursor += step

        if not window_reports:
            raise RuntimeError(
                "Walk-forward produced zero successful windows; "
                "check spec_factory and date range"
            )

        return WalkForwardReport(
            windows=windows,
            window_reports=window_reports,
            aggregated=_aggregate_walk_forward(window_reports),
        )

    # ------------------------------------------------------------------
    #  Robustness / multi-seed (Phase 4)
    # ------------------------------------------------------------------

    def run_with_seeds(
        self,
        spec_factory: Callable[[int], List[PortfolioSpec]],
        seeds: Sequence[int],
    ) -> "RobustnessReport":
        """Run the same evaluation N times with different RNG seeds.

        Designed for stochastic optimisers (PPO, Hybrid Evo) where a
        single run does not characterise the algorithm — variance across
        seeds is the honest answer.

        Parameters
        ----------
        spec_factory
            Callable invoked once per seed.  Receives the integer seed and
            must return one or more :class:`PortfolioSpec` objects, with
            consistent ``name`` values across seeds so per-algorithm
            aggregation works.  The factory is responsible for using the
            seed to drive its internal RNG.
        seeds
            Integer seeds to evaluate.  Must contain at least one entry;
            duplicates are allowed but discouraged.

        Returns
        -------
        RobustnessReport
            Per-seed reports plus per-algorithm summary statistics
            (mean, median, std, percentiles 5/25/75/95, min/max, count,
            raw values for box plots).

        Notes
        -----
        Per-seed failures (factory raises, backtest raises) are logged
        and skipped.  If every seed fails, the method raises.  Prices are
        loaded fresh for each seed via the standard :meth:`run` path,
        because the universe may differ between seeds (e.g. an Evo
        optimiser selects different assets per run).
        """
        if not seeds:
            raise ValueError("seeds must contain at least one integer")

        seed_list: List[int] = []
        seed_reports: List[BacktestReport] = []

        for seed in seeds:
            try:
                specs = spec_factory(int(seed))
            except Exception as exc:
                logger.warning(
                    "Robustness seed=%s skipped — spec_factory raised: %s",
                    seed, exc,
                )
                continue
            if not specs:
                logger.warning(
                    "Robustness seed=%s skipped — spec_factory returned no portfolios",
                    seed,
                )
                continue
            try:
                report = self.run(list(specs))
            except Exception as exc:
                logger.warning(
                    "Robustness seed=%s skipped — backtest raised: %s",
                    seed, exc,
                )
                continue
            seed_list.append(int(seed))
            seed_reports.append(report)

        if not seed_reports:
            raise RuntimeError(
                "Robustness run produced zero successful seeds; "
                "check spec_factory"
            )

        return RobustnessReport(
            seeds=seed_list,
            seed_reports=seed_reports,
            aggregated=_aggregate_robustness(seed_reports),
        )

    # ------------------------------------------------------------------
    #  Data loading
    # ------------------------------------------------------------------

    def _load_prices(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch price matrix from the repository and validate it."""
        if self._repo is None:
            self._repo = PortfolioRepository()

        df = self._repo.get_price_history(
            tickers,
            start_date=self._start,
            end_date=self._end,
        )

        if df.empty:
            raise RuntimeError(
                f"No price data for tickers {tickers} "
                f"in range [{self._start}, {self._end}]"
            )

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Forward-fill NaN gaps so one missing price does not
        # contaminate the entire value series downstream.
        n_nan_before = int(df.isna().sum().sum())
        if n_nan_before > 0:
            df = df.ffill()
            start_valid = df.iloc[0].notna()

            if not start_valid.all():
                dropped = df.columns[~start_valid].tolist()
                logger.warning(
                    "_load_prices: dropping %d tickers without стартової ціни: %s",
                    len(dropped), dropped,
                )
                df = df.loc[:, start_valid]
            n_nan_after = int(df.isna().sum().sum())
            logger.warning(
                "_load_prices: forward-filled %d NaN cells (%d remain)",
                n_nan_before - n_nan_after, n_nan_after,
            )

        return df

    # ------------------------------------------------------------------
    #  Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate(
        weights: Dict[str, float],
        prices: pd.DataFrame,
        initial_capital: float,
        rebalance_every: Optional[int] = None,
    ) -> pd.Series:
        """Convert weights + price matrix → portfolio value series.

        Parameters
        ----------
        rebalance_every : int | None
            If *None* → buy-and-hold (original behaviour).
            If an integer N → rebalance to target weights every N rows.
            Since data is weekly, N=4 ≈ monthly, N=13 ≈ quarterly.
        """
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers], dtype=np.float64)

        # Normalize so the full capital is always deployed, even when
        # weights sum to e.g. 0.995 (within the accepted tolerance).
        w_sum = w.sum()
        if w_sum > _EPSILON:
            w = w / w_sum

        subset = prices[tickers].copy()
        price_matrix = subset.values.astype(np.float64)  # (T, n_assets)
        n_rows = price_matrix.shape[0]

        # Validate first-row prices.
        first_prices = price_matrix[0]
        safe_first = np.where(
            (first_prices > _EPSILON) & np.isfinite(first_prices),
            first_prices,
            np.nan,
        )

        if np.isnan(safe_first).any():
            missing = [
                t for t, p in zip(tickers, safe_first) if np.isnan(p)
            ]
            raise ValueError(
                f"No valid opening price for tickers: {missing}. "
                "They may not have data in the selected date range."
            )

        # ── Buy-and-hold (original behaviour) ────────────────────────
        if rebalance_every is None:
            capital_per_asset = w * initial_capital
            shares = capital_per_asset / safe_first
            values = (price_matrix * shares).sum(axis=1)
            return pd.Series(values, index=subset.index, name="portfolio_value")

        # ── Periodic rebalancing ─────────────────────────────────────
        values = np.empty(n_rows, dtype=np.float64)
        current_value = initial_capital

        # Initial share purchase
        shares = (current_value * w) / safe_first
        values[0] = current_value

        for i in range(1, n_rows):
            current_prices = price_matrix[i]
            current_value = float((shares * current_prices).sum())
            values[i] = current_value

            # Rebalance at every N-th row (skip if prices are invalid)
            if i % rebalance_every == 0:
                safe_prices = np.where(
                    (current_prices > _EPSILON) & np.isfinite(current_prices),
                    current_prices,
                    np.nan,
                )
                if np.isnan(safe_prices).any():
                    bad = [t for t, p in zip(tickers, safe_prices) if np.isnan(p)]
                    logger.warning(
                        "Rebalance skipped at row %d: invalid prices for %s. "
                        "Portfolio will drift from target weights until next "
                        "successful rebalance.",
                        i, bad,
                    )
                else:
                    shares = (current_value * w) / safe_prices

        return pd.Series(values, index=subset.index, name="portfolio_value")

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        values: pd.Series,
        weights: Optional[Dict[str, float]] = None,
        prices: Optional[pd.DataFrame] = None,
        benchmark_values: Optional[pd.Series] = None,
    ) -> BacktestMetrics:
        """Derive financial metrics from a portfolio value series.

        Parameters
        ----------
        values : pd.Series
            Portfolio value series (required).
        weights, prices : optional
            If both provided, enables ``turnover`` and ``avg_n_holdings``.
        benchmark_values : pd.Series, optional
            Enables ``information_ratio`` and ``tracking_error``.
        """
        if len(values) < _MIN_SERIES_LENGTH:
            raise ValueError(
                f"Too few data points ({len(values)}) to compute metrics; "
                f"need at least {_MIN_SERIES_LENGTH}"
            )

        start_val = float(values.iloc[0])
        end_val = float(values.iloc[-1])
        total_ret = (end_val / start_val) - 1.0

        # 1. ДИНАМІЧНИЙ ФАКТОР АНУАЛІЗАЦІЇ (Рятує від змішаних дат у БД)
        days_total = (values.index[-1] - values.index[0]).days
        years = max(days_total / 365.25, _EPSILON)
        ann_factor = len(values) / years  # Самостійно вираховує, скільки торгів було в році

        cagr_val = (end_val / start_val) ** (1.0 / years) - 1.0
        returns = values.pct_change().dropna()

        # 2. Використовуємо ann_factor замість жорстких 52
        ann_vol = float(returns.std() * np.sqrt(ann_factor))

        rf_period = (1.0 + self._rf) ** (1.0 / ann_factor) - 1.0
        excess = returns - rf_period

        sharpe = (
                float(excess.mean() / (excess.std() + _EPSILON))
                * np.sqrt(ann_factor)
        )

        # Sortino
        downside = excess[excess < 0]
        if len(downside) < 2:
            sortino = float("nan")
        else:
            downside_std = float(downside.std())
            if downside_std < _EPSILON:
                sortino = float("nan")
            else:
                sortino = (
                        float(excess.mean() / downside_std)
                        * np.sqrt(ann_factor)
                )

        drawdown = float(((values / values.cummax()) - 1.0).min())

        # ── Extended metrics (delegated to app.backtesting.metrics) ──
        calmar = _metrics.calmar_ratio(cagr_val, drawdown)
        var95 = _metrics.historical_var(returns, level=0.95)
        cvar95 = _metrics.historical_cvar(returns, level=0.95)
        ulcer = _metrics.ulcer_index(values)
        ddev = _metrics.downside_deviation(returns, mar=0.0, ann_factor=ann_factor)
        best = _metrics.best_period_return(returns)
        worst = _metrics.worst_period_return(returns)
        wrate = _metrics.win_rate(returns)

        if benchmark_values is not None and len(benchmark_values) >= 2:
            bench_returns = benchmark_values.pct_change().dropna()
            ir = _metrics.information_ratio(returns, bench_returns, ann_factor)
            te = _metrics.tracking_error(returns, bench_returns, ann_factor)
        else:
            ir = float("nan")
            te = float("nan")

        if weights is not None:
            n_holdings = _metrics.count_holdings(weights)
            if prices is not None:
                to = _metrics.turnover(weights, prices, self._rebalance_every)
            else:
                to = float("nan")
        else:
            n_holdings = 0
            to = float("nan")

        return BacktestMetrics(
            total_return=total_ret,
            cagr=cagr_val,
            annualised_volatility=ann_vol,
            max_drawdown=drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            start_value=start_val,
            end_value=end_val,
            calmar_ratio=calmar,
            information_ratio=ir,
            tracking_error=te,
            var_95=var95,
            cvar_95=cvar95,
            ulcer_index=ulcer,
            downside_deviation=ddev,
            best_period_return=best,
            worst_period_return=worst,
            win_rate=wrate,
            turnover=to,
            avg_n_holdings=n_holdings,
        )

    # ------------------------------------------------------------------
    #  Benchmark
    # ------------------------------------------------------------------

    def _build_benchmark(self, prices: pd.DataFrame) -> BacktestResult:
        """Equal-weight benchmark across all available tickers.

        .. deprecated::
            This global benchmark uses *all* tickers from *all* portfolios.
            Prefer the per-portfolio ``BacktestResult.benchmark`` instead.
        """
        warnings.warn(
            "BacktestReport.benchmark (global EW benchmark) is deprecated. "
            "Use BacktestResult.benchmark (per-portfolio EW benchmark) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        n = prices.shape[1]
        eq_weights = {t: 1.0 / n for t in prices.columns}
        spec = PortfolioSpec(name="Equal-Weight Benchmark (deprecated)", weights=eq_weights)
        values = self._simulate(eq_weights, prices, self._capital, self._rebalance_every)
        metrics = self._compute_metrics(values, weights=eq_weights, prices=prices)
        return BacktestResult(spec=spec, metrics=metrics, portfolio_values=values)

    # ------------------------------------------------------------------
    #  Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_spec(
        spec: PortfolioSpec,
        available: pd.Index,
    ) -> None:
        """Raise early if the portfolio specification is invalid."""
        if not spec.weights:
            raise ValueError(f"Portfolio '{spec.name}' has no weights")

        missing = [t for t in spec.weights if t not in available]
        if missing:
            raise ValueError(
                f"Portfolio '{spec.name}': tickers {missing} have no "
                "price data in the selected date range"
            )

        negative = {t: w for t, w in spec.weights.items() if w < 0}
        if negative:
            raise ValueError(
                f"Portfolio '{spec.name}': negative weights are not "
                f"supported — {negative}"
            )

        total = sum(spec.weights.values())
        if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"Portfolio '{spec.name}': weights sum to {total:.4f}, "
                f"expected ≈ 1.0 (tolerance ±{_WEIGHT_SUM_TOLERANCE})"
            )