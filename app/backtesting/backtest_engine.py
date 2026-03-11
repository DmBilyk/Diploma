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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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


@dataclass
class BacktestResult:
    """Result for a single portfolio."""

    spec: PortfolioSpec
    metrics: BacktestMetrics
    portfolio_values: pd.Series  # indexed by datetime


@dataclass
class BacktestReport:
    """Aggregated report containing all portfolio results + benchmark."""

    results: List[BacktestResult]
    benchmark: Optional[BacktestResult]
    start_date: str
    end_date: str
    initial_capital: float


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

_MIN_SERIES_LENGTH = 4          # minimum observations to compute metrics
_ANNUALISATION_FACTOR = 52      # weekly data → annual
_EPSILON = 1e-12                # guard against division by zero
_WEIGHT_SUM_TOLERANCE = 0.05    # acceptable deviation of Σw from 1.0


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
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        self._start = start_date
        self._end = end_date
        self._capital = float(initial_capital)
        self._rf = float(risk_free_rate)
        self._repo = repo

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(self, portfolios: List[PortfolioSpec]) -> BacktestReport:
        """Execute the backtest for every portfolio in *portfolios*.

        Returns a :class:`BacktestReport` containing per-portfolio metrics,
        value series, and an equal-weight benchmark.
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

        results: List[BacktestResult] = []
        for spec in portfolios:
            self._validate_spec(spec, prices.columns)
            values = self._simulate(spec.weights, prices, self._capital)
            metrics = self._compute_metrics(values)
            results.append(BacktestResult(spec=spec, metrics=metrics, portfolio_values=values))

        benchmark = self._build_benchmark(prices)

        return BacktestReport(
            results=results,
            benchmark=benchmark,
            start_date=self._start,
            end_date=self._end,
            initial_capital=self._capital,
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

        return df

    # ------------------------------------------------------------------
    #  Simulation
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate(
        weights: Dict[str, float],
        prices: pd.DataFrame,
        initial_capital: float,
    ) -> pd.Series:
        """Convert weights + price matrix → portfolio value series.

        The simulation buys fractional shares at the first available price
        and holds them for the entire period (buy-and-hold).
        """
        tickers = list(weights.keys())
        w = np.array([weights[t] for t in tickers], dtype=np.float64)

        subset = prices[tickers].copy()

        # Replace any remaining NaN / zero first-row prices defensively.
        first_prices = subset.iloc[0].values.astype(np.float64)
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

        # Capital allocated to each asset → number of shares.
        capital_per_asset = w * initial_capital  # normalise later; use weight fractions
        shares = capital_per_asset / safe_first

        # Portfolio value at each timestamp.
        values = (subset.values * shares).sum(axis=1)
        return pd.Series(values, index=subset.index, name="portfolio_value")

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(self, values: pd.Series) -> BacktestMetrics:
        """Derive all financial metrics from a portfolio value series."""
        if len(values) < _MIN_SERIES_LENGTH:
            raise ValueError(
                f"Too few data points ({len(values)}) to compute metrics; "
                f"need at least {_MIN_SERIES_LENGTH}"
            )

        start_val = float(values.iloc[0])
        end_val = float(values.iloc[-1])

        total_ret = (end_val / start_val) - 1.0

        years = (values.index[-1] - values.index[0]).days / 365.25
        years = max(years, _EPSILON)  # guard
        cagr_val = (end_val / start_val) ** (1.0 / years) - 1.0

        returns = values.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(_ANNUALISATION_FACTOR))

        # Weekly risk-free rate for excess-return calculations.
        rf_period = (1.0 + self._rf) ** (1.0 / _ANNUALISATION_FACTOR) - 1.0
        excess = returns - rf_period

        sharpe = (
            float(excess.mean() / (returns.std() + _EPSILON))
            * np.sqrt(_ANNUALISATION_FACTOR)
        )

        # Sortino: penalise only downside deviation.
        downside = excess[excess < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else _EPSILON
        sortino = (
            float(excess.mean() / (downside_std + _EPSILON))
            * np.sqrt(_ANNUALISATION_FACTOR)
        )

        drawdown = float(((values / values.cummax()) - 1.0).min())

        return BacktestMetrics(
            total_return=total_ret,
            cagr=cagr_val,
            annualised_volatility=ann_vol,
            max_drawdown=drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            start_value=start_val,
            end_value=end_val,
        )

    # ------------------------------------------------------------------
    #  Benchmark
    # ------------------------------------------------------------------

    def _build_benchmark(self, prices: pd.DataFrame) -> BacktestResult:
        """Equal-weight benchmark across all available tickers."""
        n = prices.shape[1]
        eq_weights = {t: 1.0 / n for t in prices.columns}
        spec = PortfolioSpec(name="Equal-Weight Benchmark", weights=eq_weights)
        values = self._simulate(eq_weights, prices, self._capital)
        metrics = self._compute_metrics(values)
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
