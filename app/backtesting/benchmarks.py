"""
benchmarks.py
=============

Built-in benchmark strategies for :class:`BacktestEngine`.

A benchmark is a deterministic strategy used as a comparison baseline.
Every benchmark implements :class:`Benchmark`: given a price slice, it returns
a weight dictionary and can be passed to the engine with ``benchmarks=...``.

If no benchmarks are configured, the engine keeps the historical default:
one equal-weight benchmark for each tested portfolio.

Available benchmarks
--------------------
* :class:`EqualWeightBenchmark`     — w_i = 1/N (the naive baseline).
* :class:`MinimumVarianceBenchmark` — long-only min Σ-variance via SLSQP.
* :class:`RiskParityBenchmark`      — equal risk contribution (ERC),
  iterative scheme à la Maillard / Roncalli.
* :class:`InverseVolatilityBenchmark` — w_i ∝ 1/σ_i (a.k.a. naive RP).

Implementations use only the supplied price slice. They do not touch the
database, RNG, or global state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd

_EPSILON = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
#  Interface
# ═══════════════════════════════════════════════════════════════════════════


class Benchmark(ABC):
    """Base interface for benchmarks that produce portfolio weights.

    Implementations must be deterministic for a given input.  The
    backtest engine uses this contract to construct per-portfolio
    benchmark portfolios using the same asset universe as the strategy
    under test, so comparisons remain like-for-like.
    """

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Return a short display label, e.g. ``"EW Benchmark"``."""

    @abstractmethod
    def compute_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        """Return a fully invested ``{ticker: weight}`` mapping.

        ``prices`` is the asset-universe slice the engine wishes to
        evaluate (typically the strategy's own tickers).
        """


def _validate_prices(prices: pd.DataFrame) -> None:
    if prices is None or prices.empty:
        raise ValueError("Benchmark requires a non-empty price DataFrame")
    if prices.shape[1] == 0:
        raise ValueError("Benchmark requires at least one ticker column")


# ═══════════════════════════════════════════════════════════════════════════
#  Equal weight (1/N)
# ═══════════════════════════════════════════════════════════════════════════


class EqualWeightBenchmark(Benchmark):
    """Allocate equally across the supplied tickers."""

    short_name = "EW Benchmark"

    def compute_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        _validate_prices(prices)
        n = prices.shape[1]
        w = 1.0 / n
        return {t: w for t in prices.columns}


# ═══════════════════════════════════════════════════════════════════════════
#  Inverse volatility
# ═══════════════════════════════════════════════════════════════════════════


class InverseVolatilityBenchmark(Benchmark):
    """Allocate weights in inverse proportion to return volatility.

    Cheap to compute, robust, and a strong baseline.  Falls back to
    equal-weight on degenerate inputs (zero / NaN volatilities).
    """

    short_name = "InvVol Benchmark"

    def compute_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        _validate_prices(prices)
        rets = prices.pct_change().dropna()
        if len(rets) < 2:
            n = prices.shape[1]
            return {t: 1.0 / n for t in prices.columns}
        sigma = rets.std().replace(0.0, np.nan)
        if sigma.isna().all():
            n = prices.shape[1]
            return {t: 1.0 / n for t in prices.columns}
        inv = (1.0 / sigma).fillna(0.0)
        s = float(inv.sum())
        if s < _EPSILON:
            n = prices.shape[1]
            return {t: 1.0 / n for t in prices.columns}
        w = inv / s
        return {str(t): float(w[t]) for t in prices.columns}


# ═══════════════════════════════════════════════════════════════════════════
#  Minimum variance
# ═══════════════════════════════════════════════════════════════════════════


class MinimumVarianceBenchmark(Benchmark):
    """Build a long-only minimum-variance benchmark with SLSQP.

    Solved with scipy SLSQP starting from equal weights.  Falls back to
    equal-weight if the solver fails or the covariance is singular.
    """

    short_name = "MinVar Benchmark"

    def compute_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        _validate_prices(prices)
        n = prices.shape[1]
        rets = prices.pct_change().dropna()
        if len(rets) < n + 1:
            # Use equal weight when the covariance estimate would be unstable.
            return {t: 1.0 / n for t in prices.columns}

        cov = rets.cov().values.astype(np.float64)
        if not np.all(np.isfinite(cov)):
            return {t: 1.0 / n for t in prices.columns}

        # Add a small ridge term for numerical stability.
        cov = cov + np.eye(n) * 1e-10

        try:
            from scipy.optimize import minimize
        except ImportError:
            return {t: 1.0 / n for t in prices.columns}

        x0 = np.full(n, 1.0 / n)
        constraints = ({"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},)
        bounds = [(0.0, 1.0)] * n

        def objective(w):
            return float(w @ cov @ w)

        res = minimize(
            objective, x0, method="SLSQP",
            constraints=constraints, bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-10},
        )

        if not res.success:
            return {t: 1.0 / n for t in prices.columns}

        w = np.clip(res.x, 0.0, None)
        s = float(w.sum())
        if s < _EPSILON:
            return {t: 1.0 / n for t in prices.columns}
        w = w / s
        return {str(t): float(w[i]) for i, t in enumerate(prices.columns)}


# ═══════════════════════════════════════════════════════════════════════════
#  Risk parity (ERC)
# ═══════════════════════════════════════════════════════════════════════════


class RiskParityBenchmark(Benchmark):
    """Build an Equal-Risk-Contribution portfolio.

    Iteratively rescales weights until each asset contributes the same
    fraction of total portfolio variance.  Falls back to inverse-volatility
    on convergence failure.
    """

    short_name = "RiskParity Benchmark"

    def __init__(self, max_iter: int = 500, tol: float = 1e-8) -> None:
        self._max_iter = int(max_iter)
        self._tol = float(tol)

    def compute_weights(self, prices: pd.DataFrame) -> Dict[str, float]:
        _validate_prices(prices)
        n = prices.shape[1]
        rets = prices.pct_change().dropna()
        if len(rets) < n + 1:
            return InverseVolatilityBenchmark().compute_weights(prices)

        cov = rets.cov().values.astype(np.float64) + np.eye(n) * 1e-10
        if not np.all(np.isfinite(cov)):
            return InverseVolatilityBenchmark().compute_weights(prices)

        # Start from inverse-volatility weights for faster convergence.
        sigma = np.sqrt(np.diag(cov))
        if np.any(sigma < _EPSILON):
            return InverseVolatilityBenchmark().compute_weights(prices)
        w = 1.0 / sigma
        w = w / w.sum()

        target_rc = 1.0 / n
        for _ in range(self._max_iter):
            port_var = float(w @ cov @ w)
            if port_var < _EPSILON:
                break
            marginal = cov @ w
            rc = (w * marginal) / port_var       # Per-asset risk contribution.
            err = float(np.max(np.abs(rc - target_rc)))
            if err < self._tol:
                break
            # Move each asset toward the target contribution while preventing
            # zero weights from freezing the iteration.
            ratios = target_rc / np.maximum(rc, _EPSILON)
            w = w * np.sqrt(ratios)
            w = np.clip(w, 0.0, None)
            s = float(w.sum())
            if s < _EPSILON:
                return InverseVolatilityBenchmark().compute_weights(prices)
            w = w / s

        return {str(t): float(w[i]) for i, t in enumerate(prices.columns)}
