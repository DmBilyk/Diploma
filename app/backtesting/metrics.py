"""
metrics.py
==========

Pure functions implementing the extended financial metrics used by
:class:`BacktestEngine`.

These are kept separate from the engine so that:

* each metric has a single, easily-tested implementation;
* the engine file stays focused on simulation orchestration;
* the testing chapter of the thesis can target small, well-defined units.

All functions accept plain numpy / pandas inputs and return scalars.
None of them touch the database, the engine state, or any global config.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

_EPSILON = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
#  Risk-adjusted ratios
# ═══════════════════════════════════════════════════════════════════════════

def calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calmar ratio = CAGR / |MaxDD|.

    Returns ``nan`` when drawdown is effectively zero (no downside observed).
    """
    dd = abs(max_drawdown)
    if dd < _EPSILON:
        return float("nan")
    return float(cagr / dd)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    ann_factor: float,
) -> float:
    """Annualised information ratio of *returns* over *benchmark_returns*.

    IR = mean(active) / std(active) · √ann_factor, where
    active = portfolio - benchmark per period.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    sd = float(active.std())
    if sd < _EPSILON:
        return float("nan")
    return float(active.mean() / sd) * float(np.sqrt(ann_factor))


def tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    ann_factor: float,
) -> float:
    """Annualised tracking error: std(portfolio - benchmark) · √ann_factor."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    return float(active.std()) * float(np.sqrt(ann_factor))


# ═══════════════════════════════════════════════════════════════════════════
#  Tail-risk metrics
# ═══════════════════════════════════════════════════════════════════════════

def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR at *level* (0.95 → 95 %).

    Returned as a **positive** number representing the loss magnitude:
    e.g. 0.04 means a one-period loss of 4 % is the cutoff at the
    chosen confidence level.  Returns ``nan`` for empty input.
    """
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    quantile = float(np.quantile(r.values, 1.0 - level))
    return float(-quantile) if quantile < 0 else 0.0


def historical_cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Historical CVaR (a.k.a. Expected Shortfall) at *level*.

    Mean loss in the worst (1-level) tail, expressed as a positive number.
    """
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    cutoff = float(np.quantile(r.values, 1.0 - level))
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return 0.0
    mean_tail = float(tail.mean())
    return float(-mean_tail) if mean_tail < 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Drawdown / dispersion
# ═══════════════════════════════════════════════════════════════════════════

def ulcer_index(values: pd.Series) -> float:
    """Ulcer Index — RMS of percentage drawdowns from running peak.

    Lower is better; rewards smooth equity curves and punishes deep,
    long drawdowns more harshly than max-drawdown alone.
    """
    v = values.dropna()
    if len(v) < 2:
        return float("nan")
    peak = v.cummax()
    dd_pct = (v / peak - 1.0) * 100.0
    return float(np.sqrt((dd_pct ** 2).mean()))


def downside_deviation(
    returns: pd.Series,
    mar: float = 0.0,
    ann_factor: float = 1.0,
) -> float:
    """Annualised downside deviation below the Minimum Acceptable Return.

    Only periods with ``return < mar`` contribute.  Returns ``nan`` if
    no periods are below the threshold.
    """
    r = returns.dropna()
    downside = r[r < mar] - mar
    if len(downside) < 2:
        return float("nan")
    return float(downside.std()) * float(np.sqrt(ann_factor))


# ═══════════════════════════════════════════════════════════════════════════
#  Distribution-shape metrics
# ═══════════════════════════════════════════════════════════════════════════

def best_period_return(returns: pd.Series) -> float:
    """Maximum single-period return."""
    r = returns.dropna()
    return float(r.max()) if len(r) else float("nan")


def worst_period_return(returns: pd.Series) -> float:
    """Minimum single-period return."""
    r = returns.dropna()
    return float(r.min()) if len(r) else float("nan")


def win_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive return (0–1)."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).sum()) / float(len(r))


# ═══════════════════════════════════════════════════════════════════════════
#  Portfolio-composition metrics
# ═══════════════════════════════════════════════════════════════════════════

def count_holdings(weights: Dict[str, float], threshold: float = 1e-6) -> int:
    """Number of assets with weight strictly greater than *threshold*."""
    return int(sum(1 for w in weights.values() if w > threshold))


def turnover(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    rebalance_every: Optional[int],
) -> float:
    """Mean per-rebalance one-way turnover (sum of |Δw| / 2 across rebalances).

    Mechanics
    ---------
    Drift weights between rebalance points using the actual price path,
    then at each rebalance compute ½·Σ|w_target − w_drifted|.  This is
    the standard "one-way" turnover (a full sell-and-rebuy of x % of the
    book counts as x %, not 2x %).

    Buy-and-hold (``rebalance_every is None``) returns 0.0.
    Returns ``nan`` if there are zero rebalance events.
    """
    if rebalance_every is None:
        return 0.0
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be >= 1")

    tickers = list(weights.keys())
    w_target = np.array([weights[t] for t in tickers], dtype=np.float64)
    s = w_target.sum()
    if s > _EPSILON:
        w_target = w_target / s

    sub = prices[tickers].astype(np.float64).values
    n_rows = sub.shape[0]
    if n_rows < 2:
        return float("nan")

    first = sub[0]
    if not np.all(np.isfinite(first)) or np.any(first <= _EPSILON):
        return float("nan")

    shares = w_target / first  # unit-capital shares
    events = []
    for i in range(1, n_rows):
        if i % rebalance_every != 0:
            continue
        cur = sub[i]
        if not np.all(np.isfinite(cur)) or np.any(cur <= _EPSILON):
            continue
        positions = shares * cur
        total = positions.sum()
        if total <= _EPSILON:
            continue
        w_drift = positions / total
        events.append(0.5 * float(np.abs(w_target - w_drift).sum()))
        shares = (total * w_target) / cur  # re-balanced shares

    if not events:
        return float("nan")
    return float(np.mean(events))
