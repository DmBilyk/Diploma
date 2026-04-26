"""
statistics.py
=============

Statistical comparison utilities for backtest results (Phase 6).

These let the user move from anecdotal claims ("algorithm A has Sharpe 1.2,
B has 1.05, A is better") to defensible ones ("A's Sharpe is 0.15 higher;
the JKM test rejects equality at p = 0.03; the 95 % bootstrap CI on the
difference is [0.04, 0.27]").

Exposed
-------
* :func:`pearson_corr` — small helper used by JKM.
* :func:`jobson_korkie_memmel` — Memmel-corrected JKM test on the Sharpe
  ratio difference between two return series.
* :func:`bootstrap_ci` — non-parametric percentile bootstrap CI for any
  metric ``fn(returns) -> float``.
* :func:`paired_returns_test` — paired t-test on per-period returns.
* :func:`compare_results` — convenience wrapper that runs all three on
  two :class:`BacktestResult` objects and returns a single dict suitable
  for the HTML/PDF report.

All functions are pure and deterministic given fixed seeds; none of them
touch the database, engine state, or any global config.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

_EPSILON = 1e-12


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def pearson_corr(a: pd.Series, b: pd.Series) -> float:
    """Pearson correlation between two aligned return series.

    Returns ``nan`` if either series has variance below numerical zero.
    """
    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(aligned) < 2:
        return float("nan")
    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    if sx < _EPSILON or sy < _EPSILON:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _normal_two_sided_p(z: float) -> float:
    """Two-sided p-value from a standard-normal z-score, using ``math.erf``."""
    if math.isnan(z) or math.isinf(z):
        return float("nan")
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _sample_sharpe(returns: pd.Series) -> float:
    """Per-period Sharpe (mean / std with ddof=1).  Annualisation is the
    caller's responsibility."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd < _EPSILON:
        return float("nan")
    return float(r.mean()) / sd


# ═══════════════════════════════════════════════════════════════════════════
#  Jobson–Korkie / Memmel — Sharpe difference test
# ═══════════════════════════════════════════════════════════════════════════

def jobson_korkie_memmel(
    returns_a: pd.Series,
    returns_b: pd.Series,
    ann_factor: float = 52.0,
) -> Dict[str, float]:
    """Memmel-corrected Jobson–Korkie test on the Sharpe-ratio difference.

    Returns
    -------
    dict
        ``sharpe_a``, ``sharpe_b`` — annualised Sharpe ratios.
        ``sharpe_diff``           — sharpe_a − sharpe_b (annualised).
        ``z``                     — test statistic.
        ``p_value``               — two-sided.
        ``correlation``           — Pearson ρ between the two return series.
        ``n``                     — number of paired observations used.

    Notes
    -----
    Implements the Memmel (2003) variance correction of the
    Jobson–Korkie (1981) test:

        Var(SR_a − SR_b) ≈ (1/T) · [ 2(1 − ρ)
            + 0.5 · (SR_a² + SR_b² − 2·SR_a·SR_b·ρ²) ]

    where SR_* are the **per-period** Sharpe ratios.  We compute the test
    in per-period units and only annualise the reported Sharpe values, so
    the z-score is invariant to the choice of ``ann_factor``.
    """
    aligned = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    if len(aligned) < 4:
        return {
            "sharpe_a": float("nan"), "sharpe_b": float("nan"),
            "sharpe_diff": float("nan"), "z": float("nan"),
            "p_value": float("nan"), "correlation": float("nan"),
            "n": int(len(aligned)),
        }
    a = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    n = len(aligned)

    sr_a_pp = _sample_sharpe(a)
    sr_b_pp = _sample_sharpe(b)
    rho = pearson_corr(a, b)

    if (math.isnan(sr_a_pp) or math.isnan(sr_b_pp) or math.isnan(rho)):
        return {
            "sharpe_a": sr_a_pp * math.sqrt(ann_factor) if not math.isnan(sr_a_pp) else float("nan"),
            "sharpe_b": sr_b_pp * math.sqrt(ann_factor) if not math.isnan(sr_b_pp) else float("nan"),
            "sharpe_diff": float("nan"), "z": float("nan"),
            "p_value": float("nan"), "correlation": rho, "n": n,
        }

    var_diff = (1.0 / n) * (
        2.0 * (1.0 - rho)
        + 0.5 * (sr_a_pp ** 2 + sr_b_pp ** 2 - 2.0 * sr_a_pp * sr_b_pp * rho ** 2)
    )
    if var_diff <= _EPSILON:
        # Degenerate variance — happens when the two series are identical
        # (or perfectly correlated with equal Sharpe).  If the Sharpe
        # difference is also numerically zero, the test trivially does not
        # reject equality (z=0, p=1).  Otherwise the statistic is undefined.
        if abs(sr_a_pp - sr_b_pp) <= _EPSILON:
            z = 0.0
            p = 1.0
        else:
            z = float("nan")
            p = float("nan")
    else:
        z = (sr_a_pp - sr_b_pp) / math.sqrt(var_diff)
        p = _normal_two_sided_p(z)

    sqrt_af = math.sqrt(ann_factor)
    return {
        "sharpe_a":    float(sr_a_pp * sqrt_af),
        "sharpe_b":    float(sr_b_pp * sqrt_af),
        "sharpe_diff": float((sr_a_pp - sr_b_pp) * sqrt_af),
        "z":           float(z) if not math.isnan(z) else float("nan"),
        "p_value":     float(p) if not math.isnan(p) else float("nan"),
        "correlation": float(rho),
        "n":           int(n),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Percentile bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    metric_fn: Callable[[np.ndarray], float],
    returns: pd.Series,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Non-parametric percentile bootstrap confidence interval.

    Parameters
    ----------
    metric_fn
        Callable mapping a 1-D numpy array of returns to a scalar.  Must
        be deterministic.
    returns
        Observed return series.
    n_boot
        Number of bootstrap resamples (default 1 000).
    ci
        Confidence level, e.g. 0.95 for a 95 % interval.
    seed
        Optional integer seed for reproducibility.

    Returns
    -------
    dict
        ``estimate`` — metric on the original sample.
        ``lower``, ``upper`` — percentile bounds at ``ci`` confidence.
        ``mean``, ``std`` — of the bootstrap distribution.
        ``n_boot``, ``ci``, ``n_obs``.
    """
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be strictly between 0 and 1")
    if n_boot < 1:
        raise ValueError("n_boot must be a positive integer")

    arr = returns.dropna().values.astype(np.float64)
    n = len(arr)
    if n < 2:
        return {
            "estimate": float("nan"), "lower": float("nan"), "upper": float("nan"),
            "mean": float("nan"), "std": float("nan"),
            "n_boot": int(n_boot), "ci": float(ci), "n_obs": int(n),
        }

    rng = np.random.default_rng(seed)
    estimate = float(metric_fn(arr))

    samples = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = float(metric_fn(arr[idx]))
        except Exception:
            samples[i] = float("nan")

    valid = samples[np.isfinite(samples)]
    if len(valid) < 2:
        return {
            "estimate": estimate, "lower": float("nan"), "upper": float("nan"),
            "mean": float("nan"), "std": float("nan"),
            "n_boot": int(n_boot), "ci": float(ci), "n_obs": int(n),
        }

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(valid, 100.0 * alpha))
    hi = float(np.percentile(valid, 100.0 * (1.0 - alpha)))
    return {
        "estimate": estimate,
        "lower":    lo,
        "upper":    hi,
        "mean":     float(valid.mean()),
        "std":      float(valid.std(ddof=1)),
        "n_boot":   int(n_boot),
        "ci":       float(ci),
        "n_obs":    int(n),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Paired t-test on per-period returns
# ═══════════════════════════════════════════════════════════════════════════

def paired_returns_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
) -> Dict[str, float]:
    """Two-sided paired t-test on the per-period return difference.

    Tests H₀: mean(a − b) = 0.  Uses the standard t-statistic with n − 1
    degrees of freedom.  The p-value is computed from
    :func:`scipy.stats.t.sf` when scipy is available, falling back to a
    normal approximation otherwise.

    Returns
    -------
    dict
        ``mean_diff``, ``std_diff`` (per-period, ddof=1),
        ``t``, ``df``, ``p_value``, ``n``.
    """
    aligned = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    n = len(aligned)
    if n < 2:
        return {
            "mean_diff": float("nan"), "std_diff": float("nan"),
            "t": float("nan"), "df": int(max(n - 1, 0)),
            "p_value": float("nan"), "n": int(n),
        }
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).values.astype(np.float64)
    mean_d = float(diff.mean())
    std_d = float(diff.std(ddof=1))
    if std_d < _EPSILON:
        # Either identical or constantly-offset series.  Mean tells the story;
        # variance is zero so t is undefined.
        return {
            "mean_diff": mean_d, "std_diff": std_d,
            "t": float("nan"), "df": int(n - 1),
            "p_value": 0.0 if abs(mean_d) > _EPSILON else 1.0,
            "n": int(n),
        }
    t = mean_d / (std_d / math.sqrt(n))
    df = n - 1
    try:
        from scipy.stats import t as _t_dist  # type: ignore
        p = float(2.0 * _t_dist.sf(abs(t), df))
    except ImportError:
        p = _normal_two_sided_p(t)
    return {
        "mean_diff": mean_d, "std_diff": std_d,
        "t":         float(t),
        "df":        int(df),
        "p_value":   float(p),
        "n":         int(n),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════

def compare_results(
    result_a,
    result_b,
    ann_factor: float = 52.0,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the full statistical comparison battery on two backtest results.

    ``result_a`` and ``result_b`` are :class:`BacktestResult` objects.
    The function only reads ``portfolio_values`` and ``spec.name``, so it
    does not couple to the engine internals.

    Returns a single nested dict suitable for direct rendering into the
    HTML/PDF report.
    """
    a_name = result_a.spec.name
    b_name = result_b.spec.name
    a_rets = result_a.portfolio_values.pct_change().dropna()
    b_rets = result_b.portfolio_values.pct_change().dropna()

    jkm = jobson_korkie_memmel(a_rets, b_rets, ann_factor=ann_factor)
    paired = paired_returns_test(a_rets, b_rets)

    diff = (a_rets - b_rets).dropna()

    def _ann_mean_diff(arr: np.ndarray) -> float:
        return float(np.mean(arr) * ann_factor)

    boot = bootstrap_ci(
        _ann_mean_diff, diff, n_boot=n_boot, ci=ci, seed=seed,
    )

    return {
        "a": a_name,
        "b": b_name,
        "jkm":     jkm,
        "paired":  paired,
        "ann_mean_diff_bootstrap": boot,
    }


def compare_all_pairs(
    results,
    ann_factor: float = 52.0,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Run :func:`compare_results` on every unordered pair of results.

    Keys are ``(name_a, name_b)`` tuples with ``name_a < name_b``.
    """
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    sorted_results = sorted(results, key=lambda r: r.spec.name)
    for i, ra in enumerate(sorted_results):
        for rb in sorted_results[i + 1:]:
            key = (ra.spec.name, rb.spec.name)
            pairs[key] = compare_results(
                ra, rb,
                ann_factor=ann_factor, n_boot=n_boot, ci=ci, seed=seed,
            )
    return pairs
