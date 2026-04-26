"""
app/ai/data_prep.py
===================

Prepare market data for PPO training.

This module converts raw price history into clean return matrices that can be
passed directly to :class:`~app.ai.environment.PortfolioEnv`.

Key responsibilities
--------------------
* Load price history from ``PortfolioRepository``
* Fill short data gaps and remove tickers with weak coverage
* Compute clipped percentage returns
* Split data chronologically to avoid leakage
* Keep the asset universe small enough for stable PPO observations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from app.data.repository import PortfolioRepository

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Grouped train, validation, and test return matrices."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    tickers: List[str]

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DataSplit(train={self.train.shape}, val={self.val.shape}, "
            f"test={self.test.shape}, tickers={len(self.tickers)})"
        )


def load_and_prepare(
    repo: PortfolioRepository,
    tickers: Optional[List[str]] = None,
    start_date: str = "2010-01-01",
    end_date: str = "2023-12-31",
    max_assets: int = 50,
    min_coverage: float = 0.95,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    outlier_clip: float = 0.10,
) -> DataSplit:
    """Load prices, clean them, and create train/validation/test splits.

    Parameters
    ----------
    repo : PortfolioRepository
    tickers : list[str] | None
        Explicit ticker list.  ``None`` means all database tickers are used.
    start_date, end_date : str
        Date range for price data.
    max_assets : int
        Maximum number of assets to keep, selected by highest data coverage.
        This keeps the PPO observation space bounded.
    min_coverage : float
        Drop tickers with fewer than ``min_coverage`` fraction of non-NaN rows.
    train_frac, val_frac : float
        Proportional split sizes.  The test share is the remaining data.
    outlier_clip : float
        Clip individual returns to +/-``outlier_clip`` so bad ticks and rare
        shocks do not dominate early training.

    Returns
    -------
    DataSplit
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not (0.0 <= val_frac < 1.0):
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac must be < 1.0, got {train_frac + val_frac:.3f} "
            "(no data would remain for the test split)"
        )
    if not (0.0 < outlier_clip <= 1.0):
        raise ValueError(f"outlier_clip must be in (0, 1], got {outlier_clip}")
    if max_assets < 1:
        raise ValueError(f"max_assets must be ≥ 1, got {max_assets}")

    if tickers is None:
        tickers = repo.get_all_tickers()
    logger.info("Loading prices for %d tickers [%s → %s]", len(tickers), start_date, end_date)

    prices: pd.DataFrame = repo.get_price_history(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    logger.info("Raw price matrix: %s", prices.shape)

    # ── 1. Fill short gaps without masking long missing periods ─────────
    prices = prices.ffill(limit=5)

    # ── 2. Remove assets that would leave too many missing observations ─
    coverage = prices.notna().mean()
    ok_tickers = coverage[coverage >= min_coverage].index.tolist()
    dropped = len(tickers) - len(ok_tickers)
    if dropped:
        logger.info("Dropped %d tickers with coverage < %.0f%%", dropped, min_coverage * 100)
    prices = prices[ok_tickers]

    # ── 3. Keep the most complete assets when the universe is too large ─
    if len(ok_tickers) > max_assets:
        top = coverage[ok_tickers].nlargest(max_assets).index.tolist()
        logger.info(
            "Selecting top %d assets by coverage (from %d eligible)", max_assets, len(ok_tickers)
        )
        prices = prices[top]

    # Keep only rows where every selected asset has a valid price.
    prices = prices.dropna()
    logger.info("Clean price matrix: %s", prices.shape)

    if len(prices) < 100:
        raise ValueError(
            f"Only {len(prices)} rows remain after cleaning – "
            "check your date range or data quality."
        )

    # ── 4. Compute returns and limit extreme single-period moves ────────
    returns: pd.DataFrame = prices.pct_change().dropna()
    returns = returns.clip(-outlier_clip, outlier_clip)

    # ── 5. Split chronologically so future returns never leak backward ──
    n = len(returns)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = returns.iloc[:n_train]
    val = returns.iloc[n_train : n_train + n_val]
    test = returns.iloc[n_train + n_val :]

    logger.info(
        "Split → train=%d  val=%d  test=%d rows | assets=%d",
        len(train), len(val), len(test), returns.shape[1],
    )

    return DataSplit(
        train=train,
        val=val,
        test=test,
        tickers=list(returns.columns),
    )
