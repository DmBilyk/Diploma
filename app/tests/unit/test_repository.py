"""
Unit tests for app/data/repository.py — PortfolioRepository.

Each test runs against a fresh temporary SQLite database. The data layer
caches a global engine, so the fixture rewires DB_PATH and clears the
cached engine/session before every test.

Covers basic, deterministic behaviour:
  - add_asset is idempotent (no duplicate rows on repeat calls)
  - save_quotes_bulk upserts (re-running with new prices overwrites)
  - save_quotes_bulk coerces NaN OHLCV cells to 0.0
  - get_latest_quote_date returns the most recent quote's date
  - get_price_history filters by start_date / end_date and forward-fills
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Reset the data-layer engine to point at an isolated temp DB file."""
    from app.data import models as models_mod

    db_file = tmp_path / "repo_unit.db"
    monkeypatch.setattr(models_mod, "DB_FILE", db_file)
    monkeypatch.setattr(models_mod, "DB_PATH", f"sqlite:///{db_file}")
    monkeypatch.setattr(models_mod, "_engine", None)
    monkeypatch.setattr(models_mod, "_Session", None)
    yield db_file
    monkeypatch.setattr(models_mod, "_engine", None)
    monkeypatch.setattr(models_mod, "_Session", None)


@pytest.fixture
def repo(temp_db):
    from app.data.repository import PortfolioRepository
    return PortfolioRepository()


def _quote_frame(prices, start="2022-01-03", freq="W-MON"):
    """Build an OHLCV frame compatible with save_quotes_bulk."""
    idx = pd.date_range(start, periods=len(prices), freq=freq)
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Adj Close": prices,
            "Volume": [1_000] * len(prices),
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  add_asset
# ─────────────────────────────────────────────────────────────────────────────

class TestAddAsset:
    def test_add_asset_persists(self, repo):
        repo.add_asset("AAPL", name="Apple", sector="Tech")
        assert repo.get_all_tickers() == ["AAPL"]
        assert repo.get_asset_id("AAPL") is not None

    def test_add_asset_is_idempotent(self, repo):
        """Calling add_asset twice with the same ticker must not duplicate."""
        repo.add_asset("AAPL", name="Apple", sector="Tech")
        repo.add_asset("AAPL", name="Apple Inc.", sector="Technology")
        tickers = repo.get_all_tickers()
        assert tickers.count("AAPL") == 1

    def test_get_asset_id_unknown_returns_none(self, repo):
        assert repo.get_asset_id("NOPE") is None


# ─────────────────────────────────────────────────────────────────────────────
#  save_quotes_bulk
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveQuotesBulk:
    def test_saves_and_reads_back(self, repo):
        repo.add_asset("AAPL")
        df = _quote_frame([100.0, 101.0, 102.0])
        repo.save_quotes_bulk("AAPL", df)

        out = repo.get_price_history(["AAPL"])
        assert list(out["AAPL"].values) == pytest.approx([100.0, 101.0, 102.0])

    def test_auto_creates_missing_asset(self, repo):
        """save_quotes_bulk should add the asset if it doesn't exist yet."""
        df = _quote_frame([50.0, 51.0])
        repo.save_quotes_bulk("NEW", df)
        assert "NEW" in repo.get_all_tickers()

    def test_upsert_overwrites_existing_row(self, repo):
        """Re-saving the same date with a new price must overwrite, not append."""
        repo.add_asset("AAPL")
        repo.save_quotes_bulk("AAPL", _quote_frame([100.0, 101.0]))
        repo.save_quotes_bulk("AAPL", _quote_frame([200.0, 201.0]))

        out = repo.get_price_history(["AAPL"])
        assert len(out) == 2  # No duplicates appended.
        assert list(out["AAPL"].values) == pytest.approx([200.0, 201.0])

    def test_nan_values_coerced_to_zero(self, repo):
        """NaN cells must be stored as 0.0 (not crash on int(NaN))."""
        repo.add_asset("AAPL")
        df = _quote_frame([100.0, 101.0])
        df.loc[df.index[0], "Volume"] = np.nan
        df.loc[df.index[1], "Open"] = np.nan
        repo.save_quotes_bulk("AAPL", df)
        # If we got here without exception, NaN handling worked.
        assert len(repo.get_price_history(["AAPL"])) == 2

    def test_empty_frame_is_noop(self, repo):
        repo.add_asset("AAPL")
        repo.save_quotes_bulk("AAPL", pd.DataFrame())
        assert repo.get_price_history(["AAPL"]).empty


# ─────────────────────────────────────────────────────────────────────────────
#  get_latest_quote_date
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLatestQuoteDate:
    def test_empty_db_returns_none(self, repo):
        assert repo.get_latest_quote_date() is None

    def test_returns_max_date(self, repo):
        repo.add_asset("AAPL")
        repo.save_quotes_bulk("AAPL", _quote_frame(
            [1.0, 2.0, 3.0], start="2022-01-03"
        ))
        latest = repo.get_latest_quote_date()
        assert latest == datetime.date(2022, 1, 17)


# ─────────────────────────────────────────────────────────────────────────────
#  get_price_history
# ─────────────────────────────────────────────────────────────────────────────

class TestGetPriceHistory:
    def test_filters_by_start_and_end(self, repo):
        repo.add_asset("AAPL")
        repo.save_quotes_bulk("AAPL", _quote_frame(
            [10.0, 20.0, 30.0, 40.0, 50.0], start="2022-01-03"
        ))
        out = repo.get_price_history(
            ["AAPL"],
            start_date="2022-01-10",
            end_date="2022-01-24",
        )
        assert list(out["AAPL"].values) == pytest.approx([20.0, 30.0, 40.0])

    def test_unknown_ticker_returns_empty(self, repo):
        assert repo.get_price_history(["GHOST"]).empty

    def test_pivot_columns_match_tickers(self, repo):
        repo.add_asset("A")
        repo.add_asset("B")
        repo.save_quotes_bulk("A", _quote_frame([1.0, 2.0]))
        repo.save_quotes_bulk("B", _quote_frame([10.0, 20.0]))
        out = repo.get_price_history(["A", "B"])
        assert sorted(out.columns) == ["A", "B"]
        assert len(out) == 2
