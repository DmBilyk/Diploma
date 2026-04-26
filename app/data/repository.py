import datetime
import pandas as pd
from sqlalchemy import select, delete, text
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
from .models import Asset, Quote, Experiment, init_db


class PortfolioRepository:
    def __init__(self):
        self.Session = init_db()

    def add_asset(self, ticker: str, name: str = None, sector: str = None):
        """Add an asset if it is not already present."""
        with self.Session() as session:
            stmt = sqlite_upsert(Asset).values(ticker=ticker, name=name, sector=sector)
            stmt = stmt.on_conflict_do_nothing(index_elements=['ticker'])
            session.execute(stmt)
            session.commit()

    def get_asset_id(self, ticker: str) -> int:
        """Return the database ID for a ticker."""
        with self.Session() as session:
            stmt = select(Asset.id).where(Asset.ticker == ticker)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def get_all_tickers(self) -> list[str]:
        with self.Session() as session:
            result = session.execute(select(Asset.ticker))
            return [row[0] for row in result.all()]

    def get_all_assets(self) -> list[dict]:
        """Return all assets with their available metadata."""
        with self.Session() as session:
            stmt = select(Asset.ticker, Asset.name, Asset.sector)
            result = session.execute(stmt).all()
            return [
                {"ticker": row[0], "name": row[1], "sector": row[2]}
                for row in result
            ]

    def save_quotes_bulk(self, ticker: str, df: pd.DataFrame):
        """Persist a batch of quotes for one ticker."""
        asset_id = self.get_asset_id(ticker)
        if not asset_id:
            self.add_asset(ticker)
            asset_id = self.get_asset_id(ticker)

        def _f(val) -> float:
            """Convert a value to float, replacing NaN with 0.0."""
            return float(val) if pd.notna(val) else 0.0

        records = []
        for index, row in df.iterrows():
            records.append({
                "asset_id": asset_id,
                "date": index.date(),
                "open":      _f(row.get('Open')),
                "high":      _f(row.get('High')),
                "low":       _f(row.get('Low')),
                "close":     _f(row.get('Close')),
                "adj_close": _f(row.get('Adj Close')),
                "volume":    int(_f(row.get('Volume'))),  # _f has already made NaN safe for int().
            })

        if not records:
            return

        with self.Session() as session:
            stmt = sqlite_upsert(Quote).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['asset_id', 'date'],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "adj_close": stmt.excluded.adj_close,
                    "volume": stmt.excluded.volume
                }
            )
            session.execute(stmt)
            session.commit()

    def get_price_history(self, tickers: list[str], start_date=None, end_date=None) -> pd.DataFrame:
        """Return an adjusted-close price matrix for algorithm use."""
        with self.Session() as session:
            query = select(Quote.date, Quote.adj_close, Asset.ticker) \
                .join(Asset) \
                .where(Asset.ticker.in_(tickers))

            if start_date:
                query = query.where(Quote.date >= start_date)
            if end_date:
                query = query.where(Quote.date <= end_date)

            df = pd.read_sql(query, session.connection())

        if df.empty:
            return pd.DataFrame()

        pivot_df = df.pivot(index='date', columns='ticker', values='adj_close')
        pivot_df.index = pd.to_datetime(pivot_df.index)
        pivot_df = pivot_df.ffill()
        return pivot_df

    def get_latest_quote_date(self) -> datetime.date | None:
        """Return the most recent quote date stored in the database."""
        with self.Session() as session:
            stmt = select(Quote.date).order_by(Quote.date.desc()).limit(1)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    # --- UI helper ---
    def get_quotes(self, ticker: str) -> pd.DataFrame:
        """Return a simple quote table for one asset in the UI."""
        with self.Session() as session:
            query = select(Quote.date, Quote.adj_close) \
                .join(Asset) \
                .where(Asset.ticker == ticker) \
                .order_by(Quote.date)

            df = pd.read_sql(query, session.connection())

            # Convert dates to datetime so Matplotlib can plot them.
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])

        return df
