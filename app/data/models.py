import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Date, ForeignKey, JSON, create_engine, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

logger = logging.getLogger(__name__)

# Project root resolved from app/data/models.py -> Diploma/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "resources" / "db"
DB_FILE = DB_DIR / "portfolio.db"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = f"sqlite:///{DB_FILE}"

logger.debug("Using Database at: %s", DB_PATH)


class Base(DeclarativeBase):
    pass


class Asset(Base):
    """Tradable instrument identified by ticker."""

    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    sector: Mapped[Optional[str]] = mapped_column(String(50))
    currency: Mapped[str] = mapped_column(String(10), default="USD")

    quotes: Mapped[List["Quote"]] = relationship(back_populates="asset", cascade="all, delete-orphan")


class Quote(Base):
    """Single OHLCV observation for an asset on a given date."""

    __tablename__ = "quotes"

    __table_args__ = (
        UniqueConstraint('asset_id', 'date', name='uix_asset_date'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[int] = mapped_column(ForeignKey("assets.id"))
    date: Mapped[datetime.date] = mapped_column(Date, index=True)

    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(Integer)

    asset: Mapped["Asset"] = relationship(back_populates="quotes")


class Experiment(Base):
    """Persisted record of an optimisation/backtest run."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    algorithm: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)

    parameters: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)


_engine = None
_Session = None

def init_db():
    """Initialise the engine and session factory on first call; idempotent."""
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(
            DB_PATH,
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(_engine)
        _Session = sessionmaker(_engine)
    return _Session
