import datetime
import os
from pathlib import Path
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Date, ForeignKey, JSON, create_engine, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# --- АБСОЛЮТНИЙ ШЛЯХ (FIX) ---
# 1. Знаходимо, де лежить цей файл (app/data/models.py)
CURRENT_FILE = Path(__file__).resolve()

# 2. Вираховуємо корінь проекту (Diploma/)
# app/data/models.py -> parent -> app/data -> parent -> app -> parent -> Diploma
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

# 3. Формуємо шлях до папки з базою
DB_DIR = PROJECT_ROOT / "resources" / "db"
DB_FILE = DB_DIR / "portfolio.db"

# Створюємо папку, якщо її немає (щоб не було помилок при першому запуску)
os.makedirs(DB_DIR, exist_ok=True)

# 4. Шлях для SQLAlchemy
DB_PATH = f"sqlite:///{DB_FILE}"

print(f"DEBUG: Using Database at: {DB_PATH}")


# -----------------------------

class Base(DeclarativeBase):
    pass


class Asset(Base):
    __tablename__ = "assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(100))
    sector: Mapped[Optional[str]] = mapped_column(String(50))
    currency: Mapped[str] = mapped_column(String(10), default="USD")

    quotes: Mapped[List["Quote"]] = relationship(back_populates="asset", cascade="all, delete-orphan")


class Quote(Base):
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
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    algorithm: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)

    parameters: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)


def init_db():
    engine = create_engine(DB_PATH, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)