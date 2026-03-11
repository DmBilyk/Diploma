"""
Діагностичний скрипт для перевірки бази даних
"""
import sys
import os

# Додаємо шлях до модулів
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.data.repository import PortfolioRepository


def main():
    print("=== DATABASE DIAGNOSTIC ===\n")

    repo = PortfolioRepository()

    # 1. Перевірка кількості активів
    print("1. Checking assets...")
    assets = repo.get_all_assets()
    print(f"   Total assets: {len(assets)}")

    if assets:
        print(f"   First 5 assets:")
        for asset in assets[:5]:
            print(f"      - {asset['ticker']}: {asset['sector']}")
    else:
        print("   ⚠️ NO ASSETS FOUND!")
        return

    # 2. Перевірка котирувань для першого активу
    print("\n2. Checking quotes for first asset...")
    first_ticker = assets[0]['ticker']
    quotes = repo.get_quotes(first_ticker)

    print(f"   Ticker: {first_ticker}")
    print(f"   Quotes count: {len(quotes)}")

    if not quotes.empty:
        print(f"   Columns: {quotes.columns.tolist()}")
        print(f"   Date range: {quotes['date'].min()} to {quotes['date'].max()}")
        print(f"\n   Sample data:")
        print(quotes.head())
    else:
        print(f"   ⚠️ NO QUOTES FOUND for {first_ticker}!")

    # 3. Перевірка історії цін
    print("\n3. Checking price history...")
    tickers_sample = [a['ticker'] for a in assets[:3]]
    history = repo.get_price_history(tickers_sample)

    print(f"   Tickers: {tickers_sample}")
    print(f"   History shape: {history.shape}")
    if not history.empty:
        print(f"   Columns: {history.columns.tolist()}")
        print(f"\n   Sample:")
        print(history.head())

    print("\n✅ Diagnostic complete!")


if __name__ == "__main__":
    main()