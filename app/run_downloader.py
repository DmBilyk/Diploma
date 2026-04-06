import ssl
import time
from app.data.data_engine import DataEngine
from app.data.repository import PortfolioRepository

# --- FIX FOR MACOS SSL ERROR ---
# Це дозволяє Python качати дані з Вікіпедії без перевірки сертифікатів
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# -------------------------------

def main():
    print("=== PORTFOLIO OPTIMIZER DATA LOADER ===")

    # 1. Ініціалізація
    engine = DataEngine()
    repo = PortfolioRepository()

    # 2. Отримання списку тікерів
    print("\n[Step 1] Fetching S&P 500 tickers list...")
    tickers = engine.get_sp500_tickers()

    # Додаємо додаткові активи для різноманіття та стрес-тестів
    additional = [
        "BTC-USD", "ETH-USD", "GC=F",
        # "Зомбі-акції" (катастрофічні падіння) для перевірки алгоритмів
        "WBA", "NKLA", "PTON", "LUMN",
        # Захисні класи активів (ETF)
        "TLT", "AGG", "VNQ", "USO",
    ]
    tickers.extend(additional)
    tickers = list(set(tickers))

    print(f"Target: {len(tickers)} assets.")

    # 2.5 Перевірка наявності даних для дельта-оновлення
    print("\n[Step 2.5] Checking latest recorded data for delta update...")
    from datetime import datetime as dt, timedelta
    latest_date = repo.get_latest_quote_date()
    
    start_date = None
    if latest_date:
        if isinstance(latest_date, dt):
            start_date = latest_date - timedelta(days=7)
        else:
            start_date = dt(latest_date.year, latest_date.month, latest_date.day) - timedelta(days=7)
        print(f"   Found existing data up to {latest_date}. Preparing Delta Update.")
    else:
        print("   Database is empty or missing quotes. Preparing full 30-year sync.")

    # 3. Завантаження даних
    if start_date:
        print(f"\n[Step 3] Downloading historical data (Delta Update from {start_date.strftime('%Y-%m-%d')}, Weekly)...")
    else:
        print("\n[Step 3] Downloading historical data (30 years, Weekly)...")
        
    market_data = engine.download_market_data(tickers, start_date=start_date)

    # 4. Збереження в БД
    print(f"\n[Step 4] Saving {len(market_data)} assets to SQLite...")

    start_time = time.time()
    counter = 0

    for ticker, df in market_data.items():
        # Спочатку додаємо сам актив
        repo.add_asset(ticker, name=ticker, sector="Unknown")

        # Потім пишемо ціни
        repo.save_quotes_bulk(ticker, df)

        counter += 1
        if counter % 50 == 0:
            print(f"   Saved {counter} assets...")

    duration = time.time() - start_time
    print(f"\n✅ DONE! Database updated in {duration:.2f} seconds.")

    # Перевірка результату
    all_tickers = repo.get_all_tickers()
    print(f"Total assets in DB: {len(all_tickers)}")


if __name__ == "__main__":
    main()