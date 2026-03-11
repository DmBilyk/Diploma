import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict


class DataEngine:
    """
    Модуль завантаження даних.
    Налаштування: Суворо 30 років історії, інтервал - 1 тиждень.
    """

    def get_sp500_tickers(self) -> List[str]:
        """Парсить тікери S&P 500 з Вікіпедії (з обходом захисту від ботів)."""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            # Додаємо User-Agent, щоб Вікіпедія не блокувала запит (Error 403)
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Перевірка на помилки (404, 403 тощо)

            # Парсимо HTML-текст відповіді
            tables = pd.read_html(response.text)
            df = tables[0]

            # Замінюємо крапки на дефіси (наприклад BRK.B -> BRK-B)
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()

            print(f"✅ Successfully parsed {len(tickers)} S&P 500 tickers.")
            return tickers

        except Exception as e:
            print(f"❌ Error fetching S&P 500: {e}")
            # Резервний список, якщо парсинг не вдався
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "PG"]

    def download_market_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Завантажує дані за останні 30 років з тижневим інтервалом.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * 365)

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        INTERVAL = "1wk"

        print(f"🚀 Config: 30 Years ({start_str} to {end_str}), Interval: {INTERVAL}")

        chunk_size = 50
        valid_data = {}

        # Пакетне завантаження
        for i in range(0, len(tickers), chunk_size):
            batch = tickers[i: i + chunk_size]
            print(f"   ⬇️ Batch {i // chunk_size + 1}: {batch[:3]}... ({len(batch)} items)")

            try:
                data = yf.download(
                    batch,
                    start=start_str,
                    end=end_str,
                    interval=INTERVAL,
                    group_by='ticker',
                    auto_adjust=False,
                    threads=True,
                    progress=False
                )

                batch_results = self._process_batch_result(data, batch)
                valid_data.update(batch_results)

                time.sleep(1)

            except Exception as e:
                print(f"   ⚠️ Batch error: {e}")

        print(f"🏁 Downloaded {len(valid_data)} assets.")
        return valid_data

    def _process_batch_result(self, data: pd.DataFrame, requested_tickers: List[str]) -> Dict[str, pd.DataFrame]:
        results = {}

        # 1. Один тікер (плоский DataFrame)
        if len(requested_tickers) == 1:
            ticker = requested_tickers[0]
            if not data.empty:
                results[ticker] = self._clean_dataframe(data)
            return results

        # 2. Багато тікерів (MultiIndex)
        for ticker in requested_tickers:
            try:
                if ticker in data.columns.levels[0]:
                    df_ticker = data[ticker].copy()

                    # Відкидаємо зовсім порожні або надто короткі історії (< 2 років даних)
                    if len(df_ticker.dropna(how='all')) > 100:
                        cleaned = self._clean_dataframe(df_ticker)
                        if not cleaned.empty:
                            results[ticker] = cleaned
            except KeyError:
                continue

        return results

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очищення та нормалізація даних"""
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.rename(columns={"Adj Close": "Adj Close"})

        # Визначаємо останню реальну дату торгів ДО ffill
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if price_col in df.columns:
            last_valid_idx = df[price_col].last_valid_index()

        # Протягуємо вперед (якщо були свята)
        df = df.ffill()

        # Перевірка на "мертвий" актив (банкрутство / делістинг):
        # якщо остання реальна дата торгів раніше за кінець діапазону даних,
        # заповнюємо всі дні ПІСЛЯ неї мінімальним значенням (≈0),
        # щоб уникнути помилок ділення на нуль в оптимізаторі.
        if price_col in df.columns and last_valid_idx is not None:
            end_of_data = df.index[-1]
            if last_valid_idx < end_of_data:
                dead_mask = df.index > last_valid_idx
                numeric_cols = df.select_dtypes(include='number').columns
                df.loc[dead_mask, numeric_cols] = 0.0001

        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)

        # ❌ ВИДАЛЕНО: df = df.bfill()  <-- ЦЕ БУЛО ЗЛО
        # Ми не маємо вигадувати дані в минулому!

        # Видаляємо NaN (тобто всі дати ДО моменту реального IPO)
        if 'Adj Close' in df.columns:  # Використовуємо Adj Close як критерій
            df = df.dropna(subset=['Adj Close'])

        return df