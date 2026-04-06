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

    def download_market_data(self, tickers: List[str], start_date: datetime = None, progress_callback=None) -> Dict[
        str, pd.DataFrame]:
        end_date = datetime.now()

        if start_date is None:
            actual_start_date = end_date - timedelta(days=30 * 365)
            chunk_size = 50
        else:
            actual_start_date = start_date
            chunk_size = 100

        start_str = actual_start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        INTERVAL = "1wk"

        valid_data = {}
        total_tickers = len(tickers)


        for i in range(0, total_tickers, chunk_size):
            batch = tickers[i: i + chunk_size]


            if progress_callback:

                current_pct = 10 + int((i / total_tickers) * 80)
                progress_callback(current_pct, f"Завантаження з Yahoo: {batch[0]}... ({i}/{total_tickers})")


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

                batch_results = self._process_batch_result(data, batch, min_length=0 if start_date else 100)
                valid_data.update(batch_results)

                time.sleep(1)

            except Exception as e:
                print(f"   ⚠️ Batch error: {e}")

        return valid_data

    def _process_batch_result(self, data: pd.DataFrame, requested_tickers: List[str], min_length: int = 100) -> Dict[str, pd.DataFrame]:
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

                    # Відкидаємо зовсім порожні або надто короткі історії (якщо мануально не вказано min_length)
                    if len(df_ticker.dropna(how='all')) > min_length:
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


        if price_col in df.columns and last_valid_idx is not None:
            end_of_data = df.index[-1]

            if (end_of_data - last_valid_idx).days > 14:
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