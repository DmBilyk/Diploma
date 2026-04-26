import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict


class DataEngine:
    """
    Market data download module.
    Default setup: 30 years of history at a weekly interval.
    """

    def get_sp500_tickers(self) -> List[str]:
        """Parse S&P 500 tickers from Wikipedia with a browser-like request."""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            # Use a browser-like User-Agent so Wikipedia does not reject the request.
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            response = requests.get(url, headers=headers, verify=False, timeout=15)

            response.raise_for_status()  # Raise for HTTP errors such as 403 or 404.

            # Parse tables from the returned HTML page.
            tables = pd.read_html(response.text)
            df = tables[0]

            # Yahoo Finance uses dashes for tickers such as BRK.B -> BRK-B.
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()

            print(f"✅ Successfully parsed {len(tickers)} S&P 500 tickers.")
            return tickers

        except Exception as e:
            print(f"❌ Error fetching S&P 500: {e}")
            # Fallback universe if Wikipedia parsing fails.
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

            # Retry each batch up to three times with a short backoff.
            for attempt in range(3):
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
                    break  # Successful batch; stop retrying.

                except Exception as e:
                    if attempt < 2:
                        wait = 2 ** attempt  # 1s, then 2s.
                        print(f"   ⚠️ Batch attempt {attempt + 1}/3 failed: {e}. Retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"   ❌ Batch failed after 3 attempts ({batch[0]}...): {e}")

            time.sleep(1)

        return valid_data

    def _process_batch_result(self, data: pd.DataFrame, requested_tickers: List[str], min_length: int = 100) -> Dict[str, pd.DataFrame]:
        results = {}

        # 1. Single ticker downloads return a flat DataFrame.
        if len(requested_tickers) == 1:
            ticker = requested_tickers[0]
            if not data.empty:
                results[ticker] = self._clean_dataframe(data)
            return results

        # 2. Multi-ticker downloads return a MultiIndex column structure.
        if not isinstance(data.columns, pd.MultiIndex):
            return results

        for ticker in requested_tickers:
            try:
                if ticker in data.columns.levels[0]:
                    df_ticker = data[ticker].copy()

                    # Drop empty or too-short histories unless min_length allows them.
                    if len(df_ticker.dropna(how='all')) > min_length:
                        cleaned = self._clean_dataframe(df_ticker)
                        if not cleaned.empty:
                            results[ticker] = cleaned
            except (KeyError, AttributeError):
                continue

        return results

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalise a downloaded price DataFrame."""
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.rename(columns={"Adj Close": "Adj Close"})

        # Capture the last real trading date before forward-filling.
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        if price_col in df.columns:
            last_valid_idx = df[price_col].last_valid_index()

        # Forward-fill short market-holiday gaps.
        df = df.ffill()


        if price_col in df.columns and last_valid_idx is not None:
            end_of_data = df.index[-1]

            if (end_of_data - last_valid_idx).days > 14:
                dead_mask = df.index > last_valid_idx
                numeric_cols = df.select_dtypes(include='number').columns
                df.loc[dead_mask, numeric_cols] = 0.0001

        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)


        # Remove rows before the asset has real post-IPO adjusted-close data.
        if 'Adj Close' in df.columns:  # Use adjusted close as the validity criterion.
            df = df.dropna(subset=['Adj Close'])

        return df
