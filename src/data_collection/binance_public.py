# src/data_collection/binance_public.py
"""دریافت داده تاریخی از Binance Public Data (بدون محدودیت جغرافیایی)"""
import io
import zipfile
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


class BinancePublicData:
    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol.upper()
        self.interval = interval

    def _download_zip(self, url: str) -> pd.DataFrame:
        """دانلود و باز کردن فایل zip و حذف ردیف هدر احتمالی"""
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    df = pd.read_csv(f, header=None)
        except Exception as e:
            logger.debug(f"Failed {url}: {e}")
            return None

        if df.empty:
            return None

        # --- تشخیص و حذف ردیف هدر ---
        # اگر مقدار ستون اول قابل تبدیل به عدد نباشد، یعنی هدر است
        first_val = str(df.iloc[0, 0]).strip()
        try:
            int(float(first_val))
        except (ValueError, TypeError):
            df = df.iloc[1:].reset_index(drop=True)

        return df

    def fetch_monthly(self, year: int, month: int) -> pd.DataFrame:
        url = f"{BASE_URL}/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{year}-{month:02d}.zip"
        return self._download_zip(url)

    def fetch_historical(self, days: int = 180) -> pd.DataFrame:
        all_dfs = []
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        logger.info(
            f"Downloading {self.symbol} {self.interval} "
            f"from {start_date.date()} to {end_date.date()}"
        )

        current = start_date.replace(day=1)
        while current <= end_date:
            df = self.fetch_monthly(current.year, current.month)
            if df is not None and not df.empty:
                all_dfs.append(df)
                logger.info(f"  + {current.year}-{current.month:02d}: {len(df)} rows")
            else:
                logger.debug(f"  - {current.year}-{current.month:02d}: not available")

            # ماه بعد
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        if not all_dfs:
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)

        # تنظیم نام ستون‌ها بر اساس تعداد
        if combined.shape[1] >= 12:
            combined = combined.iloc[:, :12]
            combined.columns = COLUMNS
        elif combined.shape[1] >= 6:
            combined = combined.iloc[:, :6]
            combined.columns = COLUMNS[:6]
        else:
            logger.warning(f"Unexpected column count: {combined.shape[1]}")
            return pd.DataFrame()

        # --- تشخیص واحد زمان: میلی‌ثانیه یا میکروثانیه ---
        ts_sample = pd.to_numeric(combined["timestamp"], errors="coerce").dropna()
        if ts_sample.empty:
            logger.error("All timestamps invalid")
            return pd.DataFrame()

        first_ts = float(ts_sample.iloc[0])
        if first_ts > 1e14:
            unit = "us"  # میکروثانیه (فرمت جدید Binance)
            logger.debug(f"Detected microsecond timestamps (sample={first_ts})")
        else:
            unit = "ms"  # میلی‌ثانیه (فرمت قدیمی)
            logger.debug(f"Detected millisecond timestamps (sample={first_ts})")

        # تبدیل امن به datetime با errors="coerce" تا ردیف‌های خراب حذف شوند
        combined["timestamp"] = pd.to_datetime(
            combined["timestamp"], unit=unit, utc=True, errors="coerce"
        )
        combined = combined.dropna(subset=["timestamp"])

        combined = combined.set_index("timestamp")
        for col in ["open", "high", "low", "close", "volume"]:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined = combined[["open", "high", "low", "close", "volume"]].dropna()

        combined = combined[~combined.index.duplicated(keep="last")].sort_index()

        # فیلتر بازه زمانی
        combined = combined[combined.index >= start_date]

        logger.info(
            f"Total: {len(combined)} candles for {self.symbol} {self.interval} "
            f"({combined.index.min()} -> {combined.index.max()})"
        )
        return combined