# src/data_collection/macro_data.py
import os
from datetime import datetime, timedelta
import pandas as pd
import requests
from loguru import logger

MACRO_SERIES = {
    "dxy": "DTWEXBGS",
    "cpi": "CPIAUCSL",
    "fed_rate": "FEDFUNDS",
    "m2": "M2SL",
    "us10y": "DGS10",
}

def fetch_macro_data(years: int = 3) -> pd.DataFrame:
    """دریافت داده‌های کلان مستقیماً از FRED API با requests"""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    
    # بررسی صحت کلید (باید ۳۲ کاراکتر انگلیسی باشد)
    if len(api_key) != 32 or not api_key.isalnum():
        logger.warning("FRED_API_KEY نامعتبر است (باید ۳۲ کاراکتر انگلیسی باشد). داده ماکرو دریافت نمی‌شود.")
        return pd.DataFrame()

    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    macro_dfs = []

    for name, series_id in MACRO_SERIES.items():
        try:
            # ساخت آدرس و درخواست با requests
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series_id}&api_key={api_key}"
                f"&file_type=json&observation_start={start_date}"
            )
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            observations = data.get("observations", [])
            
            if not observations:
                logger.warning(f"داده‌ای برای {name} یافت نشد.")
                continue
                
            df = pd.DataFrame(observations)
            df = df[["date", "value"]].rename(columns={"value": name})
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df[name] = pd.to_numeric(df[name], errors="coerce")
            df = df.dropna()
            
            macro_dfs.append(df)
            logger.info(f"داده {name} دریافت شد: {len(df)} رکورد")
            
        except Exception as e:
            # جلوگیری از خطای ascii در لاگ کردن خطا
            logger.error(f"خطا در دریافت {name}: {type(e).__name__}")

    if not macro_dfs:
        return pd.DataFrame()

    # ادغام داده‌ها و پر کردن مقادیر گمشده
    macro_df = pd.concat(macro_dfs, axis=1).sort_index().ffill().dropna()
    logger.info(f"مجموع داده‌های ماکرو ادغام شد: {len(macro_df)} ردیف")
    return macro_df