"""
InsideAI Backend — main.py
Data source: SerpApi Google Trends API (replaces pytrends)

Why SerpApi?
- pytrends was archived April 2025 and Google actively rate-limits it
- SerpApi bypasses all blocks/CAPTCHAs and returns clean JSON
- 100 free searches/month — enough for FYP demo and testing
- ~2.5s per request vs pytrends timing out or returning empty

Setup:
  pip install google-search-results
  Set SERPAPI_KEY in your environment OR paste it directly in the config below.
"""

import os
import re
import unicodedata
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from serpapi import GoogleSearch

from trend_model import train_trend_predictor

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Option A: set environment variable:  set SERPAPI_KEY=your_key_here  (Windows)
# Option B: paste your key directly below (for local dev only)
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "fd64b5c69870ede36208d69471d11af63b00d36be7e3f47fd8587363bde9860d")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SUPPORTED COUNTRIES ───────────────────────────────────────────────────────
COUNTRIES = [
    {"code": "worldwide", "label": "🌍 Worldwide"},
    {"code": "US", "label": "🇺🇸 United States"},
    {"code": "GB", "label": "🇬🇧 United Kingdom"},
    {"code": "PK", "label": "🇵🇰 Pakistan"},
    {"code": "IN", "label": "🇮🇳 India"},
    {"code": "CA", "label": "🇨🇦 Canada"},
    {"code": "AU", "label": "🇦🇺 Australia"},
    {"code": "DE", "label": "🇩🇪 Germany"},
    {"code": "FR", "label": "🇫🇷 France"},
    {"code": "BR", "label": "🇧🇷 Brazil"},
    {"code": "MX", "label": "🇲🇽 Mexico"},
    {"code": "SA", "label": "🇸🇦 Saudi Arabia"},
    {"code": "AE", "label": "🇦🇪 UAE"},
    {"code": "NG", "label": "🇳🇬 Nigeria"},
    {"code": "ZA", "label": "🇿🇦 South Africa"},
    {"code": "JP", "label": "🇯🇵 Japan"},
    {"code": "KR", "label": "🇰🇷 South Korea"},
    {"code": "ID", "label": "🇮🇩 Indonesia"},
    {"code": "PH", "label": "🇵🇭 Philippines"},
    {"code": "TR", "label": "🇹🇷 Turkey"},
    {"code": "EG", "label": "🇪🇬 Egypt"},
    {"code": "IT", "label": "🇮🇹 Italy"},
    {"code": "ES", "label": "🇪🇸 Spain"},
    {"code": "NL", "label": "🇳🇱 Netherlands"},
    {"code": "SE", "label": "🇸🇪 Sweden"},
    {"code": "AR", "label": "🇦🇷 Argentina"},
    {"code": "CO", "label": "🇨🇴 Colombia"},
    {"code": "BD", "label": "🇧🇩 Bangladesh"},
    {"code": "MY", "label": "🇲🇾 Malaysia"},
    {"code": "TH", "label": "🇹🇭 Thailand"},
    {"code": "VN", "label": "🇻🇳 Vietnam"},
]
COUNTRY_CODES = {c["code"].upper(): c["code"] for c in COUNTRIES}
COUNTRY_CODES["WORLDWIDE"] = ""


# ── KEYWORD CLEANING ──────────────────────────────────────────────────────────
def clean_keyword(raw: str) -> str:
    """
    Normalize user input before sending to Google Trends.
    Google Trends is case-insensitive but we standardize to lowercase
    for consistent caching and API calls.
    """
    if not raw:
        return ""
    kw = unicodedata.normalize("NFKC", raw).strip()
    kw = re.sub(r"[^\w\s\-]", " ", kw)   # keep alphanumeric, spaces, hyphens
    kw = re.sub(r"\s+", " ", kw).strip()
    return kw.lower()


def display_keyword(cleaned: str) -> str:
    return cleaned.title()


# ── SERPAPI FETCH ─────────────────────────────────────────────────────────────
def fetch_trends_serpapi(keyword: str, geo: str = "") -> pd.DataFrame:
    """
    Fetch 5-year monthly Google Trends data via SerpApi.

    SerpApi returns weekly data for 5-year timeframe.
    We resample to monthly (Month Start) for our model.

    geo = "" → Worldwide
    geo = "US", "PK", "GB", etc. → specific country

    Returns DataFrame with columns: [date (index), keyword]
    or empty DataFrame on failure.
    """
    params = {
        "engine": "google_trends",
        "q": keyword,
        "date": "today 5-y",          # 5 years of history
        "data_type": "TIMESERIES",    # Interest over time
        "tz": "0",                    # UTC timezone for consistency
        "api_key": SERPAPI_KEY,
    }
    if geo:  # empty string = worldwide (don't pass geo param at all)
        params["geo"] = geo

    print(f"[SerpApi] Fetching: keyword='{keyword}' geo='{geo or 'WW'}'")

    try:
        search = GoogleSearch(params)
        result = search.get_dict()

        # SerpApi error handling
        if "error" in result:
            print(f"  [SerpApi] API error: {result['error']}")
            return pd.DataFrame()

        timeline = result.get("interest_over_time", {}).get("timeline_data", [])
        if not timeline:
            print(f"  [SerpApi] No timeline_data in response")
            return pd.DataFrame()

        # Parse the timeline into a DataFrame
        rows = []
        for point in timeline:
            timestamp = int(point.get("timestamp", 0))
            if timestamp == 0:
                continue
            dt = datetime.utcfromtimestamp(timestamp)
            values = point.get("values", [])
            if not values:
                continue
            val = values[0].get("extracted_value", 0)
            rows.append({"date": dt, keyword: val})

        if not rows:
            print(f"  [SerpApi] Parsed 0 rows")
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)

        # Resample weekly → monthly (Month Start)
        monthly = df[[keyword]].resample("MS").mean().dropna()
        print(f"  [SerpApi] ✅ Got {len(monthly)} monthly data points")
        return monthly

    except Exception as e:
        print(f"  [SerpApi] Exception: {type(e).__name__}: {e}")
        return pd.DataFrame()


def fetch_with_geo_fallback(keyword: str, geo: str) -> tuple:
    """
    Fetch data for the requested geo. If data is insufficient
    (< 12 data points or max value < 5), automatically fall back to Worldwide.

    Returns (DataFrame, geo_label_string)
    """
    print(f"\n{'='*55}")
    print(f"[API] keyword='{keyword}' geo='{geo or 'WW'}'")

    data = fetch_trends_serpapi(keyword, geo=geo)

    # Check quality of result
    is_good = (
        not data.empty
        and len(data) >= 12
        and data[keyword].max() >= 5
    )

    if is_good:
        label = geo if geo else "Worldwide"
        return data, label

    # Fallback to worldwide if country had insufficient data
    if geo != "":
        print(f"[API] Insufficient data for geo='{geo}', falling back to Worldwide...")
        ww_data = fetch_trends_serpapi(keyword, geo="")
        if not ww_data.empty and len(ww_data) >= 12:
            return ww_data, f"Worldwide (no data for {geo})"

    # Return whatever we got (even if thin)
    label = geo if geo else "Worldwide"
    return data, label


# ── QUALITY CHECK ─────────────────────────────────────────────────────────────
def check_quality(df: pd.DataFrame, keyword: str) -> str | None:
    """Return an error message if data is unusable, else None."""
    if df.empty:
        return f"No data found for '{keyword}'."

    max_val = df["search_volume"].max()
    mean_val = df["search_volume"].mean()

    if max_val < 5 and mean_val < 2:
        return (
            f"'{keyword}' has very little search data in this region. "
            f"Try a broader term (e.g. 'candle' not 'christmas candle') "
            f"or switch to Worldwide."
        )
    return None


# ── MAIN ENDPOINT ─────────────────────────────────────────────────────────────
@app.get("/api/predict/{keyword}")
async def get_prediction(keyword: str, geo: str = ""):
    """
    Main prediction endpoint.
    keyword: product name (auto-cleaned)
    geo: country code ("US", "PK", "") or "worldwide"
    """
    try:
        # 1. Clean keyword
        cleaned = clean_keyword(keyword)
        if not cleaned:
            return {"status": "error", "message": "Please enter a valid product name."}
        display = display_keyword(cleaned)

        # 2. Resolve geo
        geo_key = geo.upper()
        geo_code = "" if geo_key in ("", "WORLDWIDE") else geo.upper()

        # 3. Fetch data (with auto geo-fallback)
        raw_data, geo_label = fetch_with_geo_fallback(cleaned, geo_code)

        if raw_data.empty:
            return {
                "status": "error",
                "message": (
                    f"No data returned for '{display}'. "
                    f"Check spelling, try a broader term, or select Worldwide."
                ),
            }

        # 4. Build training set (up to 60 months)
        full_df = raw_data.tail(60).reset_index()
        full_df.columns = ["date", "search_volume"]
        full_df["search_volume"] = full_df["search_volume"].round(2)

        # 5. Display set (last 24 months)
        display_df = full_df.tail(24).reset_index(drop=True)

        # 6. Quality gate
        err = check_quality(display_df, display)
        if err:
            return {"status": "low_quality", "message": err, "keyword": display}

        # 7. Run forecasting model
        forecast = train_trend_predictor(full_df, display_df)

        # Detect if geo fallback was used
        fallback_used = "no data for" in geo_label.lower()

        return {
            "keyword": display,
            "keyword_raw": cleaned,
            "geo": geo_label,
            "geo_fallback": fallback_used,
            "historical": display_df.to_dict(orient="records"),
            "forecast": forecast,
            "product_type": forecast[0]["product_type"] if forecast else "unknown",
            "status": "success",
        }

    except Exception as e:
        print(f"[Server Error] {e}")
        import traceback; traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ── HELPER ENDPOINTS ──────────────────────────────────────────────────────────
@app.get("/api/countries")
async def get_countries():
    return {"countries": COUNTRIES}


@app.get("/api/health")
async def health():
    key_configured = SERPAPI_KEY != "PASTE_YOUR_SERPAPI_KEY_HERE"
    return {
        "status": "ok",
        "serpapi_key_set": key_configured,
        "warning": None if key_configured else "SerpApi key not configured! Edit main.py or set SERPAPI_KEY env var.",
    }