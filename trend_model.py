"""
trend_model.py — XGBoost + STL trend forecasting pipeline

Pipeline:
  1. STL decompose 60-month series → trend + seasonal + residual
  2. Classify product type (seasonal/trending_up/trending_down/stable)
  3. Train XGBoost on deseasonalized signal
  4. Forecast 6 months on deseasonalized signal
  5. Re-add seasonal component → final forecast
  6. Apply type-aware correction for direction consistency
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.seasonal import STL


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _clean(values: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf (STL edge artifacts) via linear interpolation."""
    s = pd.Series(values.copy()).replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.fillna(float(np.nanmean(raw)))
    return s.values


# ─────────────────────────────────────────────
# 1. STL DECOMPOSITION
# ─────────────────────────────────────────────

def decompose(series: pd.Series):
    """
    Decompose monthly series into trend + seasonal + residual.

    seasonal=13: smoothing window (must be odd, ≥ period+2)
    trend=25: longer smoother → cleaner trend, less noise leakage
    robust=True: handles outliers & zero-heavy series (holiday products)
    """
    stl = STL(series, period=12, seasonal=13, trend=25, robust=True)
    res = stl.fit()
    raw = series.values
    return (
        _clean(res.trend,    raw),
        _clean(res.seasonal, raw),
        _clean(res.resid,    raw),
    )


# ─────────────────────────────────────────────
# 2. CLASSIFICATION
# ─────────────────────────────────────────────

def classify(trend: np.ndarray, seasonal: np.ndarray, raw: np.ndarray) -> str:
    """
    Classify product trend type using STL components.

    seasonality_strength = var(seasonal) / (var(seasonal) + var(trend))
    trend_slope_pct = total % change trend makes over full period

    Extra rule: zero-spike pattern (holiday/festival products like
    'christmas candles') → always classify as seasonal.
    """
    sv = np.var(seasonal)
    tv = np.var(trend)
    ss = sv / (sv + tv + 1e-9)

    x = np.arange(len(trend))
    slope, _ = np.polyfit(x, trend, 1)
    slope_pct = (slope * len(trend)) / (np.mean(np.abs(trend)) + 1e-9)

    zero_ratio = float(np.mean(raw < 3))
    max_val    = float(np.max(raw))

    print(f"[Classify] ss={ss:.3f} slope={slope_pct:.3f} zeros={zero_ratio:.2f} max={max_val:.1f}")

    # Holiday/festival products: flat near zero, then a spike
    if zero_ratio > 0.45 and max_val > 8:
        return "seasonal"
    if ss > 0.25:
        return "seasonal"
    if slope_pct > 0.25:
        return "trending_up"
    if slope_pct < -0.25:
        return "trending_down"
    return "stable"


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

FEATURES = [
    "val", "month", "month_idx",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12",
    "slope_3", "slope_6",
    "mean_3", "mean_6", "mean_12",
    "std_6", "pct_chg",
    "sin1", "cos1", "sin2", "cos2",
]


def make_features(vals: np.ndarray, months: np.ndarray, indices: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"val": vals, "month": months, "month_idx": indices})

    # Autoregressive lags (core signal)
    df["lag_1"]  = df["val"].shift(1)
    df["lag_2"]  = df["val"].shift(2)
    df["lag_3"]  = df["val"].shift(3)
    df["lag_6"]  = df["val"].shift(6)
    df["lag_12"] = df["val"].shift(12)   # same month last year — key for seasonals

    # Momentum
    df["slope_3"] = df["val"].diff(3)
    df["slope_6"] = df["val"].diff(6)

    # Rolling context
    df["mean_3"]  = df["val"].rolling(3,  min_periods=1).mean()
    df["mean_6"]  = df["val"].rolling(6,  min_periods=1).mean()
    df["mean_12"] = df["val"].rolling(12, min_periods=1).mean()
    df["std_6"]   = df["val"].rolling(6,  min_periods=1).std().fillna(0)
    df["pct_chg"] = df["val"].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # Fourier encoding of month (cyclic seasonality)
    for k, n in [(1, "1"), (2, "2")]:
        df[f"sin{n}"] = np.sin(2 * np.pi * k * df["month"] / 12)
        df[f"cos{n}"] = np.cos(2 * np.pi * k * df["month"] / 12)

    return df.bfill().ffill().fillna(0)


# ─────────────────────────────────────────────
# 4. XGBOOST TRAINING
# ─────────────────────────────────────────────

XGB_PARAMS = {
    "seasonal":      dict(n_estimators=600, learning_rate=0.025, max_depth=4,
                          subsample=0.75, colsample_bytree=0.75, min_child_weight=5,
                          reg_alpha=0.1, reg_lambda=1.5),
    "trending_up":   dict(n_estimators=700, learning_rate=0.02,  max_depth=5,
                          subsample=0.80, colsample_bytree=0.80, min_child_weight=3,
                          reg_alpha=0.05, reg_lambda=1.0),
    "trending_down": dict(n_estimators=700, learning_rate=0.02,  max_depth=5,
                          subsample=0.80, colsample_bytree=0.80, min_child_weight=3,
                          reg_alpha=0.05, reg_lambda=1.0),
    "stable":        dict(n_estimators=400, learning_rate=0.04,  max_depth=3,
                          subsample=0.80, colsample_bytree=0.80, min_child_weight=5,
                          reg_alpha=0.1,  reg_lambda=2.0),
}


def train_xgb(df: pd.DataFrame, product_type: str) -> xgb.XGBRegressor:
    """Train XGBoost with next-month as target (shift -1)."""
    df = df.copy()
    df["target"] = df["val"].shift(-1)
    df = df.bfill().ffill().fillna(0)
    X = df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)
    y = df["target"].fillna(df["val"].mean())

    params = XGB_PARAMS.get(product_type, XGB_PARAMS["stable"])
    model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────
# 5. SINGLE-STEP PREDICTION
# ─────────────────────────────────────────────

def predict_one(model, window: list, month: int, idx: int) -> float:
    """
    window: rolling history, newest first, len >= 12 preferred.
    """
    def g(i): return float(window[i]) if i < len(window) else float(window[-1])

    v0, v1, v2, v3, v5, v11 = g(0), g(1), g(2), g(3), g(5), g(11)
    w6  = window[:6]  if len(window) >= 6  else window
    w12 = window[:12] if len(window) >= 12 else window

    row = pd.DataFrame([[
        v0, month, idx,
        v1, v2, v3, v5, v11,
        v0 - v3, v0 - v5,
        np.mean(window[:3]) if len(window) >= 3 else v0,
        np.mean(w6), np.mean(w12),
        float(np.std(w6)), (v0 - v1) / (abs(v1) + 1e-9),
        np.sin(2*np.pi*1*month/12), np.cos(2*np.pi*1*month/12),
        np.sin(2*np.pi*2*month/12), np.cos(2*np.pi*2*month/12),
    ]], columns=FEATURES)
    return float(model.predict(row)[0])


# ─────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────

def train_trend_predictor(full_data: pd.DataFrame, display_data: pd.DataFrame):
    """
    Args:
        full_data:    Up to 60 months of data (training)
        display_data: Last 24 months (shown on chart, used for seeding forecast)

    Returns:
        List of 6 forecast dicts: {month, value, product_type}
    """
    # ── Prepare ───────────────────────────────────────────────────────
    df = full_data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["month"] = df.index.month
    series = df["search_volume"]

    # ── STL ───────────────────────────────────────────────────────────
    trend, seasonal, resid = decompose(series)

    # ── Classify ──────────────────────────────────────────────────────
    product_type = classify(trend, seasonal, series.values)
    print(f"[TrendModel] product_type={product_type}")

    # ── Seasonal lookup: avg component per calendar month ─────────────
    months_arr = df["month"].values
    seas_by_month = {}
    for m in range(1, 13):
        mask = months_arr == m
        seas_by_month[m] = float(np.mean(seasonal[mask])) if mask.any() else 0.0

    # ── Deseasonalize, train ──────────────────────────────────────────
    deseas = _clean(trend + resid, series.values)
    feat_df = make_features(deseas, months_arr, np.arange(len(df)))
    model = train_xgb(feat_df, product_type)

    # ── Seed rolling window from display data ─────────────────────────
    disp = display_data.copy()
    disp["date"] = pd.to_datetime(disp["date"])
    disp = disp.set_index("date")
    disp["month"] = disp.index.month

    disp_deseas = np.array([
        row["search_volume"] - seas_by_month.get(int(row["month"]), 0.0)
        for _, row in disp.iterrows()
    ])
    disp_deseas = np.nan_to_num(disp_deseas, nan=float(series.mean()))

    # Build rolling window (newest first, length 24)
    window = list(reversed(disp_deseas[-24:])) if len(disp_deseas) >= 24 \
             else list(reversed(disp_deseas))
    while len(window) < 24:
        window.append(window[-1])

    cur_month    = int(disp["month"].iloc[-1])
    base_idx     = len(full_data)
    recent_avg   = float(np.mean(disp_deseas[-6:]))
    recent_slope = float((disp_deseas[-1] - disp_deseas[-6]) / 5) \
                   if len(disp_deseas) >= 6 else 0.0

    # ── Forecast 6 months ─────────────────────────────────────────────
    forecast = []
    for i in range(1, 7):
        pred_month = (cur_month + i - 1) % 12 + 1

        xgb_pred = predict_one(model, window, pred_month, base_idx + i)

        # Type-aware blending (XGBoost can't extrapolate trends well)
        if product_type == "stable":
            xgb_pred = xgb_pred * 0.40 + recent_avg * 0.60

        elif product_type == "trending_up":
            momentum = recent_avg + recent_slope * i * 0.6
            xgb_pred = max(xgb_pred * 0.50 + momentum * 0.50, recent_avg * 0.85)

        elif product_type == "trending_down":
            momentum = recent_avg + recent_slope * i * 0.6
            xgb_pred = min(xgb_pred * 0.50 + momentum * 0.50, recent_avg * 1.15)

        # Re-add seasonal component
        final = xgb_pred + seas_by_month.get(pred_month, 0.0)

        # Smooth the first-month transition
        if i == 1:
            last_real = float(display_data["search_volume"].iloc[-1])
            final = last_real * 0.30 + final * 0.70

        val = round(max(0.0, final), 2)
        forecast.append({"month": i, "value": val, "product_type": product_type})

        # Update rolling window
        window.insert(0, xgb_pred)
        if len(window) > 36:
            window.pop()

    return forecast