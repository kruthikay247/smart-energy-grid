"""
Shared cleaning + feature engineering.

IMPORTANT: this single module is used for
  1) building the QuickML training file   (prepare_training_data.py)
  2) the web backend                      (app.py)
  3) the grader script                    (predict.py)
so training and inference features are identical.

Only ROW-WISE features are used (no lags / rolling windows, no past demand).
The hidden test set is a split of the data, so neighbouring rows may not be
present or contiguous - row-wise features are safe in every case.
"""
import numpy as np
import pandas as pd

RAW_COLS = ["ambient_temp_c", "humidity_pct", "solar_irradiance_wm2"]

# Valid physical ranges used for validation / clipping
RANGES = {
    "ambient_temp_c": (-50.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "solar_irradiance_wm2": (0.0, 1400.0),
}

# Columns sent to QuickML (and used for training). Order matters for consistency.
FEATURE_COLS = [
    "ambient_temp_c", "humidity_pct", "solar_irradiance_wm2",
    "hour", "day_of_week", "is_weekend", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "cooling_deg", "heating_deg", "temp_x_humidity", "is_evening_peak",
]

FALLBACK_TS = pd.Timestamp("2024-01-01 12:00:00")


def clean_telemetry(df: pd.DataFrame, drop_invalid_ts: bool = True):
    """
    Validate + clean raw telemetry.
    Returns (clean_df, issues[list of human readable strings]).
    - drop_invalid_ts=True  : rows with unparsable timestamps are removed (web app)
    - drop_invalid_ts=False : rows are kept (predict.py must return 1 row per input)
    """
    issues = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column: timestamp")
    for c in RAW_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # --- timestamps
    parsed = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = int(parsed.isna().sum())
    if bad_ts:
        if drop_invalid_ts:
            issues.append(f"Dropped {bad_ts} row(s) with invalid timestamp.")
            df = df[~parsed.isna()].copy()
            parsed = parsed[~parsed.isna()]
        else:
            issues.append(f"{bad_ts} row(s) have invalid timestamp; default hour used.")
            parsed = parsed.fillna(FALLBACK_TS)
    df["_ts"] = parsed.values

    # --- numerics
    for c in RAW_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        lo, hi = RANGES[c]
        out_of_range = int(((df[c] < lo) | (df[c] > hi)).sum())
        if out_of_range:
            issues.append(f"{c}: {out_of_range} value(s) outside [{lo}, {hi}] were clipped.")
            df[c] = df[c].clip(lo, hi)

    # --- missing values: time interpolation -> ffill/bfill -> median
    if len(df) > 1:
        order = np.argsort(df["_ts"].values, kind="stable")
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        tmp = df.iloc[order].copy()
        for c in RAW_COLS:
            n_missing = int(tmp[c].isna().sum())
            if n_missing:
                issues.append(f"{c}: {n_missing} missing value(s) imputed.")
                tmp[c] = tmp[c].interpolate(limit_direction="both")
        df = tmp.iloc[inv]  # restore original row order
    for c in RAW_COLS:
        if df[c].isna().any():
            med = df[c].median()
            if np.isnan(med):  # whole column empty
                med = float(np.mean(RANGES[c]))
                issues.append(f"{c}: column empty, filled with {med:.1f}.")
            df[c] = df[c].fillna(med)

    df["timestamp"] = pd.to_datetime(df["_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S") \
        if drop_invalid_ts else df["timestamp"]
    return df.drop(columns=["_ts"]).reset_index(drop=True), issues


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all engineered columns in FEATURE_COLS."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"], errors="coerce").fillna(FALLBACK_TS)
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["month"] = ts.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["cooling_deg"] = (out["ambient_temp_c"] - 22.0).clip(lower=0)   # A/C load proxy
    out["heating_deg"] = (18.0 - out["ambient_temp_c"]).clip(lower=0)   # heater load proxy
    out["temp_x_humidity"] = out["ambient_temp_c"] * out["humidity_pct"] / 100.0
    out["is_evening_peak"] = out["hour"].between(18, 22).astype(int)
    return out
