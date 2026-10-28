"""
HackSphere predict.py - satisfies BOTH contracts in the starter kit:

  1) function:  from predict import predict ; predict(input_df) -> numpy array
     (this is what test_eval.py and the grader import)
  2) CLI:       python predict.py --input sample_test_input.csv --output predictions.csv
     (writes timestamp,target_demand_kw - one row per input row, timestamps preserved)

Set your published QuickML URL below, or via env var QUICKML_ENDPOINT (env wins).
Optional auth env vars are described in quickml_client.py. No secret belongs in this file.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from features import clean_telemetry, add_features
from quickml_client import predict_features, PLACEHOLDER_URL

QUICKML_ENDPOINT_URL = "YOUR_PUBLISHED_QUICKML_ENDPOINT_URL_HERE"


def _dummy(df: pd.DataFrame) -> np.ndarray:
    """Starter-kit dummy formula. ONLY used when ALLOW_DUMMY_FALLBACK=1 (local dev)."""
    hours = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour.fillna(12)
    base = 120.0 + 30.0 * np.sin(2 * np.pi * (hours - 6) / 24)
    cool = np.maximum(0.0, df["ambient_temp_c"].astype(float) - 24.0) * 8.5
    return (base + cool).to_numpy(dtype=float)


def predict(input_df: pd.DataFrame):
    """
    input_df columns: timestamp, ambient_temp_c, humidity_pct, solar_irradiance_wm2
    Returns a numpy array with exactly len(input_df) predicted demand values (kW).
    """
    url = os.getenv("QUICKML_ENDPOINT") or QUICKML_ENDPOINT_URL
    # keep every row (never drop) so output length always equals input length
    clean, issues = clean_telemetry(input_df, drop_invalid_ts=False)
    for i in issues:
        print("[warn]", i, file=sys.stderr)
    feats = add_features(clean)
    try:
        preds = np.array(predict_features(feats, url=url), dtype=float)
    except Exception as e:                                     # noqa: BLE001
        if os.getenv("ALLOW_DUMMY_FALLBACK") == "1":
            print(f"[warn] QuickML failed ({e}); using DUMMY fallback (dev only).", file=sys.stderr)
            return _dummy(clean)
        raise
    if len(preds) != len(input_df) or not np.isfinite(preds).all():
        raise RuntimeError("QuickML returned wrong number of predictions or NaN/inf values")
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="sample_test_input.csv")
    ap.add_argument("--output", default="predictions.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    preds = predict(df)
    pd.DataFrame({"timestamp": df["timestamp"].astype(str),
                  "target_demand_kw": np.round(preds, 2)}).to_csv(a.output, index=False)
    print(f"Wrote {len(preds)} predictions to {a.output}")


if __name__ == "__main__":
    main()
