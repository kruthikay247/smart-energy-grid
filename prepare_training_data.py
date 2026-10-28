"""
Turn train_telemetry.csv into train_features.csv for QuickML.

Usage:
    python prepare_training_data.py --input train_telemetry.csv --output train_features.csv

In QuickML: upload train_features.csv, choose target = target_demand_kw,
and REMOVE the 'timestamp' column from the feature list.
"""
import argparse
import pandas as pd
from features import clean_telemetry, add_features, FEATURE_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="train_telemetry.csv")
    ap.add_argument("--output", default="train_features.csv")
    a = ap.parse_args()

    raw = pd.read_csv(a.input)
    if "target_demand_kw" not in raw.columns:
        raise SystemExit("target_demand_kw column not found in training file")

    df, issues = clean_telemetry(raw, drop_invalid_ts=True)
    df["target_demand_kw"] = pd.to_numeric(df["target_demand_kw"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["target_demand_kw"])        # can't train on unlabeled rows
    df = add_features(df)

    cols = ["timestamp"] + FEATURE_COLS + ["target_demand_kw"]
    df[cols].to_csv(a.output, index=False)
    print(f"Wrote {a.output}: {len(df)} rows ({before - len(df)} dropped, no target)")
    for i in issues:
        print(" -", i)
    print("Features:", FEATURE_COLS)


if __name__ == "__main__":
    main()
