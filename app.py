"""
Flask backend + static frontend, deployable on Catalyst AppSail.

Routes
  GET  /                -> dashboard (static/index.html)
  GET  /api/health
  POST /api/forecast    -> JSON {"rows":[...]}  or multipart file field "file" (CSV)
  GET  /api/history     -> last saved forecast runs (Data Store, else in-memory)
"""
import io
import os
import time
import uuid

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from controller import run_controller, STORAGE, GRID
from features import clean_telemetry, add_features, RAW_COLS
from quickml_client import predict_features

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024       # 5 MB uploads

HISTORY = []            # in-memory fallback log
DATASTORE_TABLE = os.getenv("DATASTORE_TABLE", "ForecastRuns")


# ----------------------------------------------------------------- persistence
def persist(run: dict, csv_bytes: bytes = None):
    """Best-effort save to Catalyst Data Store (+ Stratus for CSV). Never breaks the request."""
    HISTORY.append(run)
    del HISTORY[:-50]
    try:
        import zcatalyst_sdk
        capp = zcatalyst_sdk.initialize(req=request)
        capp.datastore().table(DATASTORE_TABLE).insert_row({
            "run_id": run["run_id"],
            "created": run["created"],
            "peak_kw": run["summary"]["peak_demand_kw"],
            "overloads": run["summary"]["overload_hours"],
            "payload": str(run["rows"])[:60000],
        })
        bucket = os.getenv("STRATUS_BUCKET")
        if bucket and csv_bytes:
            capp.stratus().bucket(bucket).put_object(f"uploads/{run['run_id']}.csv", csv_bytes)
    except Exception as e:            # noqa: BLE001
        run.setdefault("logs", []).append(f"Persistence skipped: {type(e).__name__}")


# ----------------------------------------------------------------------- routes
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", quickml_configured=bool(os.getenv("QUICKML_ENDPOINT")))


@app.get("/api/history")
def history():
    return jsonify([{"run_id": h["run_id"], "created": h["created"], "summary": h["summary"]}
                    for h in reversed(HISTORY)])


@app.post("/api/forecast")
def forecast():
    logs, csv_bytes = [], None
    try:
        if "file" in request.files:
            csv_bytes = request.files["file"].read()
            raw = pd.read_csv(io.BytesIO(csv_bytes))
            logs.append(f"CSV received: {len(raw)} row(s)")
        else:
            payload = request.get_json(silent=True) or {}
            rows = payload.get("rows")
            if not rows:
                return jsonify(error="Send CSV file or JSON {'rows': [...]}"), 400
            raw = pd.DataFrame(rows)
            logs.append(f"Form data received: {len(raw)} row(s)")

        clean, issues = clean_telemetry(raw, drop_invalid_ts=True)
        logs += issues
        if clean.empty:
            return jsonify(error="No valid rows after cleaning", logs=logs), 400

        clean = clean.sort_values("timestamp").reset_index(drop=True)
        if len(clean) > 24:
            logs.append(f"{len(clean)} rows supplied; using first 24 (forecast window).")
            clean = clean.head(24)
        elif len(clean) < 24:
            logs.append(f"Only {len(clean)} row(s) supplied; a full forecast uses 24.")

        feats = add_features(clean)
        t0 = time.time()
        preds = predict_features(feats)
        logs.append(f"QuickML returned {len(preds)} prediction(s) in {time.time()-t0:.1f}s")

        rows_in = [{
            "timestamp": clean.loc[i, "timestamp"],
            "predicted_demand_kw": round(float(preds[i]), 2),
            **{c: float(clean.loc[i, c]) for c in RAW_COLS},
        } for i in range(len(clean))]

        soc0 = float(request.args.get("initial_soc", 80))
        result = run_controller(rows_in, initial_soc=max(0.0, min(100.0, soc0)))

        summary = {
            "hours": len(result),
            "peak_demand_kw": max(r["predicted_demand_kw"] for r in result),
            "avg_demand_kw": round(sum(r["predicted_demand_kw"] for r in result) / len(result), 2),
            "storage_hours": sum(r["mode"] == STORAGE for r in result),
            "grid_hours": sum(r["mode"] == GRID for r in result),
            "overload_hours": sum(r["overload"] for r in result),
            "min_soc_pct": min(r["battery_soc_pct"] for r in result),
            "final_soc_pct": result[-1]["battery_soc_pct"],
        }
        run = {"run_id": uuid.uuid4().hex[:12],
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "rows": result, "summary": summary, "logs": logs}
        persist(run, csv_bytes)
        return jsonify(run)

    except ValueError as e:                       # validation problems -> 400
        return jsonify(error=str(e), logs=logs), 400
    except Exception as e:                        # noqa: BLE001
        return jsonify(error=f"Forecast failed: {e}", logs=logs), 502


if __name__ == "__main__":
    port = int(os.getenv("X_ZOHO_CATALYST_LISTEN_PORT", os.getenv("PORT", "9000")))
    app.run(host="0.0.0.0", port=port)
