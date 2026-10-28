"""
Thin client for a published Zoho Catalyst QuickML endpoint.

Environment variables (never hard-code secrets):
  QUICKML_ENDPOINT        full URL of the published endpoint            (required; predict.py also has a constant)
  CATALYST_ORG_ID         value for the CATALYST-ORG header
  QUICKML_ENDPOINT_KEY    value for the X-QUICKML-ENDPOINT-KEY header (endpoint details page)
  CATALYST_ENVIRONMENT    Development (default) or Production
  -- OAuth (the endpoint page lists "Authorization: Zoho-oauthtoken", scope QuickML.deployment.READ):
     EITHER a ready token ...
  QUICKML_ACCESS_TOKEN    OAuth access token
  -- ... OR a refresh-token flow (recommended, tokens expire in 1h):
  ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN
  ZOHO_ACCOUNTS_URL       default https://accounts.zoho.com (use .in / .eu for other DCs)
  -- payload tuning (match the sample request shown on your QuickML endpoint page):
  QUICKML_PAYLOAD_STYLE   "auto" (default) -> tries a list once, falls back to "record"
                          "record" -> {"data": {col: val, ...}}   one call per row (matches the
                                      sample request on the endpoint page)
                          "batch"  -> {"data": [{...}, {...}]} -> {"predictions": [...]}
  QUICKML_BATCH_SIZE      rows per batch request (default 200)
  QUICKML_MAX_WORKERS     parallel calls in "record" mode (default 8)
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

import pandas as pd

from features import FEATURE_COLS, FALLBACK_TS

def _load_dotenv():
    """Read KEY=value lines from a local .env file (same folder) into os.environ.
    Real environment variables win. .env is git-ignored - never commit it."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v and not v.startswith("<"):          # ignore unfilled placeholders
                os.environ.setdefault(k.strip(), v)


_load_dotenv()
_token_cache = {"value": None, "exp": 0.0}
PREFERRED_KEYS = ("predicted_result", "prediction", "predictions", "predicted_value", "result",
                  "target_demand_kw", "output", "data")


def default_accounts_url() -> str:
    """accounts.zoho.<tld> matching the endpoint's data centre (.in, .eu, .com, ...)."""
    url = os.getenv("QUICKML_ENDPOINT", "")
    for tld in ("in", "eu", "com.au", "com"):
        if f"catalyst.zoho.{tld}" in url:
            return f"https://accounts.zoho.{tld}"
    return "https://accounts.zoho.com"


def _get_token():
    """Returns an access token, or None when no credentials are configured."""
    static = os.getenv("QUICKML_ACCESS_TOKEN")
    if static:
        return static
    if _token_cache["value"] and time.time() < _token_cache["exp"]:
        return _token_cache["value"]
    cid, sec, ref = (os.getenv(k) for k in ("ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN"))
    if not (cid and sec and ref):
        return None
    base = os.getenv("ZOHO_ACCOUNTS_URL") or default_accounts_url()
    r = requests.post(f"{base}/oauth/v2/token", data={
        "grant_type": "refresh_token", "client_id": cid,
        "client_secret": sec, "refresh_token": ref}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if "access_token" not in body:
        raise RuntimeError(f"Token refresh failed: {body}")
    _token_cache["value"] = body["access_token"]
    _token_cache["exp"] = time.time() + int(body.get("expires_in", 3600)) - 120
    return _token_cache["value"]


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    token = _get_token()
    if token:
        h["Authorization"] = f"Zoho-oauthtoken {token}"
    org = os.getenv("CATALYST_ORG_ID")
    if org:
        h["CATALYST-ORG"] = org
    key = os.getenv("QUICKML_ENDPOINT_KEY")
    if key:
        h["X-QUICKML-ENDPOINT-KEY"] = key
    h["Environment"] = os.getenv("CATALYST_ENVIRONMENT", "Development")   # or Production
    return h


def _extract_number(obj):
    """Find the first numeric prediction inside whatever JSON QuickML returns."""
    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj)
        except ValueError:
            return None
    if isinstance(obj, list):
        for item in obj:
            v = _extract_number(item)
            if v is not None:
                return v
        return None
    if isinstance(obj, dict):
        for k in PREFERRED_KEYS:
            if k in obj:
                v = _extract_number(obj[k])
                if v is not None:
                    return v
        for v in obj.values():
            n = _extract_number(v)
            if n is not None:
                return n
    return None


INT_COLS = ["hour", "day_of_week", "is_weekend", "month", "is_evening_peak"]


def _records(feature_df):
    """Records exactly like the endpoint's sample request: timestamp + all feature columns."""
    ts = pd.to_datetime(feature_df["timestamp"], errors="coerce").fillna(FALLBACK_TS)
    out = feature_df[FEATURE_COLS].astype(float).copy()
    out.insert(0, "timestamp", ts.dt.strftime("%Y-%m-%d %H:%M:%S").values)
    recs = out.to_dict(orient="records")
    for r in recs:
        for c in INT_COLS:
            r[c] = int(round(r[c]))                      # 21, not 21.0
        v = r["solar_irradiance_wm2"]                    # whole-number column in the training data
        r["solar_irradiance_wm2"] = int(v) if float(v).is_integer() else v
    return recs


def _post(url, payload, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, headers=_headers(), timeout=60)
            if r.status_code == 401:            # token expired -> drop cache and retry
                _token_cache["value"] = None
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            return r.json()
        except Exception as e:                  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"QuickML call failed after {retries} tries: {last}")


PLACEHOLDER_URL = "YOUR_PUBLISHED_QUICKML_ENDPOINT_URL_HERE"


def predict_features(feature_df, url=None) -> list:
    """feature_df must contain FEATURE_COLS. Returns list[float], one per row."""
    url = url or os.getenv("QUICKML_ENDPOINT")
    if not url or url == PLACEHOLDER_URL:
        raise RuntimeError("QuickML endpoint URL is not configured")
    records = _records(feature_df)
    style = os.getenv("QUICKML_PAYLOAD_STYLE", "auto").lower()

    if style == "auto":
        # The endpoint's sample request is ONE record under "data". Try a list once
        # (faster, and the starter kit's predict.py uses it); fall back to per-record calls.
        try:
            body = _post(url, {"data": records[:2]}, retries=1)
            raw = body.get("predictions", body.get("data", body)) if isinstance(body, dict) else body
            style = "batch" if isinstance(raw, list) and len(raw) == len(records[:2]) \
                and all(_extract_number(x) is not None for x in raw) else "record"
        except Exception:                      # noqa: BLE001
            style = "record"

    if style == "batch":
        size = int(os.getenv("QUICKML_BATCH_SIZE", "200"))
        preds = []
        for i in range(0, len(records), size):
            chunk = records[i:i + size]
            body = _post(url, {"data": chunk})
            raw = body.get("predictions", body.get("data", body)) if isinstance(body, dict) else body
            if not isinstance(raw, list) or len(raw) != len(chunk):
                raise RuntimeError(f"Expected {len(chunk)} predictions, got: {str(body)[:300]}")
            preds += [_extract_number(x) for x in raw]
    else:
        def one(rec):
            return _extract_number(_post(url, {"data": rec}))
        workers = int(os.getenv("QUICKML_MAX_WORKERS", "8"))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            preds = list(ex.map(one, records))

    if any(p is None for p in preds):
        raise RuntimeError("QuickML returned a response without a numeric prediction")
    return preds
