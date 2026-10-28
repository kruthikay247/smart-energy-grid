"""
Setup checker - run:  py check_setup.py
Shows which settings are present (never prints their values), tests the OAuth token,
then sends ONE row to your QuickML endpoint and reports what happened.
"""
import os
import sys

import pandas as pd
import requests

import quickml_client as q   # also loads the .env file

NEEDED = ["QUICKML_ENDPOINT", "QUICKML_ENDPOINT_KEY", "CATALYST_ORG_ID",
          "ZOHO_CLIENT_ID", "ZOHO_CLIENT_SECRET", "ZOHO_REFRESH_TOKEN"]

print("1) Settings found (window or .env file; values hidden):")
missing = []
for k in NEEDED:
    ok = bool(os.getenv(k))
    print(f"   {'OK     ' if ok else 'MISSING'}  {k}")
    if not ok:
        missing.append(k)
if missing:
    print("\nFill the MISSING ones in the .env file (copy .env.example to .env), then run this again.")
    sys.exit(1)

print("\n2) OAuth token refresh:")
base = os.getenv("ZOHO_ACCOUNTS_URL") or q.default_accounts_url()
r = requests.post(f"{base}/oauth/v2/token", data={
    "grant_type": "refresh_token", "client_id": os.environ["ZOHO_CLIENT_ID"],
    "client_secret": os.environ["ZOHO_CLIENT_SECRET"], "refresh_token": os.environ["ZOHO_REFRESH_TOKEN"]})
body = r.json()
if "access_token" not in body:
    print("   FAILED:", {k: v for k, v in body.items() if k != "refresh_token"})
    print("   -> 'invalid_client' = wrong client id/secret; 'invalid_code' = wrong/expired refresh token;")
    print(f"   -> accounts server used: {base} (must match your data centre)")
    sys.exit(1)
print("   OK - got an access token")

print("\n3) One prediction from the endpoint:")
df = pd.read_csv("sample_test_input.csv").head(1)
from predict import predict
try:
    print("   OK - predicted demand:", round(float(predict(df)[0]), 2), "kW")
except Exception as e:                       # noqa: BLE001
    print("   FAILED:", str(e)[:400])
