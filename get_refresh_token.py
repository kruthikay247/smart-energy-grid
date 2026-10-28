"""
One-time helper: turn a Self Client grant code into a long-lived refresh token.

1) Open https://api-console.zoho.in  (use .com / .eu if that is your data centre)
2) Add Client -> Self Client -> Scope: QuickML.deployment.READ -> Create -> copy the grant code
3) python get_refresh_token.py      (inputs are typed in, never saved to disk)
4) Put the printed refresh token into your environment as ZOHO_REFRESH_TOKEN
"""
import getpass

import requests

base = input("Accounts URL [https://accounts.zoho.in]: ").strip() or "https://accounts.zoho.in"
cid = input("Client ID: ").strip()
sec = getpass.getpass("Client Secret (hidden): ").strip()
code = input("Grant code: ").strip()
r = requests.post(f"{base}/oauth/v2/token", data={
    "grant_type": "authorization_code", "client_id": cid, "client_secret": sec, "code": code})
body = r.json()
print("\nRefresh token:" if "refresh_token" in body else "\nFailed:", body.get("refresh_token", body))
