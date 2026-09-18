"""Switch Yue2 Cover endpoint template image to cover-4 (REST)."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

EP = "bdn7z6r6guv3ll"
IMG = "ghcr.io/leidige/yue2-runpod-worker:cover-8"


def req(method: str, url: str, key: str, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "yue2-update-template/1.0",
        },
    )
    try:
        with urllib.request.urlopen(r, timeout=90) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def main() -> int:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        print("missing RUNPOD_API_KEY", file=sys.stderr)
        return 1
    code, raw = req("GET", f"https://rest.runpod.io/v1/endpoints/{EP}", key)
    if code != 200:
        print(f"GET endpoint HTTP {code}: {raw[:500]}")
        return 1
    ep = json.loads(raw)
    tpl_id = ep.get("templateId")
    print("endpoint", EP, "templateId", tpl_id)
    code, traw = req("GET", f"https://rest.runpod.io/v1/templates/{tpl_id}", key)
    if code != 200:
        print(f"GET template HTTP {code}: {traw[:500]}")
        return 1
    tpl = json.loads(traw)
    print("before imageName=", tpl.get("imageName"))
    patch = {"imageName": IMG}
    # Keep existing start cmd if present; YuE2 image uses handler.py at /app
    code, praw = req("PATCH", f"https://rest.runpod.io/v1/templates/{tpl_id}", key, patch)
    print(f"PATCH HTTP {code}: {praw[:800]}")
    if code not in (200, 201):
        return 2
    code, traw2 = req("GET", f"https://rest.runpod.io/v1/templates/{tpl_id}", key)
    print("after", json.loads(traw2).get("imageName") if code == 200 else traw2[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
