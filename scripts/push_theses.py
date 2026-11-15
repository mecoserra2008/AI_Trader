"""Push a theses YAML into the STOCK SCANNER app through its supported API (POST /api/v1/theses).

    PYTHONIOENCODING=utf-8 uv run python scripts/push_theses.py research/theses/2026-11-09.yaml [--api http://127.0.0.1:8010] [--dry-run]

Idempotent: the thesis_id returned for each `key` is stored in research/theses/stock_scanner_ids.json; keys already
pushed are PATCHed (same fields) instead of created twice. Never writes the app's DuckDB/Parquet files directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import yaml

IDS = Path(__file__).resolve().parents[1] / "research" / "theses" / "stock_scanner_ids.json"


def call(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body, default=str).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url} -> {e.code}: {e.read().decode()[:600]}") from None


def payload(t: dict, d: dict) -> dict:
    p = {k: v for k, v in t.items() if k != "key"}
    p.setdefault("status", d["status"])
    p.setdefault("horizon", d["horizon"])
    p.setdefault("review_every_days", d["review_every_days"])
    p["origin"] = {**d["origin"], "ref": f"{d['origin']['ref']}#{t['key']}"}
    p["tags"] = list(dict.fromkeys([*d["common_tags"], *p.get("tags", [])]))[:20]
    for c in p.get("catalysts", []) + p.get("invalidation", []):
        if isinstance(c.get("date"), date):
            c["date"] = c["date"].isoformat()
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--api", default="http://127.0.0.1:8010")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    doc = yaml.safe_load(Path(a.file).read_text(encoding="utf-8"))
    ids = json.loads(IDS.read_text(encoding="utf-8")) if IDS.exists() else {}
    base = a.api.rstrip("/") + "/api/v1/theses"
    for t in doc["theses"]:
        p = payload(t, doc["defaults"])
        if a.dry_run:
            print("DRY", t["key"], json.dumps(p, default=str)[:200])
            continue
        if t["key"] in ids:
            body = {k: v for k, v in p.items() if k not in ("subject_type", "subject_id", "origin")}
            call("PATCH", f"{base}/{ids[t['key']]}", body)
            print("updated", t["key"], ids[t["key"]])
        else:
            r = call("POST", base, p)
            tid = r.get("thesis_id") or (r.get("thesis") or {}).get("thesis_id")
            if not tid:
                sys.exit(f"no thesis_id in response for {t['key']}: {str(r)[:300]}")
            ids[t["key"]] = tid
            IDS.write_text(json.dumps(ids, indent=1), encoding="utf-8")
            print("created", t["key"], tid)


if __name__ == "__main__":
    main()
