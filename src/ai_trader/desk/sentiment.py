"""Sentiment domain (sentiment.json) and the Reddit collector. H-SENT1 (proposed, prior 0.20): sentiment has NO validated
decision weight; this layer only collects a point-in-time panel so H-SENT1 can be tested after >= 60 days.

Collector (`collect_reddit`, every 15 min from scripts/desk_collect.py)
  Official Reddit API, application-only OAuth (client_credentials) with the script-app credentials REDDIT_CLIENT_ID /
  REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT, read through ai_trader.core.config.secret (the .env is loaded by the config
  module; nothing here prints a credential, only present / missing). GET listings only: /r/<sub>/new and
  /r/<sub>/comments (up to MAX_PAGES pages of 100, stopping at the first id already stored). No X / Facebook scraping.
  Per item stored (data/research/sentiment/reddit/<seen date>.jsonl): id, kind (post | comment), sub, created_ts,
  seen_ts, tickers, tone, score, num_comments, a post title (<= 150 chars); comment bodies are never stored.
  Without keys the collector writes nothing for Reddit and the panel is built from the RSS headline store only
  (status "reddit: pending keys").
Panel (`hourly_panel`): per ticker x hour of creation x source, counting only items with seen_ts <= as_of:
  mentions (items naming the ticker, or any item in a per-coin sub), comments, posts, news items, tone sum / n.
Trend (`trend`): x_h = ln(1 + mentions in hour h). recent = the last RECENT_H complete hours, baseline = up to BASE_H
  complete hours before them and after the collection start. z_mentions = (mean x_recent - mean x_base) / sd(x_base);
  z_tone likewise on hourly mean tone (hours with items). rising if z >= 1, falling if z <= -1, else flat;
  "insufficient" while the baseline has < MIN_BASE_H hours.
Storage: <= BUDGET_MB under data/research/sentiment/ (oldest day files pruned first), files older than KEEP_DAYS removed.
"""

from __future__ import annotations

import base64
import json
import math
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from ai_trader.desk.text import TickerIndex, tone

SUBS_GENERAL = {"CryptoCurrency": None, "Bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "CryptoMarkets": None, "altcoin": None}
SUBS_COIN = {"dogecoin": "DOGE", "XRP": "XRP", "cardano": "ADA", "Chainlink": "LINK", "litecoin": "LTC", "dot": "DOT",
             "Avax": "AVAX", "SuiNetwork": "SUI", "Hedera": "HBAR", "Stellar": "XLM", "Monero": "XMR", "zec": "ZEC", "binance": "BNB",
             "Aptos": "APT", "Arbitrum": "ARB", "nearprotocol": "NEAR", "cosmosnetwork": "ATOM", "Tronix": "TRX", "UniSwap": "UNI",
             "Aave_Official": "AAVE", "bittensor_": "TAO", "Hyperliquid": "HYPE", "pepecoin": "PEPE", "ethfinance": "ETH", "BitcoinMarkets": "BTC"}
REDDIT_AUTH = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"
ALLOWED_LISTINGS = ("new", "comments")
MAX_PAGES = 3
KEYS = ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
RECENT_H, BASE_H, MIN_BASE_H = 6, 168, 24
BUDGET_MB, KEEP_DAYS = 200, 120
TITLE_MAX = 150


def store_root(root: Path) -> Path:
    p = root / "data" / "research" / "sentiment"
    p.mkdir(parents=True, exist_ok=True)
    return p


def key_status() -> dict[str, str]:
    from ai_trader.core.config import secret

    return {k: "present" if secret(k) else "missing" for k in KEYS}


class Reddit:
    """Application-only OAuth client; GET listings only. `http(method, url, headers, data) -> dict` is injectable."""

    def __init__(self, client_id: str, client_secret: str, user_agent: str, http=None):
        self._id, self._secret, self.ua = client_id, client_secret, user_agent
        self._http = http or self._urllib
        self._token: str | None = None
        self._exp = datetime.min.replace(tzinfo=UTC)
        self.calls = 0

    @classmethod
    def from_env(cls) -> Reddit | None:
        from ai_trader.core.config import secret

        vals = [secret(k) for k in KEYS]
        return cls(*vals) if all(vals) else None

    @staticmethod
    def _urllib(method: str, url: str, headers: dict, data: bytes | None) -> dict:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def _auth(self) -> str:
        if self._token and datetime.now(UTC) < self._exp:
            return self._token
        basic = base64.b64encode(f"{self._id}:{self._secret}".encode()).decode()
        d = self._http("POST", REDDIT_AUTH, {"Authorization": f"Basic {basic}", "User-Agent": self.ua,
                                             "Content-Type": "application/x-www-form-urlencoded"}, b"grant_type=client_credentials")
        if "access_token" not in d:
            raise RuntimeError(f"reddit auth failed: {d.get('error', 'no token')}")
        self._token = d["access_token"]
        self._exp = datetime.now(UTC) + timedelta(seconds=int(d.get("expires_in", 3600)) - 60)
        return self._token

    def listing(self, sub: str, which: str, after: str | None = None, limit: int = 100) -> dict:
        if which not in ALLOWED_LISTINGS:
            raise ValueError(which)
        q = {"limit": limit, "raw_json": 1}
        if after:
            q["after"] = after
        url = f"{REDDIT_API}/r/{urllib.parse.quote(sub)}/{which}?{urllib.parse.urlencode(q)}"
        self.calls += 1
        return self._http("GET", url, {"Authorization": f"bearer {self._auth()}", "User-Agent": self.ua}, None)


def parse_listing(d: dict, sub: str, seen: datetime, index: TickerIndex, default: str | None) -> list[dict]:
    out = []
    for ch in (d.get("data") or {}).get("children", []):
        kind, x = ch.get("kind"), ch.get("data") or {}
        if kind not in ("t3", "t1") or not x.get("name"):
            continue
        created = datetime.fromtimestamp(float(x.get("created_utc", 0)), UTC)
        if kind == "t3":
            text = f"{x.get('title') or ''} {x.get('selftext') or ''}"
            rec = {"id": x["name"], "kind": "post", "title": (x.get("title") or "")[:TITLE_MAX], "num_comments": int(x.get("num_comments") or 0)}
        else:
            text = x.get("body") or ""
            rec = {"id": x["name"], "kind": "comment"}
        rec.update({"sub": sub, "created_ts": created.isoformat(), "seen_ts": seen.isoformat(timespec="seconds"), "source": "reddit",
                    "tickers": index.find(text, default), "tone": round(tone(text), 4), "score": int(x.get("score") or 0)})
        out.append(rec)
    return out


def _read(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if p.is_file():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    try:
                        rows.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    return rows


def _days(d: Path, t: datetime, n: int) -> list[Path]:
    return [d / f"{(t - timedelta(days=k)).date().isoformat()}.jsonl" for k in range(n, -1, -1)]


def collect_reddit(store: Path, client: Reddit | None, index: TickerIndex, *, now: datetime | None = None,
                   subs: dict[str, str | None] | None = None, log=print) -> dict:
    t = now or datetime.now(UTC)
    if client is None:
        return {"reddit": "pending keys", "new": 0}
    subs = subs or {**SUBS_GENERAL, **SUBS_COIN}
    d = store / "reddit"
    d.mkdir(parents=True, exist_ok=True)
    seen = {r["id"] for r in _read(_days(d, t, 3))}
    stat: dict = {"reddit": "ok", "new": 0, "subs": {}}
    rows: list[dict] = []
    for sub, default in subs.items():
        n_sub = 0
        try:
            for which in ALLOWED_LISTINGS:
                after = None
                for _ in range(MAX_PAGES):
                    js = client.listing(sub, which, after)
                    items = parse_listing(js, sub, t, index, default)
                    fresh = [x for x in items if x["id"] not in seen]
                    for x in fresh:
                        seen.add(x["id"])
                    rows += fresh
                    n_sub += len(fresh)
                    after = (js.get("data") or {}).get("after")
                    if not after or len(fresh) < len(items):
                        break
            stat["subs"][sub] = n_sub
        except Exception as e:  # noqa: BLE001
            stat["subs"][sub] = f"error: {type(e).__name__}"
            log(f"  reddit r/{sub}: {type(e).__name__}")
    if rows:
        with (d / f"{t.date().isoformat()}.jsonl").open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    stat["new"] = len(rows)
    stat["calls"] = client.calls
    return stat


def prune(store: Path, now: datetime, budget_mb: float = BUDGET_MB, keep_days: int = KEEP_DAYS) -> dict:
    files = sorted((p for p in store.rglob("*.jsonl")), key=lambda p: p.name)
    removed = 0
    for p in files:
        try:
            day = datetime.fromisoformat(p.stem).replace(tzinfo=UTC)
        except ValueError:
            continue
        if day < now - timedelta(days=keep_days):
            p.unlink()
            removed += 1
    files = sorted((p for p in store.rglob("*.jsonl") if p.is_file()), key=lambda p: p.name)
    total = sum(p.stat().st_size for p in files)
    while files and total > budget_mb * 1e6:
        p = files.pop(0)
        total -= p.stat().st_size
        p.unlink()
        removed += 1
    return {"removed": removed, "mb": round(total / 1e6, 2)}


# ---------------------------------------------------------------- panel and trend

def all_items(store: Path, as_of: datetime, days: int = 8) -> list[dict]:
    rows = _read(_days(store / "reddit", as_of, days))
    for r in _read(_days(store / "news", as_of, days)):
        rows.append({"id": "news:" + r["h"], "kind": "news", "source": f"rss:{r['feed']}", "sub": None,
                     "created_ts": r.get("pub_ts") or r["seen_ts"], "seen_ts": r["seen_ts"], "tickers": r.get("tickers", []), "tone": r.get("tone", 0.0)})
    return rows


def hourly_panel(items: list[dict], as_of: datetime, hours: int = BASE_H + RECENT_H + 1) -> dict[str, dict[datetime, dict]]:
    """ticker -> hour start -> {mentions, comments, posts, news, tone_sum, tone_n, reddit, rss}; only seen_ts <= as_of."""
    lo = as_of - timedelta(hours=hours)
    out: dict[str, dict[datetime, dict]] = {}
    for r in items:
        st = datetime.fromisoformat(r["seen_ts"])
        ct = min(datetime.fromisoformat(r["created_ts"]), st)
        if st > as_of or ct < lo:
            continue
        h = ct.replace(minute=0, second=0, microsecond=0)
        for tk in r.get("tickers", []):
            c = out.setdefault(tk, {}).setdefault(h, {"mentions": 0, "comments": 0, "posts": 0, "news": 0, "tone_sum": 0.0, "tone_n": 0, "reddit": 0, "rss": 0})
            c["mentions"] += 1
            c[{"comment": "comments", "post": "posts", "news": "news"}.get(r["kind"], "news")] += 1
            c["reddit" if r.get("source") == "reddit" else "rss"] += 1
            c["tone_sum"] += float(r.get("tone") or 0.0)
            c["tone_n"] += 1
    return out


def trend(cells: dict[datetime, dict], as_of: datetime, collected_since: datetime | None) -> dict:
    end = as_of.replace(minute=0, second=0, microsecond=0)               # complete hours end at `end`
    hrs = [end - timedelta(hours=k) for k in range(BASE_H + RECENT_H, 0, -1)]
    if collected_since is not None:
        first_full = collected_since.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        hrs = [h for h in hrs if h >= first_full]
    rec, base = hrs[-RECENT_H:], hrs[:-RECENT_H]
    x = {h: math.log1p(cells.get(h, {}).get("mentions", 0)) for h in hrs}
    tn = {h: cells[h]["tone_sum"] / cells[h]["tone_n"] for h in hrs if h in cells and cells[h]["tone_n"]}
    last24 = [cells[h] for h in cells if as_of - timedelta(hours=24) <= h <= as_of]
    out = {"mentions_24h": sum(c["mentions"] for c in last24), "comments_24h": sum(c["comments"] for c in last24),
           "posts_24h": sum(c["posts"] for c in last24), "news_24h": sum(c["news"] for c in last24),
           "reddit_24h": sum(c["reddit"] for c in last24),
           "tone_24h": round(sum(c["tone_sum"] for c in last24) / max(1, sum(c["tone_n"] for c in last24)), 3),
           "mentions_recent_per_h": round(float(np.mean([math.expm1(x[h]) for h in rec])), 2) if len(rec) == RECENT_H else None,
           "baseline_hours": len(base)}
    if len(rec) < RECENT_H or len(base) < MIN_BASE_H:
        return {**out, "z_mentions": None, "z_tone": None, "trend": "insufficient"}
    xb = np.array([x[h] for h in base])
    sd = float(np.std(xb, ddof=1))
    zm = (float(np.mean([x[h] for h in rec])) - float(np.mean(xb))) / sd if sd > 0 else None
    tb = np.array([tn[h] for h in base if h in tn])
    tr = [tn[h] for h in rec if h in tn]
    zt = None
    if len(tb) >= 6 and tr and float(np.std(tb, ddof=1)) > 0:
        zt = (float(np.mean(tr)) - float(np.mean(tb))) / float(np.std(tb, ddof=1))
    lab = "flat" if zm is None else ("rising" if zm >= 1 else ("falling" if zm <= -1 else "flat"))
    return {**out, "z_mentions": None if zm is None else round(zm, 2), "z_tone": None if zt is None else round(zt, 2), "trend": lab}


def build_sentiment(as_of: datetime, store: Path, universe_bases: dict[str, str]) -> dict:
    """universe_bases: base -> Bybit symbol."""
    items = all_items(store, as_of)
    status = {}
    sp = store / "status.json"
    if sp.is_file():
        try:
            status = json.loads(sp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    seen = [datetime.fromisoformat(r["seen_ts"]) for r in items if datetime.fromisoformat(r["seen_ts"]) <= as_of]
    reddit_seen = [datetime.fromisoformat(r["seen_ts"]) for r in items if r.get("source") == "reddit" and datetime.fromisoformat(r["seen_ts"]) <= as_of]
    since = min(datetime.fromisoformat(status["collected_since"]), as_of) if status.get("collected_since") else (min(seen) if seen else None)
    panel = hourly_panel(items, as_of)
    rows = []
    for base, sym in universe_bases.items():
        t = trend(panel.get(base, {}), as_of, since)
        rows.append({"ticker": base, "symbol": sym, **t})
    rows.sort(key=lambda r: -r["mentions_24h"])
    ks = key_status()
    rstat = "ok" if all(v == "present" for v in ks.values()) and reddit_seen else ("pending keys" if any(v == "missing" for v in ks.values()) else "keys present, no items yet")
    ranked = [r for r in rows if r["z_mentions"] is not None]
    return {
        "domain": "sentiment", "as_of": as_of.isoformat(),
        "reddit": f"reddit: {rstat}", "keys": ks, "collected_since": since.isoformat() if since else None,
        "last_collect": status.get("last_collect"), "items_known": len(seen), "reddit_items_known": len(reddit_seen),
        "method": {"panel": "per ticker x creation hour; seen_ts <= as_of", "z": f"ln(1+mentions/h), last {RECENT_H}h vs up to {BASE_H}h baseline (>= {MIN_BASE_H}h)",
                   "tone": "lexicon compound score in [-1, 1] (desk.text.tone)", "sources": "reddit (official OAuth API) + RSS headlines"},
        "note": "H-SENT1 is proposed, not tested: sentiment carries no decision weight yet; never X / Facebook",
        "rising": [r["ticker"] for r in sorted(ranked, key=lambda r: -r["z_mentions"]) if r["trend"] == "rising"][:10],
        "falling": [r["ticker"] for r in sorted(ranked, key=lambda r: r["z_mentions"]) if r["trend"] == "falling"][:10],
        "tickers": rows,
    }


__all__ = [
    "SUBS_COIN",
    "SUBS_GENERAL",
    "Reddit",
    "build_sentiment",
    "collect_reddit",
    "hourly_panel",
    "key_status",
    "parse_listing",
    "prune",
    "trend",
]
