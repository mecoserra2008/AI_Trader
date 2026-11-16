"""Hourly SHADOW line of the cross-sectional network xs.v4 over every liquid Bybit crypto perp (block XS4, H-XS4).

CONTEXT ONLY, research only: never trades, never read by the decision sessions (R-XS-2 / R-XS-3). xs.v1, v2.1 and xs.v3 are
pinned track records; this module imports xs_v3 helpers read-only and changes none of their files or configs.

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_xs_v4.py --once | --loop

One run (`run_line`, as_of = now, point in time):
  universe  Bybit public tickers + instruments now: crypto USDT linear perps (symbolType ''; no innovation zone / TradFi / SPY), 24h
            turnover >= 20 M USD and top-of-book spread <= 5 bp (config/live_xs_v4.yaml). BTCUSDT is required.
  data      the REST cache data/research/xs_v4/raw is topped up (closed 1h klines, funding, 4h OI; a new token gets its full history),
            frames built by xs_v4_study.frame (known_ts <= as_of), tokens with < 120 days of bars are absent.
  model     data/models/xs_v4/frozen (spec.json, v4_seed{0,1,2}.pt; weight hashes verified every run), the study's token_rows /
            assemble on the last 130 days with a placeholder bar at the forecast hour S (xs_v3.build_rows, exact tail evaluation).
  outputs   per token: 4h / 24h relative-return q10 / q50 / q90 (bp vs the equal-weight basket of present tokens), ranks by the 24h
            and 4h medians, the low-vol rank (-lsig2), 1h / 4h vol quantiles (% |r|) and the panel log-HAR-t x hour-of-week bands.
  scoring   every earlier line whose window has closed, once: 4h / 24h Spearman IC of v4 and of the low-vol ranking at the
            non-overlapping origins (4h: 00, 04, .. UTC; 24h: 00 UTC); 1h vol pinball v4 vs log-HAR-t (every line).
Files: data/journal/live_xs_v4/<YYYY-MM-DD of S>.jsonl (id xs4-<S>), scores.jsonl, scored.json; research/live/xs_v4_LATEST.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

MODEL_ID = "xs.v4"
NOT_A_SIGNAL = "SHADOW, context only, not a signal (R-XS-2 / R-XS-3); no orders; the decision sessions do not read it"
MIN_HISTORY = 24 * 120
TOP = 5
QN = ("q10", "q50", "q90")


def _root() -> Path:
    from ai_trader.core.config import load

    return load().root


def cfg() -> dict:
    from ai_trader.research.xs_v4_data import cfg as c

    return c()


def journal_dir() -> Path:
    p = _root() / cfg()["journal"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_frozen() -> tuple[dict, list]:
    import torch

    from ai_trader.ml import xs_net as xs

    fz = _root() / "data" / "models" / "xs_v4" / "frozen"
    spec = json.loads((fz / "spec.json").read_text(encoding="utf-8"))
    for name, h in spec["files_sha256"].items():
        if _sha(fz / name) != h:
            raise RuntimeError(f"xs_v4 weights hash mismatch: {name}")
    m = spec["model"]
    models = []
    for s in range(int(m["seeds"])):
        net = xs.make(int(m["n_in"]), int(m["n_mkt"]), int(m["n_sector"]), kind=m["kind"], mix=bool(m["mix"]))
        net.load_state_dict(torch.load(fz / f"v4_seed{s}.pt", weights_only=True))
        net.eval()
        models.append(net)
    return spec, models


def code_ok(spec: dict) -> dict:
    return {f: (_root() / f).exists() and _sha(_root() / f) == h for f, h in spec.get("code_sha256", {}).items()}


def _utc(x) -> datetime:
    return x if x.tzinfo else x.replace(tzinfo=UTC)


# ------------------------------------------------------------------ one line


def run_line(now: datetime | None = None) -> dict:
    import torch

    from ai_trader.live import xs_v3
    from ai_trader.ml import xs_net as xs
    from ai_trader.research import xs_network_study as study
    from ai_trader.research import xs_v4_data as data
    from ai_trader.research import xs_v4_study as v4

    now = now or datetime.now(UTC)
    c = cfg()
    spec, models = load_frozen()
    snap = data.snapshot(c["universe"]["min_turnover_usd"], c["universe"]["max_spread_bp"])
    live = [r["symbol"] for r in snap]
    if "BTCUSDT" not in live:
        raise RuntimeError("BTCUSDT not in the live universe")
    data.download(live, verbose=False)
    frames, absent = {}, {}
    for s in live:
        f = v4.frame(s, now)
        if f.height < MIN_HISTORY:
            absent[s] = f"short history ({f.height} h)"
            continue
        frames[s] = f.drop("turnover")
    S, last_close = xs_v3.forecast_hour(frames)
    rows = xs_v3.build_rows(frames, S)
    syms = ["BTCUSDT", *[s for s in frames if s != "BTCUSDT"]]
    P = study.assemble(rows, syms)
    hit = np.flatnonzero(P["ts"] == xs_v3._us(S))
    if not len(hit):
        raise RuntimeError(f"forecast row {S.isoformat()} missing")
    i = int(hit[-1])
    so = spec["sector_order"]
    secs = v4.sectors(syms)
    sec = np.array([so.index(x) for x in secs], dtype=np.int64)
    st = spec["standardisation"]
    mu, sd = np.asarray(st["mu"], np.float32), np.asarray(st["sd"], np.float32)
    mmu, msd = np.asarray(st["mkt_mu"], float), np.asarray(st["mkt_sd"], float)
    Z = np.clip((P["X"][[i]] - mu) / sd, -6, 6)
    Z = np.nan_to_num(Z, nan=0.0).astype(np.float32)
    M = np.clip((P["M"][[i]] - mmu) / msd, -6, 6).astype(np.float32)
    pres = P["present"][[i]]
    N = len(syms)
    dummy_v = np.zeros((1, N, len(xs.VOL_H)), np.float32)
    dummy_r = np.zeros((1, N, len(xs.REL_H)), np.float32)
    B = xs.Batchable(Z, M, pres, dummy_v, dummy_v.astype(bool), dummy_r, dummy_r.astype(bool))
    old = torch.get_num_threads()
    torch.set_num_threads(int(c.get("torch_threads", 2)))
    try:
        outs = [xs.predict(m, B, sec) for m in models]
    finally:
        torch.set_num_threads(old)
    vol = np.mean([o[0] for o in outs], 0)[0]   # (N, 3, 7)
    rel = np.mean([o[1] for o in outs], 0)[0]   # (N, 2, 3)
    har = {int(k): v for k, v in spec["har_baseline"].items()}
    vh = study.har_quantiles(P, np.array([i]), har)[0]
    names = list(P["features"])
    lsig2 = P["X"][i, :, names.index("lsig2")].astype(float)
    s2 = P["s2"][i]
    present = pres[0]
    toks = {}
    for j, s in enumerate(syms):
        if not present[j]:
            absent.setdefault(s, "absent in the panel (missing core feature)")
            continue
        sj = math.sqrt(s2[j])
        t = {"sector": secs[j], "s2": float(s2[j]), "lsig2": float(lsig2[j])}
        for k, h in enumerate(xs.REL_H):
            t[f"rel{h}_bp"] = {q: float(rel[j, k, a] * math.sqrt(h) * sj * 1e4) for a, q in enumerate(QN)}
        for k, h in enumerate(xs.VOL_H[:2]):
            t[f"vol{h}h_pct"] = [float(x * math.sqrt(h) * sj * 100) for x in vol[j, k]]
            t[f"har{h}h_pct"] = [float(x * math.sqrt(h) * sj * 100) for x in vh[j, k]]
            t[f"vol{h}h_model"] = [float(x) for x in vol[j, k]]
            t[f"har{h}h_model"] = [float(x) for x in vh[j, k]]
        toks[s] = t
    pr = [s for s in syms if s in toks]

    def ranks(key) -> dict:
        v = np.array([key(toks[s]) for s in pr])
        order = np.argsort(-v)
        return {pr[o]: r + 1 for r, o in enumerate(order)}

    rk24 = ranks(lambda t: t["rel24_bp"]["q50"])
    rk4 = ranks(lambda t: t["rel4_bp"]["q50"])
    rklv = ranks(lambda t: -t["lsig2"])
    for s in pr:
        toks[s].update({"rank24": rk24[s], "rank4": rk4[s], "rank_lowvol": rklv[s]})
    line = {"id": f"xs4-{S.strftime('%Y%m%dT%H')}", "model_id": MODEL_ID, "version": spec["version"], "S": S.isoformat(), "as_of": now.isoformat(),
            "newest_close": _utc(last_close).isoformat(), "note": NOT_A_SIGNAL,
            "model": {"spec_trained_at": spec["trained_at"], "as_of_data": spec["as_of_data"], "weights_sha256": {k: v[:16] for k, v in spec["files_sha256"].items()},
                      "code_sha256_ok": code_ok(spec), "git_commit": spec.get("git_commit")},
            "universe": [{"symbol": r["symbol"], "turnover24h_musd": round(r["turnover24h"] / 1e6, 1), "spread_bp": round(r["spread_bp"], 2)} for r in snap],
            "n_present": len(pr), "absent": absent, "tokens": toks}
    p = journal_dir() / f"{S:%Y-%m-%d}.jsonl"
    seen = set()
    if p.exists():
        seen = {json.loads(x)["id"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}
    if line["id"] not in seen:
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    sc = score_closed()
    write_latest(line, sc)
    return line


# ------------------------------------------------------------------ scoring


def _closes(sym: str) -> dict[int, float]:
    from ai_trader.research.xs_v4_data import root

    p = root() / "raw" / f"kline_{sym}.parquet"
    if not p.exists():
        return {}
    k = pl.read_parquet(p, columns=["ts", "close"])
    return dict(zip((k["ts"] // 1000 + 3600).to_list(), k["close"].to_list(), strict=True))   # close time (s) -> close


def _spearman(a: list[float], b: list[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(a) < 5:
        return None
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None


def score_closed() -> dict:
    jd = journal_dir()
    sp = jd / "scored.json"
    done = set(json.loads(sp.read_text())) if sp.exists() else set()
    lines = []
    for f in sorted(jd.glob("20*.jsonl")):
        lines += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    cache: dict[str, dict] = {}
    new = []
    for ln in lines:
        S = datetime.fromisoformat(ln["S"])
        s0 = int(S.timestamp())
        for h in (1, 4, 24):
            key = f"{ln['id']}|{h}"
            if key in done:
                continue
            if h == 4 and S.hour % 4:
                continue
            if h == 24 and S.hour:
                continue
            toks = ln["tokens"]
            R = {}
            for s in toks:
                if s not in cache:
                    cache[s] = _closes(s)
                c0, c1 = cache[s].get(s0), cache[s].get(s0 + 3600 * h)
                if c0 and c1:
                    R[s] = math.log(c1 / c0)
            if s0 + 3600 * h > max((max(v) if v else 0) for v in cache.values()):
                continue                                   # window not closed yet
            row = {"id": ln["id"], "S": ln["S"], "h": h, "n": len(R)}
            if len(R) >= 5:
                m = float(np.mean(list(R.values())))
                ss = sorted(R)
                if h in (4, 24):
                    rel = [R[s] - m for s in ss]
                    row["ic_v4"] = _spearman([toks[s][f"rel{h}_bp"]["q50"] for s in ss], rel)
                    row["ic_lowvol"] = _spearman([-toks[s]["lsig2"] for s in ss], rel)
                    if row["ic_v4"] is not None and row["ic_lowvol"] is not None:
                        row["diff"] = row["ic_v4"] - row["ic_lowvol"]
                if h == 1:
                    from ai_trader.ml import xs_net as xs

                    t = np.asarray(xs.TAUS)
                    pv, ph = [], []
                    for s in ss:
                        y = abs(R[s]) / math.sqrt(toks[s]["s2"])
                        for qs, acc in ((toks[s]["vol1h_model"], pv), (toks[s]["har1h_model"], ph)):
                            e = y - np.asarray(qs)
                            acc.append(float(np.maximum(t * e, (t - 1) * e).mean()))
                    row["pin_v4"], row["pin_har"] = float(np.mean(pv)), float(np.mean(ph))
            new.append(row)
            done.add(key)
    if new:
        with (jd / "scores.jsonl").open("a", encoding="utf-8") as fh:
            for r in new:
                fh.write(json.dumps(r) + "\n")
        sp.write_text(json.dumps(sorted(done)), encoding="utf-8")
    return summary()


def summary() -> dict:
    p = journal_dir() / "scores.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
    out = {}
    for h in (4, 24):
        r = [x for x in rows if x["h"] == h and x.get("diff") is not None]
        out[f"{h}h"] = {"n": len(r), "ic_v4": float(np.mean([x["ic_v4"] for x in r])) if r else None,
                        "ic_lowvol": float(np.mean([x["ic_lowvol"] for x in r])) if r else None,
                        "diff": float(np.mean([x["diff"] for x in r])) if r else None}
    r1 = [x for x in rows if x["h"] == 1 and "pin_v4" in x]
    out["1h_vol"] = {"n": len(r1), "pin_v4": float(np.mean([x["pin_v4"] for x in r1])) if r1 else None,
                     "pin_har": float(np.mean([x["pin_har"] for x in r1])) if r1 else None}
    return out


def write_latest(line: dict, sc: dict) -> None:
    toks = line["tokens"]
    by24 = sorted(toks, key=lambda s: toks[s]["rank24"])
    bylv = sorted(toks, key=lambda s: toks[s]["rank_lowvol"])
    f = lambda x: "n/a" if x is None else f"{x:+.3f}"  # noqa: E731
    L = [f"# xs.v4 shadow line (not LATEST.md)", "",
         f"_{NOT_A_SIGNAL}. H-XS4 forward test (rank IC vs the low-vol ranking). Updated {line['as_of'][:16]} UTC, forecast hour S = {line['S'][:16]} UTC._", "",
         f"- universe now: {len(line['universe'])} perps pass 24h turnover >= 20 M USD and spread <= 5 bp; present in the panel: {line['n_present']}",
         f"- absent: {', '.join(f'{k} ({v})' for k, v in line['absent'].items()) or 'none'}",
         f"- model: trained {line['model']['spec_trained_at'][:16]}, data to {line['model']['as_of_data'][:16]}, weights {list(line['model']['weights_sha256'].values())[0]}",
         "", "| token | sector | 24h rel q10 / q50 / q90 (bp) | rank 24h | rank 4h | low-vol rank | 1h |r| q50 / q90 % (v4) | log-HAR-t q50 / q90 % |", "|---|---|---|---|---|---|---|---|"]
    for s in by24:
        t = toks[s]
        r = t["rel24_bp"]
        L.append(f"| {s} | {t['sector']} | {r['q10']:+.0f} / {r['q50']:+.0f} / {r['q90']:+.0f} | {t['rank24']} | {t['rank4']} | {t['rank_lowvol']} | "
                 f"{t['vol1h_pct'][3]:.2f} / {t['vol1h_pct'][5]:.2f} | {t['har1h_pct'][3]:.2f} / {t['har1h_pct'][5]:.2f} |")
    L += ["", f"Top 5 by 24h median: {', '.join(by24[:TOP])}. Bottom 5: {', '.join(by24[-TOP:])}. Calmest 5 (low-vol): {', '.join(bylv[:TOP])}.", "",
          "Forward scores (H-XS4; evaluated once at 60 daily origins):", "",
          f"- 4h origins: n {sc['4h']['n']}, mean IC v4 {f(sc['4h']['ic_v4'])}, low-vol {f(sc['4h']['ic_lowvol'])}, diff {f(sc['4h']['diff'])}",
          f"- 24h origins: n {sc['24h']['n']}, mean IC v4 {f(sc['24h']['ic_v4'])}, low-vol {f(sc['24h']['ic_lowvol'])}, diff {f(sc['24h']['diff'])}",
          f"- 1h vol pinball: n {sc['1h_vol']['n']}, v4 {f(sc['1h_vol']['pin_v4'])}, log-HAR-t {f(sc['1h_vol']['pin_har'])}", ""]
    p = _root() / cfg()["latest"]
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text("\n".join(L), encoding="utf-8")
    tmp.replace(p)


def safe_run() -> dict | None:
    try:
        return run_line()
    except Exception:  # noqa: BLE001 - a shadow failure must never propagate
        with (journal_dir() / "errors.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(UTC).isoformat()}\n{traceback.format_exc()}\n")
        return None


def next_run(now: datetime, minute: int = 2) -> datetime:
    t = now.replace(minute=minute, second=30, microsecond=0)
    return t if t > now else t + timedelta(hours=1)
