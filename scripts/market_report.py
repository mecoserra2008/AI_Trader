"""Daily per-token market state report (plan 2026-11-10 section 5), point in time.

    PYTHONIOENCODING=utf-8 uv run python scripts/market_report.py [--as-of 2026-11-10T22:00Z] [--symbols BTCUSDT XAUUSDT]
        [--workers 4] [--no-diary] [--a4] [--out-dir DIR (test run: no logs/, use with --no-diary)]

Universe (default): the 23 crypto perps of research.trend_zoo_study.UNIVERSE + TradFi perps (instrument symbol_type
stock / ETF / commodity / forex) with >= 30 days of 1h bars known at as_of + the 50 symbols of config/mm_multi.yaml
(short-history ones are marked). Default as_of = now - 5 minutes (flow_3s is flushed every 5 minutes).

Writes research/reports/market_state/<YYYY-MM-DD>.{md,json} (date of as_of; a later run the same day overwrites),
appends one line per token to logs/market_state.log and one diary observation. A symbol with missing data gets a card
with flags, never a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trader.core.config import load  # noqa: E402
from ai_trader.core.time import utc  # noqa: E402
from ai_trader.storage.lake import Lake  # noqa: E402
from ai_trader.tools import market_state as ms  # noqa: E402

MIN_TRADFI_DAYS = 30
HIGH_VOL_PCT = 0.8
SURPRISE_HI = 1.5
FLOW_ANOM_MIN = 3
FLOW_RATE_X = 2.0   # anomalous flow = share of scored minutes with |z| > 4 at least 2x the universe median share

_LAKE: Lake | None = None


def _init_worker() -> None:
    global _LAKE
    _LAKE = Lake()


def _card(args) -> dict:
    sym, as_of, liq, inst, fb = args
    try:
        return ms.state_card(sym, as_of, lake=_LAKE, flow=False, liquidations=liq, instrument=inst, instruments_fallback=fb)
    except Exception as e:  # noqa: BLE001
        return {"symbol": sym, "as_of": as_of.isoformat(), "error": f"{type(e).__name__}: {e}", "flags": ["error"], "price": None}


# ---------------------------------------------------------------- universe

def bar_history(lake: Lake, as_of: datetime) -> dict[str, dict]:
    """First 1h bar and count per symbol (bybit last-price bars known at as_of): an aggregate, so a direct query."""
    sql = f"""SELECT symbol, min(event_ts) AS first, count(*) AS n FROM read_parquet('{lake._glob("bars")}', hive_partitioning = true,
              union_by_name = true) WHERE venue = 'bybit' AND price_type = 'last' AND freq = '1h' AND known_ts <= ? GROUP BY symbol"""
    d = pl.from_arrow(lake.con.sql(sql, params=[as_of]).to_arrow_table())
    return {r["symbol"]: {"first": r["first"], "n": r["n"]} for r in d.iter_rows(named=True)}


def universe(lake: Lake, as_of: datetime, inst: dict[str, dict]) -> tuple[list[str], dict[str, list[str]]]:
    from ai_trader.research.trend_zoo_study import UNIVERSE

    hist = bar_history(lake, as_of)
    tradfi = sorted(s for s, r in inst.items() if r.get("symbol_type") in ms.TRADFI_TYPES and s in hist
                    and hist[s]["n"] >= MIN_TRADFI_DAYS * 24)
    mm = yaml.safe_load((ROOT / "config" / "mm_multi.yaml").read_text(encoding="utf-8"))["symbols"]
    groups = {"trend_zoo": list(UNIVERSE), "tradfi_30d": tradfi, "mm_multi": list(mm)}
    seen: list[str] = []
    for g in groups.values():
        for s in g:
            if s not in seen:
                seen.append(s)
    return seen, groups


# ---------------------------------------------------------------- formatting

def _n(x, fmt="{:.2f}", na="-"):
    return na if x is None else fmt.format(x)


def _pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def _usd(x):
    if x is None:
        return "-"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k"
    return f"{x:.0f}"


def _g(c: dict, *path):
    for p in path:
        if not isinstance(c, dict):
            return None
        c = c.get(p)
    return c


def short_flags(c: dict) -> str:
    keep = []
    for f in c.get("flags", []):
        if f.startswith("burn_in_skipped"):
            f = "no_burn_in"
        keep.append(f)
    return " ".join(keep)


def log_line(c: dict) -> str:
    def v(x, nd=4):
        return "na" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))

    kv = {
        "cls": c.get("asset_class"), "px": c.get("price_flow") or c.get("price"),
        "r1h": _g(c, "returns", "1h", "sigma_units"), "r24h": _g(c, "returns", "24h", "sigma_units"), "r7d": _g(c, "returns", "7d", "sigma_units"),
        "cell": _g(c, "regime", "cell"), "sd_rb": _g(c, "vol", "robust_sigma_day"), "sd_har": _g(c, "vol", "har_sigma_day"),
        "vpct": _g(c, "vol", "vol_pct_1y"), "vsurp": _g(c, "vol", "vol_surprise"),
        "dd": None if _g(c, "asymmetry", "drawdown_flag") is None else int(_g(c, "asymmetry", "drawdown_flag")),
        "dds": _g(c, "asymmetry", "drawdown_sigma30"), "rsneg5": _g(c, "asymmetry", "rs_neg_share_5d"),
        "jumps7d": _g(c, "jumps", "n_jumps_lm_7d"), "fund": _g(c, "funding", "rate"), "fz": _g(c, "funding", "z_90d"),
        "oi24": _g(c, "oi", "chg_24h_log"), "liqL": _g(c, "liquidations_24h", "long_usd"), "liqS": _g(c, "liquidations_24h", "short_usd"),
        "fan": _g(c, "flow", "n_anom_24h"), "fcov": _g(c, "flow", "coverage_24h"),
        "zb": _g(c, "zones_4h", "below", "dist_pct"), "za": _g(c, "zones_4h", "above", "dist_pct"),
        "hist_d": c.get("history_days"), "flags": ",".join(c.get("flags", [])) or "none",
    }
    if c.get("error"):
        kv = {"error": c["error"].replace(" ", "_")[:120], "flags": "error"}
    return f"{c.get('as_of')} {c['symbol']} " + " ".join(f"{k}={v(x)}" for k, x in kv.items())


def table(cards: list[dict]) -> list[str]:
    head = ("| symbol | px | r1h σ | r24h σ | r7d σ | cell | σ/day rb | σ/day HAR | vol pct 1y | surprise | DD (σ30) | RS- 5d | "
            "fund z | OI 24h | liq L / S 24h | flow anom (scored) | zone below / above | flags |")
    out = [head, "|" + "|".join(["---"] * (head.count("|") - 1)) + "|"]
    for c in cards:
        if c.get("price") is None:
            out.append(f"| {c['symbol']} | - | | | | | | | | | | | | | | | | {c.get('error') or short_flags(c)} |")
            continue
        dd = _g(c, "asymmetry", "drawdown_flag")
        dds = _g(c, "asymmetry", "drawdown_sigma30")
        fz = _g(c, "funding", "z_90d")
        crowd = _g(c, "funding", "crowded")
        fl = c.get("flow") or {}
        zb, za = _g(c, "zones_4h", "below", "dist_pct"), _g(c, "zones_4h", "above", "dist_pct")
        px = c.get("price_flow") or c.get("price")
        out.append("| " + " | ".join([
            f"**{c['symbol']}**" if (dd or (_g(c, 'vol', 'vol_surprise') or 0) >= SURPRISE_HI) else c["symbol"],
            f"{px:.6g}",
            _n(_g(c, "returns", "1h", "sigma_units"), "{:+.1f}"), _n(_g(c, "returns", "24h", "sigma_units"), "{:+.1f}"),
            _n(_g(c, "returns", "7d", "sigma_units"), "{:+.1f}"),
            f"{_g(c, 'regime', 'cell') or '-'}",
            _pct(_g(c, "vol", "robust_sigma_day"), 2), _pct(_g(c, "vol", "har_sigma_day"), 2),
            _n(_g(c, "vol", "vol_pct_1y"), "{:.2f}"), _n(_g(c, "vol", "vol_surprise"), "{:.2f}"),
            ("**DD** " if dd else "") + _n(dds, "{:.1f}"),
            _n(_g(c, "asymmetry", "rs_neg_share_5d"), "{:.2f}"),
            _n(fz, "{:+.1f}") + (f" ({crowd})" if crowd else ""),
            _pct(_g(c, "oi", "chg_24h_log"), 1),
            f"{_usd(_g(c, 'liquidations_24h', 'long_usd'))} / {_usd(_g(c, 'liquidations_24h', 'short_usd'))}",
            f"{_n(fl.get('n_anom_24h'), '{:d}')} ({fl.get('n_scored_24h', 0)})",
            f"{_n(zb, '{:+.1f}%')} / {_n(za, '{:+.1f}%')}",
            short_flags(c),
        ]) + " |")
    return out


def _anom_rate(c: dict) -> float | None:
    n, k = _g(c, "flow", "n_scored_24h"), _g(c, "flow", "n_anom_24h")
    return k / n if n and k is not None and n >= 120 else None


def summarise(cards: list[dict]) -> dict:
    by: dict[str, dict] = {}
    rates = sorted(r for r in (_anom_rate(c) for c in cards if c.get("price") is not None) if r is not None)
    med_rate = rates[len(rates) // 2] if rates else None
    for c in cards:
        if c.get("price") is None:
            continue
        cls = c.get("asset_class") or "unknown"
        b = by.setdefault(cls, {"n": 0, "high_vol": [], "vol_surprise": [], "drawdown": [], "anomalous_flow": [], "crowded_funding": [],
                                "cells": {}, "r24h_median": None, "_r24": []})
        b["n"] += 1
        cell = _g(c, "regime", "cell") or "none"
        b["cells"][cell] = b["cells"].get(cell, 0) + 1
        s = c["symbol"]
        p = _g(c, "vol", "vol_pct_1y")
        if p is not None and p >= HIGH_VOL_PCT:
            b["high_vol"].append((s, p))
        vs = _g(c, "vol", "vol_surprise")
        if vs is not None and vs >= SURPRISE_HI:
            b["vol_surprise"].append((s, vs))
        if _g(c, "asymmetry", "drawdown_flag"):
            b["drawdown"].append((s, _g(c, "asymmetry", "drawdown_sigma30")))
        fa, ra = _g(c, "flow", "n_anom_24h"), _anom_rate(c)
        if fa is not None and ra is not None and med_rate and fa >= FLOW_ANOM_MIN and ra >= FLOW_RATE_X * med_rate:
            b["anomalous_flow"].append((s, fa))
        fz = _g(c, "funding", "z_90d")
        if _g(c, "funding", "crowded"):
            b["crowded_funding"].append((s, fz, _g(c, "funding", "crowded")))
        r = _g(c, "returns", "24h", "sigma_units")
        if r is not None:
            b["_r24"].append(r)
    for b in by.values():
        r = sorted(b.pop("_r24"))
        b["r24h_median"] = r[len(r) // 2] if r else None
        b["flow_anom_rate_universe_median"] = med_rate
        for k in ("high_vol", "vol_surprise", "drawdown", "anomalous_flow"):
            b[k].sort(key=lambda t: -(t[1] or 0))
    return by


def summary_lines(by: dict) -> list[str]:
    out = []
    for cls in sorted(by):
        b = by[cls]
        cells = ", ".join(f"{k} {v}" for k, v in sorted(b["cells"].items()))
        out.append(f"**{cls}** ({b['n']} tokens; median 24h return {_n(b['r24h_median'], '{:+.2f}')} σ; cells: {cells})")
        out.append(f"- high vol (1y pct >= {HIGH_VOL_PCT}): " + (", ".join(f"{s} {p:.2f}" for s, p in b["high_vol"]) or "none"))
        out.append(f"- vol surprise (yesterday RV / HAR >= {SURPRISE_HI}): " + (", ".join(f"{s} {v:.2f}" for s, v in b["vol_surprise"]) or "none"))
        out.append("- drawdown (below 30d high by > 1.5 σ30·√30): " + (", ".join(f"{s} {v:.1f}" for s, v in b["drawdown"]) or "none"))
        mr = b.get("flow_anom_rate_universe_median")
        out.append(f"- anomalous flow (share of scored minutes with imbalance |z|>4 >= {FLOW_RATE_X:.0f}x the universe median "
                   f"{_pct(mr, 0)}; count in 24h): " + (", ".join(f"{s} {v}" for s, v in b["anomalous_flow"]) or "none"))
        out.append("- crowded funding (|z 90d| >= 2 with the rate on the same side): "
                   + (", ".join(f"{s} z {z:+.1f} ({w})" for s, z, w in b["crowded_funding"]) or "none"))
        out.append("")
    return out


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-diary", action="store_true")
    ap.add_argument("--a4", action="store_true", help="flow counts from features.flow_anomaly.latest_anomalies (slow; default local)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    t_start = time.time()
    as_of = utc(a.as_of) if a.as_of else datetime.now(UTC).replace(microsecond=0) - timedelta(minutes=5)
    lake = Lake()
    inst, fb = ms.instruments(as_of, lake)
    if a.symbols:
        syms, groups = list(a.symbols), {"cli": list(a.symbols)}
    else:
        syms, groups = universe(lake, as_of, inst)
    liq = ms.liquidations_window(as_of, syms, lake=lake)
    liq_first = None
    try:
        lf = pl.from_arrow(lake.con.sql(f"SELECT min(event_ts) AS t FROM read_parquet('{lake._glob('liquidations')}', hive_partitioning = true) "
                                        "WHERE known_ts <= ?", params=[as_of]).to_arrow_table())["t"][0]
        liq_first = utc(lf) if lf is not None else None
    except Exception:  # noqa: BLE001
        pass

    jobs = [(s, as_of, liq.filter(pl.col("symbol") == s), inst.get(s), fb) for s in syms]
    workers = max(1, min(4, a.workers))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        fut = ex.map(_card, jobs, chunksize=2)
        t_flow = time.time()
        try:
            flows = ms.flow_anomalies(syms, as_of, lake=lake, use_a4=a.a4)
        except Exception as e:  # noqa: BLE001
            flows = {s: {"error": f"{type(e).__name__}: {e}"} for s in syms}
        t_flow = time.time() - t_flow
        cards = list(fut)
    for c in cards:
        ms.attach_flow(c, flows.get(c["symbol"], {}))

    order = {"tradfi": 1, "crypto": 0}
    cards.sort(key=lambda c: (order.get(c.get("asset_class"), 2), c["symbol"]))
    by = summarise(cards)
    gaps = {
        "no_1h_bars": [c["symbol"] for c in cards if c.get("price") is None],
        "short_history": [c["symbol"] for c in cards if "short_history" in c.get("flags", [])],
        "stale_last_bar": [c["symbol"] for c in cards if "stale_last_bar" in c.get("flags", [])],
        "no_flow": [c["symbol"] for c in cards if "no_flow" in c.get("flags", [])],
        "no_funding": [c["symbol"] for c in cards if "no_funding" in c.get("flags", [])],
        "no_oi": [c["symbol"] for c in cards if "no_oi" in c.get("flags", [])],
        "stale_oi": [c["symbol"] for c in cards if "stale_oi" in c.get("flags", [])],
        "har_unavailable": [c["symbol"] for c in cards if "har_unavailable" in c.get("flags", [])],
        "errors": {c["symbol"]: c["error"] for c in cards if c.get("error")},
    }
    covs = [_g(c, "flow", "coverage_24h") for c in cards if _g(c, "flow", "coverage_24h") is not None]
    scored = [_g(c, "flow", "n_scored_24h") for c in cards if _g(c, "flow", "n_scored_24h") is not None]
    flow_src = next((_g(c, "flow", "source") for c in cards if _g(c, "flow", "source")), None)
    a4_err = next((_g(c, "flow", "a4_error") for c in cards if _g(c, "flow", "a4_error")), None)
    elapsed = time.time() - t_start

    root = load().root
    out_dir = Path(a.out_dir) if a.out_dir else root / "research" / "reports" / "market_state"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{as_of:%Y-%m-%d}"
    meta = {"as_of": as_of.isoformat(), "generated": datetime.now(UTC).isoformat(timespec="seconds"), "tool": "state.market_card",
            "version": ms.VERSION, "n_symbols": len(syms), "groups": groups, "flow_source": flow_src, "a4_error": a4_err,
            "flow_coverage_median": sorted(covs)[len(covs) // 2] if covs else None,
            "flow_scored_minutes_median": sorted(scored)[len(scored) // 2] if scored else None,
            "liquidations_first_ts": liq_first.isoformat() if liq_first else None, "instruments_not_pit": fb,
            "elapsed_s": round(elapsed, 1), "flow_elapsed_s": round(t_flow, 1)}
    (out_dir / f"{stem}.json").write_text(json.dumps({"meta": meta, "summary": by, "data_gaps": gaps, "cards": cards}, indent=1, default=str),
                                          encoding="utf-8")

    md = [f"# Market state {as_of:%Y-%m-%d %H:%M} UTC", "",
          f"_Tool `state.market_card` v{ms.VERSION} (src/ai_trader/tools/market_state.py), point in time as of {as_of.isoformat()}. "
          f"{len(syms)} tokens. Descriptive only: no field here is a tested trading signal. Generated in {elapsed:.0f} s._", "",
          "Columns: returns in σ units of the robust risk σ known at the start of each window; cell = v0 regime on 1h bars; "
          "σ/day rb = robust_bc risk σ × √24; HAR = next-UTC-day σ from 1h RV; vol pct = 1y percentile of the robust σ; "
          "surprise = yesterday's realised vol / its HAR forecast; DD = log drawdown from the 30d high in σ30·√30 units (bold if > 1.5); "
          "RS- 5d = downside semivariance share of RV over 5 days; fund z = latest funding vs the trailing 90 days; OI 24h = log "
          "change of the 4h OI series; flow anom = 1m signed taker imbalance minutes with |robust z| > 4 in 24h (scored minutes); "
          "zones = distance to the nearest 4h structural zone edge.", "",
          "## Summary by asset class", "", *summary_lines(by),
          "## Data gaps", ""]
    for k, v in gaps.items():
        if v:
            md.append(f"- {k}: " + (", ".join(v) if isinstance(v, list) else json.dumps(v)))
    md += [f"- flow: source `{flow_src}`; median 24h flow_3s coverage {_pct(meta['flow_coverage_median'], 0)}, median scored minutes "
           f"{meta['flow_scored_minutes_median']} (collector gaps leave many minutes without a full trailing baseline)"
           + (f"; A4 module error: {a4_err}" if a4_err else ""),
           f"- liquidations collected since {meta['liquidations_first_ts']}" + ("" if not liq_first or liq_first <= as_of - timedelta(hours=24)
                                                                               else " (24h window is PARTIAL)"),
           "", "## Table", "", *table(cards), ""]
    (out_dir / f"{stem}.md").write_text("\n".join(md), encoding="utf-8")

    log = (out_dir / "market_state.log") if a.out_dir else root / "logs" / "market_state.log"  # a test run never touches logs/
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        for c in cards:
            fh.write(log_line(c) + "\n")

    if not a.no_diary:
        from ai_trader.research.diary import entry

        body = [f"State cards for {len(syms)} tokens as of {as_of:%Y-%m-%d %H:%M} UTC (scripts/market_report.py, state.market_card "
                f"v{ms.VERSION}). Descriptive, no signals."]
        for cls in sorted(by):
            b = by[cls]
            top_cells = ", ".join(f"{k} {v}" for k, v in sorted(b["cells"].items(), key=lambda kv: -kv[1])[:3])
            body.append(
                f"{cls} ({b['n']}): median 24h {_n(b['r24h_median'], '{:+.2f}')} σ; cells {top_cells}; "
                f"high vol {', '.join(s for s, _ in b['high_vol'][:8]) or 'none'}; "
                f"vol surprise {', '.join(f'{s} {v:.2f}' for s, v in b['vol_surprise'][:6]) or 'none'}; "
                f"drawdown {', '.join(s for s, _ in b['drawdown'][:8]) or 'none'}; "
                f"anomalous flow {', '.join(f'{s} {v}' for s, v in b['anomalous_flow'][:6]) or 'none'}; "
                f"crowded funding {', '.join(f'{s} ({w})' for s, _, w in b['crowded_funding']) or 'none'}.")
        body.append(f"Data gaps: {len(gaps['no_1h_bars'])} without usable 1h bars, {len(gaps['short_history'])} short history (<30d), "
                    f"median flow coverage {_pct(meta['flow_coverage_median'], 0)} of the last 24h.")
        entry("observation", f"Market state {as_of:%Y-%m-%d %H:%M} UTC", "\n\n".join(body), tags=["market_state"],
              links=[f"research/reports/market_state/{stem}.md"])
    sys.stdout.write(f"{len(cards)} cards in {elapsed:.0f}s (flow {t_flow:.0f}s) -> {out_dir / stem}.md\n")
    sys.stdout.write("\n".join(summary_lines(by)) + "\n")


if __name__ == "__main__":
    main()
