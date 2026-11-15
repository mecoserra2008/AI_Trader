"""Cross-section domain (xs.json): relative strength, funding, OI, basis, sectors, v0 regime cells and the v3 context.

Conventions (user convention: every spread or relative measure is a log price ratio)
  ret_i(h)    = ln(C_i(T) / C_i(T - h)), T = the last 1h close <= as_of, closes of fully closed 1h bars only
  basket(h)   = mean of ret_j(h) over the basket members (crypto perps of the universe with both closes): the log return
                of the geometric equal-weight index B = prod_j (C_j / C_j(T - h))^(1/n)
  rs_i(h)     = ret_i(h) - basket(h) = Delta ln(P_i / B): the change of the log price ratio of token i to the basket
  rs_sigma    = rs_i(h) / (sd_i * sqrt(h)), sd_i = std of token i's hourly relative log returns (vs the hourly basket mean)
                over the last 168 hours (vol-standardised, house rule)
  TradFi perps (stock / ETF / commodity / forex) are ranked against their own equal-weight TradFi basket when it has >= 3
  members, never mixed into the crypto basket.
  Funding: last settled rate, its mid-rank percentile among the settlements of the last 90 days that are known at as_of
  (REST history, <= 200 settlements, so the window can be shorter: n and window_days are reported), the predicted
  next rate from the ticker, and annualised rates = rate x (24 / interval_h) x 365 (positive = longs pay shorts; a short
  collects it). OI changes are log changes (%) of the 1h OI series. Basis = 1e4 ln(mark / index) and 1e4 ln(last / index).
  None of this is a signal: R-XS-2 / R-XS-3 (relative momentum ranks backwards and the quintile spread loses after costs),
  R-AVOID-1 (funding contrarian and OI-price divergence failed), R-CARRY-1 (carry premium has decayed).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.desk.public import closed

HORIZONS = {"1h": 1, "4h": 4, "24h": 24, "7d": 168}
VOL_WINDOW_H = 168
FUNDING_WINDOW = timedelta(days=90)
TRADFI_TYPES = ("stock", "ETF", "commodity", "forex")
EXCLUDE = frozenset({"SPYUSDT"})
FLAG_RULES = {  # deterministic reasons for the geometry agent to look at a token (not signals)
    "rs_sigma_24h_abs": 2.0,
    "rs_sigma_4h_abs": 2.5,
    "funding_pct_hi": 0.95,
    "funding_pct_lo": 0.05,
    "oi_24h_abs_pct": 10.0,
    "basis_abs_bp": 25.0,
}
NOT_A_SIGNAL = "context, not a signal: R-XS-2 / R-XS-3 / R-AVOID-1 / R-CARRY-1"


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _r(x, nd: int = 2):
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def pct_rank(x: np.ndarray, v: float) -> float | None:
    """Mid-rank percentile of v among x (same definition as live.players._pct)."""
    x = x[np.isfinite(x)]
    if not len(x) or not math.isfinite(v):
        return None
    return float((np.sum(x < v) + 0.5 * np.sum(x == v)) / len(x))


# ---------------------------------------------------------------- universe

def asset_class(inst: dict | None) -> str:
    return "tradfi" if (inst or {}).get("symbolType") in TRADFI_TYPES else "crypto"


def base_of(sym: str) -> str:
    b = sym[:-4] if sym.endswith("USDT") else sym
    for p in ("1000000", "10000", "1000"):
        if b.startswith(p) and len(b) > len(p):
            return b[len(p):]
    return b


def select_universe(tickers: list[dict], instruments: list[dict], n: int = 60) -> list[dict]:
    """Top-n USDT linear perpetuals by 24h turnover, status Trading, never SPYUSDT."""
    inst = {i["symbol"]: i for i in instruments}
    rows = []
    for t in tickers:
        s = t.get("symbol", "")
        i = inst.get(s)
        if not s.endswith("USDT") or s in EXCLUDE or i is None:
            continue
        if i.get("contractType") != "LinearPerpetual" or i.get("status") != "Trading" or i.get("quoteCoin") != "USDT":
            continue
        to = _f(t.get("turnover24h")) or 0.0
        rows.append({"symbol": s, "turnover_24h_usd": to, "ticker": t, "instrument": i})
    rows.sort(key=lambda r: -r["turnover_24h_usd"])
    return rows[:n]


def sector_of(sym: str, inst: dict | None, v3_sector: str | None, yaml_sectors: dict[str, str]) -> str:
    st = (inst or {}).get("symbolType")
    if st in TRADFI_TYPES:
        return f"tradfi_{st.lower()}"
    if sym == "BTCUSDT":
        return "btc"
    if v3_sector:
        return v3_sector
    return yaml_sectors.get(sym) or "other"


def load_sectors(root: Path) -> dict[str, str]:
    import yaml

    p = root / "config" / "sectors.yaml"
    if not p.is_file():
        return {}
    return dict((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("sectors", {}))


# ---------------------------------------------------------------- relative strength

def close_grid(bars: dict[str, pl.DataFrame], as_of: datetime, n_hours: int = VOL_WINDOW_H + 168) -> tuple[datetime | None, dict[str, np.ndarray]]:
    """Hourly closes on a common grid ending at T = the latest 1h close_ts <= as_of over all symbols (NaN = missing)."""
    cl = {s: closed(b, as_of) for s, b in bars.items() if b is not None and b.height}
    ends = [b["close_ts"].max() for b in cl.values() if b.height]
    if not ends:
        return None, {}
    T = max(ends)
    grid = [T - timedelta(hours=k) for k in range(n_hours, -1, -1)]
    out = {}
    for s, b in cl.items():
        m = dict(zip(b["close_ts"].to_list(), b["close"].to_list(), strict=True))
        out[s] = np.array([m.get(t, np.nan) for t in grid], dtype=float)
    return T, out


def relative_strength(closes: dict[str, np.ndarray], basket: set[str], horizons: dict[str, int] = HORIZONS) -> dict:
    """rs_i(h) = ln(C_i(T)/C_i(T-h)) - mean_{j in basket} ln(C_j(T)/C_j(T-h)); closes[s][-1] is C(T), hourly grid."""
    syms = list(closes)
    if not syms:
        return {"basket_ret": {}, "tokens": {}}
    L = np.log(np.vstack([closes[s] for s in syms]))
    member = np.array([s in basket for s in syms])
    out_tok: dict[str, dict] = {s: {"ret": {}, "rs": {}, "rs_sigma": {}} for s in syms}
    bret = {}
    dr = np.diff(L[:, -(VOL_WINDOW_H + 1):], axis=1)                        # hourly log returns, last 168 h
    with np.errstate(invalid="ignore"):
        bh = np.array([np.nanmean(dr[member & np.isfinite(dr[:, t]), t]) if (member & np.isfinite(dr[:, t])).sum() >= 2 else np.nan
                       for t in range(dr.shape[1])])
        rel = dr - bh[None, :]
    sd = np.array([np.nanstd(r[np.isfinite(r)], ddof=1) if np.isfinite(r).sum() >= 48 else np.nan for r in rel])
    for name, h in horizons.items():
        if L.shape[1] <= h:
            continue
        r = L[:, -1] - L[:, -1 - h]
        ok = member & np.isfinite(r)
        b = float(np.mean(r[ok])) if ok.sum() >= 2 else np.nan
        bret[name] = b
        for i, s in enumerate(syms):
            if not math.isfinite(r[i]):
                continue
            rs = r[i] - b
            out_tok[s]["ret"][name] = float(r[i])
            out_tok[s]["rs"][name] = float(rs) if math.isfinite(rs) else None
            out_tok[s]["rs_sigma"][name] = float(rs / (sd[i] * math.sqrt(h))) if math.isfinite(rs) and math.isfinite(sd[i]) and sd[i] > 0 else None
    for i, s in enumerate(syms):
        out_tok[s]["sd_rel_1h"] = float(sd[i]) if math.isfinite(sd[i]) else None
    return {"basket_ret": bret, "tokens": out_tok}


def ranks(values: dict[str, float | None]) -> dict[str, int]:
    """1 = largest value; None values are not ranked."""
    v = sorted(((x, s) for s, x in values.items() if x is not None and math.isfinite(x)), reverse=True)
    return {s: i + 1 for i, (_, s) in enumerate(v)}


# ---------------------------------------------------------------- funding, OI, basis

def annualise(rate: float | None, interval_h: float | None) -> float | None:
    if rate is None or not interval_h:
        return None
    return 100.0 * rate * (24.0 / interval_h) * 365.0


def funding_stats(hist: pl.DataFrame | None, as_of: datetime, *, predicted: float | None = None, interval_h: float | None = None) -> dict:
    """Last settled funding known at as_of, its 90d percentile / z, the 7d mean and annualised rates (% / yr)."""
    out: dict = {"status": "none", "predicted": predicted, "interval_h": interval_h, "ann_pct_predicted": _r(annualise(predicted, interval_h), 1)}
    if hist is None or hist.height == 0:
        return out
    h = hist.filter((pl.col("event_ts") <= as_of) & (pl.col("known_ts") <= as_of) & (pl.col("event_ts") > as_of - FUNDING_WINDOW)).sort("event_ts")
    if h.height == 0:
        return out
    r = h["rate"].cast(pl.Float64).to_numpy()
    ts = h["event_ts"].to_list()
    if interval_h is None and len(ts) > 1:
        interval_h = (ts[-1] - ts[-2]).total_seconds() / 3600
    last = float(r[-1])
    sd = float(np.std(r, ddof=1)) if len(r) > 2 else float("nan")
    m7 = h.filter(pl.col("event_ts") > as_of - timedelta(days=7))["rate"].mean()
    out.update({
        "status": "ok", "last": last, "last_ts": ts[-1].isoformat(), "pct": _r(pct_rank(r, last), 3),
        "z": _r((last - float(np.mean(r))) / sd, 2) if sd > 0 else None, "n": len(r),
        "window_days": _r((ts[-1] - ts[0]).total_seconds() / 86400, 1), "interval_h": interval_h,
        "mean_7d": float(m7) if m7 is not None else None,
        "ann_pct_last": _r(annualise(last, interval_h), 1), "ann_pct_7d": _r(annualise(float(m7) if m7 is not None else None, interval_h), 1),
        "ann_pct_predicted": _r(annualise(predicted, interval_h), 1),
    })
    return out


def oi_change(oi: pl.DataFrame | None, as_of: datetime, mark: float | None = None, hours=(1, 4, 24, 168)) -> dict:
    """Log changes (%) of open interest over each horizon, from snapshots known at as_of."""
    if oi is None or oi.height == 0:
        return {"status": "none"}
    d = oi.filter((pl.col("event_ts") <= as_of) & (pl.col("known_ts") <= as_of)).sort("event_ts")
    if d.height == 0:
        return {"status": "none"}
    ts, v = d["event_ts"].to_list(), d["oi"].to_numpy()
    out = {"status": "ok", "ts": ts[-1].isoformat(), "oi": float(v[-1]), "oi_usd": _r(float(v[-1]) * mark, 0) if mark else None, "chg_pct": {}}
    for h in hours:
        cut = ts[-1] - timedelta(hours=h)
        idx = [i for i, t in enumerate(ts) if t <= cut]
        if idx and v[idx[-1]] > 0 and v[-1] > 0:
            out["chg_pct"][f"{h}h" if h < 168 else "7d"] = _r(100 * math.log(v[-1] / v[idx[-1]]), 2)
    return out


def basis_bp(last: float | None, mark: float | None, index: float | None) -> dict:
    def lr(a, b):
        return _r(1e4 * math.log(a / b), 1) if a and b and a > 0 and b > 0 else None
    return {"mark_index": lr(mark, index), "last_index": lr(last, index)}


# ---------------------------------------------------------------- regime and v3

def enrich_min(bars: pl.DataFrame, lam: float = 0.94) -> pl.DataFrame:
    """The columns research.regimes.label reads (ret, sigma_next), computed exactly as research.frame.enrich_bars does
    (features.transforms.ewma_sigma_ahead + one more EWMA step). A local copy because importing research.frame pulls in
    statsmodels (~6 s cold); tests/test_desk.py checks equality with enrich_bars."""
    bars = bars.sort("event_ts")
    close = bars["close"].to_numpy()
    ret = np.r_[np.nan, np.diff(np.log(close))]
    r0 = np.nan_to_num(ret, nan=0.0)
    warm = max(1, min(30, len(ret) - 1))
    sig_prev = np.full_like(r0, np.nan)
    if len(r0) > warm:
        var = float(np.nanmean(r0[:warm] ** 2))
        sig_prev[warm] = math.sqrt(var)
        for t in range(warm, len(r0) - 1):
            if math.isfinite(r0[t]):
                var = lam * var + (1 - lam) * r0[t] ** 2
            sig_prev[t + 1] = math.sqrt(var)
    sig_next = np.sqrt(lam * sig_prev ** 2 + (1 - lam) * r0 ** 2)
    return bars.with_columns(pl.Series("ret", ret), pl.Series("sigma_prev", sig_prev), pl.Series("sigma_next", sig_next))


def regime_cells(bars: dict[str, pl.DataFrame], as_of: datetime) -> dict[str, dict]:
    """v0 cell per symbol on closed 1h bars (research.regimes.label on enrich_min = research.frame.enrich_bars; the
    players.py call without the robust-vol columns label does not use). Needs ~90 days of bars; shorter histories give
    no cell. players.py reads the full history; the 200-day window used here gives the same last cell once the EWMA and
    the hysteresis have converged (180-day vol-quantile window + 20 days)."""
    from unittest import mock

    from ai_trader.core.config import load
    from ai_trader.research import regimes as RG
    CELL_LABELS = RG.CELL_LABELS
    cfg = load()

    def label(f, h):  # regimes.label re-reads the yaml config on every call; hand it one loaded config (same values)
        with mock.patch.object(RG, "load", lambda: cfg):
            return RG.label(f, h)

    out = {}
    for s, b in bars.items():
        b = closed(b, as_of) if b is not None else None
        if b is None or b.height < 2200:
            out[s] = {"status": "short history"}
            continue
        b = b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                     & (pl.col("low") <= pl.min_horizontal("open", "close")))
        try:
            f = label(enrich_min(b), 1.0)
        except Exception as e:  # noqa: BLE001
            out[s] = {"status": f"error: {type(e).__name__}"}
            continue
        last = f.tail(1).row(0, named=True)
        cl = dict(zip(f["close_ts"].to_list(), f["close"].to_list(), strict=True))
        tc = last["close_ts"]
        r24 = cl.get(tc - timedelta(hours=24))
        out[s] = {"status": "ok" if last["cell"] else "warm-up", "cell": last["cell"], "label": CELL_LABELS.get(last["cell"] or "", ""),
                  "dir_state": last["dir_state"], "vol_state": last["vol_state"], "bar_close": tc.isoformat(),
                  "close": float(last["close"]), "ret_24h_pct": 100 * math.log(last["close"] / r24) if r24 else None}
    return out


def read_v3_lines(journal_dir: Path, as_of: datetime, days: int = 2) -> list[dict]:
    out = []
    for k in range(days, -1, -1):
        p = journal_dir / f"{(as_of - timedelta(days=k)).date().isoformat()}.jsonl"
        if p.is_file():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip().startswith("{\"id\": \"xs3-"):
                    try:
                        out.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue
    return out


def v3_context(lines: list[dict], as_of: datetime) -> dict | None:
    """The latest v3 line with as_of <= as_of: per-token ranks, relative-return medians / ACI bands (bp), vol bands (%)."""
    ok = [x for x in lines if datetime.fromisoformat(x["as_of"]) <= as_of]
    if not ok:
        return None
    ln = max(ok, key=lambda x: x["as_of"])
    toks = {}
    for t in ln.get("tokens", []):
        if not t.get("present"):
            continue
        v, rel = t.get("vol", {}), t.get("rel", {})
        toks[t["symbol"]] = {
            "sector": t.get("sector"), "rank_4h": t.get("rank_4h"), "rank_24h": t.get("rank_24h"), "rank_lowvol": t.get("rank_lowvol"),
            "rel_4h_q50_bp": _r(rel.get("4h", {}).get("q_bp", [None, None, None])[1], 1), "rel_4h_aci80_bp": [_r(x, 0) for x in rel.get("4h", {}).get("aci_bp", [])],
            "rel_24h_q50_bp": _r(rel.get("24h", {}).get("q_bp", [None, None, None])[1], 1),
            "vol_1h_q50_pct": _r(v.get("1h", {}).get("v3", [None] * 7)[3], 3), "vol_1h_q90_pct": _r(v.get("1h", {}).get("v3", [None] * 7)[5], 3),
            "vol_4h_q50_pct": _r(v.get("4h", {}).get("v3", [None] * 7)[3], 3), "vol_4h_q90_pct": _r(v.get("4h", {}).get("v3", [None] * 7)[5], 3),
        }
    age = (as_of - datetime.fromisoformat(ln["as_of"])).total_seconds() / 60
    return {"id": ln["id"], "line_as_of": ln["as_of"], "age_min": round(age, 1), "note": ln.get("note"), "tokens": toks,
            "vol_units": "|r| quantiles in % (1h / 4h); rel = return minus the EW basket of the 23 universe-L tokens, bp"}


# ---------------------------------------------------------------- flags and assembly

def flags_for(tok: dict) -> list[str]:
    R, out = FLAG_RULES, []
    rs = tok.get("rs_sigma", {})
    if rs.get("24h") is not None and abs(rs["24h"]) >= R["rs_sigma_24h_abs"]:
        out.append(f"rs24h {rs['24h']:+.1f}sd")
    if rs.get("4h") is not None and abs(rs["4h"]) >= R["rs_sigma_4h_abs"]:
        out.append(f"rs4h {rs['4h']:+.1f}sd")
    fu = tok.get("funding", {})
    if fu.get("pct") is not None and (fu["pct"] >= R["funding_pct_hi"] or fu["pct"] <= R["funding_pct_lo"]):
        out.append(f"funding p{100 * fu['pct']:.0f}")
    oi = tok.get("oi", {}).get("chg_pct", {})
    if oi.get("24h") is not None and abs(oi["24h"]) >= R["oi_24h_abs_pct"]:
        out.append(f"OI24h {oi['24h']:+.1f}%")
    b = tok.get("basis_bp", {}).get("mark_index")
    if b is not None and abs(b) >= R["basis_abs_bp"]:
        out.append(f"basis {b:+.0f}bp")
    return out


def build_xs(as_of: datetime, universe: list[dict], bars1h: dict[str, pl.DataFrame], funding: dict[str, pl.DataFrame],
             oi: dict[str, pl.DataFrame], *, ticker_known: datetime | None = None, regimes: dict | None = None,
             v3: dict | None = None, yaml_sectors: dict | None = None) -> dict:
    """Assemble xs.json. Every input is filtered to as_of here (tickers only if ticker_known <= as_of)."""
    yaml_sectors = yaml_sectors or {}
    regimes = regimes or {}
    tick_ok = ticker_known is None or ticker_known <= as_of
    T, grid = close_grid({u["symbol"]: bars1h.get(u["symbol"]) for u in universe}, as_of)
    cls = {u["symbol"]: asset_class(u["instrument"]) for u in universe}
    crypto = {s for s, c in cls.items() if c == "crypto"}
    tradfi = {s for s, c in cls.items() if c == "tradfi"}
    rs_c = relative_strength({s: grid[s] for s in grid if s in crypto}, crypto)
    rs_t = relative_strength({s: grid[s] for s in grid if s in tradfi}, tradfi) if len(tradfi) >= 3 else {"basket_ret": {}, "tokens": {}}
    v3t = (v3 or {}).get("tokens", {})
    toks = []
    for u in universe:
        s, t, i = u["symbol"], u["ticker"] if tick_ok else {}, u["instrument"]
        last, mark, index = _f(t.get("lastPrice")), _f(t.get("markPrice")), _f(t.get("indexPrice"))
        ivh = _f(t.get("fundingIntervalHour")) or ((_f(i.get("fundingInterval")) or 0) / 60 or None)
        src = rs_t if s in tradfi else rs_c
        rr = src["tokens"].get(s, {"ret": {}, "rs": {}, "rs_sigma": {}})
        tok = {
            "symbol": s, "base": base_of(s), "asset_class": cls[s],
            "sector": sector_of(s, i, (v3t.get(s) or {}).get("sector"), yaml_sectors),
            "turnover_24h_usd": _r(u["turnover_24h_usd"], 0), "last": last,
            "basket": "tradfi_ew" if s in tradfi and rs_t["tokens"] else ("crypto_ew" if s in crypto else "none"),
            "ret_bp": {k: _r(1e4 * v, 1) for k, v in rr["ret"].items()},
            "rs_bp": {k: _r(1e4 * v, 1) if v is not None else None for k, v in rr["rs"].items()},
            "rs_sigma": {k: _r(v, 2) for k, v in rr["rs_sigma"].items()},
            "funding": funding_stats(funding.get(s), as_of, predicted=_f(t.get("fundingRate")), interval_h=ivh),
            "next_funding_ts": datetime.fromtimestamp(int(t["nextFundingTime"]) / 1000, UTC).isoformat() if t.get("nextFundingTime") else None,
            "oi": oi_change(oi.get(s), as_of, mark),
            "basis_bp": basis_bp(last, mark, index),
            "regime": regimes.get(s, {"status": "not in lake"}),
            "v3": v3t.get(s),
        }
        tok["flags"] = flags_for(tok)
        toks.append(tok)
    for h in HORIZONS:
        for group in (crypto, tradfi):
            rk = ranks({x["symbol"]: x["rs_bp"].get(h) for x in toks if x["symbol"] in group})
            for x in toks:
                if x["symbol"] in group:
                    x.setdefault("rank", {})[h] = rk.get(x["symbol"])
    return {
        "domain": "xs", "as_of": as_of.isoformat(), "bar_close": T.isoformat() if T else None,
        "convention": "log price ratios: ret = ln(C_T / C_T-h); rs = ret - EW-basket mean ret = Delta ln(P / geometric EW basket); bp",
        "note": NOT_A_SIGNAL,
        "basket": {"crypto_members": len([s for s in crypto if s in rs_c["tokens"]]), "crypto_ret_bp": {k: _r(1e4 * v, 1) for k, v in rs_c["basket_ret"].items()},
                   "tradfi_members": len(rs_t["tokens"]), "tradfi_ret_bp": {k: _r(1e4 * v, 1) for k, v in rs_t["basket_ret"].items()}},
        "flag_rules": FLAG_RULES, "flagged": [x["symbol"] for x in toks if x["flags"]],
        "v3": {k: v for k, v in (v3 or {}).items() if k != "tokens"} or None,
        "sectors": sector_aggregates(toks), "tokens": toks,
    }


def sector_aggregates(toks: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for t in toks:
        by.setdefault(t["sector"], []).append(t)
    out = []
    for sec, xs in by.items():
        row = {"sector": sec, "n": len(xs), "symbols": [x["symbol"] for x in xs], "rs_bp_mean": {}}
        for h in HORIZONS:
            v = [x["rs_bp"].get(h) for x in xs if x["rs_bp"].get(h) is not None]
            row["rs_bp_mean"][h] = _r(float(np.mean(v)), 1) if v else None
        fa = [x["funding"].get("ann_pct_last") for x in xs if x["funding"].get("ann_pct_last") is not None]
        oc = [x["oi"].get("chg_pct", {}).get("24h") for x in xs if x["oi"].get("chg_pct", {}).get("24h") is not None]
        row["funding_ann_pct_median"] = _r(float(np.median(fa)), 1) if fa else None
        row["oi_24h_pct_mean"] = _r(float(np.mean(oc)), 2) if oc else None
        out.append(row)
    out.sort(key=lambda r: -(r["rs_bp_mean"].get("24h") or -1e9))
    return out


__all__ = ["FLAG_RULES", "HORIZONS", "annualise", "base_of", "basis_bp", "build_xs", "close_grid", "funding_stats", "oi_change",
           "ranks", "regime_cells", "relative_strength", "select_universe", "sector_aggregates", "v3_context"]
