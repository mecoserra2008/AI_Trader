"""WS10 MICRO + COSTS: microstructure specifications and the execution-cost model (H-WS10-1..15).

Research only: no orders, no venue calls. Design frozen in research/hypotheses.yaml (H-WS10-1..15) and the diary
("WS10 design frozen before results") before any fit or cost measurement.

Data
----
* PART participant days (data/research/participants/days/<date>/{orders,types,book_1s}/<SYM>.parquet): 5 Bybit USDT perps,
  ms taker-order tape (same-ms same-side print runs) and the 1 s ob200 book state (row ts = state at ts).
  FIT = 12 pre-cutoff days + 2026-10-06..10-29, TEST = 2026-10-30..11-24 (clean_holdout).
* lake flow_3s (3 s candles, 2026-11-07..) for the 23 stress perps and Bybit spot BTC / ETH; the xvenue journal.

Passes (CLI):  uv run python -m ai_trader.research.micro_ws10 <cmd>
  day      per symbol-day reductions of the PART tape: intervals (impact fits), taker / pegged / stop simulations,
           spread decomposition, propagator statistics, metaorders  -> data/research/ws10/day/<date>/<SYM>_*.parquet|npz|json
  flow     per symbol-day reductions of flow_3s for the 23 stress perps -> data/research/ws10/flow/<date>.parquet
  (fits, cost table and re-pricing live in ai_trader.research.micro_ws10_fit)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
PART = ROOT / "data" / "research" / "participants" / "days"
OUT = ROOT / "data" / "research" / "ws10"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
TICK = {"BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.01, "XRPUSDT": 0.0001, "DOGEUSDT": 1e-5}
CUTOFF = "2026-07-01"
FIT_END = "2026-10-29"
MAKER, TAKER = 2.0, 5.5
QS = (50.0, 100.0, 200.0, 1000.0, 10000.0)
HS = (10, 60, 300)
TRAIL = 3600
PEG_Q = 100.0
PEG_T = 60
STOP_K = (0.5, 1.0, 2.0)
META_GAP_MS = 10_000
META_MIN = 4
PROP_L = 200
PROP_C = 1000
R_GRID = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000)


def split_of(d: str) -> str:
    return "PRE" if d < CUTOFF else ("FIT" if d <= FIT_END else "TEST")


def part_days() -> list[str]:
    out = []
    for p in sorted(PART.iterdir()):
        q = p / "qa.json"
        if q.exists() and json.loads(q.read_text()).get("complete"):
            out.append(p.name)
    return out


# ====================================================================== loading
def load(d: str, sym: str) -> tuple[dict, dict]:
    b = pl.read_parquet(PART / d / "book_1s" / f"{sym}.parquet").sort("ts")
    o = pl.read_parquet(PART / d / "orders" / f"{sym}.parquet")
    t = pl.read_parquet(PART / d / "types" / f"{sym}.parquet")
    if o.height != t.height or not (o["ms"] == t["ms"]).all():
        raise ValueError(f"types not aligned with orders {d} {sym}")
    o = o.with_columns(t["type"]).sort("ms", maintain_order=True)
    bk = {k: b[k].to_numpy() for k in ("ts", "bid1", "ask1", "mid")}
    for k in ("bid1_size", "ask1_size", "depth_bid_5", "depth_ask_5", "depth_bid_10", "depth_ask_10", "depth_bid_25", "depth_ask_25", "ofi"):
        bk[k] = b[k].to_numpy().astype(np.float64)
    bad = ~(np.isfinite(bk["bid1"]) & np.isfinite(bk["ask1"]) & (bk["ask1"] > bk["bid1"]) & (bk["bid1"] > 0))
    if bad.any():  # carry the last valid state forward (book gaps are rare: coverage 1.0 on PART days)
        idx = np.where(~bad, np.arange(len(bad)), 0)
        np.maximum.accumulate(idx, out=idx)
        for k in bk:
            if k not in ("ts", "ofi"):
                bk[k] = bk[k][idx]
        bk["ofi"] = np.where(bad, 0.0, bk["ofi"])
    bk["bad"] = bad
    od = {k: o[k].to_numpy() for k in ("ms", "side", "size", "notional", "n_prints", "first_price", "last_price", "type")}
    od["side"] = od["side"].astype(np.int64)
    return bk, od


# ====================================================================== helpers
def walk_premium(q: float, a1: np.ndarray, d5: np.ndarray, d10: np.ndarray, d25: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Average premium (bp vs the mid of the fill state) of a $q market order walking a piecewise-linear book:
    the first a1 dollars at hs, then linear in price to 5 / 10 / 25 bp at the cumulative band depths; beyond 25 bp the
    last density is extrapolated."""
    xs = [hs, np.full_like(hs, 5.0), np.full_like(hs, 10.0), np.full_like(hs, 25.0)]
    ds = [a1, np.maximum(a1, d5), np.maximum(np.maximum(a1, d5), d10), np.maximum(np.maximum(np.maximum(a1, d5), d10), d25)]
    xs[1] = np.maximum(xs[1], hs)
    xs[2] = np.maximum(xs[2], xs[1])
    xs[3] = np.maximum(xs[3], xs[2])
    tot = np.minimum(q, a1) * hs
    rem = np.maximum(q - a1, 0.0)
    lastx, lastd = hs, a1
    slope = np.full_like(hs, np.nan)
    for x, dd in zip(xs[1:], ds[1:], strict=True):
        width = dd - lastd
        take = np.minimum(rem, width)
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = np.where(width > 0, (x - lastx) / width, np.nan)
        tot = tot + take * lastx + 0.5 * np.where(width > 0, sl, 0.0) * take ** 2
        slope = np.where(width > 0, sl, slope)
        rem = rem - take
        lastx, lastd = np.where(width > 0, lastx + np.where(width > 0, sl, 0) * width, lastx), dd
    sl = np.where(np.isfinite(slope), slope, 25.0 / np.maximum(lastd, 1.0))
    tot = tot + rem * lastx + 0.5 * sl * rem ** 2
    return tot / q


def trailing(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing sums over rows (i-w, i] and the count of rows in the window (PIT at row i)."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(i + 1 - w, 0)
    return cs[i + 1] - cs[lo], (i + 1 - lo).astype(float)


def corr_fft(a: np.ndarray, b: np.ndarray, maxlag: int) -> np.ndarray:
    """s_k = sum_n a_{n+k} b_n for k = 0..maxlag (a = y, b = eps gives sum_n y_n eps_{n-k})."""
    n = len(a)
    m = 1 << int(math.ceil(math.log2(2 * n)))
    fa, fb = np.fft.rfft(a, m), np.fft.rfft(b, m)
    s = np.fft.irfft(fa * np.conj(fb), m)[: maxlag + 1]
    return s


# ====================================================================== the per symbol-day pass
def day_pass(d: str, sym: str, overwrite: bool = False) -> dict:
    od_dir = OUT / "day" / d
    done = od_dir / f"{sym}_stats.json"
    if done.exists() and not overwrite:
        return {"d": d, "sym": sym, "skipped": True}
    t0 = time.time()
    bk, o = load(d, sym)
    tick = TICK[sym]
    ts, bid, ask, mid = bk["ts"], bk["bid1"], bk["ask1"], bk["mid"]
    nb = len(ts)
    ms, osd, no, fp, lp, sz, typ = o["ms"], o["side"], o["notional"], o["first_price"], o["last_price"], o["size"], o["type"]
    l1b, l1a = bk["bid1_size"] * bid, bk["ask1_size"] * ask
    lm = np.log(mid)
    # orders -> book rows: row i holds (ts_i - 1000, ts_i]
    oi = np.searchsorted(ts, ms, "left")
    keep = oi < nb
    oi_k = oi[keep]
    flow_row = np.bincount(oi_k, weights=(osd * no)[keep], minlength=nb)
    vol_row = np.bincount(oi_k, weights=no[keep], minlength=nb)
    r1 = np.concatenate([[0.0], np.diff(lm)])
    s2, cnt = trailing(r1 ** 2, TRAIL)
    sig1 = np.sqrt(s2 / np.maximum(cnt - 1, 1))  # per-second sd of log mid changes, rows (i-3600, i]
    vbar = trailing(vol_row, TRAIL)[0] / cnt
    l1bar = trailing(0.5 * (l1b + l1a), TRAIL)[0] / cnt
    hour = ((ts // 1000) % 86400 // 3600).astype(np.int8)
    stats: dict = {"d": d, "sym": sym, "split": split_of(d), "n_orders": int(len(ms)), "n_rows": int(nb),
                   "bad_rows": int(bk["bad"].sum()), "notional": float(no.sum())}

    # ---------------------------------------------------------------- 1. impact intervals (h = 10, 60, 300 s)
    cs_f = np.concatenate([[0.0], np.cumsum(flow_row)])
    cs_v = np.concatenate([[0.0], np.cumsum(vol_row)])
    cs_o = np.concatenate([[0.0], np.cumsum(bk["ofi"])])
    ivs = []
    for h in HS:
        st = np.arange(600, nb - h, h)
        en = st + h
        ivs.append(pl.DataFrame({
            "h": np.full(len(st), h, np.int16), "hour": hour[st], "i": st.astype(np.int32),
            "dm": ((lm[en] - lm[st]) * 1e4).astype(np.float32),
            "flow": (cs_f[en + 1] - cs_f[st + 1]).astype(np.float32),
            "vol": (cs_v[en + 1] - cs_v[st + 1]).astype(np.float32),
            "ofi": (cs_o[en + 1] - cs_o[st + 1]).astype(np.float32),
            "sig": (sig1[st] * math.sqrt(h) * 1e4).astype(np.float32),
            "vhat": (vbar[st] * h).astype(np.float32),
            "dhat": l1bar[st].astype(np.float32),
        }))
    iv = pl.concat(ivs)

    # ---------------------------------------------------------------- 2. taker / pegged simulations, decisions each minute
    dec = np.arange(600, nb - PEG_T - 70, 60)
    f = dec + 1  # taker fill state: 1 s after the decision
    rows = []
    for side in (1, -1):
        hs_f = np.where(side > 0, ask[f] - mid[f], mid[f] - bid[f]) / mid[f] * 1e4
        a1 = l1a[f] if side > 0 else l1b[f]
        d5, d10, d25 = (bk[f"depth_{'ask' if side > 0 else 'bid'}_{k}"][f] for k in (5, 10, 25))
        drift = side * (mid[f] / mid[dec] - 1.0) * 1e4
        rec = {"i": dec.astype(np.int32), "hour": hour[dec], "side": np.full(len(dec), side, np.int8),
               "hs": hs_f.astype(np.float32), "drift": drift.astype(np.float32), "l1": a1.astype(np.float32),
               "sig60": (sig1[dec] * math.sqrt(60) * 1e4).astype(np.float32)}
        for q in QS:
            prem = walk_premium(q, a1, d5, d10, d25, hs_f)
            rec[f"taker_{int(q)}"] = (side * (mid[f] * (1 + side * prem / 1e4) / mid[dec] - 1.0) * 1e4 + TAKER).astype(np.float32)
            rec[f"walk_{int(q)}"] = (prem - hs_f).astype(np.float32)
        # move of the mid over the next 60 s in the trade direction (runaway flag) and 120 s markout of the decision mid
        rec["mv60"] = (side * (mid[dec + 60] / mid[dec] - 1.0) * 1e4).astype(np.float32)
        rows.append(pl.DataFrame(rec))
    tk = pl.concat(rows)

    ss = np.searchsorted(oi, np.arange(nb + 1), "left")  # orders of row i: [ss[i], ss[i+1])
    eps_p = tick * 0.25
    pegs = []
    for prorata in (False, True):
        for side in (1, -1):
            out = np.full((len(dec), 7), np.nan)
            for j, i0 in enumerate(dec):
                out[j] = _peg(i0, side, prorata, bid, ask, mid, l1b, l1a, ts, ss, ms, osd, no, fp, lp, eps_p, nb)
            pegs.append(pl.DataFrame({
                "i": dec.astype(np.int32), "side": np.full(len(dec), side, np.int8), "prorata": np.full(len(dec), prorata),
                "filled": out[:, 0] > 0, "how": out[:, 0].astype(np.int8), "t_fill": out[:, 1].astype(np.float32),
                "price": out[:, 2], "repegs": out[:, 3].astype(np.int16),
            }).with_columns(
                pl.Series("cost", side * (out[:, 2] / mid[dec] - 1.0) * 1e4 + np.where(out[:, 0] > 0, MAKER, TAKER)).cast(pl.Float32),
                pl.Series("mk5", out[:, 4]).cast(pl.Float32), pl.Series("mk60", out[:, 5]).cast(pl.Float32),
                pl.Series("mk300", out[:, 6]).cast(pl.Float32)))
    pg = pl.concat(pegs)

    # ---------------------------------------------------------------- 3. stop-market slippage
    pmin = np.minimum(fp, lp)
    pmax = np.maximum(fp, lp)
    stops = []
    for hh in range(1, 24):
        ih = hh * 3600
        if ih >= nb - 60:
            break
        sg = sig1[ih] * math.sqrt(3600)
        n0 = int(ss[min(ih + 1, nb)])
        if n0 >= len(ms) - 2 or not np.isfinite(sg) or sg <= 0:
            continue
        pm_dn = np.minimum.accumulate(pmin[n0:])
        pm_up = np.maximum.accumulate(pmax[n0:])
        for k in STOP_K:
            for side in (1, -1):  # side = position side; a long (1) has a sell stop below
                x = mid[ih] * math.exp(-side * k * sg)
                if side > 0:
                    jj = int(np.searchsorted(-pm_dn, -x, "left"))
                else:
                    jj = int(np.searchsorted(pm_up, x, "left"))
                if jj >= len(pm_dn):
                    continue
                n = n0 + jj
                rec = {"hour0": hh, "k": k, "side": side, "sig1h": sg * 1e4, "t_trig": (ms[n] - ts[ih]) / 1000.0,
                       "trig_hour": int((ms[n] // 1000) % 86400 // 3600), "gap": -side * ((pmin[n] if side > 0 else pmax[n]) - x) / x * 1e4,
                       "trig_prints": int(o["n_prints"][n])}
                for lat, nm in ((300, "s300"), (1000, "s1000")):
                    jl = int(np.searchsorted(ms, ms[n] + lat, "right")) - 1
                    if side > 0:  # sell at the bid
                        px = lp[jl] - (tick if osd[jl] > 0 else 0.0)
                        rec[nm] = (x - px) / x * 1e4
                    else:
                        px = lp[jl] + (tick if osd[jl] < 0 else 0.0)
                        rec[nm] = (px - x) / x * 1e4
                r = min(int(oi[n]), nb - 1)
                rec["walk100"] = float(walk_premium(100.0, np.array([l1b[r] if side > 0 else l1a[r]]),
                                                    np.array([bk["depth_bid_5" if side > 0 else "depth_ask_5"][r]]),
                                                    np.array([bk["depth_bid_10" if side > 0 else "depth_ask_10"][r]]),
                                                    np.array([bk["depth_bid_25" if side > 0 else "depth_ask_25"][r]]),
                                                    np.array([(ask[r] - bid[r]) / 2 / mid[r] * 1e4]))[0] - (ask[r] - bid[r]) / 2 / mid[r] * 1e4)
                stops.append(rec)
    sp = pl.DataFrame(stops) if stops else pl.DataFrame()

    # ---------------------------------------------------------------- 4. spread decomposition (per order, hour sums)
    m_pre = fp - osd * tick / 2.0
    vw = no / sz
    eff = osd * (vw / m_pre - 1.0) * 1e4
    dec_rows = {}
    for tau in (5, 60):
        r = np.minimum(np.searchsorted(ts, ms + tau * 1000, "left"), nb - 1)
        dec_rows[tau] = r
    rs5 = osd * (vw - mid[dec_rows[5]]) / m_pre * 1e4
    rs60 = osd * (vw - mid[dec_rows[60]]) / m_pre * 1e4
    ohour = ((ms // 1000) % 86400 // 3600).astype(np.int64)
    small = no <= 1000.0
    spr = []
    for hh in range(24):
        mk = ohour == hh
        w = no[mk]
        sw = w.sum()
        ms_ = mk & small
        spr.append({"hour": hh, "n": int(mk.sum()), "notional": float(sw),
                    "eff_w": float((w * eff[mk]).sum() / sw) if sw > 0 else np.nan,
                    "rs5_w": float((w * rs5[mk]).sum() / sw) if sw > 0 else np.nan,
                    "rs60_w": float((w * rs60[mk]).sum() / sw) if sw > 0 else np.nan,
                    "eff_e": float(eff[mk].mean()) if mk.any() else np.nan, "rs5_e": float(rs5[mk].mean()) if mk.any() else np.nan,
                    "rs60_e": float(rs60[mk].mean()) if mk.any() else np.nan,
                    "rs60_small": float(rs60[ms_].mean()) if ms_.any() else np.nan, "n_small": int(ms_.sum())})
    stats["spread_hour"] = spr
    # Roll and Huang-Stoll on event-time first prints (log bp)
    lp_ev = np.log(fp) * 1e4
    dp = np.diff(lp_ev)
    q = osd.astype(float)
    if len(dp) > 10:
        c1 = float(np.mean((dp[1:] - dp[1:].mean()) * (dp[:-1] - dp[:-1].mean())))
        stats["roll"] = {"cov1": c1, "roll_bp": float(2 * math.sqrt(-c1)) if c1 < 0 else None, "eff_full_e": float(2 * np.nanmean(eff)),
                         "quoted_bp": float(np.median((ask - bid) / mid * 1e4))}
        y = dp[1:]  # dp_t for t = 2..N-1 (dp[k] = p_{k+1} - p_k)
        qt, q1, q2 = q[2:], q[1:-1], q[:-2]
        X = np.column_stack([qt, q1, q2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        one_m2pi = float(np.sum(q1 * q2) / np.sum(q2 * q2))
        b0, b1, b2 = (float(c) for c in coef)
        alpha = -b2 / (b0 * one_m2pi) if b0 != 0 and one_m2pi != 0 else np.nan
        beta = b1 / b0 + 1.0 - alpha if b0 != 0 else np.nan
        stats["huang_stoll"] = {"S_bp": 2 * b0, "alpha": alpha, "beta": beta, "pi": (1 - one_m2pi) / 2, "b": [b0, b1, b2], "n": int(len(y))}

    # ---------------------------------------------------------------- 5. propagator statistics (event time)
    mp = np.log(m_pre) * 1e4
    yv = np.diff(mp)
    ev = q[:-1]
    n_ev = len(yv)
    r_k = corr_fft(yv, ev, PROP_L)  # sum_n y_n eps_{n-k}
    c_k = corr_fft(q, q, PROP_C)  # sum_n eps_{n+k} eps_n
    resp = np.array([float(np.mean(q[:-L] * (mp[L:] - mp[:-L]))) if L < len(mp) else np.nan for L in R_GRID])
    np.savez_compressed(_mk(od_dir) / f"{sym}_prop.npz",
                        n=n_ev, yy=float(np.sum(yv ** 2)), r=r_k, c=c_k, c_n=len(q), resp=resp, mean_eps=float(q.mean()))

    # ---------------------------------------------------------------- 6. metaorders (identical-size same-side type 2/3 runs)
    sel = np.where(np.isin(typ, (2, 3)))[0]
    meta = []
    if len(sel) > META_MIN:
        key = pl.DataFrame({"n": sel, "typ": typ[sel], "side": osd[sel], "size": sz[sel], "ms": ms[sel]}).sort(["typ", "side", "size", "ms"])
        kn = key["n"].to_numpy()
        kt, ksd, ksz, kms = (key[c].to_numpy() for c in ("typ", "side", "size", "ms"))
        brk = np.ones(len(kn), bool)
        brk[1:] = (kt[1:] != kt[:-1]) | (ksd[1:] != ksd[:-1]) | (ksz[1:] != ksz[:-1]) | (kms[1:] - kms[:-1] > META_GAP_MS)
        rid = np.cumsum(brk) - 1
        starts = np.where(brk)[0]
        ends = np.r_[starts[1:], len(kn)]
        for a, e in zip(starts, ends, strict=True):
            if e - a < META_MIN:
                continue
            first, last = int(kn[a]), int(kn[e - 1])
            t_s, t_e = ms[first], ms[last]
            i_s = int(oi[first]) - 1
            if i_s < 3600 or last + 1 >= len(ms):
                continue
            nxt = last + 1
            m_end = fp[nxt] - osd[nxt] * tick / 2.0
            side = int(ksd[a])
            rows_t = [min(int(np.searchsorted(ts, t_e + lag * 1000, "left")), nb - 1) for lag in (60, 300)]
            win = slice(max(i_s - 4 * 3600, 0), i_s + 1)
            sg_d = float(np.sqrt(np.mean(r1[win] ** 2)) * math.sqrt(86400) * 1e4)
            v_d = float(vol_row[win].mean() * 86400)
            meta.append({"side": side, "typ": int(kt[a]), "n": int(e - a), "q": float(no[kn[a:e]].sum()), "dur": (t_e - t_s) / 1000.0,
                         "hour": int((t_s // 1000) % 86400 // 3600),
                         "imp_end": side * (math.log(m_end) - math.log(m_pre[first])) * 1e4,
                         "imp60": side * (lm[rows_t[0]] - math.log(m_pre[first])) * 1e4,
                         "imp300": side * (lm[rows_t[1]] - math.log(m_pre[first])) * 1e4,
                         "sig_d": sg_d, "v_d": v_d, "l1": float(0.5 * (l1b[i_s] + l1a[i_s])), "win_h": (win.stop - win.start) / 3600.0})
        _ = rid
    mt = pl.DataFrame(meta) if meta else pl.DataFrame()

    _mk(od_dir)
    iv.write_parquet(od_dir / f"{sym}_iv.parquet", compression="zstd")
    tk.write_parquet(od_dir / f"{sym}_taker.parquet", compression="zstd")
    pg.write_parquet(od_dir / f"{sym}_peg.parquet", compression="zstd")
    if sp.height:
        sp.write_parquet(od_dir / f"{sym}_stop.parquet", compression="zstd")
    if mt.height:
        mt.write_parquet(od_dir / f"{sym}_meta.parquet", compression="zstd")
    stats["secs"] = time.time() - t0
    stats["n_meta"] = int(mt.height)
    done.write_text(json.dumps(stats, default=float))
    return {"d": d, "sym": sym, "secs": stats["secs"]}


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _peg(i0, side, prorata, bid, ask, mid, l1b, l1a, ts, ss, ms, osd, no, fp, lp, eps, nb):
    """One pegged $PEG_Q limit: join the touch at the state of row i0, process the tape second by second, re-peg to the
    end-of-second touch when it moves away; pessimistic queue (only prints at our price advance us; cancels are behind us
    unless the level shrinks below our queue ahead) or pro-rata (unexplained level decreases split pro rata).
    Returns (how, t_fill s, price, repegs, markout 5 / 60 / 300 s bp for us); how 1 through, 2 at level, 3 cross, 4 front,
    0 unfilled (price = touch crossed at the timeout, taker)."""
    p = bid[i0] if side > 0 else ask[i0]
    ahead = l1b[i0] if side > 0 else l1a[i0]
    lprev = ahead
    repegs = 0
    t_start = ts[i0]
    for k in range(1, PEG_T + 1):
        i = i0 + k
        traded = 0.0
        for n in range(ss[i], ss[i + 1]):
            if osd[n] == side:
                continue
            if side > 0:
                if lp[n] < p - eps or fp[n] < p - eps:
                    return _fill(1, ms[n], p, t_start, side, repegs, ts, mid, nb)
                if fp[n] <= p + eps:
                    traded += no[n]
                    ahead -= no[n]
                    if ahead + PEG_Q <= 0:
                        return _fill(2, ms[n], p, t_start, side, repegs, ts, mid, nb)
            else:
                if lp[n] > p + eps or fp[n] > p + eps:
                    return _fill(1, ms[n], p, t_start, side, repegs, ts, mid, nb)
                if fp[n] >= p - eps:
                    traded += no[n]
                    ahead -= no[n]
                    if ahead + PEG_Q <= 0:
                        return _fill(2, ms[n], p, t_start, side, repegs, ts, mid, nb)
        if side > 0:
            if ask[i] <= p + eps:
                return _fill(3, ts[i], p, t_start, side, repegs, ts, mid, nb)
            if bid[i] > p + eps:
                p, ahead, lprev, repegs = bid[i], l1b[i], l1b[i], repegs + 1
                continue
            if bid[i] < p - eps:
                ahead = min(ahead, 0.0)
                continue
            lnow = l1b[i]
        else:
            if bid[i] >= p - eps:
                return _fill(3, ts[i], p, t_start, side, repegs, ts, mid, nb)
            if ask[i] < p - eps:
                p, ahead, lprev, repegs = ask[i], l1a[i], l1a[i], repegs + 1
                continue
            if ask[i] > p + eps:
                ahead = min(ahead, 0.0)
                continue
            lnow = l1a[i]
        if prorata and lprev > 0 and ahead > 0:
            unexpl = max(0.0, lprev - traded - lnow)
            ahead -= unexpl * ahead / lprev
        ahead = min(max(ahead, -PEG_Q), lnow)
        lprev = lnow
    i = i0 + PEG_T + 1
    px = ask[i] if side > 0 else bid[i]
    return (0, float(PEG_T + 1), px, repegs, np.nan, np.nan, np.nan)


def _fill(how, t_ms, p, t_start, side, repegs, ts, mid, nb):
    out = [how, (t_ms - t_start) / 1000.0, p, repegs]
    for lag in (5, 60, 300):
        r = min(int(np.searchsorted(ts, t_ms + lag * 1000, "left")), nb - 1)
        out.append(side * (mid[r] / p - 1.0) * 1e4)
    return tuple(out)


def _day_job(args: tuple[str, str]) -> dict:
    d, sym = args
    try:
        return day_pass(d, sym)
    except Exception as e:  # noqa: BLE001 - logged and reported per symbol-day
        return {"d": d, "sym": sym, "error": repr(e)}


def run_days(workers: int = 2, days: list[str] | None = None, syms: tuple[str, ...] = SYMS) -> None:
    days = days or part_days()
    jobs = [(d, s) for d in days for s in syms]
    log = _mk(OUT) / "day_log.jsonl"
    t = time.time()
    with ProcessPoolExecutor(workers) as ex, log.open("a") as f:
        for r in ex.map(_day_job, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"{len(jobs)} symbol-days in {time.time() - t:.0f} s")


# ====================================================================== flow_3s pass (23 stress perps + spot)
FLOW = ROOT / "data" / "lake" / "flow_3s" / "venue=bybit"
STRESS_SYMS = ("AAVEUSDT", "ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ENAUSDT",
               "ETCUSDT", "ETHUSDT", "FILUSDT", "HYPEUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "SOLUSDT", "SUIUSDT", "TRXUSDT",
               "UNIUSDT", "XLMUSDT", "XRPUSDT")


def flow_days() -> list[str]:
    return sorted(p.name.split("=")[1] for p in (FLOW / "category=linear").iterdir() if p.name.startswith("date="))


def flow_day(d: str) -> pl.DataFrame:
    """Per symbol x hour of one flow_3s day: half-spread quantiles, depth, 3 s vol, a 3 s pegged and stop proxy, and the
    60 s impact intervals (written separately)."""
    cols = ["symbol", "event_ts", "low", "high", "close", "mid", "spread_bps", "buy_notional", "sell_notional", "depth_bid_25",
            "depth_ask_25", "book_cov"]
    parts = []
    for fpath in sorted((FLOW / "category=linear" / f"date={d}").glob("*.parquet")):  # file by file: bounded memory
        x = pl.read_parquet(fpath, columns=cols)
        x = x.filter(pl.col("symbol").is_in(STRESS_SYMS) & (pl.col("book_cov") >= 0.9) & pl.col("mid").is_not_null())
        if x.height:
            parts.append(x.drop("book_cov"))
    if not parts:
        return pl.DataFrame()
    df = pl.concat(parts).unique(["symbol", "event_ts"]).sort(["symbol", "event_ts"])
    del parts
    out, ivs = [], []
    for (sym,), g in df.group_by(["symbol"], maintain_order=True):
        ms_ = g["event_ts"].dt.epoch("ms").to_numpy()
        mid = g["mid"].to_numpy()
        sp = g["spread_bps"].to_numpy().astype(float)
        lo, hi, cl = (g[c].to_numpy() for c in ("low", "high", "close"))
        bn, sn = g["buy_notional"].to_numpy().astype(float), g["sell_notional"].to_numpy().astype(float)
        db, da = g["depth_bid_25"].to_numpy().astype(float), g["depth_ask_25"].to_numpy().astype(float)
        n = len(mid)
        if n < 2000:
            continue
        contiguous = np.r_[False, np.diff(ms_) == 3000]
        r3 = np.where(contiguous, np.r_[0.0, np.diff(np.log(mid))], np.nan) * 1e4
        hour = (ms_ // 1000 % 86400 // 3600).astype(np.int64)
        # 3 s pegged proxy (decision each minute): buy at mid*(1 - s/2), fill only on a print strictly through the price
        # (pessimistic: no queue fills), re-peg when the touch moves away, 60 s then taker; sell symmetric.
        dec = np.arange(0, n - 25, 20)
        peg_cost, peg_fill = [], []
        for side in (1, -1):
            for i0 in dec:
                if not contiguous[i0 + 1: i0 + 22].all():
                    continue
                m0 = mid[i0]
                p = m0 * (1 - side * sp[i0] / 2e4)
                filled = False
                for k in range(1, 21):
                    i = i0 + k
                    if (side > 0 and np.isfinite(lo[i]) and lo[i] < p * (1 - 1e-9)) or (side < 0 and np.isfinite(hi[i]) and hi[i] > p * (1 + 1e-9)):
                        filled = True
                        break
                    touch = mid[i] * (1 - side * sp[i] / 2e4)
                    if side * (touch - p) > 0:
                        p = touch
                if filled:
                    peg_cost.append((i0, side, side * (p / m0 - 1) * 1e4 + MAKER, 1))
                else:
                    i = i0 + 21
                    px = mid[i] * (1 + side * sp[i] / 2e4)
                    peg_cost.append((i0, side, side * (px / m0 - 1) * 1e4 + TAKER, 0))
        pc = np.array(peg_cost) if peg_cost else np.zeros((0, 4))
        # 3 s stop proxy: each hour's open, levels at k * trailing-1h sigma; trigger on the candle low / high; fill at the
        # trigger candle close minus / plus half the spread (latency up to 3 s: conservative vs 300 ms)
        stop_rows = []
        for i0 in np.where(np.r_[True, hour[1:] != hour[:-1]])[0]:
            w = r3[max(0, i0 - 1200): i0]
            w = w[np.isfinite(w)]
            if len(w) < 600:
                continue
            sg = float(np.sqrt(np.mean(w ** 2)) * math.sqrt(1200))
            for k in STOP_K:
                for side in (1, -1):
                    x = mid[i0] * math.exp(-side * k * sg / 1e4)
                    seg_lo, seg_hi = lo[i0 + 1:], hi[i0 + 1:]
                    hit = np.where(np.nan_to_num(seg_lo, nan=np.inf) <= x)[0] if side > 0 else np.where(np.nan_to_num(seg_hi, nan=-np.inf) >= x)[0]
                    if len(hit) == 0:
                        continue
                    j = i0 + 1 + hit[0]
                    px = cl[j] * (1 - side * sp[j] / 2e4)
                    stop_rows.append(side * (x - px) / x * 1e4 if side > 0 else (px - x) / x * 1e4)
        for hh in range(24):
            mk = hour == hh
            if mk.sum() < 100:
                continue
            pk = pc[np.isin(pc[:, 0].astype(int), np.where(mk)[0])] if len(pc) else pc
            rr = r3[mk]
            rr = rr[np.isfinite(rr)]
            out.append({"date": d, "symbol": sym, "hour": hh, "n3": int(mk.sum()),
                        "hs_mean": float(np.nanmean(sp[mk]) / 2), "hs_p50": float(np.nanmedian(sp[mk]) / 2),
                        "hs_p90": float(np.nanquantile(sp[mk], 0.9) / 2), "hs_p99": float(np.nanquantile(sp[mk], 0.99) / 2),
                        "depth25_med": float(np.nanmedian(np.minimum(db[mk], da[mk]))), "sig3": float(np.sqrt(np.mean(rr ** 2))) if len(rr) else np.nan,
                        "r3_p99": float(np.quantile(np.abs(rr), 0.99)) if len(rr) else np.nan,
                        "turnover": float((bn[mk] + sn[mk]).sum()),
                        "peg3_cost": float(pk[:, 2].mean()) if len(pk) else np.nan, "peg3_fill": float(pk[:, 3].mean()) if len(pk) else np.nan,
                        "peg3_n": int(len(pk))})
        sr = np.array(stop_rows)
        if len(sr):
            out.append({"date": d, "symbol": sym, "hour": -1, "stop3_mean": float(sr.mean()), "stop3_p90": float(np.quantile(sr, 0.9)),
                        "stop3_p99": float(np.quantile(sr, 0.99)), "stop3_n": int(len(sr))})
        # 60 s impact intervals (PIT trailing 60-min sigma / volume)
        st = np.arange(1200, n - 20, 20)
        ok = np.array([contiguous[a + 1: a + 21].all() for a in st])
        st = st[ok]
        if len(st):
            cs_f = np.concatenate([[0.0], np.cumsum(bn - sn)])
            cs_v = np.concatenate([[0.0], np.cumsum(bn + sn)])
            r3z = np.nan_to_num(r3)
            cs_r = np.concatenate([[0.0], np.cumsum(r3z ** 2)])
            ivs.append(pl.DataFrame({"date": d, "symbol": sym, "hour": hour[st].astype(np.int8),
                                     "dm": ((np.log(mid[st + 20]) - np.log(mid[st])) * 1e4).astype(np.float32),
                                     "flow": (cs_f[st + 21] - cs_f[st + 1]).astype(np.float32),
                                     "sig": (np.sqrt((cs_r[st + 1] - cs_r[st - 1199]) / 1200) * math.sqrt(20)).astype(np.float32),
                                     "vhat": ((cs_v[st + 1] - cs_v[st - 1199]) / 1200 * 20).astype(np.float32)}))
    res = pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()
    if ivs:
        pl.concat(ivs).write_parquet(_mk(OUT / "flow") / f"iv_{d}.parquet", compression="zstd")
    return res


def _flow_job(d: str) -> str:
    p = OUT / "flow" / f"{d}.parquet"
    if p.exists():
        return d + " skipped"
    try:
        r = flow_day(d)
        if r.height:
            r.write_parquet(_mk(OUT / "flow") / f"{d}.parquet", compression="zstd")
        return d
    except Exception as e:  # noqa: BLE001
        return f"{d} error {e!r}"


def run_flow(workers: int = 2) -> None:
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_flow_job, flow_days()):
            print(r, flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["day", "flow", "one"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--syms", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "day":
        run_days(a.workers, a.days, tuple(a.syms) if a.syms else SYMS)
    elif a.cmd == "one":
        for d in a.days:
            for s in a.syms or SYMS:
                print(day_pass(d, s, overwrite=True))
    elif a.cmd == "flow":
        run_flow(a.workers)


if __name__ == "__main__":
    main()
