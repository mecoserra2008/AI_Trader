"""Break-of-structure breakout / momentum / trend family by regime and execution arm (block BOS, H-BOS1..5).

Design frozen in the diary 2026-11-15 (kind decision, "BOS family: frozen grid ...") and data/research/bos/design.json
before any outcome. Stage 1 (this module): per (symbol, timeframe) trade tables, checkpointed in
data/research/bos/cache/<tf>/<SYM>.parquet. Stage 2 (bos_regimes_analysis.py): cells, BH, tails, stability,
allocator, execution arms, report.

One row per BoS signal of a structure (signal types CHoCH / BoS+mom are flags on the row). For every exit
(opp, trail2.5, trail4.0, time) and execution arm the row carries the net R, the exit time and (retest) the fill flag:
  m   market at the next open (+2.0 bp, taker 5.5 bp)
  p   pegged limit at the touch (next open +1.3 bp, maker 2.0 bp)
  r   retest limit at the broken level, N = 6 bars, trade-through 1 bp, stop-first on the fill bar (worst case)
  rb  the same, fill-only on the fill bar (best-case bound of the intrabar ambiguity)
  r24 retest with N = 24 (sensitivity)
Exits are taker 5.5 bp + 2.0 bp slippage. 1 R = |close_t - stop| at the signal for every arm. Funding is charged.

Shorts are simulated as longs on the mirrored price path (-price); costs use |price|.

    uv run python -m ai_trader.research.bos_regimes_study cache --tf 4h,1h
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.core.config import load
from ai_trader.features import vol_bos as vb

UNIV_L = (
    "BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT ADAUSDT LINKUSDT LTCUSDT BCHUSDT AVAXUSDT DOTUSDT BNBUSDT "
    "TRXUSDT NEARUSDT ATOMUSDT ETCUSDT FILUSDT UNIUSDT AAVEUSDT XLMUSDT SUIUSDT ENAUSDT HYPEUSDT"
).split()
UNIV_T = ("XAUUSDT XAGUSDT CLUSDT BZUSDT TSLAUSDT INTCUSDT GOOGLUSDT AAPLUSDT MSTRUSDT NVDAUSDT MUUSDT QQQUSDT "
          "SOXLUSDT SPCXUSDT PAXGUSDT XAUTUSDT").split()
TF = {"5m": dict(cls="intraday", bar_min=5, H=288), "15m": dict(cls="intraday", bar_min=15, H=96),
      "1h": dict(cls="swing", bar_min=60, H=240), "4h": dict(cls="swing", bar_min=240, H=60)}
REF_MIN = {"intraday": 60, "swing": 240}
STRUCTS = (("frac", 2), ("frac", 3), ("frac", 5), ("don", 20), ("don", 55))
SIGNALS = ("bos", "choch", "bosmom")
EXITS = ("opp", "tr25", "tr40", "time")
TRAIL_K = {"tr25": 2.5, "tr40": 4.0}
ARMS = ("m", "p", "r", "rb", "r24")
MAIN_ARMS = ("m", "p", "r")
ARM_NAME = {"m": "market", "p": "pegged", "r": "retest", "rb": "retest_bestcase", "r24": "retest_N24"}
MOM_M, MOM_THR = 20, 1.0
STOP_CLIP = (0.5, 4.0)
N_RETEST, N_RETEST2 = 6, 24
THROUGH_BP = 1.0
COST = dict(m=(2.0, 5.5), p=(1.3, 2.0), r=(0.0, 2.0))   # (entry slip bp, entry fee bp)
EXIT_SLIP, EXIT_FEE = 2.0, 5.5
CHUNK = 3000
US_H = 3_600_000_000


def root() -> Path:
    return load().root / "data" / "research" / "bos"


def cache_file(tf: str, sym: str) -> Path:
    return root() / "cache" / tf / f"{sym}.parquet"


def as_of() -> datetime:
    return datetime(2026, 11, 15, 18, 0, tzinfo=UTC)


# ====================================================================== frames
def _sanity(b: pl.DataFrame) -> pl.DataFrame:
    return b.filter((pl.col("open") > 0) & (pl.col("low") > 0) & (pl.col("high") >= pl.max_horizontal("open", "close"))
                    & (pl.col("low") <= pl.min_horizontal("open", "close")))


def _burn_start(sym: str) -> datetime | None:
    b = load().root / "data/lake/bars/venue=bybit/price_type=last/freq=1h"
    t = (pl.scan_parquet(str(b / "**" / "*.parquet"), hive_partitioning=True).filter(pl.col("symbol") == sym)
         .select(pl.col("event_ts").min()).collect().item())
    if t is None:
        return None
    rs = load().settings["research"]
    days = rs["burn_in_days_new"] if t.year >= rs["burn_in_new_listing_year"] else rs["burn_in_days"]
    return t + timedelta(days=days)


def agg(b: pl.DataFrame, minutes: int) -> pl.DataFrame:
    g = (b.sort("event_ts").group_by(pl.col("event_ts").dt.truncate(f"{minutes}m").alias("bt"), maintain_order=True)
         .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
              pl.col("volume").sum(), pl.col("turnover").sum()))
    return (g.rename({"bt": "event_ts"}).with_columns((pl.col("event_ts") + pl.duration(minutes=minutes)).alias("close_ts"))
            .with_columns(pl.col("close_ts").alias("known_ts")).sort("event_ts"))


def intraday_bars(sym: str, tf: str) -> pl.DataFrame:
    fs = sorted((root() / "bars" / "5m").glob(f"{sym}_*.parquet"))
    if not fs:
        return pl.DataFrame()
    b = pl.concat([pl.read_parquet(f) for f in fs]).unique("event_ts").sort("event_ts")
    b = b.with_columns(pl.col("event_ts").cast(pl.Datetime("us", "UTC")))
    b = _sanity(b)
    st = _burn_start(sym)
    if st is not None:
        b = b.filter(pl.col("event_ts") >= st)
    b = b.filter(pl.col("event_ts") < as_of())
    if tf == "5m":
        return b.with_columns((pl.col("event_ts") + pl.duration(minutes=5)).alias("close_ts")).with_columns(pl.col("close_ts").alias("known_ts"))
    return agg(b, 15)


def frame(sym: str, tf: str, tradfi: bool = False) -> pl.DataFrame:
    from ai_trader.research.frame import enrich_bars, market_frame

    if tf in ("5m", "15m"):
        b = intraday_bars(sym, tf)
        return enrich_bars(b) if b.height else b
    if tradfi:
        h1 = market_frame(sym, "1h", as_of(), funding=False, oi=False)
        if tf == "1h" or h1.height == 0:
            return h1
        return enrich_bars(agg(h1.select("event_ts", "open", "high", "low", "close", "volume", "turnover"), 240))
    return market_frame(sym, tf, as_of(), funding=False, oi=False)


def regime_table(sym: str, tradfi: bool = False) -> pl.DataFrame:
    """4h v0 cell, dir_score, sigma_next and trailing-365d terciles, keyed on the 4h close_ts."""
    from ai_trader.research import regimes

    f = frame(sym, "4h", tradfi=tradfi)
    if f.height < 50:
        return pl.DataFrame()
    lab = regimes.label(f, bar_hours=4.0)
    w, mp = 2190, 1095

    def pct(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        out = np.full(len(x), np.nan)
        for t in range(mp, len(x)):
            h = x[max(0, t - w):t]
            h = h[np.isfinite(h)]
            if len(h) >= mp and np.isfinite(x[t]):
                out[t] = (h < x[t]).mean()
        return out

    vp = pct(lab["sigma_next"].to_numpy())
    sp = pct(np.abs(lab["dir_score"].to_numpy()))
    def terc(p: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(p), np.minimum(np.floor(np.nan_to_num(p) * 3), 2), -1).astype(np.int8)

    cell = lab["cell"].to_list()
    ci = np.array([int(c[1:]) - 1 if c else -1 for c in cell], dtype=np.int8)
    return pl.DataFrame({"close_ts": lab["close_ts"].cast(pl.Datetime("us", "UTC")), "cell": ci, "vt": terc(vp), "st": terc(sp)})


def funding_cum(sym: str) -> tuple[np.ndarray, np.ndarray]:
    from ai_trader.storage.lake import Lake

    f = Lake().read_asof("funding", as_of(), symbols=[sym], filters={"venue": "bybit"})
    if f.height == 0:
        return np.array([0], dtype=np.int64), np.array([0.0])
    f = f.sort("event_ts").unique("event_ts", keep="last").sort("event_ts")
    ts = f["event_ts"].dt.epoch("us").to_numpy()
    return ts, np.cumsum(np.nan_to_num(f["rate"].to_numpy().astype(float)))


# ====================================================================== structure
def fractal_structure(c, h, lo, k):
    """bos (+1/-1/0), choch, broken level, trailing latest swing high / low (known at t). Same break logic as
    vol_bos.structure (each confirmed level consumed once); additionally returns the broken level."""
    hi_conf, lo_conf = vb.fractal_pivots(h, lo, k)
    n = len(c)
    bos = np.zeros(n, np.int8)
    choch = np.zeros(n, bool)
    lvl = np.full(n, np.nan)
    cur_hi = cur_lo = np.nan
    prev = 0
    for t in range(n):
        if hi_conf[t] == hi_conf[t]:
            cur_hi = hi_conf[t]
        if lo_conf[t] == lo_conf[t]:
            cur_lo = lo_conf[t]
        d = 0
        if cur_hi == cur_hi and c[t] > cur_hi:
            d, lvl[t], cur_hi = 1, cur_hi, np.nan
        elif cur_lo == cur_lo and c[t] < cur_lo:
            d, lvl[t], cur_lo = -1, cur_lo, np.nan
        if d:
            bos[t] = d
            choch[t] = prev != 0 and d != prev
            prev = d
    last_hi = pl.Series(hi_conf).fill_nan(None).forward_fill().fill_null(np.nan).to_numpy()
    last_lo = pl.Series(lo_conf).fill_nan(None).forward_fill().fill_null(np.nan).to_numpy()
    # stops at the signal: last confirmed opposite swing; trailing opposite level for the 'opp' exit: the same series
    return bos, choch, lvl, last_lo, last_hi, last_lo, last_hi


def donchian_structure(c, h, lo, n):
    hs, ls = pl.Series(h), pl.Series(lo)
    up = hs.rolling_max(n).shift(1).to_numpy()
    dn = ls.rolling_min(n).shift(1).to_numpy()
    above = np.nan_to_num(c > up, nan=False) & np.isfinite(up)
    below = np.nan_to_num(c < dn, nan=False) & np.isfinite(dn)
    on_up = above & ~np.r_[False, above[:-1]]
    on_dn = below & ~np.r_[False, below[:-1]]
    bos = on_up.astype(np.int8) - on_dn.astype(np.int8)
    bos[on_up & on_dn] = 0
    lvl = np.where(bos > 0, up, np.where(bos < 0, dn, np.nan))
    choch = np.zeros(len(c), bool)
    prev = 0
    for t in np.flatnonzero(bos):
        choch[t] = prev != 0 and bos[t] != prev
        prev = bos[t]
    stop_lo = ls.rolling_min(n).to_numpy()     # n-bar low incl. t (long stop)
    stop_hi = hs.rolling_max(n).to_numpy()
    return bos, choch, lvl, stop_lo, stop_hi, dn, up   # opp exit: close beyond the previous-n-bar opposite channel


# ====================================================================== simulation (long-mirrored)
def _gather(x: np.ndarray, start: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    idx = start[:, None] + np.arange(width)[None, :]
    ok = idx < len(x)
    return x[np.minimum(idx, len(x) - 1)], ok


def _first(mask: np.ndarray, default: int) -> np.ndarray:
    a = mask.argmax(axis=1)
    return np.where(mask.any(axis=1), a, default)


def simulate(P: dict, sig_i: np.ndarray, side: np.ndarray, stop: np.ndarray, su: np.ndarray, opp_lo: np.ndarray, opp_hi: np.ndarray,
             entry_bar: np.ndarray, entry_px_raw: np.ndarray, fill_intrabar: np.ndarray, H: int, ts: np.ndarray, fund: tuple,
             R: np.ndarray, slip_bp: float, fee_bp: float, stop_on_fill_bar: bool) -> dict[str, np.ndarray]:
    """Exits for trades entered at bar entry_bar (absolute) at entry_px_raw (mirrored, before slippage).
    Returns per exit: net R (nan = incomplete / not entered) and exit timestamp."""
    n = len(P["o"])
    s = side.astype(float)[:, None]
    O, _ = _gather(P["o"], entry_bar, H + 1)
    Hh, _ = _gather(P["h"], entry_bar, H)
    Lw, _ = _gather(P["l"], entry_bar, H)
    C, _ = _gather(P["c"], entry_bar, H)
    O, Hh, Lw, C = O * s, Hh * s, Lw * s, C * s
    lo_m = np.where(s > 0, Lw, Hh)  # mirrored low: a short's Hh * s = -high
    OL, _ = _gather(opp_lo, entry_bar, H)
    OH, _ = _gather(opp_hi, entry_bar, H)
    opp_m = np.where(s > 0, OL, -OH)
    stop_m = stop[:, None]
    E = entry_px_raw[:, None] + slip_bp * 1e-4 * np.abs(entry_px_raw[:, None])
    hit = lo_m <= stop_m
    if not stop_on_fill_bar:
        hit[:, 0] &= ~fill_intrabar
    j_stop = _first(hit, H + 10)
    st_open = O[:, :H]
    j_idx = np.minimum(j_stop, H - 1)
    o_at = np.take_along_axis(st_open, j_idx[:, None], axis=1)[:, 0]
    # fill of a stop: worse of stop and the bar open, except on an intrabar fill bar (open came before the fill)
    stop_fill = np.where((j_stop == 0) & fill_intrabar, stop, np.minimum(stop, o_at))
    out = {}
    with np.errstate(invalid="ignore"):
        best = np.maximum.accumulate(np.where(np.isfinite(C), C, -np.inf), axis=1)
        trig = {"opp": np.nan_to_num(C < opp_m, nan=False), "time": np.zeros_like(C, bool)}
        for e, k in TRAIL_K.items():
            trig[e] = C < best - k * su[:, None]
    for e in EXITS:
        j_c = np.minimum(_first(trig[e], H - 1), H - 1)
        use_stop = j_stop <= j_c
        x_open = np.take_along_axis(O, (j_c + 1)[:, None], axis=1)[:, 0]
        X = np.where(use_stop, stop_fill, x_open)
        X = X - EXIT_SLIP * 1e-4 * np.abs(X)
        jx = np.where(use_stop, j_stop, j_c + 1)
        xbar = np.minimum(entry_bar + jx, n - 1)
        x_ts = ts[xbar]
        e_ts = ts[np.minimum(entry_bar, n - 1)]
        fts, fcum = fund
        fi_e = np.searchsorted(fts, e_ts, side="right")
        fi_x = np.searchsorted(fts, x_ts, side="right")
        fc = np.r_[0.0, fcum]
        funding = side * (fc[fi_x] - fc[fi_e]) * np.abs(entry_px_raw)
        gross = X - E[:, 0]
        fees = (fee_bp * np.abs(E[:, 0]) + EXIT_FEE * np.abs(X)) * 1e-4
        net = (gross - fees - funding) / R
        net = np.where(((entry_bar + jx) < n) & np.isfinite(net), net, np.nan)
        out[e] = (net.astype(np.float32), x_ts)
    return out


def symbol_trades(sym: str, tf: str, tradfi: bool = False) -> pl.DataFrame | None:
    cfg = TF[tf]
    f = frame(sym, tf, tradfi=tradfi)
    if f.height < 2000:
        return None
    reg = regime_table(sym, tradfi=tradfi)
    if reg.height == 0:
        return None
    f = f.with_columns(pl.col("close_ts").cast(pl.Datetime("us", "UTC")))
    f = f.sort("close_ts").join_asof(reg.sort("close_ts"), on="close_ts", strategy="backward")
    ts = f["event_ts"].dt.epoch("us").to_numpy()
    cts = f["close_ts"].dt.epoch("us").to_numpy()
    o, h, lo, c = (f[k].to_numpy().astype(float) for k in ("open", "high", "low", "close"))
    sig_bar = f["sigma_next"].to_numpy().astype(float)
    su_all = sig_bar * np.sqrt(REF_MIN[cfg["cls"]] / cfg["bar_min"]) * c
    lc = np.log(c)
    mom = np.full(len(c), np.nan)
    mom[MOM_M:] = (lc[MOM_M:] - lc[:-MOM_M]) / (sig_bar[MOM_M:] * np.sqrt(MOM_M))
    cell = f["cell"].fill_null(-1).to_numpy().astype(np.int8)
    vt = f["vt"].fill_null(-1).to_numpy().astype(np.int8)
    st = f["st"].fill_null(-1).to_numpy().astype(np.int8)
    fund = funding_cum(sym)
    P = {"o": o, "h": h, "l": lo, "c": c}
    H = cfg["H"]
    warm = 200
    tabs = []
    for si, (kind, prm) in enumerate(STRUCTS):
        bos, choch, lvl, stop_lo, stop_hi, opp_lo, opp_hi = (fractal_structure if kind == "frac" else donchian_structure)(c, h, lo, prm)
        idx = np.flatnonzero(bos != 0)
        idx = idx[(idx >= warm) & (idx < len(c) - 2)]
        side = bos[idx].astype(np.int8)
        lvl_s = np.where(side > 0, stop_lo[idx], stop_hi[idx])
        su = su_all[idx]
        ok = np.isfinite(lvl_s) & np.isfinite(su) & (su > 0) & np.isfinite(lvl[idx])
        idx, side, lvl_s, su = idx[ok], side[ok], lvl_s[ok], su[ok]
        dist = np.clip((c[idx] - lvl_s) * side, STOP_CLIP[0] * su, STOP_CLIP[1] * su)
        R = dist
        stop_m = c[idx] * side - dist                         # mirrored stop
        L_m = lvl[idx] * side                                 # mirrored broken level
        cols = {"struct": np.full(len(idx), si, np.int8), "sig_ts": cts[idx], "side": side, "R_su": (dist / su).astype(np.float32),
                "choch": choch[idx], "mom": np.nan_to_num(mom[idx] * side) > MOM_THR,
                "cell": cell[idx], "vt": vt[idx], "st": st[idx]}
        res: dict[str, list] = {}
        for a in range(0, len(idx), CHUNK):
            sl = slice(a, a + CHUNK)
            ii, sd, sm, sus, RR, LL = idx[sl], side[sl], stop_m[sl], su[sl], R[sl], L_m[sl]
            eb = ii + 1
            o_next = o[np.minimum(eb, len(o) - 1)] * sd
            fz = np.zeros(len(ii), bool)
            outs = {}
            r = simulate(P, ii, sd, sm, sus, opp_lo, opp_hi, eb, o_next, fz, H, ts, fund, RR, *COST["m"], True)
            outs["m"] = r
            outs["p"] = simulate(P, ii, sd, sm, sus, opp_lo, opp_hi, eb, o_next, fz, H, ts, fund, RR, *COST["p"], True)
            for arm, N in (("r", N_RETEST), ("r24", N_RETEST2)):
                Ow, okw = _gather(o, eb, N)
                Lw, _ = _gather(lo, eb, N)
                Hw, _ = _gather(h, eb, N)
                Om = Ow * sd[:, None]
                lowm = np.where(sd[:, None] > 0, Lw, -Hw)
                thr = LL - THROUGH_BP * 1e-4 * np.abs(LL)
                gap = Om <= thr[:, None]
                thru = lowm <= thr[:, None]
                fillm = (gap | thru) & okw
                valid = sm < LL
                fj = _first(fillm, -1)
                filled = (fj >= 0) & valid
                fjj = np.maximum(fj, 0)
                gap_at = np.take_along_axis(gap, fjj[:, None], axis=1)[:, 0]
                o_at = np.take_along_axis(Om, fjj[:, None], axis=1)[:, 0]
                px = np.where(gap_at, o_at, LL)
                intrabar = ~gap_at
                ebr = eb + fjj
                for a2, sof in ((arm, True),) + ((("rb", False),) if arm == "r" else ()):
                    rr = simulate(P, ii, sd, sm, sus, opp_lo, opp_hi, ebr, px, intrabar, H, ts, fund, RR, *COST["r"], sof)
                    outs[a2] = {e: (np.where(filled, v[0], np.nan).astype(np.float32), v[1]) for e, v in rr.items()}
                    res.setdefault(f"{a2}_fill", []).append(filled)
            for arm, r in outs.items():
                for e, (net, xts) in r.items():
                    res.setdefault(f"{arm}_{e}", []).append(net)
                    if arm in ("m", "r", "r24"):
                        res.setdefault(f"{arm}_{e}_x", []).append(xts)
        for k, v in res.items():
            cols[k] = np.concatenate(v) if v else np.array([])
        tabs.append(pl.DataFrame(cols))
        del res
        gc.collect()
    df = pl.concat(tabs)
    return df.with_columns(pl.lit(sym).alias("sym"))


def build(tfs: list[str], symbols: list[str], tradfi: bool = False) -> None:
    for tf in tfs:
        for sym in symbols:
            p = cache_file(tf + ("_T" if tradfi else ""), sym)
            if p.exists():
                continue
            t0 = time.time()
            try:
                d = symbol_trades(sym, tf, tradfi=tradfi)
            except Exception as ex:  # noqa: BLE001 (record and continue; reported in the manifest)
                print(f"FAIL {tf} {sym}: {ex!r}", flush=True)
                continue
            if d is None or d.height == 0:
                print(f"skip {tf} {sym}", flush=True)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.parquet")
            d.write_parquet(tmp, compression="zstd")
            os.replace(tmp, p)
            print(f"cached {tf} {sym} signals={d.height} {time.time() - t0:.0f}s", flush=True)
            del d
            gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache"])
    ap.add_argument("--tf", default="4h,1h")
    ap.add_argument("--symbols", default=",".join(UNIV_L))
    ap.add_argument("--tradfi", action="store_true")
    a = ap.parse_args()
    syms = UNIV_T if a.tradfi and a.symbols == ",".join(UNIV_L) else a.symbols.split(",")
    build(a.tf.split(","), syms, tradfi=a.tradfi)


if __name__ == "__main__":
    main()
