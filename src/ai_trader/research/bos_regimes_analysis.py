"""Stage 2 of the BOS study (H-BOS1..5): cells, BH, tails, stability, allocator, execution arms.

Reads the trade tables of bos_regimes_study (data/research/bos/cache/<tf>/<SYM>.parquet) and writes
data/research/bos/agg/<tf>.npz + <tf>_cells.parquet (stage `agg`, one timeframe at a time, low memory), then
data/research/bos/results.json (stage `infer`). Design: diary 2026-11-15 18:05 UTC (decision) / design.json.

Cell = (variant, arm, regime). variant = (tf, structure, signal, exit); arm in market / pegged / retest (BH family),
retest_bestcase / retest_N24 (sensitivity); regime = ALL, v0 cell x side (18), vol tercile (3), trend tercile (3).
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import polars as pl

from ai_trader.research import bos_regimes_study as S

ND = NormalDist()
T0_US = int(datetime(2020, 1, 6, tzinfo=UTC).timestamp() * 1e6)      # Monday
D0_US = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1e6)
WEEK_US, DAY_US = 7 * 86_400_000_000, 86_400_000_000
CUT_US = int(datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1e6)
W = int((int(datetime(2026, 11, 16, tzinfo=UTC).timestamp() * 1e6) - T0_US) // WEEK_US) + 1
ND_DAYS = int((int(datetime(2026, 11, 16, tzinfo=UTC).timestamp() * 1e6) - D0_US) // DAY_US) + 1
W_CUT = (CUT_US - T0_US) // WEEK_US                                  # first holdout week (2026-06-29 week straddles; signals split by ts)
B_BOOT = 400
MIN_N, MIN_WEEKS = 30, 10
TFS = ("5m", "15m", "1h", "4h")
CLASS = {"5m": "intraday", "15m": "intraday", "1h": "swing", "4h": "swing"}
SEGS = ("llm_in_sample", "clean_holdout")
REG_NAMES = (["ALL"] + [f"R{c + 1}{'L' if s == 0 else 'S'}" for c in range(9) for s in range(2)]
             + ["vol_T1", "vol_T2", "vol_T3", "trend_S1", "trend_S2", "trend_S3"])
NREG = len(REG_NAMES)
X_COL = {"m": "m", "p": "m", "r": "r", "rb": "r", "r24": "r24"}
FILL = {"r": "r_fill", "rb": "rb_fill", "r24": "r24_fill"}
HBINS = np.r_[-np.inf, np.arange(-5.0, 60.0001, 0.02), np.inf]
RNG = np.random.default_rng(20261115)
BOOTW = RNG.poisson(1.0, size=(B_BOOT, W)).astype(np.float32)


def agg_dir() -> Path:
    return S.root() / "agg"


def variant_id(tf: str, s: int, g: int, e: int) -> int:
    return TFS.index(tf) * 60 + s * 12 + g * 4 + e


def variant_name(v: int) -> str:
    tf = TFS[v // 60]
    r = v % 60
    s, g, e = r // 12, (r % 12) // 4, r % 4
    kind, prm = S.STRUCTS[s]
    return f"{tf}|{kind}{prm}|{S.SIGNALS[g]}|{S.EXITS[e]}"


def files(tf: str) -> list[Path]:
    return sorted((S.root() / "cache" / tf).glob("*.parquet"))


def read_cols(tf: str, cols: list[str]) -> pl.DataFrame:
    fs = files(tf)
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs]) if fs else pl.DataFrame()


def _keys(df: pl.DataFrame, xcol: str) -> pl.DataFrame:
    """week, segment (purged), regime ids."""
    return df.with_columns(
        ((pl.col("sig_ts") - T0_US) // WEEK_US).cast(pl.Int32).alias("wk"),
        ((pl.col(xcol) - D0_US) // DAY_US).cast(pl.Int32).alias("xday"),
        ((pl.col(xcol) - T0_US) // WEEK_US).cast(pl.Int32).alias("xwk"),
        pl.when(pl.col("sig_ts") >= CUT_US).then(1).when(pl.col(xcol) < CUT_US).then(0).otherwise(-1).cast(pl.Int8).alias("seg"),
        pl.when(pl.col("cell") >= 0).then(1 + pl.col("cell").cast(pl.Int16) * 2 + (pl.col("side") < 0).cast(pl.Int16)).otherwise(-1).alias("rcs"),
        pl.when(pl.col("vt") >= 0).then(19 + pl.col("vt").cast(pl.Int16)).otherwise(-1).alias("rvt"),
        pl.when(pl.col("st") >= 0).then(22 + pl.col("st").cast(pl.Int16)).otherwise(-1).alias("rst"),
    )


# ====================================================================== stage agg
def agg_tf(tf: str) -> None:
    """Per timeframe: per-cell sufficient statistics for every (signal, struct, exit, arm, regime, segment)."""
    out = agg_dir()
    out.mkdir(parents=True, exist_ok=True)
    cls = CLASS[tf]
    base = ["struct", "sig_ts", "side", "choch", "mom", "cell", "vt", "st"]
    ncell = 60 * len(S.ARMS) * NREG * 2
    WS = np.zeros((ncell, W), np.float32)          # weekly sums (by signal week)
    WN = np.zeros((ncell, W), np.float32)          # weekly counts
    stats = np.full((ncell, 8), np.nan)             # n, sum, n_win, sum_win, sum_loss, n_loss, top5_sum, n_signals
    hist_c = np.zeros((len(S.ARMS), NREG, 2, len(HBINS) - 1), np.float64)   # pooled family histograms (H-BOS2)
    hist_s = np.zeros_like(hist_c)
    fam_ws = np.zeros((len(S.ARMS), NREG, 2, W), np.float64)
    fam_wn = np.zeros_like(fam_ws)
    alloc = []
    for ai, a in enumerate(S.ARMS):
        for ei, e in enumerate(S.EXITS):
            xc = f"{X_COL[a]}_{e}_x"
            cols = base + [f"{a}_{e}", xc] + ([FILL[a]] if a in FILL else [])
            df = _keys(read_cols(tf, cols), xc).rename({f"{a}_{e}": "v"})
            if a in FILL:
                # per-signal count: filled-or-missed signals that are complete (missed = 0 R); filled-but-incomplete dropped
                df = df.with_columns(pl.when(pl.col(FILL[a])).then(pl.col("v")).otherwise(0.0).alias("v_sig"),
                                     (pl.col(FILL[a]) & pl.col("v").is_not_nan()).alias("tr"),
                                     (~pl.col(FILL[a]) | pl.col("v").is_not_nan()).alias("sigok"))
            else:
                df = df.with_columns(pl.col("v").alias("v_sig"), pl.col("v").is_not_nan().alias("tr"), pl.col("v").is_not_nan().alias("sigok"))
            # missed retests: their 'seg' uses the market exit-ts column; recompute seg on sig_ts for missed rows
            df = df.with_columns(pl.when(~pl.col("tr") & pl.col("sigok")).then(
                pl.when(pl.col("sig_ts") >= CUT_US).then(1).otherwise(0)).otherwise(pl.col("seg")).cast(pl.Int8).alias("seg"))
            df = df.filter((pl.col("seg") >= 0) & pl.col("sigok"))
            for s in range(len(S.STRUCTS)):
                d0 = df.filter(pl.col("struct") == s)
                for g, flt in enumerate((None, "choch", "mom")):
                    d = d0 if flt is None else d0.filter(pl.col(flt))
                    if d.height == 0:
                        continue
                    vid = s * 12 + g * 4 + ei
                    v = d["v"].to_numpy().astype(np.float64)
                    tr = d["tr"].to_numpy()
                    wk = d["wk"].to_numpy()
                    seg = d["seg"].to_numpy()
                    regs = [np.zeros(d.height, np.int16), d["rcs"].to_numpy(), d["rvt"].to_numpy(), d["rst"].to_numpy()]
                    for rr in regs:
                        for sg in (0, 1):
                            for rid in np.unique(rr[rr >= 0]):
                                m = (rr == rid) & (seg == sg)
                                if not m.any():
                                    continue
                                ci = ((vid * len(S.ARMS) + ai) * NREG + rid) * 2 + sg
                                mt = m & tr
                                x = v[mt]
                                stats[ci, 7] = m.sum()
                                if len(x) == 0:
                                    stats[ci, :7] = [0, 0, 0, 0, 0, 0, 0]
                                    continue
                                k = max(1, int(np.ceil(0.05 * len(x))))
                                top = np.partition(x, len(x) - k)[len(x) - k:].sum()
                                w_ = x > 0
                                stats[ci, :7] = [len(x), x.sum(), w_.sum(), x[w_].sum(), x[~w_].sum(), (~w_).sum(), top]
                                WS[ci] = np.bincount(wk[mt], weights=x, minlength=W)[:W]
                                WN[ci] = np.bincount(wk[mt], minlength=W)[:W]
                                if True:  # family pool over every variant of the timeframe (H-BOS2)
                                    hi = np.searchsorted(HBINS, x, side="right") - 1
                                    hist_c[ai, rid, sg] += np.bincount(hi, minlength=len(HBINS) - 1)
                                    hist_s[ai, rid, sg] += np.bincount(hi, weights=x, minlength=len(HBINS) - 1)
                                    fam_ws[ai, rid, sg] += np.bincount(wk[mt], weights=x, minlength=W)[:W]
                                    fam_wn[ai, rid, sg] += np.bincount(wk[mt], minlength=W)[:W]
                    if a == "p":
                        ok = tr
                        alloc.append(pl.DataFrame({
                            "vid": np.full(ok.sum(), variant_id(tf, s, g, ei), np.int16),
                            "rcs": d["rcs"].to_numpy()[ok].astype(np.int8), "wk": wk[ok].astype(np.int16),
                            "xday": d["xday"].to_numpy()[ok].astype(np.int16), "xwk": d["xwk"].to_numpy()[ok].astype(np.int16),
                            "seg": seg[ok], "v": v[ok].astype(np.float32)}))
            del df
            gc.collect()
            print(f"agg {tf} {a} {e}", flush=True)
    np.savez_compressed(out / f"{tf}.npz", WS=WS, WN=WN, stats=stats, hist_c=hist_c, hist_s=hist_s, fam_ws=fam_ws, fam_wn=fam_wn)
    if alloc:
        pl.concat(alloc).write_parquet(out / f"{tf}_alloc_p.parquet")
    exec_pass(tf)
    print(f"done {tf} class={cls}", flush=True)


def exec_pass(tf: str) -> None:
    """H-BOS5 per-signal paired arms (missed = 0 R), pooled over the variants of the timeframe; plus missed-winner counts."""
    out = {}
    base = ["struct", "sig_ts", "side", "choch", "mom", "cell", "vt", "st", "r_fill", "r24_fill"]
    acc = {}
    for e in S.EXITS:
        cols = base + [f"m_{e}", f"p_{e}", f"r_{e}", f"rb_{e}", f"r24_{e}", f"m_{e}_x"]
        df = _keys(read_cols(tf, cols), f"m_{e}_x")
        df = df.filter((pl.col("seg") >= 0) & pl.col(f"m_{e}").is_not_nan())
        # retest: missed -> 0; filled-but-incomplete -> drop the signal
        for a, fc in (("r", "r_fill"), ("rb", "r_fill"), ("r24", "r24_fill")):
            df = df.with_columns(pl.when(pl.col(fc)).then(pl.col(f"{a}_{e}")).otherwise(0.0).alias(f"{a}_{e}"))
        df = df.filter(pl.col(f"r_{e}").is_not_nan() & pl.col(f"r24_{e}").is_not_nan() & pl.col(f"rb_{e}").is_not_nan())
        for g, flt in enumerate((None, "choch", "mom")):
            d = df if flt is None else df.filter(pl.col(flt))
            for sg in (0, 1):
                x = d.filter(pl.col("seg") == sg)
                if x.height == 0:
                    continue
                wk = x["wk"].to_numpy()
                m, p, r, rb, r24 = (x[f"{a}_{e}"].to_numpy().astype(float) for a in ("m", "p", "r", "rb", "r24"))
                fill = x["r_fill"].to_numpy()
                for nm, val in (("m", m), ("p", p), ("r", r), ("rb", rb), ("r24", r24), ("r-m", r - m), ("r-p", r - p),
                                ("p-m", p - m), ("rb-m", rb - m), ("r24-m", r24 - m), ("rb-p", rb - p), ("r24-p", r24 - p)):
                    k = (g, sg, nm)
                    a_ = acc.setdefault(k, [np.zeros(W), np.zeros(W)])
                    a_[0] += np.bincount(wk, weights=val, minlength=W)[:W]
                    a_[1] += np.bincount(wk, minlength=W)[:W]
                # missed-winner analysis on the market arm
                k = (g, sg, "mw")
                q95 = np.quantile(m, 0.95)
                c = acc.setdefault(k, np.zeros(8))
                win = m > 0
                top = m >= q95
                c += [len(m), win.sum(), (win & ~fill).sum(), top.sum(), (top & ~fill).sum(), (~fill).sum(), m[top].sum(), m[top & ~fill].sum()]
        del df
        gc.collect()
    for k, v in acc.items():
        out["|".join(map(str, k))] = np.stack(v) if isinstance(v, list) else v
    np.savez_compressed(agg_dir() / f"{tf}_exec.npz", **out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["agg"])
    ap.add_argument("--tf", default="4h,1h")
    a = ap.parse_args()
    for tf in a.tf.split(","):
        agg_tf(tf)


if __name__ == "__main__":
    main()

