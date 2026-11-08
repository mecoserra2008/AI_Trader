"""Directional-change (DC) pivots: port of the TradingView indicator "Pivot geometry trend (DC pivots)".

Evaluated on bar closes. For each threshold multiple K an independent engine confirms a swing pivot
once price has reversed by delta_t = K * sigma_{t-1} from the running extreme of the current run,
where sigma is the EWMA std of bar-to-bar close CHANGES in price points (half-life `vol_hl`).
A pivot is known only at its confirmation bar, which is what makes everything here causal.

Per K three votes in {-1, 0, +1}:
  STRUCT    stateless Dow structure: higher high & higher low = +1, lower high & lower low = -1.
  PIVOT_BO  stateful: close above the last swing high = +1, below the last swing low = -1, else hold.
  FLAG_BO   stateful continuation: after a higher low, a close above H0 with retracement
            (H0-L0)/(H0-L1) <= max_retr enters long with a trailing structural stop at L0 (ratchets up
            with new lows); symmetric short. Flat on a close through the stop; no re-entry on the exit bar.
The per-model ensemble is the mean of the K votes, emitted only when `ready`.

Order within a bar follows the Pine script: pivot update, then STRUCT, PIVOT_BO, FLAG_BO using the
updated H0/L0/H1/L1/lastTy.

Port decisions (where the Pine is silent):
  * bar_index is the row index of the frame passed in (truncating at the end does not change it).
  * When delta is invalid (NaN or <= 0, only possible during the first bars) the engine and the three
    signal states are not updated for that bar; they hold their previous values.
  * Non-finite closes are skipped the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

KS = (3, 4, 6, 8, 12)


def indicator_sigma(close: np.ndarray, vol_hl: float = 60, warm: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """EWMA std of close-to-close changes (price points) and the `ready` flag (bar_index > warm, sigma > 0)."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    alpha = 1.0 - 0.5 ** (1.0 / vol_hl)
    sig = np.full(n, np.nan)
    mean, var, started = 0.0, 0.0, False
    for t in range(1, n):
        chg = close[t] - close[t - 1]
        if not np.isfinite(chg):
            sig[t] = np.sqrt(var) if started else np.nan
            continue
        if not started:
            mean, var, started = chg, 0.0, True
        else:
            dev = chg - mean
            mean += alpha * dev
            var = (1.0 - alpha) * (var + alpha * dev * dev)
        sig[t] = np.sqrt(var)
    with np.errstate(invalid="ignore"):
        ready = (np.arange(n) > warm) & np.isfinite(sig) & (sig > 0)
    return sig, ready


@dataclass
class EngineResult:
    H0: np.ndarray
    H1: np.ndarray
    L0: np.ndarray
    L1: np.ndarray
    last_ty: np.ndarray       # +1 last confirmed pivot was a high, -1 a low, 0 none yet
    m: np.ndarray             # 0 warm-up, +1 up-run, -1 down-run
    ext: np.ndarray           # running extreme of the current run
    struct: np.ndarray
    pivot_bo: np.ndarray
    flag_bo: np.ndarray
    fstop: np.ndarray         # FLAG_BO trailing stop (NaN when flat)
    flag_entry: np.ndarray    # +1/-1 on a FLAG_BO entry bar, else 0
    events: pl.DataFrame      # confirmed pivots: price, type (+1 high / -1 low), pivot_bar, confirm_bar


def dc_engine(close: np.ndarray, sigma_prev: np.ndarray, k: float, max_retr: float = 0.618) -> EngineResult:
    """One K. `sigma_prev[t]` is the indicator sigma of bar t-1 (delta_t = k * sigma_prev[t])."""
    p_all = np.asarray(close, dtype=float)
    d_all = k * np.asarray(sigma_prev, dtype=float)
    n = len(p_all)
    nan = np.nan
    out = {c: np.full(n, nan) for c in ("H0", "H1", "L0", "L1", "ext", "fstop")}
    last_ty_a, m_a = np.zeros(n), np.zeros(n)
    struct_a, pbo_a, fbo_a, fent_a = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    ev_price: list[float] = []
    ev_type: list[int] = []
    ev_pbar: list[int] = []
    ev_cbar: list[int] = []

    H0 = H1 = L0 = L1 = nan
    last_ty = 0
    m = 0
    ext = hiR = loR = nan
    extB = hiB = loB = -1
    first = True
    pbo = 0
    fbo = 0
    fstop = nan

    for t in range(n):
        p, d = p_all[t], d_all[t]
        if np.isfinite(p) and np.isfinite(d) and d > 0:
            piv = None  # (price, type, bar)
            if first:
                ext = hiR = loR = p
                extB = hiB = loB = t
                first = False
            if m == 0:
                if p > hiR:
                    hiR, hiB = p, t
                if p < loR:
                    loR, loB = p, t
                if p - loR >= d:
                    piv = (loR, -1, loB)
                    m, ext, extB = 1, p, t
                elif hiR - p >= d:
                    piv = (hiR, 1, hiB)
                    m, ext, extB = -1, p, t
            elif m == 1:
                if ext - p >= d:  # reversal test before the extreme update
                    piv = (ext, 1, extB)
                    m, ext, extB = -1, p, t
                elif p > ext:
                    ext, extB = p, t
            else:
                if p - ext >= d:
                    piv = (ext, -1, extB)
                    m, ext, extB = 1, p, t
                elif p < ext:
                    ext, extB = p, t
            if piv is not None:
                price, ty, bar = piv
                if ty == 1:
                    H1, H0, last_ty = H0, price, 1
                else:
                    L1, L0, last_ty = L0, price, -1
                ev_price.append(price)
                ev_type.append(ty)
                ev_pbar.append(bar)
                ev_cbar.append(t)

            # STRUCT (stateless)
            st = 0
            if np.isfinite(H1) and np.isfinite(L1):
                if H0 > H1 and L0 > L1:
                    st = 1
                elif H0 < H1 and L0 < L1:
                    st = -1
            struct_a[t] = st

            # PIVOT_BO (stateful)
            if np.isfinite(H0) and p > H0:
                pbo = 1
            elif np.isfinite(L0) and p < L0:
                pbo = -1

            # FLAG_BO (stateful)
            exited = False
            entry = 0
            if fbo == 1:
                fstop = L0 if not np.isfinite(fstop) else max(fstop, L0 if np.isfinite(L0) else fstop)
                if p < fstop:
                    fbo, fstop, exited = 0, nan, True
            elif fbo == -1:
                fstop = H0 if not np.isfinite(fstop) else min(fstop, H0 if np.isfinite(H0) else fstop)
                if p > fstop:
                    fbo, fstop, exited = 0, nan, True
            if fbo == 0 and not exited and np.isfinite(H0) and np.isfinite(L0):
                if last_ty == -1 and np.isfinite(L1) and L0 > L1 and p > H0:
                    den = H0 - L1
                    if den > 0 and (H0 - L0) / den <= max_retr:
                        fbo, fstop, entry = 1, L0, 1
                elif last_ty == 1 and np.isfinite(H1) and H0 < H1 and p < L0:
                    den = H1 - L0
                    if den > 0 and (H0 - L0) / den <= max_retr:
                        fbo, fstop, entry = -1, H0, -1
            fent_a[t] = entry
        else:
            struct_a[t] = struct_a[t - 1] if t else 0.0
        pbo_a[t], fbo_a[t] = pbo, fbo
        out["fstop"][t] = fstop
        out["H0"][t], out["H1"][t], out["L0"][t], out["L1"][t] = H0, H1, L0, L1
        out["ext"][t] = ext
        last_ty_a[t], m_a[t] = last_ty, m

    events = pl.DataFrame(
        {"price": ev_price, "type": ev_type, "pivot_bar": ev_pbar, "confirm_bar": ev_cbar},
        schema={"price": pl.Float64, "type": pl.Int64, "pivot_bar": pl.Int64, "confirm_bar": pl.Int64},
    )
    return EngineResult(out["H0"], out["H1"], out["L0"], out["L1"], last_ty_a, m_a, out["ext"],
                        struct_a, pbo_a, fbo_a, out["fstop"], fent_a, events)


def run_engines(close: np.ndarray, ks=KS, vol_hl: float = 60, warm: int = 60, max_retr: float = 0.618):
    """Indicator sigma, ready flag and one EngineResult per K."""
    close = np.asarray(close, dtype=float)
    sig, ready = indicator_sigma(close, vol_hl, warm)
    sig_prev = np.r_[np.nan, sig[:-1]] if len(sig) else sig
    return sig, ready, {int(k): dc_engine(close, sig_prev, k, max_retr) for k in ks}


def ensemble(results: dict[int, EngineResult], ready: np.ndarray) -> dict[str, np.ndarray]:
    """Mean of the K votes per model, zero where not ready."""
    out = {}
    for model, attr in (("struct", "struct"), ("pivot_bo", "pivot_bo"), ("flag_bo", "flag_bo")):
        votes = np.vstack([getattr(r, attr) for r in results.values()])
        out[model] = np.where(ready, votes.mean(axis=0), 0.0)
    return out


def geometry(frame: pl.DataFrame, ks=KS, vol_hl: float = 60, warm: int = 60, max_retr: float = 0.618, level_k: int = 6) -> pl.DataFrame:
    """Per-bar ensembles (geo_struct, geo_pivot_bo, geo_flag_bo), indicator sigma/ready and the level_k levels."""
    close = frame["close"].cast(pl.Float64).fill_null(float("nan")).to_numpy()
    sig, ready, res = run_engines(close, ks, vol_hl, warm, max_retr)
    ens = ensemble(res, ready)
    cols = {
        "geo_sigma": sig, "geo_ready": ready,
        "geo_struct": ens["struct"], "geo_pivot_bo": ens["pivot_bo"], "geo_flag_bo": ens["flag_bo"],
    }
    if level_k in res:
        r = res[level_k]
        cols.update({f"geo_H0_{level_k}": r.H0, f"geo_H1_{level_k}": r.H1, f"geo_L0_{level_k}": r.L0,
                     f"geo_L1_{level_k}": r.L1, f"geo_last_ty_{level_k}": r.last_ty, f"geo_fstop_{level_k}": r.fstop})
    return pl.DataFrame(cols)
