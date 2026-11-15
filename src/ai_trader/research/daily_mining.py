"""FDR-controlled hypothesis mining on the universe-L daily panel (H-MINE2, program 2026-11-12, agent H). Research only.

    PYTHONIOENCODING=utf-8 uv run --extra ml python -m ai_trader.research.daily_mining declare    # family + diary, no statistic
    ... discover   # llm_in_sample statistics, BH-FDR 10%, freeze the list (+ diary)
    ... confirm    # clean_holdout, once; refuses without the frozen file; records H-MINE2 once
    ... report     # research/reports/daily_mining_2026-11-12.{json,md}

Panel. ai_trader.ml.features universe L (23 crypto perps), complete rows (the 38,275-row H-NN1 sample), one row per
symbol and UTC day D, decision time T = 00:00 UTC of D+1, every feature point in time. Targets:
  vr1  log(RV_{D+1} / F_HAR), F_HAR = H-NN1's walk-forward per-asset log-HAR forecast (monthly purged refits, wf_preds,
       from 2022-04): the part of next-day variance the HAR baseline misses                    [kind 'vol']
  r1   next-day vol-scaled return ret_{D+1} / sigma_D, winsorised at +-5                        [kind 'return']
  r5   next-5-day vol-scaled return sum ret_{D+1..D+5} / (sigma_D sqrt 5), five consecutive calendar days, +-5
       winsorised; its own segments: llm_in_sample iff D + 6 days <= cutoff, clean_holdout iff D + 1 day >= cutoff
       (rows whose 5-day label straddles the cutoff are dropped)                               [kind 'return']
Segments: DISCOVERY = llm_in_sample, CONFIRMATION = clean_holdout (research.knowledge_cutoff, 2026-07-01).

States (discretised; every quantile edge fitted on the DISCOVERY rows only, frozen in the family file and re-used
verbatim on the holdout; bin = number of edges strictly below the value, so ties of a discrete feature fall low):
  F1  terciles of each of the top-10 H-NN3 TreeSHAP features (in-sample main effects: log_sigma, lrv_w, lrv_q,
      next_is_weekend, dow_sin, lrv_m, next_wk_hours, lrv_d, rq_ratio_d, v0_dir_score), + top-minus-bottom contrast
  F2  quintiles of the top-3 (log_sigma, lrv_w, lrv_q), + Q5-minus-Q1 contrast
  F3  all pairs of terciles among the top-5 (10 pairs, 3 x 3 cells)
  F4  forecast weekday (7)
  F5  v0 regime cell (9: direction x vol state at the 20:00 4h bar of D)
  F6  funding_z_90d tercile, + top-minus-bottom contrast
  F7  drawdown flag: dd_90d_z in the bottom fifth of the discovery rows (deep 90d drawdown) vs not, + flag contrast
  F8  BTC's v0 trend x vol state on D, alts only (9)
Effect = mean target in the cell minus the mean over the state's population (all rows, or alts for F8), or a
contrast (top cell minus bottom cell) where the state is ordered. Cells empty in the discovery rows are not effects.
Family = every (target, state, cell / contrast) effect; size stated in the declaration diary entry.

Inference. Cluster-robust SE of the effect from its influence function, clusters = 10-day calendar blocks (counted from
the cutoff), which covers the cross-sectional dependence of a date and the 5-day label overlap of r5; small-sample
factor G / (G - 1), t with df = (clusters touched by the cell) - 1. A date-clustered SE is reported alongside
(secondary). Discovery: an effect is testable iff every cell it uses has >= 100 discovery rows on >= 30 dates;
untestable effects keep p = 1 and stay in the family. Benjamini-Hochberg over the whole family at q = 0.10.
Confirmation: each frozen discovery once on clean_holdout, frozen edges, one-sided in the discovered sign, p < 0.05
(10-day block clusters); it needs >= 20 holdout rows in each cell and >= 5 clusters, otherwise it counts as not
replicated. Replication rate overall, by target kind (vol vs return), by target and by family.

H-MINE2 (pooled, recorded once): supports iff rate >= 0.5 and vol rate > return rate; contradicts if rate < 0.5, or if
both kinds have discoveries and vol rate <= return rate; inconclusive if rate >= 0.5 but one kind has no discoveries
(the comparison cannot be made).

Known-rule tags (fixed in KNOWN before discovery; descriptive): each discovery is tagged 'known' (restates an existing
rule), 'contradicts' (would contradict an existing rule) or 'novel', with the rule ids, from its target kind, its
state family and the sign of the effect (known_tag). Overlap with agent C's LAT1 cells (weekday group x own-vol
tercile x v0 direction) is tagged separately (overlap_c).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

TAG = "2026-11-12"
HID = "H-MINE2"
ROOT = Path(__file__).resolve().parents[3]
MINING_DIR = ROOT / "data" / "research" / "mining"
FAMILY_FILE = MINING_DIR / "daily_family.json"
DISCOVERY_FILE = MINING_DIR / "daily_discovery.json"
FROZEN = MINING_DIR / "daily_discoveries_frozen.json"
CONFIRM_FILE = MINING_DIR / "daily_confirmation.json"
REPORT_JSON = ROOT / "research" / "reports" / f"daily_mining_{TAG}.json"
REPORT_MD = ROOT / "research" / "reports" / f"daily_mining_{TAG}.md"

DISC, CONF = "llm_in_sample", "clean_holdout"
TOP10 = ("log_sigma", "lrv_w", "lrv_q", "next_is_weekend", "dow_sin", "lrv_m", "next_wk_hours", "lrv_d", "rq_ratio_d", "v0_dir_score")
TOP5, TOP3 = TOP10[:5], TOP10[:3]
TARGETS = {"vr1": "vol", "r1": "return", "r5": "return"}
TARGET_WORDS = {"vr1": "next-day RV relative to the log-HAR forecast", "r1": "next-day vol-scaled return",
                "r5": "next-5-day vol-scaled return"}
CLIP = 5.0
BLOCK_DAYS = 10
FDR_Q = 0.10
ALPHA_CONF = 0.05
MIN_ROWS_DISC, MIN_DATES_DISC = 100, 30
MIN_ROWS_CONF, MIN_CLUSTERS_CONF = 20, 5
TERC_P = (1 / 3, 2 / 3)
QUINT_P = (0.2, 0.4, 0.6, 0.8)
DD_P = (0.2,)
TERC_W = ("in the bottom third", "in the middle third", "in the top third")
QUINT_W = ("in the bottom fifth", "in the 2nd fifth", "in the middle fifth", "in the 4th fifth", "in the top fifth")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
V0_CELLS = ("R1 quiet trend up", "R2 trend up", "R3 momentum burst", "R4 compression", "R5 balanced range", "R6 whipsaw",
            "R7 grind down", "R8 trend down", "R9 liquidation")
CAL_COLS = {"next_is_weekend", "dow_sin", "next_wk_hours", "wd"}
VOL_LEVEL_COLS = {"log_sigma", "lrv_w", "lrv_q", "lrv_m", "lrv_d"}
COL_WORDS = {
    "log_sigma": "robust 1h sigma (log)", "lrv_w": "7-day mean log RV", "lrv_q": "90-day mean log RV",
    "next_is_weekend": "forecast day is Sat/Sun", "dow_sin": "weekday sine (Mon 0, Tue-Thu > 0, Fri-Sun < 0)",
    "lrv_m": "30-day mean log RV", "next_wk_hours": "TradFi-weekend hours in the forecast day", "lrv_d": "day-D log RV",
    "rq_ratio_d": "day-D quarticity ratio (one-big-hour share)", "v0_dir_score": "v0 trend score",
    "funding_z_90d": "funding z (90d)", "dd_90d_z": "90d drawdown z",
}


# ------------------------------------------------------------------ family specification


@dataclass(frozen=True)
class Dim:
    col: str
    kind: str                      # 'q' (discovery quantile edges) or 'cat' (fixed levels)
    probs: tuple[float, ...] = ()  # quantile probabilities for 'q'
    levels: tuple[float, ...] = ()  # values for 'cat' (value -> index)
    words: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return f"{self.col}:{len(self.probs)}" if self.kind == "q" else self.col

    @property
    def size(self) -> int:
        return len(self.probs) + 1 if self.kind == "q" else len(self.levels)


@dataclass(frozen=True)
class State:
    id: str
    family: str
    dims: tuple[Dim, ...]
    population: str = "all"   # 'all' or 'alts'
    contrast: bool = False    # top-minus-bottom along the (single) ordered dim

    @property
    def n_cells(self) -> int:
        return int(np.prod([d.size for d in self.dims]))


def _terc(c: str) -> Dim:
    return Dim(c, "q", TERC_P, words=TERC_W)


def states() -> list[State]:
    out = [State(f"T3:{c}", "F1", (_terc(c),), contrast=True) for c in TOP10]
    out += [State(f"Q5:{c}", "F2", (Dim(c, "q", QUINT_P, words=QUINT_W),), contrast=True) for c in TOP3]
    out += [State(f"T3xT3:{a}x{b}", "F3", (_terc(a), _terc(b))) for i, a in enumerate(TOP5) for b in TOP5[i + 1:]]
    out.append(State("weekday", "F4", (Dim("wd", "cat", levels=tuple(float(k) for k in range(1, 8)), words=tuple(f"forecast day {w}" for w in WEEKDAYS)),)))
    out.append(State("v0cell", "F5", (Dim("v0cell", "cat", levels=tuple(float(k) for k in range(9)), words=tuple(f"v0 cell {w}" for w in V0_CELLS)),)))
    out.append(State("T3:funding_z_90d", "F6", (_terc("funding_z_90d"),), contrast=True))
    out.append(State("ddflag", "F7", (Dim("dd_90d_z", "q", DD_P, words=("in a deep 90d drawdown (bottom fifth)", "not in a deep drawdown")),), contrast=True))
    out.append(State("btc_v0cell", "F8", (Dim("btc_v0cell", "cat", levels=tuple(float(k) for k in range(9)), words=tuple(f"BTC v0 cell {w}" for w in V0_CELLS)),),
                     population="alts"))
    return out


def fit_edges(x: np.ndarray, probs: tuple[float, ...]) -> list[float]:
    """Quantile edges from the given (discovery) values only."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, probs)]


def assign_bins(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin = number of edges strictly below x (ties of a discrete feature fall into the lower bin); NaN -> -1."""
    x = np.asarray(x, dtype=float)
    b = np.searchsorted(np.asarray(edges, dtype=float), x, side="left")
    return np.where(np.isfinite(x), b, -1)


def fit_all_edges(disc: pl.DataFrame, sts: list[State] | None = None) -> dict[str, list[float]]:
    """Every quantile edge of the family, fitted on the DISCOVERY frame only (callers pass llm_in_sample rows)."""
    if disc["segment"].unique().to_list() != [DISC]:
        raise ValueError("edges must be fitted on llm_in_sample rows only")
    edges: dict[str, list[float]] = {}
    for s in sts or states():
        for d in s.dims:
            if d.kind == "q" and d.key not in edges:
                edges[d.key] = fit_edges(disc[d.col].cast(pl.Float64).to_numpy(), d.probs)
    return edges


def state_codes(df: pl.DataFrame, s: State, edges: dict[str, list[float]]) -> np.ndarray:
    """Cell id per row in [0, n_cells), -1 if outside the state's population or a dimension is missing."""
    code = np.zeros(df.height, dtype=int)
    bad = np.zeros(df.height, dtype=bool)
    for d in s.dims:
        x = df[d.col].cast(pl.Float64).fill_null(np.nan).to_numpy()
        if d.kind == "q":
            b = assign_bins(x, edges[d.key])
        else:
            b = np.full(len(x), -1)
            for i, lv in enumerate(d.levels):
                b[x == lv] = i
        bad |= b < 0
        code = code * d.size + np.maximum(b, 0)
    if s.population == "alts":
        bad |= (df["symbol"] == "BTCUSDT").to_numpy()
    return np.where(bad, -1, code)


def _coords(s: State, cell: int) -> list[int]:
    out = []
    for d in reversed(s.dims):
        out.append(cell % d.size)
        cell //= d.size
    return out[::-1]


# discrete calendar features: the weekday set of each tercile bin (checked on the discovery rows before declaring)
CAL_BIN_WORDS = {
    ("next_is_weekend", 0): "forecast day Mon-Fri", ("next_is_weekend", 2): "forecast day Sat/Sun",
    ("dow_sin", 0): "forecast day Fri-Sun", ("dow_sin", 1): "forecast day Mon or Thu", ("dow_sin", 2): "forecast day Tue or Wed",
    ("next_wk_hours", 0): "forecast day Mon-Thu (no TradFi-weekend hours)", ("next_wk_hours", 1): "forecast day Fri (TradFi weekend starts)",
    ("next_wk_hours", 2): "forecast day Sat/Sun (TradFi weekend)",
}


def _dim_label(d: Dim, b: int, edges: dict[str, list[float]]) -> str:
    if d.kind == "cat":
        return d.words[b]
    if len(d.probs) == 2 and (d.col, b) in CAL_BIN_WORDS:
        return CAL_BIN_WORDS[(d.col, b)]
    e = edges[d.key]
    rng = f"<= {e[0]:.3g}" if b == 0 else f"> {e[-1]:.3g}" if b == len(e) else f"{e[b - 1]:.3g} to {e[b]:.3g}"
    return f"{COL_WORDS.get(d.col, d.col)} {d.words[b]} ({rng})"


def describe_cell(s: State, cell: int, edges: dict[str, list[float]]) -> str:
    txt = ", ".join(_dim_label(d, b, edges) for d, b in zip(s.dims, _coords(s, cell), strict=True))
    return txt + (" [alts only]" if s.population == "alts" else "")


@dataclass
class Effect:
    id: str
    target: str
    kind: str
    family: str
    state: str
    cell: int = -1             # cell effect (vs population mean)
    hi: int = -1               # contrast hi - lo
    lo: int = -1
    describe: str = ""
    n_disc: int = 0
    dates_disc: int = 0
    testable: bool = False
    known: dict = field(default_factory=dict)

    @property
    def is_contrast(self) -> bool:
        return self.hi >= 0


# ------------------------------------------------------------------ known rules (fixed before discovery)

KNOWN = {
    # (target kind, condition) -> rules; evaluated by known_tag with the effect's sign
    "vol_calendar": ["R-VOL-1", "R-VOL-27", "R-ML-1", "R-DS-1", "R-DS-2", "BELIEFS V1/V10/V17"],
    "vol_calendar_x_level": ["R-ML-2", "R-SP-3"],
    "vol_v0": ["R-DS-4", "R-VOL-8", "R-ML-2"],
    "vol_level_meanrev": ["R-VOL-8 (high vol over-forecast)", "BELIEFS V9"],
    "vol_funding_up": ["R-LEV-5", "BELIEFS H1 (funding crowding tilts next-day vol up)"],
    "vol_btc_state": ["R-VOL-7", "R-VOL-25 (no BTC vol spillover beyond own HAR)"],
    "vol_drawdown": ["R-LEV-3 (drawdowns create no asymmetry)"],
    "return_any": ["R-DS-6", "R-BUDGET-1", "R-ML-5", "R-PX-6 (no direction from states / weekday / regime)"],
    "return_weekend": ["R-PX-9 (no weekend direction)"],
    "return_trend": ["R-PX-3 (thin trend edge)"],
}


def known_tag(e: Effect, sign: float, sts: dict[str, State]) -> dict:
    """Tag a discovery: 'known' (restates an existing rule), 'contradicts' (would contradict one) or 'novel'."""
    s = sts[e.state]
    cols = {d.col for d in s.dims}
    hi_side = None  # ordered dims: is the effect about the top (True) or the bottom (False) bins of every dim?
    if all(d.kind == "q" for d in s.dims):
        if e.is_contrast:
            hi_side = True
        else:
            bs = _coords(s, e.cell)
            if all(b == d.size - 1 for d, b in zip(s.dims, bs, strict=True)):
                hi_side = True
            elif all(b == 0 for b in bs):
                hi_side = False
    if e.kind == "vol":
        if cols & CAL_COLS:
            rules = list(KNOWN["vol_calendar"]) + (KNOWN["vol_calendar_x_level"] if cols & VOL_LEVEL_COLS else [])
            return {"status": "known", "rules": rules, "overlap_c": True}
        if cols & {"v0cell", "v0_dir_score"}:
            return {"status": "known", "rules": KNOWN["vol_v0"], "overlap_c": True}
        if cols <= VOL_LEVEL_COLS | {"rq_ratio_d"} and cols & VOL_LEVEL_COLS:
            # mean reversion beyond the forecast: high level -> below forecast, low level -> above
            if hi_side is not None and ((hi_side and sign < 0) or (not hi_side and sign > 0)):
                return {"status": "known", "rules": KNOWN["vol_level_meanrev"], "overlap_c": True}
            return {"status": "novel", "rules": [], "overlap_c": True}
        if "funding_z_90d" in cols:
            if hi_side is not None and ((hi_side and sign > 0) or (not hi_side and sign < 0)):
                return {"status": "known", "rules": KNOWN["vol_funding_up"], "overlap_c": False}
            return {"status": "novel", "rules": [], "overlap_c": False}
        if "btc_v0cell" in cols:
            return {"status": "contradicts", "rules": KNOWN["vol_btc_state"], "overlap_c": False}
        if "dd_90d_z" in cols:
            return {"status": "novel", "rules": KNOWN["vol_drawdown"], "overlap_c": False}
        return {"status": "novel", "rules": [], "overlap_c": False}
    rules = list(KNOWN["return_any"])
    if cols & CAL_COLS:
        rules += KNOWN["return_weekend"]
    if cols & {"v0cell", "v0_dir_score", "btc_v0cell"}:
        rules += KNOWN["return_trend"]
    return {"status": "contradicts", "rules": rules, "overlap_c": bool(cols & (CAL_COLS | {"v0cell"}))}


# Reviewed after the discovery list was frozen and BEFORE any clean_holdout statistic (diary 'H-MINE2 known-rule tags
# reviewed'): the mechanical map missed that the quarticity-ratio effect on the HAR residual is the HAR-Q correction.
# Descriptive only; the frozen list and the test are unchanged. The mechanical tag is kept as 'known_mechanical'.
KNOWN_REVIEW = {
    "vr1|T3:rq_ratio_d|": {"status": "known", "rules": ["R-VOL-23 (HAR-Q: a day dominated by one big hour gets less weight)",
                                                         "R-VOL-3 / BELIEFS V4 (spike excess variance decays within hours)"], "overlap_c": False},
}


def review_tag(d: dict) -> dict:
    for prefix, tag in KNOWN_REVIEW.items():
        if d["id"].startswith(prefix):
            return tag | {"reviewed": True}
    return d["known"]


# ------------------------------------------------------------------ data


def build_frame() -> pl.DataFrame:
    """Complete-row panel + targets vr1 / r1 / r5 (+ seg5), weekday, v0 cell, BTC's v0 cell, date int and clusters."""
    from ai_trader.research.evaluate import knowledge_cutoff
    from ai_trader.research.ml_signals_study import load_panel, root

    full, c = load_panel()
    cut = knowledge_cutoff().date()
    wf = pl.read_parquet(root() / "wf_preds.parquet").select("symbol", "T", "F_har")
    c = c.join(wf, on=["symbol", "T"], how="left")
    # r5 from the full panel: 5 consecutive calendar days of ret_next
    parts = []
    for s in full["symbol"].unique().sort().to_list():
        f = full.filter(pl.col("symbol") == s).sort("date")
        d = f["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
        r = f["ret_next"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        sig = np.exp(f["log_sigma"].cast(pl.Float64).fill_null(np.nan).to_numpy())
        n = len(r)
        r5 = np.full(n, np.nan)
        if n >= 5:
            win = np.lib.stride_tricks.sliding_window_view(r, 5).sum(1)
            span = d[4:] - d[:-4]
            ok = span == 4
            r5[: n - 4] = np.where(ok, win, np.nan)
        with np.errstate(all="ignore"):
            z5 = r5 / (sig * math.sqrt(5))
        parts.append(pl.DataFrame({"symbol": [s] * n, "date": f["date"], "r5_raw": z5}))
    r5df = pl.concat(parts)
    c = c.join(r5df, on=["symbol", "date"], how="left")
    btc = c.filter(pl.col("symbol") == "BTCUSDT").select("date", ((1 - pl.col("v0_dir")) * 3 + pl.col("v0_vol")).alias("btc_v0cell"))
    c = c.join(btc, on="date", how="left")
    # v0 cell index: up/low=R1 .. down/high=R9 -> (dir up=0, range=1, down=2) * 3 + vol
    c = c.with_columns(
        ((1 - pl.col("v0_dir")) * 3 + pl.col("v0_vol")).alias("v0cell"),
        (pl.col("date") + pl.duration(days=1)).dt.weekday().cast(pl.Float64).alias("wd"),
        (pl.col("rv_next") / pl.col("F_har")).log().alias("vr1"),
        pl.col("ret_next_z").clip(-CLIP, CLIP).alias("r1"),
        pl.col("r5_raw").clip(-CLIP, CLIP).alias("r5"),
        pl.when(pl.col("date") + pl.duration(days=6) <= cut).then(pl.lit(DISC))
        .when(pl.col("date") + pl.duration(days=1) >= cut).then(pl.lit(CONF)).otherwise(None).alias("seg5"),
    )
    di = c["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    cut_i = np.datetime64(cut, "D").astype(np.int64)
    c = c.with_columns(pl.Series("date_i", di), pl.Series("block", np.floor_divide(di - cut_i, BLOCK_DAYS)))
    return c.with_columns(pl.col(pl.Float64).fill_nan(None))


def target_rows(frame: pl.DataFrame, target: str, segment: str, frozen: Path | None = None) -> pl.DataFrame:
    """Rows of one target in one segment. The holdout is behind the code wall: it needs the frozen discovery file."""
    if segment == CONF:
        require_frozen(frozen or FROZEN)
    elif segment != DISC:
        raise ValueError(segment)
    segcol = "seg5" if target == "r5" else "segment"
    return frame.filter((pl.col(segcol) == segment) & pl.col(target).is_not_null() & pl.col(target).is_finite())


# ------------------------------------------------------------------ statistics


def effect_stat(v: np.ndarray, codes: np.ndarray, clusters: np.ndarray, dates: np.ndarray, cell: int = -1, hi: int = -1,
                lo: int = -1, sided: int = 0) -> dict[str, float]:
    """Cell mean minus population mean (cell >= 0) or contrast hi - lo, with cluster-robust SE from the influence
    function (clusters: 10-day blocks; a date-clustered SE is computed alongside). sided 0 = two-sided p, +1 / -1 =
    one-sided in that direction. df = clusters touched by the cell(s) - 1."""
    from scipy import stats

    ok = (codes >= 0) & np.isfinite(v)
    v, codes, clusters, dates = v[ok], codes[ok], clusters[ok], dates[ok]
    n = len(v)
    nan = {"d": math.nan, "se": math.nan, "t": math.nan, "df": 0, "p": 1.0, "se_date": math.nan, "t_date": math.nan, "p_date": 1.0,
           "n": 0, "n_dates": 0, "n_clusters": 0, "n_pop": n}
    if n == 0:
        return nan
    m = float(v.mean())
    if cell >= 0:
        a = codes == cell
        na = int(a.sum())
        if na == 0:
            return nan
        ma = float(v[a].mean())
        d = ma - m
        psi = np.where(a, (v - ma) / na, 0.0) - (v - m) / n
        used = a
        n_cell, n_dates = na, len(np.unique(dates[a]))
        n_cl = len(np.unique(clusters[a]))
    else:
        h, lw = codes == hi, codes == lo
        nh, nl = int(h.sum()), int(lw.sum())
        if nh == 0 or nl == 0:
            return nan
        mh, ml = float(v[h].mean()), float(v[lw].mean())
        d = mh - ml
        psi = np.where(h, (v - mh) / nh, 0.0) - np.where(lw, (v - ml) / nl, 0.0)
        used = h | lw
        n_cell = min(nh, nl)
        n_dates = min(len(np.unique(dates[h])), len(np.unique(dates[lw])))
        n_cl = min(len(np.unique(clusters[h])), len(np.unique(clusters[lw])))
    out = {"d": d, "n": n_cell, "n_dates": n_dates, "n_clusters": n_cl, "n_pop": n}
    for tag, g in (("", clusters), ("_date", dates)):
        _, gi = np.unique(g, return_inverse=True)
        sums = np.bincount(gi, weights=psi)
        G = len(sums)
        var = G / (G - 1) * float((sums ** 2).sum()) if G > 1 else math.nan
        se = math.sqrt(var) if var > 0 else math.nan
        t = d / se if se and np.isfinite(se) else math.nan
        df = max(1, (n_cl if tag == "" else len(np.unique(dates[used]))) - 1)
        if not np.isfinite(t):
            p = 1.0
        elif sided == 0:
            p = float(2 * stats.t.sf(abs(t), df))
        else:
            p = float(stats.t.sf(sided * t, df))
        out |= {f"se{tag}": se, f"t{tag}": t, f"p{tag}": min(p, 1.0)}
        if tag == "":
            out["df"] = df
    return out


def bh(p: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    from ai_trader.research.micro_mining import bh as _bh

    return _bh(p, q)


def bh_adjusted(p: np.ndarray) -> np.ndarray:
    from ai_trader.research.micro_mining import bh_adjusted as _bha

    return _bha(p)


# ------------------------------------------------------------------ steps


class FrozenMissing(RuntimeError):
    pass


def require_frozen(frozen: Path = FROZEN) -> dict[str, Any]:
    if not Path(frozen).is_file():
        raise FrozenMissing(f"{frozen} does not exist: the H-MINE2 discovery list must be frozen before any clean_holdout statistic")
    obj = json.loads(Path(frozen).read_text(encoding="utf-8"))
    if obj.get("hypothesis") != HID or not obj.get("frozen_at") or "discoveries" not in obj or "edges" not in obj:
        raise FrozenMissing(f"{frozen} is not a frozen H-MINE2 discovery file")
    return obj


def effect_family(frame: pl.DataFrame, edges: dict[str, list[float]]) -> list[Effect]:
    """The declared family. Uses discovery FEATURES only (cell row / date counts), never a target value."""
    sts = states()
    fam: list[Effect] = []
    for tgt, kind in TARGETS.items():
        segcol = "seg5" if tgt == "r5" else "segment"
        dd = frame.filter((pl.col(segcol) == DISC) & pl.col(tgt).is_not_null())  # availability of the label only
        dates = dd["date_i"].to_numpy()
        for s in sts:
            codes = state_codes(dd, s, edges)
            cnt = np.bincount(codes[codes >= 0], minlength=s.n_cells)
            ndt = np.array([len(np.unique(dates[codes == k])) for k in range(s.n_cells)])
            for k in range(s.n_cells):
                if cnt[k] == 0:
                    continue
                fam.append(Effect(f"{tgt}|{s.id}|c{k}", tgt, kind, s.family, s.id, cell=k, describe=describe_cell(s, k, edges),
                                  n_disc=int(cnt[k]), dates_disc=int(ndt[k]),
                                  testable=bool(cnt[k] >= MIN_ROWS_DISC and ndt[k] >= MIN_DATES_DISC)))
            if s.contrast:
                hi, lo = s.n_cells - 1, 0
                if cnt[hi] > 0 and cnt[lo] > 0:
                    fam.append(Effect(f"{tgt}|{s.id}|hi-lo", tgt, kind, s.family, s.id, hi=hi, lo=lo,
                                      describe=f"{describe_cell(s, hi, edges)} minus {describe_cell(s, lo, edges)}",
                                      n_disc=int(min(cnt[hi], cnt[lo])), dates_disc=int(min(ndt[hi], ndt[lo])),
                                      testable=bool(min(cnt[hi], cnt[lo]) >= MIN_ROWS_DISC and min(ndt[hi], ndt[lo]) >= MIN_DATES_DISC)))
    return fam


def family_counts(fam: list[Effect]) -> dict[str, Any]:
    out: dict[str, Any] = {"total": len(fam), "testable": sum(e.testable for e in fam)}
    for key in ("target", "kind", "family"):
        vals = sorted({getattr(e, key) for e in fam})
        out[f"by_{key}"] = {v: sum(getattr(e, key) == v for e in fam) for v in vals}
    return out


def _effect_from(d: dict) -> Effect:
    return Effect(**{k: d[k] for k in Effect.__dataclass_fields__ if k in d})


def run_declare(write_diary: bool = True) -> dict[str, Any]:
    if FAMILY_FILE.exists():
        raise FileExistsError(f"{FAMILY_FILE} exists; the family is declared once")
    frame = build_frame()
    disc_all = frame.filter(pl.col("segment") == DISC)
    edges = fit_all_edges(disc_all)
    fam = effect_family(frame, edges)
    counts = family_counts(fam)
    obj = {"hypothesis": HID, "declared_at": datetime.now(UTC).isoformat(timespec="seconds"), "edges": edges,
           "edges_fitted_on": f"{DISC} complete rows ({disc_all.height})", "counts": counts,
           "states": [{"id": s.id, "family": s.family, "population": s.population, "contrast": s.contrast,
                       "dims": [asdict(d) for d in s.dims]} for s in states()],
           "protocol": {"fdr_q": FDR_Q, "alpha_conf": ALPHA_CONF, "block_days": BLOCK_DAYS, "clip": CLIP,
                        "min_disc": [MIN_ROWS_DISC, MIN_DATES_DISC], "min_conf": [MIN_ROWS_CONF, MIN_CLUSTERS_CONF]},
           "effects": [asdict(e) for e in fam]}
    MINING_DIR.mkdir(parents=True, exist_ok=True)
    FAMILY_FILE.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    if write_diary:
        _diary_declare(counts, edges)
    return obj


def _diary_declare(counts: dict, edges: dict) -> None:
    from ai_trader.research import diary

    fam_lines = "\n".join(f"- {k}: {v}" for k, v in counts["by_family"].items())
    body = f"""Effect family of H-MINE2, declared in src/ai_trader/research/daily_mining.py (states, effect_family) and written to data/research/mining/daily_family.json BEFORE any target statistic is computed. Only feature-side counts (rows / dates per cell on llm_in_sample rows whose label exists) were computed; no target mean, SE or p-value.

**Panel:** universe-L daily complete rows (ai_trader.ml.features, the H-NN1 sample), decision at 00:00 UTC of D+1, point in time.

**Targets:** vr1 = log(RV_next / walk-forward per-asset log-HAR forecast of H-NN1) [vol]; r1 = next-day vol-scaled return, +-5 winsorised [return]; r5 = next-5-day vol-scaled return (5 consecutive days, sigma_D sqrt 5), +-5 winsorised, own segments (in-sample iff D+6 <= cutoff) [return].

**States (edges fitted on llm_in_sample rows only, frozen):** F1 terciles of the top-10 H-NN3 TreeSHAP features ({', '.join(TOP10)}) + top-bottom contrast; F2 quintiles of the top-3 + Q5-Q1; F3 all 10 pairs of terciles among the top-5; F4 forecast weekday; F5 v0 regime cell (9); F6 funding_z_90d tercile + contrast; F7 drawdown flag (dd_90d_z bottom fifth) + contrast; F8 BTC v0 trend x vol cell, alts only. Cells empty on discovery are not effects.

**Effect:** cell mean minus the population mean (or top-minus-bottom contrast). **Family size: {counts['total']} effects** ({counts['by_kind'].get('vol', 0)} vol, {counts['by_kind'].get('return', 0)} return; by target {counts['by_target']}); {counts['testable']} testable (>= {MIN_ROWS_DISC} rows on >= {MIN_DATES_DISC} dates per cell), the rest keep p = 1.
{fam_lines}

**Inference:** cluster-robust SE (influence function), clusters = {BLOCK_DAYS}-day calendar blocks, t with df = clusters touched - 1; date-clustered SE reported as secondary. BH at FDR {FDR_Q} over the whole family on llm_in_sample.

**Confirmation (fixed now):** each frozen discovery once on clean_holdout with the frozen edges, one-sided in the discovered sign, p < {ALPHA_CONF} ({BLOCK_DAYS}-day clusters), needs >= {MIN_ROWS_CONF} rows per cell and >= {MIN_CLUSTERS_CONF} clusters (else not replicated). Verdict (pooled, once): supports iff rate >= 0.5 and vol rate > return rate; contradicts if rate < 0.5 or (both kinds discovered and vol rate <= return rate); inconclusive if rate >= 0.5 but one kind has no discoveries. Known-rule tags (KNOWN / known_tag) fixed in code now. Code wall: target_rows(clean_holdout) raises without data/research/mining/daily_discoveries_frozen.json."""
    diary.entry("hypothesis", f"H-MINE2 effect family declared ({counts['total']} effects) before any statistic", body,
                tags=["mining", "fdr", "daily"], hypotheses=[HID], links=["src/ai_trader/research/daily_mining.py"])


def load_family() -> tuple[dict, list[Effect]]:
    if not FAMILY_FILE.is_file():
        raise FileNotFoundError("declare the family first")
    obj = json.loads(FAMILY_FILE.read_text(encoding="utf-8"))
    return obj, [_effect_from(d) for d in obj["effects"]]


def _stats_for(frame: pl.DataFrame, fam: list[Effect], edges: dict, segment: str, sided: dict[str, int] | None = None,
               frozen: Path | None = None) -> dict[str, dict]:
    sts = {s.id: s for s in states()}
    out: dict[str, dict] = {}
    for tgt in TARGETS:
        effs = [e for e in fam if e.target == tgt]
        if not effs:
            continue
        rows = target_rows(frame, tgt, segment, frozen=frozen)
        v = rows[tgt].to_numpy().astype(float)
        cl, dt = rows["block"].to_numpy(), rows["date_i"].to_numpy()
        codes_cache: dict[str, np.ndarray] = {}
        for e in effs:
            if e.state not in codes_cache:
                codes_cache[e.state] = state_codes(rows, sts[e.state], edges)
            sd = 0 if sided is None else sided[e.id]
            out[e.id] = effect_stat(v, codes_cache[e.state], cl, dt, cell=e.cell, hi=e.hi, lo=e.lo, sided=sd)
            out[e.id]["pop_mean"] = float(np.nanmean(v[codes_cache[e.state] >= 0])) if (codes_cache[e.state] >= 0).any() else math.nan
    return out


def run_discover(write_diary: bool = True) -> dict[str, Any]:
    if FROZEN.exists():
        raise FileExistsError(f"{FROZEN} exists; discovery has been frozen already")
    fobj, fam = load_family()
    edges = fobj["edges"]
    frame = build_frame()
    st = _stats_for(frame, fam, edges, DISC)
    p = np.array([st[e.id]["p"] if e.testable else 1.0 for e in fam])
    rej = bh(p, FDR_Q)
    padj = bh_adjusted(p)
    sts = {s.id: s for s in states()}
    rows = []
    for e, r, pa, pp in zip(fam, rej, padj, p, strict=True):
        s = st[e.id]
        sign = float(np.sign(s["d"])) if np.isfinite(s["d"]) else 0.0
        row = asdict(e) | {k: s[k] for k in ("d", "se", "t", "df", "se_date", "t_date", "p_date", "n", "n_dates", "n_clusters", "n_pop", "pop_mean")}
        row |= {"p": float(pp), "p_bh": float(pa), "bh_reject": bool(r), "direction": sign}
        if r:
            row["known"] = known_tag(e, sign, sts)
        rows.append(row)
    disc = {"hypothesis": HID, "segment": DISC, "computed_at": datetime.now(UTC).isoformat(timespec="seconds"), "family": fobj["counts"],
            "n_discoveries": int(rej.sum()), "bh_threshold_p": float(max([pp for pp, r in zip(p, rej, strict=True) if r], default=0.0)),
            "effects": rows}
    DISCOVERY_FILE.write_text(json.dumps(disc, indent=1, default=float), encoding="utf-8")
    frozen = freeze(disc, edges)
    if write_diary:
        _diary_freeze(disc, frozen)
    return disc


def freeze(disc: dict, edges: dict, frozen: Path = FROZEN) -> dict:
    if frozen.exists():
        raise FileExistsError(f"{frozen} already exists; the discovery list is frozen once")
    ds = [{k: r[k] for k in ("id", "target", "kind", "family", "state", "cell", "hi", "lo", "describe", "d", "t", "p", "p_bh", "n", "n_dates",
                            "direction", "known")} for r in disc["effects"] if r["bh_reject"]]
    obj = {"hypothesis": HID, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"), "segment_discovery": DISC, "segment_confirmation": CONF,
           "family": disc["family"], "fdr_q": FDR_Q, "alpha_confirmation": ALPHA_CONF, "edges": edges, "n_discoveries": len(ds), "discoveries": ds}
    frozen.parent.mkdir(parents=True, exist_ok=True)
    tmp = frozen.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, frozen)
    return obj


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _diary_freeze(disc: dict, frozen: dict) -> None:
    from ai_trader.research import diary

    ds = frozen["discoveries"]
    by_t = {t: sum(d["target"] == t for d in ds) for t in TARGETS}
    by_s = {s: sum(d["known"]["status"] == s for d in ds) for s in ("known", "contradicts", "novel")}
    top = sorted(ds, key=lambda d: d["p"])[:12]
    lines = "\n".join(f"- {d['id']}: {d['describe']}; d {d['d']:+.3f}, t {d['t']:.1f}, p_bh {d['p_bh']:.2g} [{d['known']['status']}]" for d in top)
    body = f"""The H-MINE2 discovery list is frozen in data/research/mining/daily_discoveries_frozen.json (sha256 {_sha(FROZEN)}; frozen_at {frozen['frozen_at']}), together with the discovery-fitted edges. No clean_holdout statistic has been computed (target_rows raises on clean_holdout until this file exists).

**Discovery (llm_in_sample):** family {disc['family']['total']} effects ({disc['family']['testable']} testable); BH at FDR {FDR_Q}: **{len(ds)} discoveries** (largest rejected p {disc['bh_threshold_p']:.3g}). By target: {by_t}. Known-rule tags: {by_s}.

Strongest 12 by p:
{lines}"""
    diary.entry("hypothesis", "H-MINE2 discovery list frozen", body, tags=["mining", "fdr", "daily"], hypotheses=[HID],
                links=["data/research/mining/daily_discoveries_frozen.json"])


def _rate(xs: list[bool]) -> dict[str, Any]:
    n, k = len(xs), int(sum(xs))
    if n == 0:
        return {"n": 0, "replicated": 0, "rate": None, "wilson95": None}
    p, z = k / n, 1.959964
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"n": n, "replicated": k, "rate": p, "wilson95": [max(0.0, c - h), min(1.0, c + h)]}


def verdict(summary: dict) -> str:
    tot, vol, ret = summary["overall"], summary["by_kind"]["vol"], summary["by_kind"]["return"]
    if tot["n"] == 0:
        return "inconclusive"
    if tot["rate"] < 0.5:
        return "contradicts"
    if vol["n"] == 0 or ret["n"] == 0:
        return "inconclusive"
    return "supports" if vol["rate"] > ret["rate"] else "contradicts"


def run_confirm(record: bool = True, frozen_path: Path = FROZEN) -> dict[str, Any]:
    fz = require_frozen(frozen_path)
    if CONFIRM_FILE.exists():
        raise FileExistsError(f"{CONFIRM_FILE} exists; confirmation runs once")
    edges = fz["edges"]
    ds = fz["discoveries"]
    fam = [_effect_from(d) for d in ds]
    frame = build_frame()
    sided = {d["id"]: int(d["direction"]) for d in ds}
    st = _stats_for(frame, fam, edges, CONF, sided=sided, frozen=frozen_path)
    rows = []
    for d in ds:
        s = st[d["id"]]
        testable = s["n"] >= MIN_ROWS_CONF and s["n_clusters"] >= MIN_CLUSTERS_CONF
        rep = bool(testable and np.isfinite(s["d"]) and np.sign(s["d"]) == d["direction"] and s["p"] < ALPHA_CONF)
        rows.append(d | {"known": review_tag(d), "known_mechanical": d["known"], "ho": {k: s[k] for k in ("d", "se", "t", "df", "p", "t_date", "p_date", "n", "n_dates", "n_clusters", "n_pop", "pop_mean")},
                         "ho_testable": bool(testable), "replicated": rep,
                         "replicated_date_clustered": bool(testable and np.sign(s["d"]) == d["direction"] and s["p_date"] < ALPHA_CONF)})
    summ = summarise(rows)
    v = verdict(summ)
    out = {"hypothesis": HID, "computed_at": datetime.now(UTC).isoformat(timespec="seconds"), "frozen_sha256": _sha(frozen_path),
           "summary": summ, "verdict": v, "effects": rows}
    CONFIRM_FILE.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    if record:
        _record(out)
    return out


def summarise(rows: list[dict]) -> dict[str, Any]:
    rep = [r["replicated"] for r in rows]
    out = {"overall": _rate(rep),
           "by_kind": {k: _rate([r["replicated"] for r in rows if r["kind"] == k]) for k in ("vol", "return")},
           "by_target": {t: _rate([r["replicated"] for r in rows if r["target"] == t]) for t in TARGETS},
           "by_family": {f: _rate([r["replicated"] for r in rows if r["family"] == f]) for f in sorted({r["family"] for r in rows})},
           "by_known": {s: _rate([r["replicated"] for r in rows if r["known"]["status"] == s]) for s in ("known", "contradicts", "novel")},
           "date_clustered_overall": _rate([r["replicated_date_clustered"] for r in rows]),
           "untestable": sum(not r["ho_testable"] for r in rows),
           "sign_agree": _rate([bool(np.isfinite(r["ho"]["d"]) and np.sign(r["ho"]["d"]) == r["direction"]) for r in rows])}
    out["restating_known"] = {"discoveries": sum(r["known"]["status"] == "known" for r in rows),
                              "replicated": sum(r["replicated"] and r["known"]["status"] == "known" for r in rows)}
    return out


def _record(out: dict) -> None:
    from ai_trader.research import hypotheses

    if hypotheses.results(HID):
        print(f"{HID} already recorded; not recording again")
        return
    s = out["summary"]
    m = {"n_family": json.loads(FROZEN.read_text(encoding="utf-8"))["family"]["total"], "n_discoveries": s["overall"]["n"],
         "replication_rate": s["overall"]["rate"], "replication_wilson95": s["overall"]["wilson95"],
         "vol": s["by_kind"]["vol"], "return": s["by_kind"]["return"], "by_target": s["by_target"],
         "restating_known": s["restating_known"], "date_clustered_rate": s["date_clustered_overall"]["rate"],
         "untestable_on_holdout": s["untestable"], "sign_agreement": s["sign_agree"]["rate"]}
    note = ("discovery = llm_in_sample, BH-FDR 10%, 10-day-block clustered t; confirmation once on clean_holdout, one-sided p < 0.05; "
            f"vol {s['by_kind']['vol']['replicated']}/{s['by_kind']['vol']['n']}, return {s['by_kind']['return']['replicated']}/{s['by_kind']['return']['n']}; "
            f"{s['restating_known']['discoveries']} discoveries restate known rules")
    hypotheses.record(HID, "pooled", out["verdict"], m, note=note, data="research/reports/daily_mining_2026-11-12.md")


# ------------------------------------------------------------------ report


def _effect_words(r: dict, seg: str = "disc") -> str:
    d = r["d"] if seg == "disc" else r["ho"]["d"]
    if r["kind"] == "vol":
        return f"{(math.exp(d) - 1) * 100:+.1f}%"
    return f"{d:+.3f} sigma"


def plain(r: dict) -> str:
    tw = TARGET_WORDS[r["target"]]
    if r["kind"] == "vol":
        size = f"{tw} about {(math.exp(r['d']) - 1) * 100:+.0f}% in-sample / {(math.exp(r['ho']['d']) - 1) * 100:+.0f}% holdout"
    else:
        size = f"{tw} {r['d']:+.3f} / {r['ho']['d']:+.3f} sigma (in-sample / holdout)"
    if r["hi"] >= 0:
        return f"{r['describe']}: {size} (contrast)"
    return f"When {r['describe']}: {size}, relative to the average row"


def detail(r: dict) -> str:
    ho = r["ho"]
    return (f"holdout {ho['n']} rows on {ho['n_dates']} dates ({ho['n_clusters']} ten-day blocks), t {ho['t']:.2f}, one-sided p {ho['p']:.3g}; "
            f"date-clustered p {ho['p_date']:.3g}")


def run_report() -> dict[str, Any]:
    fobj, fam = load_family()
    disc = json.loads(DISCOVERY_FILE.read_text(encoding="utf-8"))
    conf = json.loads(CONFIRM_FILE.read_text(encoding="utf-8"))
    rows = conf["effects"]
    s = conf["summary"]
    novel_rep = [r for r in rows if r["replicated"] and r["known"]["status"] != "known"]
    known_rep = [r for r in rows if r["replicated"] and r["known"]["status"] == "known"]
    rules = rule_candidates(novel_rep)
    rep_json = {"hypothesis": HID, "tag": TAG, "family": fobj["counts"], "edges": fobj["edges"], "protocol": fobj["protocol"],
                "discovery": {"n": disc["n_discoveries"], "bh_threshold_p": disc["bh_threshold_p"]}, "summary": s, "verdict": conf["verdict"],
                "frozen_sha256": conf["frozen_sha256"], "discoveries": rows, "rule_candidates": rules}
    REPORT_JSON.write_text(json.dumps(rep_json, indent=1, default=float), encoding="utf-8")
    REPORT_MD.write_text(report_md(fobj, disc, conf, novel_rep, known_rep, rules), encoding="utf-8")
    return rep_json


def _pct(x: dict) -> str:
    if x["n"] == 0:
        return "- (0)"
    lo, hi = x["wilson95"]
    return f"{x['replicated']}/{x['n']} = {100 * x['rate']:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def rule_candidates(novel_rep: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(sorted(novel_rep, key=lambda r: r["ho"]["p"])):
        if r["kind"] == "vol":
            mult = math.exp(r["d"])
            then = (f"central next-day variance expectation = log-HAR x {mult:.2f} relative to the average residual; "
                    "stops and risk keep the unshrunk risk sigma (R-VOL-2)")
        else:
            then = (f"treat the {TARGET_WORDS[r['target']]} as tilted by about {r['d']:+.2f} sigma; context for the decision layer only, "
                    "not a reason to change risk limits")
        desc = r["describe"]
        if desc.endswith(" [alts only]"):
            desc = "the perp is an alt (not BTC) and " + desc[: -len(" [alts only]")].replace("BTC v0 cell", "BTC's v0 cell is")
        cond = desc if r["hi"] < 0 else f"comparing {desc}"
        conflict = (f"; it contradicts {', '.join(r['known']['rules'])}, which stay in force until a forward test confirms this"
                    if r["known"]["status"] == "contradicts" else "")
        out.append({
            "id": f"R-MINE-{20 + i}",
            "kind": "expect",
            "rule": f"WHEN {cond} THEN expect {TARGET_WORDS[r['target']]} {_effect_words(r)} vs the average (in-sample), {_effect_words(r, 'ho')} holdout",
            "when": f"daily (1d) decision at 00:00 UTC for a universe-L crypto perp; H-MINE2 state {r['state']} (edges fitted on llm_in_sample)",
            "then": then,
            "scope": {"assets": "crypto perps (universe L)", "horizon": "5d" if r["target"] == "r5" else "1d", "sides": "both"},
            "effect": (f"{r['target']} effect {r['d']:+.4f} (t {r['t']:.2f}, {r['n']} rows, {r['n_dates']} dates) in-sample, "
                       f"{r['ho']['d']:+.4f} (t {r['ho']['t']:.2f}, one-sided p {r['ho']['p']:.3g}, {r['ho']['n']} rows, {r['ho']['n_dates']} dates) holdout; "
                       f"date-clustered holdout p {r['ho']['p_date']:.3g}; BH-FDR 10% discovery among {conf_family_total()} effects"),
            "confidence": "C",
            "evidence": ["H-MINE2"],
            "review": ("register as a forward hypothesis; demote if the effect on dates after 2026-11-12 changes sign or falls below half"
                       + conflict),
            "status": "provisional",
            "known_tag": r["known"]["status"] + (": " + ", ".join(r["known"]["rules"]) if r["known"]["rules"] else ""),
        })
    return out


def conf_family_total() -> int:
    return int(json.loads(FAMILY_FILE.read_text(encoding="utf-8"))["counts"]["total"])


def _q(x: Any) -> str:
    return json.dumps(str(x), ensure_ascii=False)


def rules_yaml(rules: list[dict]) -> str:
    if not rules:
        return "[]\n"
    out = []
    for r in rules:
        out.append(f"- id: {_q(r['id'])}")
        for k in ("kind", "rule", "when", "then"):
            out.append(f"  {k}: {_q(r[k])}")
        out.append("  scope:")
        for k, v in r["scope"].items():
            out.append(f"    {k}: {_q(v)}")
        out.append(f"  effect: {_q(r['effect'])}")
        out.append(f"  confidence: {_q(r['confidence'])}")
        out.append("  evidence: [" + ", ".join(_q(x) for x in r["evidence"]) + "]")
        out.append(f"  review: {_q(r['review'])}")
        out.append(f"  status: {_q(r['status'])}")
    return "\n".join(out) + "\n"


def _row_md(r: dict) -> str:
    ho = r["ho"]
    return (f"| {r['id']} | {r['describe']} | {_effect_words(r)} | {r['t']:.1f} | {r['p_bh']:.2g} | {r['n']} | "
            f"{_effect_words(r, 'ho') if np.isfinite(ho['d']) else '-'} | {ho['t']:.2f} | {ho['p']:.3g} | {ho['n']} | "
            f"{'**yes**' if r['replicated'] else 'no'} | {r['known']['status']} |")


def report_md(fobj: dict, disc: dict, conf: dict, novel_rep: list[dict], known_rep: list[dict], rules: list[dict]) -> str:
    s = conf["summary"]
    c = fobj["counts"]
    rows = sorted(conf["effects"], key=lambda r: (r["target"], r["p"]))
    L = []
    L.append("# Daily-panel hypothesis mining with FDR control (H-MINE2)\n")
    L.append(f"_Program {TAG} (full-speed round, agent H). Data as of {TAG}; knowledge cutoff 2026-07-01. Code: `src/ai_trader/research/daily_mining.py`; "
             "tests `tests/test_daily_mining.py`; artefacts `data/research/mining/daily_{family,discovery,discoveries_frozen,confirmation}.json`. "
             "Deterministic research only; the LLM decision layer is not tested here (principle 1)._\n")
    L.append("## Summary\n")
    L.append(f"- **Family:** {c['total']} effects declared before any statistic ({c['by_kind'].get('vol', 0)} vol, {c['by_kind'].get('return', 0)} return; "
             f"{c['testable']} testable). Targets: vr1 = log(RV_next / log-HAR), r1 = next-day vol-scaled return, r5 = next-5-day vol-scaled return.")
    L.append(f"- **Discovery (llm_in_sample, BH-FDR 10%, 10-day-block clustered t):** {disc['n_discoveries']} discoveries "
             f"(vr1 {s['by_target']['vr1']['n']}, r1 {s['by_target']['r1']['n']}, r5 {s['by_target']['r5']['n']}); "
             f"{s['restating_known']['discoveries']} of them restate existing rules (mostly the weekday and v0-trend vol effects).")
    L.append(f"- **Confirmation (clean_holdout, once, one-sided p < 0.05):** replicated {_pct(s['overall'])}; "
             f"vol {_pct(s['by_kind']['vol'])}, return {_pct(s['by_kind']['return'])}. Sign agreement {_pct(s['sign_agree'])}. "
             f"Untestable on the holdout (too few rows / clusters): {s['untestable']}.")
    L.append(f"- **H-MINE2 verdict (pooled): {conf['verdict']}.** Registry criterion: rate >= 50% and vol rate > return rate.")
    L.append(f"- **Novel replicated effects:** {len(novel_rep)}; replicated restatements of known rules: {len(known_rep)}.")
    nonk = [r for r in conf["effects"] if r["known"]["status"] != "known"]
    ret = s["by_kind"]["return"]
    L.append(f"- **Read with care:** the 'supports' rests on restated vol rules ({s['by_known']['known']['replicated']}/{s['by_known']['known']['n']} replicate); "
             f"discoveries that are not restatements replicate {sum(r['replicated'] for r in nonk)}/{len(nonk)}. "
             f"Return targets: {ret['replicated']}/{ret['n']} replicated against {ALPHA_CONF * ret['n']:.2f} expected by chance at one-sided 5%, so there is no "
             "evidence of a replicating return effect. Both novel replications lose significance with date-clustered SEs (secondary), and one rests on 9 holdout dates.\n")
    L.append("## Protocol\n")
    L.append("- Panel: universe-L complete daily rows (H-NN1 sample, 52 point-in-time features), decision at 00:00 UTC of D+1. vr1 uses H-NN1's walk-forward per-asset log-HAR "
             "forecast (monthly purged refits, from 2022-04). r1 / r5 are winsorised at +-5 sigma; r5 needs 5 consecutive days and its own segment rule (in-sample iff D + 6 <= cutoff).")
    L.append("- States (edges fitted on llm_in_sample rows only, frozen and re-used on the holdout): F1 terciles of the top-10 H-NN3 TreeSHAP features + top-bottom contrast; "
             "F2 quintiles of log_sigma, lrv_w, lrv_q + Q5-Q1; F3 the 10 tercile pairs of the top-5; F4 weekday; F5 v0 cell; F6 funding-z tercile + contrast; "
             "F7 deep-drawdown flag (dd_90d_z bottom fifth) + contrast; F8 BTC v0 cell (alts).")
    L.append("- Effect = cell mean minus the state's population mean (or top minus bottom). SE from the influence function, clustered by 10-day blocks, t with df = blocks touched - 1; "
             "date-clustered SE shown as a secondary check. BH at q = 0.10 over the whole family; untestable effects (< 100 rows or < 30 dates) keep p = 1.")
    L.append("- The discovery list and edges were frozen (diary 'H-MINE2 discovery list frozen', sha256 "
             f"`{conf['frozen_sha256'][:16]}...`) before any holdout statistic; `target_rows(clean_holdout)` raises without the frozen file.\n")
    L.append("### Family by state family and target\n")
    L.append("| family | effects | | target | effects |")
    L.append("|---|---|---|---|---|")
    fams, tg = list(c["by_family"].items()), list(c["by_target"].items())
    for i in range(max(len(fams), len(tg))):
        a = fams[i] if i < len(fams) else ("", "")
        b = tg[i] if i < len(tg) else ("", "")
        L.append(f"| {a[0]} | {a[1]} | | {b[0]} | {b[1]} |")
    L.append("")
    L.append("## Replication\n")
    L.append("| split | discoveries | replicated (Wilson 95%) |")
    L.append("|---|---|---|")
    L.append(f"| overall | {s['overall']['n']} | {_pct(s['overall'])} |")
    for k, v in s["by_kind"].items():
        L.append(f"| kind {k} | {v['n']} | {_pct(v)} |")
    for k, v in s["by_target"].items():
        L.append(f"| target {k} | {v['n']} | {_pct(v)} |")
    for k, v in s["by_family"].items():
        L.append(f"| family {k} | {v['n']} | {_pct(v)} |")
    for k, v in s["by_known"].items():
        L.append(f"| tagged {k} | {v['n']} | {_pct(v)} |")
    L.append(f"| overall, date-clustered SE (secondary) | {s['date_clustered_overall']['n']} | {_pct(s['date_clustered_overall'])} |")
    L.append("")
    L.append("## Discoveries and their holdout test\n")
    L.append("Effect: vr1 as a ratio to the average log-HAR residual (exp(d) - 1); r1 / r5 in sigma units. t = 10-day-block clustered. Holdout p is one-sided in the discovered sign.\n")
    L.append("| id | state | effect in | t in | p_bh | rows in | effect ho | t ho | p ho | rows ho | replicated | tag |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(_row_md(r))
    L.append("")
    L.append("## Replicated effects in plain words\n")
    L.append("### Novel (or contradicting an existing rule)\n")
    if novel_rep:
        for r in sorted(novel_rep, key=lambda r: r["ho"]["p"]):
            tag = r["known"]["status"] + (f" ({', '.join(r['known']['rules'])})" if r["known"]["rules"] else "")
            L.append(f"- {plain(r)}. {detail(r)}; tag {tag}.")
    else:
        L.append("- none")
    L.append("\n### Already known (restate an existing rule), grouped by rule\n")
    if known_rep:
        groups: dict[str, list[dict]] = {}
        for r in known_rep:
            groups.setdefault(", ".join(r["known"]["rules"][:2]), []).append(r)
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            ex = sorted(rs, key=lambda r: r["ho"]["p"])[:3]
            L.append(f"- **{g}** ({len(rs)} replicated): " + "; ".join(plain(r) for r in ex) + ".")
    else:
        L.append("- none")
    nk = [r for r in conf["effects"] if r["known"]["status"] == "known" and not r["replicated"]]
    if nk:
        L.append(f"\nKnown-rule discoveries that did **not** replicate ({len(nk)}): " + "; ".join(
            f"{r['describe']} ({_effect_words(r)} in, {_effect_words(r, 'ho')} ho, p {r['ho']['p']:.2f})" for r in sorted(nk, key=lambda r: r["ho"]["p"]))
            + ".")
    L.append("\n### Overlap with agent C's cells (discrete_states_2026-11-12)\n")
    ov = [r for r in conf["effects"] if r["known"].get("overlap_c")]
    L.append(f"{len(ov)} of {len(conf['effects'])} discoveries use a state agent C's LAT1 lattice also uses (weekday / calendar, v0 trend, own vol level); "
             f"{sum(r['replicated'] for r in ov)} of them replicated. Discoveries outside that lattice: {len(conf['effects']) - len(ov)}, "
             f"replicated {sum(r['replicated'] for r in conf['effects'] if not r['known'].get('overlap_c'))}.\n")
    L.append("## Caveats\n")
    L.append("- The holdout is about 133 dates (2026-07-01 to 2026-11-10), i.e. 13-14 ten-day blocks; one-sided tests with df <= 13 have little power for small effects, so a non-replication is weak evidence of absence.")
    L.append("- Effects within a state are strongly dependent (cells of one partition, the same feature in F1 / F2 / F3, three vol-level copies), so the effective number of independent discoveries is much smaller than the count. BH is valid under positive dependence; the replication rate counts dependent effects separately.")
    L.append("- The top-10 features come from H-NN3's in-sample TreeSHAP ranking, computed on in-sample walk-forward rows; the choice of states is therefore informed by in-sample data (mild look-ahead within discovery, none in the holdout).")
    L.append("- Effects are relative to each segment's own population mean, so a level shift between segments (e.g. a lower average log-HAR residual) does not count; the vr1 effect is geometric (mean log ratio).")
    L.append("- Known-rule tags are a mechanical mapping (KNOWN / known_tag in the code), fixed before discovery; they can over- or under-state novelty. "
             "One correction was made after freezing and before any holdout statistic (diary 'H-MINE2 known-rule tags reviewed'): the three day-D quarticity-ratio "
             "vol effects are the HAR-Q correction (R-VOL-23), so they are tagged known; the mechanical tag is kept as known_mechanical in the JSON.")
    L.append("- vr1 uses H-NN1's walk-forward log-HAR forecast, so the HAR-Q and weekday effects are relative to plain log-HAR; against log-HAR + weekday dummies "
             "(H-NN1 exploratory) the calendar cells would mostly vanish by construction.")
    L.append("- The BTC-whipsaw cell (R-MINE-20) is a date-level state: its holdout evidence is 9 dates in 6 blocks, and alts' vol then is one correlated bet (R-VOL-25).")
    L.append("- Rule candidates below are mined, replicated once on a short holdout, and graded C; they must be registered as forward hypotheses before promotion.\n")
    L.append("## Rule candidates\n")
    L.append("Novel replicated effects only (known-rule restatements are excluded).\n")
    L.append("```yaml")
    L.append(rules_yaml(rules).rstrip("\n"))
    L.append("```\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["declare", "discover", "confirm", "report"])
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    if a.step == "declare":
        o = run_declare()
        print(json.dumps(o["counts"], indent=1))
    elif a.step == "discover":
        d = run_discover()
        print("discoveries", d["n_discoveries"], "threshold p", d["bh_threshold_p"])
    elif a.step == "confirm":
        o = run_confirm(record=not a.no_record)
        print(json.dumps(o["summary"], indent=1, default=float), o["verdict"])
    else:
        run_report()
        print(REPORT_MD)


if __name__ == "__main__":
    main()


__all__ = ["Dim", "Effect", "FrozenMissing", "State", "assign_bins", "bh", "build_frame", "effect_family", "effect_stat", "fit_all_edges",
           "fit_edges", "require_frozen", "run_confirm", "run_declare", "run_discover", "states", "target_rows"]
