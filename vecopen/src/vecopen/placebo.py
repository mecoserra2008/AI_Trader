"""Phase 3: placebo levels (SPEC section 3), streaming only (standing rule 2).

``PlaceboGenerator.update(bar, treated_new) -> list[PlaceboSpec]`` is fed every bar after the treated
``TouchDetector`` has processed it, together with the treated levels born at that bar. Each ``PlaceboSpec`` carries a
``Leg`` to pass to ``TouchDetector.register`` of the placebo registry of its family: a plain ``LevelRegistry`` with a
family tag and no detectors, so the zone, the state machine, the fills, polarity, expiry and confluence are the same
code as for treated levels. For a placebo ``Leg``, ``origin_o`` is the placebo price, ``direction`` its side g at
birth (``events.side_at_birth``), ``confirm_close`` the price at birth, ``atr`` its frozen ATR and
``leg_size_atr = |confirm_close - price| / atr``; the other origin fields describe the source candle.

Families (the choices where SPEC is silent are marked D in ``reports/phase_3.md``)
---------------------------------------------------------------------------------
1. ``p1_nonorigin``: opens of candles that are not leg origins, on the same timeframe. A non-origin trade bar c
   with ``ATR_{c-1} > 0`` becomes a placebo level when price departs from its open as a treated level's does: at the
   close of the first tau-bar ``c+k``, ``k in [0, n_confirm]``, with ``|C_{c+k} - O_c| >= theta_L ATR_{c-1}`` and no
   bar in ``c+1 .. c+k`` trading back to the open (the leg rule's strict test, in the departure direction). The
   candle direction, the wick test and the lockout are not applied: those are what make an origin. "Non-origin" is
   checked at the departure bar against every treated origin confirmed so far, which covers every origin in the
   departure direction (it confirms at that same bar). An origin in the opposite direction could in principle confirm
   later (only through a gap); such placebos are counted as ``late_origin`` and excluded from matching. This is the
   candidate pool; each treated first fill is then matched to one pool first fill (``match_events``).
2. ``p2_displaced``: for every treated level, two placebos at ``price + upsilon_1 A`` and ``price - upsilon_2 A``
   (variants ``up`` / ``down``), with independent ``upsilon ~ U[stats.placebo_displacement_atr]`` = U[1, 3] from the
   seeded generator (``rng_for``), drawn in stream order, up before down. Both signs are always used, so the
   displacement is sign-balanced by construction and needs no sign draw. Born with the treated level (same instant,
   timeframe, ATR and zone width).
3. ``p3_candle``: for every treated level, its origin candle's close, midpoint ``(H + L) / 2`` and extreme
   (variants ``close`` / ``mid`` / ``extreme``). The extreme is the far end of the candle in the leg direction (the low
   of a down-leg candle, the high of an up-leg candle): the open-side extreme lies within ``omega_max`` of the range
   from the open and equals it whenever omega = 0, so it could not be a distinct price. Born with the treated level.

Matching (family 1, offline: it takes event tables, not bars, and uses only covariates known at t0)
--------------------------------------------------------------------------------------------------
``match_events`` pairs each treated first fill with its nearest family-1 first fill, 1:1 with replacement, exactly
within the calendar year of t0 and by Euclidean distance on the covariates ``log(1 + age)``, ``departure_atr`` and
``regime``, each scaled by its standard deviation over the treated and pool events of the cell. A cell is one
(asset, timeframe) for 15m to 4h. For 1d, 1w and 1M (``CROSS_ASSET_TFS``) a cell is one timeframe across all assets
of the run: one asset has only about 150, 20 and 7 family-1 levels on these timeframes in five years (BTCUSDT), too
few to match on, and all three covariates are scale-free (log tau-bars, ATR units, a log ATR ratio). ``smd`` gives
standardised mean differences ``(mean_T - mean_C) / sqrt((var_T + var_C) / 2)`` on the matched pairs.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vecopen.config import LoadedConfig
from vecopen.events import LevelInfo, TfVol, side_at_birth
from vecopen.legs import DEFAULT_ATR_METHOD, Bar, Leg, config_data

P1, P2, P3 = "p1_nonorigin", "p2_displaced", "p3_candle"
FAMILIES = (P1, P2, P3)
MATCH_COVARIATES = ("age_log", "departure_atr", "regime")
CROSS_ASSET_TFS = ("1d", "1w", "1M")     # matched across assets (per-asset pools are too small); module docstring


def rng_for(asset: str, config: LoadedConfig | Mapping[str, Any] | None = None) -> np.random.Generator:
    """The seeded generator of one asset: ``default_rng([config seed, crc32(asset)])``, independent of asset order."""
    return np.random.default_rng([int(config_data(config)["seed"]), zlib.crc32(asset.encode("utf-8"))])


@dataclass(frozen=True, slots=True)
class PlaceboSpec:
    family: str
    leg: Leg
    source_level_id: int | None = None
    variant: str | None = None
    displacement_atr: float | None = None


class _Cand:
    __slots__ = ("idx", "o", "h", "l", "c", "ts_open", "ts_close", "atr", "thr", "alive")

    def __init__(self, idx: int, bar: Bar, atr: float, thr: float) -> None:
        self.idx, self.atr, self.thr = idx, atr, thr
        self.o, self.h, self.l, self.c = bar.o, bar.h, bar.l, bar.c
        self.ts_open, self.ts_close = bar.ts_open, bar.ts_close
        self.alive = {-1: True, 1: True}


class _TfState:
    __slots__ = ("vol", "pending", "origins", "p1_born")

    def __init__(self, vol: TfVol) -> None:
        self.vol = vol
        self.pending: list[_Cand] = []
        self.origins: set[int] = set()        # origin indices of treated legs confirmed so far (recent)
        self.p1_born: dict[int, int] = {}     # candle index -> birth trade index, recent family-1 placebos


class PlaceboGenerator:
    """Streaming generator of the three placebo families (module docstring)."""

    def __init__(self, config: LoadedConfig | Mapping[str, Any] | None, timeframes: Sequence[str], *,
                 rng: np.random.Generator) -> None:
        cfg = config_data(config)
        legs = cfg["legs"]
        self.theta = float(legs["theta_L_atr"])
        self.n = int(legs["n_confirm"])
        lo, hi = (float(x) for x in cfg["stats"]["placebo_displacement_atr"])
        if not 0 < lo <= hi:
            raise ValueError(f"invalid stats.placebo_displacement_atr {cfg['stats']['placebo_displacement_atr']}")
        self.u_lo, self.u_hi = lo, hi
        self.rng = rng
        method = str(legs.get("atr_method", DEFAULT_ATR_METHOD))
        a_max = int(cfg["levels"]["a_max_bars"])
        self.state = {tf: _TfState(TfVol(int(legs["atr_len"]), method, a_max)) for tf in timeframes}
        self.late_origin: list[tuple[str, int]] = []      # (tf, candle index) of family-1 placebos at a later origin

    def update(self, bar: Bar, treated_new: Sequence[LevelInfo]) -> list[PlaceboSpec]:
        st = self.state.get(bar.tf)
        if st is None:
            if treated_new:
                raise ValueError(f"treated births on a {bar.tf} bar, which has no placebo state")
            return []
        for info in treated_new:
            if info.tf != bar.tf or info.birth_ts != bar.ts_close or info.leg is None:
                raise ValueError("treated_new must be the treated levels born at this bar")
            idx = info.leg.origin_index
            st.origins.add(idx)
            if idx in st.p1_born:
                self.late_origin.append((bar.tf, idx))
        out: list[PlaceboSpec] = []
        i = st.vol.update(bar)
        if i is not None:
            out += self._p1(st, bar, i)
            lim = i - self.n - 1
            st.origins = {x for x in st.origins if x >= lim} if len(st.origins) > 4 * (self.n + 2) else st.origins
            for k in [k for k in st.p1_born if k < lim]:
                del st.p1_born[k]
        for info in treated_new:
            out += self._p2(info)
            out += self._p3(info)
        return out

    # -- family 1
    def _p1(self, st: _TfState, bar: Bar, i: int) -> list[PlaceboSpec]:
        out: list[PlaceboSpec] = []
        keep: list[_Cand] = []
        for p in st.pending:
            born = 0
            for s in (-1, 1):
                if not p.alive[s]:
                    continue
                x = bar.h if s < 0 else bar.l
                if s * (x - p.o) <= 0:              # traded back to the open: dead in direction s
                    p.alive[s] = False
                    continue
                if s * (bar.c - p.o) >= p.thr:
                    born = s
                    break
            if born:
                if p.idx not in st.origins:
                    out.append(self._p1_spec(st, p, born, bar, i))
            elif i - p.idx < self.n and (p.alive[-1] or p.alive[1]):
                keep.append(p)
        st.pending = keep
        atr_prev, _ = st.vol.at(i)
        if atr_prev is not None and atr_prev > 0:
            p = _Cand(i, bar, atr_prev, self.theta * atr_prev)
            s = (bar.c > bar.o) - (bar.c < bar.o)
            if s and s * (bar.c - bar.o) >= p.thr:   # k = 0
                if i not in st.origins:
                    out.append(self._p1_spec(st, p, s, bar, i))
            elif self.n > 0:
                st.pending.append(p)
        return out

    def _p1_spec(self, st: _TfState, p: _Cand, s: int, bar: Bar, i: int) -> PlaceboSpec:
        rng_ = p.h - p.l
        omega = ((p.h - p.o) if s < 0 else (p.o - p.l)) / rng_ if rng_ > 0 else float("nan")
        st.p1_born[p.idx] = i
        leg = Leg(tf=bar.tf, direction=s, origin_index=p.idx, origin_ts_open=p.ts_open, origin_ts_close=p.ts_close,
                  origin_o=p.o, origin_h=p.h, origin_l=p.l, origin_c=p.c, omega=omega, confirm_index=i,
                  k_star=i - p.idx, birth_ts=bar.ts_close, confirm_close=bar.c, atr=p.atr,
                  leg_size_atr=s * (bar.c - p.o) / p.atr, leg_vol=None)
        return PlaceboSpec(P1, leg)

    # -- families 2 and 3
    @staticmethod
    def _at(leg: Leg, price: float) -> Leg:
        ref = leg.confirm_close
        return replace(leg, direction=side_at_birth(ref, price, leg.direction), origin_o=price,
                       leg_size_atr=abs(ref - price) / leg.atr)

    def _p2(self, info: LevelInfo) -> list[PlaceboSpec]:
        leg = info.leg
        out = []
        for sgn, name in ((1, "up"), (-1, "down")):
            u = float(self.rng.uniform(self.u_lo, self.u_hi))
            out.append(PlaceboSpec(P2, self._at(leg, leg.origin_o + sgn * u * leg.atr), info.level_id, name, sgn * u))
        return out

    def _p3(self, info: LevelInfo) -> list[PlaceboSpec]:
        leg = info.leg
        extreme = leg.origin_l if leg.direction < 0 else leg.origin_h
        prices = (("close", leg.origin_c), ("mid", (leg.origin_h + leg.origin_l) / 2), ("extreme", extreme))
        return [PlaceboSpec(P3, self._at(leg, x), info.level_id, name) for name, x in prices]


# ---------------------------------------------------------------- matching and balance (offline, covariates only)

def _cov(rows: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.asarray(rows[c], dtype=float) for c in MATCH_COVARIATES])


def match_events(treated: Mapping[str, np.ndarray], pool: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """1:1 nearest-neighbour matching with replacement (module docstring) within ONE (asset, timeframe) cell.

    ``treated`` / ``pool``: columns ``event_key``, ``year`` and the ``MATCH_COVARIATES``. Rows with a missing
    covariate are not matched. Returns ``treated_key``, ``placebo_key`` (None if no pool event in the same year),
    ``distance``."""
    from scipy.spatial import cKDTree

    xt, xp = _cov(treated), _cov(pool)
    okt, okp = np.isfinite(xt).all(axis=1), np.isfinite(xp).all(axis=1)
    both = np.vstack([xt[okt], xp[okp]])
    sd = both.std(axis=0, ddof=1) if len(both) > 1 else np.ones(xt.shape[1])
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    yt, yp = np.asarray(treated["year"]), np.asarray(pool["year"])
    kt, kp = np.asarray(treated["event_key"], dtype=object), np.asarray(pool["event_key"], dtype=object)
    out_p = np.full(len(kt), None, dtype=object)
    out_d = np.full(len(kt), np.nan)
    for y in np.unique(yt[okt]):
        ti = np.flatnonzero(okt & (yt == y))
        pi = np.flatnonzero(okp & (yp == y))
        if len(pi) == 0:
            continue
        tree = cKDTree(xp[pi] / sd)
        d, j = tree.query(xt[ti] / sd, k=1)
        out_p[ti] = kp[pi[np.atleast_1d(j)]]
        out_d[ti] = d
    return {"treated_key": kt, "placebo_key": out_p, "distance": out_d}


def smd(xt: np.ndarray, xc: np.ndarray) -> float:
    """Standardised mean difference with the pooled (average-variance) denominator; nan if undefined."""
    xt, xc = np.asarray(xt, float), np.asarray(xc, float)
    if len(xt) < 2 or len(xc) < 2:
        return float("nan")
    den = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
    return float((xt.mean() - xc.mean()) / den) if den > 0 else 0.0


@dataclass
class Moments:
    """Running n, sum and sum of squares, so balance can be pooled over assets processed one at a time."""

    n: int = 0
    s: float = 0.0
    ss: float = 0.0

    def add(self, x: np.ndarray) -> None:
        x = np.asarray(x, float)
        self.n += len(x)
        self.s += float(x.sum())
        self.ss += float((x * x).sum())

    def merge(self, n: int, s: float, ss: float) -> None:
        self.n += int(n)
        self.s += float(s)
        self.ss += float(ss)

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")

    @property
    def var(self) -> float:
        return (self.ss - self.n * self.mean ** 2) / (self.n - 1) if self.n > 1 else float("nan")


def smd_moments(t: Moments, c: Moments) -> float:
    if t.n < 2 or c.n < 2:
        return float("nan")
    den = np.sqrt(max((t.var + c.var) / 2, 0.0))
    return float((t.mean - c.mean) / den) if den > 0 else 0.0
