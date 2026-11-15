"""Hourly live line of the frozen cross-sectional network v3 (xs.v3), CONTEXT ONLY (research only; reads the lake, never
trades). Rules R-XS-2 / R-XS-3: the relative-return ranking is mostly a low-volatility ranking and its top-minus-bottom
spread loses after costs, so nothing here is a signal; the decision sessions (agent.decision) do not read it. R-XS-1:
the 1h vol head beats panel log-HAR-t x hour-of-week, so its 1h bands are shown next to v1's.

    run from scripts/live_nn.py (hourly.run_hour, after v1, v2.1 and players; wrapped so a v3 failure changes nothing
    of v1 / v2.1)

What one run computes (`run_line`, as_of = the run time; everything point in time, known_ts <= as_of):
  inputs   for the 23 universe-L symbols of spec.json: market_frame(sym, '1h', as_of, funding, 4h OI) on the full
           history (one lake query per table for all symbols: PrefetchLake), a placeholder bar at the forecast hour S,
           then the study's own xs_network_study.token_rows and assemble (no re-implemented features). Two speed-ups,
           both exact and tested: the expanding hour-of-week profiles (seasonality.seasonal_factor, xs_net.
           seasonal_multi) are evaluated only on the last SEASONAL_TAIL rows (same sums in the same order, same _shape),
           and assemble runs on the last TAIL_HOURS hours (every window it uses is shorter; float rounding of the
           rolling sums only). Row S uses bars <= S-1 and funding / OI known by the close of S-1. A token is absent
           (masked exactly as in training) if any core feature is missing: short history, a stale feed, a gap.
           Fewer than MIN_TOKENS present tokens: no line.
  model    data/models/xs_v3/frozen: spec.json, v3_seed{0,1,2}.pt; the weights' and the four source files' SHA-256
           are checked against spec.json every run (a mismatch stops v3). Standardisation and clipping with the
           study's block(); the three seeds' quantiles averaged; torch at 2 threads (restored afterwards).
  outputs  per token: 1h / 4h vol quantiles (|r| in %, = model units x sqrt(h) s), the panel log-HAR-t x hour-of-week
           quantiles (spec.har_baseline, the incumbent of R-XS-1), the 4h / 24h relative-return quantiles q10 / q50 / q90
           (return minus the equal-weight basket of present tokens, bp) and the 80% ACI band; ranks by the 4h and 24h
           rel median and the low-vol rank (by -lsig2) beside them; the 2D map coordinates (fixed projection,
           data/models/xs_v3/live/projection.json, fitted once on the frozen refit's fit block); attention received
           (descriptive, R-XS-4) and the attention matrix.
  ACI      per token and horizon, alpha 0.20, gamma 0.005, window 500 (the study's protocol); warm start = spec.aci
           (the last update replayed from scores_tail); updated only at the study's non-overlapping origins (4h: 00, 04,
           .. UTC; 24h: 00 UTC) once the window has closed, before the current band is issued. The 24h band is known to
           under-cover (holdout 0.749 at 80%) and is labelled so. State: data/models/xs_v3/live/aci_state.json.
  scoring  every earlier line whose window has closed, once, into scores.jsonl: 1h / 4h vol pinball v3 vs log-HAR-t
           and q90 coverage; 4h / 24h rank IC of the rel median and of the low-vol ranking, ACI / raw band hits.
Outputs: data/journal/live_xs_v3/<YYYY-MM-DD of S>.jsonl (one line per forecast hour S, id xs3-<S>), scores.jsonl
there, the ACI state, and a short block in research/live/LATEST.md (written by hourly.compose_latest).
"""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from ai_trader.live import nn

MODEL_ID = "xs.v3"
VERSION = "live_xs_v3.1"
NOT_A_SIGNAL = "context only, not a signal (R-XS-2 / R-XS-3); the decision sessions do not read it"
TAIL_HOURS = 24 * 130            # assemble window: > fund_z90's 2160 + 1 lagged values and beta's 720 bars
SEASONAL_TAIL = 48               # rows of the expanding hour-of-week profiles evaluated (row S is the last)
MIN_HISTORY = 24 * 120           # the study's build(): shorter frames are skipped
STALE_AFTER = nn.STALE_AFTER
REPLAY_AFTER = nn.REPLAY_AFTER
ACI_LEVEL = 80
VOL_SHOW = (1, 4)                # vol horizons shown / scored (24h: log-HAR-t stays the reference, R-XS-1)
REL_H = (4, 24)
TOP = 5
SCORE_GIVE_UP = timedelta(hours=72)
QNAMES = nn.QNAMES
TORCH_THREADS = 2


# ------------------------------------------------------------------ paths


@dataclass
class PathsV3:
    root: Path
    models_root: Path | None = None   # tests: frozen model + projection from the repo, everything else under root

    @classmethod
    def default(cls) -> PathsV3:
        from ai_trader.core.config import load

        return cls(load().root)

    @property
    def frozen(self) -> Path:
        return (self.models_root or self.root) / "data" / "models" / "xs_v3" / "frozen"

    @property
    def projection(self) -> Path:
        return (self.models_root or self.root) / "data" / "models" / "xs_v3" / "live" / "projection.json"

    @property
    def code_root(self) -> Path:
        return self.models_root or self.root

    @property
    def journal(self) -> Path:
        return self.root / "data" / "journal" / "live_xs_v3"

    @property
    def state(self) -> Path:
        return self.root / "data" / "models" / "xs_v3" / "live" / "aci_state.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _us(t: datetime) -> int:
    return round(t.timestamp() * 1_000_000)


def _from_us(x: int) -> datetime:
    return datetime.fromtimestamp(int(x) / 1_000_000, tz=UTC)


# ------------------------------------------------------------------ frozen model


@dataclass
class Frozen:
    spec: dict
    spec_sha: str
    models: list
    weights_sha: dict
    code_ok: dict

    @property
    def symbols(self) -> list[str]:
        return list(self.spec["symbols"])

    @property
    def sec(self) -> np.ndarray:
        return np.asarray(self.spec["sector_index"], dtype=np.int64)

    def std(self) -> dict:
        s = self.spec["standardisation"]
        return {"mu": np.asarray(s["mu"], float), "sd": np.asarray(s["sd"], float), "mmu": np.asarray(s["mkt_mu"], float),
                "msd": np.asarray(s["mkt_sd"], float)}

    def har(self) -> dict:
        return {int(k): v for k, v in self.spec["har_baseline"].items()}

    def card(self) -> dict:
        return {"model_id": MODEL_ID, "version": self.spec["version"], "spec_sha256": self.spec_sha[:16],
                "weights_sha256": {k: v[:16] for k, v in self.weights_sha.items()}, "code_sha256_ok": self.code_ok,
                "git_commit": self.spec.get("git_commit"), "refit_quarter": self.spec.get("refit_quarter"),
                "fit_range": [_from_us(x).isoformat() for x in self.spec["fit_range_us"]],
                "trained_through": trained_through(self.spec).isoformat(), "as_of_data": self.spec.get("as_of_data")}


def trained_through(spec: dict) -> datetime:
    """Labels of the refit's training rows (fit + early-stopping validation) end 1 day before its quarter start."""
    y, q = spec["refit_quarter"].split("Q")
    return datetime(int(y), 3 * (int(q) - 1) + 1, 1, tzinfo=UTC) - timedelta(days=1)


def verify(spec: dict, frozen_dir: Path, code_root: Path) -> tuple[dict, dict]:
    """SHA-256 of the six weight files and of the four source files against spec.json; raises on any mismatch."""
    w = {}
    for name, h in spec["files_sha256"].items():
        got = _sha(frozen_dir / name)
        if got != h:
            raise RuntimeError(f"xs_v3 weights hash mismatch: {name}")
        w[name] = got
    code = {}
    for f, h in spec["code_sha256"].items():
        code[f] = _sha(code_root / f) == h
        if not code[f]:
            raise RuntimeError(f"xs_v3 code hash mismatch (train/serve skew guard): {f}")
    return w, code


@lru_cache(maxsize=2)
def _load_cached(frozen_dir: str, code_root: str, spec_sha: str) -> Frozen:
    import torch

    from ai_trader.ml import xs_net as xs

    fd = Path(frozen_dir)
    spec = json.loads((fd / "spec.json").read_text(encoding="utf-8"))
    w, code = verify(spec, fd, Path(code_root))
    cfg = spec["model"]
    models = []
    for s in range(int(cfg["seeds"])):
        m = xs.make(int(cfg["n_in"]), int(cfg["n_mkt"]), int(cfg["n_sector"]), kind=cfg["kind"], mix=bool(cfg["mix"]))
        m.load_state_dict(torch.load(fd / f"v3_seed{s}.pt", weights_only=True))
        m.eval()
        models.append(m)
    return Frozen(spec, spec_sha, models, w, code)


def load_frozen(paths: PathsV3) -> Frozen:
    sp = paths.frozen / "spec.json"
    fz = _load_cached(str(paths.frozen), str(paths.code_root), _sha(sp))
    verify(fz.spec, paths.frozen, paths.code_root)   # every run: a file changed under a cached model is caught
    return fz


# ------------------------------------------------------------------ point-in-time inputs


class PrefetchLake:
    """market_frame's three reads (1h bars, funding, 4h OI) for many symbols from one query per table. A per-symbol
    read with the same as_of and filters returns exactly the rows of Lake.read_asof (same query, filtered by symbol,
    same order); anything else goes to the real lake."""

    SPECS: ClassVar[dict[str, dict[str, str]]] = {"bars": {"venue": "bybit", "price_type": "last", "freq": "1h"}, "funding": {"venue": "bybit"},
             "open_interest": {"venue": "bybit", "interval": "4h"}}

    def __init__(self, lake, as_of: datetime, symbols: list[str]):
        self.lake, self.as_of = lake, as_of
        self.t = {k: lake.read_asof(k, as_of, symbols=list(symbols), filters=f) for k, f in self.SPECS.items()}

    def bars(self, symbol: str, freq: str, as_of: datetime, *, price_type: str = "last", start=None, venue: str = "bybit") -> pl.DataFrame:
        return self.read_asof("bars", as_of, symbols=[symbol], start=start, filters={"venue": venue, "price_type": price_type, "freq": freq})

    def read_asof(self, table: str, as_of: datetime, *, symbols=None, start=None, end=None, filters=None, columns=None) -> pl.DataFrame:
        same = (table in self.t and as_of == self.as_of and start is None and end is None and columns is None and symbols is not None
                and len(symbols) == 1 and {k: str(v) for k, v in (filters or {}).items()} == {k: str(v) for k, v in self.SPECS[table].items()})
        if not same:
            return self.lake.read_asof(table, as_of, symbols=symbols, start=start, end=end, filters=filters, columns=columns)
        d = self.t[table]
        return d.filter(pl.col("symbol") == symbols[0]) if d.height else d

    def data_max(self) -> datetime | None:
        ks = [nn._utc(d["known_ts"].max()) for d in self.t.values() if d.height and "known_ts" in d.columns]
        return max(ks) if ks else None


def _tail_profile(how, u, hs, tail: int, kappa: float, bandwidth: int, cap: float, floor: float, min_weeks: int):
    """seasonal_factor (hs None) / xs_net.seasonal_multi (hs tuple) evaluated on the last `tail` rows only (NaN before).
    Exact: the per-hour-of-week sums are accumulated in the same order (np.bincount is a sequential loop) and the
    profile is the same seasonality._shape."""
    from ai_trader.features.seasonality import HOW, _kernel, _shape

    how = np.asarray(how, dtype=np.int64)
    u = np.asarray(u, dtype=float)
    n = len(u)
    fin = np.isfinite(u)
    x = np.where(fin, np.clip(np.where(fin, u, 0.0), 0.0, cap), 0.0)
    t0 = max(0, n - tail)
    sums = np.bincount(how[:t0][fin[:t0]], weights=x[:t0][fin[:t0]], minlength=HOW).astype(float)
    counts = np.bincount(how[:t0][fin[:t0]], minlength=HOW).astype(float)
    seen = int(fin[:t0].sum())
    need = min_weeks * HOW
    kern = _kernel(bandwidth)
    out = np.full(n, np.nan) if hs is None else {h: np.full(n, np.nan) for h in hs}
    for t in range(t0, n):
        if seen >= need:
            cache = _shape(sums, counts, kappa, kern, floor)
            if hs is None:
                out[t] = cache[how[t]]
            else:
                cum = np.r_[0.0, np.cumsum(np.r_[cache, cache])]
                k0 = how[t]
                for h in hs:
                    out[h][t] = (cum[k0 + h] - cum[k0]) / h
        if fin[t]:
            sums[how[t]] += x[t]
            counts[how[t]] += 1
            seen += 1
    return out


@contextmanager
def fast_seasonal(tail: int = SEASONAL_TAIL):
    """Swap the two expanding hour-of-week profiles for their tail-only evaluation while the study's token_rows runs."""
    from ai_trader.features import seasonality as sz
    from ai_trader.ml import xs_net as xs

    orig_f, orig_m = sz.seasonal_factor, xs.seasonal_multi

    def f_fast(how, u, *, window_weeks=None, kappa=20.0, bandwidth=1, cap=50.0, floor=0.1, min_weeks=8):
        if window_weeks:
            return orig_f(how, u, window_weeks=window_weeks, kappa=kappa, bandwidth=bandwidth, cap=cap, floor=floor, min_weeks=min_weeks)
        return _tail_profile(how, u, None, tail, kappa, bandwidth, cap, floor, min_weeks)

    def m_fast(how, u, hs=(4, 24), *, kappa=20.0, bandwidth=1, cap=50.0, floor=0.1, min_weeks=8):
        return _tail_profile(how, u, tuple(hs), tail, kappa, bandwidth, cap, floor, min_weeks)

    sz.seasonal_factor, xs.seasonal_multi = f_fast, m_fast
    try:
        yield
    finally:
        sz.seasonal_factor, xs.seasonal_multi = orig_f, orig_m


FRAME_COLS = ("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "ret", "sigma_risk_next", "funding_rate", "open_interest")


@contextmanager
def lean_frame():
    """While market_frame runs, skip the two per-bar recursions whose columns v3 never reads (FRAME_COLS keeps none of
    them): the plain EWMA of enrich_bars (sigma_prev / sigma_next / z) and the weekend-aware VolModel of add_robust_vol
    (sigma_rb_next / wk_m). They come back as NaN; sigma_risk_next (use_weekend=False) is computed by the original run.
    Every FRAME_COLS column is bitwise unchanged (tested)."""
    from ai_trader.features import vol_models as vm
    from ai_trader.research import frame as fr

    orig_e, orig_r = fr.ewma_sigma_ahead, vm.run

    def e_skip(r, *a, **k):
        return np.full(len(r), np.nan)

    def r_lean(r, weekend, cfg=None):
        cfg = vm.VolModel() if cfg is None else cfg
        if cfg.use_weekend:
            n = len(r)
            return {"sigma_prev": np.full(n, np.nan), "v_next": np.full(n, np.nan), "m": np.full(n, np.nan)}
        return orig_r(r, weekend, cfg)

    fr.ewma_sigma_ahead, vm.run = e_skip, r_lean
    try:
        yield
    finally:
        fr.ewma_sigma_ahead, vm.run = orig_e, orig_r


def token_frames(as_of: datetime, symbols: list[str], lake, lean: bool = True) -> tuple[dict[str, pl.DataFrame], dict[str, str], PrefetchLake]:
    """market_frame per symbol (the study's _frame columns), point in time; absent reasons for the others."""
    from ai_trader.research.frame import market_frame

    pf = PrefetchLake(lake, as_of, symbols)
    frames, absent = {}, {}
    for s in symbols:
        if lean:
            with lean_frame():
                f = market_frame(s, "1h", as_of, funding=True, oi=True, oi_interval="4h", lake=pf)
        else:
            f = market_frame(s, "1h", as_of, funding=True, oi=True, oi_interval="4h", lake=pf)
        if f.height == 0:
            absent[s] = "no data"
            continue
        if f.height < MIN_HISTORY:
            absent[s] = f"short history ({f.height} h < {MIN_HISTORY})"
            continue
        frames[s] = f.select(*FRAME_COLS)
    return frames, absent, pf


def forecast_hour(frames: dict[str, pl.DataFrame]) -> tuple[datetime, datetime]:
    """S = the hour that opens at the newest bar close (max over tokens); returns (S, newest close)."""
    last = max(nn._utc(f["close_ts"].max()) for f in frames.values())
    return last.replace(minute=0, second=0, microsecond=0), last


def build_rows(frames: dict[str, pl.DataFrame], S: datetime, tail_hours: int = TAIL_HOURS, seasonal_tail: int = SEASONAL_TAIL) -> dict[str, pl.DataFrame]:
    """The study's token_rows per symbol on its full history plus a placeholder bar at S (as nn.hourly_frame), kept for
    hours >= S - tail_hours."""
    from ai_trader.research import xs_network_study as study

    out = {}
    lo = S - timedelta(hours=tail_hours)
    for s, f in frames.items():
        dt = f.schema["event_ts"]
        if nn._utc(f["event_ts"].max()) >= S:
            g = f.filter(pl.col("event_ts") < pl.lit(S).cast(dt))   # never happens at as_of (bar S not closed); replays only
        else:
            g = f
        ph = pl.DataFrame({"event_ts": [S]}).with_columns(pl.col("event_ts").cast(dt))
        g = pl.concat([g, ph], how="diagonal_relaxed")
        if seasonal_tail:
            with fast_seasonal(seasonal_tail):
                r = study.token_rows(g)
        else:
            r = study.token_rows(g)
        out[s] = r.filter(pl.col("event_ts") >= pl.lit(lo).cast(r.schema["event_ts"]))
    return out


def panel_row(frames: dict[str, pl.DataFrame], symbols: list[str], S: datetime, **kw) -> tuple[dict, int]:
    """assemble() on the tail; returns the panel dict and the index of hour S."""
    from ai_trader.research import xs_network_study as study

    rows = build_rows(frames, S, **kw)
    P = study.assemble(rows, symbols)
    hit = np.flatnonzero(P["ts"] == _us(S))
    if not len(hit):
        raise RuntimeError(f"forecast row {S.isoformat()} missing from the panel")
    return P, int(hit[-1])


# ------------------------------------------------------------------ ACI


def new_aci_state(spec: dict, as_of: datetime) -> dict:
    """Warm start from spec.aci: scores_tail as the window; alpha_last is the alpha used at the stream's last origin, so
    that origin's update is replayed (its Q from the 499 earlier scores)."""
    a = spec["aci"]
    gamma, window, alpha = float(a["gamma"]), int(a["window"]), 1 - float(a["level"])
    st = {"created": datetime.now(UTC).isoformat(), "as_of": as_of.isoformat(), "spec_version": spec["version"]}
    for h in REL_H:
        toks = {}
        for sym, d in a["state"][f"{h}h"].items():
            sc = [float(x) for x in d["scores_tail"]]
            o = nn.OnlineACI(ACI_LEVEL, gamma, window, float(d["alpha_last"]), sc[:-1])
            Q = o.Q()
            err = 1.0 if not math.isfinite(Q) else float(sc[-1] > Q)
            o = nn.OnlineACI(ACI_LEVEL, gamma, window, float(d["alpha_last"]) + gamma * (alpha - err), sc[-window:])
            toks[sym] = o
        st[f"{h}h"] = {"applied_until": None, "tokens": toks}
    return st


def dump_state(st: dict) -> dict:
    out = {k: v for k, v in st.items() if not k.endswith("h")}
    for h in REL_H:
        k = f"{h}h"
        out[k] = {"applied_until": st[k]["applied_until"], "tokens": {s: o.to_dict() for s, o in st[k]["tokens"].items()}}
    return out


def load_state(paths: PathsV3) -> dict | None:
    if not paths.state.is_file():
        return None
    d = json.loads(paths.state.read_text(encoding="utf-8"))
    for h in REL_H:
        k = f"{h}h"
        d[k] = {"applied_until": d[k]["applied_until"], "tokens": {s: nn.OnlineACI.from_dict(o) for s, o in d[k]["tokens"].items()}}
    return d


def save_state(paths: PathsV3, st: dict) -> None:
    paths.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.state.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dump_state(st)), encoding="utf-8")
    tmp.replace(paths.state)


def is_origin(S: datetime, h: int) -> bool:
    return (S.hour % 4 == 0) if h == 4 else (S.hour == 0)


# ------------------------------------------------------------------ maths (pure, tested)


def ranks_desc(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rank 1 = largest value among masked finite entries (ties: average rank); NaN elsewhere."""
    from scipy.stats import rankdata

    out = np.full(len(v), np.nan)
    k = mask & np.isfinite(v)
    if k.any():
        out[k] = rankdata(-v[k])
    return out


def spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_n: int = 5) -> float | None:
    from ai_trader.ml import xs_net as xs

    r = xs.spearman_rows(np.asarray(a, float)[None, :], np.asarray(b, float)[None, :], np.asarray(mask, bool)[None, :], min_n=min_n)[0]
    return float(r) if np.isfinite(r) else None


def relative_realised(R: np.ndarray, member: np.ndarray, min_n: int = 5) -> np.ndarray:
    from ai_trader.ml import xs_net as xs

    return xs.relative(np.asarray(R, float)[None, :], np.asarray(member, bool)[None, :], min_n=min_n)[0]


def pinball7(y: float, q: list[float]) -> float:
    from ai_trader.ml import xs_net as xs

    t = np.asarray(xs.TAUS)
    e = y - np.asarray(q, float)
    return float(np.maximum(t * e, (t - 1) * e).mean())


def attention_summary(w: np.ndarray, present: np.ndarray, symbols: list[str]) -> dict:
    """w (N, N+1): query i -> key j (last key = the market token), averaged over heads and seeds. Received by key j =
    mean over the other present queries of w[i, j] x the number of keys (1 = uniform), as in the study."""
    N = len(symbols)
    idx = np.flatnonzero(present)
    nk = len(idx) + 1
    keys = [*idx.tolist(), N]
    rec = {}
    for j in keys:
        q = [i for i in idx if i != j]
        rec["MARKET" if j == N else symbols[j]] = float(np.mean(w[q, j]) * nk) if q else None
    hubs = sorted(((k, v) for k, v in rec.items() if v is not None), key=lambda kv: -kv[1])
    return {"keys": ["MARKET" if j == N else symbols[j] for j in keys], "queries": [symbols[i] for i in idx],
            "w": [[float(f"{w[i, j]:.4f}") for j in keys] for i in idx], "received": rec, "hubs": hubs[:TOP],
            "market_share": float(np.mean(w[idx, N])) if len(idx) else None,
            "self_share": float(np.mean(w[idx, idx])) if len(idx) else None, "n_keys": nk}


# ------------------------------------------------------------------ projection (fitted once, on training data)


def fit_projection(paths: PathsV3 | None = None, chunk: int = 2000) -> dict:
    """The fixed 2D map: per-token mean head input [z_i, a_i] of each frozen seed over the refit's fit block
    (spec.fit_range_us, valid hours of the study's cached panel), the three seeds concatenated (3 x 16 dims, no
    cross-seed alignment needed), centred over tokens, top-2 principal axes (sign: BTC positive on both). Written once
    to data/models/xs_v3/live/projection.json; live rows are projected onto it, never refitted."""
    from ai_trader.research import xs_network_study as study

    paths = paths or PathsV3.default()
    fz = load_frozen(paths)
    P = study.load_panel()
    if list(P["symbols"]) != fz.symbols or list(P["features"]) != list(fz.spec["features"]):
        raise RuntimeError("cached panel does not match spec.json symbols / features")
    lo, hi = fz.spec["fit_range_us"]
    fit = np.flatnonzero(P["valid"] & (P["ts"] >= lo) & (P["ts"] <= hi))
    st = fz.std()
    N, D2 = len(fz.symbols), 2 * int(fz.spec["model"]["d"])
    acc = np.zeros((len(fz.models), N, D2))
    cnt = np.zeros(N)
    for a in range(0, len(fit), chunk):
        idx = fit[a:a + chunk]
        B = study.block(P, idx, st)
        n_j = B.P.sum(0)
        for s, m in enumerate(fz.models):
            r = study.representation(m, B, fz.sec)
            acc[s] += np.where(n_j[:, None] > 0, r * n_j[:, None], 0.0)
        cnt += n_j
    rep = np.concatenate([acc[s] / cnt[:, None] for s in range(len(fz.models))], axis=1)   # (N, 48)
    ok = np.all(np.isfinite(rep), axis=1)
    center = rep[ok].mean(0)
    R = rep[ok] - center
    _, Sv, Vt = np.linalg.svd(R, full_matrices=False)
    comps = Vt[:2].copy()
    btc = fz.symbols.index("BTCUSDT")
    for k in range(2):
        if (rep[btc] - center) @ comps[k] < 0:
            comps[k] *= -1
    xy = (rep - center) @ comps.T
    out = {"method": "PCA of the concatenated seed representations [z_i, a_i] (3 x 16), per-token means over the frozen refit's fit block",
           "spec_version": fz.spec["version"], "spec_sha256": fz.spec_sha, "fit_range_us": [int(lo), int(hi)], "n_hours": len(fit),
           "symbols": fz.symbols, "center": center.tolist(), "components": comps.tolist(),
           "explained": (Sv[:2] ** 2 / (Sv ** 2).sum()).tolist(),
           "anchors": {s: (xy[j].tolist() if ok[j] else None) for j, s in enumerate(fz.symbols)},
           "created": datetime.now(UTC).isoformat()}
    paths.projection.parent.mkdir(parents=True, exist_ok=True)
    paths.projection.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def load_projection(paths: PathsV3, spec_sha: str) -> tuple[dict | None, str | None]:
    if not paths.projection.is_file():
        return None, None
    pj = json.loads(paths.projection.read_text(encoding="utf-8"))
    if pj.get("spec_sha256") != spec_sha:
        return None, None
    return pj, _sha(paths.projection)


# ------------------------------------------------------------------ journal


def read_lines(paths: PathsV3, since: datetime | None = None) -> list[dict]:
    """Journal lines (all, or those in files dated >= since's date)."""
    if not paths.journal.is_dir():
        return []
    rows = []
    for f in sorted(paths.journal.glob("20*.jsonl")):
        if since is not None and f.stem < f"{since:%Y-%m-%d}":
            continue
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted(rows, key=lambda r: (r["forecast_hour"], r["logged_at"]))


def read_scores(paths: PathsV3) -> list[dict]:
    p = paths.journal / "scores.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def closes_by_ts(frames: dict[str, pl.DataFrame], since: datetime) -> dict[str, dict[int, float]]:
    out = {}
    for s, f in frames.items():
        g = f.filter(pl.col("close_ts") >= pl.lit(since).cast(f.schema["close_ts"])).select("close_ts", "close")
        ts = g["close_ts"].dt.replace_time_zone(None).cast(pl.Datetime("us")).cast(pl.Int64).to_numpy()
        out[s] = dict(zip(ts.tolist(), g["close"].cast(pl.Float64).to_list(), strict=True))
    return out


def score_line(line: dict, h: int, closes: dict[str, dict[int, float]]) -> dict | None:
    """Realised outcome of one earlier line at horizon h (hours) from lake closes; None if no token has its close."""
    S = nn._utc(line["forecast_hour"])
    end = _us(S + timedelta(hours=h))
    toks = [t for t in line["tokens"] if t["present"]]
    syms = [t["symbol"] for t in toks]
    R = np.array([math.log(closes[t["symbol"]][end] / t["close_ref"]) if t.get("close_ref") and end in closes.get(t["symbol"], {}) else np.nan
                  for t in toks])
    if not np.isfinite(R).any():
        return None
    s = np.array([t["s_pct"] / 100 for t in toks])
    out = {"id": line["id"], "horizon": f"{h}h", "forecast_hour": line["forecast_hour"], "window_end": (S + timedelta(hours=h)).isoformat(),
           "origin": is_origin(S, h) if h in REL_H else True, "n": int(np.isfinite(R).sum())}
    if h in VOL_SHOW:
        k = f"{h}h"
        a = np.abs(R) / (math.sqrt(h) * s)
        pv, ph, cv, chh = [], [], [], []
        for j, t in enumerate(toks):
            if not np.isfinite(a[j]):
                continue
            sc = 100 * math.sqrt(h) * s[j]
            qv = [x / sc for x in t["vol"][k]["v3"]]
            qh = [x / sc for x in t["vol"][k]["har"]]
            pv.append(pinball7(a[j], qv))
            ph.append(pinball7(a[j], qh))
            cv.append(float(a[j] <= qv[5]))
            chh.append(float(a[j] <= qh[5]))
        out["vol"] = {"pinball_v3": float(np.mean(pv)), "pinball_har": float(np.mean(ph)), "cov_q90_v3": float(np.mean(cv)),
                      "cov_q90_har": float(np.mean(chh)), "n": len(pv)}
    if h in REL_H:
        k = f"{h}h"
        fin = np.isfinite(R)
        rel = relative_realised(R, fin)
        if np.isfinite(rel).sum() >= 5:
            med = np.array([t["rel"][k]["q_bp"][1] for t in toks], float)
            lowv = -np.array([t["lsig2"] for t in toks], float)
            m = np.isfinite(rel)
            rel_bp = 1e4 * rel
            hit = [bool(t["rel"][k]["aci_bp"][0] <= rel_bp[j] <= t["rel"][k]["aci_bp"][1]) for j, t in enumerate(toks) if m[j]]
            hit_raw = [bool(t["rel"][k]["q_bp"][0] <= rel_bp[j] <= t["rel"][k]["q_bp"][2]) for j, t in enumerate(toks) if m[j]]
            out["rel"] = {"ic_v3": spearman(med, rel, m), "ic_lowvol": spearman(lowv, rel, m), "hit_aci80": float(np.mean(hit)),
                          "hit_raw80": float(np.mean(hit_raw)), "n": int(m.sum()),
                          "realised_bp": {syms[j]: float(f"{rel_bp[j]:.2f}") for j in range(len(toks)) if m[j]}}
    return out


def aci_update(st: dict, line: dict, h: int, closes: dict[str, dict[int, float]]) -> int:
    """Resolve the issued band of an origin line once its window closed (tokens present then, with a close now)."""
    S = nn._utc(line["forecast_hour"])
    k = f"{h}h"
    au = st[k]["applied_until"]
    if not is_origin(S, h) or (au is not None and nn._utc(au) >= S):
        return 0
    end = _us(S + timedelta(hours=h))
    toks = [t for t in line["tokens"] if t["present"]]
    R = np.array([math.log(closes[t["symbol"]][end] / t["close_ref"]) if t.get("close_ref") and end in closes.get(t["symbol"], {}) else np.nan
                  for t in toks])
    rel = relative_realised(R, np.isfinite(R))
    n = 0
    for j, t in enumerate(toks):
        o = st[k]["tokens"].get(t["symbol"])
        if o is None or not np.isfinite(rel[j]):
            continue
        r = t["rel"][k]
        y = rel[j] / (r["scale_bp"] / 1e4)
        lo, hi = r["q_std"][0], r["q_std"][2]
        blo, bhi = r["aci_std"]
        err = float(not (blo <= y <= bhi)) if r.get("Q") is not None else 1.0
        o.resolve(err, max(lo - y, y - hi), S.isoformat(), float(not (lo <= y <= hi)))
        n += 1
    st[k]["applied_until"] = S.isoformat()
    return n


def running(scores: list[dict]) -> dict:
    out = {}
    for h in (1, 4):
        v = [s["vol"] for s in scores if s["horizon"] == f"{h}h" and "vol" in s]
        if v:
            pv, ph = float(np.mean([x["pinball_v3"] for x in v])), float(np.mean([x["pinball_har"] for x in v]))
            out[f"vol_{h}h"] = {"n_lines": len(v), "pinball_v3": pv, "pinball_har": ph, "ratio": pv / ph if ph > 0 else None,
                                "cov_q90_v3": float(np.mean([x["cov_q90_v3"] for x in v]))}
    for h in REL_H:
        r = [s for s in scores if s["horizon"] == f"{h}h" and "rel" in s and s["rel"]["ic_v3"] is not None]
        if r:
            o = [s for s in r if s["origin"]]
            out[f"rel_{h}h"] = {"n_lines": len(r), "ic_v3": float(np.mean([s["rel"]["ic_v3"] for s in r])),
                                "ic_lowvol": float(np.mean([s["rel"]["ic_lowvol"] for s in r if s["rel"]["ic_lowvol"] is not None] or [np.nan])),
                                "n_origins": len(o), "ic_v3_origins": float(np.mean([s["rel"]["ic_v3"] for s in o])) if o else None,
                                "hit_aci80_origins": float(np.mean([s["rel"]["hit_aci80"] for s in o])) if o else None,
                                "hit_aci80_all": float(np.mean([s["rel"]["hit_aci80"] for s in r]))}
    return out


# ------------------------------------------------------------------ the line


def _r(x, nd: int = 6):
    return None if x is None or not np.isfinite(x) else float(f"{float(x):.{nd}g}")


def v1_compare(v1: dict | None) -> dict | None:
    if not v1 or "net" not in v1:
        return None
    n1 = v1["net"]["1h"]
    q = n1["quantiles_pct"]
    if q.get("q05") is None or q.get("q95") is None:
        return None
    return {"id": v1["id"], "forecast_hour": v1["forecast_hour"], "half90_pct": (q["q95"] - q["q05"]) / 2, "sigma_pct": n1.get("sigma_pct"),
            "aci90_half_pct": (n1["aci"]["90"]["hi_pct"] - n1["aci"]["90"]["lo_pct"]) / 2 if n1.get("aci", {}).get("90", {}).get("lo_pct") is not None else None}


def run_line(as_of: datetime, *, paths: PathsV3 | None = None, lake=None, state: dict | None = None, persist_state: bool = True,
             logged_at: datetime | None = None, v1: dict | None = None, clock=None) -> tuple[dict, dict, bool, list[dict]]:
    """Compute the v3 line. Returns (line, state, state_is_persistable, new_scores). Writes nothing."""
    import time

    import torch

    from ai_trader.ml import xs_net as xs
    from ai_trader.research import xs_network_study as study
    from ai_trader.storage.lake import Lake

    clock = clock or time.time
    t0 = clock()
    paths = paths or PathsV3.default()
    lake = lake or Lake()
    as_of = nn._utc(as_of)
    logged_at = logged_at or datetime.now(UTC)
    flags: list[str] = []
    alarms: list[str] = []
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(TORCH_THREADS)
        fz = load_frozen(paths)
        syms = fz.symbols
        if trained_through(fz.spec) > as_of:
            flags.append("model_fitted_after_as_of")

        # ---- inputs
        frames, absent, pf = token_frames(as_of, syms, lake)
        if len(frames) < xs.MIN_TOKENS:
            raise RuntimeError(f"only {len(frames)} tokens with enough history (< {xs.MIN_TOKENS})")
        S, last_close = forecast_hour(frames)
        if as_of - last_close > STALE_AFTER:
            flags.append("stale")
            alarms.append(f"stale data: newest bar closed {(as_of - last_close).total_seconds() / 60:.0f} min before as_of")
        if logged_at - (S + timedelta(minutes=3)) > REPLAY_AFTER:
            flags.append("replay")
        closes = closes_by_ts(frames, S - timedelta(days=4))

        # ---- state, then score / resolve earlier lines (before this hour's band is issued)
        persistable = persist_state
        st = state if state is not None else load_state(paths)
        if st is not None and nn._utc(st["as_of"]) > as_of:
            st, persistable = None, False
            flags.append("aci_state_ahead_of_as_of_rebuilt_in_memory")
        if st is None:
            st = new_aci_state(fz.spec, as_of)
            flags.append("aci_state_initialised")
        old_scores = read_scores(paths)
        done = {(s["id"], s["horizon"]) for s in old_scores}
        new_scores, n_upd = [], {f"{h}h": 0 for h in REL_H}
        for ln in read_lines(paths, since=S - timedelta(days=4)):
            Sl = nn._utc(ln["forecast_hour"])
            if Sl >= S or ln.get("schema") != VERSION:
                continue
            for h in sorted({*VOL_SHOW, *REL_H}):
                if Sl + timedelta(hours=h) > last_close:
                    continue
                if h in REL_H:
                    n_upd[f"{h}h"] += aci_update(st, ln, h, closes)
                if (ln["id"], f"{h}h") in done:
                    continue
                sc = score_line(ln, h, closes)
                if sc is None:
                    if last_close - (Sl + timedelta(hours=h)) < SCORE_GIVE_UP:
                        continue
                    sc = {"id": ln["id"], "horizon": f"{h}h", "forecast_hour": ln["forecast_hour"], "status": "no realised closes"}
                new_scores.append({"scored_at": logged_at.isoformat(), "scored_by": f"xs3-{S:%Y%m%dT%H}", **sc})
                done.add((ln["id"], f"{h}h"))

        # ---- panel row for hour S
        P, i = panel_row(frames, syms, S)
        present = P["present"][i].copy()
        for j, s in enumerate(syms):
            if s in frames and not present[j]:
                lc = nn._utc(frames[s]["close_ts"].max())
                absent[s] = f"stale (last bar closed {lc:%m-%d %H:%M})" if lc < S else "incomplete features"
        n_pres = int(present.sum())
        if n_pres < xs.MIN_TOKENS:
            raise RuntimeError(f"only {n_pres} tokens present at {S.isoformat()} (< {xs.MIN_TOKENS}): no line")
        idx = np.array([i])
        B = study.block(P, idx, fz.std())
        vols, rels, ws, reps = [], [], [], []
        for m in fz.models:
            v, r, w = xs.predict(m, B, fz.sec, attn=True)
            vols.append(v[0])
            rels.append(r[0])
            ws.append(w[0].astype(np.float64))
            reps.append(study.representation(m, B, fz.sec))
        vol, rel, W = np.mean(vols, 0), np.mean(rels, 0), np.mean(ws, 0)
        rep = np.concatenate(reps, axis=1)
        harq = study.har_quantiles(P, idx, fz.har())[0].astype(float)    # (N, 3, 7) units of sqrt(h) s

        pj, pj_sha = load_projection(paths, fz.spec_sha)
        if pj is None:
            flags.append("no_projection")
            xy = np.full((len(syms), 2), np.nan)
        else:
            xy = (rep - np.asarray(pj["center"])) @ np.asarray(pj["components"]).T

        s = np.sqrt(np.where(P["s2"][i] > 0, P["s2"][i], np.nan))
        lsig2 = P["X"][i, :, list(fz.spec["features"]).index("lsig2")].astype(float)
        med = {h: rel[:, k, 1] * math.sqrt(h) * s for k, h in enumerate(REL_H)}
        rk = {h: ranks_desc(med[h], present) for h in REL_H}
        rk_low = ranks_desc(-lsig2, present)
        att = attention_summary(W, present, syms)
        sectors_cfg = study.sectors(syms)
        if sectors_cfg != list(fz.spec["sectors"]):
            flags.append("sectors_yaml_differs_from_spec")
        tokens = []
        vi = {h: list(xs.VOL_H).index(h) for h in VOL_SHOW}
        for j, sym in enumerate(syms):
            t = {"symbol": sym, "sector": sectors_cfg[j], "sector_model": fz.spec["sectors"][j], "present": bool(present[j])}
            f = frames.get(sym)
            if f is not None:
                t["last_bar_close"] = nn._utc(f["close_ts"].max()).isoformat()
            if not present[j]:
                t["absent"] = absent.get(sym, "absent")
                tokens.append(t)
                continue
            cref = closes.get(sym, {}).get(_us(S))
            t.update({"close_ref": cref, "s_pct": _r(100 * s[j]), "lsig2": _r(lsig2[j]), "vol": {}, "rel": {},
                      "rank_4h": _r(rk[4][j], 4), "rank_24h": _r(rk[24][j], 4), "rank_lowvol": _r(rk_low[j], 4),
                      "xy": [_r(xy[j, 0], 5), _r(xy[j, 1], 5)], "att_received": _r(att["received"].get(sym), 4)})
            for h in VOL_SHOW:
                sc = 100 * math.sqrt(h) * s[j]
                t["vol"][f"{h}h"] = {"v3": [_r(x * sc) for x in vol[j, vi[h]]], "har": [_r(x * sc) for x in harq[j, vi[h]]]}
            for k, h in enumerate(REL_H):
                o = st[f"{h}h"]["tokens"].get(sym)
                q = rel[j, k]
                scale = math.sqrt(h) * s[j]
                if o is not None:
                    blo, bhi, Q = o.band(float(q[0]), float(q[2]))
                    Qv, at = (Q if math.isfinite(Q) else None), o.alpha_t
                else:
                    blo, bhi, Qv, at = float(q[0]), float(q[2]), None, None
                t["rel"][f"{h}h"] = {"q_bp": [_r(x * scale * 1e4) for x in q], "aci_bp": [_r(blo * scale * 1e4), _r(bhi * scale * 1e4)],
                                     "q_std": [_r(x) for x in q], "aci_std": [_r(blo), _r(bhi)], "Q": _r(Qv), "alpha_t": _r(at),
                                     "scale_bp": _r(scale * 1e4)}
            tokens.append(t)

        # ---- summary
        pres = [t for t in tokens if t["present"]]
        by4 = sorted(pres, key=lambda t: t["rank_4h"])
        low5 = {t["symbol"] for t in sorted(pres, key=lambda t: t["rank_lowvol"])[:TOP]}
        summary = {"rank_corr_lowvol_4h": spearman(med[4], -lsig2, present), "rank_corr_lowvol_24h": spearman(med[24], -lsig2, present),
                   "top5_4h": [t["symbol"] for t in by4[:TOP]], "bottom5_4h": [t["symbol"] for t in by4[-TOP:]],
                   "top5_lowvol": sorted(low5, key=lambda x: next(t["rank_lowvol"] for t in pres if t["symbol"] == x)),
                   "top5_overlap_lowvol": sum(t["symbol"] in low5 for t in by4[:TOP])}
        all_scores = old_scores + new_scores
        X = P["X"][i]
        ihash = hashlib.sha256(json.dumps([S.isoformat(), [[_r(v, 9) for v in row] for row in X.tolist()], [_r(v, 9) for v in P["M"][i].tolist()],
                                           present.astype(int).tolist()]).encode()).hexdigest()[:20]
        code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
        line = {
            "id": f"xs3-{S:%Y%m%dT%H}", "schema": VERSION, "model_id": MODEL_ID, "version": fz.spec["version"], "context_only": True,
            "note": NOT_A_SIGNAL, "logged_at": logged_at.isoformat(), "as_of": as_of.isoformat(), "forecast_hour": S.isoformat(),
            "windows": {f"{h}h": [S.isoformat(), (S + timedelta(hours=h)).isoformat()] for h in sorted({*VOL_SHOW, *REL_H})},
            "data_max_ts": (pf.data_max() or last_close).isoformat(), "last_bar_close": last_close.isoformat(),
            "flags": sorted(set(flags)), "alarms": alarms,
            "models": {**fz.card(), "projection_sha256": pj_sha[:16] if pj_sha else None, "live_code_sha256": code, "tool_version": VERSION},
            "inputs_hash": ihash, "n_present": n_pres, "n_symbols": len(syms),
            "absent": {k: v for k, v in absent.items() if k in syms and not present[syms.index(k)]},
            "tokens": tokens, "attention": att,
            "map": {"explained": pj["explained"] if pj else None, "anchors": pj["anchors"] if pj else None,
                    "method": pj["method"] if pj else None},
            "summary": summary, "vs_v1": v1_compare(v1), "running": running(all_scores),
            "aci_updates": n_upd, "aci_24h_note": "24h ACI band known to under-cover (holdout 0.749 at 80%, 6/23 tokens within 3 pp)",
            "scored": [f"{x['id']}|{x['horizon']}" for x in new_scores],
        }
        st["as_of"] = as_of.isoformat()
        line["runtime_s"] = round(clock() - t0, 1)
        return nn._clean(line), st, persistable, [nn._clean(x) for x in new_scores]
    finally:
        torch.set_num_threads(prev_threads)


def write_outputs(paths: PathsV3, line: dict, st: dict, persistable: bool, scores: list[dict], *, force: bool = False) -> dict:
    out = {"journal": None, "scores": 0, "state": False}
    paths.journal.mkdir(parents=True, exist_ok=True)
    existing = {r["id"] for r in read_lines(paths)}
    if line["id"] in existing and not force:
        out["journal"] = f"skipped: {line['id']} already logged"
    else:
        if line["id"] in existing:
            line = {**line, "id": f"{line['id']}-r{sum(1 for i in existing if i.startswith(line['id']))}", "flags": sorted({*line["flags"], "rerun"})}
        p = paths.journal / f"{line['forecast_hour'][:10]}.jsonl"
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        out["journal"] = str(p)
    if scores:
        with (paths.journal / "scores.jsonl").open("a", encoding="utf-8") as fh:
            for s in scores:
                fh.write(json.dumps(s, default=str) + "\n")
        out["scores"] = len(scores)
    if persistable and not str(out["journal"]).startswith("skipped"):
        save_state(paths, st)
        out["state"] = True
    return out


# ------------------------------------------------------------------ LATEST.md block


def _short(sym: str) -> str:
    return sym.removesuffix("USDT")


def _bp(x) -> str:
    return "n/a" if x is None else f"{x:+.0f}"


def render_block(line: dict) -> list[str]:
    """At most 8 lines for research/live/LATEST.md."""
    toks = {t["symbol"]: t for t in line["tokens"] if t["present"]}
    sm = line["summary"]

    def item(s):
        r = toks[s]["rel"]["4h"]
        return f"{_short(s)} {_bp(r['q_bp'][1])} [{_bp(r['aci_bp'][0])}, {_bp(r['aci_bp'][1])}]"

    L = [f"\n## Cross-section v3 (context only, not a signal: R-XS-2/R-XS-3; {line['id']}, {line['n_present']}/{line['n_symbols']} tokens)",
         f"- 4h relative median vs EW basket, bp [80% ACI]: top {', '.join(item(s) for s in sm['top5_4h'])}",
         f"- bottom {', '.join(item(s) for s in sm['bottom5_4h'])}"]
    c4, c24 = sm.get("rank_corr_lowvol_4h"), sm.get("rank_corr_lowvol_24h")
    L.append(f"- mostly low vol: rank corr with -lsig2 4h {_f(c4)} / 24h {_f(c24)}; top 5 shared with the low-vol top 5: {sm['top5_overlap_lowvol']}/5 "
             f"({', '.join(_short(x) for x in sm['top5_lowvol'])})")
    b = toks.get("BTCUSDT")
    if b:
        v = b["vol"]["1h"]
        cmp = line.get("vs_v1")
        s = f"- BTC 1h |r|: v3 q50 {v['v3'][3]:.2f}%, q90 {v['v3'][5]:.2f}% (log-HAR-t q90 {v['har'][5]:.2f}%)"
        if cmp:
            s += f" vs v1 90% half-width {cmp['half90_pct']:.2f}% (ACI90 {_f(cmp.get('aci90_half_pct'), sign=False)}%)"
        L.append(s)
    ru = line.get("running", {})
    parts = []
    for h in REL_H:
        r = ru.get(f"rel_{h}h")
        parts.append(f"{h}h IC v3 {_f(r['ic_v3'], 3)} vs low-vol {_f(r['ic_lowvol'], 3)} (n {r['n_lines']})" if r else f"{h}h IC n/a")
    v = ru.get("vol_1h")
    parts.append(f"1h vol pinball v3/HAR {v['ratio']:.3f} (n {v['n_lines']})" if v and v.get("ratio") else "1h vol pinball n/a")
    L.append("- running (scored lines): " + "; ".join(parts))
    L.append("- 24h ACI band under-covers (holdout 0.749 at 80%); the decision sessions do not read this block"
             + (f"; flags: {', '.join(line['flags'])}" if line["flags"] else ""))
    return L


def _f(v, nd: int = 2, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


__all__ = ["MODEL_ID", "VERSION", "PathsV3", "PrefetchLake", "attention_summary", "fast_seasonal", "fit_projection", "load_frozen",
           "new_aci_state", "panel_row", "ranks_desc", "read_lines", "read_scores", "render_block", "run_line", "running", "score_line",
           "write_outputs"]
