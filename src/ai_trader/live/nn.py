"""Live hourly line of the frozen BTC networks (research only; reads the lake, never trades).

    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_nn.py [--as-of ISO] [--dry-run]

Meant for hh:03 UTC, just after each 1h bar closes. Everything is point in time: bars / funding / OI come from
Lake.read_asof(as_of) (known_ts <= as_of), the forecast bar S is the bar that opens at the close of the newest known bar.

What one run computes (one journal line, `run_line`):
  pnn 1h   the frozen 2-seed polynomial network of agent F2 (data/models/pnn/wf_1h/<refit>.pt, refit month taken from
           frozen_1h.json), the H-PNN1 walk-forward path: ml.hourly.symbol_hourly features of bar S (bars <= S-1 only;
           an empty placeholder bar is appended exactly as tools.vol_ml does), the checkpoint's fit-block Standardizer,
           clip +-6, mean of the two seeds' quantiles (model units y = r / sqrt(s2)) x sqrt(s2) -> % log return of
           bar S (close S-1h... close S+1h is the window [S, S+1h)).
  pnn 24h  the frozen 1d network (wf_1d/<refit>.pt) on a ROLLING day: its 14 daily features are computed by
           ml.features.symbol_features on bars, funding and OI shifted back by k = S.hour hours, so the 'UTC day' of the
           construction is the 24 h ending at S; the calendar inputs are recomputed on the real window (dow of the
           window's majority date, TradFi-weekend hours inside [S, S+24h)). At k = 0 this is exactly the training
           construction (tests check it against the study's cached panel). Window [S, S+24h). Trained on UTC days only,
           so k > 0 is a mild extrapolation (flagged in the line as 'rolling_day').
  sigma    the network's sigma = (q90 - q10) / 2.5631 (normal-equivalent of the 80% band), in %.
  ACI      adaptive conformal (Gibbs-Candes; the pnn.aci update, implemented step-wise in OnlineACI and tested equal
           to pnn.aci) at 80% ([q10, q90]), 90% and 95% ([q05, q95]), CQR scores in model units. 1h: gamma 0.005,
           window 1440, warm-started from the end of F2's walk-forward stream (latent_1h.parquet, rows whose label is
           known at as_of), then advanced over every closed bar with the frozen network. 24h: outcomes arrive 24 h
           later and windows overlap, so each resolved forecast moves alpha by gamma / 24 (= one daily step per day),
           window 2000; warm start from latent_1d.parquet. State: data/models/pnn/live_aci_state.json.
  latent   z_S (seed-0 bottleneck of the 1h net), zhat_{S+1} = A z_S + B u_{S+1} + c (the checkpoint's LatentDynamics);
           the realised error of last hour's zhat_S vs persistence z_{S-1} (latent sd units) and the 168 h R^2 gain.
  field    agent F1's frozen field (data/models/field/index.json 'latest'), cards.card on the state at as_of.
  H-NN10   the newest BTCUSDT line of data/research/vol_ml_forward/<date>.jsonl logged by as_of (read only).
  news     ai_trader.news.alt.daily_alt_features(as_of) (no network: refresh=False) and RSS forward counts.
  alarms   stale data; running coverage of each band over the last 72 / 168 resolved forecasts outside the binomial
           99% interval; latent model worse than persistence over 168 h; any network input with |z| > 4 against its
           training standardisation; models fitted after as_of (only possible in replays).
  scoring  every earlier line whose window has closed is scored once into data/journal/live_nn/scores.jsonl.
Outputs: data/journal/live_nn/<YYYY-MM-DD of S>.jsonl (append-only; one line per forecast bar S, id nn-<S>),
research/live/LATEST.md (overwritten), the ACI state. Lines computed more than 15 min after S + 3 min are flagged 'replay'.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl

SYMBOL = "BTCUSDT"
VERSION = "live_nn.v1"
LEVELS = (80, 90, 95)
BAND = {80: (1, 5), 90: (0, 6), 95: (0, 6)}  # indices into pnn.TAUS
QNAMES = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
NORM80 = 2.5631031310892007  # 2 * Phi^-1(0.90)
GAMMA = {"1h": 0.005, "24h": 0.005 / 24}
WINDOW = {"1h": 1440, "24h": 2000}
HIST = 168
STALE_AFTER = timedelta(hours=1, minutes=5)
REPLAY_AFTER = timedelta(minutes=18)
Z_ALARM = 4.0
H_DELTA = {"1h": timedelta(hours=1), "24h": timedelta(hours=24)}
NET_H = {"1h": "1h", "24h": "1d"}  # live horizon -> network checkpoint
TS_COLS = ("event_ts", "close_ts", "known_ts")


# ------------------------------------------------------------------ paths


@dataclass
class Paths:
    root: Path
    models_root: Path | None = None   # tests: read the frozen models from the repo, write everything else under root

    @classmethod
    def default(cls) -> Paths:
        from ai_trader.core.config import load

        return cls(load().root)

    @property
    def pnn(self) -> Path:
        return (self.models_root or self.root) / "data" / "models" / "pnn"

    @property
    def field(self) -> Path:
        return (self.models_root or self.root) / "data" / "models" / "field"

    @property
    def journal(self) -> Path:
        return self.root / "data" / "journal" / "live_nn"

    @property
    def state(self) -> Path:
        return self.root / "data" / "models" / "pnn" / "live_aci_state.json"

    @property
    def latest_md(self) -> Path:
        return self.root / "research" / "live" / "LATEST.md"

    @property
    def vol_forward(self) -> Path:
        return self.root / "data" / "research" / "vol_ml_forward"


def sha256_file(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _iso(t) -> str | None:
    return None if t is None else (t.isoformat() if hasattr(t, "isoformat") else str(t))


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.fromisoformat(t.replace("Z", "+00:00"))
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


# ------------------------------------------------------------------ frozen networks


@dataclass
class Net:
    h: str
    month: str
    features: tuple
    mu: np.ndarray
    sd: np.ndarray
    fits: list
    dyn: object
    sha_pt: str
    sha_frozen: str

    def zraw(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, float) - self.mu) / self.sd

    def _Z(self, X):
        from ai_trader.ml import pnn

        return np.clip(np.nan_to_num(self.zraw(X), nan=0.0), -pnn.CLIP, pnn.CLIP)

    def quantiles(self, X: np.ndarray) -> np.ndarray:
        """2-seed mean quantiles in model units; NaN rows where any input is missing."""
        from ai_trader.ml import pnn

        X = np.atleast_2d(np.asarray(X, float))
        q = pnn.ensemble_quantiles([f.predict(self._Z(X)) for f in self.fits])
        q[~np.isfinite(X).all(axis=1)] = np.nan
        return q

    def latent(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        z = self.fits[0].latent(self._Z(X))
        z[~np.isfinite(X).all(axis=1)] = np.nan
        return z

    @property
    def train_end(self) -> datetime:
        """Labels of the refit end before the refit month starts."""
        y, m = (int(v) for v in self.month.split("-"))
        return datetime(y, m, 1, tzinfo=UTC)

    def card(self) -> dict:
        return {"model": f"pnn_{self.h}", "refit_month": self.month, "pt_sha256": self.sha_pt, "frozen_json_sha256": self.sha_frozen,
                "n_seeds": len(self.fits), "features": len(self.features)}


@lru_cache(maxsize=4)
def load_net(h: str, pnn_dir: str) -> Net:
    import torch

    from ai_trader.ml import pnn

    torch.set_num_threads(1)
    d = Path(pnn_dir)
    fz = d / f"frozen_{h}.json"
    spec = json.loads(fz.read_text(encoding="utf-8"))
    pt = d / f"wf_{h}" / f"{spec['refit_month']}.pt"
    ck = torch.load(pt, weights_only=False)
    fits = [pnn.QFit("pnn", s, ck["pnn_meta"], 0.0, 0) for s in ck["pnn"]]
    dy = ck["dyn"]
    dyn = pnn.LatentDynamics(*(np.asarray(dy[k], float) for k in ("A", "B", "c", "zmu", "zsd")))
    mu, sd = np.asarray(ck["std"]["mu"], float), np.asarray(ck["std"]["sd"], float)
    if not np.allclose(mu, spec["std_mu"]) or not np.allclose(sd, spec["std_sd"]):
        raise RuntimeError(f"frozen_{h}.json and {pt.name} disagree on the standardiser")
    return Net(h, spec["refit_month"], tuple(spec["features"]), mu, sd, fits, dyn, sha256_file(pt), sha256_file(fz))


# ------------------------------------------------------------------ features (point in time)


def hourly_frame(f: pl.DataFrame) -> tuple[pl.DataFrame, datetime]:
    """ml.hourly features for every bar of the market frame plus the next (not yet opened / closed) bar S, and the
    signed return r of each closed bar (null where the previous bar is missing). Same placeholder method as
    tools.vol_ml._next_bar_row."""
    from ai_trader.ml.hourly import symbol_hourly

    g = f.select("event_ts", "ret", "sigma_risk_next").sort("event_ts").unique("event_ts", keep="last", maintain_order=True)
    nxt = g["event_ts"][-1] + timedelta(hours=1)
    ph = pl.DataFrame({"event_ts": [nxt], "ret": [None], "sigma_risk_next": [None]}, schema=g.schema)
    h = symbol_hourly(pl.concat([g, ph]))
    h = h.join(g.select("event_ts", pl.col("ret").cast(pl.Float64).alias("r_raw")), on="event_ts", how="left").with_columns(
        pl.when(pl.col("r2").is_not_null()).then(pl.col("r_raw")).otherwise(None).alias("r")).drop("r_raw")
    return h, nxt


def _np(df: pl.DataFrame, cols) -> np.ndarray:
    return np.column_stack([df[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in cols]).astype(float)


def _shift(df: pl.DataFrame, d: timedelta) -> pl.DataFrame:
    return df.with_columns([pl.col(c) - d for c in TS_COLS if c in df.columns]) if df.height else df


class ShiftedLake:
    """A read-only view of the lake with every timestamp moved back by `shift` (for the rolling-day construction)."""

    def __init__(self, lake, shift: timedelta):
        self.lake, self.shift = lake, shift

    def read_asof(self, table: str, as_of: datetime, **kw):
        if kw.get("start") is not None:
            kw["start"] = kw["start"] + self.shift
        return _shift(self.lake.read_asof(table, as_of + self.shift, **kw), self.shift)


def resample_4h(f1s: pl.DataFrame) -> pl.DataFrame:
    """4h bars (complete blocks of 4 hourly bars, aligned to the frame's midnight) with market_frame's columns. Used
    when the shift is not a multiple of 4 h: the regime features (vol_of_vol_30d) need a 4h bar ending each day."""
    from ai_trader.research.frame import add_robust_vol, enrich_bars

    b = (f1s.sort("event_ts").group_by_dynamic("event_ts", every="4h", closed="left", label="left")
         .agg(pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(), pl.col("volume").sum(),
              pl.col("turnover").sum() if "turnover" in f1s.columns else pl.lit(None).alias("turnover"),
              pl.col("close_ts").max(), pl.col("known_ts").max(), pl.len().alias("_n"))
         .filter(pl.col("_n") == 4).drop("_n"))
    return add_robust_vol(enrich_bars(b))


def daily_row(f1: pl.DataFrame, f4: pl.DataFrame, lake, S: datetime, as_of: datetime) -> dict:
    """The 1d network's 14 inputs for the 24 h window [S, S+24h), from data known at as_of (>= S)."""
    from ai_trader.ml.features import symbol_features
    from ai_trader.research.calendar import add_calendar

    k = timedelta(hours=S.hour)
    f1s = _shift(f1, k)
    f4s = _shift(f4, k) if S.hour % 4 == 0 else resample_4h(f1s)
    d = symbol_features(SYMBOL, f1s, f4s, ShiftedLake(lake, k), as_of - k)
    T0 = (S - k).replace(tzinfo=None)
    row = d.filter(pl.col("T").dt.replace_time_zone(None).cast(pl.Datetime("us")) == pl.lit(T0).cast(pl.Datetime("us")))
    if row.height != 1:
        return {"ok": False, "reason": f"no daily row for window {S.isoformat()}"}
    out = {c: row[c].cast(pl.Float64)[0] for c in row.columns if row[c].dtype.is_numeric()}
    maj = (S + timedelta(hours=12)).date()
    wd = maj.isoweekday()
    out["dow_sin"], out["dow_cos"] = math.sin(2 * math.pi * (wd - 1) / 7), math.cos(2 * math.pi * (wd - 1) / 7)
    hh = pl.DataFrame({"event_ts": pl.datetime_range(S, S + timedelta(hours=23), "1h", eager=True, time_zone="UTC")})
    out["next_wk_hours"] = float(add_calendar(hh)["tradfi_weekend"].cast(pl.Float64).sum())
    out["ok"] = True
    out["shift_hours"] = S.hour
    return out


# ------------------------------------------------------------------ online ACI


@dataclass
class OnlineACI:
    """Gibbs-Candes adaptive conformal, one stream, the same update and rank rule as pnn.aci (tested equal)."""

    level: int
    gamma: float
    window: int
    alpha_t: float | None = None
    scores: list = field(default_factory=list)
    hist: list = field(default_factory=list)   # [[ts, err_aci, err_raw], ...] last HIST resolved forecasts
    n: int = 0

    def __post_init__(self):
        if self.alpha_t is None:
            self.alpha_t = self.alpha

    @property
    def alpha(self) -> float:
        return 1 - self.level / 100

    def Q(self) -> float:
        srt = sorted(self.scores)
        m, at = len(srt), self.alpha_t
        if m == 0:
            return 0.0
        if at <= 0:
            return srt[-1]
        if at >= 1:
            return -math.inf
        k = int(math.ceil((m + 1) * (1 - at)))
        return srt[min(max(k, 1), m) - 1]

    def band(self, lo: float, hi: float) -> tuple[float, float, float]:
        Q = self.Q()
        if not math.isfinite(Q):
            mid = 0.5 * (lo + hi)
            return mid, mid, Q
        return lo - Q, hi + Q, Q

    def resolve(self, err: float, score: float, ts: str | None = None, err_raw: float | None = None) -> None:
        """Outcome of a forecast issued earlier (its band was computed then): move alpha, store the score."""
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
        if math.isfinite(score):
            self.scores.append(float(score))
            if len(self.scores) > self.window:
                del self.scores[: len(self.scores) - self.window]
        self.hist.append([ts, float(err), None if err_raw is None else float(err_raw)])
        if len(self.hist) > HIST:
            del self.hist[: len(self.hist) - HIST]
        self.n += 1

    def step(self, lo: float, hi: float, y: float, ts: str | None = None) -> float:
        """Issue the band for (lo, hi), reveal y, update (pnn.aci's loop body)."""
        blo, bhi, Q = self.band(lo, hi)
        e = float(not (blo <= y <= bhi)) if math.isfinite(Q) else 1.0
        self.resolve(e, max(lo - y, y - hi), ts, float(not (lo <= y <= hi)))
        return e

    def to_dict(self) -> dict:
        return {"level": self.level, "gamma": self.gamma, "window": self.window, "alpha_t": self.alpha_t, "n": self.n,
                "scores": [float(f"{s:.7g}") for s in self.scores], "hist": self.hist}

    @classmethod
    def from_dict(cls, d: dict) -> OnlineACI:
        return cls(int(d["level"]), float(d["gamma"]), int(d["window"]), float(d["alpha_t"]), list(d["scores"]), list(d["hist"]), int(d.get("n", 0)))


def _warm_from_parquet(path: Path, h: str, as_of: datetime) -> tuple[dict[int, OnlineACI], str | None]:
    """Continue F2's walk-forward ACI: last aci_alpha + its last update, last WINDOW CQR scores, rows with label <= as_of."""
    step = H_DELTA[h]
    d = pl.read_parquet(path).sort("ts")
    d = d.filter(pl.col("ts") + step <= as_of)
    out = {}
    if d.height == 0:
        return {L: OnlineACI(L, GAMMA[h], WINDOW[h]) for L in LEVELS}, None
    sig = d["sigma_ref"].cast(pl.Float64).to_numpy()
    y = d["ret"].cast(pl.Float64).to_numpy() / sig
    for L in LEVELS:
        i, j = BAND[L]
        lo = d[QNAMES[i]].cast(pl.Float64).to_numpy() / sig
        hi = d[QNAMES[j]].cast(pl.Float64).to_numpy() / sig
        s = np.maximum(lo - y, y - hi)
        alo, ahi = d[f"aci_lo{L}"].cast(pl.Float64).to_numpy(), d[f"aci_hi{L}"].cast(pl.Float64).to_numpy()
        r = d["ret"].cast(pl.Float64).to_numpy()
        err = (~((alo <= r) & (r <= ahi))).astype(float)
        raw = (~((lo <= y) & (y <= hi))).astype(float)
        a_last = float(d[f"aci_alpha{L}"][-1])
        g_wf = 0.005  # the study's gamma (per step of the walk-forward stream)
        ts = [str(t) for t in d["ts"].to_list()[-HIST:]]
        o = OnlineACI(L, GAMMA[h], WINDOW[h], a_last + g_wf * ((1 - L / 100) - err[-1]),
                      [float(v) for v in s[np.isfinite(s)][-WINDOW[h]:]],
                      [[t, float(e), float(rw)] for t, e, rw in zip(ts, err[-HIST:], raw[-HIST:], strict=True)], int(d.height))
        out[L] = o
    return out, str(d["ts"][-1])


# ------------------------------------------------------------------ state


def new_state(paths: Paths, as_of: datetime) -> dict:
    a1, last1 = _warm_from_parquet(paths.pnn / "latent_1h.parquet", "1h", as_of)
    a24, last24 = _warm_from_parquet(paths.pnn / "latent_1d.parquet", "24h", as_of)
    return {"schema": "live_aci_state.v1", "created": datetime.now(UTC).isoformat(), "as_of": as_of.isoformat(),
            "warm": {"1h": f"latent_1h.parquet to {last1}", "24h": f"latent_1d.parquet to {last24}"},
            "1h": {"last_ts": last1, "aci": a1}, "24h": {"aci": a24, "pending": [], "last_resolved_end": None}}


def load_state(paths: Paths) -> dict | None:
    if not paths.state.is_file():
        return None
    d = json.loads(paths.state.read_text(encoding="utf-8"))
    for h in ("1h", "24h"):
        d[h]["aci"] = {int(k): OnlineACI.from_dict(v) for k, v in d[h]["aci"].items()}
    return d


def dump_state(st: dict) -> dict:
    out = json.loads(json.dumps({k: v for k, v in st.items() if k not in ("1h", "24h")}, default=str))
    for h in ("1h", "24h"):
        out[h] = {k: v for k, v in st[h].items() if k != "aci"}
        out[h]["aci"] = {str(L): a.to_dict() for L, a in st[h]["aci"].items()}
    return out


def save_state(paths: Paths, st: dict) -> None:
    paths.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.state.with_suffix(".tmp")
    tmp.write_text(json.dumps(dump_state(st), default=str), encoding="utf-8")
    tmp.replace(paths.state)


def coverage(a: OnlineACI, n: int, col: int = 1) -> dict:
    h = [x[col] for x in a.hist[-n:] if x[col] is not None]
    if not h:
        return {"n": 0}
    cov = 1 - float(np.mean(h))
    p = a.level / 100
    half = 2.576 * math.sqrt(p * (1 - p) / len(h))
    return {"n": len(h), "cov": round(cov, 4), "nominal": p, "alarm": bool(abs(cov - p) > half)}


# ------------------------------------------------------------------ realised outcomes


def closes(f: pl.DataFrame) -> dict:
    """close_ts -> close of each 1h bar (UTC datetimes)."""
    return {_utc(t): float(c) for t, c in zip(f["close_ts"].to_list(), f["close"].to_list(), strict=True)}


def realised(cl: dict, S: datetime, h: str) -> float | None:
    a, b = cl.get(_utc(S)), cl.get(_utc(S) + H_DELTA[h])
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return 100 * math.log(b / a)


def pinball(y: float, q: dict, taus: dict) -> float:
    return float(np.mean([max(t * (y - q[k]), (t - 1) * (y - q[k])) for k, t in taus.items()]))


TAU7 = {"q05": 0.05, "q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90, "q95": 0.95}


# ------------------------------------------------------------------ journal io


def read_lines(paths: Paths) -> list[dict]:
    rows = []
    for f in sorted(paths.journal.glob("20*.jsonl")) if paths.journal.is_dir() else []:
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted(rows, key=lambda r: (r["forecast_hour"], r["logged_at"]))


def read_scores(paths: Paths) -> list[dict]:
    p = paths.journal / "scores.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def score_line(line: dict, h: str, r: float) -> dict:
    n = line["net"][h]
    q = n["quantiles_pct"]
    out = {"id": line["id"], "horizon": h, "window_start": n["window_start"], "window_end": n["window_end"], "r_pct": r, "abs_r_pct": abs(r),
           "sigma_pct": n["sigma_pct"], "z": r / n["sigma_pct"] if n["sigma_pct"] else None, "pinball7": pinball(r, q, TAU7),
           "pinball3": pinball(r, q, {"q10": 0.1, "q50": 0.5, "q90": 0.9}),
           "hit_raw80": q["q10"] <= r <= q["q90"], "hit_raw90": q["q05"] <= r <= q["q95"], "flags": line.get("flags", [])}
    for L in LEVELS:
        b = n["aci"][str(L)]
        out[f"hit_aci{L}"] = b["lo_pct"] <= r <= b["hi_pct"]
    return out


# ------------------------------------------------------------------ side inputs


def field_card(paths: Paths, as_of: datetime, lake) -> dict:
    from ai_trader.field.cards import card
    from ai_trader.field.model import Field
    from ai_trader.field.state import COORDS, state_asof

    idx = json.loads((paths.field / "index.json").read_text(encoding="utf-8"))
    fp = paths.field / idx["latest"]
    fld = Field.load(fp)
    st, _ = state_asof(as_of, lake=lake)
    st = st.filter(pl.col("known_ts") <= as_of).drop_nulls(list(COORDS))
    c = card(fld, st, st.height - 1, moves=(0.01, 0.02, 0.03), n_paths=4000, seed=0)
    fit_as_of = fld.meta.get("fit_as_of")
    hz = {k: {"quantiles_pct": {q: round(v, 4) for q, v in h["quantiles_pct"].items()}, "sd_pct": round(h["sd_pct"], 4),
              "p_jump": round(h["p_jump"], 4), "p_touch": {m: round(p, 4) for m, p in h["p_touch_pct"].items()},
              "p_up_close": round(h["p_up_close"], 4)} for k, h in c["horizons"].items()}
    return {"file": fp.name, "sha256": sha256_file(fp), "fit_as_of": fit_as_of, "state_known_ts": c["as_of"], "close": c["close"],
            "cell": c["cell"], "one_hour": {k: round(v, 5) for k, v in c["one_hour"].items()}, "horizons": hz,
            "drivers_D": [[d["term"], round(d["contribution"], 4)] for d in c["drivers_D"][:3]],
            "drivers_lam": [[d["term"], round(d["contribution"], 4)] for d in c["drivers_lam"][:3]],
            "state": {k: round(v, 4) for k, v in c["state"].items()}, "text": c["text"],
            "fit_after_as_of": bool(fit_as_of and _utc(fit_as_of) > as_of)}


def hnn10_line(paths: Paths, as_of: datetime) -> dict:
    best = None
    for d in (as_of.date() - timedelta(days=1), as_of.date()):
        p = paths.vol_forward / f"{d:%Y-%m-%d}.jsonl"
        if not p.is_file():
            continue
        for x in p.read_text(encoding="utf-8").splitlines():
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("symbol") != SYMBOL or not r.get("logged_at") or _utc(r["logged_at"]) > as_of:
                continue
            if best is None or r["logged_at"] >= best["logged_at"]:
                best = r
    if best is None:
        return {"status": "missing"}
    keep = ("logged_at", "forecast_for", "sigma_next_hour_ml", "sigma_next_hour_baseline", "ratio", "top_shap_groups", "flags", "model_id")
    return {"status": "ok", **{k: best.get(k) for k in keep}}


def news_block(as_of: datetime) -> dict:
    out: dict = {}
    try:
        from ai_trader.news.alt import ALT_COLS, RAW_COLS, daily_alt_features

        row = daily_alt_features(as_of)
        out["alt"] = {c: (None if row[c][0] is None else round(float(row[c][0]), 4)) for c in (*ALT_COLS, *RAW_COLS) if c in row.columns}
        out["alt_days"] = {c: str(row[c][0]) for c in row.columns if c.startswith("day_")}
    except ImportError:
        out["alt"] = "pending"
    except Exception as e:  # noqa: BLE001 - a side input must not stop the line
        out["alt"] = f"error: {type(e).__name__}: {e}"[:200]
    try:
        from ai_trader.news.rss import hourly_counts
        from ai_trader.news.store import Manifest

        hc = hourly_counts(Manifest())
        if hc.height:
            hc = hc.filter(pl.col("known_ts") <= as_of)
            n1 = int(hc.filter(pl.col("known_ts") > as_of - timedelta(hours=1))["n"].sum())
            n24 = hc.filter((pl.col("known_ts") > as_of - timedelta(hours=24)) & ~pl.col("first_poll"))
            last = hc["known_ts"].max() if hc.height else None
            out["rss"] = {"items_last_1h": n1, "items_last_24h_excl_backlog": int(n24["n"].sum()), "last_hour_known": _iso(last),
                          "polling_stale": bool(last is None or as_of - last > timedelta(hours=3))}
        else:
            out["rss"] = "no polls yet"
    except Exception as e:  # noqa: BLE001
        out["rss"] = f"error: {type(e).__name__}: {e}"[:200]
    return out


# ------------------------------------------------------------------ the run


def _qdict(q_model: np.ndarray, scale: float) -> dict:
    return {k: float(100 * v * scale) for k, v in zip(QNAMES, q_model, strict=True)}


def _zalarms(net: Net, x: np.ndarray, tag: str) -> list[str]:
    z = net.zraw(x[None, :])[0]
    return [f"input_z:{tag}:{c}={v:+.1f}" for c, v in zip(net.features, z, strict=True) if np.isfinite(v) and abs(v) > Z_ALARM]


def run_line(as_of: datetime, *, paths: Paths | None = None, lake=None, state: dict | None = None, persist_state: bool = True,
             with_field: bool = True, with_news: bool = True, logged_at: datetime | None = None) -> tuple[dict, dict, bool, list[dict]]:
    """Compute the line. Returns (line, state, state_is_persistable, new_scores). Writes nothing."""
    from ai_trader.ml import pnn
    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.frame import market_frame
    from ai_trader.research.pnn_study import DAILY_FEATURES
    from ai_trader.storage.lake import Lake

    paths = paths or Paths.default()
    lake = lake or Lake()
    as_of = _utc(as_of)
    logged_at = logged_at or datetime.now(UTC)
    flags: list[str] = []
    alarms: list[str] = []

    f1 = market_frame(SYMBOL, "1h", as_of, funding=False, oi=False, lake=lake)
    f4 = market_frame(SYMBOL, "4h", as_of, funding=False, oi=False, lake=lake)
    if f1.height < 24 * 120:
        raise RuntimeError("insufficient 1h history")
    data_max = _utc(f1["known_ts"].max())
    last_close = _utc(f1["close_ts"].max())
    if as_of - last_close > STALE_AFTER:
        flags.append("stale")
        alarms.append(f"stale data: newest bar closed {(as_of - last_close).total_seconds() / 60:.0f} min before as_of")
    H, S = hourly_frame(f1)
    S = _utc(S)
    if logged_at - (S + timedelta(minutes=3)) > REPLAY_AFTER:
        flags.append("replay")
    cl = closes(f1)

    # ---- 1h network over recent rows (catch-up + current)
    n1 = load_net("1h", str(paths.pnn))
    tail = H.tail(24 * 10)
    X1 = _np(tail, HOURLY_FEATURES)
    s2 = tail["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    q1 = n1.quantiles(X1)
    z1 = n1.latent(X1)
    ts1 = [_utc(t) for t in tail["event_ts"].to_list()]
    r1 = tail["r"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    if ts1[-1] != S:
        raise RuntimeError("forecast row missing")
    x_now = X1[-1]
    if not np.isfinite(x_now).all():
        flags.append("incomplete_features_1h")
    alarms += _zalarms(n1, x_now, "1h")
    if n1.train_end > as_of:
        flags.append("pnn_1h_trained_after_as_of")

    # ---- state
    persistable = persist_state
    st = state if state is not None else load_state(paths)
    if st is not None and st["1h"]["last_ts"] and _utc(st["1h"]["last_ts"]) >= S:
        st, persistable = None, False
        flags.append("aci_state_ahead_of_as_of_rebuilt_in_memory")
    if st is not None and any(_utc(p["window_start"]) > S for p in st["24h"]["pending"]):
        st, persistable = None, False
        flags.append("aci_state_ahead_of_as_of_rebuilt_in_memory")
    if st is None:
        st = new_state(paths, as_of)
        flags.append("aci_state_initialised")

    # 1h ACI catch-up over closed bars after last_ts (the frozen net on each)
    last = _utc(st["1h"]["last_ts"]) if st["1h"]["last_ts"] else None
    if last is not None and last < ts1[0] - timedelta(hours=1):
        # gap longer than the tail: recompute over the full frame
        Xa = _np(H, HOURLY_FEATURES)
        qa, s2a, ra, tsa = n1.quantiles(Xa), H["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy(), H["r"].cast(pl.Float64).fill_null(np.nan).to_numpy(), [_utc(t) for t in H["event_ts"].to_list()]
    else:
        qa, s2a, ra, tsa = q1, s2, r1, ts1
    n_caught = 0
    for i, t in enumerate(tsa):
        if t >= S or (last is not None and t <= last) or not np.isfinite(ra[i]) or not np.isfinite(qa[i]).all() or not (s2a[i] > 0):
            continue
        y = ra[i] / math.sqrt(s2a[i])
        for L in LEVELS:
            a, b = BAND[L]
            st["1h"]["aci"][L].step(qa[i, a], qa[i, b], y, t.isoformat())
        st["1h"]["last_ts"] = t.isoformat()
        n_caught += 1

    # ---- 1h line
    net: dict = {}
    sc1 = math.sqrt(s2[-1]) if s2[-1] > 0 else float("nan")
    qd = _qdict(q1[-1], sc1)
    net["1h"] = {"window_start": S.isoformat(), "window_end": (S + H_DELTA["1h"]).isoformat(), "quantiles_pct": qd,
                 "sigma_pct": (qd["q90"] - qd["q10"]) / NORM80, "sigma_ref_pct": 100 * sc1, "aci": {}}
    for L in LEVELS:
        a, b = BAND[L]
        o = st["1h"]["aci"][L]
        lo, hi, Q = o.band(q1[-1, a], q1[-1, b])
        net["1h"]["aci"][str(L)] = {"lo_pct": 100 * lo * sc1, "hi_pct": 100 * hi * sc1, "Q": Q, "alpha_t": o.alpha_t}

    # ---- 24h network (rolling day)
    n24 = load_net("1d", str(paths.pnn))
    if n24.train_end > as_of:
        flags.append("pnn_1d_trained_after_as_of")
    dr = daily_row(f1, f4, lake, S, as_of)
    if dr.get("ok"):
        x24 = np.array([dr.get(c, np.nan) if dr.get(c) is not None else np.nan for c in DAILY_FEATURES], float)
        sig24 = math.exp(0.5 * x24[DAILY_FEATURES.index("lrv_w")])
        q24 = n24.quantiles(x24)[0]
        zl24 = n24.latent(x24)[0]
        if not np.isfinite(x24).all():
            flags.append("incomplete_features_24h")
        alarms += _zalarms(n24, x24, "24h")
    else:
        x24, sig24, q24, zl24 = np.full(len(DAILY_FEATURES), np.nan), float("nan"), np.full(7, np.nan), np.full(4, np.nan)
        flags.append("no_daily_row")
    # resolve pending 24h forecasts whose window has closed
    keep, n_res = [], 0
    for p in sorted(st["24h"]["pending"], key=lambda p: p["window_start"]):
        ws = _utc(p["window_start"])
        if ws + H_DELTA["24h"] > last_close:
            keep.append(p)
            continue
        r = realised(cl, ws, "24h")
        if r is None:
            if last_close - ws > timedelta(hours=72):
                continue
            keep.append(p)
            continue
        y = r / 100 / p["sig"]
        for L in LEVELS:
            o = st["24h"]["aci"][L]
            b = p["band"][str(L)]
            err = float(not (b[0] <= y <= b[1]))
            o.resolve(err, max(p["raw"][str(L)][0] - y, y - p["raw"][str(L)][1]), ws.isoformat(),
                      float(not (p["raw"][str(L)][0] <= y <= p["raw"][str(L)][1])))
        st["24h"]["last_resolved_end"] = (ws + H_DELTA["24h"]).isoformat()
        n_res += 1
    st["24h"]["pending"] = keep
    qd24 = _qdict(q24, sig24)
    net["24h"] = {"window_start": S.isoformat(), "window_end": (S + H_DELTA["24h"]).isoformat(), "quantiles_pct": qd24,
                  "sigma_pct": (qd24["q90"] - qd24["q10"]) / NORM80, "sigma_ref_pct": 100 * sig24, "aci": {},
                  "construction": f"rolling day, shift {S.hour} h" + (" (= training construction)" if S.hour == 0 else "")}
    pend = {"window_start": S.isoformat(), "sig": sig24, "band": {}, "raw": {}}
    for L in LEVELS:
        a, b = BAND[L]
        o = st["24h"]["aci"][L]
        lo, hi, Q = o.band(q24[a], q24[b])
        net["24h"]["aci"][str(L)] = {"lo_pct": 100 * lo * sig24, "hi_pct": 100 * hi * sig24, "Q": Q, "alpha_t": o.alpha_t}
        pend["band"][str(L)], pend["raw"][str(L)] = [lo, hi], [float(q24[a]), float(q24[b])]
    if np.isfinite(q24).all() and not any(p["window_start"] == pend["window_start"] for p in st["24h"]["pending"]):
        st["24h"]["pending"].append(pend)
    if S.hour:
        flags.append("rolling_day")

    # ---- coverage alarms
    cov = {}
    for h in ("1h", "24h"):
        for L in LEVELS:
            o = st[h]["aci"][L]
            for n in (72, 168):
                c = coverage(o, n)
                cov[f"{h}_{L}_{n}"] = c
                if c.get("alarm"):
                    alarms.append(f"coverage {h} ACI {L}% over last {c['n']}: {c['cov']:.3f} vs {L / 100:.2f}")
            c = coverage(o, 168, col=2)
            cov[f"{h}_{L}_168_raw"] = c

    # ---- latent (1h)
    zmu, zsd = n1.dyn.zmu, n1.dyn.zsd
    nxt_t = [t + timedelta(hours=1) for t in ts1]
    u = np.array([[math.sin(2 * math.pi * t.hour / 24), math.cos(2 * math.pi * t.hour / 24),
                   math.sin(2 * math.pi * (t.isoweekday() - 1) / 7), math.cos(2 * math.pi * (t.isoweekday() - 1) / 7)] for t in nxt_t])
    zhat = n1.dyn.predict(z1, u)  # zhat[i] predicts the latent of row i+1
    lat = {"z": z1[-1].tolist(), "zhat_next": zhat[-1].tolist(), "zmu": zmu.tolist(), "zsd": zsd.tolist()}
    if len(ts1) >= 2 and ts1[-2] == S - timedelta(hours=1):
        e_m = (z1[-1] - zhat[-2]) / zsd
        e_p = (z1[-1] - z1[-2]) / zsd
        lat["last_error"] = {"for": S.isoformat(), "model_rmse": float(np.sqrt(np.mean(e_m ** 2))), "persistence_rmse": float(np.sqrt(np.mean(e_p ** 2))),
                             "model_err": e_m.tolist()}
    pairs = [i for i in range(len(ts1) - 1) if ts1[i + 1] - ts1[i] == timedelta(hours=1) and np.isfinite(z1[i]).all() and np.isfinite(z1[i + 1]).all()][-HIST:]
    if len(pairs) >= 24:
        g = pnn.latent_r2_gain(z1[pairs], z1[[i + 1 for i in pairs]], zhat[pairs], zmu, zsd)
        lat["r2_gain_168"] = {"n": len(pairs), "gain": g["gain"], "r2_model": g["r2_model"], "r2_persist": g["r2_persist"]}
        if g["gain"] < 0:
            alarms.append(f"latent dynamics worse than persistence over {len(pairs)} h (R^2 gain {g['gain']:+.3f}; holdout was +0.143)")
    lat["z_24h_net"] = zl24.tolist()

    # ---- side inputs
    fc: dict
    if with_field:
        try:
            fc = field_card(paths, as_of, lake)
            if fc.get("fit_after_as_of"):
                flags.append("field_fit_after_as_of")
        except Exception as e:  # noqa: BLE001
            fc = {"status": f"error: {type(e).__name__}: {e}"[:300]}
    else:
        fc = {"status": "skipped"}
    hn = hnn10_line(paths, as_of)
    nw = news_block(as_of) if with_news else {"alt": "skipped"}

    # ---- score earlier lines (window closed), once
    done = {(s["id"], s["horizon"]) for s in read_scores(paths)}
    new_scores = []
    for ln in read_lines(paths):
        if _utc(ln["forecast_hour"]) >= S:
            continue
        for h in ("1h", "24h"):
            if (ln["id"], h) in done or h not in ln.get("net", {}):
                continue
            ws = _utc(ln["net"][h]["window_start"])
            if ws + H_DELTA[h] > last_close:
                continue
            r = realised(cl, ws, h)
            nh = ln["net"][h]
            if r is None or not all(v is not None for v in nh["quantiles_pct"].values()) or nh.get("sigma_pct") is None \
                    or any(nh["aci"][str(L)]["lo_pct"] is None for L in LEVELS):
                continue
            new_scores.append({"scored_at": logged_at.isoformat(), "scored_by": f"nn-{S:%Y%m%dT%H}", **score_line(ln, h, r)})
            done.add((ln["id"], h))

    # ---- assemble
    ihash = hashlib.sha256(json.dumps([S.isoformat(), [float(f"{v:.12g}") for v in x_now], [float(f"{v:.12g}") for v in x24]]).encode()).hexdigest()[:20]
    code = hashlib.sha256(b"".join(Path(m.__file__).read_bytes() for m in (pnn, __import__("ai_trader.field.cards", fromlist=["x"]), __import__("ai_trader.live.nn", fromlist=["x"])))).hexdigest()[:16]
    line = {
        "id": f"nn-{S:%Y%m%dT%H}", "schema": VERSION, "logged_at": logged_at.isoformat(), "as_of": as_of.isoformat(), "symbol": SYMBOL,
        "forecast_hour": S.isoformat(), "data_max_ts": data_max.isoformat(), "last_bar_close": last_close.isoformat(),
        "close": cl.get(last_close), "flags": sorted(set(flags)), "alarms": alarms,
        "models": {"pnn_1h": n1.card(), "pnn_1d": n24.card(), "field": {k: fc.get(k) for k in ("file", "sha256", "fit_as_of")},
                   "hnn10": hn.get("model_id"), "code_sha256": code, "tool_version": VERSION},
        "inputs_hash": ihash, "inputs": {"1h": dict(zip(HOURLY_FEATURES, map(float, x_now), strict=True)),
                                         "24h": dict(zip(DAILY_FEATURES, map(float, x24), strict=True))},
        "net": net, "latent": lat, "coverage": cov,
        "aci_updates": {"1h_caught_up": n_caught, "24h_resolved": n_res, "24h_pending": len(st["24h"]["pending"])},
        "field": {k: v for k, v in fc.items() if k != "text"}, "hnn10": hn, "news": nw,
        "scored": [f"{s['id']}|{s['horizon']}" for s in new_scores],
    }
    st["as_of"] = as_of.isoformat()
    return _clean(line), st, persistable, [_clean(s) for s in new_scores]


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, datetime):
        return o.isoformat()
    return o


# ------------------------------------------------------------------ human summary


def _f(v, nd=2, sign=True) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def render_latest(line: dict, prev: dict | None, scores: list[dict]) -> str:
    n1, n24 = line["net"]["1h"], line["net"]["24h"]
    S = line["forecast_hour"][:16].replace("T", " ")
    L = [f"# BTC live network: bar {S} UTC (line {line['id']})",
         f"as_of {line['as_of'][:19]}Z, newest bar closed {line['last_bar_close'][11:16]}, close {line['close']:,.1f}. "
         f"Flags: {', '.join(line['flags']) or 'none'}. Research only, no direction call."]
    for tag, n in (("Next hour", n1), ("Next 24h", n24)):
        q = n["quantiles_pct"]
        a = n["aci"]
        L.append(f"\n## {tag} [{n['window_start'][5:16]} -> {n['window_end'][5:16]}], % log return")
        L.append(f"- PNN median {_f(q['q50'])}, 80% [{_f(q['q10'])}, {_f(q['q90'])}], 90% [{_f(q['q05'])}, {_f(q['q95'])}], sigma {_f(n['sigma_pct'], sign=False)}%")
        L.append("- ACI " + "; ".join(f"{k}% [{_f(v['lo_pct'])}, {_f(v['hi_pct'])}] alpha_t {_f(v['alpha_t'], 4, False)}" for k, v in a.items()))
        if tag == "Next hour":
            hn = line.get("hnn10", {})
            if hn.get("status") == "ok" and hn.get("sigma_next_hour_ml") is not None:
                L.append(f"- H-NN10 vol (bar {str(hn['forecast_for'])[11:16]}): ML {100 * hn['sigma_next_hour_ml']:.3f}% vs baseline "
                         f"{100 * hn['sigma_next_hour_baseline']:.3f}% (ratio {hn['ratio']:.2f})"
                         + ("" if str(hn["forecast_for"])[:13] == line["forecast_hour"][:13] else " - not this bar's line"))
            else:
                L.append("- H-NN10 vol: no BTC line for this bar logged by as_of")
        elif "construction" in n:
            L.append(f"- 24h net: {n['construction']}")
    fc = line.get("field", {})
    if "horizons" in fc:
        L.append(f"\n## Field card ({fc['file']}, regime {fc['cell']})")
        for k, h in fc["horizons"].items():
            q, t = h["quantiles_pct"], h["p_touch"]
            L.append(f"- {k}: 90% [{_f(q['q05'])}, {_f(q['q95'])}] med {_f(q['q50'])}; touch +1/-1 {100 * t['+1%']:.0f}/{100 * t['-1%']:.0f}%, "
                     f"+2/-2 {100 * t['+2%']:.0f}/{100 * t['-2%']:.0f}%, +3/-3 {100 * t['+3%']:.0f}/{100 * t['-3%']:.0f}%; P(jump) {100 * h['p_jump']:.0f}%")
        L.append("- variance drivers: " + ", ".join(f"{a} {b:+.2f}" for a, b in fc["drivers_D"]) + "; jump drivers: "
                 + ", ".join(f"{a} {b:+.2f}" for a, b in fc["drivers_lam"]))
    else:
        L.append(f"\n## Field card: {fc.get('status', 'n/a')}")
    lat = line["latent"]
    L.append("\n## Latent (1h net, 4 dims)")
    L.append(f"- z {[round(v, 2) for v in lat['z']]}, predicted next {[round(v, 2) for v in lat['zhat_next']]}")
    if "last_error" in lat:
        e = lat["last_error"]
        L.append(f"- last hour: model error {e['model_rmse']:.2f} vs persistence {e['persistence_rmse']:.2f} (latent sd); "
                 + (f"168h R^2 gain {lat['r2_gain_168']['gain']:+.3f}" if "r2_gain_168" in lat else ""))
    nw = line.get("news", {})
    alt = nw.get("alt")
    alt_s = alt if isinstance(alt, str) else ", ".join(f"{k} {v:+.2f}" for k, v in alt.items() if v is not None and not k.startswith("raw_"))
    rss = nw.get("rss")
    rss_s = rss if isinstance(rss, str) else f"RSS items last 1h {rss['items_last_1h']}, 24h {rss['items_last_24h_excl_backlog']}" + (" (polling stale)" if rss["polling_stale"] else "")
    L.append(f"\n## News\n- alt: {alt_s or 'no values'}\n- {rss_s}")
    cv = line["coverage"]
    L.append("\n## Running coverage (ACI, last 72 / 168 resolved)")
    for h in ("1h", "24h"):
        L.append(f"- {h}: " + "; ".join(f"{lv}% {_f(cv[f'{h}_{lv}_72'].get('cov'), 3, False)} / {_f(cv[f'{h}_{lv}_168'].get('cov'), 3, False)}"
                                        for lv in LEVELS) + f" (n {cv[f'{h}_80_168'].get('n', 0)})")
    if scores:
        L.append("\n## Scored this run")
        for s in scores[-4:]:
            L.append(f"- {s['id']} {s['horizon']}: r {_f(s['r_pct'], 3)}% (z {_f(s['z'])}), in ACI80 {s['hit_aci80']}, pinball {s['pinball7']:.4f}")
    L.append("\n## Alarms")
    L += [f"- {a}" for a in line["alarms"]] or ["- none"]
    if prev:
        p1, p24 = prev["net"]["1h"], prev["net"]["24h"]
        ch = [f"1h sigma {_f(p1['sigma_pct'], sign=False)} -> {_f(n1['sigma_pct'], sign=False)}%",
              f"24h sigma {_f(p24['sigma_pct'], sign=False)} -> {_f(n24['sigma_pct'], sign=False)}%",
              f"1h alpha80 {_f(p1['aci']['80']['alpha_t'], 4, False)} -> {_f(n1['aci']['80']['alpha_t'], 4, False)}"]
        pf, cf = prev.get("field", {}).get("horizons", {}).get("1h"), fc.get("horizons", {}).get("1h")
        if pf and cf:
            ch.append(f"field 1h P(jump) {100 * pf['p_jump']:.1f} -> {100 * cf['p_jump']:.1f}%")
        new_al = [a for a in line["alarms"] if a not in prev.get("alarms", [])]
        L.append(f"\n## Changed vs previous line ({prev['id']})\n- " + "; ".join(ch) + (f"\n- new alarms: {len(new_al)}" if new_al else ""))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ write


def write_outputs(paths: Paths, line: dict, st: dict, persistable: bool, scores: list[dict], *, force: bool = False) -> dict:
    out = {"journal": None, "scores": 0, "state": False, "latest": None}
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
    if persistable:
        save_state(paths, st)
        out["state"] = True
    prev = [r for r in read_lines(paths) if r["forecast_hour"] < line["forecast_hour"]]
    paths.latest_md.parent.mkdir(parents=True, exist_ok=True)
    paths.latest_md.write_text(render_latest(line, prev[-1] if prev else None, scores), encoding="utf-8")
    out["latest"] = str(paths.latest_md)
    return out


__all__ = ["LEVELS", "VERSION", "OnlineACI", "Paths", "ShiftedLake", "daily_row", "hourly_frame", "load_net", "pinball", "realised",
           "render_latest", "run_line", "score_line", "write_outputs"]
