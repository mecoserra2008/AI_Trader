"""The live 'brain' of the frozen 1h BTC network: browser export, intra-hour nowcast inputs, recorder and H-BR1 checkpoints.

Research only. Reads the journals and the frozen models; writes only under data/journal/brain/ (snapshots, checkpoints)
and data/models/pnn/brain_1h.json (the export). Never trades.

Network export (`export_net`)
  Both seeds of the live v1 1h network (data/models/pnn/wf_1h/<refit_month>.pt, refit month from frozen_1h.json):
  per seed enc (4 x 19 x 4 Chebyshev coefficients, 4 x 19 residual SiLU weights, 4 biases) and head (7 x 4 x 4, 7 x 4, 7),
  the fit-block standardiser (mu, sd), clip 6, scale 0.5, the latent dynamics and the feature names. The browser runs
  the same forward pass in float64 (brain_core.js); `forward` here is the float64 numpy reference. The live line uses
  torch float32, so stored quantiles agree to ~1e-6 (tests check JS == numpy to 1e-9 and both == stored lines to 2e-6).

Intra-hour nowcast (`nowcast_inputs`, recipe brain_recipe.v1, the pre-declared H-BR1 recipe)
  Forecast bar S = the newest v1 line's forecast_hour; P0 = the line's close (close of bar S-1 = open of S);
  f = elapsed share of the hour. From ticks / closed 1m candles with S <= t <= now only:
    P = last price, H / L = high / low since S (P0 included), r_p = ln(P / P0).
  Return terms (scaled to an hourly equivalent: under Brownian scaling r_p / sqrt(f) has the law of a full-hour return)
    u = r_p^2 / (f s2),  s2 = exp(lsig2) (the frozen base variance)
    lu_1   := ln(max(u, 1e-4))                          (the partial hour replaces the last bar's return)
    lu_m6  := ln(exp(lu_m6) + (u - exp(lu_1_frozen)) / 6)      (its 1/6 share of the 6 h mean replaced)
    lu_m24 := ln(exp(lu_m24) + (u - exp(lu_1_frozen)) / 24)    (its 1/24 share of the 24 h mean replaced)
  Range terms (no time scaling: a jump is a size, not a rate)
    J = 1{ max(ln H/P0, ln P0/L) > 4 s_pre },  s_pre = exp(lsig2 / 2)
    jump24 := max(jump24, J);  since_jump := 1/168 if J else frozen
  Everything else (lsig2, lfac, fund_*, lu_2..lu_24, har_*, dow_*) stays frozen until the next close.
  Warm-up: for f < 5 min the nowcast equals the frozen inputs (too little of the hour to scale).
  Remaining-hour quantiles (% log return from the current price to the hour's close):
    nowcast  100 q_now sqrt(s2) sqrt(1 - f);   frozen  100 q_frozen sqrt(s2) sqrt(1 - f)    (H-BR1's comparison)
  Labelled everywhere as an intra-hour nowcast, not the official hourly forecast.

Recorder (`Recorder`, run in a daemon thread of the dashboard server or by scripts/brain_recorder.py; one at a time by
a heartbeat lock): once a minute it pulls Bybit public REST 1m klines, rebuilds the nowcast at the close of the newest
closed minute and appends a snapshot (inputs, edge outputs, node activations) to data/journal/brain/<date>.jsonl when
the inputs changed. Limits: >= 1 s between snapshots, <= MAX_DAY_BYTES per day file (~2.6 KB a snapshot, ~3.8 MB/day).
H-BR1 checkpoints at minutes 15 / 30 / 45 go to data/journal/brain/checkpoints.jsonl, once per (hour, minute), and are
refused when the evaluation would be later than LATE_S after the checkpoint (or the base line is not for this hour).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

VERSION = "brain.v1"
RECIPE_VERSION = "brain_recipe.v1"
EXPORT_SCHEMA = "brain_net.v1"
TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QNAMES = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
NORM80 = 2.5631031310892007
HOUR_MS = 3_600_000
F_WARM = 5 / 60
U_FLOOR = 1e-4
JUMP_Z = 4.0
CHECKPOINT_MIN = (15, 30, 45)
LATE_S = 120.0
MIN_GAP_S = 1.0
MAX_DAY_BYTES = 45_000_000
SCORE_MIN_HOURS = 300
LOCK_STALE_S = 180.0
LOOP_S = 60.0
SIG = 7  # significant digits stored in snapshots

FAMILY = {
    "volatility": ("lsig2", "lfac", "har_d", "har_w", "har_m"),
    "returns": ("lu_1", "lu_2", "lu_3", "lu_6", "lu_12", "lu_24", "lu_m6", "lu_m24"),
    "jumps": ("jump24", "since_jump"),
    "funding": ("fund_after", "fund_before"),
    "calendar": ("dow_sin", "dow_cos"),
}
FEATURE_FAMILY = {c: fam for fam, cs in FAMILY.items() for c in cs}

RECIPE = [
    {"feature": "lsig2", "changes": False, "how": "frozen: log base variance s2 of bar S (robust vol of bars <= S-1); also the scale of every quantile"},
    {"feature": "lfac", "changes": False, "how": "frozen: hour-of-week seasonal variance factor of bar S"},
    {"feature": "fund_after", "changes": False, "how": "frozen: calendar flag, bar S opens at 00/08/16 UTC"},
    {"feature": "fund_before", "changes": False, "how": "frozen: calendar flag, bar S opens at 23/07/15 UTC"},
    {"feature": "lu_1", "changes": True, "how": "ln(max(u, 1e-4)), u = r_p^2 / (f s2): the partial-hour return scaled to an hourly equivalent replaces the last bar's return"},
    {"feature": "lu_2", "changes": False, "how": "frozen: squared return of bar S-2 / s2"},
    {"feature": "lu_3", "changes": False, "how": "frozen: bar S-3"},
    {"feature": "lu_6", "changes": False, "how": "frozen: bar S-6"},
    {"feature": "lu_12", "changes": False, "how": "frozen: bar S-12"},
    {"feature": "lu_24", "changes": False, "how": "frozen: bar S-24"},
    {"feature": "lu_m6", "changes": True, "how": "ln(exp(lu_m6) + (u - exp(lu_1 frozen)) / 6): the last bar's 1/6 share of the 6 h mean replaced by u"},
    {"feature": "lu_m24", "changes": True, "how": "ln(exp(lu_m24) + (u - exp(lu_1 frozen)) / 24): the last bar's 1/24 share replaced by u"},
    {"feature": "jump24", "changes": True, "how": "max(frozen, J), J = 1 if the partial-hour range from the open exceeds 4 s_pre (max(ln H/P0, ln P0/L) > 4 s_pre; unscaled)"},
    {"feature": "since_jump", "changes": True, "how": "1/168 if J (a jump in the current hour) else frozen"},
    {"feature": "har_d", "changes": False, "how": "frozen: yesterday's log RV minus log(24 s2)"},
    {"feature": "har_w", "changes": False, "how": "frozen: 7-day mean log RV minus log(24 s2)"},
    {"feature": "har_m", "changes": False, "how": "frozen: 30-day mean log RV minus log(24 s2)"},
    {"feature": "dow_sin", "changes": False, "how": "frozen: day of week of bar S"},
    {"feature": "dow_cos", "changes": False, "how": "frozen: day of week of bar S"},
]
for _r in RECIPE:
    _r["family"] = FEATURE_FAMILY[_r["feature"]]
CHANGING = tuple(r["feature"] for r in RECIPE if r["changes"])
NOWCAST_LABEL = "intra-hour nowcast, not the official hourly forecast"


def _utc(t) -> datetime:
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


def _ms(t: datetime) -> int:
    return round(_utc(t).timestamp() * 1000)


def _r(v, sig: int = SIG):
    if isinstance(v, list | tuple | np.ndarray):
        return [_r(x, sig) for x in v]
    v = float(v)
    return float(f"{v:.{sig}g}") if math.isfinite(v) else None


# ------------------------------------------------------------------ export


def export_net(pnn_dir: Path | str) -> dict:
    """Both seeds of the live 1h network as plain JSON (the browser's model file)."""
    import torch

    torch.set_num_threads(1)
    d = Path(pnn_dir)
    fz = d / "frozen_1h.json"
    spec = json.loads(fz.read_text(encoding="utf-8"))
    pt = d / "wf_1h" / f"{spec['refit_month']}.pt"
    ck = torch.load(pt, weights_only=False)
    mu, sd = [float(v) for v in ck["std"]["mu"]], [float(v) for v in ck["std"]["sd"]]
    if not np.allclose(mu, spec["std_mu"]) or not np.allclose(sd, spec["std_sd"]):
        raise RuntimeError("frozen_1h.json and the checkpoint disagree on the standardiser")

    def arr(t):
        return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t, dtype=np.float64).tolist()

    seeds = []
    for st in ck["pnn"]:
        seeds.append({lay: {"coef": arr(st[f"{lay}.coef"]), "w_res": arr(st[f"{lay}.w_res"]), "bias": arr(st[f"{lay}.bias"])}
                      for lay in ("enc", "head")})
    dy = ck["dyn"]
    h = hashlib.sha256()
    h.update(pt.read_bytes())
    return {"schema": EXPORT_SCHEMA, "horizon": "1h", "refit_month": spec["refit_month"], "pt": f"wf_1h/{pt.name}",
            "pt_sha256": h.hexdigest()[:16], "features": list(spec["features"]),
            "family": {c: FEATURE_FAMILY[c] for c in spec["features"]}, "std_mu": mu, "std_sd": sd,
            "clip": float(spec["clip"]), "scale": float(spec["scale"]), "degree": int(spec["degree"]), "taus": list(TAUS),
            "latent": int(ck["pnn_meta"]["latent"]), "seeds": seeds,
            "dynamics": {k: np.asarray(dy[k], dtype=np.float64).tolist() for k in ("A", "B", "c", "zmu", "zsd")},
            "head_rule": spec.get("head_rule"), "ensemble": "mean of the seeds' quantiles (model units y = r / sqrt(s2))",
            "edge": "phi(x) = w_res silu(x) + sum_{d=1..4} c_d T_d(tanh(scale x)); node = bias + sum of its edges"}


STATS_FROM = "2025-01-01"


def panel_path(pnn_dir: Path | str) -> Path | None:
    ps = sorted((Path(pnn_dir) / "cache").glob("btc_1h_*.parquet"))
    return ps[-1] if ps else None


def unit_stats(net: dict, panel: Path | str | None, start: str = STATS_FROM) -> dict | None:
    """Mean / sd of every unit and edge of the network over the hourly panel from `start` (the page's firing scales)."""
    if panel is None or not Path(panel).is_file():
        return None
    import polars as pl

    d = pl.read_parquet(panel).filter(pl.col("event_ts") >= datetime.fromisoformat(start).replace(tzinfo=UTC))
    X = np.column_stack([d[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in net["features"]])
    X = X[np.isfinite(X).all(axis=1)]
    fb = forward_batch(net, X)
    sd = lambda a: np.maximum(a.std(axis=0), 1e-9).tolist()
    return {"panel": Path(panel).name, "from": start, "n": len(X), "x_mu": fb["x"].mean(axis=0).tolist(), "x_sd": sd(fb["x"]),
            "z_mu": [s["z"].mean(axis=0).tolist() for s in fb["seeds"]], "z_sd": [sd(s["z"]) for s in fb["seeds"]],
            "o_mu": [s["o"].mean(axis=0).tolist() for s in fb["seeds"]], "o_sd": [sd(s["o"]) for s in fb["seeds"]],
            "enc_edge_sd": [sd(s["enc_edges"]) for s in fb["seeds"]], "head_edge_sd": [sd(s["head_edges"]) for s in fb["seeds"]],
            "q_mu": fb["q"].mean(axis=0).tolist(), "q_sd": sd(fb["q"])}


def net_path(pnn_dir: Path | str) -> Path:
    return Path(pnn_dir) / "brain_1h.json"


def write_net(pnn_dir: Path | str) -> Path:
    p = net_path(pnn_dir)
    net = export_net(pnn_dir)
    net["unit_stats"] = unit_stats(net, panel_path(pnn_dir))
    p.write_text(json.dumps(net), encoding="utf-8")
    return p


@lru_cache(maxsize=2)
def _load_net_cached(path: str, mtime: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_net(pnn_dir: Path | str) -> dict:
    """The exported net; (re)exported when missing or older than the checkpoint / frozen spec."""
    d = Path(pnn_dir)
    p = net_path(d)
    spec = json.loads((d / "frozen_1h.json").read_text(encoding="utf-8"))
    pt = d / "wf_1h" / f"{spec['refit_month']}.pt"
    if not p.is_file() or p.stat().st_mtime < max(pt.stat().st_mtime, (d / "frozen_1h.json").stat().st_mtime):
        write_net(d)
    net = _load_net_cached(str(p), p.stat().st_mtime_ns)
    if net.get("refit_month") != spec["refit_month"]:
        write_net(d)
        net = _load_net_cached(str(p), p.stat().st_mtime_ns)
    return net


# ------------------------------------------------------------------ float64 forward pass (reference for brain_core.js)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softplus(x):
    x = np.asarray(x, float)
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))  # torch's threshold 20


def _basis(x, scale: float, degree: int) -> np.ndarray:
    u = np.tanh(scale * x)
    ts = [u, 2 * u * u - 1]
    for _ in range(2, degree):
        ts.append(2 * u * ts[-1] - ts[-2])
    return np.stack(ts[:degree], axis=-1)  # (n_in, D)


def _layer(lay: dict, x: np.ndarray, scale: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    coef, w, b = (np.asarray(lay[k], float) for k in ("coef", "w_res", "bias"))
    E = np.einsum("id,oid->oi", _basis(x, scale, degree), coef) + _silu(x)[None, :] * w  # (out, in)
    return E, b + E.sum(axis=1)


def quantiles_from_raw(o: np.ndarray) -> np.ndarray:
    sp = _softplus
    q50 = o[3]
    q75 = q50 + sp(o[4])
    q90 = q75 + sp(o[5])
    q95 = q90 + sp(o[6])
    q25 = q50 - sp(o[2])
    q10 = q25 - sp(o[1])
    q05 = q10 - sp(o[0])
    return np.array([q05, q10, q25, q50, q75, q90, q95], float)


def standardise(net: dict, raw) -> np.ndarray:
    raw = np.asarray(raw, float)
    z = (raw - np.asarray(net["std_mu"], float)) / np.asarray(net["std_sd"], float)
    return np.clip(np.nan_to_num(z, nan=0.0), -net["clip"], net["clip"])


def forward(net: dict, raw) -> dict:
    """Float64 forward pass, decomposed per edge. raw: the 19 inputs (feature order of the net)."""
    x = standardise(net, raw)
    sc, D = float(net["scale"]), int(net["degree"])
    seeds, qs = [], []
    for s in net["seeds"]:
        E1, z = _layer(s["enc"], x, sc, D)
        E2, o = _layer(s["head"], z, sc, D)
        q = quantiles_from_raw(o)
        seeds.append({"enc_edges": E1, "z": z, "head_edges": E2, "o": o, "q": q})
        qs.append(q)
    return {"x": x, "seeds": seeds, "q": np.mean(qs, axis=0), "missing": ~np.isfinite(np.asarray(raw, float))}


def forward_batch(net: dict, raw: np.ndarray) -> dict:
    """Vectorised forward over rows (N, 19): x, per-seed z (N, 4), o (N, 7), q (N, 7), enc edges (N, 4, 19), ensemble q."""
    X = np.asarray(raw, float)
    Z = (X - np.asarray(net["std_mu"], float)) / np.asarray(net["std_sd"], float)
    Z = np.clip(np.nan_to_num(Z, nan=0.0), -net["clip"], net["clip"])
    sc, D = float(net["scale"]), int(net["degree"])

    def lay(L, x):
        coef, w, b = (np.asarray(L[k], float) for k in ("coef", "w_res", "bias"))
        u = np.tanh(sc * x)
        ts = [u, 2 * u * u - 1]
        for _ in range(2, D):
            ts.append(2 * u * ts[-1] - ts[-2])
        B = np.stack(ts[:D], axis=-1)  # (N, in, D)
        E = np.einsum("nid,oid->noi", B, coef) + _silu(x)[:, None, :] * w[None]
        return E, b[None, :] + E.sum(axis=2)

    out = {"x": Z, "seeds": []}
    qs = []
    for s in net["seeds"]:
        E1, z = lay(s["enc"], Z)
        E2, o = lay(s["head"], z)
        sp = _softplus
        q50 = o[:, 3]
        q75 = q50 + sp(o[:, 4]); q90 = q75 + sp(o[:, 5]); q95 = q90 + sp(o[:, 6])
        q25 = q50 - sp(o[:, 2]); q10 = q25 - sp(o[:, 1]); q05 = q10 - sp(o[:, 0])
        q = np.stack([q05, q10, q25, q50, q75, q90, q95], axis=1)
        out["seeds"].append({"enc_edges": E1, "z": z, "head_edges": E2, "o": o, "q": q})
        qs.append(q)
    out["q"] = np.mean(qs, axis=0)
    return out


# ------------------------------------------------------------------ base (frozen inputs at the last close) and recipe


def base_from_line(line: dict, net: dict) -> dict:
    """The frozen top-of-hour state the nowcast starts from: exactly the v1 line's stored 1h inputs and quantiles."""
    feats = net["features"]
    raw = [line["inputs"]["1h"].get(c) for c in feats]
    n1 = line["net"]["1h"]
    x = np.array([np.nan if v is None else float(v) for v in raw], float)
    lsig2 = x[feats.index("lsig2")]
    s2 = math.exp(lsig2) if math.isfinite(lsig2) else float("nan")
    sref = (n1.get("sigma_ref_pct") or float("nan")) / 100
    fw = forward(net, x)
    S = _utc(line["forecast_hour"])
    return {"schema": VERSION, "recipe_version": RECIPE_VERSION, "label": NOWCAST_LABEL, "line_id": line["id"],
            "forecast_hour": S.isoformat(), "window_end": (S + timedelta(hours=1)).isoformat(), "S_ms": _ms(S),
            "logged_at": line.get("logged_at"), "flags": line.get("flags", []), "p0": line.get("close"),
            "features": feats, "inputs": raw, "s2": s2, "s_pre": math.sqrt(s2) if math.isfinite(s2) else None,
            "sigma_ref": sref, "sigma_ref_pct": n1.get("sigma_ref_pct"),
            "frozen_quantiles_pct": [n1["quantiles_pct"].get(k) for k in QNAMES],
            "frozen_quantiles_model": fw["q"].tolist(), "frozen_sigma_pct": n1.get("sigma_pct"),
            "aci": n1.get("aci"), "latent_stored": line.get("latent", {}).get("z"),
            "recipe": RECIPE, "changing": list(CHANGING), "families": {k: list(v) for k, v in FAMILY.items()},
            "constants": {"f_warm": F_WARM, "u_floor": U_FLOOR, "jump_z": JUMP_Z, "checkpoints_min": list(CHECKPOINT_MIN)}}


def partial_state(p0: float, S_ms: int, now_ms: int, ticks=(), candles=()) -> dict:
    """Last price, high and low of the forecast hour from data with S <= t <= now only.
    ticks: [(ts_ms, price)]; candles: Bybit 1m rows [start_ms, open, high, low, close, ...] - a candle is used only once it
    has closed (start + 60 s <= now); its close time is start + 60 s."""
    P, H, L, t_last, n = float(p0), float(p0), float(p0), S_ms, 0
    ev = []
    for c in candles:
        st = int(c[0])
        if st >= S_ms and st + 60_000 <= now_ms and st < S_ms + HOUR_MS:
            ev.append((st + 60_000, float(c[4]), float(c[2]), float(c[3])))
    for ts, p in ticks:
        ts = int(ts)
        if S_ms <= ts <= now_ms and ts <= S_ms + HOUR_MS:
            ev.append((ts, float(p), float(p), float(p)))
    ev.sort(key=lambda e: e[0])
    for ts, p, hi, lo in ev:
        if not (p > 0 and hi > 0 and lo > 0):
            continue
        P, t_last, n = p, ts, n + 1
        H, L = max(H, hi, p), min(L, lo, p)
    return {"P": P, "H": H, "L": L, "t_last_ms": t_last, "n": n}


def nowcast_inputs(base: dict, state: dict, now_ms: int) -> tuple[list[float], dict]:
    """Recipe brain_recipe.v1: the 19 raw inputs at now from the frozen base and the partial-hour state."""
    feats = base["features"]
    x = [np.nan if v is None else float(v) for v in base["inputs"]]
    idx = {c: i for i, c in enumerate(feats)}
    f = (now_ms - base["S_ms"]) / HOUR_MS
    f = min(max(f, 0.0), 1.0)
    p0 = float(base["p0"])
    info = {"f": f, "warm": f < F_WARM, "P": state["P"], "H": state["H"], "L": state["L"], "r_p": math.log(state["P"] / p0),
            "range": math.log(state["H"] / state["L"]), "u": None, "J": 0}
    if f < F_WARM:
        return x, info
    s2 = base["s2"]
    u = info["r_p"] ** 2 / (f * s2)
    u_last = math.exp(x[idx["lu_1"]])
    x[idx["lu_1"]] = math.log(max(u, U_FLOOR))
    x[idx["lu_m6"]] = math.log(max(math.exp(x[idx["lu_m6"]]) + (u - u_last) / 6, 1e-12))
    x[idx["lu_m24"]] = math.log(max(math.exp(x[idx["lu_m24"]]) + (u - u_last) / 24, 1e-12))
    exc = max(math.log(state["H"] / p0), math.log(p0 / state["L"]))
    J = 1 if exc > JUMP_Z * math.sqrt(s2) else 0
    if J:
        x[idx["jump24"]] = max(x[idx["jump24"]], 1.0)
        x[idx["since_jump"]] = 1 / 168
    info.update({"u": u, "J": J, "excursion": exc})
    return x, info


def remaining_pct(q_model, base: dict, f: float) -> list[float]:
    """% log return quantiles from the current price to the hour's close."""
    k = 100 * math.sqrt(base["s2"]) * math.sqrt(max(1.0 - f, 0.0))
    return [float(k * v) for v in q_model]


# ------------------------------------------------------------------ recorder (snapshots + H-BR1 checkpoints)


def brain_dir(root: Path | str) -> Path:
    return Path(root) / "data" / "journal" / "brain"


def snapshot(base: dict, net: dict, x: list[float], info: dict, t_ms: int, source: str) -> dict:
    fw = forward(net, x)
    f = info["f"]
    return {"t": datetime.fromtimestamp(t_ms / 1000, UTC).isoformat(), "t_ms": t_ms, "line_id": base["line_id"],
            "S": base["forecast_hour"], "f": _r(f), "source": source, "recipe": RECIPE_VERSION, "warm": bool(info["warm"]),
            "price": info["P"], "high": info["H"], "low": info["L"], "r_p": _r(info["r_p"]), "u": _r(info["u"]) if info["u"] is not None else None,
            "J": info["J"], "inputs": _r(x), "x": _r(fw["x"]),
            "seeds": [{"enc_edges": _r(s["enc_edges"].tolist()), "z": _r(s["z"]), "head_edges": _r(s["head_edges"].tolist()),
                       "o": _r(s["o"]), "q": _r(s["q"])} for s in fw["seeds"]],
            "q_model": _r(fw["q"]), "q_rem_pct": _r(remaining_pct(fw["q"], base, f)),
            "q_rem_frozen_pct": _r(remaining_pct(base["frozen_quantiles_model"], base, f))}


class SnapshotWriter:
    """Append-only daily jsonl with a rate limit (MIN_GAP_S) and a per-day size cap (MAX_DAY_BYTES)."""

    def __init__(self, root: Path | str, min_gap_s: float = MIN_GAP_S, max_day_bytes: int = MAX_DAY_BYTES):
        self.dir = brain_dir(root)
        self.min_gap_s, self.max_day_bytes = min_gap_s, max_day_bytes
        self.last_t_ms: int | None = None
        self.last_inputs: list | None = None
        self.refused = {"rate": 0, "size": 0, "unchanged": 0}

    def path(self, t_ms: int) -> Path:
        return self.dir / f"{datetime.fromtimestamp(t_ms / 1000, UTC):%Y-%m-%d}.jsonl"

    def write(self, snap: dict) -> str:
        t = int(snap["t_ms"])
        if self.last_inputs is not None and snap["inputs"] == self.last_inputs and snap["line_id"] == getattr(self, "_last_line", None):
            self.refused["unchanged"] += 1
            return "unchanged"
        if self.last_t_ms is not None and t - self.last_t_ms < self.min_gap_s * 1000:
            self.refused["rate"] += 1
            return "rate"
        p = self.path(t)
        body = (json.dumps(snap, separators=(",", ":")) + "\n").encode("utf-8")
        cur = p.stat().st_size if p.is_file() else 0
        if cur + len(body) > self.max_day_bytes:
            self.refused["size"] += 1
            return "size"
        self.dir.mkdir(parents=True, exist_ok=True)
        with p.open("ab") as fh:
            fh.write(body)
        self.last_t_ms, self.last_inputs, self._last_line = t, snap["inputs"], snap["line_id"]
        return "ok"


def checkpoints_path(root: Path | str) -> Path:
    return brain_dir(root) / "checkpoints.jsonl"


def read_checkpoints(root: Path | str) -> list[dict]:
    p = checkpoints_path(root)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def log_checkpoint(root: Path | str, base: dict, net: dict, minute: int, state: dict, eval_ms: int, now_ms: int,
                   logged: set | None = None) -> tuple[str, dict | None]:
    """H-BR1 checkpoint at S + minute: refused when late (now - checkpoint > LATE_S), early, for another hour, already
    logged, or when the partial state is newer than the checkpoint (it must use data up to the checkpoint only)."""
    if minute not in CHECKPOINT_MIN:
        return "bad_minute", None
    cp_ms = base["S_ms"] + minute * 60_000
    if now_ms < cp_ms or eval_ms != cp_ms:
        return "early", None
    if now_ms - cp_ms > LATE_S * 1000:
        return "late", None
    if state["t_last_ms"] > cp_ms:
        return "lookahead", None
    key = (base["forecast_hour"], minute)
    done = logged if logged is not None else {(c["S"], c["minute"]) for c in read_checkpoints(root)}
    if key in done:
        return "duplicate", None
    x, info = nowcast_inputs(base, state, cp_ms)
    fw = forward(net, x)
    f = info["f"]
    row = {"id": f"br-{_utc(base['forecast_hour']):%Y%m%dT%H}-m{minute}", "schema": VERSION, "hypothesis": "H-BR1",
           "recipe": RECIPE_VERSION, "S": base["forecast_hour"], "minute": minute, "f": f, "line_id": base["line_id"],
           "checkpoint": datetime.fromtimestamp(cp_ms / 1000, UTC).isoformat(),
           "logged_at": datetime.fromtimestamp(now_ms / 1000, UTC).isoformat(), "lag_s": (now_ms - cp_ms) / 1000,
           "price": info["P"], "high": info["H"], "low": info["L"], "p0": base["p0"], "n_obs": state["n"],
           "inputs": _r(x), "q_model": _r(fw["q"]), "nowcast_rem_pct": dict(zip(QNAMES, _r(remaining_pct(fw["q"], base, f)), strict=True)),
           "frozen_rem_pct": dict(zip(QNAMES, _r(remaining_pct(base["frozen_quantiles_model"], base, f)), strict=True)),
           "u": info["u"], "J": info["J"]}
    p = checkpoints_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    done.add(key)
    return "ok", row


def bybit_1m(fetch, limit: int = 90) -> list[list]:
    d = fetch({"category": "linear", "symbol": "BTCUSDT", "interval": "1", "limit": limit})
    if d.get("retCode") not in (0, None):
        raise RuntimeError(f"bybit retCode {d.get('retCode')}: {d.get('retMsg')}")
    rows = [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in d.get("result", {}).get("list", [])]
    return sorted(rows)


class Recorder:
    """One step a minute: newest v1 line -> base; Bybit 1m klines -> partial state at the newest closed minute ->
    snapshot (if inputs changed) and any due H-BR1 checkpoint."""

    def __init__(self, root: Path | str, store, pnn_dir: Path | str, fetch, writer: SnapshotWriter | None = None):
        self.root, self.store, self.pnn_dir, self.fetch = Path(root), store, Path(pnn_dir), fetch
        self.writer = writer or SnapshotWriter(root)
        self.logged = {(c["S"], c["minute"]) for c in read_checkpoints(root)}
        self.status: dict = {"started": datetime.now(UTC).isoformat(), "steps": 0, "last": None, "checkpoints": {}, "errors": 0}

    def step(self, now_ms: int | None = None, candles: list | None = None) -> dict:
        now_ms = now_ms or int(time.time() * 1000)
        out: dict = {"now": datetime.fromtimestamp(now_ms / 1000, UTC).isoformat()}
        v1 = self.store.v1()
        if not v1:
            out["status"] = "no_line"
            return out
        line = v1[-1]
        S_ms = _ms(_utc(line["forecast_hour"]))
        if not (S_ms <= now_ms < S_ms + HOUR_MS):
            out["status"] = "line_not_current"
            return out
        net = load_net(self.pnn_dir)
        base = base_from_line(line, net)
        if base["p0"] is None or not (base["s2"] > 0):
            out["status"] = "base_incomplete"
            return out
        if candles is None:
            candles = bybit_1m(self.fetch)
        closed = [c for c in candles if c[0] >= S_ms and c[0] + 60_000 <= now_ms]
        eval_ms = (closed[-1][0] + 60_000) if closed else S_ms
        st = partial_state(base["p0"], S_ms, eval_ms, candles=candles)
        x, info = nowcast_inputs(base, st, eval_ms)
        out["snapshot"] = self.writer.write(snapshot(base, net, x, info, eval_ms, "rest_1m"))
        for m in CHECKPOINT_MIN:
            cp = S_ms + m * 60_000
            if (base["forecast_hour"], m) in self.logged or now_ms < cp:
                continue
            st_cp = partial_state(base["p0"], S_ms, cp, candles=candles)
            res, _ = log_checkpoint(self.root, base, net, m, st_cp, cp, now_ms, self.logged)
            out[f"checkpoint_{m}"] = res
            self.status["checkpoints"][res] = self.status["checkpoints"].get(res, 0) + 1
            if res == "late":
                self.logged.add((base["forecast_hour"], m))  # do not retry a missed checkpoint
        out["status"] = "ok"
        self.status["steps"] += 1
        self.status["last"] = out
        return out


# ------------------------------------------------------------------ H-BR1 scoring (forward only)


def pinball7(y: float, q: dict) -> float:
    return float(np.mean([max(t * (y - q[k]), (t - 1) * (y - q[k])) for k, t in zip(QNAMES, TAUS, strict=True)]))


def hour_closes(v1_lines: list[dict]) -> dict:
    """close at each hour boundary T (UTC iso) from the v1 lines: the line for forecast hour T stores the close of the bar
    that closed at T (its last_bar_close). Stale lines are skipped."""
    out = {}
    for ln in v1_lines:
        if "stale" in ln.get("flags", []) or ln.get("close") is None or not ln.get("last_bar_close"):
            continue
        T = _utc(ln["last_bar_close"])
        if T == _utc(ln["forecast_hour"]):
            out.setdefault(T.isoformat(), float(ln["close"]))
    return out


def score_checkpoints(cps: list[dict], closes: dict) -> list[dict]:
    """One row per resolved checkpoint: remaining-hour log return (%), pinball of nowcast and frozen, paired difference."""
    rows = []
    for c in cps:
        end = (_utc(c["S"]) + timedelta(hours=1)).isoformat()
        if end not in closes or not c.get("price"):
            continue
        y = 100 * math.log(closes[end] / float(c["price"]))
        pn, pf = pinball7(y, c["nowcast_rem_pct"]), pinball7(y, c["frozen_rem_pct"])
        rows.append({"id": c["id"], "S": c["S"], "minute": c["minute"], "day": c["S"][:10], "r_rem_pct": y, "pinball_nowcast": pn,
                     "pinball_frozen": pf, "d": pn - pf})
    return rows


def hbr1_summary(rows: list[dict], min_hours: int = SCORE_MIN_HOURS) -> dict:
    """Paired pinball difference (nowcast - frozen) per (hour, checkpoint); NW t on the daily means. A verdict only once
    at least `min_hours` distinct hours are resolved: supports iff mean < 0 and t < -2; contradicts iff t > 2;
    else inconclusive."""
    from ai_trader.research.evaluate import newey_west_t

    hours = sorted({r["S"] for r in rows})
    out: dict = {"n_rows": len(rows), "n_hours": len(hours), "min_hours": min_hours, "verdict": None}
    if not rows:
        return out
    d = np.array([r["d"] for r in rows], float)
    days = sorted({r["day"] for r in rows})
    daily = np.array([np.mean([r["d"] for r in rows if r["day"] == dd]) for dd in days], float)
    m, t, _n = newey_west_t(daily)
    out.update({"mean_d": float(d.mean()), "n_days": len(days), "nw_t_daily": t, "mean_daily": m,
                "by_minute": {str(mm): {"n": int(sum(1 for r in rows if r["minute"] == mm)),
                                        "mean_d": float(np.mean([r["d"] for r in rows if r["minute"] == mm])) if any(r["minute"] == mm for r in rows) else None}
                              for mm in CHECKPOINT_MIN},
                "mean_pinball_nowcast": float(np.mean([r["pinball_nowcast"] for r in rows])),
                "mean_pinball_frozen": float(np.mean([r["pinball_frozen"] for r in rows]))})
    if len(hours) >= min_hours and math.isfinite(t):
        out["verdict"] = "supports" if (out["mean_d"] < 0 and t < -2) else ("contradicts" if t > 2 else "inconclusive")
    return out


# ------------------------------------------------------------------ single-recorder lock and loop


def _lock_path(root) -> Path:
    return brain_dir(root) / "recorder.lock"


def acquire_lock(root, owner: str, now: float | None = None) -> bool:
    """Heartbeat lock: free if missing, ours, or older than LOCK_STALE_S."""
    p = _lock_path(root)
    now = now or time.time()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("owner") != owner and now - float(d.get("heartbeat", 0)) < LOCK_STALE_S:
            return False
    except (OSError, ValueError):
        pass
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"owner": owner, "pid": os.getpid(), "heartbeat": now}), encoding="utf-8")
    tmp.replace(p)
    return True


def run_loop(rec: Recorder, owner: str, stop: threading.Event, period_s: float = LOOP_S, offset_s: float = 4.0) -> None:
    """Wake a few seconds after each minute boundary; step only while holding the lock."""
    while not stop.is_set():
        now = time.time()
        nxt = (math.floor(now / period_s) + 1) * period_s + offset_s
        if stop.wait(max(0.5, nxt - now)):
            break
        try:
            if acquire_lock(rec.root, owner):
                rec.step()
        except Exception as e:  # noqa: BLE001 - network blips must not kill the recorder
            rec.status["errors"] += 1
            rec.status["last_error"] = f"{type(e).__name__}: {e}"[:200]


_REC: dict = {}


def ensure_recorder(app) -> dict:
    """Start the in-server recorder thread once (only with the real Bybit fetcher and AI_TRADER_BRAIN_RECORD != '0')."""
    from ai_trader.livedash import server as srv

    if os.environ.get("AI_TRADER_BRAIN_RECORD", "1") == "0" or app.fetch is not srv.bybit_fetch:
        return {"running": False, "reason": "disabled"}
    if "thread" not in _REC:
        rec = Recorder(app.store.root, app.store, app.store.pnn_dir, app.fetch)
        stop = threading.Event()
        th = threading.Thread(target=run_loop, args=(rec, f"server-{os.getpid()}", stop), name="brain-recorder", daemon=True)
        th.start()
        _REC.update(thread=th, rec=rec, stop=stop)
    return {"running": _REC["thread"].is_alive(), **_REC["rec"].status}


# ------------------------------------------------------------------ http routes (registered by one line in server.py)

STATIC = {"brain.html": "text/html; charset=utf-8", "brain.js": "text/javascript; charset=utf-8",
          "brain_core.js": "text/javascript; charset=utf-8"}


def handles(path: str) -> bool:
    return path in STATIC or path.startswith("api/brain/")


def recent_checkpoints(root, n: int = 24) -> list[dict]:
    return read_checkpoints(root)[-n:]


def route(app, h, path: str, q: dict):
    """Serve the brain page and its endpoints. h is the request handler (its _send / _json)."""
    if path in STATIC:
        f = app.static / path
        if not f.is_file():
            return h._json(404, {"error": f"{path} not found"})
        return h._send(200, f.read_bytes(), STATIC[path])
    if path == "api/brain/net":
        return h._json(200, load_net(app.store.pnn_dir))
    if path == "api/brain/base":
        rec = ensure_recorder(app)
        v1 = app.store.v1()
        if not v1:
            return h._json(404, {"error": "no v1 line yet"})
        net = load_net(app.store.pnn_dir)
        b = base_from_line(v1[-1], net)
        now_ms = int(time.time() * 1000)
        b["current"] = b["S_ms"] <= now_ms < b["S_ms"] + HOUR_MS
        b["server_now_ms"] = now_ms
        b["net_pt_sha256"] = net["pt_sha256"]
        b["recorder"] = rec
        b["checkpoints"] = recent_checkpoints(app.store.root, 9)
        return h._json(200, b)
    if path == "api/brain/checkpoints":
        return h._json(200, {"rows": recent_checkpoints(app.store.root, 72)})
    return h._json(404, {"error": "not found"})


__all__ = ["CHANGING", "CHECKPOINT_MIN", "FAMILY", "RECIPE", "Recorder", "SnapshotWriter", "base_from_line", "export_net", "forward",
           "forward_batch", "handles", "load_net", "log_checkpoint", "nowcast_inputs", "partial_state", "remaining_pct", "route"]
