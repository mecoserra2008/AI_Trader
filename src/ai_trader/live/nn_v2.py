"""Live BTC network v2: a parallel shadow of live_nn.v1 that differs only in the conformal layer (research only).

    run from scripts/live_nn.py after the v1 line is written (v1 is computed and written first, so it cannot depend on v2)
    PYTHONIOENCODING=utf-8 uv run --extra ml python scripts/live_nn.py --v2-replay-from 2026-11-13T20:00Z   (one-off replay)

Settings: config/live_nn_v2.yaml (hash of the parsed content in every line; pre-declared in the diary 2026-11-15).
  v2.1     bands are ISSUED at alpha_eff = max(alpha_t, nominal alpha / 4) on both horizons (alpha_t < 0 makes the ACI
           interval infinite); the ACI state keeps the unclipped alpha_t and updates on the pre-floor band as in v2.
  network  the same frozen 1h / 1d PNN checkpoints, features and construction as v1 (live.nn; tests check the
           quantiles are equal to v1's).
  ACI      the same Gibbs-Candes update and CQR scores as v1 with gamma 0.02 (1h) and 0.02 / 24 per resolved 24h
           forecast, warm-started from the same walk-forward stream (latent_<h>.parquet). The ACI stream is updated with
           the miscoverage of its own (pre-floor) band.
  floor    1h only. While a jump cluster is active (any of the bars S-6h .. S-1h had |r / sigma_eff| > 3, sigma_eff =
           (hi80 - lo80) / 2.5631 of the v2 ISSUED 80% band of that bar), each issued band's half-width is at least
           z_q x max(net sigma, field 1h sigma, H-NN10 ML sigma), widened symmetrically about the ACI band's midpoint.
Outputs: data/journal/live_nn_v2/<date of S>.jsonl (one line per forecast bar, id v21-<S> in v2.1), scores.jsonl in the same
directory, state data/models/pnn/live_aci_state_v2.json. Lines computed more than 18 min after S + 3 min are flagged
'replay' and count for no verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ai_trader.live import nn

VERSION = "live_nn.v2.1"
MODEL_ID = "live_nn.v2.1"
LEVELS = nn.LEVELS
Z_Q = {80: 1.2815515655446004, 90: 1.6448536269514722, 95: 1.959963984540054}
TAU_BAND = {80: (0.10, 0.90), 90: (0.05, 0.95), 95: (0.025, 0.975)}
ISSUED_KEEP = timedelta(hours=48)
ZHIST_KEEP = 48


# ------------------------------------------------------------------ config


def config_path(root: Path | None = None) -> Path:
    if root is None:
        from ai_trader.core.config import load

        root = load().root
    return Path(root) / "config" / "live_nn_v2.yaml"


def load_config(path: Path | None = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or config_path()).read_text(encoding="utf-8"))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Settings:
    gamma: dict
    window: dict
    lookback_h: int
    z_trigger: float
    cfg_hash: str
    alpha_min: dict = None   # level -> minimum alpha used to ISSUE a band (v2.1); the ACI state keeps the unclipped alpha_t

    @classmethod
    def from_config(cls, cfg: dict) -> Settings:
        a, j = cfg["aci"], cfg["jump_floor"]
        if cfg.get("version") != VERSION or [int(x) for x in a["levels"]] != list(LEVELS):
            raise ValueError("config/live_nn_v2.yaml does not describe live_nn.v2 at levels 80/90/95")
        div = float(a["alpha_floor"]["divisor"])
        return cls({"1h": float(a["gamma_1h"]), "24h": float(a["gamma_24h"])}, {"1h": int(a["window_1h"]), "24h": int(a["window_24h"])},
                   int(j["lookback_hours"]), float(j["z_trigger"]), config_hash(cfg), {L: (1 - L / 100) / div for L in LEVELS})


def q_at(o: nn.OnlineACI, alpha: float) -> float:
    """OnlineACI.Q() evaluated at another alpha (same rank rule)."""
    srt = sorted(o.scores)
    m = len(srt)
    if m == 0:
        return 0.0
    if alpha <= 0:
        return srt[-1]
    if alpha >= 1:
        return -math.inf
    k = int(math.ceil((m + 1) * (1 - alpha)))
    return srt[min(max(k, 1), m) - 1]


def issue_band(o: nn.OnlineACI, lo: float, hi: float, alpha_min: float) -> tuple[float, float, float, float]:
    """The issued conformal band at alpha_eff = max(alpha_t, alpha_min), model units: (lo, hi, Q, alpha_eff)."""
    ae = max(o.alpha_t, alpha_min)
    Q = q_at(o, ae)
    if not math.isfinite(Q):
        mid = 0.5 * (lo + hi)
        return mid, mid, Q, ae
    return lo - Q, hi + Q, Q, ae


# ------------------------------------------------------------------ paths


@dataclass
class PathsV2:
    root: Path
    models_root: Path | None = None
    config: Path | None = None

    @classmethod
    def default(cls) -> PathsV2:
        from ai_trader.core.config import load

        return cls(load().root)

    @property
    def v1(self) -> nn.Paths:
        return nn.Paths(self.root, self.models_root)

    @property
    def journal(self) -> Path:
        return self.root / "data" / "journal" / "live_nn_v2"

    @property
    def state(self) -> Path:
        return self.root / "data" / "models" / "pnn" / "live_aci_state_v2.json"

    @property
    def config_file(self) -> Path:
        return self.config or config_path(self.models_root or self.root)


# ------------------------------------------------------------------ floor / scoring maths


def floor_band(lo: float, hi: float, level: int, sigma_floor: float | None, active: bool) -> tuple[float, float, bool]:
    """Issued band (% units) from the ACI band: the half-width is at least z_q x sigma_floor while a cluster is active."""
    if not active or sigma_floor is None or not math.isfinite(sigma_floor) or sigma_floor <= 0:
        return lo, hi, False
    f = Z_Q[level] * sigma_floor
    if (hi - lo) / 2 >= f:
        return lo, hi, False
    c = 0.5 * (lo + hi)
    return c - f, c + f, True


def cluster(zhist: list, S: datetime, lookback_h: int, z_trigger: float) -> list:
    """[ts, z] of the bars S - lookback .. S - 1h with |z| > z_trigger (non-empty = cluster active)."""
    lo = S - timedelta(hours=lookback_h)
    return [[ts, z] for ts, _r, _s, z in zhist if z is not None and lo <= nn._utc(ts) < S and abs(z) > z_trigger]


def band_pinball(y: float, bands: dict) -> float:
    """Mean pinball loss of the 6 band endpoints {level: (lo, hi)} (taus 0.10/0.90, 0.05/0.95, 0.025/0.975)."""
    out = []
    for L in LEVELS:
        lo, hi = bands[L]
        for q, t in zip((lo, hi), TAU_BAND[L], strict=True):
            out.append(max(t * (y - q), (t - 1) * (y - q)))
    return float(np.mean(out))


# ------------------------------------------------------------------ state


def _ihist_from(a: nn.OnlineACI) -> list:
    return [[x[0], x[1]] for x in a.hist]


def new_state(paths: PathsV2, as_of: datetime, cfg: Settings) -> dict:
    """Warm start from the same source v1 used (F2's walk-forward stream), with v2's gamma."""
    out = {"schema": "live_aci_state.v2.1", "version": VERSION, "config_hash": cfg.cfg_hash, "created": datetime.now(UTC).isoformat(),
           "as_of": as_of.isoformat(), "warm": {}}
    for h, fn in (("1h", "latent_1h.parquet"), ("24h", "latent_1d.parquet")):
        a, last = nn._warm_from_parquet(paths.v1.pnn / fn, h, as_of)
        for o in a.values():
            o.gamma, o.window = cfg.gamma[h], cfg.window[h]
            if len(o.scores) > o.window:
                del o.scores[: len(o.scores) - o.window]
        out["warm"][h] = f"{fn} to {last}"
        if h == "1h":
            out["1h"] = {"last_ts": last, "aci": a, "issued": {}, "zhist": [], "ihist": {str(L): _ihist_from(a[L]) for L in LEVELS}}
        else:
            out["24h"] = {"aci": a, "pending": [], "last_resolved_end": None, "ihist": {str(L): _ihist_from(a[L]) for L in LEVELS}}
    return out


def load_state(paths: PathsV2) -> dict | None:
    if not paths.state.is_file():
        return None
    d = json.loads(paths.state.read_text(encoding="utf-8"))
    for h in ("1h", "24h"):
        d[h]["aci"] = {int(k): nn.OnlineACI.from_dict(v) for k, v in d[h]["aci"].items()}
    return d


def dump_state(st: dict) -> dict:
    return nn.dump_state(st)


def save_state(paths: PathsV2, st: dict) -> None:
    paths.state.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.state.with_suffix(".tmp")
    tmp.write_text(json.dumps(dump_state(st), default=str), encoding="utf-8")
    tmp.replace(paths.state)


def _cov(errs: list, level: int, n: int) -> dict:
    h = [e for _t, e in errs[-n:] if e is not None]
    if not h:
        return {"n": 0}
    cov = 1 - float(np.mean(h))
    p = level / 100
    half = 2.576 * math.sqrt(p * (1 - p) / len(h))
    return {"n": len(h), "cov": round(cov, 4), "nominal": p, "alarm": bool(abs(cov - p) > half)}


# ------------------------------------------------------------------ journal io


def read_lines(paths: PathsV2) -> list[dict]:
    rows = []
    for f in sorted(paths.journal.glob("20*.jsonl")) if paths.journal.is_dir() else []:
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return sorted(rows, key=lambda r: (r["forecast_hour"], r["logged_at"]))


def read_scores(paths: PathsV2) -> list[dict]:
    p = paths.journal / "scores.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.is_file() else []


def score_line(line: dict, h: str, r: float) -> dict:
    n = line["net"][h]
    q = n["quantiles_pct"]
    b = {L: (n["band"][str(L)]["lo_pct"], n["band"][str(L)]["hi_pct"]) for L in LEVELS}
    out = {"id": line["id"], "horizon": h, "window_start": n["window_start"], "window_end": n["window_end"], "r_pct": r,
           "sigma_pct": n["sigma_pct"], "sigma_eff_pct": n.get("sigma_eff_pct"),
           "z_eff": r / n["sigma_eff_pct"] if n.get("sigma_eff_pct") else None, "pinball7": nn.pinball(r, q, nn.TAU7),
           "pinball_bands": band_pinball(r, b), "floor_active": bool(n.get("floor", {}).get("active")), "flags": line.get("flags", [])}
    for L in LEVELS:
        out[f"hit_band{L}"] = b[L][0] <= r <= b[L][1]
        a = n["aci"][str(L)]
        out[f"hit_aci{L}"] = a["lo_pct"] is not None and a["lo_pct"] <= r <= a["hi_pct"]
        out[f"width{L}"] = b[L][1] - b[L][0]
    return out


# ------------------------------------------------------------------ the network (same as v1)


def network(as_of: datetime, paths: PathsV2, lake) -> dict:
    """The frozen networks' raw output at as_of, exactly as live.nn.run_line computes it (bars <= S-1 for bar S)."""
    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.frame import market_frame
    from ai_trader.research.pnn_study import DAILY_FEATURES

    flags: list[str] = []
    alarms: list[str] = []
    f1 = market_frame(nn.SYMBOL, "1h", as_of, funding=False, oi=False, lake=lake)
    f4 = market_frame(nn.SYMBOL, "4h", as_of, funding=False, oi=False, lake=lake)
    if f1.height < 24 * 120:
        raise RuntimeError("insufficient 1h history")
    data_max, last_close = nn._utc(f1["known_ts"].max()), nn._utc(f1["close_ts"].max())
    if as_of - last_close > nn.STALE_AFTER:
        flags.append("stale")
        alarms.append(f"stale data: newest bar closed {(as_of - last_close).total_seconds() / 60:.0f} min before as_of")
    H, S = nn.hourly_frame(f1)
    S = nn._utc(S)
    n1 = nn.load_net("1h", str(paths.v1.pnn))
    if n1.train_end > as_of:
        flags.append("pnn_1h_trained_after_as_of")
    tail = H.tail(24 * 10)
    X1 = nn._np(tail, HOURLY_FEATURES)
    ts1 = [nn._utc(t) for t in tail["event_ts"].to_list()]
    if ts1[-1] != S:
        raise RuntimeError("forecast row missing")
    if not np.isfinite(X1[-1]).all():
        flags.append("incomplete_features_1h")
    n24 = nn.load_net("1d", str(paths.v1.pnn))
    if n24.train_end > as_of:
        flags.append("pnn_1d_trained_after_as_of")
    dr = nn.daily_row(f1, f4, lake, S, as_of)
    if dr.get("ok"):
        x24 = np.array([dr.get(c, np.nan) if dr.get(c) is not None else np.nan for c in DAILY_FEATURES], float)
        sig24 = math.exp(0.5 * x24[DAILY_FEATURES.index("lrv_w")])
        q24 = n24.quantiles(x24)[0]
        if not np.isfinite(x24).all():
            flags.append("incomplete_features_24h")
    else:
        x24, sig24, q24 = np.full(len(DAILY_FEATURES), np.nan), float("nan"), np.full(7, np.nan)
        flags.append("no_daily_row")
    if S.hour:
        flags.append("rolling_day")

    def full():
        Xa = nn._np(H, HOURLY_FEATURES)
        return (n1.quantiles(Xa), H["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy(), H["r"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                [nn._utc(t) for t in H["event_ts"].to_list()])

    return {"S": S, "data_max": data_max, "last_close": last_close, "cl": nn.closes(f1), "flags": flags, "alarms": alarms,
            "q1": n1.quantiles(X1), "s2": tail["s2"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
            "r1": tail["r"].cast(pl.Float64).fill_null(np.nan).to_numpy(), "ts1": ts1, "x_now": X1[-1], "x24": x24, "q24": q24, "sig24": sig24,
            "full": full, "cards": {"pnn_1h": n1.card(), "pnn_1d": n24.card()}}


# ------------------------------------------------------------------ side inputs of the floor


def floor_components(sigma_net_pct: float | None, field: dict | None, hnn10: dict | None, S: datetime, as_of: datetime) -> tuple[dict, list[str]]:
    comps, notes = {"net": sigma_net_pct}, []
    if field and "horizons" in field and not field.get("fit_after_as_of"):
        comps["field"] = field["horizons"]["1h"]["sd_pct"]
    elif field and field.get("fit_after_as_of"):
        notes.append("floor_field_fit_after_as_of_omitted")
    if hnn10 and hnn10.get("status") == "ok" and hnn10.get("sigma_next_hour_ml") is not None and hnn10.get("forecast_for") \
            and nn._utc(hnn10["forecast_for"]) >= S - timedelta(hours=1) and nn._utc(hnn10["logged_at"]) <= as_of:
        comps["hnn10"] = 100 * float(hnn10["sigma_next_hour_ml"])
    return comps, notes


def _sigma_floor(comps: dict) -> float | None:
    v = [x for x in comps.values() if x is not None and math.isfinite(x) and x > 0]
    return max(v) if v else None


def _band_1h(st: dict, q: np.ndarray, sc: float, comps: dict, active: bool, alpha_min: dict) -> dict:
    """ACI bands (unclipped alpha_t: the stream the state updates on) and issued bands in % (alpha_eff = max(alpha_t,
    alpha_min), then the jump floor) for one bar, from the state before that bar is resolved."""
    aci, band = {}, {}
    sf = _sigma_floor(comps)
    for L in LEVELS:
        a, b = nn.BAND[L]
        o = st["1h"]["aci"][L]
        lo, hi, Q = o.band(q[a], q[b])
        lp, hp = 100 * lo * sc, 100 * hi * sc
        aci[str(L)] = {"lo_pct": lp, "hi_pct": hp, "Q": Q, "alpha_t": o.alpha_t}
        ilo, ihi, iQ, ae = issue_band(o, q[a], q[b], alpha_min[L])
        blo, bhi, fl = floor_band(100 * ilo * sc, 100 * ihi * sc, L, sf, active)
        band[str(L)] = {"lo_pct": blo, "hi_pct": bhi, "floored": fl, "alpha_eff": ae, "Q": iQ}
    sig_eff = (band["80"]["hi_pct"] - band["80"]["lo_pct"]) / nn.NORM80
    return {"aci": aci, "band": band, "sigma_eff_pct": sig_eff, "sigma_floor_pct": sf}


# ------------------------------------------------------------------ the run


def run_line(as_of: datetime, *, paths: PathsV2 | None = None, lake=None, state: dict | None = None, persist_state: bool = True,
             field: dict | None = None, with_field: bool = True, hnn10: dict | None = None, logged_at: datetime | None = None,
             cfg: Settings | None = None, flag_replay: bool = False) -> tuple[dict, dict, bool, list[dict]]:
    """Compute the v2 line. `field` / `hnn10`: the v1 line's blocks for the same as_of (saves recomputing the field card);
    computed here when None. Returns (line, state, persistable, new_scores). Writes nothing."""
    from ai_trader.ml import pnn
    from ai_trader.ml.hourly import HOURLY_FEATURES
    from ai_trader.research.pnn_study import DAILY_FEATURES
    from ai_trader.storage.lake import Lake

    paths = paths or PathsV2.default()
    lake = lake or Lake()
    cfg = cfg or Settings.from_config(load_config(paths.config_file))
    as_of = nn._utc(as_of)
    logged_at = logged_at or datetime.now(UTC)
    N = network(as_of, paths, lake)
    S, cl, last_close = N["S"], N["cl"], N["last_close"]
    flags, alarms = list(N["flags"]), list(N["alarms"])
    if flag_replay or logged_at - (S + timedelta(minutes=3)) > nn.REPLAY_AFTER:
        flags.append("replay")

    # ---- state
    persistable = persist_state
    st = state if state is not None else load_state(paths)
    if st is not None and st.get("config_hash") != cfg.cfg_hash:
        raise RuntimeError(f"v2 state was built with config {st.get('config_hash')}, config is now {cfg.cfg_hash}: a new version needs a new state")
    if st is not None and ((st["1h"]["last_ts"] and nn._utc(st["1h"]["last_ts"]) >= S) or any(nn._utc(p["window_start"]) > S for p in st["24h"]["pending"])):
        st, persistable = None, False
        flags.append("aci_state_ahead_of_as_of_rebuilt_in_memory")
    if st is None:
        st = new_state(paths, as_of, cfg)
        flags.append("aci_state_initialised")

    # ---- 1h catch-up over closed bars after last_ts
    last = nn._utc(st["1h"]["last_ts"]) if st["1h"]["last_ts"] else None
    ts1 = N["ts1"]
    if last is not None and last < ts1[0] - timedelta(hours=1):
        qa, s2a, ra, tsa = N["full"]()
    else:
        qa, s2a, ra, tsa = N["q1"], N["s2"], N["r1"], ts1
    s1 = st["1h"]
    n_caught, n_floored_catchup = 0, 0
    for i, t in enumerate(tsa):
        if t >= S or (last is not None and t <= last) or not np.isfinite(ra[i]) or not np.isfinite(qa[i]).all() or not (s2a[i] > 0):
            continue
        sc = math.sqrt(s2a[i])
        tk = t.isoformat()
        iss = s1["issued"].get(tk)
        if iss is None:   # no v2 line was issued for this bar: in-memory band, floor from the net and H-NN10 only
            sig_net = (100 * (qa[i, 5] - qa[i, 1]) * sc) / nn.NORM80
            comps, _ = floor_components(sig_net, None, nn.hnn10_line(paths.v1, t + timedelta(minutes=3)), t, t + timedelta(minutes=3))
            b = _band_1h(st, qa[i], sc, comps, bool(cluster(s1["zhist"], t, cfg.lookback_h, cfg.z_trigger)), cfg.alpha_min)
            iss = {"band": {k: [v["lo_pct"], v["hi_pct"]] for k, v in b["band"].items()}, "sigma_eff_pct": b["sigma_eff_pct"], "source": "catch_up"}
            n_floored_catchup += any(v["floored"] for v in b["band"].values())
        r_pct = 100 * ra[i]
        y = ra[i] / sc
        for L in LEVELS:
            a, bb = nn.BAND[L]
            s1["aci"][L].step(qa[i, a], qa[i, bb], y, tk)
            lo, hi = iss["band"][str(L)]
            s1["ihist"][str(L)].append([tk, float(not (lo <= r_pct <= hi))])
            del s1["ihist"][str(L)][: max(0, len(s1["ihist"][str(L)]) - nn.HIST)]
        se = iss["sigma_eff_pct"]
        s1["zhist"].append([tk, r_pct, se, r_pct / se if se and se > 0 else None])
        del s1["zhist"][: max(0, len(s1["zhist"]) - ZHIST_KEEP)]
        s1["last_ts"] = tk
        n_caught += 1
    s1["issued"] = {k: v for k, v in s1["issued"].items() if nn._utc(k) >= S - ISSUED_KEEP}

    # ---- 1h line
    q1, s2 = N["q1"], N["s2"]
    sc1 = math.sqrt(s2[-1]) if s2[-1] > 0 else float("nan")
    qd = nn._qdict(q1[-1], sc1)
    sig_net = (qd["q90"] - qd["q10"]) / nn.NORM80
    if field is None and with_field:
        try:
            field = nn.field_card(paths.v1, as_of, lake)
        except Exception as e:  # noqa: BLE001
            field = {"status": f"error: {type(e).__name__}: {e}"[:300]}
    hn = hnn10 if hnn10 is not None else nn.hnn10_line(paths.v1, as_of)
    comps, notes = floor_components(sig_net, field, hn, S, as_of)
    flags += notes
    trig = cluster(s1["zhist"], S, cfg.lookback_h, cfg.z_trigger)
    b1 = _band_1h(st, q1[-1], sc1, comps, bool(trig), cfg.alpha_min)
    net: dict = {"1h": {"window_start": S.isoformat(), "window_end": (S + nn.H_DELTA["1h"]).isoformat(), "quantiles_pct": qd, "sigma_pct": sig_net,
                        "sigma_ref_pct": 100 * sc1, "aci": b1["aci"], "band": b1["band"], "sigma_eff_pct": b1["sigma_eff_pct"],
                        "floor": {"active": bool(trig), "trigger": [[t, round(z, 2)] for t, z in trig], "components_pct": comps,
                                  "sigma_floor_pct": b1["sigma_floor_pct"], "applied": [L for L in LEVELS if b1["band"][str(L)]["floored"]]}}}
    if all(math.isfinite(v["lo_pct"]) for v in b1["band"].values()):
        s1["issued"][S.isoformat()] = {"band": {k: [v["lo_pct"], v["hi_pct"]] for k, v in b1["band"].items()}, "sigma_eff_pct": b1["sigma_eff_pct"],
                                       "source": "line"}
    if trig:
        flags.append("jump_cluster")

    # ---- 24h (same as v1, v2 gamma; no floor)
    q24, sig24 = N["q24"], N["sig24"]
    s24 = st["24h"]
    keep, n_res = [], 0
    for p in sorted(s24["pending"], key=lambda p: p["window_start"]):
        ws = nn._utc(p["window_start"])
        if ws + nn.H_DELTA["24h"] > last_close:
            keep.append(p)
            continue
        r = nn.realised(cl, ws, "24h")
        if r is None:
            if last_close - ws <= timedelta(hours=72):
                keep.append(p)
            continue
        y = r / 100 / p["sig"]
        for L in LEVELS:
            b, raw = p["band"][str(L)], p["raw"][str(L)]
            s24["aci"][L].resolve(float(not (b[0] <= y <= b[1])), max(raw[0] - y, y - raw[1]), ws.isoformat(), float(not (raw[0] <= y <= raw[1])))
            ib = p.get("issued", p["band"])[str(L)]
            ih = s24["ihist"][str(L)]
            ih.append([ws.isoformat(), float(not (ib[0] <= y <= ib[1]))])
            del ih[: max(0, len(ih) - nn.HIST)]
        s24["last_resolved_end"] = (ws + nn.H_DELTA["24h"]).isoformat()
        n_res += 1
    s24["pending"] = keep
    qd24 = nn._qdict(q24, sig24)
    net["24h"] = {"window_start": S.isoformat(), "window_end": (S + nn.H_DELTA["24h"]).isoformat(), "quantiles_pct": qd24,
                  "sigma_pct": (qd24["q90"] - qd24["q10"]) / nn.NORM80, "sigma_ref_pct": 100 * sig24, "aci": {}, "band": {},
                  "construction": f"rolling day, shift {S.hour} h" + (" (= training construction)" if S.hour == 0 else "")}
    pend = {"window_start": S.isoformat(), "sig": sig24, "band": {}, "raw": {}}
    for L in LEVELS:
        a, b = nn.BAND[L]
        o = s24["aci"][L]
        lo, hi, Q = o.band(q24[a], q24[b])
        ilo, ihi, iQ, ae = issue_band(o, q24[a], q24[b], cfg.alpha_min[L])
        net["24h"]["aci"][str(L)] = {"lo_pct": 100 * lo * sig24, "hi_pct": 100 * hi * sig24, "Q": Q, "alpha_t": o.alpha_t}
        net["24h"]["band"][str(L)] = {"lo_pct": 100 * ilo * sig24, "hi_pct": 100 * ihi * sig24, "floored": False, "alpha_eff": ae, "Q": iQ}
        pend["band"][str(L)], pend["raw"][str(L)] = [lo, hi], [float(q24[a]), float(q24[b])]
        pend.setdefault("issued", {})[str(L)] = [ilo, ihi]
    net["24h"]["sigma_eff_pct"] = (net["24h"]["band"]["80"]["hi_pct"] - net["24h"]["band"]["80"]["lo_pct"]) / nn.NORM80
    if np.isfinite(q24).all() and not any(p["window_start"] == pend["window_start"] for p in s24["pending"]):
        s24["pending"].append(pend)

    # ---- coverage (issued bands; for 24h the issued band is the ACI band)
    cov = {}
    for h in ("1h", "24h"):
        for L in LEVELS:
            errs = s1["ihist"][str(L)] if h == "1h" else s24["ihist"][str(L)]
            for n in (72, 168):
                c = _cov(errs, L, n)
                cov[f"{h}_{L}_{n}"] = c
                if c.get("alarm"):
                    alarms.append(f"coverage {h} v2 {L}% over last {c['n']}: {c['cov']:.3f} vs {L / 100:.2f}")
            if h == "1h":
                cov[f"1h_{L}_168_aci"] = nn.coverage(s1["aci"][L], 168)

    # ---- score earlier v2 lines (window closed), once
    done = {(s["id"], s["horizon"]) for s in read_scores(paths)}
    new_scores = []
    for ln in read_lines(paths):
        if nn._utc(ln["forecast_hour"]) >= S:
            continue
        for h in ("1h", "24h"):
            if (ln["id"], h) in done or h not in ln.get("net", {}):
                continue
            ws = nn._utc(ln["net"][h]["window_start"])
            if ws + nn.H_DELTA[h] > last_close:
                continue
            r = nn.realised(cl, ws, h)
            nh = ln["net"][h]
            if r is None or any(v is None for v in nh["quantiles_pct"].values()) or any(nh["band"][str(L)]["lo_pct"] is None for L in LEVELS):
                continue
            new_scores.append({"scored_at": logged_at.isoformat(), "scored_by": f"v21-{S:%Y%m%dT%H}", **score_line(ln, h, r)})
            done.add((ln["id"], h))

    ihash = hashlib.sha256(json.dumps([S.isoformat(), [float(f"{v:.12g}") for v in N["x_now"]], [float(f"{v:.12g}") for v in N["x24"]]]).encode()).hexdigest()[:20]
    code = hashlib.sha256(b"".join(Path(m.__file__).read_bytes() for m in (pnn, nn, __import__("ai_trader.live.nn_v2", fromlist=["x"])))).hexdigest()[:16]
    line = {
        "id": f"v21-{S:%Y%m%dT%H}", "schema": VERSION, "model_id": MODEL_ID, "version": VERSION, "config_hash": cfg.cfg_hash,
        "logged_at": logged_at.isoformat(), "as_of": as_of.isoformat(), "symbol": nn.SYMBOL, "forecast_hour": S.isoformat(),
        "data_max_ts": N["data_max"].isoformat(), "last_bar_close": last_close.isoformat(), "close": cl.get(last_close),
        "flags": sorted(set(flags)), "alarms": alarms,
        "models": {**N["cards"], "field": {k: (field or {}).get(k) for k in ("file", "sha256", "fit_as_of")}, "hnn10": (hn or {}).get("model_id"),
                   "code_sha256": code, "tool_version": VERSION, "v1_line": f"nn-{S:%Y%m%dT%H}"},
        "inputs_hash": ihash, "inputs": {"1h": dict(zip(HOURLY_FEATURES, map(float, N["x_now"]), strict=True)),
                                         "24h": dict(zip(DAILY_FEATURES, map(float, N["x24"]), strict=True))},
        "net": net, "coverage": cov,
        "aci_updates": {"1h_caught_up": n_caught, "1h_catchup_floored": n_floored_catchup, "24h_resolved": n_res, "24h_pending": len(s24["pending"])},
        "scored": [f"{s['id']}|{s['horizon']}" for s in new_scores],
    }
    st["as_of"] = as_of.isoformat()
    return nn._clean(line), st, persistable, [nn._clean(s) for s in new_scores]


def write_outputs(paths: PathsV2, line: dict, st: dict, persistable: bool, scores: list[dict], *, force: bool = False) -> dict:
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
    if persistable:
        save_state(paths, st)
        out["state"] = True
    return out


# ------------------------------------------------------------------ v1 vs v2 (descriptive)


def _first(lines: list[dict]) -> dict:
    out = {}
    for ln in sorted(lines, key=lambda r: r["logged_at"]):
        if "stale" in ln.get("flags", []):
            continue
        out.setdefault(ln["forecast_hour"], ln)
    return out


def compare(v1_lines: list[dict], v2_lines: list[dict], realised, since: datetime | None = None) -> dict:
    """Same forecast bars, same windows: coverage of v1's ACI bands vs v2's issued bands, band pinball, width.
    realised(window_start, horizon) -> % log return or None. Descriptive only (H-LIVE1 still compares Claude vs v1)."""
    from ai_trader.research.evaluate import newey_west_t

    a, b = _first(v1_lines), _first(v2_lines)
    res = {}
    for h in ("1h", "24h"):
        rows = []
        for fh in sorted(set(a) & set(b)):
            if since is not None and nn._utc(fh) < since:
                continue
            l1, l2 = a[fh]["net"].get(h), b[fh]["net"].get(h)
            if not l1 or not l2:
                continue
            r = realised(nn._utc(l1["window_start"]), h)
            if r is None or any(l1["aci"][str(L)]["lo_pct"] is None or l2["band"][str(L)]["lo_pct"] is None for L in LEVELS):
                continue
            b1 = {L: (l1["aci"][str(L)]["lo_pct"], l1["aci"][str(L)]["hi_pct"]) for L in LEVELS}
            b2 = {L: (l2["band"][str(L)]["lo_pct"], l2["band"][str(L)]["hi_pct"]) for L in LEVELS}
            rows.append({"fh": fh, "day": fh[:10], "replay": "replay" in b[fh].get("flags", []), "p1": band_pinball(r, b1), "p2": band_pinball(r, b2),
                         **{f"h1_{L}": b1[L][0] <= r <= b1[L][1] for L in LEVELS}, **{f"h2_{L}": b2[L][0] <= r <= b2[L][1] for L in LEVELS},
                         "w1": b1[80][1] - b1[80][0], "w2": b2[80][1] - b2[80][0]})
        n = len(rows)
        if not n:
            res[h] = {"n": 0}
            continue
        d = np.array([x["p2"] - x["p1"] for x in rows])
        m, t, _ = newey_west_t(d) if n >= 5 else (float(d.mean()), float("nan"), n)
        res[h] = {"n": n, "n_replay_v2": sum(x["replay"] for x in rows), "first": rows[0]["fh"], "last": rows[-1]["fh"],
                  **{f"cov_v1_{L}": float(np.mean([x[f"h1_{L}"] for x in rows])) for L in LEVELS},
                  **{f"cov_v2_{L}": float(np.mean([x[f"h2_{L}"] for x in rows])) for L in LEVELS},
                  "pinball_bands_v1": float(np.mean([x["p1"] for x in rows])), "pinball_bands_v2": float(np.mean([x["p2"] for x in rows])),
                  "mean_d": float(d.mean()), "nw_t": t, "width80_v1": float(np.mean([x["w1"] for x in rows])), "width80_v2": float(np.mean([x["w2"] for x in rows]))}
    return res


# ------------------------------------------------------------------ summary block for LATEST.md


def _b(v: dict) -> str:
    return f"[{nn._f(v['lo_pct'])}, {nn._f(v['hi_pct'])}]"


def render_block(line: dict, cmp: dict | None) -> list[str]:
    n1, n24 = line["net"]["1h"], line["net"]["24h"]
    fl = n1["floor"]
    ftxt = (f"floor ACTIVE (|z|>3 at {', '.join(t[11:16] for t, _ in fl['trigger'])}; floor sigma {nn._f(fl['sigma_floor_pct'], sign=False)}%, "
            f"applied {'/'.join(map(str, fl['applied'])) or 'none'})") if fl["active"] else "floor inactive"
    cv = line["coverage"]
    al1 = "/".join(f"{v['alpha_t']:.4f}" for v in n1["aci"].values())
    al24 = "/".join(f"{v['alpha_t']:.4f}" for v in n24["aci"].values())
    L = [f"\n## v2.1 shadow (ACI gamma 0.02, issue alpha >= nominal/4, jump-cluster floor; cfg {line['config_hash']}; {line['id']}"
         + (", replay" if "replay" in line["flags"] else "") + ")",
         "- 1h: " + "; ".join(f"{k}% {_b(v)}" for k, v in n1["band"].items()) + f"; alpha_t (unclipped) 80/90/95 {al1}; {ftxt}",
         "- 24h: " + "; ".join(f"{k}% {_b(v)}" for k, v in n24["band"].items()) + f"; alpha_t {al24}",
         "- running coverage (issued, 72 / 168): " + " | ".join(
             f"{h} " + ", ".join(f"{lv}% {nn._f(cv[f'{h}_{lv}_72'].get('cov'), 3, False)}/{nn._f(cv[f'{h}_{lv}_168'].get('cov'), 3, False)}" for lv in LEVELS)
             for h in ("1h", "24h"))]
    if cmp:
        parts = []
        for h in ("1h", "24h"):
            c = cmp.get(h, {})
            if not c.get("n"):
                continue
            parts.append(f"{h} n {c['n']}: cov80 {c['cov_v1_80']:.3f} vs {c['cov_v2_80']:.3f}, cov95 {c['cov_v1_95']:.3f} vs {c['cov_v2_95']:.3f}, "
                         f"band pinball {c['pinball_bands_v1']:.4f} vs {c['pinball_bands_v2']:.4f}")
        if parts:
            L.append(f"- v1 vs v2 since {cmp.get('since', '')[5:16]} (same bars; v2 mostly replay): " + "; ".join(parts))
    return L


__all__ = ["MODEL_ID", "VERSION", "PathsV2", "Settings", "band_pinball", "cluster", "compare", "config_hash", "floor_band", "load_config",
           "render_block", "run_line", "write_outputs"]
