"""Phase 3 command line: ``python -m vecopen phase3`` (the reproduce command of ``reports/phase_3.md``).

Per asset, one at a time (memory), reading ``data/research/<ASSET>.parquet`` only:

1. EVENT STEP (no forward data). Base 5m bars and the closed bars of every primary timeframe (phase 1's ``resample``
   with the asset's session) are merged into the as-of feed order (``levels.bar_stream``) and fed one bar at a time
   to four ``TouchDetector``s, one per family, each over its own ``LevelRegistry``. After every tau-bar, the
   ``PlaceboGenerator`` turns that bar and the treated levels born at it into placebo levels, which are registered
   in the placebo registries before the base bar of the same close.
2. LABEL STEP (forward-looking, separate). A second pass over the base bars feeds ``labels.LabelStream``; each event
   is added at its t0 and labelled from the bars after it.
3. Tables written to ``data/research/events/<ASSET>/{levels,events,labels}.parquet``; the in-memory frames are freed.

After the last asset: MATCHING of family 1 (``match_all``; it reads covariate columns of the saved event tables,
never a label) into ``<ASSET>/matches.parquet``, the covariate balance, ``reports/phase_3_tables.md``,
``data/research/events/manifest.json``, the vecopen tests and one registry line (phase 3, kind ``build``).

No statistic that compares treated and placebo OUTCOMES is computed or printed here (that is gate G2, phase 5).
Placebo first-fill counts are not printed either, nor stored in the manifest or the registry line, and neither are
event or label row counts: with the births they would give a touch rate (H2). The report holds treated counts,
placebo births, matching counts, covariate balance and a treated-only label summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vecopen import registry
from vecopen.config import PROJECT_ROOT, load_config

EVENTS_DIR = PROJECT_ROOT / "data" / "research" / "events"
TABLES_PATH = PROJECT_ROOT / "reports" / "phase_3_tables.md"
TREATED = "treated"


def _year(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, UTC).year


def _dt(ms: int | None):
    return None if ms is None else datetime.fromtimestamp(ms / 1000, UTC)


# ---------------------------------------------------------------- rows

def level_row(asset: str, x: Any) -> dict[str, Any]:
    return {"asset": asset, "family": x.family, "level_id": x.level_id, "event_key": f"{x.family}:{x.level_id}",
            "tf": x.tf, "birth_ts": _dt(x.birth_ts), "price": x.price, "zone_lo": x.zone_lo, "zone_hi": x.zone_hi,
            "atr": x.atr, "g": x.g, "leg_direction": x.leg_direction, "origin_index": x.origin_index,
            "origin_ts_close": _dt(x.origin_ts_close), "k_star": x.k_star, "leg_size_atr": x.leg_size_atr,
            "leg_vol": x.leg_vol, "ref_close": x.ref_close, "departure_atr": x.departure_atr, "d_edge": x.d_edge,
            "regime": x.regime, "sigma_birth": x.sigma_birth, "h_base": x.h_base,
            "source_level_id": x.source_level_id, "variant": x.variant, "displacement_atr": x.displacement_atr,
            "omega": x.leg.omega if x.leg is not None else None}


def event_row(asset: str, e: Any, kappa: float, k_pt: float, k_sl: float) -> dict[str, Any]:
    x = e.level
    a = x.atr
    r = level_row(asset, x)
    r.update({"t0": _dt(e.t0), "year": _year(e.t0), "age": e.age, "age_log": math.log1p(e.age),
              "age_hours": (e.t0 - x.birth_ts) / 3_600_000, "polarity": e.polarity, "confluence": e.confluence,
              "approach_side": e.approach_side, "prev_close": e.prev_close, "sigma_t0": e.sigma_t0,
              "touch_o": e.touch_o, "touch_h": e.touch_h, "touch_l": e.touch_l, "touch_c": e.touch_c,
              # analytic-baseline inputs with the touch at the near zone edge (SPEC section 3), price units
              "a_fade_edge": (k_pt - kappa) * a, "b_fade_edge": (2 * kappa + k_sl) * a,
              "a_follow_edge": (k_pt + kappa) * a, "b_follow_edge": k_sl * a})
    return r


def label_row(asset: str, lab: Any) -> dict[str, Any]:
    r = {"asset": asset, "family": lab.family, "level_id": lab.level_id, "event_key": f"{lab.family}:{lab.level_id}",
         "tf": lab.tf, "t0": _dt(lab.t0), "h_base": lab.h_base, "horizon_end_ts": _dt(lab.horizon_end_ts),
         "entry_ts": _dt(lab.entry_ts), "entry": lab.entry, "bars_seen": lab.bars_seen, "censored": lab.censored,
         "res": lab.res, "res_side": lab.res_side, "res_end_ts": _dt(lab.res_end_ts), "mae_atr": lab.mae_atr,
         "mae_end_ts": _dt(lab.mae_end_ts)}
    for b in (lab.fade, lab.follow):
        p = b.side
        r.update({f"{p}_dir": b.direction, f"{p}_target": b.target, f"{p}_stop": b.stop, f"{p}_a": b.a,
                  f"{p}_b": b.b, f"{p}_label": b.label, f"{p}_r": b.r, f"{p}_pnl_atr": b.pnl_atr, f"{p}_exit": b.exit,
                  f"{p}_reason": b.reason,
                  f"{p}_end_ts": _dt(b.end_ts), f"{p}_tie": b.tie})
    return r


# ---------------------------------------------------------------- one asset

class Frames:
    """Row sink that turns every ``chunk`` dict rows into a polars frame (compact), concatenated at the end."""

    def __init__(self, chunk: int = 20_000) -> None:
        self.rows: list[dict[str, Any]] = []
        self.frames: list[Any] = []
        self.chunk = chunk

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        import polars as pl

        if self.rows:
            self.frames.append(pl.DataFrame(self.rows, infer_schema_length=None))
            self.rows = []

    def frame(self) -> Any:
        import polars as pl

        self.flush()
        out = pl.concat(self.frames, how="vertical_relaxed") if self.frames else pl.DataFrame()
        self.frames = []
        return out


def event_step(asset: str, df: Any, higher: dict[str, Any], cfg: Any, session: Any,
               level_sink: Any = None) -> dict[str, Any]:
    """The streaming event step (module docstring, step 1): a loop over ``update``; it never sees a later bar.
    Level rows go to ``level_sink.append`` (default: a list returned as ``level_rows``)."""
    from vecopen.events import TouchDetector
    from vecopen.levels import LevelRegistry, bar_stream
    from vecopen.placebo import FAMILIES, PlaceboGenerator, rng_for

    tfs = list(higher)
    treated = TouchDetector(LevelRegistry(cfg, timeframes=tfs, session=session), cfg, session=session)
    fams = {f: TouchDetector(LevelRegistry(cfg, timeframes=tfs, session=session, family=f), cfg, session=session)
            for f in FAMILIES}
    gen = PlaceboGenerator(cfg, treated.registry.timeframes, rng=rng_for(asset, cfg))
    base_tf = treated.base_tf
    sink = level_sink if level_sink is not None else []
    events: list[Any] = []
    fill_ts: dict[str, int] = {}
    expire: dict[str, int] = {}
    n_base = 0
    for b in bar_stream(df, higher, base_tf):
        n_base += b.tf == base_tf
        for td in (treated, *fams.values()):
            for e in td.update(b):
                events.append(e)
                fill_ts[f"{e.family}:{e.level_id}"] = e.t0
            for x in td.expired:
                expire[f"{td.family}:{x.level_id}"] = x.ts
        for x in treated.new_levels:
            sink.append(level_row(asset, x))
        if b.tf != base_tf:
            for sp in gen.update(b, treated.new_levels):
                x = fams[sp.family].register(sp.leg, source_level_id=sp.source_level_id, variant=sp.variant,
                                             displacement_atr=sp.displacement_atr)
                sink.append(level_row(asset, x))
    return {"events": events, "level_rows": sink if level_sink is None else None, "fill_ts": fill_ts,
            "expire": expire, "base_tf": base_tf, "n_base": n_base, "late_origin": list(gen.late_origin),
            "detectors": {TREATED: treated, **fams}}


def label_step(events: list[Any], df: Any, cfg: Any, base_tf: str) -> Any:
    """THE LABELLING STEP (forward-looking): ``events`` sorted by t0, then one pass over the base bars. Yields labels."""
    from vecopen.labels import LabelStream
    from vecopen.levels import _bars_from_frame

    ls = LabelStream(cfg, base_tf)
    k = 0
    for b in _bars_from_frame(df, base_tf):
        yield from ls.update(b)
        while k < len(events) and events[k].t0 == b.ts_close:
            ls.add(events[k])
            k += 1
    if k != len(events):
        raise AssertionError(f"{len(events) - k} events are not at a base-bar close")
    yield from ls.finish()


def _key_frame(d: dict[str, int], col: str) -> Any:
    import polars as pl

    return pl.DataFrame({"event_key": pl.Series(list(d), dtype=pl.String),
                         col: pl.Series(list(d.values()), dtype=pl.Int64)}).with_columns(
        pl.col(col).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC").dt.cast_time_unit("us"))


def run_asset(asset: str, cfg: Any, out_dir: Path) -> dict[str, Any]:
    import gc

    import polars as pl

    from vecopen.bars import load_sessions, resample
    from vecopen.data import load_bars
    from vecopen.labels import label_params
    from vecopen.placebo import P1

    t_start = time.perf_counter()
    df = load_bars(asset)
    df = df.select([c for c in ("ts_close", "o", "h", "l", "c", "v") if c in df.columns])
    session = load_sessions().for_asset(asset)
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    higher = {tf: resample(df, tf, session) for tf in tfs}
    kappa = float(cfg.data["levels"]["kappa_atr"])
    k_pt, k_sl = label_params(cfg)

    # ---- 1. event step: streaming, no forward data
    lv_sink = Frames()
    st = event_step(asset, df, higher, cfg, session, level_sink=lv_sink)
    del higher
    events, base_tf, n_base = st["events"], st["base_tf"], st["n_base"]
    late_src = pl.DataFrame({"tf": [x[0] for x in st["late_origin"]], "origin_index": [x[1] for x in st["late_origin"]]},
                            schema={"tf": pl.String, "origin_index": pl.Int64})
    levels_df = lv_sink.frame()
    late_keys = levels_df.filter(pl.col("family") == P1).join(late_src, on=["tf", "origin_index"])["event_key"]
    levels_df = (levels_df.join(_key_frame(st["fill_ts"], "fill_ts"), on="event_key", how="left")
                 .join(_key_frame(st["expire"], "expire_ts"), on="event_key", how="left")
                 .with_columns(late_origin=pl.col("event_key").is_in(late_keys.implode())))
    del st
    gc.collect()
    t_events = time.perf_counter()

    # ---- 2. label step: separate, forward-looking
    events.sort(key=lambda e: e.t0)                       # stable: family and level-id order kept within a t0
    lab_sink = Frames()
    for x in label_step(events, df, cfg, base_tf):
        lab_sink.append(label_row(asset, x))
    labels_df = lab_sink.frame()
    t_labels = time.perf_counter()
    ev_sink = Frames()
    for e in events:
        ev_sink.append(event_row(asset, e, kappa, k_pt, k_sl))
    del events
    events_df = ev_sink.frame().with_columns(late_origin=pl.col("event_key").is_in(late_keys.implode()))
    del df
    gc.collect()

    # ---- 4. write
    d = out_dir / asset
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, frame in (("levels", levels_df), ("events", events_df), ("labels", labels_df)):
        files[name] = _write(frame, d / f"{name}.parquet")
    t_end = time.perf_counter()
    summary = summarise(asset, levels_df, events_df, labels_df, tfs)
    summary.update({"files": files, "base_bars": n_base, "late_origin": int(late_keys.len()),
                    "seconds": {"events": round(t_events - t_start, 1), "labels": round(t_labels - t_events, 1),
                                "total": round(t_end - t_start, 1)}})
    return summary


# ---------------------------------------------------------------- per-asset summary (counts and covariates only)

def _moments(x: Any) -> list[float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return [int(len(a)), float(a.sum()), float((a * a).sum())]


def _write(frame: Any, p: Path) -> dict[str, Any]:
    frame.write_parquet(p, compression="zstd", statistics=True)
    rel = p.relative_to(PROJECT_ROOT).as_posix() if p.is_relative_to(PROJECT_ROOT) else p.as_posix()
    return {"path": rel, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}   # no row counts (module docstring)


def match_all(summaries: list[dict[str, Any]], tfs: list[str], out_dir: Path) -> None:
    """Step 3, after every asset's event table is written: family-1 matching (``placebo.match_events``), exact on
    year, in cells of one (asset, timeframe) for 15m..4h and one timeframe across all assets of the run for
    ``placebo.CROSS_ASSET_TFS``. Writes ``<ASSET>/matches.parquet`` and adds matching counts, pooled moments and
    per-(asset, timeframe) SMDs to the summaries. Reads covariate columns only (no label is read)."""
    import polars as pl

    from vecopen.placebo import CROSS_ASSET_TFS, MATCH_COVARIATES, P1, match_events, smd

    cols = ["asset", "event_key", "family", "tf", "year", *MATCH_COVARIATES, "late_origin"]
    ev = pl.concat([pl.read_parquet(out_dir / s["asset"] / "events.parquet", columns=cols)
                    .filter(pl.col("family").is_in([TREATED, P1])) for s in summaries])
    ev = ev.with_columns(gkey=pl.col("asset") + "/" + pl.col("event_key"))
    parts = []
    c2 = ["year", *MATCH_COVARIATES]
    for tf in tfs:
        x = ev.filter(pl.col("tf") == tf)
        groups = [None] if tf in CROSS_ASSET_TFS else [s["asset"] for s in summaries]
        for g in groups:
            y = x if g is None else x.filter(pl.col("asset") == g)
            t_ = y.filter(pl.col("family") == TREATED)
            p_ = y.filter((pl.col("family") == P1) & ~pl.col("late_origin"))
            if t_.height == 0:
                continue
            m = match_events({"event_key": t_["gkey"].to_numpy(), **{c: t_[c].to_numpy() for c in c2}},
                             {"event_key": p_["gkey"].to_numpy(), **{c: p_[c].to_numpy() for c in c2}})
            parts.append(pl.DataFrame({"asset": t_["asset"], "tf": tf, "year": t_["year"],
                                       "treated_key": t_["event_key"], "placebo_gkey": m["placebo_key"].tolist(),
                                       "distance": m["distance"], "cell": "asset" if g is not None else "all_assets"},
                                      schema_overrides={"placebo_gkey": pl.String}))
    mm = pl.concat(parts)
    mm = mm.with_columns(placebo_asset=pl.col("placebo_gkey").str.split("/").list.get(0),
                         placebo_key=pl.col("placebo_gkey").str.split("/").list.get(1)).drop("placebo_gkey")
    cov = ev.select(["asset", "event_key", *MATCH_COVARIATES])
    mm = (mm.join(cov.rename({c: f"t_{c}" for c in MATCH_COVARIATES}), left_on=["asset", "treated_key"],
                  right_on=["asset", "event_key"], how="left")
          .join(cov.rename({"asset": "placebo_asset", **{c: f"p_{c}" for c in MATCH_COVARIATES}}),
                left_on=["placebo_asset", "placebo_key"], right_on=["placebo_asset", "event_key"], how="left"))
    ok_all = mm.filter(pl.col("placebo_key").is_not_null())
    glob = {}
    for tf in tfs:
        u = ok_all.filter(pl.col("tf") == tf).group_by(["placebo_asset", "placebo_key"]).len()
        glob[tf] = {"distinct": u.height, "max_reuse": int(u["len"].max()) if u.height else 0}
    for s in summaries:
        a = s["asset"]
        mine = mm.filter(pl.col("asset") == a)
        s["files"]["matches"] = _write(mine.select(["asset", "tf", "year", "treated_key", "placebo_asset",
                                                    "placebo_key", "distance", "cell"]), out_dir / a / "matches.parquet")
        mt, mom, cells = {}, {}, {}
        for tf in tfs:
            x = mine.filter(pl.col("tf") == tf)
            ok = x.filter(pl.col("placebo_key").is_not_null())
            uses = Counter(zip(ok["placebo_asset"].to_list(), ok["placebo_key"].to_list()))
            mt[tf] = {"treated": x.height, "matched": ok.height, "unmatched": x.height - ok.height,
                      "max_reuse": max(uses.values()) if uses else 0,
                      "other_asset": int((ok["placebo_asset"] != a).sum()),
                      "median_distance": float(ok["distance"].median()) if ok.height else None}
            mom[tf] = {f"{sd}_{c}": _moments(ok[f"{sd}_{c}"]) for sd in ("t", "p") for c in MATCH_COVARIATES}
            if ok.height:
                cells[tf] = {"n": ok.height, "smd": {c: smd(ok[f"t_{c}"].to_numpy(), ok[f"p_{c}"].to_numpy())
                                                     for c in MATCH_COVARIATES}}
        s["matching"], s["_matched_moments"], s["_cells"] = mt, mom, cells
        s["_distinct_global"] = glob


def summarise(asset: str, levels: Any, events: Any, labels: Any, tfs: list[str]) -> dict[str, Any]:
    import polars as pl

    from vecopen.placebo import P1

    out: dict[str, Any] = {"asset": asset}
    out["births"] = {f: {tf: int(levels.filter((pl.col("family") == f) & (pl.col("tf") == tf)).height) for tf in tfs}
                     for f in levels["family"].unique().sort().to_list()}
    tev = events.filter(pl.col("family") == TREATED)
    out["treated_events"] = {tf: int(tev.filter(pl.col("tf") == tf).height) for tf in tfs}
    out["treated_approach_mismatch"] = int(tev.filter(pl.col("approach_side") != pl.col("g")).height)
    out["treated_missing_sigma"] = int(tev.filter(pl.col("sigma_t0").is_null() | pl.col("sigma_birth").is_null()).height)
    out["treated_missing_regime"] = int(tev.filter(pl.col("regime").is_null()).height)

    # treated-only label summary (raw sums, pooled later)
    tl = labels.filter(pl.col("family") == TREATED)
    lab: dict[str, Any] = {}
    for tf in tfs:
        x = tl.filter(pl.col("tf") == tf)
        done = x.filter(~pl.col("censored"))
        r = {"events": x.height, "complete": done.height, "censored": x.height - done.height,
             "res_n": int(x["res"].is_not_null().sum()), "res_sum": int(x["res"].fill_null(0).sum()),
             "mae_n": int(done["mae_atr"].is_not_null().sum()), "mae_sum": float(done["mae_atr"].fill_null(0).sum())}
        for p in ("fade", "follow"):
            valid = x.filter(pl.col(f"{p}_label").is_not_null())
            r[f"{p}_n"] = valid.height
            r[f"{p}_hit"] = int(valid[f"{p}_label"].sum()) if valid.height else 0
            r[f"{p}_r_sum"] = float(valid[f"{p}_r"].sum()) if valid.height else 0.0
            r[f"{p}_pnl_sum"] = float(valid[f"{p}_pnl_atr"].sum()) if valid.height else 0.0
            r[f"{p}_r_max"] = float(valid[f"{p}_r"].max()) if valid.height else float("nan")
            r[f"{p}_void_target"] = int((x[f"{p}_reason"] == "void_target").sum())
            r[f"{p}_void_stop"] = int((x[f"{p}_reason"] == "void_stop").sum())
            r[f"{p}_tie"] = int(x[f"{p}_tie"].sum())
            r[f"{p}_time"] = int((x[f"{p}_reason"] == "time").sum())
        lab[tf] = r
    out["treated_labels"] = lab

    pool = levels.filter((pl.col("family") == P1) & ~pl.col("late_origin"))
    tlv = levels.filter(pl.col("family") == TREATED)
    out["_birth_moments"] = {tf: {f"{s}_{c}": _moments(fr.filter(pl.col("tf") == tf)[c].drop_nulls())
                                  for s, fr in (("t", tlv), ("p", pool)) for c in ("departure_atr", "regime")}
                             for tf in tfs}
    p2 = levels.filter(pl.col("family") == "p2_displaced")
    out["p2_sign_counts"] = {v: int((p2["variant"] == v).sum()) for v in ("up", "down")}
    out["p2_upsilon_range"] = [float(p2["displacement_atr"].abs().min()), float(p2["displacement_atr"].abs().max())] \
        if p2.height else None
    tr = levels.filter(pl.col("family") == TREATED)
    out["treated_levels"] = tr.height
    out["treated_omega_zero"] = int((tr["omega"] == 0).sum())
    return out


# ---------------------------------------------------------------- report tables

def _f(x: float | None, nd: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def build_tables(summaries: list[dict[str, Any]], tfs: list[str]) -> tuple[str, dict[str, Any]]:
    from vecopen.placebo import FAMILIES, MATCH_COVARIATES, Moments, smd_moments

    assets = [s["asset"] for s in summaries]
    L = ["# Phase 3 tables (crypto research sample)", "",
         "Generated by `uv run --extra vecopen python -m vecopen phase3`. Do not edit by hand. Counts and covariates "
         "only: no statistic comparing treated and placebo outcomes is computed here (gate G2, phase 5).", ""]
    metrics: dict[str, Any] = {}

    # 1. treated events per asset x tf
    L += ["## 1. Treated first-fill events per asset and timeframe", "",
          "| asset | " + " | ".join(tfs) + " | total |", "|---|" + "---|" * (len(tfs) + 1)]
    tot = Counter()
    for s in summaries:
        row = s["treated_events"]
        tot.update(row)
        L.append(f"| {s['asset']} | " + " | ".join(f"{row[tf]:,}" for tf in tfs) + f" | {sum(row.values()):,} |")
    L.append("| **all** | " + " | ".join(f"**{tot[tf]:,}**" for tf in tfs) + f" | **{sum(tot.values()):,}** |")
    metrics["treated_events"] = {tf: tot[tf] for tf in tfs}
    metrics["treated_events_total"] = sum(tot.values())

    # 2. levels born per family x tf
    fams = [TREATED, *FAMILIES]
    births = {f: Counter() for f in fams}
    for s in summaries:
        for f in fams:
            births[f].update(s["births"].get(f, {}))
    L += ["", "## 2. Levels born per family and timeframe (all 20 assets)", "",
          "Placebo first-fill counts are deliberately not shown: with the births they would give a touch rate, an H2 "
          "outcome.", "",
          "| family | " + " | ".join(tfs) + " | total |", "|---|" + "---|" * (len(tfs) + 1)]
    for f in fams:
        L.append(f"| {f} | " + " | ".join(f"{births[f][tf]:,}" for tf in tfs) + f" | {sum(births[f].values()):,} |")
    metrics["births"] = {f: dict(births[f]) for f in fams}
    L += ["", "Levels born per asset and family:", "", "| asset | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for s in summaries:
        L.append(f"| {s['asset']} | " + " | ".join(f"{sum(s['births'].get(f, {}).values()):,}" for f in fams) + " |")
    p2s = Counter()
    for s in summaries:
        p2s.update(s["p2_sign_counts"])
    lo = min(s["p2_upsilon_range"][0] for s in summaries if s["p2_upsilon_range"])
    hi = max(s["p2_upsilon_range"][1] for s in summaries if s["p2_upsilon_range"])
    late = sum(s["late_origin"] for s in summaries)
    om0, ntr = sum(s["treated_omega_zero"] for s in summaries), sum(s["treated_levels"] for s in summaries)
    L += ["", f"Family 2: {p2s['up']:,} up and {p2s['down']:,} down; |upsilon| drawn in [{lo:.4f}, {hi:.4f}]. "
          f"Family 1 placebos whose candle later became a treated origin (excluded from matching): {late}. "
          f"Treated levels with omega = 0 (open = the open-side extreme, so that extreme is not a distinct price): "
          f"{om0:,} of {ntr:,}."]
    metrics.update({"treated_omega_zero": om0, "treated_levels": ntr})
    metrics.update({"p2_signs": dict(p2s), "p2_upsilon_minmax": [lo, hi], "p1_late_origin": late})

    # 3. matching counts
    from vecopen.placebo import CROSS_ASSET_TFS

    L += ["", "## 3. Family 1 matching: 1:1 nearest neighbour with replacement, exact on year", "",
          "Cell = one asset for 15m to 4h; all assets of the run for 1d, 1w and 1M (decision D-P5).", "",
          "| tf | cell | treated events | matched | unmatched | distinct placebo events used | max reuse of one "
          "placebo | matched to another asset | median distance (SD units, median over assets) |",
          "|---|---|---|---|---|---|---|---|---|"]
    mt_all = {}
    for tf in tfs:
        rows = [s["matching"][tf] for s in summaries if tf in s["matching"]]
        if not rows:
            continue
        a = {k: sum(r[k] for r in rows) for k in ("treated", "matched", "unmatched", "other_asset")}
        g = summaries[0]["_distinct_global"][tf]
        a["distinct_placebo"], a["max_reuse"] = g["distinct"], g["max_reuse"]
        dists = [r["median_distance"] for r in rows if r["median_distance"] is not None]
        a["median_of_asset_medians"] = float(np.median(dists)) if dists else None
        mt_all[tf] = a
        cell = "all assets" if tf in CROSS_ASSET_TFS else "asset"
        L.append(f"| {tf} | {cell} | {a['treated']:,} | {a['matched']:,} | {a['unmatched']:,} | "
                 f"{a['distinct_placebo']:,} | {a['max_reuse']} | {a['other_asset']:,} | "
                 f"{_f(a['median_of_asset_medians'])} |")
    metrics["matching"] = mt_all

    # 4. balance on the matched sample
    L += ["", "## 4. Covariate balance of the matched sample (treated first fills vs matched family-1 first fills)", "",
          "SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2) over the matched pairs (a placebo used twice counts "
          "twice). Pre-matching balance of age at fill is not shown: for unmatched levels the age at first fill is "
          "the time to touch, an H2 outcome.", "",
          "| tf | covariate | n pairs | mean treated | mean placebo | SD treated | SD placebo | SMD |",
          "|---|---|---|---|---|---|---|---|"]
    bal: dict[str, Any] = {}
    overall = {c: (Moments(), Moments()) for c in MATCH_COVARIATES}
    for tf in [*tfs, "all"]:
        if tf == "all":
            mom = overall
        else:
            mom = {c: (Moments(), Moments()) for c in MATCH_COVARIATES}
            for s in summaries:
                cv = s["_matched_moments"].get(tf)
                if not cv:
                    continue
                for c in MATCH_COVARIATES:
                    for k, side in ((0, "t"), (1, "p")):
                        mom[c][k].merge(*cv[f"{side}_{c}"])
                        overall[c][k].merge(*cv[f"{side}_{c}"])
        bal[tf] = {}
        for c in MATCH_COVARIATES:
            t, p = mom[c]
            v = smd_moments(t, p)
            bal[tf][c] = {"n": t.n, "mean_t": t.mean, "mean_p": p.mean, "sd_t": math.sqrt(t.var) if t.n > 1 else None,
                          "sd_p": math.sqrt(p.var) if p.n > 1 else None, "smd": v}
            b = bal[tf][c]
            L.append(f"| {'**all**' if tf == 'all' else tf} | {c} | {t.n:,} | {_f(b['mean_t'])} | {_f(b['mean_p'])} | "
                     f"{_f(b['sd_t'])} | {_f(b['sd_p'])} | {_f(v)} |")
    metrics["balance"] = bal
    worst = max(abs(bal[tf][c]["smd"]) for tf in bal for c in MATCH_COVARIATES if math.isfinite(bal[tf][c]["smd"]))
    metrics["balance_max_abs_smd_by_tf"] = worst

    # per asset x tf cells
    cells = []
    for s in summaries:
        for tf, cv in s["_cells"].items():
            cells.append((s["asset"], tf, cv["n"], cv["smd"]))
    L += ["", "Per (asset, timeframe) cell: number of cells, and cells with |SMD| >= 0.1 on any covariate, by cell size:", "",
          "| cell size (pairs) | cells | cells with any abs(SMD) >= 0.1 | max abs(SMD) |", "|---|---|---|---|"]
    cell_tab = {}
    for name, lo_, hi_ in (("< 30", 0, 30), ("30 to 299", 30, 300), (">= 300", 300, 10**12)):
        cs = [x for x in cells if lo_ <= x[2] < hi_]
        bad = [x for x in cs if any(math.isfinite(v) and abs(v) >= 0.1 for v in x[3].values())]
        mx = max((abs(v) for x in cs for v in x[3].values() if math.isfinite(v)), default=float("nan"))
        cell_tab[name] = {"cells": len(cs), "bad": len(bad), "max_abs_smd": mx,
                          "bad_cells": [(a, tf, n, {k: round(v, 3) for k, v in d.items()}) for a, tf, n, d in bad]}
        L.append(f"| {name} | {len(cs)} | {len(bad)} | {_f(mx)} |")
    metrics["balance_cells"] = cell_tab
    bad_big = cell_tab[">= 300"]["bad_cells"] + cell_tab["30 to 299"]["bad_cells"]
    if bad_big:
        L += ["", "Cells with at least 30 pairs and some |SMD| >= 0.1:", "", "| asset | tf | pairs | SMD age_log | "
              "SMD departure_atr | SMD regime |", "|---|---|---|---|---|---|"]
        for a, tf, n, d in sorted(bad_big, key=lambda x: (x[1], x[0])):
            L.append(f"| {a} | {tf} | {n} | {d['age_log']} | {d['departure_atr']} | {d['regime']} |")

    # 5. pre-matching balance at birth (levels, birth covariates only)
    L += ["", "## 5. Before matching, at birth: treated levels vs the family-1 pool (birth covariates only)", "",
          "| tf | covariate | n treated | n pool | mean treated | mean pool | SMD |", "|---|---|---|---|---|---|---|"]
    pre = {}
    for tf in tfs:
        pre[tf] = {}
        for c in ("departure_atr", "regime"):
            t, p = Moments(), Moments()
            for s in summaries:
                bc = s["_birth_moments"][tf]
                t.merge(*bc[f"t_{c}"])
                p.merge(*bc[f"p_{c}"])
            v = smd_moments(t, p)
            pre[tf][c] = {"n_t": t.n, "n_p": p.n, "mean_t": t.mean, "mean_p": p.mean, "smd": v}
            L.append(f"| {tf} | {c} | {t.n:,} | {p.n:,} | {_f(t.mean)} | {_f(p.mean)} | {_f(v)} |")
    metrics["prematch_birth_balance"] = pre

    # 6. treated label summary
    L += ["", "## 6. Labels summary, TREATED events only (all 20 assets pooled)", "",
          "Rates are over complete, non-void labels. Placebo labels are written to the tables but not summarised.", "",
          "| tf | events | complete | censored | resolution rate | mean MAE (ATR) | fade n | fade target-first | "
          "fade mean PnL (ATR) | fade mean R | fade max R | follow n | follow target-first | follow mean PnL (ATR) | "
          "follow mean R | follow max R |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    labsum = {}
    for tf in tfs:
        a = Counter()
        for s in summaries:
            a.update({k: v for k, v in s["treated_labels"][tf].items() if not k.endswith("_r_max")})
        for p in ("fade", "follow"):
            mx = [s["treated_labels"][tf][f"{p}_r_max"] for s in summaries]
            a[f"{p}_r_max"] = max((x for x in mx if math.isfinite(x)), default=float("nan"))
        labsum[tf] = dict(a)
        div = lambda x, y: x / y if y else None  # noqa: E731
        cols = [f"{a['events']:,}", f"{a['complete']:,}", f"{a['censored']:,}", _f(div(a["res_sum"], a["res_n"])),
                _f(div(a["mae_sum"], a["mae_n"]))]
        for p in ("fade", "follow"):
            cols += [f"{a[p + '_n']:,}", _f(div(a[p + "_hit"], a[p + "_n"])), _f(div(a[p + "_pnl_sum"], a[p + "_n"])),
                     _f(div(a[p + "_r_sum"], a[p + "_n"])), _f(a[p + "_r_max"], 1)]
        L.append(f"| {tf} | " + " | ".join(cols) + " |")
    L += ["", "Label mechanics, treated:", "",
          "| tf | fade void at target | fade void at stop | fade ties (stop first) | fade time exits | follow void at "
          "target | follow void at stop | follow ties (stop first) | follow time exits |", "|---|---|---|---|---|---|---|---|---|"]
    for tf in tfs:
        a = labsum[tf]
        L.append(f"| {tf} | {a.get('fade_void_target', 0):,} | {a.get('fade_void_stop', 0):,} | {a.get('fade_tie', 0):,} | "
                 f"{a.get('fade_time', 0):,} | {a.get('follow_void_target', 0):,} | {a.get('follow_void_stop', 0):,} | "
                 f"{a.get('follow_tie', 0):,} | {a.get('follow_time', 0):,} |")
    metrics["treated_labels"] = labsum
    mism = sum(s["treated_approach_mismatch"] for s in summaries)
    ms = sum(s["treated_missing_sigma"] for s in summaries)
    mr = sum(s["treated_missing_regime"] for s in summaries)
    L += ["", f"Treated events approached from the side opposite to g (a gap jumped the zone first): {mism}. "
          f"Treated events without sigma: {ms}; without regime: {mr}.",
          "Every event of every family is labelled; placebo label counts are not shown (they equal placebo touch counts)."]
    metrics.update({"treated_approach_mismatch": mism, "treated_missing_sigma": ms, "treated_missing_regime": mr})

    # 7. files and runtime
    L += ["", "## 7. Runtime per asset", "",
          "Event and label row counts are not shown: across families they would give placebo touch counts.", "",
          "| asset | event step (s) | label step (s) | asset total (s) |", "|---|---|---|---|"]
    for s in summaries:
        sec = s["seconds"]
        L.append(f"| {s['asset']} | {sec['events']} | {sec['labels']} | {sec['total']} |")
    metrics["assets"] = assets
    return "\n".join(L) + "\n", metrics


# ---------------------------------------------------------------- command

def cmd_phase3(args: argparse.Namespace) -> int:
    from vecopen.__main__ import run_tests
    from vecopen.phase2 import peak_rss_mb

    cfg = load_config()
    assets = args.assets.split(",") if args.assets else list(cfg.data["assets"]["crypto"]["symbols"])
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    t0 = time.perf_counter()
    for a in assets:
        s = run_asset(a, cfg, EVENTS_DIR)
        summaries.append(s)
        print(f"{a}: treated events {sum(s['treated_events'].values()):,}; {s['seconds']}; "
              f"peak RSS {peak_rss_mb():.0f} MB", flush=True)
    match_all(summaries, tfs, EVENTS_DIR)
    md, metrics = build_tables(summaries, tfs)
    TABLES_PATH.write_text(md, encoding="utf-8", newline="\n")
    print(md)
    manifest = {"config_sha256": cfg.sha256, "generated_utc": datetime.now(UTC).isoformat(),
                "assets": {s["asset"]: s["files"] for s in summaries},
                "columns": "see vecopen/reports/phase_3.md"}
    (EVENTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    counts = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "pytest_exit_code": 0}
    if not args.skip_tests:
        counts = run_tests(PROJECT_ROOT / "tests")
    print(f"tests: {counts}")
    green = counts["pytest_exit_code"] == 0 and counts["failed"] == 0 and counts["errors"] == 0
    metrics.update({"tests": counts, "runtime_seconds": round(time.perf_counter() - t0, 1),
                    "peak_rss_mb": round(peak_rss_mb(), 1),
                    "tables_sha256": hashlib.sha256(md.encode("utf-8")).hexdigest(),
                    "manifest_sha256": hashlib.sha256((EVENTS_DIR / "manifest.json").read_bytes()).hexdigest(),
                    "files": {s["asset"]: {k: v["sha256"] for k, v in s["files"].items()} for s in summaries}})
    metrics = json.loads(json.dumps(metrics, default=_jsonable, allow_nan=True), parse_constant=lambda _: None)
    if args.dry_run:
        print("dry run: registry not written")
    else:
        rec = registry.append(3, metrics, kind="build", config=cfg,
                              note="phase 3 events and labels: TouchDetector, label_event triple barrier, "
                                   f"PlaceboGenerator (3 families) + matching; {len(assets)} crypto assets")
        print(f"registry line {rec['seq'] + 1} appended (kind build): line_sha256 {rec['line_sha256']} "
              f"git_commit {rec['git_commit']} diff_sha256 {rec['diff_sha256']}")
    return 0 if green else 1


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


# ---------------------------------------------------------------- amendment 2026-11-14: balance per timeframe group
#
# ``python -m vecopen phase3-balance``. SPEC section 9 (user decision, 2026-11-14): phase 3 balance and the G2 family
# are judged per timeframe group. This step reads the STORED phase 3 outputs of the authoritative run (registry line
# ``PHASE3_AUTHORITATIVE_LINE``): ``matches.parquet`` and the covariate columns of ``events.parquet``, and nothing
# else. It never opens ``labels.parquet`` or ``levels.parquet``, recomputes no event and changes no matching. The only
# count it reports is the number of matched pairs, which equals the treated first-fill count already published.

TF_GROUPS: dict[str, tuple[str, ...]] = {"intraday": ("15m", "30m", "1h", "2h", "4h"),
                                         "daily_plus": ("1d", "1w", "1M")}
BALANCE_SMD_MAX = 0.1                    # SPEC section 8, phase 3 acceptance: |SMD| below 0.1
PHASE3_AUTHORITATIVE_LINE = "50d886626fb13408831e8abc042aba361198fd56032ce8ccff5627a9a5fa8e77"
GROUP_BALANCE_PATH = PROJECT_ROOT / "reports" / "phase_3_group_balance.md"
_EVENT_COV_COLS = ("asset", "event_key", "family")      # plus MATCH_COVARIATES; no outcome column
_MATCH_COLS = ("asset", "tf", "treated_key", "placebo_asset", "placebo_key")


def load_matched_covariates(events_dir: Path, assets: list[str]) -> Any:
    """One row per matched pair: ``asset``, ``tf``, ``t_<cov>`` and ``p_<cov>`` for the ``MATCH_COVARIATES``.

    Reads ``<ASSET>/matches.parquet`` and the key and covariate columns of ``<ASSET>/events.parquet`` (families
    treated and family 1 only). The placebo side is looked up in ``placebo_asset``'s events (cross-asset cells).
    Unmatched treated events are dropped. Raises if a pair has a missing or non-finite covariate."""
    import polars as pl

    from vecopen.placebo import MATCH_COVARIATES, P1

    cols = [*_EVENT_COV_COLS, *MATCH_COVARIATES]
    cov = pl.concat([pl.scan_parquet(events_dir / a / "events.parquet").select(cols)
                     .filter(pl.col("family").is_in([TREATED, P1])).collect() for a in assets])
    mm = pl.concat([pl.read_parquet(events_dir / a / "matches.parquet", columns=list(_MATCH_COLS)) for a in assets])
    mm = mm.filter(pl.col("placebo_key").is_not_null())
    t = cov.filter(pl.col("family") == TREATED).drop("family").rename({c: f"t_{c}" for c in MATCH_COVARIATES})
    p = (cov.filter(pl.col("family") == P1).drop("family")
         .rename({"asset": "placebo_asset", **{c: f"p_{c}" for c in MATCH_COVARIATES}}))
    out = (mm.join(t, left_on=["asset", "treated_key"], right_on=["asset", "event_key"], how="left")
           .join(p, left_on=["placebo_asset", "placebo_key"], right_on=["placebo_asset", "event_key"], how="left"))
    val = [f"{s}_{c}" for s in ("t", "p") for c in MATCH_COVARIATES]
    bad = out.filter(pl.any_horizontal([pl.col(c).is_null() | ~pl.col(c).is_finite() for c in val]))
    if bad.height:
        raise ValueError(f"{bad.height} matched pairs lack a finite covariate on one side (key mismatch?)")
    return out.select(["asset", "tf", *val])


def _balance_entry(mom: dict[str, tuple[Any, Any]]) -> dict[str, Any]:
    from vecopen.placebo import MATCH_COVARIATES, smd_moments

    covs = {}
    for c in MATCH_COVARIATES:
        t, p = mom[c]
        covs[c] = {"mean_t": t.mean, "mean_p": p.mean, "sd_t": math.sqrt(t.var) if t.n > 1 else None,
                   "sd_p": math.sqrt(p.var) if p.n > 1 else None, "smd": smd_moments(t, p)}
    vals = [v["smd"] for v in covs.values()]
    mx = max(abs(v) for v in vals) if all(math.isfinite(v) for v in vals) else float("nan")
    n = mom[MATCH_COVARIATES[0]][0].n
    return {"n_pairs": n, "covariates": covs, "max_abs_smd": mx,
            "pass": bool(math.isfinite(mx) and mx < BALANCE_SMD_MAX)}


def pooled_balance(pairs: Any, groups: dict[str, tuple[str, ...]] = TF_GROUPS) -> dict[str, Any]:
    """Matched-pair SMDs pooled with ``placebo.Moments`` / ``smd_moments``: per timeframe group (over every asset
    and timeframe of the group), per timeframe (the breakdown) and over everything (``all``). Each (asset, timeframe)
    cell is accumulated once and merged into its timeframe, its group and ``all``."""
    from vecopen.placebo import MATCH_COVARIATES, Moments

    tf_of = {tf: g for g, tfs in groups.items() for tf in tfs}
    unknown = sorted(set(pairs["tf"].unique().to_list()) - set(tf_of))
    if unknown:
        raise ValueError(f"timeframes in no group: {unknown}")

    def fresh() -> dict[str, tuple[Any, Any]]:
        return {c: (Moments(), Moments()) for c in MATCH_COVARIATES}

    by_tf = {tf: fresh() for g in groups for tf in groups[g]}
    by_group = {g: fresh() for g in groups}
    overall = fresh()
    for (_, tf), cell in pairs.partition_by(["asset", "tf"], as_dict=True).items():
        for c in MATCH_COVARIATES:
            t, p = Moments(), Moments()
            t.add(cell[f"t_{c}"].to_numpy())
            p.add(cell[f"p_{c}"].to_numpy())
            for tgt in (by_tf[tf], by_group[tf_of[tf]], overall):
                tgt[c][0].merge(t.n, t.s, t.ss)
                tgt[c][1].merge(p.n, p.s, p.ss)
    out_g = {g: {"timeframes": list(groups[g]), **_balance_entry(by_group[g])} for g in groups}
    out_tf = {tf: {"group": tf_of[tf], **_balance_entry(m)} for tf, m in by_tf.items() if m[MATCH_COVARIATES[0]][0].n}
    return {"groups": out_g, "by_tf": out_tf, "all": _balance_entry(overall),
            "threshold": BALANCE_SMD_MAX, "met": all(v["pass"] for v in out_g.values())}


def group_balance_md(bal: dict[str, Any], provenance: list[str] | None = None) -> str:
    from vecopen.placebo import MATCH_COVARIATES

    hdr = "| pairs | " + " | ".join(f"SMD {c}" for c in MATCH_COVARIATES) + " | max abs SMD | below 0.1 |"
    sep = "|---|" + "---|" * (len(MATCH_COVARIATES) + 3)

    def row(e: dict[str, Any]) -> str:
        return (f"{e['n_pairs']:,} | " + " | ".join(_f(e["covariates"][c]["smd"]) for c in MATCH_COVARIATES)
                + f" | {_f(e['max_abs_smd'])} | {'yes' if e['pass'] else '**no**'} |")

    L = ["# Phase 3 amendment: covariate balance per timeframe group", "",
         "Generated by `uv run --extra vecopen python -m vecopen phase3-balance`. Do not edit by hand. Covariates only: "
         "matched treated first fills vs their matched family-1 first fills, read from the stored phase 3 outputs "
         "(no event recomputed, no label or level table read). SMD = (mean_T - mean_C) / sqrt((var_T + var_C) / 2), "
         "pooled with `placebo.Moments` over every asset and timeframe of the group; a placebo used twice counts twice. "
         "Groups per SPEC section 9 (user decision, 2026-11-14).", ""]
    if provenance:
        L += [*provenance, ""]
    L += ["## Groups (acceptance: max |SMD| < 0.1 in each group)", "", "| group | timeframes " + hdr,
          "|---" + sep]
    for g, e in bal["groups"].items():
        L.append(f"| {g} | {', '.join(e['timeframes'])} | " + row(e))
    L += ["", f"Acceptance {'met' if bal['met'] else '**not met**'} in "
          + ", ".join(f"{g}: {'yes' if e['pass'] else 'no'}" for g, e in bal["groups"].items()) + ".", "",
          "## Breakdown per timeframe", "", "| tf | group " + hdr, "|---" + sep]
    for tf, e in bal["by_tf"].items():
        L.append(f"| {tf} | {e['group']} | " + row(e))
    L.append("| **all** | | " + row(bal["all"]))
    L += ["", "## Means and SDs per group", "",
          "| group | covariate | mean treated | mean placebo | SD treated | SD placebo | SMD |", "|---|---|---|---|---|---|---|"]
    for g, e in bal["groups"].items():
        for c in MATCH_COVARIATES:
            v = e["covariates"][c]
            L.append(f"| {g} | {c} | {_f(v['mean_t'])} | {_f(v['mean_p'])} | {_f(v['sd_t'])} | {_f(v['sd_p'])} | "
                     f"{_f(v['smd'])} |")
    return "\n".join(L) + "\n"


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_phase3_inputs(events_dir: Path, source: dict[str, Any]) -> tuple[list[str], str]:
    """Check that the stored events and matches tables are byte-identical to the authoritative phase 3 run: the
    manifest's sha256 equals ``source``'s ``manifest_sha256`` and every file read equals its manifest sha256.
    Returns the assets and the manifest sha256."""
    mp = events_dir / "manifest.json"
    msha = _sha256_file(mp)
    if msha != source["metrics"]["manifest_sha256"]:
        raise ValueError(f"manifest.json sha256 {msha} differs from registry line {source['line_sha256'][:12]}")
    man = json.loads(mp.read_text(encoding="utf-8"))
    assets = list(source["metrics"]["assets"])
    if sorted(man["assets"]) != sorted(assets):
        raise ValueError("manifest assets differ from the registry line")
    for a in assets:
        for name in ("events", "matches"):
            want = man["assets"][a][name]["sha256"]
            if _sha256_file(events_dir / a / f"{name}.parquet") != want:
                raise ValueError(f"{a}/{name}.parquet differs from the manifest of the authoritative phase 3 run")
    return assets, msha


def cmd_phase3_balance(args: argparse.Namespace) -> int:
    from vecopen.__main__ import run_tests
    from vecopen.phase2 import peak_rss_mb
    from vecopen.placebo import MATCH_COVARIATES

    cfg = load_config()
    t0 = time.perf_counter()
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    grouped = [tf for g in TF_GROUPS.values() for tf in g]
    if sorted(grouped) != sorted(tfs) or len(set(grouped)) != len(grouped):
        raise ValueError(f"TF_GROUPS {TF_GROUPS} do not partition timeframes_primary {tfs}")
    src = [r for r in registry.read() if r["line_sha256"] == PHASE3_AUTHORITATIVE_LINE]
    if len(src) != 1:
        raise ValueError("authoritative phase 3 registry line not found")
    src = src[0]
    assets, msha = verify_phase3_inputs(EVENTS_DIR, src)
    pairs = load_matched_covariates(EVENTS_DIR, assets)
    bal = pooled_balance(pairs)
    del pairs
    # cross-check: the per-timeframe breakdown reproduces the stored phase 3 balance table
    ref = src["metrics"]["balance"]
    diffs = [abs(bal["by_tf"][tf]["covariates"][c]["smd"] - ref[tf][c]["smd"]) for tf in bal["by_tf"]
             for c in MATCH_COVARIATES]
    diffs += [abs(bal["all"]["covariates"][c]["smd"] - ref["all"][c]["smd"]) for c in MATCH_COVARIATES]
    same_n = all(bal["by_tf"][tf]["n_pairs"] == ref[tf][MATCH_COVARIATES[0]]["n"] for tf in bal["by_tf"])
    xmax = max(diffs)
    if xmax > 1e-9 or not same_n:
        raise ValueError(f"per-timeframe balance does not reproduce registry line {src['seq'] + 1} "
                         f"(max |diff| {xmax:.3g}, same pair counts {same_n})")
    prov = [f"Source: phase 3 registry line {src['seq'] + 1} (`line_sha256 {src['line_sha256']}`); "
            f"`manifest.json` sha256 `{msha}`; the {len(assets)} assets' `events.parquet` and `matches.parquet` "
            "match the manifest byte for byte.",
            f"Cross-check: the per-timeframe breakdown reproduces that line's balance table (max |SMD difference| "
            f"{xmax:.1e}, identical pair counts)."]
    md = group_balance_md(bal, prov)
    GROUP_BALANCE_PATH.write_text(md, encoding="utf-8", newline="\n")
    print(md)
    counts = {"tests": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "pytest_exit_code": 0}
    if not args.skip_tests:
        counts = run_tests(PROJECT_ROOT / "tests")
    print(f"tests: {counts}")
    green = counts["pytest_exit_code"] == 0 and counts["failed"] == 0 and counts["errors"] == 0
    metrics = {"amendment": "phase 3 covariate balance per timeframe group (SPEC section 9, user decision "
                            "2026-11-14); covariates only, from stored outputs",
               "group_balance": bal["groups"], "by_tf": bal["by_tf"], "all": bal["all"],
               "threshold": BALANCE_SMD_MAX, "acceptance_met": bal["met"],
               "source_line_sha256": src["line_sha256"], "manifest_sha256": msha,
               "crosscheck_by_tf_max_abs_diff": xmax,
               "tables_sha256": hashlib.sha256(md.encode("utf-8")).hexdigest(), "tests": counts,
               "runtime_seconds": round(time.perf_counter() - t0, 1), "peak_rss_mb": round(peak_rss_mb(), 1)}
    metrics = json.loads(json.dumps(metrics, default=_jsonable, allow_nan=True), parse_constant=lambda _: None)
    if args.dry_run:
        print("dry run: registry not written")
    else:
        rec = registry.append(3, metrics, kind="build", config=cfg,
                              note="phase 3 amendment: matched-sample covariate balance per timeframe group "
                                   "(intraday 15m-4h, daily_plus 1d-1M), from the stored outputs of line "
                                   f"{src['seq'] + 1}; no outcome read, matching unchanged")
        print(f"registry line {rec['seq'] + 1} appended (kind build): line_sha256 {rec['line_sha256']} "
              f"git_commit {rec['git_commit']} diff_sha256 {rec['diff_sha256']}")
    print(f"group balance acceptance {'met' if bal['met'] else 'NOT met'}; peak RSS {peak_rss_mb():.0f} MB")
    return 0 if green and bal["met"] else 1


def add_commands(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("phase3", help="phase 3 reproduce command: events, labels, placebos, matching, tables")
    p.add_argument("--assets", default="", help="comma-separated subset (default: all crypto symbols)")
    p.add_argument("--dry-run", action="store_true", help="do not append a registry line")
    p.add_argument("--skip-tests", action="store_true")
    p.set_defaults(fn=cmd_phase3)
    b = sub.add_parser("phase3-balance", help="phase 3 amendment: covariate balance per timeframe group, from the "
                                              "stored phase 3 outputs (covariates only)")
    b.add_argument("--dry-run", action="store_true", help="do not append a registry line")
    b.add_argument("--skip-tests", action="store_true")
    b.set_defaults(fn=cmd_phase3_balance)
