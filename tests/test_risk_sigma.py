"""ai_trader.risk.sigma (decision D1): the one stop-sigma entry point, its frame routing, the shadow scorer and the gate.

1. Regression: with spec robust_bc the routed frame is bitwise the pre-D1 add_robust_vol (and the real lake frame too).
2. Point in time: store rows / gate decisions known after as_of never change a sigma at as_of.
3. The clock, release and range terms are in the model (design width, signs) and the forecaster reproduces WS04.
4. No weekend shrink in the stop sigma (clock floored on weekend bars; robust_bc runs use_weekend=False).
5. Switch semantics, pinned base consumers, shadow scorer and the latched gate.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ai_trader.research.frame import add_robust_vol, enrich_bars
from ai_trader.risk import sigma as S
from ai_trader.risk import sigma_shadow as sh
from ai_trader.risk import sigma_store as st

T0 = datetime(2024, 1, 1, tzinfo=UTC)
CFG = {**S.DEFAULTS, "nu": dict(S.NU)}


def old_add_robust_vol(df: pl.DataFrame) -> pl.DataFrame:
    """research.frame.add_robust_vol as it was before D1 (commit 76e0caf), verbatim."""
    from ai_trader.features.vol_models import VolModel, run
    from ai_trader.research.calendar import add_calendar

    if df.height < 40:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in ("sigma_rb_next", "sigma_risk_next", "wk_m"))
    wk = add_calendar(df.select("event_ts"))["tradfi_weekend"].to_numpy()
    nxt_wk = add_calendar(df.select(pl.col("close_ts").alias("event_ts")))["tradfi_weekend"].to_numpy()
    r = np.nan_to_num(df["ret"].to_numpy())
    seasonal = run(r, wk, VolModel(bias_correct=True))
    risk = run(r, wk, VolModel(bias_correct=True, use_weekend=False))
    return df.with_columns(
        pl.Series("sigma_rb_next", np.sqrt(seasonal["v_next"]) * np.where(nxt_wk, seasonal["m"], 1.0)),
        pl.Series("sigma_risk_next", np.sqrt(risk["v_next"])),
        pl.Series("wk_m", seasonal["m"]),
    )


def _cfg(**kw) -> dict:
    return {**CFG, **kw}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolated 5m store, journal and gate; caches cleared."""
    d = tmp_path / "rv1h"
    d.mkdir()
    monkeypatch.setattr(st, "store_dir", lambda: d)
    j = tmp_path / "journal"
    j.mkdir()
    monkeypatch.setattr(sh, "journal_dir", lambda: j)
    monkeypatch.setattr(S, "gate_path", lambda: j / "gate.jsonl")
    monkeypatch.setattr(S, "_PROCESS_BASE", False)
    S._series_cached.cache_clear()
    S._gate_lines.cache_clear()
    yield tmp_path
    S._series_cached.cache_clear()


def synth_hours(start: datetime, n: int, seed: int = 0, clock=None, rel_mult: float = 1.0) -> pl.DataFrame:
    """Hourly store rows: persistent log-vol, optional hour-of-week clock multiplier (array of 168), release multiplier."""
    rng = np.random.default_rng(seed)
    ts = pl.datetime_range(start, start + timedelta(hours=n - 1), "1h", time_zone="UTC", eager=True).cast(pl.Datetime("us", "UTC"))
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = 0.98 * lv[i - 1] + 0.12 * rng.normal()
    var = 4e-5 * np.exp(lv)
    if clock is not None:
        from ai_trader.research.vol_spec import cells

        var = var * np.asarray(clock)[cells(ts, "1h")]
    if rel_mult != 1.0:
        from ai_trader.research.vol_spec import release_flags

        var = var * np.where(release_flags(ts, 1), rel_mult, 1.0)
    rv = var * rng.chisquare(12, n) / 12
    r = np.sqrt(var) * rng.standard_t(5, n) / math.sqrt(5 / 3)
    close = 100 * np.exp(np.cumsum(r))
    opn = np.r_[100.0, close[:-1]]
    sd = np.sqrt(rv)
    high = np.maximum(opn, close) * np.exp(np.abs(rng.normal(size=n)) * sd)
    low = np.minimum(opn, close) * np.exp(-np.abs(rng.normal(size=n)) * sd)
    return pl.DataFrame({"event_ts": ts, "open": opn, "high": high, "low": low, "close": close, "rv": rv,
                         "n5": np.full(n, 12, dtype=np.int64)}).with_columns(
        (pl.col("event_ts") + pl.duration(hours=1, seconds=2)).alias("known_ts")).select(*st.COLS)


def bars_from_hours(h: pl.DataFrame) -> pl.DataFrame:
    return h.select("event_ts", (pl.col("event_ts") + pl.duration(hours=1)).alias("close_ts"),
                    (pl.col("event_ts") + pl.duration(hours=1, seconds=2)).alias("known_ts"), "open", "high", "low", "close",
                    pl.lit(1.0).alias("volume"), pl.lit(1.0).alias("turnover"))


# ====================================================================== 1. regression


def test_regression_frame_identical_under_robust_bc(store, monkeypatch):
    monkeypatch.setattr(S, "settings", lambda: _cfg(spec="robust_bc", switch_on=None))
    h = synth_hours(T0, 24 * 60, seed=1)
    st._write("SYNUSDT", h)
    b = enrich_bars(bars_from_hours(h))
    new = add_robust_vol(b, symbol="SYNUSDT", freq="1h", as_of=T0 + timedelta(days=61))
    old = old_add_robust_vol(b)
    for c in old.columns:
        assert new[c].equals(old[c], check_dtypes=True), c          # bitwise, NaN-aware
    assert np.array_equal(new["sigma_rbc_next"].to_numpy(), new["sigma_risk_next"].to_numpy(), equal_nan=True)
    # no symbol (hand-built / aggregated frames): always robust_bc, even if a new spec is configured
    monkeypatch.setattr(S, "settings", lambda: _cfg(spec="loghar_cr_range"))
    assert add_robust_vol(b)["sigma_risk_next"].equals(old["sigma_risk_next"])


def test_regression_real_lake_frame(store, monkeypatch):
    """The real BTC 1h market frame (if the lake is here) equals the pre-D1 computation on the same bars."""
    from ai_trader.research.frame import market_frame
    from ai_trader.storage.lake import Lake

    if not Lake().has_data("bars"):
        pytest.skip("no lake")
    as_of = datetime(2026, 11, 20, tzinfo=UTC)
    f = market_frame("BTCUSDT", "1h", as_of, start=as_of - timedelta(days=200), funding=False, oi=False)
    if f.height < 1000:
        pytest.skip("no BTC 1h history")
    base = f.select("event_ts", "close_ts", "known_ts", "open", "high", "low", "close", "volume", "turnover")
    old = old_add_robust_vol(enrich_bars(base))
    for c in ("sigma_risk_next", "sigma_rb_next", "wk_m"):
        assert f[c].equals(old[c]), c
    assert S.spec_in_force(as_of) == "robust_bc"      # the shipped config: robust_bc until switch_on (2026-12-11)


def test_1d_is_vol_har(store):
    """At 1d both new specs are vol.har: the point route equals the vol.har tool when the previous UTC day is complete."""
    from ai_trader.storage.lake import Lake
    from ai_trader.tools.vol import har

    if not Lake().has_data("bars"):
        pytest.skip("no lake")
    as_of = datetime(2026, 11, 20, 0, 30, tzinfo=UTC)
    try:
        ref = har("BTCUSDT", as_of=as_of).value["sigma_next_day"]
    except ValueError:
        pytest.skip("no BTC 1h history")
    for sp in ("loghar_cr", "loghar_cr_range"):
        r = S.sigma_risk("BTCUSDT", "1d", as_of, spec=sp, fallback=False)
        assert r.fallback is None and abs(r.sigma / ref - 1) < 1e-12 and r.bar_start == datetime(2026, 11, 20, tzinfo=UTC)


# ====================================================================== 2. point in time


def test_point_in_time_store_and_frame(store, monkeypatch):
    monkeypatch.setattr(S, "settings", lambda: _cfg())
    h = synth_hours(T0, 24 * 430, seed=2)
    as_of = T0 + timedelta(days=420, hours=5, minutes=3)
    st._write("SYNUSDT", h.filter(pl.col("known_ts") <= as_of))
    a = S.sigma_risk("SYNUSDT", "1h", as_of, spec="loghar_cr_range", fallback=False)
    a4 = S.sigma_risk("SYNUSDT", "4h", as_of, spec="loghar_cr", fallback=False)
    assert a.fallback is None and np.isfinite(a.sigma) and a.bar_start == as_of.replace(minute=0)
    assert a4.bar_start == as_of.replace(hour=4, minute=0)
    # the future arrives (with absurd values): nothing at as_of may change
    fut = h.filter(pl.col("known_ts") > as_of).with_columns(pl.col("rv") * 1e4, pl.col("high") * 2.0)
    st._write("SYNUSDT", pl.concat([h.filter(pl.col("known_ts") <= as_of), fut]))
    S._series_cached.cache_clear()
    b = S.sigma_risk("SYNUSDT", "1h", as_of, spec="loghar_cr_range", fallback=False)
    b4 = S.sigma_risk("SYNUSDT", "4h", as_of, spec="loghar_cr", fallback=False)
    assert b.sigma == a.sigma and b4.sigma == a4.sigma
    # frame route: rows known at as_of are unchanged; the point equals the frame's last row
    bars = enrich_bars(bars_from_hours(h.filter(pl.col("known_ts") <= as_of)))
    v, spec = S.stop_sigma_column(old_add_robust_vol(bars).with_columns(pl.col("sigma_risk_next").alias("sigma_rbc_next")),
                                  symbol="SYNUSDT", freq="1h", as_of=as_of, spec="loghar_cr_range")
    assert spec == "loghar_cr_range" and abs(v[-1] / a.sigma - 1) < 1e-12


def test_point_equals_series_and_fallback(store, monkeypatch):
    monkeypatch.setattr(S, "settings", lambda: _cfg())
    h = synth_hours(T0, 24 * 450, seed=3)
    st._write("SYNUSDT", h)
    as_of = h["known_ts"].max()
    for g in ("1h", "4h"):
        for sp in ("loghar_cr", "loghar_cr_range"):
            p = S.sigma_risk("SYNUSDT", g, as_of, spec=sp, fallback=False)
            ser = S._series_cached("SYNUSDT", g, sp, int(as_of.timestamp() * 1e6), S._store_mtime("SYNUSDT"))
            row = ser.filter(pl.col("event_ts") == pl.lit(p.bar_start).cast(pl.Datetime("us", "UTC")))
            assert abs(math.sqrt(row["stop"][0]) / p.sigma - 1) < 1e-10, (g, sp)
    # stale store: no forecast for a bar two hours past the data -> NaN without fallback
    late = S.sigma_risk("SYNUSDT", "1h", as_of + timedelta(hours=2), spec="loghar_cr", fallback=False)
    assert not np.isfinite(late.sigma) and late.fallback
    # short history: no fit within a year of the first bar
    st._write("NEWUSDT", synth_hours(T0, 24 * 200, seed=4))
    short = S.sigma_risk("NEWUSDT", "1h", T0 + timedelta(days=199, minutes=3), spec="loghar_cr", fallback=False)
    assert "shorter than a year" in (short.fallback or "")


# ====================================================================== 3. terms in the model


def _fit(G: pl.DataFrame, g: str, spec: str) -> dict:
    out = S.forecast(G, g, spec)
    assert "fit" in out
    return out


def test_design_has_clock_release_and_range_terms(store):
    h = synth_hours(T0, 24 * 420, seed=5)
    G1, G4 = S.grid(h, "1h"), S.grid(h, "4h")
    assert len(_fit(G1, "1h", "loghar_cr")["fit"]["coef"]) == 1 + 5 + 1          # const, 5 HAR lags, release
    assert len(_fit(G1, "1h", "loghar_cr_range")["fit"]["coef"]) == 1 + 5 + 1 + 3  # + range over 1 / 6 / 24 bars
    assert len(_fit(G4, "4h", "loghar_cr")["fit"]["coef"]) == 1 + 4 + 1
    assert len(_fit(G4, "4h", "loghar_cr_range")["fit"]["coef"]) == 1 + 4 + 1 + 2
    # the range terms move the forecast; loghar_cr does not see them
    G1b = G1.with_columns(pl.when(pl.col("event_ts") == G1["event_ts"][-2]).then(pl.col("pk") * 50).otherwise(pl.col("pk")).alias("pk"))
    a, b = S.forecast(G1, "1h", "loghar_cr_range")["stop"][-1], S.forecast(G1b, "1h", "loghar_cr_range")["stop"][-1]
    c, d = S.forecast(G1, "1h", "loghar_cr")["stop"][-1], S.forecast(G1b, "1h", "loghar_cr")["stop"][-1]
    assert b > a and c == d


def test_clock_shapes_the_forecast(store):
    clock = np.ones(168)
    clock[[d * 24 + 14 for d in range(5)]] = 6.0       # 14:00 UTC on weekdays: 6x variance
    h = synth_hours(T0, 24 * 420, seed=6, clock=clock)
    out = S.forecast(S.grid(h, "1h"), "1h", "loghar_cr")
    ts = h["event_ts"]
    m = np.isfinite(out["central"])
    hr = ts.dt.hour().to_numpy()
    wd = ts.dt.weekday().to_numpy()
    busy = m & (hr == 14) & (wd <= 5)
    quiet = m & (hr == 3) & (wd <= 5)
    assert np.median(out["central"][busy]) > 3 * np.median(out["central"][quiet])


def test_release_dummy_is_used(store):
    from ai_trader.research.vol_spec import release_flags

    try:
        release_flags(pl.Series([T0]).cast(pl.Datetime("us", "UTC")), 1)
    except Exception:
        pytest.skip("no release schedule (data/research/news/events/events.parquet)")
    h = synth_hours(T0, 24 * 500, seed=7, rel_mult=12.0)
    out = S.forecast(S.grid(h, "1h"), "1h", "loghar_cr")
    rel = release_flags(h["event_ts"], 1).astype(bool)
    assert out["fit"]["coef"][-1] > 1.0                  # log-variance loading of the release dummy (true: log 12 = 2.5)
    m = np.isfinite(out["central"])
    assert (rel & m).sum() >= 5
    assert np.median(out["central"][rel & m]) > 4 * np.median(out["central"][~rel & m])


def test_reproduces_ws04_loghar_cr():
    """The forecaster equals WS04's saved loghar_cr (vol_spec.forecast_grid) on WS04's own grid, if present."""
    from ai_trader.core.config import load

    d = load().root / "data" / "research" / "ws04"
    gp, fp = d / "grid_BTCUSDT_4h.parquet", d / "fc_BTCUSDT_4h.parquet"
    if not (gp.is_file() and fp.is_file()):
        pytest.skip("WS04 outputs not present")
    G = pl.read_parquet(gp).select("event_ts", "rv", pl.lit(None, dtype=pl.Float64).alias("pk"))
    out = S.forecast(G, "4h", "loghar_cr", start=datetime(2023, 1, 1, tzinfo=UTC))
    fc = pl.read_parquet(fp).select("event_ts", "loghar_cr")
    m = pl.DataFrame({"event_ts": G["event_ts"], "mine": out["central"]}).join(fc, on="event_ts").drop_nans().drop_nulls()
    assert m.height > 5000
    assert float(((m["mine"] / m["loghar_cr"]) - 1).abs().max()) < 1e-12


# ====================================================================== 4. no weekend shrink in the stop sigma


def test_no_weekend_shrink_in_stop_sigma(store):
    from ai_trader.research.calendar import add_calendar

    clock = np.ones(168)
    clock[5 * 24:] = 0.25                               # Saturday + Sunday: a quarter of the weekday variance
    h = synth_hours(T0, 24 * 420, seed=8, clock=clock)
    out = S.forecast(S.grid(h, "1h"), "1h", "loghar_cr_range")
    ts = h["event_ts"]
    wk = add_calendar(pl.DataFrame({"event_ts": ts}))["tradfi_weekend"].to_numpy().astype(bool)
    m = np.isfinite(out["stop"])
    assert np.allclose(out["stop"][m & ~wk], out["central"][m & ~wk])            # weekdays: the clock as fitted
    assert (out["stop"][m & wk] >= out["central"][m & wk] - 1e-18).all()         # weekend: never below the central
    sat = m & (ts.dt.weekday().to_numpy() == 6)
    # the 0.25 weekend shrink is undone: true ratio 4, fitted ~2.5 because the clock is shrunk toward its mean (kappa 20)
    assert np.median(out["stop"][sat] / out["central"][sat]) > 2.0
    # the robust_bc stop sigma is the use_weekend=False run (no weekend multiplier), unlike the central sigma_rb_next
    f = add_robust_vol(enrich_bars(bars_from_hours(h.head(24 * 60))))
    from ai_trader.features.vol_models import VolModel, run

    wk2 = add_calendar(f.select("event_ts"))["tradfi_weekend"].to_numpy()
    ref = np.sqrt(run(np.nan_to_num(f["ret"].to_numpy()), wk2, VolModel(bias_correct=True, use_weekend=False))["v_next"])
    assert np.array_equal(f["sigma_risk_next"].to_numpy(), ref, equal_nan=True)


def test_weekend_4h_bar_touching_weekend_is_floored():
    ts = pl.Series([datetime(2026, 11, 27, 20, tzinfo=UTC), datetime(2026, 11, 27, 12, tzinfo=UTC)]).cast(pl.Datetime("us", "UTC"))
    wk = S.weekend_bars(ts, "4h")
    assert wk.tolist() == [True, False]      # Fri 20-24 UTC touches the TradFi weekend (Fri 17:00 New York)


# ====================================================================== 5. switch, roles, shadow, gate


def test_spec_in_force_switch_and_latched_gate(store):
    cfg = _cfg(spec="robust_bc", shadow="loghar_cr_range", switch_on="2026-12-11", switch_gate="score")
    before, after = datetime(2026, 12, 10, 23, tzinfo=UTC), datetime(2026, 12, 11, 1, tzinfo=UTC)
    assert S.spec_in_force(before, cfg) == "robust_bc"
    assert S.spec_in_force(after, cfg) == "robust_bc"                    # no gate decision yet
    gp = S.gate_path()
    gp.write_text(json.dumps({"switch_on": "2026-12-11", "decided_at": "2026-12-11T00:02:30+00:00", "pass": True, "shadow": "loghar_cr_range"}) + "\n"
                  + json.dumps({"switch_on": "2026-12-11", "decided_at": "2026-12-11T05:02:30+00:00", "pass": False, "shadow": "loghar_cr_range"}) + "\n")
    assert S.spec_in_force(datetime(2026, 12, 11, 0, 1, tzinfo=UTC), cfg) == "robust_bc"   # decided after as_of
    assert S.spec_in_force(after, cfg) == "loghar_cr_range"
    assert S.spec_in_force(datetime(2026, 12, 20, tzinfo=UTC), cfg) == "loghar_cr_range"   # latched: later lines ignored
    assert S.spec_in_force(after, {**cfg, "switch_gate": "none"}) == "loghar_cr_range"
    assert S.spec_in_force(after, {**cfg, "switch_on": "2026-12-12"}) == "robust_bc"
    assert S.spec_in_force(before, {**cfg, "spec": "loghar_cr"}) == "loghar_cr"            # flipped by hand
    with pytest.raises(ValueError):
        S.sigma_risk("X", "1h", after, spec="garch")


def test_switch_moves_stops_but_not_the_pinned_base(store, monkeypatch):
    """With a new spec in force the frame's sigma_risk_next changes; the networks' s2 (ml.hourly.symbol_hourly) and the
    robust_bc tool column do not."""
    from ai_trader.ml.hourly import symbol_hourly

    h = synth_hours(T0, 24 * 420, seed=9)
    st._write("SYNUSDT", h)
    b = enrich_bars(bars_from_hours(h))
    as_of = h["known_ts"].max()
    monkeypatch.setattr(S, "settings", lambda: _cfg(spec="robust_bc"))
    f0 = add_robust_vol(b, symbol="SYNUSDT", freq="1h", as_of=as_of)
    monkeypatch.setattr(S, "settings", lambda: _cfg(spec="loghar_cr_range"))
    f1 = add_robust_vol(b, symbol="SYNUSDT", freq="1h", as_of=as_of)
    moved = (f1["sigma_risk_next"] != f0["sigma_risk_next"]).sum()
    assert moved > 24 * 30                                               # the stop sigma switched after the first refit
    assert f1["sigma_rbc_next"].equals(f0["sigma_rbc_next"])
    assert f1["sigma_risk_next"][: 24 * 300].equals(f0["sigma_risk_next"][: 24 * 300])   # before the first refit: robust_bc
    assert symbol_hourly(S.pin_base(f1)).equals(symbol_hourly(f0))        # networks' inputs pinned (callers pass pin_base)
    assert S.pin_base(f1)["sigma_risk_next"].equals(f0["sigma_risk_next"])
    # process pin (scripts/live_nn.py): every frame keeps robust_bc even with a new spec in force
    try:
        S.pin_process_base()
        f2 = add_robust_vol(b, symbol="SYNUSDT", freq="1h", as_of=as_of)
    finally:
        S.pin_process_base(False)
    assert f2["sigma_risk_next"].equals(f0["sigma_risk_next"])


def test_hash_guarded_network_sources_untouched():
    """xs_v3 / xs_v4 / live_nn hash these sources (train/serve skew guard, logged code hashes): D1 must not edit them."""
    from ai_trader.core.config import load

    root = load().root
    spec = root / "data" / "models" / "xs_v3" / "frozen" / "spec.json"
    if not spec.is_file():
        pytest.skip("no xs_v3 frozen spec")
    import hashlib

    for f, h in json.loads(spec.read_text(encoding="utf-8"))["code_sha256"].items():
        assert hashlib.sha256((root / f).read_bytes()).hexdigest() == h, f
    for f in ("src/ai_trader/live/nn.py", "src/ai_trader/live/nn_v2.py", "src/ai_trader/live/xs_v3.py", "src/ai_trader/ml/hourly.py"):
        assert "risk.sigma" not in (root / f).read_text(encoding="utf-8"), f


def test_store_hours_from_5m():
    t = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    n = 12 * 3 + 1
    ts = pl.datetime_range(t - timedelta(minutes=5), t - timedelta(minutes=5) + timedelta(minutes=5 * (n - 1)), "5m", time_zone="UTC", eager=True)
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    d5 = pl.DataFrame({"event_ts": ts, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c}).head(n - 6)  # last hour incomplete
    h = st.hours_from_5m(d5, first_hour=t)
    assert h["event_ts"].to_list() == [t, t + timedelta(hours=1)]          # 12:00 hour is incomplete in time
    r = np.diff(np.log(c))
    assert abs(h["rv"][0] - float((r[:12] ** 2).sum())) < 1e-15
    assert h["known_ts"][0] == t + timedelta(hours=1, seconds=2)


def test_shadow_scoring_and_gate_latch(store, monkeypatch):
    monkeypatch.setattr(S, "settings", lambda: _cfg(switch_on="2025-02-10"))
    h = synth_hours(T0, 24 * 410, seed=10)
    for s in ("AUSDT", "BUSDT"):
        st._write(s, h)
    lines = []
    for k in range(3):
        bar = T0 + timedelta(days=400, hours=k)
        for s in ("AUSDT", "BUSDT"):
            lines.append({"symbol": s, "run_ts": (bar + timedelta(minutes=2)).isoformat(), "as_of": (bar + timedelta(minutes=2)).isoformat(),
                          "h": {"1h": {"bar_start": bar.isoformat(), "late": False, "robust_bc": 0.01, "loghar_cr": 0.006, "loghar_cr_central": 0.006,
                                       "loghar_cr_range": 0.0063, "loghar_cr_range_central": 0.0063}}})
    lines.append({**lines[0], "run_ts": "2099-01-01T00:00:00+00:00", "h": {"1h": {**lines[0]["h"]["1h"], "robust_bc": 9.0}}})  # duplicate: later
    lines.append({**lines[1], "replay": True, "h": {"1h": {**lines[1]["h"]["1h"], "bar_start": (T0 + timedelta(days=401)).isoformat()}}})
    df = sh.scored_table(T0 + timedelta(days=405), lines=lines)
    assert df.height == 6 and df.filter(pl.col("robust_bc") > 1).height == 0       # first line kept, replay ignored
    one = df.row(0, named=True)
    from ai_trader.research.vol_spec import qlike

    d = sh.pooled_dm(df, "1h", "loghar_cr", "robust_bc")
    assert d["n_bars"] == 3
    ref = float(np.mean([np.mean(qlike(df.filter(pl.col("bar_start") == b)["rv"].to_numpy(), 0.006 ** 2)) for b in df["bar_start"].unique()]))
    assert abs(d["qlike_a"] - ref) < 1e-12 and one["rv"] > 0
    # gate: too few data -> a 'pass: false, insufficient' decision, appended once and latched
    summ = sh.summary(df, S.settings())
    now = datetime(2025, 2, 11, tzinfo=UTC)
    g1 = sh.maybe_gate(summ, now, S.settings())
    g2 = sh.maybe_gate(summ, now + timedelta(hours=1), S.settings())
    lines_g = S.gate_path().read_text().splitlines()
    assert g1["status"] == "decided now" and g1["insufficient"] and not g1["pass"]
    assert g2["status"] == "decided" and len(lines_g) == 1
    assert S.spec_in_force(now + timedelta(hours=2), S.settings()) == "robust_bc"


def test_gate_rule_registered_thresholds():
    def summ(n1, n4, d1, t1, d4, var):
        return {"horizons": {"1h": {"dm_gate": {"loghar_cr_range-robust_bc": {"n_bars": n1, "diff": d1, "t_hln": t1, "rel": -0.3}},
                                    "var_1pct_lower": {"loghar_cr_range": {"all": {"rate": var, "n": 5000}}}},
                             "4h": {"dm_gate": {"loghar_cr_range-robust_bc": {"n_bars": n4, "diff": d4, "t_hln": -1.0, "rel": -0.3}}}}}

    assert sh.gate_rule(summ(200, 50, -0.1, -3.0, -0.05, 0.012))[0] is True
    assert sh.gate_rule(summ(100, 50, -0.1, -3.0, -0.05, 0.012))[0] is None      # < 168 pooled 1h bars
    assert sh.gate_rule(summ(200, 50, -0.1, -1.5, -0.05, 0.012))[0] is False     # t not < -2
    assert sh.gate_rule(summ(200, 50, -0.1, -3.0, +0.05, 0.012))[0] is False     # 4h worse
    assert sh.gate_rule(summ(200, 50, -0.1, -3.0, -0.05, 0.025))[0] is False     # VaR breached > 2%
