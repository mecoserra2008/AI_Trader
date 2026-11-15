"""Gate G1 leak tests (SPEC section 6, automated leak tests 1 to 3), written by the leak auditor.

Fast tests (always run):
* static audit: the streaming modules cannot reach a data file, a frame loader or the label module, and keep no
  mutable module-level state; only the phase 3 command imports ``vecopen.labels``; the event and matching table
  builders never read a label;
* the harness pipeline is the ``phase3.event_step`` loop (identical outputs on a real slice), and the label step does
  not mutate events;
* the session conventions come from the config (rule 4);
* the canonical state digest is exact, survives pickling, and rejects non-closed state;
* tests 1 to 3 pass on a real slice used as the full data, and non-vacuity: the forbidden open-time join and a
  one-bar look-ahead are both caught; the audit flags outputs stamped late or holding a future timestamp.

Real-data tests (all 20 crypto assets, 200 timestamps each) are run by the reproduce command
``uv run --extra vecopen python vecopen/tests/leak/run_g1.py``. Here they run only with ``VECOPEN_LEAK_FULL=1``,
because each asset takes minutes and ``vecopen/tests`` is run by every phase's reproduce command.
"""

from __future__ import annotations

import ast
import inspect
import os
import pickle
import re
from pathlib import Path

import leakkit as lk
import polars as pl
import pytest
from vecopen.bars import load_sessions, resample
from vecopen.config import PROJECT_ROOT, load_config
from vecopen.data import load_bars, research_path
from vecopen.legs import Bar
from vecopen.levels import LevelBorn, bar_stream

SRC = PROJECT_ROOT / "src" / "vecopen"
STREAMING = ("legs.py", "levels.py", "events.py", "placebo.py")
CFG = load_config()
ASSETS = list(CFG.data["assets"]["crypto"]["symbols"])
HAVE_DATA = research_path("BTCUSDT").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="research data not present")


def _tree(name: str) -> ast.Module:
    return ast.parse((SRC / name).read_text(encoding="utf-8"))


def _imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
    return out


# ---------------------------------------------------------------- static audit

@pytest.mark.parametrize("name", STREAMING)
def test_streaming_modules_cannot_reach_data_files_or_labels(name):
    tree = _tree(name)
    imps = _imports(tree)
    banned = ("polars", "pandas", "pyarrow", "duckdb", "vecopen.data", "vecopen.labels", "vecopen.split",
              "vecopen.ingest", "vecopen.gaps", "vecopen.phase1", "vecopen.phase2", "vecopen.phase3", "pickle",
              "requests", "urllib", "socket", "os", "glob", "pathlib")
    assert not [m for m in imps if m.split(".")[0] in banned or any(m.startswith(b) for b in banned)], (name, imps)
    io_calls = {"open", "read_parquet", "scan_parquet", "read_csv", "read_bytes", "read_text", "write_parquet",
                "load_bars", "loads", "load", "urlopen", "__import__", "exec", "eval", "globals"}
    called = {(n.func.id if isinstance(n.func, ast.Name) else n.func.attr) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    # load_config is the frozen config (read when no config is passed); it is not data
    assert not (called & io_calls), (name, called & io_calls)


@pytest.mark.parametrize("name", STREAMING)
def test_streaming_modules_keep_no_mutable_module_state(name):
    """Every module-level container is a constant: never mutated, never rebound through ``global``."""
    tree = _tree(name)
    mutable = set()
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            val = n.value
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if isinstance(val, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)) or (
                    isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id in
                    ("dict", "list", "set", "deque", "defaultdict", "Counter")):
                mutable |= {t.id for t in targets if isinstance(t, ast.Name)}
    mutators = {"append", "extend", "insert", "pop", "popitem", "remove", "clear", "update", "add", "discard",
                "setdefault", "appendleft", "sort"}
    for n in ast.walk(tree):
        assert not isinstance(n, (ast.Global, ast.Nonlocal)), (name, "global/nonlocal statement")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            assert not (n.func.value.id in mutable and n.func.attr in mutators), (name, n.func.value.id, n.func.attr)
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.Delete)):
            for t in (n.targets if isinstance(n, (ast.Assign, ast.Delete)) else [n.target]):
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                    assert t.value.id not in mutable, (name, t.value.id)


def test_only_the_phase3_command_imports_labels():
    importers = sorted(p.name for p in SRC.glob("*.py") if p.name != "labels.py"
                       and any(m.endswith("labels") for m in _imports(_tree(p.name))))
    assert importers == ["phase3.py"], importers


def test_event_level_and_matching_builders_never_touch_labels():
    from vecopen import phase3

    for fn in (phase3.event_step, phase3.event_row, phase3.level_row, phase3.match_all):
        src = inspect.getsource(fn)
        assert not re.search(r"label_step|LabelStream|label_row|labels\.parquet|labels_df|\blab\b", src), fn.__name__
    m = inspect.getsource(phase3.match_all)
    assert '"events.parquet"' in m and "columns=cols" in m
    cols_line = re.search(r"cols = \[(.*)\]", m).group(1)
    assert not any(x in cols_line for x in ("fade", "follow", "res", "mae", "entry", "fill_ts", "expire_ts"))


# ---------------------------------------------------------------- the harness is the production loop

def _slice(asset: str, start_day: int, days: int) -> pl.DataFrame:
    return load_bars(asset).select(list(lk.COLS)).slice(288 * start_day, 288 * days)


@needs_data
def test_harness_pipeline_equals_phase3_event_step():
    from vecopen import phase3

    cfg = CFG
    s = load_sessions().for_asset("ETHUSDT")
    sl = _slice("ETHUSDT", 500, 35)
    tfs = list(cfg.data["levels"]["timeframes_primary"])
    higher = {tf: resample(sl, tf, s) for tf in tfs}
    ref = phase3.event_step("ETHUSDT", sl, higher, cfg, s)
    events, rows = [], []
    with lk.spy():
        pipe = lk.Pipeline(cfg, s, "ETHUSDT")
        for b in bar_stream(sl, higher, "5m"):
            so = pipe.step(b)
            events += [e for _, e in so.touches]
            rows += [phase3.level_row("ETHUSDT", x) for x in so.births]
            rows += [phase3.level_row("ETHUSDT", x) for _, x in so.placebos]
    assert repr(events) == repr(ref["events"]) and len(events) > 100
    assert rows == ref["level_rows"]


@needs_data
def test_label_step_does_not_mutate_events():
    """Labels look forward by design; running them must leave every event object unchanged (nothing flows back)."""
    from vecopen.labels import LabelStream
    from vecopen.levels import _bars_from_frame

    s = load_sessions().for_asset("BTCUSDT")
    sl = _slice("BTCUSDT", 900, 20)
    higher = {tf: resample(sl, tf, s) for tf in CFG.data["levels"]["timeframes_primary"]}
    ev = []
    with lk.spy():
        pipe = lk.Pipeline(CFG, s, "BTCUSDT")
        for b in bar_stream(sl, higher, "5m"):
            ev += [e for _, e in pipe.step(b).touches]
    ev.sort(key=lambda e: e.t0)
    before = repr(ev)
    ls = LabelStream(CFG, "5m")
    k = n_lab = 0
    for b in _bars_from_frame(sl, "5m"):
        n_lab += len(ls.update(b))
        while k < len(ev) and ev[k].t0 == b.ts_close:
            ls.add(ev[k])
            k += 1
    n_lab += len(ls.finish())
    assert n_lab == len(ev) > 0
    assert repr(ev) == before


# ---------------------------------------------------------------- rule 4: sessions from the config

def test_session_conventions_come_from_the_config():
    sess = load_sessions()
    assert sess.sha256 == "9add7ce4e40a89ffce64e1f8a8f4cdcc47e82efdcad178cc798de1e80e2a2607"   # phase 1 registry
    assert CFG.data["assets"]["crypto"]["session"] == "utc_24h"
    for a in ASSETS:
        s = sess.for_asset(a)
        assert (s.name, s.tz, s.day_start) == ("utc_24h", "UTC", "00:00")


@needs_data
def test_session_anchor_moves_the_opens_non_vacuity():
    sess = load_sessions()
    sl = _slice("BTCUSDT", 300, 20)
    utc = resample(sl, "1d", sess.for_asset("BTCUSDT"))
    fx = resample(sl, "1d", sess.conventions["fx_ny17"])
    assert (utc["ts_open"].dt.hour() == 0).all() and not (fx["ts_open"].dt.hour() == 0).any()
    assert not utc["o"].equals(fx["o"])


# ---------------------------------------------------------------- canonical digest

@needs_data
def test_canon_is_exact_survives_pickle_and_rejects_open_state():
    s = load_sessions().for_asset("BTCUSDT")
    sl = _slice("BTCUSDT", 700, 12)
    higher = {tf: resample(sl, tf, s) for tf in CFG.data["levels"]["timeframes_primary"]}
    bars = list(bar_stream(sl, higher, "5m"))
    with lk.spy():
        p1, p2 = lk.Pipeline(CFG, s, "BTCUSDT"), lk.Pipeline(CFG, s, "BTCUSDT")
        for b in bars:
            p1.step(b)
        for b in bars[:-1]:
            p2.step(b)
        d1 = lk.canon(p1)
        assert lk.canon(pickle.loads(pickle.dumps(p1)))[0] == d1[0]
        assert lk.canon(p2)[0] != d1[0]
        p2.step(bars[-1])
        assert lk.canon(p2)[0] == d1[0]
        assert d1[1] == bars[-1].ts_close                      # the latest timestamp held by the state is now
    p1.treated.leak = sl                                       # a frame reachable from the state
    with pytest.raises(lk.ClosureViolation):
        lk.canon(p1)
    p1.treated.leak = iter(bars)
    with pytest.raises(lk.ClosureViolation):
        lk.canon(p1)


def test_audit_flags_late_emission_and_future_timestamps():
    from vecopen.levels import PolarityFlip

    au = lk.Audit.__new__(lk.Audit)                    # bare audit: no arrays, only the test-3 bookkeeping
    au.fail, au.ok, au.examples, au.ok_placebo_any = lk.Counter(), lk.Counter(), {}, set()
    au.hwm, au.last_key, au.ptr, au.fed, au.tf_ts = -1, None, {}, {}, {}
    au.prov = {"15m": {}}
    au.live, au.zones, au.treated_born, au.cov_rows, au.independent = {}, {}, {}, [], False
    au.n_treated_events = au.n_treated_births = 0
    au.nonvac = lk.Counter()
    t = 1_600_000_000_000
    empty = [("p1_nonorigin", (), (), ()), ("p2_displaced", (), (), ()), ("p3_candle", (), (), ())]
    so = lk.StepOut([("treated", (PolarityFlip(t - 300_000, 1, "15m", "support", 1),), (), ()), *empty], [], [], [], [])
    au.bar(Bar("5m", t, 1.0, 1.0, 1.0, 1.0, 1.0, t - 300_000), so)          # an output stamped one bar late
    assert au.fail["t3_output_stamp_is_now"] == 1 and au.fail["t3_hwm_le_stamp"] == 1    # emitted after its inputs
    lb = LevelBorn(t + 300_000, 0, "15m", -1, 1.0, 0.9, 1.1, 1.0, t + 900_000, 0, 2.0, None)   # origin after birth
    so = lk.StepOut([("treated", (lb,), (), ()), *empty], [], [], [], [])
    au.bar(Bar("5m", t + 300_000, 1.0, 1.0, 1.0, 1.0, 1.0, t), so)
    assert au.fail["t3_output_fields_le_stamp"] == 1
    au.prov = {"15m": {t + 600_000: t + 900_000}}                            # a 15m bar keyed before its data
    au.fed = {"15m": 0}
    so = lk.StepOut([("treated", (), (), ()), *empty], [], [], [], [])
    au.bar(Bar("15m", t + 600_000, 1.0, 1.0, 1.0, 1.0, 1.0, t), so)
    assert au.fail["t3_bar_provenance"] == 1 and au.fail["t3_hwm_le_stamp"] == 1      # no output here: unchanged


# ---------------------------------------------------------------- tests 1-3 on a real slice, and non-vacuity

@pytest.fixture(scope="module")
def slice_path(tmp_path_factory):
    if not HAVE_DATA:
        pytest.skip("research data not present")
    p = tmp_path_factory.mktemp("leak") / "slice.parquet"
    df = _slice("BTCUSDT", 420, 30)
    df.write_parquet(p)
    return p, df


def _run(slice_path, variant, n):
    p, df = slice_path
    return lk.run_asset("BTCUSDT", n=n, variant=variant, df=df, df_path=str(p), log=lambda m: None)


def test_tests_1_to_3_pass_on_a_real_slice(slice_path):
    r = _run(slice_path, "clean", 10)
    assert r["test1"]["pass"] and r["test1"]["literal_pass"], r["test1"]["failures"]
    assert r["test2"]["pass"], r["test2"]["failures"]
    assert r["test3"]["pass"] and r["rules_pass"], {c: v for c, v in r["audit"]["checks"].items() if v["fail"]}
    assert r["state_timestamps_le_t"] and r["late_origin_flags"] == 0
    assert r["audit"]["treated_events"] > 50


def test_non_vacuity_open_time_join_fails_all_three_tests(slice_path):
    r = _run(slice_path, "open_keyed", 6)
    assert r["test1"]["failed"] == r["test1"]["n"] and r["test2"]["failed"] == r["test2"]["n"]
    assert not r["test3"]["pass"]
    assert sum(v["max_src_after_close"] for v in r["provenance"].values()) > 0


def test_non_vacuity_one_bar_look_ahead_is_caught(slice_path):
    r = _run(slice_path, "peek", 16)
    assert r["test1"]["failed"] > 0 and r["test2"]["failed"] > 0


# ---------------------------------------------------------------- every asset (the reproduce command runs these)

@pytest.mark.skipif(os.environ.get("VECOPEN_LEAK_FULL") != "1",
                    reason="full-history run per asset; use vecopen/tests/leak/run_g1.py or VECOPEN_LEAK_FULL=1")
@pytest.mark.parametrize("asset", ASSETS)
def test_g1_full_history(asset):
    if not research_path(asset).exists():
        pytest.skip(f"no research data for {asset}")
    r = lk.run_asset(asset, 200, log=lambda m: None)
    r["stored"] = lk.stored_covariates_check(asset, r.pop("_cov_rows"))
    import run_g1

    v = run_g1.verdict(r)
    assert all(v.values()), (v, r["test1"]["failures"], r["test2"]["failures"])


def test_no_cfd_research_data_yet():
    """CFDs are not audited because none exists; this test fails (on purpose) once CFD data lands, so G1 is re-run."""
    cfd = CFG.data["assets"]["cfd"]["candidates"]
    assert not [a for a in cfd if research_path(a).exists()], "CFD research data exists: re-run G1 on the CFDs"


def test_holdout_is_never_a_harness_input():
    """The harness reads research bars through ``load_bars`` / explicit ``data/research`` paths only."""
    for f in ("leakkit.py", "run_g1.py"):
        for line in Path(lk.__file__).with_name(f).read_text(encoding="utf-8").splitlines():
            assert "holdout" not in line.lower() or "never" in line.lower(), (f, line)
