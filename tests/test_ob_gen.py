"""Tests for the queue-reactive order-book generator (ai_trader.sim.ob_gen, agent F5).

RefBook below is a minimal price-time FIFO reference book written for these tests only (docs/MATCHING_ENGINE.md:
until at-match exists the generator is checked against it)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ai_trader.sim import ob_gen as og

DATE = "2026-01-01"
D0 = og.day0_ms(DATE)
P0 = 50_000.0


# ---------------------------------------------------------------------------------------------- synthetic ob200 stream


def _p_cancel(q: float) -> float:
    """Ground truth of the synthetic stream: touch cancellations get likelier with the queue size (queue-reactive)."""
    return min(0.04 + 0.06 * q, 0.8)


def synthetic_stream(tmp: Path, n: int = 6000, seed: int = 1) -> tuple[Path, dict]:
    rng = np.random.default_rng(seed)
    t = og.TICK
    best_bid = int(P0 / t)
    bids = {best_bid - k: round(float(rng.uniform(0.1, 1.0)), 3) for k in range(0, 300, 1) if k == 0 or rng.random() < 0.6}
    asks = {best_bid + 1 + k: round(float(rng.uniform(0.1, 1.0)), 3) for k in range(0, 300, 1) if k == 0 or rng.random() < 0.6}
    bids[best_bid] = 1.0
    asks[best_bid + 1] = 1.0

    def lv(d):
        return [[f"{p * t:.1f}", f"{s:.3f}"] for p, s in sorted(d.items())]

    lines = [{"topic": "orderbook.200.BTCUSDT", "type": "snapshot", "ts": D0 + 50, "cts": D0 + 49,
              "data": {"s": "BTCUSDT", "b": lv(bids), "a": lv(asks), "u": 1, "seq": 1}}]
    tr_ms, tr_side, tr_px, tr_sz = [], [], [], []
    for k in range(1, n):
        ts = D0 + 50 + 100 * k
        old_b, old_a = dict(bids), dict(asks)
        for book, best in ((bids, max(bids)), (asks, min(asks))):
            q = book[best]
            if rng.random() < _p_cancel(q) and q > 0.002:
                book[best] = round(max(q - q * float(rng.uniform(0.05, 0.5)), 0.001), 3)
            elif rng.random() < 0.3:
                book[best] = round(q + float(rng.lognormal(-1.5, 0.7)), 3)
        if k % 40 == 0:   # occasional one-tick move up or down
            if rng.random() < 0.5:
                a = min(asks)
                del asks[a]
                bids[a - 1 + 1 - 1] = bids.get(a - 1, 0) or 1.0
                nb = max(bids) + 1 if max(bids) + 1 < min(asks) else None
                if nb is not None:
                    bids[nb] = 1.0
            else:
                b = max(bids)
                del bids[b]
                na = min(asks) - 1 if min(asks) - 1 > max(bids) else None
                if na is not None:
                    asks[na] = 1.0
        if k % 10 == 0:
            tr_ms.append(ts - 5)
            tr_side.append(1 if rng.random() < 0.5 else -1)
            tr_px.append((max(bids) + 0.5) * t)
            tr_sz.append(0.001)

        def delta(old, new):
            out = [[f"{p * t:.1f}", "0"] for p in old if p not in new]
            out += [[f"{p * t:.1f}", f"{s:.3f}"] for p, s in new.items() if old.get(p) != s]
            return out

        lines.append({"topic": "orderbook.200.BTCUSDT", "type": "delta", "ts": ts, "cts": ts - 1,
                      "data": {"s": "BTCUSDT", "b": delta(old_b, bids), "a": delta(old_a, asks), "u": k + 1, "seq": k + 1}})
    p = tmp / f"{DATE}_BTCUSDT_ob200.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    tr = {"ms": np.asarray(tr_ms, np.int64), "side": np.asarray(tr_side, np.int8), "price": np.asarray(tr_px), "size": np.asarray(tr_sz)}
    return p, tr


@pytest.fixture(scope="module")
def cal_root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("obgen")
    ob, tr = synthetic_stream(tmp)
    cal = og.extract_calibration(ob, tr, DATE, real_out=tmp / "real" / DATE, log=lambda *_: None)
    og.write_json(tmp / "cal" / f"{DATE}.json", cal)
    return tmp, cal


@pytest.fixture(scope="module")
def tables(cal_root):
    root, _ = cal_root
    return og.load_tables([DATE], root=root)


def day_cal() -> og.DayCal:
    return og.DayCal(date=DATE, s_touch=1.0, s_deep=1.0, vedges=[1e-9, 4e-9],
                     arrival={"family": "expmix3", "params": {"w1": 0.3, "w2": 0.4, "w3": 0.3, "rate1": 200.0, "rate2": 4.0, "rate3": 0.5}},
                     size={"kind": "size", "mu": 6.0, "sigma": 1.5, "xmin": 50_000.0, "alpha": 2.5, "p_tail": 0.01},
                     sign={"kind": "translated_sign", "c": 1.0, "gamma": 0.7, "p_buy": 0.5, "short": {"1": 0.5, "2": 0.4, "5": 0.3}},
                     orders_per_s=3.0, theta_scale=0.5, phi=0.3)


def flow_path(n_bars: int = 6, seed: int = 3) -> og.FlowPath:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 3e-4, n_bars)
    close = P0 * np.exp(np.cumsum(r))
    nb = rng.poisson(90, n_bars)
    ns = rng.poisson(90, n_bars)
    return og.FlowPath(D0 + 3_600_000, 60_000, P0, close, rv=np.full(n_bars, 6e-7), buy_notional=nb * 2000.0,
                       sell_notional=ns * 2000.0, n_buy=nb, n_sell=ns)


# ---------------------------------------------------------------------------------------------- reference book


class RefBook:
    """Minimal price-time FIFO book that consumes at-match JSONL events (tests only)."""

    def __init__(self, band: float = 0.01):
        self.side = {"Buy": {}, "Sell": {}}
        self.live: dict[int, tuple[str, float]] = {}
        self.trades: list[tuple[int, int, float, float]] = []
        self.band = band
        self.last_ts = -math.inf

    def best(self, s: str):
        lv = self.side[s]
        if not lv:
            return None
        return max(lv) if s == "Buy" else min(lv)

    def apply(self, e: dict) -> None:
        assert e["ts_ms"] >= self.last_ts, "ts_ms must be non-decreasing"
        self.last_ts = e["ts_ms"]
        if e["kind"] == "cancel":
            assert e["id"] in self.live, f"cancel of a dead id {e['id']}"
            s, p = self.live.pop(e["id"])
            q = self.side[s][p]
            q[:] = [o for o in q if o[0] != e["id"]]
            if not q:
                del self.side[s][p]
            return
        assert e["kind"] == "new"
        assert e["id"] not in self.live
        qty = round(e["qty"], 3)
        assert qty >= 0.001 and abs(qty * 1000 - round(qty * 1000)) < 1e-6
        s = e["side"]
        opp = "Sell" if s == "Buy" else "Buy"
        if e["type"] == "Limit":
            p = round(e["price"], 1)
            assert abs(p * 10 - round(p * 10)) < 1e-6
            ob = self.best(opp)
            assert e["tif"] == "PostOnly"
            if ob is not None:
                assert (p < ob) if s == "Buy" else (p > ob), "PostOnly order would cross"
            self.side[s].setdefault(p, []).append([e["id"], qty])
            self.live[e["id"]] = (s, p)
            return
        assert e["type"] == "Market" and e["tif"] == "IOC"
        ob = self.best(opp)
        if ob is None:
            return
        lim = ob * (1 + self.band) if s == "Buy" else ob * (1 - self.band)
        left = qty
        while left > 1e-9 and self.side[opp]:
            p = self.best(opp)
            if (s == "Buy" and p > lim) or (s == "Sell" and p < lim):
                break
            q = self.side[opp][p]
            while q and left > 1e-9:
                take = min(q[0][1], left)
                q[0][1] = round(q[0][1] - take, 3)
                left = round(left - take, 3)
                self.trades.append((e["ts_ms"], 1 if s == "Buy" else -1, p, take))
                if q[0][1] <= 1e-9:
                    self.live.pop(q[0][0], None)
                    q.pop(0)
            if not q:
                del self.side[opp][p]


# ---------------------------------------------------------------------------------------------- tests


def test_calibration_recovers_queue_reactive_cancels(cal_root):
    _, cal = cal_root
    shp = (og.NB, og.NQ, og.NV, og.NS)
    risk = np.asarray(cal["risk"]).reshape(shp)[0].sum(axis=(1, 2))
    nc = np.asarray(cal["n_can"]).reshape(shp)[0].sum(axis=(1, 2))
    rate = np.where(risk >= 100, nc / np.maximum(risk, 1), np.nan)
    good = np.flatnonzero(np.isfinite(rate))
    assert len(good) >= 3
    # cancel intensity at the touch increases with the queue size (the synthetic truth)
    assert rate[good[-1]] > rate[good[0]] + 0.03
    assert np.corrcoef(good, rate[good])[0, 1] > 0.6
    assert cal["aes"] > 0 and cal["counts"]["intervals_unchanged"] > 1000


def test_generator_intensities_follow_the_tables(tables):
    g = og.Generator(tables, day_cal(), seed=5)
    sink = og.EventSink(None)
    bk = og.GenBook(int(P0 / og.TICK), sink)
    g.init_book(bk, D0, P0 / og.TICK)
    lots_small = max(1, int(round(0.3 * g.aes_lots)))
    lots_big = max(lots_small + 1, int(round(8 * g.aes_lots)))
    freq = {}
    for lots in (lots_small, lots_big):
        n_can = 0
        trials = 3000
        for _ in range(trials):
            b = bk.best[1]
            cur = int(bk.lots[1][b - bk.base])
            if cur != lots:     # reset the touch queue to the target size
                if cur > lots:
                    bk.cancel_lots(D0, 1, b, cur - lots)
                else:
                    bk.add(D0, 1, b, lots - cur)
            before = int(bk.lots[1][b - bk.base])
            g.level_events(bk, D0, 1, True)
            after = int(bk.lots[1][b - bk.base]) if bk.best[1] == b else 0
            n_can += after < before
        freq[lots] = n_can / trials
    qb = og.qbin(np.asarray([lots_small, lots_big]) / g.aes_lots)
    p_tab = [float(tables.p_can[0, q, 1, 0]) for q in qb]
    # empirical frequencies match the tables (adds can mask a cancel in the same interval: allow a small gap)
    assert abs(freq[lots_small] - p_tab[0]) < 0.05
    assert abs(freq[lots_big] - p_tab[1]) < 0.06
    assert freq[lots_big] > freq[lots_small]


def _generate(tables, tmp: Path, seed: int = 7):
    ev = tmp / "events.jsonl"
    out = og.Generator(tables, day_cal(), seed=seed).run(flow_path(), events_path=ev, warmup_s=30)
    return ev, out


def test_generated_events_are_valid_atmatch_input_and_match_reference(tables, tmp_path):
    ev, out = _generate(tables, tmp_path)
    ref = RefBook()
    keys = {"ts_ms", "kind", "id", "side", "type", "tif", "price", "qty", "reduce_only", "owner"}
    n = 0
    with ev.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "new":
                assert keys <= set(e), e
            else:
                assert {"ts_ms", "kind", "id"} <= set(e)
            ref.apply(e)
            n += 1
    assert n == out.n_events > 1000
    # the generator's own book matched exactly like the reference book
    gen = sorted((int(t), int(s), round(p * og.TICK, 1), round(q * og.QTY_STEP, 3)) for t, s, p, q in out.trades.tolist())
    rb = sorted((int(t), int(s), round(p, 1), round(q, 3)) for t, s, p, q in ref.trades)
    assert len(gen) > 50 and gen == rb
    # never crossed at the 1 s states
    b1 = np.asarray([r[2] for r in out.book_rows])
    a1 = np.asarray([r[3] for r in out.book_rows])
    assert np.all(a1 > b1)


def test_path_counts_and_notional_are_honoured(tables):
    p = flow_path()
    g = og.Generator(tables, day_cal(), seed=11)
    ms, sd, lots = g.taker_orders(p, [])
    bar = (ms - p.t0_ms) // p.bar_ms
    for b in range(p.n):
        m = bar == b
        assert (sd[m] > 0).sum() == pytest.approx(p.n_buy[b], abs=2)
        assert (sd[m] < 0).sum() == pytest.approx(p.n_sell[b], abs=2)
        px = p.open0 if b == 0 else p.close[b - 1]
        assert (lots[m & (sd > 0)] * og.QTY_STEP * px).sum() == pytest.approx(p.buy_notional[b], rel=0.15)


def test_aggregates_match_micro_days_schema(tables, tmp_path):
    _, out = _generate(tables, tmp_path, seed=9)
    od = tmp_path / "agg"
    og.aggregate(od, og.prints_dict(out.trades), out.book_rows, D0)
    real = og.MICRO / "2026-11-02"
    for sub in ("trades_1s", "book_1s", "orders"):
        g = pl.read_parquet(od / sub / "BTCUSDT.parquet")
        rp = real / sub / "BTCUSDT.parquet"
        if rp.is_file():
            r = pl.read_parquet(rp)
            assert g.schema == r.schema, sub
        assert g.height > 0
    b = pl.read_parquet(od / "book_1s" / "BTCUSDT.parquet")
    d = b["ts"].dt.epoch("ms").diff().drop_nulls()
    assert (d == 1000).all()          # dense 1 s grid


def test_manifest_prevents_redownload(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(url, dest):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return {"status": "ok", "bytes": 1}

    monkeypatch.setattr(og, "RAW", tmp_path / "raw")
    monkeypatch.setattr(og, "ROOT", tmp_path)
    monkeypatch.setattr(og, "storage_gb", lambda root=None: {"total_gb": 1.0})
    man = og.Manifest(tmp_path / "manifest.json")
    paths = og.fetch_day("2026-02-02", man, fetch=fake_fetch, max_gb=30)
    assert paths is not None and len(calls) == 2
    og.fetch_day("2026-02-02", man, fetch=fake_fetch, max_gb=30)       # files on disk: no new download
    assert len(calls) == 2
    man.put("calibration", "2026-02-02", {"status": "done"})
    og.cleanup_day("2026-02-02", man)
    assert og.fetch_day("2026-02-02", og.Manifest(tmp_path / "manifest.json"), fetch=fake_fetch, max_gb=30) is None
    assert len(calls) == 2
    # storage guard
    monkeypatch.setattr(og, "storage_gb", lambda root=None: {"total_gb": 29.5})
    with pytest.raises(RuntimeError):
        og.fetch_day("2026-02-03", man, fetch=fake_fetch, max_gb=30)


@pytest.mark.skipif(og.atx_binary() is None, reason="atx match-sim not built")
def test_engine_reproduces_the_generator_book(tables, tmp_path):
    ev, out = _generate(tables, tmp_path, seed=13)
    st = og.run_engine(ev, tmp_path / "engine")
    assert st["bad_input"] == 0 and og.engine_rejects(tmp_path / "engine") == 0
    pe, pi = og.engine_prints(tmp_path / "engine"), og.prints_dict(out.trades)
    assert len(pe["ms"]) == len(pi["ms"]) > 0
    assert np.array_equal(pe["ms"], pi["ms"]) and np.array_equal(pe["side"], pi["side"])
    assert np.allclose(pe["price"], pi["price"]) and np.allclose(pe["size"], pi["size"])
