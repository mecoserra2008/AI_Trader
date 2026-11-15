"""Market / limit-order execution engine on 3 s paths + a faithful Python port of the `atx mm` quote rule (H-SIM4).

Port (crates/atx/src/mm.rs, read only): `target_quotes`, `min_lot`, `snap_beyond`, `sigma_1h`, `trade_imbalance`
semantics and the requote hysteresis of the run loop (replace a resting quote if it is wrong in kind, drifted
> 1 sigma, or drifted > requote_sigma after min_rest_s). Rounding to tick / lot uses exact decimals like
at_core::num::round_to_step; Rust's f64::round (half away from zero) is reproduced.

Engine (`Account`, `run_mm`):
- limit (post-only) orders rest from the candle after they are placed and fill with the per-candle probability of a
  `FillModel` (H-SIM1), full quantity, at the limit price, maker fee 2 bps of notional;
- market orders pay half the spread + square-root impact (impact_bps = impact_coef_bps * sqrt(notional / depth_25))
  + taker fee 5.5 bps; used for the end-of-path flatten and the per-symbol loss stop;
- the account is a fill ledger: cash = -sum(side * qty * price) - fees; equity = cash + inventory * mid.
Replay simplifications (stated in the report): one symbol at a time (portfolio net / gross caps see no other
symbols), no structural levels (levels_file zones and 15 m pivots are off), book top derived from the 3 s mid and
TWA spread, flow = 0.5 * 5-min taker notional imbalance + 0.5 * obi_25 of the latest candle, decisions every
`refresh` candles. Gaps (candles not ok) cancel resting quotes; inventory is carried across the gap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from ai_trader.sim.fill_model import FillModel
from ai_trader.sim.flow_data import Grid

MAKER_FEE_BPS = 2.0
TAKER_FEE_BPS = 5.5


# ---------------------------------------------------------------------------------------------- config

@dataclass
class MmCfg:
    """Quote-rule parameters (names as in MmCfg, crates/atx/src/mm.rs). Defaults = config/mm_multi.yaml."""
    refresh_s: int = 10
    sigma_lookback_min: int = 240
    sigma_fast_min: int = 30
    sigma_floor_bps: float = 20.0
    ask_offset_sigma: float = 0.45
    bid_offset_sigma: float = 0.90
    requote_sigma: float = 0.30
    min_rest_s: int = 45
    inventory_shift_sigma: float = 0.60
    take_profit_sigma: float = 0.60
    max_short_usd: float = 25.0
    max_long_usd: float = 15.0
    max_gross_usd: float = 400.0
    reduce_lots_max: int = 3
    flow_window_s: int = 300
    flow_shift_sigma: float = 0.30
    tox_horizon_s: int = 60
    tox_alpha: float = 0.30
    tox_widen: float = 1.0
    tox_pause_sigma: float = 0.75
    shock_pause_sigma: float = 3.0
    shock_cooldown_s: int = 120
    level_snap_sigma: float = 0.5
    level_buffer_sigma: float = 0.05
    symbol_loss_limit_usd: float = 1.5
    exit_window_min: int = 30
    max_net_usd: float = 60.0
    portfolio_shift_sigma: float = 0.6

    # fields with #[serde(default)] in MmCfg: absent from a yaml file they are 0, not the mm_multi values above
    SERDE_DEFAULT_ZERO = ("sigma_fast_min", "reduce_lots_max", "shock_pause_sigma", "shock_cooldown_s", "symbol_loss_limit_usd",
                          "max_net_usd", "portfolio_shift_sigma")

    @classmethod
    def from_yaml(cls, path: Path) -> MmCfg:
        import yaml
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        names = set(cls.__dataclass_fields__)
        base = {k: 0 for k in cls.SERDE_DEFAULT_ZERO}
        base.update({k: v for k, v in d.items() if k in names})
        return cls(**base)


@dataclass
class Rule:
    """InstrumentRules subset used by the quote rule."""
    tick_size: float
    qty_step: float
    min_qty: float
    min_notional: float = 5.0


@dataclass
class Inputs:
    bid1: float
    ask1: float
    sigma: float
    inv_qty: float = 0.0
    avg_entry: float = 0.0
    gross_others_usd: float = 0.0
    flow: float = 0.0
    tox: float = 0.0
    exit_only: bool = False
    paused: bool = False
    bias_sigma: float = 0.0
    cap_short_usd: float = 0.0
    cap_long_usd: float = 0.0
    net_others_usd: float = 0.0
    pend_sell_others_usd: float = 0.0
    pend_buy_others_usd: float = 0.0


@dataclass(frozen=True)
class Quote:
    price: float
    qty: float
    reduce_only: bool


@dataclass
class Levels:
    res: list[float] = field(default_factory=list)
    sup: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------------------------- port

def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x))).normalize()


def round_to_step(value: float, step: float, mode: str) -> float:
    """at_core::num::round_to_step: exact decimal rounding to a multiple of step (mode: down | up | nearest)."""
    if step <= 0.0:
        return value
    v, s = _dec(value), _dec(step)
    rnd = {"down": ROUND_FLOOR, "up": ROUND_CEILING, "nearest": ROUND_HALF_UP}[mode]
    units = (v / s).to_integral_value(rounding=rnd)
    return float(units * s)


def _rust_round(x: float) -> float:
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def min_lot(rule: Rule, px: float) -> float:
    by_notional = rule.min_notional * 1.05 / px if rule.min_notional > 0.0 else 0.0
    return round_to_step(max(rule.min_qty, by_notional), rule.qty_step, "up")


def snap_beyond(ask: float, bid: float, mid: float, sigma: float, lv: Levels, snap: float, buffer: float) -> tuple[float, float]:
    sn, bf = snap * sigma * mid, buffer * sigma * mid
    a = ask
    for r in lv.res:
        if ask <= r <= ask + sn:
            a = max(a, r + bf)
    b = bid
    for x in lv.sup:
        if bid - sn <= x <= bid:
            b = min(b, x - bf)
    return a, b


def sigma_1h(closes, floor_bps: float) -> float:
    """Robust 1h sigma from 1m closes: mean |log r| * sqrt(pi/2) * sqrt(60), floored (mm.rs sigma_1h)."""
    r = [abs(math.log(b / a)) for a, b in zip(closes[:-1], closes[1:], strict=True) if a > 0.0 and b > 0.0]
    floor = floor_bps / 1e4
    if len(r) < 30:
        return floor
    s = sum(r) / len(r) * math.sqrt(math.pi / 2.0) * math.sqrt(60.0)
    return max(s, floor)


def trade_imbalance(trades, since_ms: int) -> float:
    s = a = 0.0
    for t, sign, notional in trades:
        if t >= since_ms:
            s += sign * notional
            a += notional
    return s / a if a > 0.0 else 0.0


def target_quotes(cfg: MmCfg, rule: Rule, x: Inputs, lv: Levels | None = None) -> tuple[Quote | None, Quote | None]:
    """(bid, ask) exactly as crates/atx/src/mm.rs::target_quotes."""
    lv = lv or Levels()
    if not (x.bid1 > 0.0 and x.ask1 > x.bid1 and x.sigma > 0.0):
        return None, None
    tick, sig = rule.tick_size, x.sigma
    mid = 0.5 * (x.bid1 + x.ask1)
    inv_usd = x.inv_qty * mid
    cap_s = x.cap_short_usd if x.cap_short_usd > 0.0 else cfg.max_short_usd
    cap_l = x.cap_long_usd if x.cap_long_usd > 0.0 else cfg.max_long_usd
    inv_frac = min(1.0, max(-1.0, inv_usd / cap_s if inv_usd < 0.0 else inv_usd / cap_l))
    widen = 1.0 + cfg.tox_widen * max(-x.tox, 0.0)
    pause_adding = x.tox < -cfg.tox_pause_sigma or x.exit_only or x.paused
    net_frac = min(1.0, max(-1.0, (x.net_others_usd + inv_usd) / cfg.max_net_usd)) if cfg.max_net_usd > 0.0 else 0.0
    flow = min(1.0, max(-1.0, x.flow))
    r = mid * (1.0 + (-cfg.inventory_shift_sigma * inv_frac - cfg.portfolio_shift_sigma * net_frac + cfg.flow_shift_sigma * flow + x.bias_sigma) * sig)
    ask, bid = snap_beyond(r * (1.0 + cfg.ask_offset_sigma * sig * widen), r * (1.0 - cfg.bid_offset_sigma * sig * widen), mid, sig, lv,
                           cfg.level_snap_sigma, cfg.level_buffer_sigma)
    tp = cfg.take_profit_sigma * sig
    if x.inv_qty < 0.0 and (x.exit_only or (x.avg_entry > 0.0 and mid <= x.avg_entry * (1.0 - tp))):
        bid = max(bid, x.bid1)
    if x.inv_qty > 0.0 and (x.exit_only or (x.avg_entry > 0.0 and mid >= x.avg_entry * (1.0 + tp))):
        ask = min(ask, x.ask1)
    ask_px = max(round_to_step(ask, tick, "up"), x.bid1 + tick)
    bid_px = min(round_to_step(bid, tick, "down"), x.ask1 - tick)
    lot = min_lot(rule, mid)
    flat = abs(x.inv_qty) < rule.qty_step * 0.5

    def adds_ok(after: float) -> bool:
        within = -after <= cap_s if after < 0.0 else after <= cap_l
        pending = -x.pend_sell_others_usd if after < inv_usd else x.pend_buy_others_usd
        net_after = x.net_others_usd + pending + after
        net_before = x.net_others_usd + inv_usd
        net_ok = cfg.max_net_usd <= 0.0 or abs(net_after) <= cfg.max_net_usd or abs(net_after) < abs(net_before)
        return (within or flat) and net_ok and x.gross_others_usd + abs(after) <= cfg.max_gross_usd

    k = float(max(cfg.reduce_lots_max, 1))

    def reduce_qty(q: float) -> float:
        lots = 1.0 + _rust_round((k - 1.0) * abs(inv_frac))
        want = round_to_step(min(lots * lot, abs(q)), rule.qty_step, "down")
        return round_to_step(abs(q), rule.qty_step, "down") if want <= 0.0 else want

    if x.inv_qty > 0.0 and not flat:
        a = Quote(ask_px, reduce_qty(x.inv_qty), True)
    elif not pause_adding and adds_ok(inv_usd - lot * mid):
        a = Quote(ask_px, lot, False)
    else:
        a = None
    if x.inv_qty < 0.0 and not flat:
        b = Quote(bid_px, reduce_qty(x.inv_qty), True)
    elif not pause_adding and adds_ok(inv_usd + lot * mid):
        b = Quote(bid_px, lot, False)
    else:
        b = None
    return (b if b is not None and b.qty > 0.0 else None), (a if a is not None and a.qty > 0.0 else None)


# ------------------------------------------------------------------------------------------- engine

@dataclass
class Account:
    """Fill ledger of one symbol. side +1 buy / -1 sell."""
    cash: float = 0.0
    inv: float = 0.0
    avg_entry: float = 0.0
    fees: float = 0.0
    turnover: float = 0.0
    n_maker: int = 0
    n_taker: int = 0
    taker_cost: float = 0.0   # half-spread + impact paid on market orders (quote ccy)

    def fill(self, side: int, qty: float, px: float, fee_bps: float) -> None:
        if qty <= 0:
            return
        notional = qty * px
        fee = notional * fee_bps / 1e4
        self.cash -= side * notional + fee
        self.fees += fee
        self.turnover += notional
        old = self.inv
        new = old + side * qty
        if old == 0 or (old > 0) == (side > 0):
            self.avg_entry = (abs(old) * self.avg_entry + qty * px) / (abs(old) + qty)
        elif abs(new) < 1e-12:
            self.avg_entry = 0.0
        elif (new > 0) != (old > 0):
            self.avg_entry = px  # flipped through flat: the remainder is a new position at this price
        self.inv = 0.0 if abs(new) < 1e-12 else new

    def limit_fill(self, side: int, qty: float, px: float, fee_bps: float = MAKER_FEE_BPS) -> None:
        self.fill(side, qty, px, fee_bps)
        self.n_maker += 1

    def market(self, side: int, qty: float, mid: float, spread_bps: float, depth: float, impact_coef_bps: float = 10.0,
               fee_bps: float = TAKER_FEE_BPS) -> float:
        """Market order: price = mid * (1 + side * (spread/2 + impact) / 1e4). Returns the execution price."""
        if qty <= 0:
            return mid
        notional = qty * mid
        imp = impact_coef_bps * math.sqrt(notional / depth) if depth and depth > 0 and math.isfinite(depth) else 0.0
        px = mid * (1.0 + side * (0.5 * spread_bps + imp) / 1e4)
        self.taker_cost += abs(px - mid) * qty
        self.fill(side, qty, px, fee_bps)
        self.n_taker += 1
        return px

    def equity(self, mid: float) -> float:
        return self.cash + self.inv * mid


@dataclass
class Resting:
    side: int
    price: float
    qty: float
    reduce_only: bool
    placed_ms: int


def book_top(mid: float, spread_bps: float, tick: float) -> tuple[float, float]:
    """Best bid / ask on the tick grid around a 3 s mid and its TWA spread (at least one tick wide)."""
    half = mid * spread_bps / 2e4
    bid1 = math.floor((mid - half) / tick + 1e-9) * tick
    ask1 = math.ceil((mid + half) / tick - 1e-9) * tick
    if ask1 <= bid1:
        ask1 = bid1 + tick
    return bid1, ask1


def run_mm(g: Grid, rule: Rule, cfg: MmCfg, fm: FillModel, rng: np.random.Generator, *, start: int = 0, end: int | None = None,
           refresh: int = 3, warmup: int = 6000, impact_coef_bps: float = 10.0) -> dict:
    """Replay the quoter on grid g, trading candles [start, end); earlier candles (>= start - warmup) feed sigma only."""
    n = len(g) if end is None else min(end, len(g))
    t_ms, ok, mid_a, spr_a, hi_a, lo_a = g.t_ms, g.ok, g.mid, g.spread_bps, g.high, g.low
    obi = np.nan_to_num(g.obi_25, nan=0.0)
    cb = np.r_[0.0, np.cumsum(np.nan_to_num(g.buy))]
    cs = np.r_[0.0, np.cumsum(np.nan_to_num(g.sell))]
    fw = max(1, cfg.flow_window_s // 3)
    tick = rule.tick_size
    acc = Account()
    closes: list[float] = []
    sigma = cfg.sigma_floor_bps / 1e4
    sigma_at = -10**9
    tox = 0.0
    paused_until = -1
    last_mid, last_ms = 0.0, 0
    resting: dict[int, Resting | None] = {1: None, -1: None}
    pend_mk: list[tuple[int, float, int]] = []  # (side, price, fill candle)
    stopped = False
    exit_from = n - cfg.exit_window_min * 20
    n_dec = 0
    max_inv_usd = 0.0
    last_ok = None
    for i in range(max(0, start - warmup), n):
        if not ok[i]:
            resting[1] = resting[-1] = None
            continue
        m = float(mid_a[i])
        last_ok = i
        if (int(t_ms[i]) + 3000) % 60000 == 0:
            closes.append(m)
            if len(closes) > cfg.sigma_lookback_min + 1:
                del closes[0]
        if i < start:
            continue
        # fills on this candle (quotes placed at an earlier decision)
        for side in (1, -1) if rng.random() < 0.5 else (-1, 1):
            q = resting[side]
            if q is None:
                continue
            if rng.random() < fm.p_candle(side, q.price, hi_a[i], lo_a[i], tick):
                qty = q.qty
                if q.reduce_only:
                    qty = min(qty, abs(acc.inv)) if acc.inv * side < 0 else 0.0
                if qty > 0:
                    acc.limit_fill(side, qty, q.price)
                    pend_mk.append((side, q.price, i))
                resting[side] = None
        # toxicity markouts due
        if pend_mk and i - pend_mk[0][2] >= cfg.tox_horizon_s // 3:
            keep = []
            for side, px, j in pend_mk:
                if i - j >= cfg.tox_horizon_s // 3:
                    tox = (1.0 - cfg.tox_alpha) * tox + cfg.tox_alpha * side * (m - px) / px / max(sigma, 1e-6)
                else:
                    keep.append((side, px, j))
            pend_mk = keep
        max_inv_usd = max(max_inv_usd, abs(acc.inv) * m)
        if stopped or (i - start) % refresh:
            continue
        n_dec += 1
        now = int(t_ms[i])
        if now - sigma_at >= 15 * 60_000:
            slow = sigma_1h(closes, cfg.sigma_floor_bps)
            # mm.rs slices the LAST sigma_fast_min closes (29 returns for 30), below sigma_1h's 30-return minimum, so the
            # live fast sigma is always the floor; the port keeps that behaviour (finding reported in the study).
            fast = sigma_1h(closes[-cfg.sigma_fast_min:], cfg.sigma_floor_bps) if cfg.sigma_fast_min > 0 and len(closes) > cfg.sigma_fast_min else 0.0
            sigma, sigma_at = max(slow, fast), now
        if cfg.symbol_loss_limit_usd > 0 and acc.equity(m) <= -cfg.symbol_loss_limit_usd:
            if acc.inv:
                acc.market(-1 if acc.inv > 0 else 1, abs(acc.inv), m, float(spr_a[i]), float(g.depth[i]), impact_coef_bps)
            resting[1] = resting[-1] = None
            stopped = True
            continue
        if last_mid > 0 and cfg.shock_pause_sigma > 0:
            dt_s = max((now - last_ms) / 1000.0, float(cfg.refresh_s))
            if abs(math.log(m / last_mid)) > cfg.shock_pause_sigma * sigma * math.sqrt(dt_s / 3600.0):
                paused_until = now + cfg.shock_cooldown_s * 1000
        last_mid, last_ms = m, now
        tb, ts_ = cb[i + 1] - cb[max(0, i + 1 - fw)], cs[i + 1] - cs[max(0, i + 1 - fw)]
        ti = (tb - ts_) / (tb + ts_) if tb + ts_ > 0 else 0.0
        flow = 0.5 * ti + 0.5 * float(obi[i])
        bid1, ask1 = book_top(m, float(spr_a[i]), tick)
        x = Inputs(bid1=bid1, ask1=ask1, sigma=sigma, inv_qty=acc.inv, avg_entry=acc.avg_entry, flow=flow, tox=tox,
                   exit_only=i >= exit_from, paused=now < paused_until)
        want_b, want_a = target_quotes(cfg, rule, x)
        mid_q = 0.5 * (bid1 + ask1)
        for side, want in ((1, want_b), (-1, want_a)):
            cur = resting[side]
            if want is not None and cur is not None:
                drift = abs(cur.price - want.price) / mid_q / sigma
                age_s = (now - cur.placed_ms) / 1000.0
                kind = cur.reduce_only != want.reduce_only or abs(cur.qty - want.qty) > rule.qty_step * 0.5
                replace_ = kind or drift > 1.0 or (drift > cfg.requote_sigma and age_s >= cfg.min_rest_s)
            else:
                replace_ = want is None and cur is not None
            if replace_:
                resting[side] = None
            if want is not None and (cur is None or replace_):
                resting[side] = Resting(side, want.price, want.qty, want.reduce_only, now)
    # end of path: flatten at market
    if last_ok is not None and acc.inv:
        acc.market(-1 if acc.inv > 0 else 1, abs(acc.inv), float(mid_a[last_ok]), float(spr_a[last_ok]), float(g.depth[last_ok]), impact_coef_bps)
    pnl = acc.cash  # flat
    return {"pnl": pnl, "turnover": acc.turnover, "fees": acc.fees, "maker_fills": acc.n_maker, "taker_fills": acc.n_taker,
            "taker_cost": acc.taker_cost, "pnl_bps": pnl / acc.turnover * 1e4 if acc.turnover > 0 else float("nan"),
            "max_inv_usd": max_inv_usd, "decisions": n_dec, "stopped": stopped}


def grid_params(base: MmCfg | None = None) -> list[MmCfg]:
    """The H-SIM4 grid: ask offset x bid offset (3 x 3) x take-profit (2) x requote threshold (2) = 36 sets."""
    base = base or MmCfg()
    out = []
    for a in (0.2, 0.45, 0.9):
        for b in (0.2, 0.45, 0.9):
            for tp in (0.3, 0.9):
                for rq in (0.15, 0.5):
                    out.append(replace(base, ask_offset_sigma=a, bid_offset_sigma=b, take_profit_sigma=tp, requote_sigma=rq))
    return out


def param_key(c: MmCfg) -> str:
    return f"a{c.ask_offset_sigma:g}_b{c.bid_offset_sigma:g}_tp{c.take_profit_sigma:g}_rq{c.requote_sigma:g}"


def cfg_dict(c: MmCfg) -> dict:
    return asdict(c)
