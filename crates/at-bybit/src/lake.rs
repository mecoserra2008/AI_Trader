//! Venue records -> lake batches. **All known_ts rules for Bybit data live here** (docs/SCHEMA.md).
//!
//! | table          | event_ts            | known_ts                                  |
//! |----------------|---------------------|-------------------------------------------|
//! | bars           | bar open            | bar close + latency                       |
//! | flow_bars      | bar open            | bar close + latency                       |
//! | funding        | settlement time     | settlement + latency                      |
//! | open_interest  | snapshot timestamp  | timestamp + one period + latency (conservative: the venue may update the current bucket) |
//! | ls_ratio       | snapshot timestamp  | timestamp + one period + latency (same reason) |
//! | trades_raw     | trade time          | trade time + latency                      |
//! | instruments    | snapshot (fetch) time | fetch time                              |

use at_core::time::{Freq, Ms};
use at_store::Batch;

use crate::VENUE;
use crate::archive::{FlowBar, Trade};
use crate::market::{Funding, Instrument, Kline, LongShortRatio, OpenInterest, PriceType};

pub fn instruments(category: &str, xs: &[Instrument], fetched_at: Ms) -> Batch {
    let n = xs.len();
    let s = |g: fn(&Instrument) -> &String| xs.iter().map(|x| g(x).clone()).collect::<Vec<_>>();
    let num = |g: fn(&Instrument) -> f64| xs.iter().map(g).collect::<Vec<_>>();
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("category", category, n)
        .utf8("symbol", s(|x| &x.symbol))
        .utf8("base", s(|x| &x.base))
        .utf8("quote", s(|x| &x.quote))
        .utf8("settle", s(|x| &x.settle))
        .utf8("contract_type", s(|x| &x.contract_type))
        .utf8("symbol_type", s(|x| &x.symbol_type))
        .utf8("status", s(|x| &x.status))
        .ts_opt("launch_ts", xs.iter().map(|x| x.launch_ts).collect())
        .f64("tick_size", num(|x| x.tick_size))
        .f64("qty_step", num(|x| x.qty_step))
        .f64("min_qty", num(|x| x.min_qty))
        .f64("max_qty", num(|x| x.max_qty))
        .f64("max_mkt_qty", num(|x| x.max_mkt_qty))
        .f64("min_notional", num(|x| x.min_notional))
        .f64("max_leverage", num(|x| x.max_leverage))
        .i32("funding_interval_min", xs.iter().map(|x| x.funding_interval_min).collect())
        .ts("event_ts", vec![fetched_at; n])
        .ts("known_ts", vec![fetched_at; n])
}

pub fn klines(symbol: &str, freq: Freq, price_type: PriceType, xs: &[Kline], latency_ms: Ms) -> Batch {
    let n = xs.len();
    let dur = freq.duration_ms();
    let has_volume = price_type == PriceType::Last;
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("price_type", price_type.as_str(), n)
        .utf8_const("freq", freq.as_str(), n)
        .utf8_const("symbol", symbol, n)
        .ts("event_ts", xs.iter().map(|k| k.open_ts).collect())
        .ts("close_ts", xs.iter().map(|k| k.open_ts + dur).collect())
        .f64("open", xs.iter().map(|k| k.open).collect())
        .f64("high", xs.iter().map(|k| k.high).collect())
        .f64("low", xs.iter().map(|k| k.low).collect())
        .f64("close", xs.iter().map(|k| k.close).collect())
        .f64_opt("volume", xs.iter().map(|k| k.volume.filter(|_| has_volume)).collect())
        .f64_opt("turnover", xs.iter().map(|k| k.turnover.filter(|_| has_volume)).collect())
        .ts("known_ts", xs.iter().map(|k| k.open_ts + dur + latency_ms).collect())
}

pub fn flow_bars(symbol: &str, freq: Freq, xs: &[FlowBar], latency_ms: Ms) -> Batch {
    let n = xs.len();
    let dur = freq.duration_ms();
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("freq", freq.as_str(), n)
        .utf8_const("symbol", symbol, n)
        .ts("event_ts", xs.iter().map(|b| b.open_ts).collect())
        .ts("close_ts", xs.iter().map(|b| b.open_ts + dur).collect())
        .f64("buy_volume", xs.iter().map(|b| b.buy_volume).collect())
        .f64("sell_volume", xs.iter().map(|b| b.sell_volume).collect())
        .f64("buy_turnover", xs.iter().map(|b| b.buy_turnover).collect())
        .f64("sell_turnover", xs.iter().map(|b| b.sell_turnover).collect())
        .i64("n_trades", xs.iter().map(|b| b.n_trades).collect())
        .i64("n_buy", xs.iter().map(|b| b.n_buy).collect())
        .i64("n_sell", xs.iter().map(|b| b.n_sell).collect())
        .f64_opt("vwap", xs.iter().map(|b| b.vwap).collect())
        .f64("max_trade_size", xs.iter().map(|b| b.max_trade_size).collect())
        .ts("known_ts", xs.iter().map(|b| b.open_ts + dur + latency_ms).collect())
}

pub fn trades(symbol: &str, xs: &[Trade], latency_ms: Ms) -> Batch {
    let n = xs.len();
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("symbol", symbol, n)
        .ts("event_ts", xs.iter().map(|t| t.ts).collect())
        .utf8("trade_id", xs.iter().map(|t| t.trade_id.clone()).collect())
        .utf8("side", xs.iter().map(|t| if t.is_buy { "Buy" } else { "Sell" }.to_string()).collect())
        .f64("price", xs.iter().map(|t| t.price).collect())
        .f64("size", xs.iter().map(|t| t.size).collect())
        .ts("known_ts", xs.iter().map(|t| t.ts + latency_ms).collect())
}

pub fn funding(symbol: &str, xs: &[Funding], latency_ms: Ms) -> Batch {
    let n = xs.len();
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("symbol", symbol, n)
        .ts("event_ts", xs.iter().map(|x| x.ts).collect())
        .f64("rate", xs.iter().map(|x| x.rate).collect())
        .ts("known_ts", xs.iter().map(|x| x.ts + latency_ms).collect())
}

pub fn open_interest(symbol: &str, interval: &str, period_ms: Ms, xs: &[OpenInterest], latency_ms: Ms) -> Batch {
    let n = xs.len();
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("interval", interval, n)
        .utf8_const("symbol", symbol, n)
        .ts("event_ts", xs.iter().map(|x| x.ts).collect())
        .f64("open_interest", xs.iter().map(|x| x.open_interest).collect())
        .ts("known_ts", xs.iter().map(|x| x.ts + period_ms + latency_ms).collect())
}

pub fn ls_ratio(symbol: &str, period: &str, period_ms: Ms, xs: &[LongShortRatio], latency_ms: Ms) -> Batch {
    let n = xs.len();
    Batch::new()
        .utf8_const("venue", VENUE, n)
        .utf8_const("period", period, n)
        .utf8_const("symbol", symbol, n)
        .ts("event_ts", xs.iter().map(|x| x.ts).collect())
        .f64("buy_ratio", xs.iter().map(|x| x.buy_ratio).collect())
        .f64("sell_ratio", xs.iter().map(|x| x.sell_ratio).collect())
        .ts("known_ts", xs.iter().map(|x| x.ts + period_ms + latency_ms).collect())
}
