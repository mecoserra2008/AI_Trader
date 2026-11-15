//! Historical ticks (`ProtoOAGetTickDataReq`): delta decoding and backwards paging on `hasMore`.
//!
//! A response lists ticks newest first. The first tick carries the absolute time (ms) and price (1/100000);
//! each following tick carries the difference to the previous one. Spotware's samples accumulate
//! (`t += dt`, `p += dp`) with negative time deltas; if a page instead decodes forwards in time the opposite
//! sign convention is applied, so either convention yields the same ticks (logged once as a warning).

use crate::pb;
use crate::session::Session;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "bid" => Ok(Self::Bid),
            "ask" => Ok(Self::Ask),
            o => anyhow::bail!("side must be bid or ask, got {o:?}"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bid => "bid",
            Self::Ask => "ask",
        }
    }
    fn quote_type(self) -> i32 {
        match self {
            Self::Bid => pb::ProtoOaQuoteType::Bid as i32,
            Self::Ask => pb::ProtoOaQuoteType::Ask as i32,
        }
    }
}

/// One tick: event time (ms since epoch) and raw price (1/100000).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tick {
    pub ts: i64,
    pub raw: i64,
}

/// Decode one page (newest first) into absolute ticks, newest first.
pub fn decode_page(data: &[pb::ProtoOaTickData]) -> Vec<Tick> {
    let Some(first) = data.first() else { return Vec::new() };
    let acc = |sign: i64| {
        let (mut t, mut p) = (first.timestamp, first.tick);
        let mut v = Vec::with_capacity(data.len());
        v.push(Tick { ts: t, raw: p });
        for d in &data[1..] {
            t += sign * d.timestamp;
            p += sign * d.tick;
            v.push(Tick { ts: t, raw: p });
        }
        v
    };
    let v = acc(1);
    // newest first: every later entry must be at or before the first one
    if v.iter().all(|x| x.ts <= first.timestamp) {
        return v;
    }
    tracing::warn!("tick page decodes forwards in time with t += dt; using t -= dt");
    acc(-1)
}

/// Page backwards from `to_ms` until `from_ms` (both ms since epoch, `[from, to)`); returns ticks in
/// chronological order, deduplicated at page boundaries.
pub async fn fetch(sess: &mut Session, account: i64, symbol_id: i64, side: Side, from_ms: i64, to_ms: i64, mut progress: impl FnMut(usize, i64)) -> anyhow::Result<Vec<Tick>> {
    let mut out: Vec<Tick> = Vec::new();
    let mut to = to_ms;
    let mut pages = 0usize;
    loop {
        let req = pb::ProtoOaGetTickDataReq {
            payload_type: None,
            ctid_trader_account_id: account,
            symbol_id,
            r#type: side.quote_type(),
            from_timestamp: Some(from_ms),
            to_timestamp: Some(to),
        };
        let res = sess.request(&req).await?;
        pages += 1;
        let page = decode_page(&res.tick_data);
        let oldest = page.iter().map(|t| t.ts).min();
        out.extend(page.into_iter().filter(|t| t.ts >= from_ms && t.ts < to_ms));
        progress(pages, oldest.unwrap_or(from_ms));
        match oldest {
            Some(o) if res.has_more && o > from_ms => {
                // next page ends at the oldest tick seen; if the page did not move the boundary, step past it
                to = if o < to { o } else { o - 1 };
            }
            _ => break,
        }
        anyhow::ensure!(pages < 100_000, "tick paging did not terminate");
    }
    out.sort_by_key(|t| t.ts);
    // pages overlap at the boundary millisecond: drop exact repeats of (ts, price) that come from two pages
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td(timestamp: i64, tick: i64) -> pb::ProtoOaTickData {
        pb::ProtoOaTickData { timestamp, tick }
    }

    #[test]
    fn decodes_negative_deltas() {
        let p = decode_page(&[td(1_000_000, 200_000), td(-10, 5), td(-5, -15)]);
        assert_eq!(p, vec![Tick { ts: 1_000_000, raw: 200_000 }, Tick { ts: 999_990, raw: 200_005 }, Tick { ts: 999_985, raw: 199_990 }]);
    }

    #[test]
    fn decodes_positive_deltas_as_differences() {
        let p = decode_page(&[td(1_000_000, 200_000), td(10, -5), td(5, 15)]);
        assert_eq!(p, vec![Tick { ts: 1_000_000, raw: 200_000 }, Tick { ts: 999_990, raw: 200_005 }, Tick { ts: 999_985, raw: 199_990 }]);
    }

    #[test]
    fn empty_page() {
        assert!(decode_page(&[]).is_empty());
    }
}
