//! The client against a fake Open API server speaking canned protobuf frames over an in-memory duplex.

use std::sync::{Arc, Mutex};

use at_ctrader::codec::{Frame, OaMsg, split_frames};
use at_ctrader::pb;
use at_ctrader::ticks::{self, Side, Tick};
use at_ctrader::trendbars::{self, Bar, OnDisk, Period};
use at_ctrader::{CtraderCreds, CtraderEnv, Session, flow};
use prost::Message;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const ACCOUNT: u64 = 4242;

#[derive(Clone)]
struct Fake {
    scope: i32,
    /// absolute ticks, chronological
    ticks: Vec<Tick>,
    page: usize,
    rate_limit_first_tick_req: bool,
    log: Arc<Mutex<Vec<u32>>>,
    /// M5 trend bars, chronological; at most `bar_chunk` per response (`hasMore` beyond), keeping the newest or the
    /// oldest; windows longer than `bar_max_range_ms` get a boundary error.
    bars: Vec<Bar>,
    bar_chunk: usize,
    keep_newest: bool,
    bar_max_range_ms: i64,
    trendbar_requests: Arc<Mutex<Vec<(i64, i64)>>>,
}

fn reply<M: OaMsg>(m: &M, id: &Option<String>) -> Vec<u8> {
    Frame::of(m, id.clone()).encode()
}

/// Newest-first page with Spotware delta encoding (negative time deltas).
fn encode_page(t: &[Tick]) -> Vec<pb::ProtoOaTickData> {
    let mut out = Vec::new();
    let mut prev: Option<Tick> = None;
    for x in t.iter().rev() {
        out.push(match prev {
            None => pb::ProtoOaTickData { timestamp: x.ts, tick: x.raw },
            Some(p) => pb::ProtoOaTickData { timestamp: x.ts - p.ts, tick: x.raw - p.raw },
        });
        prev = Some(*x);
    }
    out
}

fn handle(f: &Frame, st: &mut Fake) -> Vec<u8> {
    st.log.lock().unwrap().push(f.payload_type);
    let id = &f.client_msg_id;
    let acct = ACCOUNT as i64;
    match f.payload_type {
        p if p == pb::ProtoOaApplicationAuthReq::PAYLOAD => {
            let r: pb::ProtoOaApplicationAuthReq = f.parse().unwrap();
            if r.client_secret != "good-secret" {
                return reply(&pb::ProtoOaErrorRes { error_code: "CH_CLIENT_AUTH_FAILURE".into(), description: Some("bad secret".into()), ..Default::default() }, id);
            }
            // an unsolicited event before the response must be queued, not mistaken for it
            let mut out = reply(&pb::ProtoHeartbeatEvent::default(), &None);
            out.extend(reply(&pb::ProtoOaApplicationAuthRes::default(), id));
            out
        }
        p if p == pb::ProtoOaVersionReq::PAYLOAD => reply(&pb::ProtoOaVersionRes { version: "99".into(), ..Default::default() }, id),
        p if p == pb::ProtoOaGetAccountListByAccessTokenReq::PAYLOAD => reply(
            &pb::ProtoOaGetAccountListByAccessTokenRes {
                access_token: "x".into(),
                permission_scope: Some(st.scope),
                ctid_trader_account: vec![
                    pb::ProtoOaCtidTraderAccount {
                        ctid_trader_account_id: 1,
                        is_live: Some(false),
                        broker_title_short: Some("BlackBull".into()),
                        trader_login: Some(11),
                        ..Default::default()
                    },
                    pb::ProtoOaCtidTraderAccount {
                        ctid_trader_account_id: ACCOUNT,
                        is_live: Some(true),
                        broker_title_short: Some("BlackBull".into()),
                        trader_login: Some(77),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
            id,
        ),
        p if p == pb::ProtoOaAccountAuthReq::PAYLOAD => reply(&pb::ProtoOaAccountAuthRes { ctid_trader_account_id: acct, ..Default::default() }, id),
        p if p == pb::ProtoOaTraderReq::PAYLOAD => reply(
            &pb::ProtoOaTraderRes {
                ctid_trader_account_id: acct,
                trader: pb::ProtoOaTrader {
                    ctid_trader_account_id: acct,
                    deposit_asset_id: 9,
                    broker_name: Some("blackbull".into()),
                    leverage_in_cents: Some(50_000),
                    ..Default::default()
                },
                ..Default::default()
            },
            id,
        ),
        p if p == pb::ProtoOaAssetListReq::PAYLOAD => reply(
            &pb::ProtoOaAssetListRes {
                ctid_trader_account_id: acct,
                asset: vec![pb::ProtoOaAsset { asset_id: 9, name: "USD".into(), ..Default::default() }],
                ..Default::default()
            },
            id,
        ),
        p if p == pb::ProtoOaSymbolsListReq::PAYLOAD => reply(
            &pb::ProtoOaSymbolsListRes {
                ctid_trader_account_id: acct,
                symbol: vec![
                    pb::ProtoOaLightSymbol { symbol_id: 41, symbol_name: Some("XAUUSD".into()), ..Default::default() },
                    pb::ProtoOaLightSymbol { symbol_id: 1001, symbol_name: Some("NVDA.NAS".into()), ..Default::default() },
                ],
                ..Default::default()
            },
            id,
        ),
        p if p == pb::ProtoOaSymbolCategoryListReq::PAYLOAD => reply(&pb::ProtoOaSymbolCategoryListRes { ctid_trader_account_id: acct, ..Default::default() }, id),
        p if p == pb::ProtoOaGetTickDataReq::PAYLOAD => {
            if st.rate_limit_first_tick_req {
                st.rate_limit_first_tick_req = false;
                return reply(&pb::ProtoOaErrorRes { error_code: "BLOCKED_PAYLOAD_TYPE".into(), retry_after: Some(1), ..Default::default() }, id);
            }
            let r: pb::ProtoOaGetTickDataReq = f.parse().unwrap();
            let (from, to) = (r.from_timestamp.unwrap(), r.to_timestamp.unwrap());
            // inclusive at both ends (the harder case for the pager)
            let sel: Vec<Tick> = st.ticks.iter().copied().filter(|t| t.ts >= from && t.ts <= to).collect();
            let has_more = sel.len() > st.page;
            let chunk = &sel[sel.len().saturating_sub(st.page)..];
            reply(&pb::ProtoOaGetTickDataRes { ctid_trader_account_id: acct, tick_data: encode_page(chunk), has_more, ..Default::default() }, id)
        }
        p if p == pb::ProtoOaGetTrendbarsReq::PAYLOAD => {
            let r: pb::ProtoOaGetTrendbarsReq = f.parse().unwrap();
            assert_eq!(r.period, Period::M5.pb());
            let (from, to) = (r.from_timestamp.unwrap(), r.to_timestamp.unwrap());
            st.trendbar_requests.lock().unwrap().push((from, to));
            if to - from > st.bar_max_range_ms {
                return reply(&pb::ProtoOaErrorRes { error_code: "INCORRECT_BOUNDARIES".into(), description: Some("period too long".into()), ..Default::default() }, id);
            }
            // inclusive at both ends, like the tick fake
            let sel: Vec<&Bar> = st.bars.iter().filter(|b| b.ts_open >= from && b.ts_open <= to).collect();
            let has_more = sel.len() > st.bar_chunk;
            let chunk: Vec<&Bar> = if !has_more {
                sel
            } else if st.keep_newest {
                sel[sel.len() - st.bar_chunk..].to_vec()
            } else {
                sel[..st.bar_chunk].to_vec()
            };
            let trendbar = chunk.iter().map(|b| trendbars::encode(b, Period::M5)).collect();
            reply(
                &pb::ProtoOaGetTrendbarsRes {
                    ctid_trader_account_id: acct,
                    period: r.period,
                    trendbar,
                    symbol_id: Some(r.symbol_id),
                    has_more: Some(has_more),
                    ..Default::default()
                },
                id,
            )
        }
        p if p == pb::ProtoOaSubscribeSpotsReq::PAYLOAD => {
            let mut out = reply(&pb::ProtoOaSubscribeSpotsRes { ctid_trader_account_id: acct, ..Default::default() }, id);
            out.extend(reply(
                &pb::ProtoOaSpotEvent { ctid_trader_account_id: acct, symbol_id: 41, bid: Some(400_012_300), ask: Some(400_045_600), timestamp: Some(5), ..Default::default() },
                &None,
            ));
            out
        }
        p if p == pb::ProtoHeartbeatEvent::PAYLOAD => Vec::new(),
        other => panic!("fake server got unexpected payload type {other}"),
    }
}

fn spawn_fake(st: Fake) -> Session {
    let (client, mut server) = tokio::io::duplex(1 << 20);
    tokio::spawn(async move {
        let mut st = st;
        let mut buf = Vec::new();
        let mut tmp = vec![0u8; 65536];
        loop {
            let n = match server.read(&mut tmp).await {
                Ok(0) | Err(_) => return,
                Ok(n) => n,
            };
            buf.extend_from_slice(&tmp[..n]);
            for f in split_frames(&mut buf).unwrap() {
                let out = handle(&f, &mut st);
                if !out.is_empty() && server.write_all(&out).await.is_err() {
                    return;
                }
            }
        }
    });
    Session::from_stream(client)
}

fn creds(secret: &str) -> CtraderCreds {
    CtraderCreds { client_id: "id".into(), client_secret: secret.into(), access_token: "tok".into(), refresh_token: None, env: CtraderEnv::Live, account_id: None }
}

fn fake(scope: i32) -> Fake {
    Fake {
        scope,
        ticks: vec![],
        page: 5,
        rate_limit_first_tick_req: false,
        log: Arc::new(Mutex::new(vec![])),
        bars: vec![],
        bar_chunk: 1000,
        keep_newest: true,
        bar_max_range_ms: i64::MAX,
        trendbar_requests: Arc::new(Mutex::new(vec![])),
    }
}

#[tokio::test]
async fn check_read_only_token() {
    let mut s = spawn_fake(fake(pb::ProtoOaClientPermissionScope::ScopeView as i32));
    let (r, err) = flow::check(&mut s, &creds("good-secret")).await;
    assert!(err.is_none(), "{err:?}");
    assert!(r.read_only);
    assert_eq!(r.server_version, "99");
    assert_eq!(r.account.as_ref().unwrap().ctid_trader_account_id, ACCOUNT, "live env picks the live account");
    assert_eq!(r.broker_name.as_deref(), Some("blackbull"));
    assert_eq!(r.deposit_asset.as_deref(), Some("USD"));
    assert_eq!(r.leverage, Some(500.0));
    assert_eq!(r.n_symbols, Some(2));
}

#[tokio::test]
async fn check_refuses_trading_scope_before_account_auth() {
    let f = fake(pb::ProtoOaClientPermissionScope::ScopeTrade as i32);
    let log = f.log.clone();
    let mut s = spawn_fake(f);
    let (r, err) = flow::check(&mut s, &creds("good-secret")).await;
    let msg = err.expect("must refuse").to_string();
    assert!(msg.contains("read-only") && msg.contains("accounts"), "{msg}");
    assert!(!r.read_only);
    assert!(!log.lock().unwrap().contains(&pb::ProtoOaAccountAuthReq::PAYLOAD), "no account auth on a trading token");
    // and login() refuses too
    let mut s2 = spawn_fake(fake(pb::ProtoOaClientPermissionScope::ScopeTrade as i32));
    assert!(flow::login(&mut s2, &creds("good-secret")).await.is_err());
}

#[tokio::test]
async fn bad_secret_is_an_error() {
    let mut s = spawn_fake(fake(0));
    let (_, err) = flow::check(&mut s, &creds("wrong")).await;
    assert!(err.unwrap().to_string().contains("CH_CLIENT_AUTH_FAILURE"));
}

#[tokio::test]
async fn ticks_page_backwards_with_rate_limit_retry() {
    let mut f = fake(0);
    // 23 ticks 100 ms apart, two share a millisecond with different prices
    let mut t: Vec<Tick> = (0..22).map(|i| Tick { ts: 1_000_000 + i * 100, raw: 400_000_000 + (i * 7) % 13 }).collect();
    t.push(Tick { ts: 1_000_000 + 21 * 100, raw: 400_000_099 });
    t.sort();
    f.ticks = t.clone();
    f.rate_limit_first_tick_req = true;
    let mut s = spawn_fake(f);
    s.hist_interval = std::time::Duration::from_millis(1);
    let (acct, _) = flow::login(&mut s, &creds("good-secret")).await.unwrap();
    let mut pages = 0;
    let got = ticks::fetch(&mut s, acct, 41, Side::Bid, 1_000_000, 1_000_000 + 2_200, |p, _| pages = p).await.unwrap();
    let mut want: Vec<Tick> = t.into_iter().filter(|x| x.ts < 1_000_000 + 2_200).collect();
    want.sort_by_key(|x| x.ts);
    assert_eq!(got.len(), want.len());
    assert_eq!(got.iter().map(|x| x.ts).collect::<Vec<_>>(), want.iter().map(|x| x.ts).collect::<Vec<_>>());
    assert!(pages >= 5, "pages {pages}");
}

#[tokio::test]
async fn spot_subscription_delivers_events() {
    let mut s = spawn_fake(fake(0));
    let (acct, _) = flow::login(&mut s, &creds("good-secret")).await.unwrap();
    s.subscribe_spots(acct, &[41]).await.unwrap();
    let f = s.next_event(std::time::Duration::from_secs(2)).await.unwrap().expect("spot event");
    let e = pb::ProtoOaSpotEvent::decode(f.payload.as_slice()).unwrap();
    assert_eq!(e.symbol_id, 41);
    assert!((at_ctrader::price(e.bid.unwrap() as i64) - 4000.123).abs() < 1e-9);
    s.heartbeat().await.unwrap();
}

#[tokio::test]
async fn raw_sender_refuses_order_messages() {
    let mut s = spawn_fake(fake(0));
    let f = Frame { payload_type: pb::ProtoOaPayloadType::ProtoOaNewOrderReq as u32, payload: vec![], client_msg_id: None };
    let e = s.send_frame(&f).await.unwrap_err().to_string();
    assert!(e.contains("allowlist"), "{e}");
}

const MIN5: i64 = 300_000;

fn ms(s: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(s).unwrap().timestamp_millis()
}

/// M5 bars on weekdays only (Mon 00:00 - Fri 22:00 UTC), from `from` to `to`, deterministic prices.
fn weekday_bars(from: i64, to: i64) -> Vec<Bar> {
    use chrono::{Datelike, Timelike};
    let mut v = Vec::new();
    let mut t = from;
    while t < to {
        let dt = chrono::DateTime::from_timestamp_millis(t).unwrap();
        let wd = dt.weekday().num_days_from_monday();
        if wd < 4 || (wd == 4 && dt.hour() < 22) {
            let i = (t / MIN5) % 97;
            v.push(Bar { ts_open: t, o: 110_000 + i, h: 110_050 + i, l: 109_950 + i, c: 110_010 + i, volume: 3 + i });
        }
        t += MIN5;
    }
    v
}

const FAR_FUTURE: &str = "2030-01-01T00:00:00Z";

#[tokio::test]
async fn trendbars_page_by_bisection_whichever_end_the_server_keeps() {
    let (start, end) = (ms("2024-01-10T00:00:00Z"), ms("2024-03-05T00:00:00Z"));
    let want = weekday_bars(start, end);
    for keep_newest in [true, false] {
        let mut f = fake(0);
        f.bars = want.clone();
        f.bar_chunk = 700;
        f.keep_newest = keep_newest;
        let mut s = spawn_fake(f);
        s.hist_interval = std::time::Duration::from_millis(1);
        let (acct, _) = flow::login(&mut s, &creds("good-secret")).await.unwrap();
        let mut got = Vec::new();
        let mut months = Vec::new();
        let (from, to) = (ms("2023-06-01T00:00:00Z"), ms("2024-04-01T00:00:00Z"));
        let rep = trendbars::fetch_history(
            &mut s,
            acct,
            41,
            Period::M5,
            from,
            to,
            ms(FAR_FUTURE),
            2,
            |_| None,
            |m, bars, partial| {
                months.push((m.name.clone(), partial));
                got.extend_from_slice(bars);
                Ok(())
            },
        )
        .await
        .unwrap();
        got.sort_by_key(|b| b.ts_open);
        assert_eq!(got, want, "keep_newest={keep_newest}");
        assert_eq!(rep.n_bars_fetched, want.len());
        assert_eq!(rep.earliest_ts, Some(start));
        assert_eq!(rep.latest_ts, Some(want.last().unwrap().ts_open));
        // Mar, Feb, Jan have data; Dec and Nov are empty -> stop (max_empty 2) long before 2023-06
        assert_eq!(rep.stopped, "history_start");
        assert_eq!(months.iter().map(|m| m.0.as_str()).collect::<Vec<_>>(), ["2024-03", "2024-02", "2024-01", "2023-12", "2023-11"]);
        assert!(months.iter().all(|m| !m.1), "complete months are final");
        assert!(rep.requests > months.len(), "truncated windows were split");
    }
}

#[tokio::test]
async fn trendbars_split_windows_on_boundary_errors_and_skip_months_on_disk() {
    let (start, end) = (ms("2024-01-01T00:00:00Z"), ms("2024-03-01T00:00:00Z"));
    let want = weekday_bars(start, end);
    let mut f = fake(0);
    f.bars = want.clone();
    f.bar_max_range_ms = 10 * 86_400_000; // the server rejects windows over 10 days
    let reqs = f.trendbar_requests.clone();
    let mut s = spawn_fake(f);
    s.hist_interval = std::time::Duration::from_millis(1);
    let (acct, _) = flow::login(&mut s, &creds("good-secret")).await.unwrap();
    let mut got = Vec::new();
    let mut fetched = Vec::new();
    // February is already on disk: it must not be requested again
    let feb = OnDisk { n: 1, first_ts: Some(ms("2024-02-01T00:00:00Z")), last_ts: Some(ms("2024-02-29T21:55:00Z")) };
    let rep = trendbars::fetch_history(
        &mut s,
        acct,
        41,
        Period::M5,
        start,
        end,
        ms(FAR_FUTURE),
        2,
        |m| (m.name == "2024-02").then_some(feb),
        |m, bars, _| {
            fetched.push(m.name.clone());
            got.extend_from_slice(bars);
            Ok(())
        },
    )
    .await
    .unwrap();
    assert_eq!(fetched, ["2024-01"]);
    assert_eq!(rep.months_skipped, 1);
    got.sort_by_key(|b| b.ts_open);
    let jan: Vec<Bar> = want.iter().copied().filter(|b| b.ts_open < ms("2024-02-01T00:00:00Z")).collect();
    assert_eq!(got, jan);
    assert_eq!(rep.latest_ts, feb.last_ts, "bars on disk count towards the history extent");
    let reqs = reqs.lock().unwrap();
    assert!(reqs.iter().all(|(a, b)| *a >= start && *b <= ms("2024-02-01T00:00:00Z")), "no request touched February");
    assert!(reqs.iter().any(|(a, b)| b - a > 10 * 86_400_000), "the month was tried whole first");
}

#[tokio::test]
async fn trendbars_current_month_is_partial() {
    let mut f = fake(0);
    f.bars = weekday_bars(ms("2024-01-01T00:00:00Z"), ms("2024-01-20T00:00:00Z"));
    let mut s = spawn_fake(f);
    s.hist_interval = std::time::Duration::from_millis(1);
    let (acct, _) = flow::login(&mut s, &creds("good-secret")).await.unwrap();
    let mut flags = Vec::new();
    let now = ms("2024-01-20T00:00:00Z");
    trendbars::fetch_history(
        &mut s,
        acct,
        41,
        Period::M5,
        ms("2023-12-01T00:00:00Z"),
        now,
        now,
        2,
        |_| None,
        |m, _, p| {
            flags.push((m.name.clone(), p));
            Ok(())
        },
    )
    .await
    .unwrap();
    assert_eq!(flags, [("2024-01".to_string(), true), ("2023-12".to_string(), false)]);
}

#[tokio::test]
async fn trendbars_request_is_on_the_read_only_allowlist() {
    let f = fake(0);
    let log = f.log.clone();
    let mut s = spawn_fake(f);
    let (acct, _) = flow::login(&mut s, &creds("good-secret")).await.unwrap();
    let mut n = 0;
    trendbars::fetch_window(&mut s, acct, 41, Period::M5, 0, 3_600_000, &mut n).await.unwrap();
    assert_eq!(n, 1);
    assert!(log.lock().unwrap().contains(&pb::ProtoOaGetTrendbarsReq::PAYLOAD));
}
