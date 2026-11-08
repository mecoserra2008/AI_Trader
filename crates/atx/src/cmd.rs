use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use at_alt as alt;
use at_bybit::market::{MarketData, PriceType, period_ms};
use at_bybit::{BybitRest, Credentials, archive, lake as bl, ws};
use at_core::config::{self, Environment, RiskConfig, Settings};
use at_core::num::{Round, round_to_step};
use at_core::time::{DAY, Freq, Ms, format_date, now_ms, parse_ts};
use at_core::types::{EntryType, InstrumentRules, Side, TradeProposal};
use at_exec::Broker;
use at_exec::bybit::BybitBroker;
use at_risk::SizedOrder;
use at_store::{Lake, WriteReport};
use serde_yaml::Value as Yaml;

use crate::Range;

pub struct Ctx {
    pub s: Settings,
    pub risk: RiskConfig,
    pub lake: Lake,
    sources: Yaml,
    raw: Yaml,
}

impl Ctx {
    pub fn load() -> anyhow::Result<Self> {
        let (s, risk) = config::load_default()?;
        let lake = Lake::open(s.lake_dir(), &s.schema_path())?;
        let sources: Yaml = serde_yaml::from_str(&std::fs::read_to_string(s.root.join("config/datasources.yaml"))?)?;
        let raw: Yaml = serde_yaml::from_str(&std::fs::read_to_string(s.root.join("config/settings.yaml"))?)?;
        Ok(Self { s, risk, lake, sources, raw })
    }

    fn rest_for(&self, env: Environment, creds: Option<Credentials>) -> anyhow::Result<BybitRest> {
        let b = &self.s.venues.bybit;
        BybitRest::new(&self.s.bybit_endpoints(env)?.rest, creds, b.recv_window_ms, b.rest_rate_limit_per_s)
    }

    /// Public market data always comes from `market_data_env` (mainnet).
    pub(crate) fn market_rest(&self) -> anyhow::Result<BybitRest> {
        self.rest_for(self.s.venues.bybit.market_data_env, None)
    }

    /// Execution environment with credentials, if present.
    pub(crate) fn exec_rest(&self) -> anyhow::Result<BybitRest> {
        let env = self.s.environment;
        let creds = config::bybit_credentials(env).map(|(k, s)| Credentials { api_key: k, api_secret: s });
        self.rest_for(env, creds)
    }

    fn range(&self, r: &Range) -> anyhow::Result<(Ms, Ms)> {
        let start = parse_ts(r.start.as_deref().unwrap_or(&self.s.data.history_start))?;
        let end = r.end.as_deref().map(parse_ts).transpose()?.unwrap_or_else(now_ms);
        anyhow::ensure!(start < end, "start must be before end");
        Ok((start, end))
    }

    pub(crate) fn latency(&self, table: &str) -> Ms {
        self.s.data.latency(table)
    }

    pub(crate) fn alt_latency(&self, source: &str) -> Ms {
        self.sources["defaults"]["known_ts_latency_ms"][source].as_i64().unwrap_or(DAY)
    }

    pub(crate) fn category(&self) -> &str {
        &self.s.venues.bybit.category
    }

    /// The raw settings.yaml document (sections without a typed struct, e.g. `flow`, `live`).
    pub(crate) fn raw(&self) -> &Yaml {
        &self.raw
    }
}

fn report(table: &str, what: &str, r: &WriteReport) {
    println!("{table:<14} {what:<24} {:>9} rows  {:>3} files", r.rows, r.files.len());
}

async fn symbols(ctx: &Ctx, arg: &str) -> anyhow::Result<Vec<String>> {
    if arg == "universe" {
        return universe(ctx).await;
    }
    Ok(arg.split(',').map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).collect())
}

// ---------------------------------------------------------------- check

async fn probe<F, T>(name: &str, fut: F) -> bool
where
    F: Future<Output = anyhow::Result<T>>,
    T: std::fmt::Display,
{
    let t0 = Instant::now();
    match tokio::time::timeout(Duration::from_secs(20), fut).await {
        Ok(Ok(info)) => {
            println!("  OK    {name:<34} {:>6} ms  {info}", t0.elapsed().as_millis());
            true
        }
        Ok(Err(e)) => {
            println!("  FAIL  {name:<34} {:>6} ms  {e:#}", t0.elapsed().as_millis());
            false
        }
        Err(_) => {
            println!("  FAIL  {name:<34}  timeout");
            false
        }
    }
}

fn skip(name: &str, why: &str) {
    println!("  --    {name:<34}            {why}");
}

async fn http_status(http: &reqwest::Client, url: &str) -> anyhow::Result<String> {
    let r = http.get(url).send().await?;
    let st = r.status();
    if st.as_u16() == 429 {
        return Ok("HTTP 429: reachable, rate-limited (respect source limits when ingesting)".into());
    }
    if !st.is_success() {
        bail!("HTTP {st}");
    }
    Ok(format!("HTTP {}", st.as_u16()))
}

pub async fn check(ctx: &Ctx) -> anyhow::Result<()> {
    let mut ok = true;
    println!("environment: {}  (execution venue)", ctx.s.environment.as_str());
    println!("\nBybit REST");
    for env in [Environment::Mainnet, Environment::Demo, Environment::Testnet] {
        let rest = ctx.rest_for(env, None)?;
        ok &= probe(&format!("bybit {} {}", env.as_str(), rest.base()), async {
            let t0 = now_ms();
            let server = MarketData::new(&rest, ctx.category()).server_time_ms().await?;
            let skew = server - (t0 + now_ms()) / 2;
            Ok(format!("clock skew {skew:+} ms"))
        })
        .await;
    }

    println!("\nBybit credentials ({})", ctx.s.environment.as_str());
    let exec = ctx.exec_rest()?;
    if exec.has_credentials() {
        ok &= probe("wallet balance (private, signed)", async {
            let w = at_bybit::trading::Trading::new(&exec, ctx.category(), &ctx.s.venues.bybit.settle_coin).wallet().await?;
            Ok(format!("equity {:.2} USDT", w.total_equity))
        })
        .await;
    } else {
        skip("private REST", &format!("no BYBIT_{}_API_KEY/SECRET in .env", ctx.s.environment.as_str().to_uppercase()));
    }

    println!("\nBybit WebSocket");
    let ws_url = ctx.s.bybit_endpoints(ctx.s.venues.bybit.market_data_env)?.ws_public.clone();
    ok &= probe("public stream tickers.BTCUSDT", async {
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let h = tokio::spawn(ws::run(ws_url, vec!["tickers.BTCUSDT".into()], None, tx));
        let m = rx.recv().await.context("no message")?;
        h.abort();
        Ok(format!("first message on {}", m.topic))
    })
    .await;

    println!("\nData sources");
    let http = alt::http()?;
    let day = format_date(now_ms() - 2 * DAY);
    let archive = archive::archive_url(&ctx.s.venues.bybit.public_trades_archive, "BTCUSDT", &day);
    ok &= probe("bybit trade archive", async { http.head(&archive).send().await?.error_for_status().map(|_| format!("{day} present")).map_err(Into::into) }).await;
    ok &= probe("alternative.me fear & greed", http_status(&http, "https://api.alternative.me/fng/?limit=1")).await;
    ok &= probe("deribit public api", http_status(&http, "https://www.deribit.com/api/v2/public/test")).await;
    ok &= probe("defillama", http_status(&http, "https://api.llama.fi/v2/chains")).await;
    ok &= probe("binance data vision", async {
        http.head(format!("https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{day}.zip")).send().await?.error_for_status()?;
        Ok("reachable")
    })
    .await;
    ok &= probe("gdelt doc api", http_status(&http, "https://api.gdeltproject.org/api/v2/doc/doc?query=bitcoin&mode=timelinevol&format=json&timespan=1h")).await;
    ok &= probe("polymarket gamma api", http_status(&http, "https://gamma-api.polymarket.com/markets?limit=1")).await;
    match config::secret("FRED_API_KEY") {
        Some(k) => ok &= probe("fred / alfred", http_status(&http, &format!("https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key={k}&file_type=json"))).await,
        None => skip("fred / alfred", "no FRED_API_KEY"),
    }
    match config::secret("CRYPTOPANIC_API_KEY") {
        Some(k) => ok &= probe("cryptopanic", http_status(&http, &format!("https://cryptopanic.com/api/developer/v2/posts/?auth_token={k}"))).await,
        None => skip("cryptopanic", "no CRYPTOPANIC_API_KEY"),
    }

    println!("\nKeys present (not called)");
    for k in ["ANTHROPIC_API_KEY", "COINGECKO_API_KEY", "COINDESK_API_KEY", "TARDIS_API_KEY", "COINGLASS_API_KEY", "GLASSNODE_API_KEY"] {
        println!("  {}  {k}", if config::secret(k).is_some() { "yes" } else { " no" });
    }

    println!("\nLake");
    std::fs::create_dir_all(ctx.lake.root())?;
    println!("  OK    {}  ({} tables in schema)", ctx.lake.root().display(), ctx.lake.schema().tables.len());

    println!("\n{}", if ok { "all probes passed" } else { "some probes FAILED" });
    Ok(())
}

// ---------------------------------------------------------------- universe & instruments

pub async fn universe(ctx: &Ctx) -> anyhow::Result<Vec<String>> {
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    let u = &ctx.s.universe;
    let instruments: HashMap<String, _> = md.instruments().await?.into_iter().map(|i| (i.symbol.clone(), i)).collect();
    let min_launch = now_ms() - u.min_listing_age_days * DAY;
    let mut rows: Vec<(String, f64)> = md
        .tickers()
        .await?
        .into_iter()
        .filter(|t| {
            instruments.get(&t.symbol).is_some_and(|i| {
                i.status == "Trading" && i.contract_type == u.contract_type && u.symbol_types.contains(&i.symbol_type) && i.quote == u.quote && i.launch_ts.unwrap_or(0) <= min_launch
            })
        })
        .filter(|t| t.turnover_24h >= u.min_24h_turnover_usd && !u.exclude.contains(&t.symbol))
        .map(|t| (t.symbol, t.turnover_24h))
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut out: Vec<String> = rows.into_iter().take(u.max_symbols).map(|r| r.0).collect();
    for s in &u.always_include {
        if !out.contains(s) {
            out.push(s.clone());
        }
    }
    Ok(out)
}

pub async fn instruments(ctx: &Ctx) -> anyhow::Result<()> {
    let rest = ctx.market_rest()?;
    let xs = MarketData::new(&rest, ctx.category()).instruments().await?;
    let r = ctx.lake.write("instruments", &bl::instruments(ctx.category(), &xs, now_ms()))?;
    report("instruments", ctx.category(), &r);
    Ok(())
}

// ---------------------------------------------------------------- market data

pub async fn klines(ctx: &Ctx, syms: &str, freq: &str, price_type: &str, range: &Range) -> anyhow::Result<()> {
    let (start, end) = ctx.range(range)?;
    let freq = Freq::parse(freq)?;
    let pt = PriceType::parse(price_type)?;
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    for sym in symbols(ctx, syms).await? {
        let xs = md.klines(&sym, freq, pt, start, end).await?;
        let r = ctx.lake.write("bars", &bl::klines(&sym, freq, pt, &xs, ctx.latency("bars")))?;
        report("bars", &format!("{sym} {} {freq}", pt.as_str()), &r);
    }
    Ok(())
}

pub async fn funding(ctx: &Ctx, syms: &str, range: &Range) -> anyhow::Result<()> {
    let (start, end) = ctx.range(range)?;
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    for sym in symbols(ctx, syms).await? {
        let xs = md.funding_history(&sym, start, end).await?;
        report("funding", &sym, &ctx.lake.write("funding", &bl::funding(&sym, &xs, ctx.latency("funding")))?);
    }
    Ok(())
}

pub async fn open_interest(ctx: &Ctx, syms: &str, interval: Option<String>, range: &Range) -> anyhow::Result<()> {
    let (start, end) = ctx.range(range)?;
    let interval = interval.unwrap_or_else(|| ctx.s.data.open_interest_interval.clone());
    let pm = period_ms(&interval)?;
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    for sym in symbols(ctx, syms).await? {
        let xs = md.open_interest(&sym, &interval, start, end).await?;
        let b = bl::open_interest(&sym, &interval, pm, &xs, ctx.latency("open_interest"));
        report("open_interest", &format!("{sym} {interval}"), &ctx.lake.write("open_interest", &b)?);
    }
    Ok(())
}

pub async fn ls_ratio(ctx: &Ctx, syms: &str, period: Option<String>, range: &Range) -> anyhow::Result<()> {
    let (start, end) = ctx.range(range)?;
    let period = period.unwrap_or_else(|| ctx.s.data.ls_ratio_period.clone());
    let pm = period_ms(&period)?;
    let rest = ctx.market_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    for sym in symbols(ctx, syms).await? {
        let xs = md.long_short_ratio(&sym, &period, start, end).await?;
        let b = bl::ls_ratio(&sym, &period, pm, &xs, ctx.latency("ls_ratio"));
        report("ls_ratio", &format!("{sym} {period}"), &ctx.lake.write("ls_ratio", &b)?);
    }
    Ok(())
}

pub async fn trades(ctx: &Ctx, symbol: &str, from: &str, to: &str, freqs: Option<String>, raw: bool, cache: bool) -> anyhow::Result<()> {
    let freqs: Vec<Freq> = match freqs {
        Some(f) => f.split(',').map(Freq::parse).collect::<Result<_, _>>()?,
        None => ctx.s.data.flow_freqs.iter().map(|f| Freq::parse(f)).collect::<Result<_, _>>()?,
    };
    let http = alt::http()?;
    let cache_dir = cache.then(|| ctx.s.abs(&ctx.s.paths.raw_cache).join("bybit_trades"));
    let symbol = symbol.to_uppercase();
    let (mut day, last) = (parse_ts(from)?, parse_ts(to)?);
    while day <= last {
        let date = format_date(day);
        match archive::fetch_day(&http, &ctx.s.venues.bybit.public_trades_archive, &symbol, &date, cache_dir.as_deref()).await? {
            None => println!("trades         {symbol} {date}            no archive file"),
            Some(ts) => {
                for &f in &freqs {
                    let bars = archive::aggregate(&ts, f);
                    let r = ctx.lake.write("flow_bars", &bl::flow_bars(&symbol, f, &bars, ctx.latency("flow_bars")))?;
                    report("flow_bars", &format!("{symbol} {date} {f}"), &r);
                }
                if raw {
                    let r = ctx.lake.write("trades_raw", &bl::trades(&symbol, &ts, ctx.latency("trades")))?;
                    report("trades_raw", &format!("{symbol} {date}"), &r);
                }
            }
        }
        day += DAY;
    }
    Ok(())
}

// ---------------------------------------------------------------- alternative data

pub async fn alt_fng(ctx: &Ctx) -> anyhow::Result<()> {
    let xs = alt::fng::fetch_all(&alt::http()?, ctx.alt_latency("alternative_me_fng")).await?;
    report("alt_series", "alternative_me fear_greed", &ctx.lake.write("alt_series", &alt::to_batch(alt::fng::SOURCE, alt::fng::PIT, &xs))?);
    Ok(())
}

pub async fn alt_dvol(ctx: &Ctx, currencies: &str, resolution: &str, range: &Range) -> anyhow::Result<()> {
    let (start, end) = ctx.range(range)?;
    let http = alt::http()?;
    for c in currencies.split(',').map(str::trim) {
        let xs = alt::deribit::dvol(&http, c, resolution, start, end, ctx.alt_latency("deribit_dvol")).await?;
        report("alt_series", &format!("deribit dvol {c}"), &ctx.lake.write("alt_series", &alt::to_batch(alt::deribit::SOURCE, alt::deribit::PIT, &xs))?);
    }
    Ok(())
}

pub async fn alt_stablecoins(ctx: &Ctx) -> anyhow::Result<()> {
    let xs = alt::defillama::stablecoins_total(&alt::http()?, ctx.alt_latency("defillama_stablecoins")).await?;
    report("alt_series", "defillama stablecoins", &ctx.lake.write("alt_series", &alt::to_batch(alt::defillama::SOURCE, alt::defillama::PIT, &xs))?);
    Ok(())
}

pub async fn alt_fred(ctx: &Ctx, series: Option<String>) -> anyhow::Result<()> {
    let key = config::secret("FRED_API_KEY").context("FRED_API_KEY missing in .env (free: https://fred.stlouisfed.org/docs/api/api_key.html)")?;
    let list: Vec<String> = match series {
        Some(s) => s.split(',').map(|x| x.trim().to_string()).collect(),
        None => ctx.sources["defaults"]["fred_alfred"]["series"]
            .as_sequence()
            .map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default(),
    };
    let http = alt::http()?;
    for sid in list {
        let xs = alt::fred::observations_all_vintages(&http, &key, &sid, ctx.alt_latency("fred_alfred")).await?;
        report("alt_series", &format!("fred {sid}"), &ctx.lake.write("alt_series", &alt::to_batch(alt::fred::SOURCE, alt::fred::PIT, &xs))?);
    }
    Ok(())
}

// ---------------------------------------------------------------- account & streams

pub async fn account(ctx: &Ctx) -> anyhow::Result<()> {
    let rest = ctx.exec_rest()?;
    anyhow::ensure!(rest.has_credentials(), "no API credentials for {} in .env", ctx.s.environment.as_str());
    let t = at_bybit::trading::Trading::new(&rest, ctx.category(), &ctx.s.venues.bybit.settle_coin);
    let w = t.wallet().await?;
    println!("environment {}  equity {:.2}  available {:.2}", ctx.s.environment.as_str(), w.total_equity, w.total_available);
    let ps = t.positions().await?;
    if ps.is_empty() {
        println!("no open positions");
    }
    for p in ps {
        println!("{:<12} {:<4} size {:<10} avg {:<12} mark {:<12} SL {:<10} TP {:<10} uPnL {:.2}", p.symbol, p.side, p.size, p.avg_price, p.mark_price, p.stop_loss, p.take_profit, p.unrealised_pnl);
    }
    Ok(())
}

pub async fn stream(ctx: &Ctx, syms: &str, topics: &str, seconds: u64) -> anyhow::Result<()> {
    let syms = symbols(ctx, syms).await?;
    let topics: Vec<String> = topics.split(',').flat_map(|t| syms.iter().map(move |s| format!("{}.{s}", t.trim()))).collect();
    let url = ctx.s.bybit_endpoints(ctx.s.venues.bybit.market_data_env)?.ws_public.clone();
    let (tx, mut rx) = tokio::sync::mpsc::channel(10_000);
    let handle = tokio::spawn(ws::run(url, topics.clone(), None, tx));
    let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
    let mut counts: BTreeMap<String, u64> = BTreeMap::new();
    while let Ok(Some(m)) = tokio::time::timeout_at(deadline, rx.recv()).await {
        *counts.entry(m.topic).or_default() += 1;
    }
    handle.abort();
    println!("{} topics subscribed, {seconds}s:", topics.len());
    for t in &topics {
        println!("  {:<36} {:>7} messages", t, counts.get(t).copied().unwrap_or(0));
    }
    Ok(())
}

/// Smoke test of the real execution path (BybitBroker) on demo/testnet: minimum-size market
/// order with SL/TP 3% away, confirm the position, then close it.
pub async fn order_roundtrip(ctx: &Ctx, symbol: &str, yes: bool) -> anyhow::Result<()> {
    if ctx.s.environment == Environment::Mainnet {
        bail!("order-roundtrip is disabled on mainnet");
    }
    if !yes {
        bail!("this places a real order on {}; re-run with --yes", ctx.s.environment.as_str());
    }
    let symbol = symbol.to_uppercase();
    let rest = ctx.exec_rest()?;
    let md = MarketData::new(&rest, ctx.category());
    let inst = md.instruments().await?.into_iter().find(|i| i.symbol == symbol).with_context(|| format!("{symbol} not listed"))?;
    let price = md.tickers().await?.into_iter().find(|t| t.symbol == symbol).context("no ticker")?.last_price;
    let rule = InstrumentRules {
        symbol: symbol.clone(),
        tick_size: inst.tick_size,
        qty_step: inst.qty_step,
        min_qty: inst.min_qty,
        max_qty: inst.max_qty,
        max_mkt_qty: inst.max_mkt_qty,
        min_notional: inst.min_notional,
        max_leverage: inst.max_leverage,
    };
    let qty = round_to_step((inst.min_notional * 1.2 / price).max(inst.min_qty), inst.qty_step, Round::Up);
    let stop = round_to_step(price * 0.97, inst.tick_size, Round::Down);
    let tp = round_to_step(price * 1.03, inst.tick_size, Round::Up);
    let proposal = TradeProposal {
        id: format!("roundtrip-{}", now_ms()),
        symbol: symbol.clone(),
        side: Side::Long,
        entry_type: EntryType::Market,
        entry: price,
        stop,
        take_profit: tp,
        p_tp: 0.5,
        cluster: "test".into(),
        channel: "smoke-test".into(),
    };
    let order = SizedOrder { proposal: proposal.clone(), entry: price, stop, take_profit: tp, qty, notional: qty * price, risk_amount: qty * (price - stop), reward_risk: 1.0, cost_r: 0.0, net_expected_r: 0.0 };
    let mut broker = BybitBroker::new(&ctx.s, rest, HashMap::from([(symbol.clone(), rule)]))?;
    println!("{}: BUY {qty} {symbol} @ market (~{price}), SL {stop}, TP {tp}", broker.name());
    let id = broker.submit(&order, &proposal.id).await?;
    println!("  order id {id}");
    tokio::time::sleep(Duration::from_secs(2)).await;
    let pos = broker.positions().await?;
    match pos.iter().find(|p| p.symbol == symbol) {
        Some(p) => println!("  position: {} qty {} @ {} SL {} TP {}", p.symbol, p.qty, p.entry_price, p.stop, p.take_profit),
        None => println!("  WARNING: position not visible yet"),
    }
    broker.close(&symbol).await?;
    tokio::time::sleep(Duration::from_secs(2)).await;
    let still_open = broker.positions().await?.iter().any(|p| p.symbol == symbol);
    println!("  closed: {}", if still_open { "NO, check the account" } else { "yes" });
    Ok(())
}

// ---------------------------------------------------------------- order-flow collector

pub async fn collect(ctx: Ctx, minutes: Option<u64>, category: Option<String>, symbols: Option<String>, depth: Option<u32>) -> anyhow::Result<()> {
    use at_flow::collector::{CollectorCfg, StreamCfg, run};
    let f = &ctx.raw["flow"];
    let base = f["ws_public_base"].as_str().context("flow.ws_public_base missing")?.trim_end_matches('/').to_string();
    let override_syms: Option<Vec<String>> = symbols.map(|s| s.split(',').map(|x| x.trim().to_uppercase()).collect());
    let mut streams = Vec::new();
    for st in f["streams"].as_sequence().context("flow.streams missing")? {
        let cat = st["category"].as_str().context("stream.category")?.to_string();
        if category.as_deref().is_some_and(|c| c != cat) {
            continue;
        }
        let syms = override_syms.clone().unwrap_or_else(|| {
            st["symbols"].as_sequence().map(|v| v.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
        });
        streams.push(StreamCfg { ws_url: format!("{base}/{cat}"), category: cat, symbols: syms, depth: depth.unwrap_or(st["depth"].as_u64().unwrap_or(50) as u32) });
    }
    anyhow::ensure!(!streams.is_empty(), "no streams selected");
    for s in &streams {
        println!("stream {:<7} depth {:<4} {} symbols  {}", s.category, s.depth, s.symbols.len(), s.ws_url);
    }
    let cfg = CollectorCfg {
        venue: at_bybit::VENUE.into(),
        streams,
        symbols_per_conn: f["symbols_per_conn"].as_u64().unwrap_or(25) as usize,
        grace_ms: f["grace_ms"].as_i64().unwrap_or(1500),
        flush_secs: f["flush_secs"].as_u64().unwrap_or(300),
        run_for: minutes.map(|m| Duration::from_secs(m * 60)),
    };
    let t0 = Instant::now();
    let stats = run(cfg, std::sync::Arc::new(ctx.lake)).await?;
    println!("collector stopped after {:.1} min: {stats:?}", t0.elapsed().as_secs_f64() / 60.0);
    Ok(())
}

// ---------------------------------------------------------------- live market-data updater

pub async fn update(ctx: &Ctx, once: bool) -> anyhow::Result<()> {
    crate::updater::run(ctx, once).await
}
