//! `atx ctrader ...`: read-only cTrader Open API (BlackBull) for the cross-venue study (H-ARB1, H-ARB2).
//! Never sends order messages (at-ctrader allowlist) and refuses tokens with trading scope.
//! Output goes to data/research/xvenue/ (git-ignored), never the lake.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use at_core::time::{format_date, now_ms, parse_ts};
use at_ctrader::codec::OaMsg;
use at_ctrader::creds::{self, CtraderCreds, CtraderEnv};
use at_ctrader::ticks::{self, Side};
use at_ctrader::trendbars::{self, OnDisk, Period};
use at_ctrader::{Session, flow, pb, store};
use clap::Subcommand;
use prost::Message as _;
use serde_json::json;

use crate::cmd::Ctx;

#[derive(Subcommand)]
pub enum CtraderCmd {
    /// Keys present/missing, app auth, accounts, token scope (refuses trading scope), account auth, symbol count.
    Check,
    /// Symbol list -> data/research/xvenue/ctrader_symbols.json; --resolve maps config/xvenue.yaml -> ctrader_map.json (with symbol details).
    Symbols {
        #[arg(long)]
        resolve: bool,
        /// Only print names containing this text (case-insensitive).
        #[arg(long)]
        filter: Option<String>,
        /// Candidate file for --resolve, xvenue format (`instruments: [{bybit|asset, ticker, ctrader_candidates}]`); default config/xvenue.yaml.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Where --resolve writes the map (default data/research/xvenue/ctrader_map.json).
        #[arg(long)]
        map_out: Option<PathBuf>,
    },
    /// Historical trend bars (ProtoOAGetTrendbarsReq; one calendar month per window, split on hasMore or range errors,
    /// <= ~4.5 req/s), walking back until the history start -> <out>/<symbol>/<period>/<YYYY-MM>.parquet.
    Trendbars {
        /// cTrader symbol name (as in `symbols`) or numeric symbol id.
        #[arg(long)]
        symbol: String,
        /// M1 | M5 | M15 | M30 | H1 | H4 | D1 | W1 | MN1
        #[arg(long, default_value = "M5")]
        period: String,
        /// Earliest start (YYYY-MM-DD or RFC 3339, UTC); the walk stops earlier at the server's history start.
        #[arg(long, default_value = "2000-01-01")]
        from: String,
        /// End, exclusive (default: now).
        #[arg(long)]
        to: Option<String>,
        /// Output root (default data/research/xvenue/ctrader_trendbars).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Consecutive empty months, after data was seen, that mark the history start.
        #[arg(long, default_value_t = 3)]
        max_empty: usize,
        /// Re-download months that already have a final file.
        #[arg(long)]
        force: bool,
    },
    /// Historical ticks (ProtoOAGetTickDataReq, paged on hasMore, <= ~4.5 req/s) -> ctrader_ticks/<symbol>/<date>_<side>.parquet.
    Ticks {
        /// cTrader symbol name (as in `symbols`) or numeric symbol id.
        #[arg(long)]
        symbol: String,
        /// Start (YYYY-MM-DD or RFC 3339, UTC).
        #[arg(long)]
        from: String,
        /// End, exclusive.
        #[arg(long)]
        to: String,
        /// bid | ask | both
        #[arg(long, default_value = "both")]
        side: String,
        /// Re-download days that already have a file.
        #[arg(long)]
        force: bool,
    },
    /// Live spots (ProtoOASubscribeSpotsReq) with heartbeats and reconnect -> ctrader_spots/<date>.jsonl.
    Stream {
        /// Comma-separated cTrader symbol names.
        #[arg(long)]
        symbols: String,
        /// Stop after this many seconds (default: until Ctrl-C).
        #[arg(long)]
        seconds: Option<u64>,
    },
    /// Round trip to the Open API: TCP + TLS connect, then N version requests (no keys needed for connect timing).
    Ping {
        #[arg(long, default_value_t = 20)]
        n: usize,
        /// live | demo (default CTRADER_ENV, else live).
        #[arg(long)]
        env: Option<String>,
    },
}

fn out_dir(ctx: &Ctx) -> PathBuf {
    ctx.s.abs(&ctx.s.paths.data_root).join("research").join("xvenue")
}

fn write_json(path: &PathBuf, v: &serde_json::Value) -> anyhow::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(v)?)?;
    Ok(())
}

fn print_presence() -> bool {
    println!("cTrader keys in .env (values never shown):");
    let mut ok = true;
    for (k, present) in creds::presence() {
        let req = creds::REQUIRED.contains(&k);
        println!("  {k:<24} {}{}", if present { "present" } else { "missing" }, if req { "" } else { "  (optional)" });
        ok &= present || !req;
    }
    ok
}

async fn login(c: &CtraderCreds) -> anyhow::Result<(Session, i64)> {
    let (mut s, _) = Session::connect(c.env).await.with_context(|| format!("connecting to {}:5035", c.env.host()))?;
    let (id, _) = flow::login(&mut s, c).await?;
    Ok((s, id))
}

pub async fn run(ctx: &Ctx, cmd: CtraderCmd) -> anyhow::Result<()> {
    match cmd {
        CtraderCmd::Check => check(ctx).await,
        CtraderCmd::Symbols { resolve, filter, config, map_out } => symbols(ctx, resolve, filter, config, map_out).await,
        CtraderCmd::Trendbars { symbol, period, from, to, out, max_empty, force } => trendbars_cmd(ctx, &symbol, &period, &from, to.as_deref(), out, max_empty, force).await,
        CtraderCmd::Ticks { symbol, from, to, side, force } => ticks_cmd(ctx, &symbol, &from, &to, &side, force).await,
        CtraderCmd::Stream { symbols, seconds } => stream(ctx, &symbols, seconds).await,
        CtraderCmd::Ping { n, env } => ping(ctx, n, env).await,
    }
}

async fn check(ctx: &Ctx) -> anyhow::Result<()> {
    if !print_presence() {
        anyhow::bail!("add the missing keys to .env (CTRADER_ENV = live or demo), then rerun `atx ctrader check`");
    }
    let c = CtraderCreds::from_env()?;
    println!("connecting to {}:5035 ({})", c.env.host(), c.env.as_str());
    let (mut s, t) = Session::connect(c.env).await?;
    println!("  tcp {:.0} ms, tls {:.0} ms", t.tcp_ms, t.tls_ms);
    let (r, err) = flow::check(&mut s, &c).await;
    println!("server version   {}", r.server_version);
    println!("token scope      {}", r.scope);
    println!("accounts on token: {}", r.accounts.len());
    for a in &r.accounts {
        println!("  ctid {:>10}  {}  broker {}", a.ctid_trader_account_id, if a.is_live { "live" } else { "demo" }, a.broker);
    }
    write_json(&out_dir(ctx).join("ctrader_check.json"), &json!({"checked_at_ms": now_ms(), "report": r, "error": err.as_ref().map(|e| e.to_string())}))?;
    if let Some(e) = err {
        if !r.read_only && !r.scope.is_empty() {
            println!("\nREFUSED: {e}");
        }
        return Err(e);
    }
    let a = r.account.as_ref().expect("set when no error");
    println!("using account    ctid {} ({}, broker {})", a.ctid_trader_account_id, if a.is_live { "live" } else { "demo" }, a.broker);
    println!("broker name      {}", r.broker_name.as_deref().unwrap_or("?"));
    println!("environment      {}", r.env);
    println!("deposit asset    {}", r.deposit_asset.as_deref().unwrap_or("?"));
    println!("account type     {}  swap-free {:?}  leverage 1:{}", r.account_type.as_deref().unwrap_or("?"), r.swap_free, r.leverage.unwrap_or(0.0));
    println!("symbols          {} ({} categories)", r.n_symbols.unwrap_or(0), r.n_categories.unwrap_or(0));
    println!("read-only        yes: OK for data. Next: atx ctrader symbols --resolve");
    Ok(())
}

#[derive(serde::Deserialize)]
struct XvenueCfg {
    instruments: Vec<XInst>,
}

#[derive(serde::Deserialize)]
struct XInst {
    #[serde(alias = "asset")]
    bybit: String,
    #[serde(default)]
    ctrader_candidates: Vec<String>,
    #[serde(default)]
    ticker: Option<String>,
}

/// Exact candidate (case-insensitive) first, then names that are `<ticker><sep><suffix>` (e.g. NVDA.NAS, NVDA.US).
fn resolve_one(inst: &XInst, names: &BTreeMap<String, i64>) -> (Option<(String, i64)>, Vec<String>) {
    let lower: BTreeMap<String, (&String, i64)> = names.iter().map(|(n, id)| (n.to_ascii_lowercase(), (n, *id))).collect();
    for c in &inst.ctrader_candidates {
        if let Some((n, id)) = lower.get(&c.to_ascii_lowercase()) {
            return (Some(((*n).clone(), *id)), vec![]);
        }
    }
    let Some(t) = inst.ticker.as_ref() else { return (None, vec![]) };
    let t = t.to_ascii_lowercase();
    let hits: Vec<String> = names
        .keys()
        .filter(|n| {
            let l = n.to_ascii_lowercase();
            l == t || (l.starts_with(&t) && l[t.len()..].starts_with(['.', '_', '-', '#', ' ']))
        })
        .cloned()
        .collect();
    match hits.as_slice() {
        [one] => (Some((one.clone(), names[one])), vec![]),
        _ => (None, hits),
    }
}

fn symbol_json(d: &pb::ProtoOaSymbol) -> serde_json::Value {
    let ct = d.commission_type.and_then(|v| pb::ProtoOaCommissionType::try_from(v).ok()).map(|v| v.as_str_name());
    let sw = d.swap_calculation_type.and_then(|v| pb::ProtoOaSwapCalculationType::try_from(v).ok()).map(|v| v.as_str_name());
    let rollover = d.swap_rollover3_days.and_then(|v| pb::ProtoOaDayOfWeek::try_from(v).ok()).map(|v| v.as_str_name());
    // preciseTradingCommissionRate: x 1e8 for non-percentage types, x 1e5 for PERCENTAGE_OF_VALUE (then in percent)
    let rate = d.precise_trading_commission_rate.map(|r| if ct == Some("PERCENTAGE_OF_VALUE") { r as f64 / 1e5 } else { r as f64 / 1e8 });
    json!({
        "symbol_id": d.symbol_id, "digits": d.digits, "pip_position": d.pip_position, "lot_size_cents": d.lot_size,
        "min_volume_cents": d.min_volume, "step_volume_cents": d.step_volume, "measurement_units": d.measurement_units,
        "commission_type": ct, "commission_rate": rate, "precise_min_commission": d.precise_min_commission.map(|v| v as f64 / 1e8),
        "min_commission_asset": d.min_commission_asset, "pnl_conversion_fee_rate_bp": d.pnl_conversion_fee_rate,
        "swap_long": d.swap_long, "swap_short": d.swap_short, "swap_calculation_type": sw, "swap_rollover_3days": rollover,
        "swap_period_h": d.swap_period, "swap_time_min_utc": d.swap_time, "charge_swap_at_weekends": d.charge_swap_at_weekends,
        "schedule_time_zone": d.schedule_time_zone,
        "schedule": d.schedule.iter().map(|i| [i.start_second, i.end_second]).collect::<Vec<_>>(),
        "holidays": d.holiday.iter().map(|h| json!({"name": h.name, "date_days": h.holiday_date, "recurring": h.is_recurring, "start_s": h.start_second, "end_s": h.end_second, "tz": h.schedule_time_zone})).collect::<Vec<_>>(),
        "trading_mode": d.trading_mode, "enable_short_selling": d.enable_short_selling,
    })
}

async fn symbols(ctx: &Ctx, resolve: bool, filter: Option<String>, config: Option<PathBuf>, map_out: Option<PathBuf>) -> anyhow::Result<()> {
    let c = CtraderCreds::from_env().context("run `atx ctrader check` first")?;
    let (mut s, acct) = login(&c).await?;
    let list = s.symbols(acct).await?;
    let cats: BTreeMap<i64, String> = s.categories(acct).await.unwrap_or_default().into_iter().map(|x| (x.id, x.name)).collect();
    let rows: Vec<serde_json::Value> = list
        .iter()
        .map(|x| json!({"id": x.symbol_id, "name": x.symbol_name, "description": x.description, "enabled": x.enabled, "category": x.symbol_category_id.and_then(|i| cats.get(&i))}))
        .collect();
    let dir = out_dir(ctx);
    write_json(&dir.join("ctrader_symbols.json"), &json!({"fetched_at_ms": now_ms(), "env": c.env.as_str(), "symbols": rows}))?;
    let f = filter.map(|x| x.to_ascii_lowercase());
    for x in &list {
        let name = x.symbol_name.clone().unwrap_or_default();
        if f.as_ref().is_none_or(|f| name.to_ascii_lowercase().contains(f)) {
            println!("{:>8}  {:<20} {}", x.symbol_id, name, x.description.as_deref().unwrap_or(""));
        }
    }
    println!("{} symbols -> {}", list.len(), dir.join("ctrader_symbols.json").display());
    if !resolve {
        return Ok(());
    }
    let cfg_path = config.unwrap_or_else(|| ctx.s.root.join("config/xvenue.yaml"));
    let cfg: XvenueCfg = serde_yaml::from_str(&std::fs::read_to_string(&cfg_path).with_context(|| format!("reading {}", cfg_path.display()))?)?;
    let names: BTreeMap<String, i64> = list.iter().filter_map(|x| Some((x.symbol_name.clone()?, x.symbol_id))).collect();
    let mut resolved = Vec::new();
    let mut ids = Vec::new();
    for inst in &cfg.instruments {
        anyhow::ensure!(inst.bybit != "SPYUSDT", "SPYUSDT must never be in config/xvenue.yaml");
        let (hit, ambiguous) = resolve_one(inst, &names);
        if let Some((_, id)) = &hit {
            ids.push(*id);
        }
        resolved.push((inst.bybit.clone(), hit, ambiguous));
    }
    let details: BTreeMap<i64, serde_json::Value> = s.symbol_details(acct, &ids).await?.iter().map(|d| (d.symbol_id, symbol_json(d))).collect();
    let mut map = serde_json::Map::new();
    println!("\n{:<14} {:<18} {:>8}  commission / swap", "bybit", "ctrader", "id");
    for (b, hit, amb) in resolved {
        match hit {
            Some((n, id)) => {
                let d = details.get(&id).cloned().unwrap_or(serde_json::Value::Null);
                println!("{b:<14} {n:<18} {id:>8}  {} {} / long {} short {}", d["commission_type"], d["commission_rate"], d["swap_long"], d["swap_short"]);
                map.insert(b, json!({"ctrader_name": n, "ctrader_id": id, "details": d}));
            }
            None => {
                println!("{b:<14} {:<18} {:>8}  candidates: {amb:?}", "UNRESOLVED", "-");
                map.insert(b, json!({"ctrader_name": null, "ctrader_id": null, "ambiguous": amb}));
            }
        }
    }
    let p = map_out.unwrap_or_else(|| dir.join("ctrader_map.json"));
    write_json(&p, &json!({"resolved_at_ms": now_ms(), "env": c.env.as_str(), "account_ctid": acct, "candidates": cfg_path.display().to_string(), "map": map}))?;
    println!("-> {}  (edit ctrader_candidates in {} for unresolved rows and rerun)", p.display(), cfg_path.display());
    Ok(())
}

async fn resolve_symbol(s: &mut Session, acct: i64, sym: &str) -> anyhow::Result<(String, i64)> {
    let list = s.symbols(acct).await?;
    if let Ok(id) = sym.parse::<i64>() {
        let name = list.iter().find(|x| x.symbol_id == id).and_then(|x| x.symbol_name.clone()).unwrap_or_else(|| sym.to_string());
        return Ok((name, id));
    }
    list.iter()
        .find(|x| x.symbol_name.as_deref().is_some_and(|n| n.eq_ignore_ascii_case(sym)))
        .map(|x| (x.symbol_name.clone().unwrap_or_default(), x.symbol_id))
        .ok_or_else(|| anyhow::anyhow!("symbol {sym:?} not found; see `atx ctrader symbols --filter ...`"))
}

async fn ticks_cmd(ctx: &Ctx, symbol: &str, from: &str, to: &str, side: &str, force: bool) -> anyhow::Result<()> {
    let sides = match side {
        "both" => vec![Side::Bid, Side::Ask],
        s => vec![Side::parse(s)?],
    };
    let (from_ms, to_ms) = (parse_ts(from)?, parse_ts(to)?);
    anyhow::ensure!(from_ms < to_ms, "--from must be before --to");
    anyhow::ensure!(to_ms <= now_ms() + 60_000, "--to is in the future");
    let c = CtraderCreds::from_env().context("run `atx ctrader check` first")?;
    let (mut s, acct) = login(&c).await?;
    let (name, id) = resolve_symbol(&mut s, acct, symbol).await?;
    let root = out_dir(ctx).join("ctrader_ticks");
    for d0 in store::days(from_ms, to_ms) {
        let (a, b) = (from_ms.max(d0), to_ms.min(d0 + store::DAY_MS));
        let date = format_date(d0);
        for &sd in &sides {
            let path = store::tick_path(&root, &name, &date, sd);
            if path.exists() && !force {
                println!("{name} {date} {}: exists, skipped", sd.as_str());
                continue;
            }
            let t0 = Instant::now();
            let ticks = ticks::fetch(&mut s, acct, id, sd, a, b, |p, oldest| {
                if p % 50 == 0 {
                    tracing::info!("{name} {date} {}: {p} pages, back to {}", sd.as_str(), at_core::time::to_datetime(oldest).format("%H:%M:%S"));
                }
            })
            .await?;
            let known = now_ms();
            store::write_ticks(&path, &ticks, known)?;
            let meta = json!({"symbol": name, "symbol_id": id, "date": date, "side": sd.as_str(), "from_ms": a, "to_ms": b, "n": ticks.len(), "known_ts": known, "env": c.env.as_str(), "secs": t0.elapsed().as_secs_f64()});
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(root.join("_meta.jsonl"))?;
            writeln!(f, "{meta}")?;
            println!("{name} {date} {}: {} ticks in {:.1} s -> {}", sd.as_str(), ticks.len(), t0.elapsed().as_secs_f64(), path.display());
        }
    }
    Ok(())
}

/// Row count and min / max `ts_open` of a trend bar file, from the parquet footer (no data pages are read).
fn trendbar_file_info(path: &std::path::Path) -> Option<OnDisk> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::file::statistics::Statistics;
    let r = SerializedFileReader::new(std::fs::File::open(path).ok()?).ok()?;
    let md = r.metadata();
    let n = md.file_metadata().num_rows() as usize;
    let (mut lo, mut hi): (Option<i64>, Option<i64>) = (None, None);
    for rg in md.row_groups() {
        if let Some(Statistics::Int64(st)) = rg.column(0).statistics() {
            lo = st.min_opt().map(|v| lo.map_or(*v, |x| x.min(*v)));
            hi = st.max_opt().map(|v| hi.map_or(*v, |x| x.max(*v)));
        }
    }
    Some(OnDisk { n, first_ts: lo, last_ts: hi })
}

#[allow(clippy::too_many_arguments)]
async fn trendbars_cmd(ctx: &Ctx, symbol: &str, period: &str, from: &str, to: Option<&str>, out: Option<PathBuf>, max_empty: usize, force: bool) -> anyhow::Result<()> {
    let period = Period::parse(period)?;
    let now = now_ms();
    let (from_ms, to_ms) = (
        parse_ts(from)?,
        match to {
            Some(t) => parse_ts(t)?,
            None => now,
        },
    );
    anyhow::ensure!(from_ms < to_ms, "--from must be before --to");
    anyhow::ensure!(to_ms <= now + 60_000, "--to is in the future");
    let c = CtraderCreds::from_env().context("run `atx ctrader check` first")?;
    let (mut s, acct) = login(&c).await?;
    let (name, id) = resolve_symbol(&mut s, acct, symbol).await?;
    let root = out.unwrap_or_else(|| out_dir(ctx).join("ctrader_trendbars"));
    let dir = root.join(store::safe_name(&name)).join(period.as_str());
    std::fs::create_dir_all(&dir)?;
    let t0 = Instant::now();
    let mut meta = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("_meta.jsonl"))?;
    let rep = trendbars::fetch_history(
        &mut s,
        acct,
        id,
        period,
        from_ms,
        to_ms,
        now,
        max_empty,
        |m| {
            let p = store::trendbar_path(&root, &name, period, &m.name, false);
            if force || !p.exists() { None } else { trendbar_file_info(&p) }
        },
        |m, bars, partial| {
            let known = now_ms();
            let p = store::trendbar_path(&root, &name, period, &m.name, partial);
            store::write_trendbars(&p, bars, known)?;
            if !partial {
                let stale = store::trendbar_path(&root, &name, period, &m.name, true);
                if stale.exists() {
                    std::fs::remove_file(stale)?;
                }
            }
            let line = json!({"symbol": name, "symbol_id": id, "period": period.as_str(), "month": m.name, "from_ms": m.a, "to_ms": m.b, "partial": partial,
                "n": bars.len(), "first_ts": bars.first().map(|b| b.ts_open), "last_ts": bars.last().map(|b| b.ts_open), "known_ts": known, "env": c.env.as_str()});
            writeln!(meta, "{line}")?;
            println!("{name} {} {}{}: {} bars", period.as_str(), m.name, if partial { " (partial)" } else { "" }, bars.len());
            Ok(())
        },
    )
    .await?;
    let fmt = |t: Option<i64>| t.map(|v| at_core::time::to_datetime(v).to_rfc3339());
    let summary = json!({"symbol": name, "symbol_id": id, "period": period.as_str(), "env": c.env.as_str(), "fetched_at_ms": now_ms(), "requested_from": from,
        "earliest": fmt(rep.earliest_ts), "latest": fmt(rep.latest_ts), "secs": t0.elapsed().as_secs_f64(), "report": rep});
    write_json(&dir.join("_summary.json"), &summary)?;
    println!(
        "{name} {}: {} months fetched, {} on disk, {} requests, history {} .. {} ({}) -> {}",
        period.as_str(),
        rep.months_fetched,
        rep.months_skipped,
        rep.requests,
        fmt(rep.earliest_ts).unwrap_or_else(|| "none".into()),
        fmt(rep.latest_ts).unwrap_or_else(|| "none".into()),
        rep.stopped,
        dir.display()
    );
    Ok(())
}

async fn stream(ctx: &Ctx, symbols: &str, seconds: Option<u64>) -> anyhow::Result<()> {
    let c = CtraderCreds::from_env().context("run `atx ctrader check` first")?;
    let want: Vec<String> = symbols.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect();
    let end = seconds.map(|s| Instant::now() + Duration::from_secs(s));
    let dir = out_dir(ctx).join("ctrader_spots");
    std::fs::create_dir_all(&dir)?;
    let mut backoff = 1u64;
    let (mut n_events, mut n_reconnects) = (0u64, 0u64);
    'outer: loop {
        let attempt: anyhow::Result<bool> = async {
            let (mut s, acct) = login(&c).await?;
            let mut ids = BTreeMap::new();
            for w in &want {
                let (n, id) = resolve_symbol(&mut s, acct, w).await?;
                ids.insert(id, n);
            }
            s.subscribe_spots(acct, &ids.keys().copied().collect::<Vec<_>>()).await?;
            println!("subscribed {} symbols on {}", ids.len(), c.env.as_str());
            backoff = 1;
            // spot events only carry the side that changed: keep the last bid / ask per symbol
            let mut last: BTreeMap<i64, (Option<u64>, Option<u64>)> = BTreeMap::new();
            loop {
                if end.is_some_and(|e| Instant::now() >= e) {
                    return Ok(true);
                }
                s.keepalive().await?;
                let Some(f) = s.next_event(Duration::from_secs(1)).await? else { continue };
                if f.is::<pb::ProtoOaSpotEvent>() {
                    let e = pb::ProtoOaSpotEvent::decode(f.payload.as_slice())?;
                    let l = last.entry(e.symbol_id).or_default();
                    l.0 = e.bid.or(l.0);
                    l.1 = e.ask.or(l.1);
                    let recv = now_ms();
                    let line = json!({"recv_ms": recv, "ts": e.timestamp, "symbol": ids.get(&e.symbol_id), "id": e.symbol_id,
                        "bid": l.0.map(|v| at_ctrader::price(v as i64)), "ask": l.1.map(|v| at_ctrader::price(v as i64))});
                    let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(format!("{}.jsonl", format_date(recv))))?;
                    writeln!(fh, "{line}")?;
                    n_events += 1;
                    if n_events % 1000 == 0 {
                        tracing::info!("{n_events} spot events");
                    }
                } else if f.is::<pb::ProtoOaAccountsTokenInvalidatedEvent>() {
                    anyhow::bail!("access token invalidated (expired or revoked): issue a new read-only token and update .env");
                } else if f.is::<pb::ProtoOaClientDisconnectEvent>() || f.is::<pb::ProtoOaAccountDisconnectEvent>() {
                    return Ok(false);
                } else if let Some(e) = f.error() {
                    tracing::warn!("server error event: {e}");
                }
            }
        }
        .await;
        match attempt {
            Ok(true) => break 'outer,
            Err(e) if e.to_string().contains("token invalidated") || e.to_string().contains("read-only token") => return Err(e),
            Ok(false) | Err(_) => {
                if let Err(e) = &attempt {
                    tracing::warn!("stream error: {e:#}; reconnecting in {backoff} s");
                }
                n_reconnects += 1;
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(30);
            }
        }
    }
    println!("{n_events} spot events, {n_reconnects} reconnects -> {}", dir.display());
    Ok(())
}

fn quantiles(mut v: Vec<f64>) -> serde_json::Value {
    if v.is_empty() {
        return serde_json::Value::Null;
    }
    v.sort_by(f64::total_cmp);
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    json!({"n": v.len(), "p50": q(0.5), "p90": q(0.9), "max": v[v.len() - 1], "min": v[0]})
}

async fn ping(ctx: &Ctx, n: usize, env: Option<String>) -> anyhow::Result<()> {
    let env = match env.or_else(|| at_core::config::secret("CTRADER_ENV")) {
        Some(e) => CtraderEnv::parse(&e)?,
        None => CtraderEnv::Live,
    };
    let creds = CtraderCreds::from_env().ok();
    let (mut tcp, mut tls, mut rtt) = (vec![], vec![], vec![]);
    for i in 0..n.max(1) {
        let (mut s, t) = Session::connect(env).await?;
        tcp.push(t.tcp_ms);
        tls.push(t.tls_ms);
        // request/response round trips on the first connection only (an app-auth'd session)
        if i == 0 {
            if let Some(c) = &creds {
                s.app_auth(c).await?;
            }
            for _ in 0..n {
                let t0 = Instant::now();
                let r = s.request(&pb::ProtoOaVersionReq { payload_type: None }).await;
                match r {
                    Ok(_) => rtt.push(t0.elapsed().as_secs_f64() * 1e3),
                    Err(e) => {
                        tracing::warn!("version request failed ({e}); round trips need CTRADER_CLIENT_ID / _SECRET");
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
        if i + 1 >= n.min(5) {
            break; // 5 fresh connections are enough for connect timing
        }
    }
    let res = json!({"measured_at_ms": now_ms(), "env": env.as_str(), "host": env.host(), "payload": pb::ProtoOaVersionReq::PAYLOAD,
        "tcp_connect_ms": quantiles(tcp), "tls_handshake_ms": quantiles(tls), "request_rtt_ms": quantiles(rtt)});
    println!("{}", serde_json::to_string_pretty(&res)?);
    write_json(&out_dir(ctx).join(format!("latency_ctrader_{}.json", format_date(now_ms()))), &res)?;
    Ok(())
}
