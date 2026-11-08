//! `atx`: AI_Trader command line.
//!
//!   atx check                                   connectivity to every configured endpoint
//!   atx universe                                current tradable universe (live filters)
//!   atx instruments                             snapshot instrument rules into the lake
//!   atx klines  --symbols BTCUSDT --freq 1h     bars (last/mark/index/premium)
//!   atx funding --symbols BTCUSDT               realised funding
//!   atx oi      --symbols BTCUSDT               open interest
//!   atx lsr     --symbols BTCUSDT               account long/short ratio
//!   atx trades  --symbol BTCUSDT --from D --to D   archive trades -> flow bars
//!   atx alt fng|dvol|stablecoins|fred           alternative data
//!   atx account                                 equity and positions (needs keys)
//!   atx stream  --symbols BTCUSDT --seconds 30  live websocket smoke test
//!   atx order-roundtrip --symbol BTCUSDT --yes  demo/testnet only: open + close a minimum-size order
//!   atx collect [--minutes N] [--category linear] [--symbols A,B]   live order flow -> flow_3s
//!   atx update [--once | --loop]                live bars/funding/OI/DVOL/F&G -> lake (config: live)
//!   atx propose --file P.json [--submit]        proposals -> risk engine -> (demo/testnet) bracket orders
//!   atx positions [--manage [--breakeven]]      open positions in R, open orders; optional breakeven rule
//!   atx flatten --symbol SYM --yes              cancel orders + close the position (mainnet: two-key guard)
//!   atx mm --until RFC3339 [--dry-run] [--yes]  weekend skewed passive quoting (config/mm.yaml, H-MM1)

mod cmd;
mod exec;
mod mm;
mod updater;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "atx", version, about = "AI_Trader core: connectivity, ingestion, execution")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args, Clone)]
pub struct Range {
    /// Start (YYYY-MM-DD or RFC 3339). Default: data.history_start.
    #[arg(long)]
    start: Option<String>,
    /// End, exclusive. Default: now.
    #[arg(long)]
    end: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Check connectivity to all configured endpoints and credentials.
    Check,
    /// Print the current universe (live filters; NOT point-in-time, don't use for history).
    Universe,
    /// Snapshot instrument metadata into the lake.
    Instruments,
    Klines {
        /// Comma-separated symbols, or "universe".
        #[arg(long)]
        symbols: String,
        #[arg(long, default_value = "1h")]
        freq: String,
        #[arg(long, default_value = "last")]
        price_type: String,
        #[command(flatten)]
        range: Range,
    },
    Funding {
        #[arg(long)]
        symbols: String,
        #[command(flatten)]
        range: Range,
    },
    Oi {
        #[arg(long)]
        symbols: String,
        #[arg(long)]
        interval: Option<String>,
        #[command(flatten)]
        range: Range,
    },
    Lsr {
        #[arg(long)]
        symbols: String,
        #[arg(long)]
        period: Option<String>,
        #[command(flatten)]
        range: Range,
    },
    /// Download daily trade archives and aggregate into flow bars.
    Trades {
        #[arg(long)]
        symbol: String,
        /// First day (YYYY-MM-DD).
        #[arg(long)]
        from: String,
        /// Last day, inclusive (YYYY-MM-DD).
        #[arg(long)]
        to: String,
        /// Comma-separated flow bar frequencies. Default: data.flow_freqs.
        #[arg(long)]
        freqs: Option<String>,
        /// Also store raw trades (rolling retention table).
        #[arg(long)]
        raw: bool,
        /// Keep downloaded .csv.gz files in paths.raw_cache.
        #[arg(long)]
        cache: bool,
    },
    Alt {
        #[command(subcommand)]
        source: AltCmd,
    },
    /// Show equity and open positions for the configured environment.
    Account,
    /// Subscribe to public streams and report message counts.
    Stream {
        #[arg(long, default_value = "BTCUSDT")]
        symbols: String,
        /// Comma-separated topic prefixes: kline.1, publicTrade, tickers, orderbook.1, allLiquidation
        #[arg(long, default_value = "kline.1,publicTrade,tickers,allLiquidation")]
        topics: String,
        #[arg(long, default_value_t = 30)]
        seconds: u64,
    },
    /// Live order-flow collector: trades + L2 books -> 3-second candles in flow_3s (config: flow.streams).
    Collect {
        /// Stop after N minutes (default: run until Ctrl-C).
        #[arg(long)]
        minutes: Option<u64>,
        /// Only run streams of this category (linear | spot | inverse).
        #[arg(long)]
        category: Option<String>,
        /// Override the symbol list of every selected stream.
        #[arg(long)]
        symbols: Option<String>,
        /// Override the order-book depth of every selected stream.
        #[arg(long)]
        depth: Option<u32>,
    },
    /// Live market-data updater: keeps bars, funding, OI, DVOL and Fear & Greed current (config: live).
    Update {
        /// Run a single cycle and exit.
        #[arg(long, conflicts_with = "run_loop")]
        once: bool,
        /// Run until Ctrl-C (the default).
        #[arg(long = "loop", id = "run_loop")]
        run_loop: bool,
    },
    /// Demo/testnet only: open a minimum-size market order with SL/TP, then close it.
    OrderRoundtrip {
        #[arg(long, default_value = "BTCUSDT")]
        symbol: String,
        /// Required confirmation.
        #[arg(long)]
        yes: bool,
    },
    /// Size ranked trade proposals with the risk engine; with --submit place them (demo/testnet only).
    Propose {
        /// JSON file: {"session_id", "proposals": [TradeProposal...], "clusters"?: {symbol: cluster}} (docs/EXECUTION.md).
        #[arg(long)]
        file: std::path::PathBuf,
        /// Place the accepted orders. Refused on mainnet.
        #[arg(long)]
        submit: bool,
        /// Overrides the file's session_id (order link ids are "<session>-<proposal id>").
        #[arg(long)]
        session_id: Option<String>,
    },
    /// Open positions (entry, mark, unrealised R, stop/TP) and open orders.
    Positions {
        /// Apply the enabled deterministic exit rules (never closes on time).
        #[arg(long)]
        manage: bool,
        /// Rule: move the stop to entry once unrealised >= 1R.
        #[arg(long, requires = "manage")]
        breakeven: bool,
        /// Session id for journal lines (default "atx-manage").
        #[arg(long)]
        session_id: Option<String>,
    },
    /// Cancel all open orders for a symbol and close its position (journaled as manual_action). Mainnet: two-key guard.
    Flatten {
        #[arg(long)]
        symbol: String,
        /// Required confirmation.
        #[arg(long)]
        yes: bool,
        /// Session id for the journal line (default "atx-manual").
        #[arg(long)]
        session_id: Option<String>,
    },
    /// Skewed passive quoting until a deadline, then cancel everything and flatten (config/mm.yaml, H-MM1).
    Mm {
        /// Stop time (RFC 3339, e.g. 2026-11-08T12:00:00Z).
        #[arg(long)]
        until: String,
        /// Compute and journal quotes without sending anything.
        #[arg(long)]
        dry_run: bool,
        /// Required on mainnet.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum AltCmd {
    /// Crypto Fear & Greed index (full history).
    Fng,
    /// Deribit DVOL implied-vol index.
    Dvol {
        #[arg(long, default_value = "BTC,ETH")]
        currencies: String,
        #[arg(long, default_value = "3600")]
        resolution: String,
        #[command(flatten)]
        range: Range,
    },
    /// DefiLlama total stablecoin supply.
    Stablecoins,
    /// FRED series with all ALFRED vintages (needs FRED_API_KEY). Default: configured list.
    Fred {
        #[arg(long)]
        series: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_target(false)
        .init();
    let cli = Cli::parse();
    let ctx = cmd::Ctx::load()?;
    match cli.command {
        Command::Check => cmd::check(&ctx).await,
        Command::Universe => cmd::universe(&ctx).await.map(|u| {
            println!("{} symbols: {}", u.len(), u.join(","));
        }),
        Command::Instruments => cmd::instruments(&ctx).await,
        Command::Klines { symbols, freq, price_type, range } => cmd::klines(&ctx, &symbols, &freq, &price_type, &range).await,
        Command::Funding { symbols, range } => cmd::funding(&ctx, &symbols, &range).await,
        Command::Oi { symbols, interval, range } => cmd::open_interest(&ctx, &symbols, interval, &range).await,
        Command::Lsr { symbols, period, range } => cmd::ls_ratio(&ctx, &symbols, period, &range).await,
        Command::Trades { symbol, from, to, freqs, raw, cache } => cmd::trades(&ctx, &symbol, &from, &to, freqs, raw, cache).await,
        Command::Alt { source } => match source {
            AltCmd::Fng => cmd::alt_fng(&ctx).await,
            AltCmd::Dvol { currencies, resolution, range } => cmd::alt_dvol(&ctx, &currencies, &resolution, &range).await,
            AltCmd::Stablecoins => cmd::alt_stablecoins(&ctx).await,
            AltCmd::Fred { series } => cmd::alt_fred(&ctx, series).await,
        },
        Command::Account => cmd::account(&ctx).await,
        Command::Stream { symbols, topics, seconds } => cmd::stream(&ctx, &symbols, &topics, seconds).await,
        Command::OrderRoundtrip { symbol, yes } => cmd::order_roundtrip(&ctx, &symbol, yes).await,
        Command::Propose { file, submit, session_id } => exec::propose(&ctx, &file, submit, session_id).await,
        Command::Positions { manage, breakeven, session_id } => exec::positions(&ctx, manage, breakeven, session_id).await,
        Command::Flatten { symbol, yes, session_id } => exec::flatten(&ctx, &symbol, yes, session_id).await,
        Command::Mm { until, dry_run, yes } => mm::run(&ctx, &until, dry_run, yes).await,
        Command::Collect { minutes, category, symbols, depth } => cmd::collect(ctx, minutes, category, symbols, depth).await,
        Command::Update { once, run_loop: _ } => cmd::update(&ctx, once).await,
    }
}
