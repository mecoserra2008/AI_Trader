//! Loads `config/settings.yaml` + `config/risk.yaml` and secrets from `.env`.
//! Only the fields the Rust side uses are typed; unknown keys are ignored.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Environment {
    Demo,
    Testnet,
    Mainnet,
}

impl Environment {
    pub fn as_str(self) -> &'static str {
        match self {
            Environment::Demo => "demo",
            Environment::Testnet => "testnet",
            Environment::Mainnet => "mainnet",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "demo" => Ok(Environment::Demo),
            "testnet" => Ok(Environment::Testnet),
            "mainnet" => Ok(Environment::Mainnet),
            other => Err(Error::Config(format!("unknown environment {other:?}"))),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    pub environment: Environment,
    pub execution: ExecutionCfg,
    pub paths: Paths,
    pub venues: Venues,
    pub universe: UniverseCfg,
    pub data: DataCfg,
    #[serde(skip)]
    pub root: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExecutionCfg {
    pub allow_mainnet: bool,
    pub stop_trigger_by: String,
    pub tp_trigger_by: String,
    pub default_leverage: f64,
    pub entry_order_type: String,
    pub limit_time_in_force: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Paths {
    pub data_root: PathBuf,
    pub lake: PathBuf,
    pub journal: PathBuf,
    pub model_states: PathBuf,
    pub raw_cache: PathBuf,
    pub logs: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Venues {
    pub bybit: BybitCfg,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BybitCfg {
    pub category: String,
    pub settle_coin: String,
    pub recv_window_ms: u64,
    pub rest_rate_limit_per_s: f64,
    pub market_data_env: Environment,
    pub public_trades_archive: String,
    pub endpoints: BTreeMap<Environment, BybitEndpoints>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BybitEndpoints {
    pub rest: String,
    pub ws_public: String,
    pub ws_private: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UniverseCfg {
    pub category: String,
    pub quote: String,
    pub contract_type: String,
    pub symbol_types: Vec<String>,
    pub min_24h_turnover_usd: f64,
    pub min_listing_age_days: i64,
    pub max_symbols: usize,
    pub always_include: Vec<String>,
    pub exclude: Vec<String>,
    pub market_proxy: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct DataCfg {
    pub history_start: String,
    pub bar_freqs: Vec<String>,
    pub price_types: Vec<String>,
    pub flow_freqs: Vec<String>,
    pub open_interest_interval: String,
    pub ls_ratio_period: String,
    pub known_ts_latency_ms: BTreeMap<String, i64>,
}

impl DataCfg {
    pub fn latency(&self, table: &str) -> i64 {
        self.known_ts_latency_ms.get(table).copied().unwrap_or(0)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct RiskConfig {
    pub risk_per_trade_pct: f64,
    pub max_new_trades_per_day: u32,
    pub max_open_positions: u32,
    pub max_positions_per_cluster: u32,
    pub daily_loss_limit_pct: f64,
    pub max_gross_exposure_x: f64,
    pub max_stop_distance_pct: f64,
    pub min_net_expected_r: f64,
    pub costs: Costs,
    pub slot_replacement_hurdle_r: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Costs {
    pub taker_bps: f64,
    pub maker_bps: f64,
    pub slippage_bps: f64,
}

impl Settings {
    pub fn bybit_endpoints(&self, env: Environment) -> Result<&BybitEndpoints> {
        self.venues
            .bybit
            .endpoints
            .get(&env)
            .ok_or_else(|| Error::Config(format!("no bybit endpoints for {}", env.as_str())))
    }

    pub fn abs(&self, p: &Path) -> PathBuf {
        if p.is_absolute() { p.to_path_buf() } else { self.root.join(p) }
    }

    pub fn lake_dir(&self) -> PathBuf {
        self.abs(&self.paths.lake)
    }

    pub fn schema_path(&self) -> PathBuf {
        self.root.join("config").join("schema.yaml")
    }

    /// Two-key guard for real money: config flag AND environment variable.
    pub fn ensure_execution_allowed(&self) -> Result<()> {
        if self.environment == Environment::Mainnet {
            let env_ok = std::env::var("AI_TRADER_ALLOW_MAINNET").map(|v| v == "1").unwrap_or(false);
            if !(self.execution.allow_mainnet && env_ok) {
                return Err(Error::Config(
                    "mainnet execution blocked: needs execution.allow_mainnet=true AND AI_TRADER_ALLOW_MAINNET=1".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Project root: `AI_TRADER_ROOT`, else the first ancestor of the cwd containing `config/settings.yaml`.
pub fn find_root() -> Result<PathBuf> {
    if let Ok(r) = std::env::var("AI_TRADER_ROOT") {
        return Ok(PathBuf::from(r));
    }
    let cwd = std::env::current_dir().map_err(|e| Error::Config(e.to_string()))?;
    cwd.ancestors()
        .find(|d| d.join("config").join("settings.yaml").is_file())
        .map(Path::to_path_buf)
        .ok_or_else(|| Error::Config("config/settings.yaml not found in any parent directory".into()))
}

fn read_yaml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
    serde_yaml::from_str(&text).map_err(|e| Error::Config(format!("{}: {e}", path.display())))
}

/// Load settings from `<root>/config`, applying `.env` and the `AI_TRADER_ENVIRONMENT` override.
pub fn load(root: &Path) -> Result<(Settings, RiskConfig)> {
    let _ = dotenvy::from_path(root.join(".env"));
    let mut settings: Settings = read_yaml(&root.join("config").join("settings.yaml"))?;
    settings.root = root.to_path_buf();
    if let Ok(env) = std::env::var("AI_TRADER_ENVIRONMENT") {
        settings.environment = Environment::parse(&env)?;
    }
    let risk: RiskConfig = read_yaml(&root.join("config").join("risk.yaml"))?;
    Ok((settings, risk))
}

pub fn load_default() -> Result<(Settings, RiskConfig)> {
    load(&find_root()?)
}

pub fn secret(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// `(api_key, api_secret)` for a Bybit environment, from `BYBIT_<ENV>_API_KEY/_SECRET`.
pub fn bybit_credentials(env: Environment) -> Option<(String, String)> {
    let p = env.as_str().to_ascii_uppercase();
    Some((secret(&format!("BYBIT_{p}_API_KEY"))?, secret(&format!("BYBIT_{p}_API_SECRET"))?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_project_config() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (s, r) = load(&root).expect("config loads");
        assert!(s.bybit_endpoints(Environment::Demo).is_ok());
        assert!(r.risk_per_trade_pct > 0.0 && r.risk_per_trade_pct <= 2.0);
        assert!(!s.execution.allow_mainnet, "mainnet must be off by default");
    }
}
