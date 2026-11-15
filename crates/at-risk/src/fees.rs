//! Per-symbol fee schedule (`config/fees.yaml`), shared by risk sizing and paper fills.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, bail, ensure};
use at_core::types::EntryType;
use serde::{Deserialize, Serialize};

/// Fees and slippage in basis points of notional.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeeRates {
    pub maker_bps: f64,
    pub taker_bps: f64,
    pub slippage_bps: f64,
}

impl FeeRates {
    /// Entry leg: maker for a resting limit, taker + slippage for a market order.
    pub fn entry_bps(&self, entry: EntryType) -> f64 {
        match entry {
            EntryType::Market => self.taker_bps + self.slippage_bps,
            EntryType::Limit => self.maker_bps,
        }
    }

    /// Exit leg: stops and targets trigger market orders (taker + slippage).
    pub fn exit_bps(&self) -> f64 {
        self.taker_bps + self.slippage_bps
    }

    /// Round-trip cost as a fraction of price.
    pub fn round_trip_frac(&self, entry: EntryType) -> f64 {
        (self.entry_bps(entry) + self.exit_bps()) / 1e4
    }

    /// Round trip of a limit entry pegged to the touch (R-EXE-2, H-EXE1): the maker fee plus the peg's adverse slip
    /// beyond the decision-time mid, then the usual taker exit.
    pub fn round_trip_frac_pegged(&self, slip_bps: f64) -> f64 {
        (self.maker_bps + slip_bps + self.exit_bps()) / 1e4
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SymbolFee {
    Class(String),
    Rates(FeeRates),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeeSchedule {
    pub default_class: String,
    pub classes: BTreeMap<String, FeeRates>,
    #[serde(default)]
    pub symbols: BTreeMap<String, SymbolFee>,
}

impl FeeSchedule {
    pub fn from_yaml(text: &str) -> anyhow::Result<Self> {
        let f: FeeSchedule = serde_yaml::from_str(text).context("fee schedule")?;
        f.validate()?;
        Ok(f)
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        Self::from_yaml(&std::fs::read_to_string(path).with_context(|| path.display().to_string())?).with_context(|| path.display().to_string())
    }

    fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.classes.contains_key(&self.default_class), "default_class {:?} is not in classes", self.default_class);
        let ok = |r: &FeeRates| [r.maker_bps, r.taker_bps, r.slippage_bps].iter().all(|x| x.is_finite() && *x >= 0.0);
        for (k, r) in &self.classes {
            ensure!(ok(r), "class {k}: fees must be finite and >= 0 (maker rebates are not modelled)");
        }
        for (s, f) in &self.symbols {
            match f {
                SymbolFee::Class(c) if !self.classes.contains_key(c) => bail!("symbol {s}: unknown class {c:?}"),
                SymbolFee::Rates(r) if !ok(r) => bail!("symbol {s}: fees must be finite and >= 0"),
                _ => {}
            }
        }
        Ok(())
    }

    /// Class name for `symbol`: explicit override, else the hint (instrument metadata or the intent's
    /// asset_class) if it names a known class, else the default class.
    pub fn class_of(&self, symbol: &str, hint: Option<&str>) -> String {
        match self.symbols.get(symbol) {
            Some(SymbolFee::Class(c)) => c.clone(),
            Some(SymbolFee::Rates(_)) => format!("symbol:{symbol}"),
            None => hint.map(|h| h.trim().to_ascii_lowercase()).filter(|h| self.classes.contains_key(h)).unwrap_or_else(|| self.default_class.clone()),
        }
    }

    pub fn rates(&self, symbol: &str, hint: Option<&str>) -> FeeRates {
        if let Some(SymbolFee::Rates(r)) = self.symbols.get(symbol) {
            return *r;
        }
        self.classes[&self.class_of(symbol, hint)]
    }

    /// Record a class learned from venue metadata, unless the file already overrides the symbol.
    pub fn set_class_if_absent(&mut self, symbol: &str, class: &str) {
        if self.classes.contains_key(class) {
            self.symbols.entry(symbol.to_string()).or_insert_with(|| SymbolFee::Class(class.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_fee_file_loads_with_the_agreed_schedule() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f = FeeSchedule::load(&root.join("config/fees.yaml")).unwrap();
        let c = f.rates("BTCUSDT", Some("crypto"));
        assert_eq!((c.maker_bps, c.taker_bps), (2.0, 5.5));
        let t = f.rates("NVDAUSDT", Some("tradfi"));
        assert_eq!((t.maker_bps, t.taker_bps), (0.0, 2.75));
        assert_eq!(f.class_of("XYZUSDT", Some("unknown")), "crypto", "unknown classes fall back to the default");
    }

    #[test]
    fn overrides_and_round_trip() {
        let mut f = FeeSchedule::from_yaml(
            "default_class: crypto\nclasses:\n  crypto: {maker_bps: 2, taker_bps: 5.5, slippage_bps: 2}\n  tradfi: {maker_bps: 0, taker_bps: 2.75, slippage_bps: 2}\nsymbols:\n  XAUUSDT: tradfi\n  ODDUSDT: {maker_bps: 1, taker_bps: 1, slippage_bps: 1}\n",
        )
        .unwrap();
        assert_eq!(f.class_of("XAUUSDT", Some("crypto")), "tradfi", "explicit override wins over the hint");
        assert_eq!(f.rates("ODDUSDT", None).taker_bps, 1.0);
        f.set_class_if_absent("AAPLUSDT", "tradfi");
        f.set_class_if_absent("XAUUSDT", "crypto");
        assert_eq!(f.class_of("AAPLUSDT", None), "tradfi");
        assert_eq!(f.class_of("XAUUSDT", None), "tradfi");
        let c = f.rates("BTCUSDT", None);
        assert!((c.round_trip_frac(EntryType::Market) - 15e-4).abs() < 1e-12);
        assert!((c.round_trip_frac(EntryType::Limit) - 9.5e-4).abs() < 1e-12);
        assert!(FeeSchedule::from_yaml("default_class: x\nclasses: {}\n").is_err());
        assert!(FeeSchedule::from_yaml("default_class: c\nclasses:\n  c: {maker_bps: -1, taker_bps: 1, slippage_bps: 1}\n").is_err());
    }
}
