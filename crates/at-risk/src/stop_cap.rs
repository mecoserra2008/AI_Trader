//! Stop-distance cap in risk-sigma units (H-STOP1, user decision 2026-11-14).
//!
//! The cap starts as a PRIOR (`placement.max_stop_distance_sigma: 4.0` in `config/risk.yaml`) and is updated weekly
//! (Monday 00:00 UTC) from closed paper trades by the pinned conjugate rule in `ai_trader.risk.stop_posterior`, which
//! writes `data/models/risk/stop_cap.json`. This module only READS that file:
//! - no file: the yaml value is the cap (the prior);
//! - a file: its `applied_cap_sigma` is the cap, after checking it is finite and inside the hard bounds
//!   `placement.stop_cap_bounds_sigma` from `config/risk.yaml`. A file that fails any check is an error (fail closed):
//!   the engine never trades on a cap it cannot verify, and nothing here falls back silently.
//!
//! The LLM never sets the cap: it is not an intent field, and the only writer is the deterministic weekly update.

use std::path::Path;

use anyhow::{Context, bail, ensure};
use serde::Deserialize;

/// Default location under the project root (`paths.model_states` = data/models).
pub const STOP_CAP_FILE: &str = "data/models/risk/stop_cap.json";

/// The fields of `stop_cap.json` the engine relies on; everything else in the file is ignored here.
#[derive(Clone, Debug, Deserialize)]
pub struct StopCapFile {
    pub applied_cap_sigma: f64,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub n: Option<u64>,
    #[serde(default)]
    pub effective_from: Option<String>,
}

/// The cap the engine enforces and where it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct StopCap {
    pub cap_sigma: f64,
    pub source: String,
}

/// Check the hard bounds themselves: two positive finite numbers, lo < hi.
pub fn check_bounds(bounds: [f64; 2]) -> anyhow::Result<()> {
    let [lo, hi] = bounds;
    ensure!(lo.is_finite() && hi.is_finite() && lo > 0.0 && lo < hi, "placement.stop_cap_bounds_sigma {bounds:?}: need 0 < lo < hi");
    Ok(())
}

/// Resolve the cap from the yaml fallback, the yaml hard bounds and the (optional) contents of `stop_cap.json`.
pub fn resolve(yaml_cap: f64, bounds: Option<[f64; 2]>, file_text: Option<&str>) -> anyhow::Result<StopCap> {
    ensure!(yaml_cap.is_finite() && yaml_cap > 0.0, "placement.max_stop_distance_sigma {yaml_cap} must be a positive number");
    if let Some(b) = bounds {
        check_bounds(b)?;
        ensure!(b[0] <= yaml_cap && yaml_cap <= b[1], "placement.max_stop_distance_sigma {yaml_cap} lies outside stop_cap_bounds_sigma {b:?}");
    }
    let Some(text) = file_text else {
        return Ok(StopCap { cap_sigma: yaml_cap, source: format!("risk.yaml placement.max_stop_distance_sigma (prior, no {STOP_CAP_FILE})") });
    };
    let Some([lo, hi]) = bounds else {
        bail!("{STOP_CAP_FILE} exists but config/risk.yaml has no placement.stop_cap_bounds_sigma to validate it against");
    };
    let f: StopCapFile = serde_json::from_str(text).with_context(|| format!("{STOP_CAP_FILE}: not a stop-cap file (needs applied_cap_sigma)"))?;
    let c = f.applied_cap_sigma;
    ensure!(c.is_finite(), "{STOP_CAP_FILE}: applied_cap_sigma is not finite");
    ensure!((lo..=hi).contains(&c), "{STOP_CAP_FILE}: applied_cap_sigma {c} outside the hard bounds [{lo}, {hi}] (config/risk.yaml)");
    let v = f.version.as_deref().unwrap_or("?");
    let n = f.n.map(|n| n.to_string()).unwrap_or_else(|| "?".into());
    let from = f.effective_from.as_deref().unwrap_or("?");
    Ok(StopCap { cap_sigma: c, source: format!("{STOP_CAP_FILE} (version {v}, n {n}, effective {from}; bounds [{lo}, {hi}], yaml fallback {yaml_cap})") })
}

/// Read `path` if it exists and resolve the cap (see [`resolve`]).
pub fn load(path: &Path, yaml_cap: f64, bounds: Option<[f64; 2]>) -> anyhow::Result<StopCap> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => Some(t),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e).with_context(|| path.display().to_string()),
    };
    resolve(yaml_cap, bounds, text.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Option<[f64; 2]> = Some([2.5, 6.0]);

    #[test]
    fn no_file_falls_back_to_the_yaml_prior() {
        let c = resolve(4.0, B, None).unwrap();
        assert_eq!(c.cap_sigma, 4.0);
        assert!(c.source.contains("prior"));
        // bounds are optional while there is no file
        assert_eq!(resolve(4.0, None, None).unwrap().cap_sigma, 4.0);
    }

    #[test]
    fn file_inside_the_bounds_is_used() {
        let c = resolve(4.0, B, Some(r#"{"applied_cap_sigma": 3.4, "version": "stop_posterior.v1", "n": 12, "extra": [1, 2]}"#)).unwrap();
        assert_eq!(c.cap_sigma, 3.4);
        assert!(c.source.contains("n 12") && c.source.contains("stop_posterior.v1"));
        // the bounds themselves are allowed
        assert_eq!(resolve(4.0, B, Some(r#"{"applied_cap_sigma": 2.5}"#)).unwrap().cap_sigma, 2.5);
        assert_eq!(resolve(4.0, B, Some(r#"{"applied_cap_sigma": 6.0}"#)).unwrap().cap_sigma, 6.0);
    }

    #[test]
    fn file_outside_the_bounds_or_corrupt_is_refused() {
        for bad in
            [r#"{"applied_cap_sigma": 2.49}"#, r#"{"applied_cap_sigma": 6.01}"#, r#"{"applied_cap_sigma": 40}"#, r#"{"cap": 4.0}"#, "not json", r#"{"applied_cap_sigma": null}"#]
        {
            assert!(resolve(4.0, B, Some(bad)).is_err(), "{bad} must be refused");
        }
        // a file without yaml bounds cannot be validated
        assert!(resolve(4.0, None, Some(r#"{"applied_cap_sigma": 4.0}"#)).is_err());
    }

    #[test]
    fn bad_bounds_or_yaml_value_are_refused() {
        assert!(resolve(4.0, Some([6.0, 2.5]), None).is_err());
        assert!(resolve(4.0, Some([0.0, 6.0]), None).is_err());
        assert!(resolve(7.0, B, None).is_err(), "the yaml prior must itself lie inside the bounds");
        assert!(resolve(f64::NAN, B, None).is_err());
    }

    #[test]
    fn load_reads_the_file_when_present() {
        let dir = std::env::temp_dir().join(format!("at_risk_stop_cap_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("stop_cap.json");
        let _ = std::fs::remove_file(&p);
        assert_eq!(load(&p, 4.0, B).unwrap().cap_sigma, 4.0);
        std::fs::write(&p, r#"{"applied_cap_sigma": 5.25}"#).unwrap();
        assert_eq!(load(&p, 4.0, B).unwrap().cap_sigma, 5.25);
        std::fs::write(&p, r#"{"applied_cap_sigma": 9.0}"#).unwrap();
        assert!(load(&p, 4.0, B).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
