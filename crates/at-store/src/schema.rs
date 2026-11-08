use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, bail};
use serde::Deserialize;

/// Partition names computed from `event_ts` when they are not regular columns.
pub const DERIVED_PARTITIONS: [&str; 3] = ["year", "month", "date"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColType {
    Utf8,
    F64,
    /// Compact float for features (7 significant digits): halves storage vs f64.
    F32,
    I64,
    I32,
    Bool,
    TsMs,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ColumnSpec {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: ColType,
    #[serde(default)]
    pub nullable: bool,
    #[serde(default)]
    pub doc: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TableSpec {
    #[serde(default)]
    pub doc: String,
    pub partition_by: Vec<String>,
    pub primary_key: Vec<String>,
    pub columns: Vec<ColumnSpec>,
}

impl TableSpec {
    pub fn column(&self, name: &str) -> Option<&ColumnSpec> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn is_partition(&self, name: &str) -> bool {
        self.partition_by.iter().any(|p| p == name)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Schema {
    pub version: u32,
    pub tables: BTreeMap<String, TableSpec>,
}

impl Schema {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let schema: Schema = serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        schema.validate()?;
        Ok(schema)
    }

    /// Structural invariants every table must satisfy.
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, t) in &self.tables {
            for required in ["event_ts", "known_ts"] {
                match t.column(required) {
                    Some(c) if c.ty == ColType::TsMs && !c.nullable => {}
                    _ => bail!("table {name}: {required} must be a non-nullable ts_ms column"),
                }
            }
            for k in &t.primary_key {
                if t.column(k).is_none() && !t.is_partition(k) {
                    bail!("table {name}: primary key column {k} not defined");
                }
            }
            for p in &t.partition_by {
                if t.column(p).is_none() && !DERIVED_PARTITIONS.contains(&p.as_str()) {
                    bail!("table {name}: partition {p} is neither a column nor derived (year/month/date)");
                }
            }
        }
        Ok(())
    }

    pub fn table(&self, name: &str) -> anyhow::Result<&TableSpec> {
        self.tables.get(name).with_context(|| format!("unknown lake table {name:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_schema_is_valid() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/schema.yaml");
        let s = Schema::load(&p).unwrap();
        assert!(s.tables.contains_key("bars"));
        assert!(s.table("bars").unwrap().is_partition("freq"));
    }
}
