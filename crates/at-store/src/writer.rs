use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, bail};
use arrow::array::{ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, TimestampMillisecondArray, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use arrow::record_batch::RecordBatch;
use at_core::time::{format_date, now_ms, to_datetime};
use chrono::Datelike;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::batch::{Batch, Col};
use crate::schema::{ColType, ColumnSpec, Schema, TableSpec};

#[derive(Debug, Default)]
pub struct WriteReport {
    pub rows: usize,
    pub files: Vec<PathBuf>,
}

pub struct Lake {
    root: PathBuf,
    schema: Schema,
}

impl Lake {
    pub fn open(root: impl Into<PathBuf>, schema_path: &Path) -> anyhow::Result<Self> {
        Ok(Self { root: root.into(), schema: Schema::load(schema_path)? })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Validate `batch` against the table spec, split it by partition and write one Parquet file
    /// per partition. Appends only: revisions are new rows with a later `known_ts`.
    pub fn write(&self, table: &str, batch: &Batch) -> anyhow::Result<WriteReport> {
        let spec = self.schema.table(table)?;
        let n = batch.len();
        if n == 0 {
            return Ok(WriteReport::default());
        }
        let cols = validate(table, spec, batch, n)?;
        let groups = partition_groups(spec, &cols, n)?;

        let file_specs: Vec<&ColumnSpec> = spec.columns.iter().filter(|c| !spec.is_partition(&c.name)).collect();
        let arrow_schema = Arc::new(ArrowSchema::new(
            file_specs.iter().map(|c| Field::new(&c.name, arrow_type(c.ty), c.nullable)).collect::<Vec<_>>(),
        ));
        let arrays: Vec<ArrayRef> = file_specs.iter().map(|c| to_array(&cols[&c.name])).collect();
        let props = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();

        let mut report = WriteReport { rows: n, files: Vec::new() };
        for (key, idx) in groups {
            let mut dir = self.root.join(table);
            for (p, v) in spec.partition_by.iter().zip(&key) {
                dir.push(format!("{p}={v}"));
            }
            std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

            let indices = UInt32Array::from(idx);
            let taken = arrays.iter().map(|a| take(a.as_ref(), &indices, None)).collect::<Result<Vec<_>, _>>()?;
            let rb = RecordBatch::try_new(arrow_schema.clone(), taken)?;

            let id = uuid::Uuid::new_v4().simple().to_string();
            let final_path = dir.join(format!("part-{}-{}.parquet", now_ms(), &id[..8]));
            let tmp_path = final_path.with_extension("parquet.tmp");
            {
                let file = File::create(&tmp_path)?;
                let mut w = ArrowWriter::try_new(file, arrow_schema.clone(), Some(props.clone()))?;
                w.write(&rb)?;
                w.close()?;
            }
            std::fs::rename(&tmp_path, &final_path)?;
            report.files.push(final_path);
        }
        Ok(report)
    }
}

/// Check every column against the spec; returns the full column set (missing nullable columns filled with nulls).
fn validate<'a>(table: &str, spec: &TableSpec, batch: &'a Batch, n: usize) -> anyhow::Result<BTreeMap<String, std::borrow::Cow<'a, Col>>> {
    use std::borrow::Cow;
    for name in batch.cols.keys() {
        if spec.column(name).is_none() {
            bail!("{table}: column {name:?} is not in the schema");
        }
    }
    let mut out = BTreeMap::new();
    for c in &spec.columns {
        let col = match batch.col(&c.name) {
            Some(col) => Cow::Borrowed(col),
            None if c.nullable => Cow::Owned(null_col(c.ty, n)),
            None => bail!("{table}: required column {:?} missing", c.name),
        };
        if col.len() != n {
            bail!("{table}: column {:?} has {} rows, expected {n}", c.name, col.len());
        }
        if col.ty() != c.ty {
            bail!("{table}: column {:?} is {:?}, schema says {:?}", c.name, col.ty(), c.ty);
        }
        if !c.nullable && col.null_count() > 0 {
            bail!("{table}: non-nullable column {:?} contains nulls", c.name);
        }
        out.insert(c.name.clone(), col);
    }
    // known_ts must never precede event_ts: knowing a value before it happened is look-ahead by construction.
    if let (Col::TsMs(ev), Col::TsMs(kn)) = (&*out["event_ts"], &*out["known_ts"])
        && let Some(i) = ev.iter().zip(kn).position(|(e, k)| k < e)
    {
        bail!("{table}: row {i} has known_ts < event_ts");
    }
    Ok(out)
}

fn partition_groups(spec: &TableSpec, cols: &BTreeMap<String, std::borrow::Cow<'_, Col>>, n: usize) -> anyhow::Result<BTreeMap<Vec<String>, Vec<u32>>> {
    let Col::TsMs(event_ts) = &*cols["event_ts"] else { unreachable!("validated") };
    let mut groups: BTreeMap<Vec<String>, Vec<u32>> = BTreeMap::new();
    for (i, ev) in event_ts.iter().enumerate().take(n) {
        let ev = ev.expect("event_ts is non-nullable (validated)");
        let mut key = Vec::with_capacity(spec.partition_by.len());
        for p in &spec.partition_by {
            let v = match cols.get(p.as_str()) {
                Some(c) => c.value_string(i).with_context(|| format!("partition column {p} is null at row {i}"))?,
                None => {
                    let dt = to_datetime(ev);
                    match p.as_str() {
                        "year" => dt.year().to_string(),
                        "month" => format!("{:02}", dt.month()),
                        "date" => format_date(ev),
                        other => bail!("unknown derived partition {other}"),
                    }
                }
            };
            if v.contains(['/', '\\', '=']) {
                bail!("partition value {v:?} for {p} contains a path character");
            }
            key.push(v);
        }
        groups.entry(key).or_default().push(i as u32);
    }
    Ok(groups)
}

fn arrow_type(t: ColType) -> DataType {
    match t {
        ColType::Utf8 => DataType::Utf8,
        ColType::F64 => DataType::Float64,
        ColType::F32 => DataType::Float32,
        ColType::I64 => DataType::Int64,
        ColType::I32 => DataType::Int32,
        ColType::Bool => DataType::Boolean,
        ColType::TsMs => DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
    }
}

fn to_array(c: &Col) -> ArrayRef {
    match c {
        Col::Utf8(v) => Arc::new(StringArray::from(v.clone())),
        Col::F64(v) => Arc::new(Float64Array::from(v.clone())),
        Col::F32(v) => Arc::new(Float32Array::from(v.clone())),
        Col::I64(v) => Arc::new(Int64Array::from(v.clone())),
        Col::I32(v) => Arc::new(Int32Array::from(v.clone())),
        Col::Bool(v) => Arc::new(BooleanArray::from(v.clone())),
        Col::TsMs(v) => Arc::new(TimestampMillisecondArray::from(v.clone()).with_timezone("UTC")),
    }
}

fn null_col(t: ColType, n: usize) -> Col {
    match t {
        ColType::Utf8 => Col::Utf8(vec![None; n]),
        ColType::F64 => Col::F64(vec![None; n]),
        ColType::F32 => Col::F32(vec![None; n]),
        ColType::I64 => Col::I64(vec![None; n]),
        ColType::I32 => Col::I32(vec![None; n]),
        ColType::Bool => Col::Bool(vec![None; n]),
        ColType::TsMs => Col::TsMs(vec![None; n]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::file::reader::{FileReader, SerializedFileReader};

    fn lake() -> (Lake, PathBuf) {
        let root = std::env::temp_dir().join(format!("at-store-test-{}", uuid::Uuid::new_v4().simple()));
        let schema = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config/schema.yaml");
        (Lake::open(&root, &schema).unwrap(), root)
    }

    #[test]
    fn writes_partitioned_funding() {
        let (lake, root) = lake();
        let t = vec![1_735_689_600_000, 1_767_225_600_000]; // 2025-01-01, 2026-01-01
        let b = Batch::new()
            .utf8_const("venue", "bybit", 2)
            .utf8_const("symbol", "BTCUSDT", 2)
            .ts("event_ts", t.clone())
            .f64("rate", vec![0.0001, -0.0002])
            .ts("known_ts", t);
        let rep = lake.write("funding", &b).unwrap();
        assert_eq!(rep.rows, 2);
        assert_eq!(rep.files.len(), 2, "one file per year partition");
        assert!(rep.files.iter().any(|p| p.to_string_lossy().contains("year=2026")));
        let r = SerializedFileReader::new(File::open(&rep.files[0]).unwrap()).unwrap();
        assert_eq!(r.metadata().file_metadata().num_rows(), 1);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rejects_lookahead_rows_and_bad_columns() {
        let (lake, root) = lake();
        let b = Batch::new()
            .utf8_const("venue", "bybit", 1)
            .utf8_const("symbol", "BTCUSDT", 1)
            .ts("event_ts", vec![100])
            .f64("rate", vec![0.0])
            .ts("known_ts", vec![99]);
        assert!(lake.write("funding", &b).unwrap_err().to_string().contains("known_ts < event_ts"));
        let b = b.f64("not_a_column", vec![1.0]);
        assert!(lake.write("funding", &b).is_err());
        std::fs::remove_dir_all(root).ok();
    }
}
