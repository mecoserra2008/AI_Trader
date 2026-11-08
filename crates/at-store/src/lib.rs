//! Bitemporal Parquet lake. `config/schema.yaml` is the contract: this crate validates every
//! write against it, routes rows to hive partitions and writes zstd Parquet atomically
//! (tmp file + rename, so concurrent readers never see partial files).
//!
//! Reads are done from Python/DuckDB (`src/ai_trader/storage/lake.py`) with the as-of rule
//! `known_ts <= as_of`, keeping the latest `known_ts` per primary key.

pub mod batch;
pub mod schema;
pub mod writer;

pub use batch::{Batch, Col};
pub use schema::{ColType, ColumnSpec, Schema, TableSpec};
pub use writer::{Lake, WriteReport};
