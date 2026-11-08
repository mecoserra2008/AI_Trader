//! Shared foundations: time and frequency handling, the look-ahead guard, domain types,
//! configuration loading and secrets.

pub mod config;
pub mod error;
pub mod num;
pub mod time;
pub mod types;

pub use error::{Error, Result};
