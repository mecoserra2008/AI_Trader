//! Hot-path estimators. Everything here is O(1) per update (recursive filters) or closed-form,
//! so intraday updates across the whole universe cost microseconds and a few bytes of state
//! per asset. Parameters are fitted offline in Python and loaded here (ARCHITECTURE §2.2).

pub mod chain;
pub mod ewma;
pub mod garch;
pub mod passage;
pub mod range_vol;
pub mod robust;
