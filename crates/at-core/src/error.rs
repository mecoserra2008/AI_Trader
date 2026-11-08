#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A computation saw data that was not knowable at its `as_of` time.
    #[error("look-ahead violation: data_max_ts {data_max_ts} > as_of {as_of}")]
    LookAhead { data_max_ts: i64, as_of: i64 },

    #[error("config error: {0}")]
    Config(String),

    #[error("venue error {code}: {msg}")]
    Venue { code: i64, msg: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
