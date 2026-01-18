use std::time::Duration;
use thiserror::Error;

/// Unified error type across all providers.
#[derive(Error, Debug)]
pub enum Error {
    /// Rate limited by the provider. Check `retry_after` for suggested wait time.
    #[error("rate limited")]
    RateLimited { retry_after: Option<Duration> },

    /// Invalid or missing API key.
    #[error("unauthorized")]
    Unauthorized,

    /// Server error (5xx status codes).
    #[error("server error ({0})")]
    Server(u16),

    /// API error with provider-specific message.
    #[error("{message}")]
    Api { status: u16, message: String },

    /// Request or connection timeout.
    #[error("timeout")]
    Timeout,

    /// JSON or SSE parsing error.
    #[error("parse: {0}")]
    Parse(String),

    /// Invalid model ID format.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Missing API key for provider.
    #[error("missing API key for {0}")]
    MissingApiKey(String),

    /// HTTP/network error.
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),

    /// Stream was already consumed.
    #[error("stream already finalized")]
    StreamConsumed,

    /// Invalid configuration.
    #[error("config: {0}")]
    Config(String),
}

impl Error {
    /// Returns true if this error is retryable.
    #[inline]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Error::RateLimited { .. } | Error::Server(_) | Error::Timeout
        )
    }

    /// Create an API error from status and message.
    pub fn api(status: u16, message: impl Into<String>) -> Self {
        Self::Api {
            status,
            message: message.into(),
        }
    }

    /// Create a parse error.
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(msg.into())
    }
}
