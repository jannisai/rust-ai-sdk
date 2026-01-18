//! High-performance streaming LLM client with zero-copy parsing and usage tracking.
//!
//! # Example
//! ```no_run
//! use rust_ai_sdk::{Client, Message, Role};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), rust_ai_sdk::Error> {
//!     let client = Client::from_env()?;
//!     let messages = vec![Message::user("Hello!")];
//!
//!     let mut stream = client
//!         .stream("cerebras/llama3.1-70b", &messages)
//!         .max_tokens(256)
//!         .send()
//!         .await?;
//!
//!     while let Some(chunk) = stream.next().await {
//!         let chunk = chunk?;
//!         if let Some(text) = chunk.text() {
//!             print!("{text}");
//!         }
//!     }
//!
//!     let result = stream.finalize()?;
//!     println!("\nTokens: {} in, {} out", result.usage.input_tokens, result.usage.output_tokens);
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod cost;
pub mod error;
pub mod providers;
pub mod sse;
pub mod stream;
pub mod types;

pub use client::{Client, ClientBuilder, RequestBuilder};
pub use cost::{Cost, CostTracker, ModelPricing, PricingRegistry};
pub use error::Error;
pub use stream::CompletionStream;
pub use types::*;

/// Result type alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
