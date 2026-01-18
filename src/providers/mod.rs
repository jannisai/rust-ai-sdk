//! Provider implementations for different LLM APIs.

pub mod cerebras;
pub mod claude;
pub mod gemini;
pub mod openai;

use crate::error::Error;
use crate::stream::ProviderParser;
use crate::types::{Message, Tool};
use reqwest::header::HeaderMap;
use serde_json::Value;

/// Provider configuration and behavior.
pub trait Provider: Send + Sync {
    /// Provider name (e.g., "cerebras", "openai").
    fn name(&self) -> &str;

    /// Base URL for API requests.
    fn base_url(&self) -> &str;

    /// Build request headers including auth.
    fn headers(&self, api_key: &str) -> HeaderMap;

    /// Build request body for streaming completion.
    fn build_stream_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error>;

    /// Build request body for non-streaming completion.
    fn build_complete_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error>;

    /// Create a parser for streaming responses.
    fn create_parser(&self) -> Box<dyn ProviderParser + Send>;

    /// Parse a non-streaming response.
    fn parse_response(&self, body: &str) -> Result<crate::types::CompletionResult, Error>;

    /// API endpoint path for chat completions.
    fn chat_endpoint(&self) -> &'static str {
        "/chat/completions"
    }

    /// Build full URL for streaming request.
    fn stream_url(&self, _model: &str, _api_key: &str) -> String {
        format!("{}{}", self.base_url(), self.chat_endpoint())
    }

    /// Build full URL for non-streaming request.
    fn complete_url(&self, _model: &str, _api_key: &str) -> String {
        format!("{}{}", self.base_url(), self.chat_endpoint())
    }
}

/// Request configuration shared across providers.
#[derive(Debug, Clone, Default)]
pub struct RequestConfig {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    pub system: Option<String>,
    /// Extra provider-specific fields.
    pub extra: Option<Value>,
}

/// Tool choice configuration.
#[derive(Debug, Clone)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

impl ToolChoice {
    pub fn to_value(&self) -> Value {
        match self {
            ToolChoice::Auto => Value::String("auto".to_string()),
            ToolChoice::None => Value::String("none".to_string()),
            ToolChoice::Required => Value::String("required".to_string()),
            ToolChoice::Function(name) => serde_json::json!({
                "type": "function",
                "function": {"name": name}
            }),
        }
    }
}

/// Get provider by name.
pub fn get_provider(name: &str) -> Option<Box<dyn Provider>> {
    get_provider_with_base_url(name, None)
}

/// Get provider by name with optional custom base URL.
pub fn get_provider_with_base_url(name: &str, base_url: Option<&str>) -> Option<Box<dyn Provider>> {
    match name {
        "cerebras" => Some(Box::new(match base_url {
            Some(url) => cerebras::CerebrasProvider::with_base_url(url),
            None => cerebras::CerebrasProvider::new(),
        })),
        "claude" => Some(Box::new(match base_url {
            Some(url) => claude::ClaudeProvider::with_base_url(url),
            None => claude::ClaudeProvider::new(),
        })),
        "gemini" => Some(Box::new(match base_url {
            Some(url) => gemini::GeminiProvider::new().with_base_url(url),
            None => gemini::GeminiProvider::new(),
        })),
        "openai" => Some(Box::new(match base_url {
            Some(url) => openai::OpenAIProvider::with_base_url(url),
            None => openai::OpenAIProvider::new(),
        })),
        _ => None,
    }
}
