//! HTTP client with retry logic and request builders.

use crate::error::Error;
use crate::providers::{get_provider_with_base_url, Provider, RequestConfig, ToolChoice};
use crate::stream::CompletionStream;
use crate::types::*;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

/// Main client for making LLM API requests.
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    api_keys: Arc<HashMap<String, String>>,
    base_urls: Arc<HashMap<String, String>>,
    config: ClientConfig,
}

/// Client configuration.
#[derive(Clone, Debug)]
pub struct ClientConfig {
    /// Request timeout.
    pub timeout: Duration,
    /// Maximum retry attempts.
    pub max_retries: u32,
    /// Initial retry backoff.
    pub retry_backoff: Duration,
    /// Maximum retry backoff.
    pub max_backoff: Duration,
    /// Backoff multiplier.
    pub backoff_multiplier: f32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(120),
            max_retries: 3,
            retry_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

/// Builder for Client.
pub struct ClientBuilder {
    api_keys: HashMap<String, String>,
    base_urls: HashMap<String, String>,
    config: ClientConfig,
    http_builder: reqwest::ClientBuilder,
}

impl ClientBuilder {
    /// Create a new client builder.
    pub fn new() -> Self {
        Self {
            api_keys: HashMap::new(),
            base_urls: HashMap::new(),
            config: ClientConfig::default(),
            http_builder: reqwest::Client::builder()
                .pool_max_idle_per_host(10)
                .pool_idle_timeout(Duration::from_secs(90))
                .tcp_nodelay(true),
        }
    }

    /// Add an API key for a provider.
    pub fn api_key(mut self, provider: &str, key: impl Into<String>) -> Self {
        self.api_keys.insert(provider.to_string(), key.into());
        self
    }

    /// Set a custom base URL for a provider (useful for testing with mock servers).
    pub fn base_url(mut self, provider: &str, url: impl Into<String>) -> Self {
        self.base_urls.insert(provider.to_string(), url.into());
        self
    }

    /// Set request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set maximum retry attempts.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    /// Set initial retry backoff.
    pub fn retry_backoff(mut self, backoff: Duration) -> Self {
        self.config.retry_backoff = backoff;
        self
    }

    /// Load API keys from environment variables.
    pub fn from_env(mut self) -> Self {
        let env_mappings = [
            ("cerebras", "CEREBRAS_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
        ];

        for (provider, env_var) in env_mappings {
            if let Ok(key) = env::var(env_var) {
                self.api_keys.insert(provider.to_string(), key);
            }
        }

        self
    }

    /// Build the client.
    pub fn build(self) -> Result<Client, Error> {
        let http = self
            .http_builder
            .timeout(self.config.timeout)
            .build()
            .map_err(|e| Error::Config(e.to_string()))?;

        Ok(Client {
            http,
            api_keys: Arc::new(self.api_keys),
            base_urls: Arc::new(self.base_urls),
            config: self.config,
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Client {
    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self, Error> {
        ClientBuilder::new().from_env().build()
    }

    /// Create a new client builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Start building a streaming request.
    pub fn stream<'a>(&'a self, model: &str, messages: &'a [Message]) -> RequestBuilder<'a> {
        RequestBuilder {
            client: self,
            model: model.to_string(),
            messages,
            config: RequestConfig::default(),
            streaming: true,
        }
    }

    /// Start building a non-streaming request.
    pub fn complete<'a>(&'a self, model: &str, messages: &'a [Message]) -> RequestBuilder<'a> {
        RequestBuilder {
            client: self,
            model: model.to_string(),
            messages,
            config: RequestConfig::default(),
            streaming: false,
        }
    }

    /// Get API key for a provider.
    fn get_api_key(&self, provider: &str) -> Result<&str, Error> {
        self.api_keys
            .get(provider)
            .map(std::string::String::as_str)
            .ok_or_else(|| Error::MissingApiKey(provider.to_string()))
    }

    /// Get custom base URL for a provider, if configured.
    fn get_base_url(&self, provider: &str) -> Option<&str> {
        self.base_urls
            .get(provider)
            .map(std::string::String::as_str)
    }

    /// Execute a streaming request with retry.
    async fn execute_stream(
        &self,
        provider: &dyn Provider,
        api_key: &str,
        body: serde_json::Value,
        model: String,
    ) -> Result<
        CompletionStream<impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin>,
        Error,
    > {
        let url = provider.stream_url(&model, api_key);
        let headers = provider.headers(api_key);

        let mut attempt = 0;
        let mut backoff = self.config.retry_backoff;

        loop {
            attempt += 1;

            let response = self
                .http
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        let stream = resp.bytes_stream();
                        let parser = provider.create_parser();
                        return Ok(CompletionStream::new(Box::pin(stream), parser, model));
                    }

                    // Handle errors
                    let error = self.handle_error_response(resp).await;

                    if !error.is_retryable() || attempt >= self.config.max_retries {
                        return Err(error);
                    }

                    // Extract retry-after if available
                    if let Error::RateLimited {
                        retry_after: Some(duration),
                    } = &error
                    {
                        backoff = *duration;
                    }
                }
                Err(e) => {
                    if e.is_timeout() {
                        if attempt >= self.config.max_retries {
                            return Err(Error::Timeout);
                        }
                    } else if e.is_connect() {
                        if attempt >= self.config.max_retries {
                            return Err(Error::Http(e));
                        }
                    } else {
                        return Err(Error::Http(e));
                    }
                }
            }

            // Exponential backoff with jitter
            let jitter = fastrand::f32() * 0.3 + 0.85; // 0.85-1.15
            let sleep_duration = Duration::from_secs_f32(backoff.as_secs_f32() * jitter);
            sleep(sleep_duration).await;

            backoff = Duration::from_secs_f32(
                (backoff.as_secs_f32() * self.config.backoff_multiplier)
                    .min(self.config.max_backoff.as_secs_f32()),
            );
        }
    }

    /// Execute a non-streaming request with retry.
    async fn execute_complete(
        &self,
        provider: &dyn Provider,
        api_key: &str,
        body: serde_json::Value,
        model: &str,
    ) -> Result<CompletionResult, Error> {
        let url = provider.complete_url(model, api_key);
        let headers = provider.headers(api_key);

        let mut attempt = 0;
        let mut backoff = self.config.retry_backoff;

        loop {
            attempt += 1;

            let response = self
                .http
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();

                    if status.is_success() {
                        let text = resp.text().await.map_err(Error::Http)?;
                        return provider.parse_response(&text);
                    }

                    let error = self.handle_error_response(resp).await;

                    if !error.is_retryable() || attempt >= self.config.max_retries {
                        return Err(error);
                    }

                    if let Error::RateLimited {
                        retry_after: Some(duration),
                    } = &error
                    {
                        backoff = *duration;
                    }
                }
                Err(e) => {
                    if e.is_timeout() {
                        if attempt >= self.config.max_retries {
                            return Err(Error::Timeout);
                        }
                    } else if e.is_connect() {
                        if attempt >= self.config.max_retries {
                            return Err(Error::Http(e));
                        }
                    } else {
                        return Err(Error::Http(e));
                    }
                }
            }

            let jitter = fastrand::f32() * 0.3 + 0.85;
            let sleep_duration = Duration::from_secs_f32(backoff.as_secs_f32() * jitter);
            sleep(sleep_duration).await;

            backoff = Duration::from_secs_f32(
                (backoff.as_secs_f32() * self.config.backoff_multiplier)
                    .min(self.config.max_backoff.as_secs_f32()),
            );
        }
    }

    /// Convert error response to Error type.
    async fn handle_error_response(&self, resp: reqwest::Response) -> Error {
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();

        let body = resp.text().await.unwrap_or_default();

        match status {
            401 => Error::Unauthorized,
            429 => {
                let retry_after = parse_retry_after(&headers);
                Error::RateLimited { retry_after }
            }
            500..=599 => Error::Server(status),
            _ => {
                // Try to extract error message from JSON
                let message = serde_json::from_str::<serde_json::Value>(&body)
                    .ok()
                    .and_then(|v| {
                        v["error"]["message"]
                            .as_str()
                            .map(std::string::ToString::to_string)
                    })
                    .unwrap_or(body);
                Error::api(status, message)
            }
        }
    }
}

/// Parse Retry-After header.
fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers.get(RETRY_AFTER).and_then(|v| {
        v.to_str().ok().and_then(|s| {
            // Try parsing as seconds
            s.parse::<u64>().ok().map(Duration::from_secs)
            // Or as HTTP date (not implemented for simplicity)
        })
    })
}

/// Builder for individual requests.
pub struct RequestBuilder<'a> {
    client: &'a Client,
    model: String,
    messages: &'a [Message],
    config: RequestConfig,
    streaming: bool,
}

impl RequestBuilder<'_> {
    /// Set maximum tokens to generate.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    /// Set temperature for sampling.
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = Some(temp);
        self
    }

    /// Set top-p for nucleus sampling.
    pub fn top_p(mut self, p: f32) -> Self {
        self.config.top_p = Some(p);
        self
    }

    /// Set stop sequences.
    pub fn stop(mut self, sequences: Vec<String>) -> Self {
        self.config.stop = Some(sequences);
        self
    }

    /// Set tools for function calling.
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.config.tools = Some(tools);
        self
    }

    /// Set tool choice.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.config.tool_choice = Some(choice);
        self
    }

    /// Set system message (for providers that support it separately).
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.config.system = Some(system.into());
        self
    }

    /// Add extra provider-specific fields.
    pub fn extra(mut self, extra: serde_json::Value) -> Self {
        self.config.extra = Some(extra);
        self
    }

    /// Send the streaming request.
    pub async fn send(
        self,
    ) -> Result<
        CompletionStream<impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin>,
        Error,
    > {
        let model_id = ModelId::parse(&self.model)?;
        let base_url = self.client.get_base_url(&model_id.provider);
        let provider =
            get_provider_with_base_url(&model_id.provider, base_url).ok_or_else(|| {
                Error::InvalidModel(format!("unknown provider: {}", model_id.provider))
            })?;
        let api_key = self.client.get_api_key(&model_id.provider)?;

        if self.streaming {
            let body = provider.build_stream_body(&model_id.model, self.messages, &self.config)?;
            self.client
                .execute_stream(provider.as_ref(), api_key, body, model_id.model)
                .await
        } else {
            // For non-streaming, we'd need a different return type
            // This is a limitation of the current API design
            Err(Error::Config(
                "use send_complete() for non-streaming".into(),
            ))
        }
    }

    /// Send a non-streaming request.
    pub async fn send_complete(self) -> Result<CompletionResult, Error> {
        let model_id = ModelId::parse(&self.model)?;
        let base_url = self.client.get_base_url(&model_id.provider);
        let provider =
            get_provider_with_base_url(&model_id.provider, base_url).ok_or_else(|| {
                Error::InvalidModel(format!("unknown provider: {}", model_id.provider))
            })?;
        let api_key = self.client.get_api_key(&model_id.provider)?;

        let body = provider.build_complete_body(&model_id.model, self.messages, &self.config)?;
        self.client
            .execute_complete(provider.as_ref(), api_key, body, &model_id.model)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder() {
        let client = Client::builder()
            .api_key("cerebras", "test-key")
            .timeout(Duration::from_secs(60))
            .max_retries(5)
            .build()
            .unwrap();

        assert_eq!(client.config.max_retries, 5);
        assert_eq!(client.api_keys.get("cerebras").unwrap(), "test-key");
    }

    #[test]
    fn test_request_builder() {
        let client = Client::builder()
            .api_key("cerebras", "test")
            .build()
            .unwrap();

        let messages = vec![Message::user("Hi")];
        let builder = client
            .stream("cerebras/llama3.1-70b", &messages)
            .max_tokens(100)
            .temperature(0.7)
            .top_p(0.9);

        assert_eq!(builder.config.max_tokens, Some(100));
        assert_eq!(builder.config.temperature, Some(0.7));
        assert_eq!(builder.config.top_p, Some(0.9));
    }
}
