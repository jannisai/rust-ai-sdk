//! Google Gemini provider implementation.
//!
//! Gemini uses a different API format than OpenAI-compatible providers.
//! - Auth via `x-goog-api-key` header or `?key=` query param
//! - Different message format (`contents` with `parts`)
//! - Streaming via SSE with `?alt=sse`
//! - Usage in every chunk (keep last)
//! - No `[DONE]` marker - stream ends on connection close

use crate::error::Error;
use crate::providers::{Provider, RequestConfig};
use crate::stream::ProviderParser;
use crate::types::*;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::Value;

/// Gemini API provider.
pub struct GeminiProvider {
    base_url: String,
    /// API key stored for query param auth
    api_key_in_query: bool,
}

impl GeminiProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
            api_key_in_query: false,
        }
    }

    /// Use query parameter for API key instead of header.
    pub fn with_query_auth(mut self) -> Self {
        self.api_key_in_query = true;
        self
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Convert messages to Gemini format.
    fn convert_messages(&self, messages: &[Message]) -> Vec<Value> {
        messages
            .iter()
            .filter_map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "model",
                    Role::System => return None, // Handled separately
                    Role::Tool => "function", // Tool responses
                };

                let parts = match &m.content {
                    MessageContent::Text(text) => {
                        vec![serde_json::json!({"text": text})]
                    }
                    MessageContent::Parts(parts) => parts
                        .iter()
                        .map(|p| match p {
                            ContentPart::Text { text } => serde_json::json!({"text": text}),
                            ContentPart::ImageUrl { image_url } => {
                                serde_json::json!({
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_url.url.strip_prefix("data:image/jpeg;base64,")
                                            .unwrap_or(&image_url.url)
                                    }
                                })
                            }
                        })
                        .collect(),
                };

                Some(serde_json::json!({
                    "role": role,
                    "parts": parts
                }))
            })
            .collect()
    }

    /// Extract system instruction from messages.
    fn extract_system(&self, messages: &[Message]) -> Option<Value> {
        messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.content.as_text())
            .map(|text| {
                serde_json::json!({
                    "parts": [{"text": text}]
                })
            })
    }

    /// Convert tools to Gemini format.
    fn convert_tools(&self, tools: &[Tool]) -> Value {
        let function_declarations: Vec<Value> = tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.function.name,
                    "description": t.function.description,
                    "parameters": t.function.parameters
                })
            })
            .collect();

        serde_json::json!([{
            "function_declarations": function_declarations
        }])
    }
}

impl Default for GeminiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for GeminiProvider {
    fn name(&self) -> &'static str {
        "gemini"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn headers(&self, api_key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        // Use header auth by default
        if !self.api_key_in_query {
            if let Ok(key) = HeaderValue::from_str(api_key) {
                headers.insert("x-goog-api-key", key);
            }
        }
        headers
    }

    fn build_stream_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        self.build_body(model, messages, config)
    }

    fn build_complete_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        self.build_body(model, messages, config)
    }

    fn create_parser(&self) -> Box<dyn ProviderParser + Send> {
        Box::new(GeminiParser::new())
    }

    fn parse_response(&self, body: &str) -> Result<CompletionResult, Error> {
        let resp: GeminiResponse =
            serde_json::from_str(body).map_err(|e| Error::parse(e.to_string()))?;

        let candidate = resp
            .candidates
            .first()
            .ok_or_else(|| Error::parse("no candidates"))?;

        let content = candidate
            .content
            .parts
            .iter()
            .filter_map(|p| p.text.as_ref())
            .cloned()
            .collect::<String>();

        let tool_calls = candidate
            .content
            .parts
            .iter()
            .filter_map(|p| p.function_call.as_ref())
            .enumerate()
            .map(|(i, fc)| ToolCall {
                id: format!("call_{}", i),
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: fc.name.clone(),
                    arguments: serde_json::to_string(&fc.args).unwrap_or_default(),
                },
            })
            .collect();

        let finish_reason = match candidate.finish_reason.as_deref() {
            Some("STOP") => FinishReason::Stop,
            Some("MAX_TOKENS") => FinishReason::Length,
            Some("TOOL_CALLS") => FinishReason::ToolCalls,
            Some("SAFETY") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };

        let usage = resp.usage_metadata.map_or(Usage::default(), |u| Usage {
            input_tokens: u.prompt_token_count,
            output_tokens: u.candidates_token_count.unwrap_or(0),
            cache_read_input_tokens: u.cached_content_token_count.unwrap_or(0),
            ..Default::default()
        });

        Ok(CompletionResult {
            content,
            usage,
            model: resp.model_version.unwrap_or_default(),
            finish_reason,
            tool_calls,
        })
    }

    fn chat_endpoint(&self) -> &'static str {
        "" // Gemini uses model-specific endpoints
    }

    fn stream_url(&self, model: &str, api_key: &str) -> String {
        let base = format!(
            "{}/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        );
        if self.api_key_in_query {
            format!("{}&key={}", base, api_key)
        } else {
            base
        }
    }

    fn complete_url(&self, model: &str, api_key: &str) -> String {
        let base = format!("{}/models/{}:generateContent", self.base_url, model);
        if self.api_key_in_query {
            format!("{}?key={}", base, api_key)
        } else {
            base
        }
    }
}

impl GeminiProvider {
    fn build_body(
        &self,
        _model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        let contents = self.convert_messages(messages);

        let mut body = serde_json::json!({
            "contents": contents
        });

        // System instruction
        if let Some(system) = self.extract_system(messages) {
            body["system_instruction"] = system;
        }
        if let Some(system) = &config.system {
            body["system_instruction"] = serde_json::json!({
                "parts": [{"text": system}]
            });
        }

        // Generation config
        let mut gen_config = serde_json::json!({});
        if let Some(max_tokens) = config.max_tokens {
            gen_config["maxOutputTokens"] = Value::Number(max_tokens.into());
        }
        if let Some(temp) = config.temperature {
            gen_config["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = config.top_p {
            gen_config["topP"] = serde_json::json!(top_p);
        }
        if let Some(stop) = &config.stop {
            gen_config["stopSequences"] = serde_json::to_value(stop).unwrap_or(Value::Null);
        }
        if gen_config.as_object().is_some_and(|o| !o.is_empty()) {
            body["generationConfig"] = gen_config;
        }

        // Tools
        if let Some(tools) = &config.tools {
            body["tools"] = self.convert_tools(tools);
        }

        // Tool config (tool choice)
        if let Some(tool_choice) = &config.tool_choice {
            body["tool_config"] = match tool_choice {
                crate::providers::ToolChoice::Auto => serde_json::json!({
                    "function_calling_config": {"mode": "AUTO"}
                }),
                crate::providers::ToolChoice::None => serde_json::json!({
                    "function_calling_config": {"mode": "NONE"}
                }),
                crate::providers::ToolChoice::Required => serde_json::json!({
                    "function_calling_config": {"mode": "ANY"}
                }),
                crate::providers::ToolChoice::Function(name) => serde_json::json!({
                    "function_calling_config": {
                        "mode": "ANY",
                        "allowed_function_names": [name]
                    }
                }),
            };
        }

        // Merge extra fields
        if let Some(Value::Object(map)) = &config.extra {
            if let Value::Object(ref mut body_map) = body {
                for (k, v) in map {
                    body_map.insert(k.clone(), v.clone());
                }
            }
        }

        Ok(body)
    }
}

/// Streaming response parser for Gemini.
pub struct GeminiParser {
    last_usage: Option<Usage>,
}

impl GeminiParser {
    pub fn new() -> Self {
        Self { last_usage: None }
    }
}

impl Default for GeminiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderParser for GeminiParser {
    fn parse_chunk(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        let chunk: GeminiStreamChunk =
            serde_json::from_str(data).map_err(|e| Error::parse(e.to_string()))?;

        // Update usage (keep last)
        if let Some(usage) = &chunk.usage_metadata {
            self.last_usage = Some(Usage {
                input_tokens: usage.prompt_token_count,
                output_tokens: usage.candidates_token_count.unwrap_or(0),
                cache_read_input_tokens: usage.cached_content_token_count.unwrap_or(0),
                ..Default::default()
            });
        }

        let candidate = if let Some(c) = chunk.candidates.first() { c } else {
            // Usage-only chunk
            if let Some(usage) = self.last_usage.clone() {
                return Ok(Some(StreamChunk::usage(usage)));
            }
            return Ok(None);
        };

        // Extract text from parts
        let text: String = candidate
            .content
            .as_ref()
            .map(|c| {
                c.parts
                    .iter()
                    .filter_map(|p| p.text.as_ref())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        // Check for function calls
        let tool_call_delta = candidate.content.as_ref().and_then(|c| {
            c.parts.iter().find_map(|p| {
                p.function_call.as_ref().map(|fc| ToolCallDelta {
                    index: 0,
                    id: Some(format!("call_{}", fastrand::u32(..))),
                    function_name: Some(fc.name.clone()),
                    function_arguments: Some(serde_json::to_string(&fc.args).unwrap_or_default()),
                })
            })
        });

        let mut chunk = if !text.is_empty() {
            StreamChunk::text_owned(text)
        } else if tool_call_delta.is_some() {
            let mut c = StreamChunk::empty(ChunkKind::ToolDelta);
            c.tool_call_delta = tool_call_delta;
            c
        } else {
            StreamChunk::empty(ChunkKind::Unknown)
        };

        // Set finish reason
        if let Some(reason) = &candidate.finish_reason {
            chunk.finish_reason = Some(match reason.as_str() {
                "STOP" => FinishReason::Stop,
                "MAX_TOKENS" => FinishReason::Length,
                "SAFETY" => FinishReason::ContentFilter,
                _ => FinishReason::Unknown,
            });
        }

        // Attach usage
        chunk.usage = self.last_usage.clone();

        Ok(Some(chunk))
    }

    fn is_end_of_stream(&self, _data: &str) -> bool {
        // Gemini doesn't have a [DONE] marker - stream ends on connection close
        false
    }
}

// --- Serde types for Gemini API ---

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    usage_metadata: Option<GeminiUsage>,
    model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: GeminiContent,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    #[allow(dead_code)]
    role: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiPart {
    text: Option<String>,
    function_call: Option<GeminiFunctionCall>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    prompt_token_count: u32,
    candidates_token_count: Option<u32>,
    cached_content_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiStreamChunk {
    candidates: Vec<GeminiStreamCandidate>,
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiStreamCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages() {
        let provider = GeminiProvider::new();
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

        let contents = provider.convert_messages(&messages);
        // System message is filtered out
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
    }

    #[test]
    fn test_extract_system() {
        let provider = GeminiProvider::new();
        let messages = vec![Message::system("Be helpful"), Message::user("Hi")];

        let system = provider.extract_system(&messages);
        assert!(system.is_some());
    }

    #[test]
    fn test_parse_response() {
        let provider = GeminiProvider::new();
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello!"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        }"#;

        let result = provider.parse_response(json).unwrap();
        assert_eq!(result.content, "Hello!");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        assert_eq!(result.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn test_parse_stream_chunk() {
        let mut parser = GeminiParser::new();
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hi"}],
                    "role": "model"
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 1
            }
        }"#;

        let chunk = parser.parse_chunk(json).unwrap().unwrap();
        assert_eq!(chunk.text().unwrap().as_ref(), "Hi");
        assert!(chunk.usage.is_some());
    }

    #[test]
    fn test_parse_function_call() {
        let mut parser = GeminiParser::new();
        let json = r#"{
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Tokyo"}
                        }
                    }],
                    "role": "model"
                }
            }]
        }"#;

        let chunk = parser.parse_chunk(json).unwrap().unwrap();
        assert_eq!(chunk.kind, ChunkKind::ToolDelta);
        let delta = chunk.tool_call_delta.unwrap();
        assert_eq!(delta.function_name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_build_body_with_tools() {
        let provider = GeminiProvider::new();
        let messages = vec![Message::user("What's the weather?")];
        let tools = vec![Tool::function(
            "get_weather",
            "Get weather",
            serde_json::json!({"type": "object", "properties": {}}),
        )];

        let config = RequestConfig {
            max_tokens: Some(100),
            tools: Some(tools),
            ..Default::default()
        };

        let body = provider
            .build_body("gemini-pro", &messages, &config)
            .unwrap();
        assert!(body["tools"].is_array());
        assert!(body["generationConfig"]["maxOutputTokens"].is_number());
    }
}
