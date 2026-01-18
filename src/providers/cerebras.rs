//! Cerebras provider implementation.
//!
//! Cerebras uses OpenAI-compatible API with some extensions.
//! Streaming uses SSE with `[DONE]` marker.

use crate::error::Error;
use crate::providers::{Provider, RequestConfig};
use crate::stream::ProviderParser;
use crate::types::*;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::Value;

/// Cerebras API provider.
pub struct CerebrasProvider {
    base_url: String,
}

impl CerebrasProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.cerebras.ai/v1".to_string(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl Default for CerebrasProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for CerebrasProvider {
    fn name(&self) -> &'static str {
        "cerebras"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn headers(&self, api_key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Ok(auth) = HeaderValue::from_str(&format!("Bearer {}", api_key)) {
            headers.insert(AUTHORIZATION, auth);
        }
        headers
    }

    fn build_stream_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        let mut body = self.build_base_body(model, messages, config)?;

        // Enable streaming with usage tracking
        body["stream"] = Value::Bool(true);
        body["stream_options"] = serde_json::json!({
            "include_usage": true
        });

        Ok(body)
    }

    fn build_complete_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        let mut body = self.build_base_body(model, messages, config)?;
        body["stream"] = Value::Bool(false);
        Ok(body)
    }

    fn create_parser(&self) -> Box<dyn ProviderParser + Send> {
        Box::new(CerebrasParser::new())
    }

    fn parse_response(&self, body: &str) -> Result<CompletionResult, Error> {
        let resp: CerebrasResponse =
            serde_json::from_str(body).map_err(|e| Error::parse(e.to_string()))?;

        let choice = resp
            .choices
            .first()
            .ok_or_else(|| Error::parse("no choices"))?;

        Ok(CompletionResult {
            content: choice.message.content.clone().unwrap_or_default(),
            usage: Usage {
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
                ..Default::default()
            },
            model: resp.model,
            finish_reason: parse_finish_reason(choice.finish_reason.as_deref()),
            tool_calls: choice.message.tool_calls.clone().unwrap_or_default(),
        })
    }
}

impl CerebrasProvider {
    fn build_base_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        // Convert messages to Cerebras format
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut obj = serde_json::json!({
                    "role": m.role,
                    "content": match &m.content {
                        MessageContent::Text(s) => Value::String(s.clone()),
                        MessageContent::Parts(parts) => serde_json::to_value(parts).unwrap_or(Value::Null),
                    }
                });

                if let Some(name) = &m.name {
                    obj["name"] = Value::String(name.clone());
                }
                if let Some(tool_call_id) = &m.tool_call_id {
                    obj["tool_call_id"] = Value::String(tool_call_id.clone());
                }
                if let Some(tool_calls) = &m.tool_calls {
                    obj["tool_calls"] = serde_json::to_value(tool_calls).unwrap_or(Value::Null);
                }

                obj
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": msgs,
        });

        // Add optional parameters
        if let Some(max_tokens) = config.max_tokens {
            body["max_tokens"] = Value::Number(max_tokens.into());
        }
        if let Some(temperature) = config.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(top_p) = config.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(stop) = &config.stop {
            body["stop"] = serde_json::to_value(stop).unwrap_or(Value::Null);
        }

        // Tool calling support
        if let Some(tools) = &config.tools {
            body["tools"] = serde_json::to_value(tools).unwrap_or(Value::Null);
        }
        if let Some(tool_choice) = &config.tool_choice {
            body["tool_choice"] = tool_choice.to_value();
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

/// Streaming response parser for Cerebras.
pub struct CerebrasParser {
    /// Scratch buffer for JSON parsing to avoid allocations.
    #[cfg(feature = "simd-json")]
    scratch: Vec<u8>,
}

impl CerebrasParser {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "simd-json")]
            scratch: Vec::with_capacity(4096),
        }
    }

    /// Parse a streaming chunk using zero-copy where possible.
    fn parse_chunk_inner(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        // Fast path: parse JSON
        #[cfg(feature = "simd-json")]
        let chunk: CerebrasStreamChunk = {
            let mut data_bytes = data.as_bytes().to_vec();
            simd_json::from_slice(&mut data_bytes).map_err(|e| Error::parse(e.to_string()))?
        };

        #[cfg(not(feature = "simd-json"))]
        let chunk: CerebrasStreamChunk =
            serde_json::from_str(data).map_err(|e| Error::parse(e.to_string()))?;

        // Check for usage-only chunk (no choices, just usage)
        if chunk.choices.is_empty() {
            if let Some(usage) = chunk.usage {
                return Ok(Some(StreamChunk::usage(Usage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    ..Default::default()
                })));
            }
            return Ok(None);
        }

        let choice = &chunk.choices[0];
        let delta = &choice.delta;

        // Build chunk based on content
        let mut stream_chunk = if let Some(content) = &delta.content {
            if content.is_empty() {
                StreamChunk::empty(ChunkKind::Text)
            } else {
                StreamChunk::text_owned(content.clone())
            }
        } else if delta.tool_calls.is_some() {
            StreamChunk::empty(ChunkKind::ToolDelta)
        } else {
            StreamChunk::empty(ChunkKind::Unknown)
        };

        // Handle tool call deltas
        if let Some(tool_calls) = &delta.tool_calls {
            if let Some(tc) = tool_calls.first() {
                stream_chunk.tool_call_delta = Some(ToolCallDelta {
                    index: tc.index,
                    id: tc.id.clone(),
                    function_name: tc.function.as_ref().and_then(|f| f.name.clone()),
                    function_arguments: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                });
                stream_chunk.kind = ChunkKind::ToolDelta;
            }
        }

        // Set finish reason
        if let Some(reason) = &choice.finish_reason {
            stream_chunk.finish_reason = Some(parse_finish_reason(Some(reason)));
        }

        // Set usage if present
        if let Some(usage) = chunk.usage {
            stream_chunk.usage = Some(Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                ..Default::default()
            });
        }

        Ok(Some(stream_chunk))
    }
}

impl Default for CerebrasParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderParser for CerebrasParser {
    fn parse_chunk(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        self.parse_chunk_inner(data)
    }

    fn is_end_of_stream(&self, data: &str) -> bool {
        data == "[DONE]"
    }
}

/// Parse finish reason string to enum.
fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

// --- Serde types for Cerebras API ---

#[derive(Debug, Deserialize)]
struct CerebrasResponse {
    model: String,
    choices: Vec<CerebrasChoice>,
    usage: CerebrasUsage,
}

#[derive(Debug, Deserialize)]
struct CerebrasChoice {
    message: CerebrasMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CerebrasMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct CerebrasUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct CerebrasStreamChunk {
    #[allow(dead_code)]
    id: Option<String>,
    choices: Vec<CerebrasStreamChoice>,
    usage: Option<CerebrasUsage>,
}

#[derive(Debug, Deserialize)]
struct CerebrasStreamChoice {
    delta: CerebrasStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CerebrasStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<CerebrasToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct CerebrasToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<CerebrasFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct CerebrasFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_chunk() {
        let mut parser = CerebrasParser::new();
        let data = r#"{"id":"123","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#;

        let chunk = parser.parse_chunk(data).unwrap().unwrap();
        assert_eq!(chunk.text().unwrap().as_ref(), "Hello");
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn test_parse_finish_chunk() {
        let mut parser = CerebrasParser::new();
        let data = r#"{"id":"123","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

        let chunk = parser.parse_chunk(data).unwrap().unwrap();
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn test_parse_usage_chunk() {
        let mut parser = CerebrasParser::new();
        let data =
            r#"{"id":"123","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}"#;

        let chunk = parser.parse_chunk(data).unwrap().unwrap();
        assert_eq!(chunk.kind, ChunkKind::UsageOnly);
        let usage = chunk.usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
    }

    #[test]
    fn test_parse_tool_call_delta() {
        let mut parser = CerebrasParser::new();
        let data = r#"{"id":"123","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","function":{"name":"get_weather","arguments":"{\"loc"}}]},"finish_reason":null}]}"#;

        let chunk = parser.parse_chunk(data).unwrap().unwrap();
        assert_eq!(chunk.kind, ChunkKind::ToolDelta);
        let delta = chunk.tool_call_delta.unwrap();
        assert_eq!(delta.index, 0);
        assert_eq!(delta.id, Some("call_123".to_string()));
        assert_eq!(delta.function_name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_is_done() {
        let parser = CerebrasParser::new();
        assert!(parser.is_end_of_stream("[DONE]"));
        assert!(!parser.is_end_of_stream("{}"));
    }

    #[test]
    fn test_build_body_with_tools() {
        let provider = CerebrasProvider::new();
        let messages = vec![Message::user("What's the weather?")];
        let tools = vec![Tool::function(
            "get_weather",
            "Get current weather",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
        )];

        let config = RequestConfig {
            max_tokens: Some(100),
            tools: Some(tools),
            ..Default::default()
        };

        let body = provider
            .build_stream_body("llama3.1-70b", &messages, &config)
            .unwrap();

        assert!(body["tools"].is_array());
        assert!(body["stream"].as_bool().unwrap());
        assert!(body["stream_options"]["include_usage"].as_bool().unwrap());
    }
}
