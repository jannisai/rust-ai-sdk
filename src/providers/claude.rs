//! Anthropic Claude provider implementation.
//!
//! Claude uses the Messages API with distinct streaming event types.
//! - Auth via `x-api-key` header (not Bearer token)
//! - Requires `anthropic-version` header
//! - Streaming uses named SSE events: message_start, content_block_delta, etc.
//! - Content blocks can be text, tool_use, or thinking

use crate::error::Error;
use crate::providers::{Provider, RequestConfig, ToolChoice};
use crate::stream::ProviderParser;
use crate::types::*;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::Value;

const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Claude API provider.
pub struct ClaudeProvider {
    base_url: String,
}

impl ClaudeProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.anthropic.com".to_string(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl Default for ClaudeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for ClaudeProvider {
    fn name(&self) -> &'static str {
        "claude"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn headers(&self, api_key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        if let Ok(key) = HeaderValue::from_str(api_key) {
            headers.insert("x-api-key", key);
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
        body["stream"] = Value::Bool(true);
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
        Box::new(ClaudeParser::new())
    }

    fn parse_response(&self, body: &str) -> Result<CompletionResult, Error> {
        let resp: ClaudeResponse =
            serde_json::from_str(body).map_err(|e| Error::parse(e.to_string()))?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in &resp.content {
            match block {
                ContentBlock::Text { text } => {
                    content.push_str(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    });
                }
                ContentBlock::Thinking { .. } => {
                    // Thinking blocks are not included in content
                }
            }
        }

        let finish_reason = match resp.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("tool_use") => FinishReason::ToolCalls,
            Some("stop_sequence") => FinishReason::Stop,
            _ => FinishReason::Unknown,
        };

        Ok(CompletionResult {
            content,
            usage: Usage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
                cache_read_input_tokens: resp.usage.cache_read_input_tokens.unwrap_or(0),
                cache_creation_input_tokens: resp.usage.cache_creation_input_tokens.unwrap_or(0),
            },
            model: resp.model,
            finish_reason,
            tool_calls,
        })
    }

    fn chat_endpoint(&self) -> &'static str {
        "/v1/messages"
    }
}

impl ClaudeProvider {
    fn build_base_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        // Extract system message
        let system_text = messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.content.as_text())
            .or(config.system.as_deref());

        // Convert non-system messages
        let msgs: Vec<Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| self.convert_message(m))
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": msgs,
            "max_tokens": config.max_tokens.unwrap_or(4096),
        });

        if let Some(system) = system_text {
            body["system"] = Value::String(system.to_string());
        }

        if let Some(temp) = config.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = config.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(stop) = &config.stop {
            body["stop_sequences"] = serde_json::to_value(stop).unwrap_or(Value::Null);
        }

        // Tools
        if let Some(tools) = &config.tools {
            let claude_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.function.name,
                        "description": t.function.description,
                        "input_schema": t.function.parameters
                    })
                })
                .collect();
            body["tools"] = Value::Array(claude_tools);
        }

        // Tool choice
        if let Some(tool_choice) = &config.tool_choice {
            body["tool_choice"] = match tool_choice {
                ToolChoice::Auto => serde_json::json!({"type": "auto"}),
                ToolChoice::None => serde_json::json!({"type": "none"}),
                ToolChoice::Required => serde_json::json!({"type": "any"}),
                ToolChoice::Function(name) => serde_json::json!({
                    "type": "tool",
                    "name": name
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

    fn convert_message(&self, msg: &Message) -> Value {
        let role = match msg.role {
            Role::User | Role::System => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user", // Tool results come as user messages
        };

        let content = match &msg.content {
            MessageContent::Text(text) => {
                if msg.role == Role::Tool {
                    // Tool result format
                    serde_json::json!([{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id.as_deref().unwrap_or(""),
                        "content": text
                    }])
                } else {
                    Value::String(text.clone())
                }
            }
            MessageContent::Parts(parts) => {
                let blocks: Vec<Value> = parts
                    .iter()
                    .map(|p| match p {
                        ContentPart::Text { text } => {
                            serde_json::json!({"type": "text", "text": text})
                        }
                        ContentPart::ImageUrl { image_url } => {
                            // Extract base64 data and media type
                            let url = &image_url.url;
                            if let Some(rest) = url.strip_prefix("data:") {
                                if let Some((media_type, data)) = rest.split_once(";base64,") {
                                    return serde_json::json!({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data
                                        }
                                    });
                                }
                            }
                            // Fallback to URL (Claude supports this too)
                            serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": url
                                }
                            })
                        }
                    })
                    .collect();
                Value::Array(blocks)
            }
        };

        serde_json::json!({
            "role": role,
            "content": content
        })
    }
}

/// Streaming response parser for Claude.
pub struct ClaudeParser {
    current_usage: Option<Usage>,
    current_block_type: Option<String>,
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    tool_index: usize,
}

impl ClaudeParser {
    pub fn new() -> Self {
        Self {
            current_usage: None,
            current_block_type: None,
            current_tool_id: None,
            current_tool_name: None,
            tool_index: 0,
        }
    }
}

impl Default for ClaudeParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderParser for ClaudeParser {
    fn parse_chunk(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        let event: ClaudeStreamEvent =
            serde_json::from_str(data).map_err(|e| Error::parse(e.to_string()))?;

        match event {
            ClaudeStreamEvent::MessageStart { message } => {
                // Initial usage from message_start
                self.current_usage = Some(Usage {
                    input_tokens: message.usage.input_tokens,
                    output_tokens: message.usage.output_tokens,
                    cache_read_input_tokens: message.usage.cache_read_input_tokens.unwrap_or(0),
                    cache_creation_input_tokens: message
                        .usage
                        .cache_creation_input_tokens
                        .unwrap_or(0),
                });
                Ok(None)
            }
            ClaudeStreamEvent::ContentBlockStart { content_block, .. } => {
                match content_block {
                    StreamContentBlock::Text { .. } => {
                        self.current_block_type = Some("text".to_string());
                    }
                    StreamContentBlock::ToolUse { id, name, .. } => {
                        self.current_block_type = Some("tool_use".to_string());
                        self.current_tool_id = Some(id);
                        self.current_tool_name = Some(name);
                    }
                    StreamContentBlock::Thinking { .. } => {
                        self.current_block_type = Some("thinking".to_string());
                    }
                }
                Ok(None)
            }
            ClaudeStreamEvent::ContentBlockDelta { delta, .. } => match delta {
                StreamDelta::TextDelta { text } => Ok(Some(StreamChunk::text_owned(text))),
                StreamDelta::InputJsonDelta { partial_json } => {
                    let mut chunk = StreamChunk::empty(ChunkKind::ToolDelta);
                    chunk.tool_call_delta = Some(ToolCallDelta {
                        index: self.tool_index,
                        id: self.current_tool_id.clone(),
                        function_name: self.current_tool_name.clone(),
                        function_arguments: Some(partial_json),
                    });
                    Ok(Some(chunk))
                }
                StreamDelta::ThinkingDelta { .. } | StreamDelta::SignatureDelta { .. } => {
                    // Skip thinking deltas for now
                    Ok(None)
                }
            },
            ClaudeStreamEvent::ContentBlockStop { .. } => {
                if self.current_block_type.as_deref() == Some("tool_use") {
                    self.tool_index += 1;
                }
                self.current_block_type = None;
                self.current_tool_id = None;
                self.current_tool_name = None;
                Ok(None)
            }
            ClaudeStreamEvent::MessageDelta { delta, usage } => {
                let finish_reason = match delta.stop_reason.as_deref() {
                    Some("end_turn") => Some(FinishReason::Stop),
                    Some("max_tokens") => Some(FinishReason::Length),
                    Some("tool_use") => Some(FinishReason::ToolCalls),
                    Some("stop_sequence") => Some(FinishReason::Stop),
                    _ => None,
                };

                // Update output tokens from message_delta
                if let Some(ref mut u) = self.current_usage {
                    u.output_tokens = usage.output_tokens;
                }

                let mut chunk = StreamChunk::empty(ChunkKind::Unknown);
                chunk.usage = self.current_usage.clone();
                chunk.finish_reason = finish_reason;
                Ok(Some(chunk))
            }
            ClaudeStreamEvent::MessageStop => Ok(None),
            ClaudeStreamEvent::Ping => Ok(None),
            ClaudeStreamEvent::Error { error } => Err(Error::Api {
                status: 0,
                message: error.message,
            }),
        }
    }

    fn is_end_of_stream(&self, _data: &str) -> bool {
        // Claude uses message_stop event, handled in parse_chunk
        false
    }
}

// --- Serde types for Claude API ---

#[derive(Debug, Deserialize)]
struct ClaudeResponse {
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: ClaudeUsage,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
}

#[derive(Debug, Deserialize, Clone)]
struct ClaudeUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClaudeStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: StreamMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: StreamContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: StreamDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDeltaData,
        usage: MessageDeltaUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "error")]
    Error { error: ClaudeError },
}

#[derive(Debug, Deserialize)]
struct StreamMessage {
    usage: ClaudeUsage,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
}

#[allow(dead_code, clippy::enum_variant_names)]
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MessageDeltaUsage {
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ClaudeError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_response() {
        let provider = ClaudeProvider::new();
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;

        let result = provider.parse_response(json).unwrap();
        assert_eq!(result.content, "Hello!");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        assert_eq!(result.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn test_parse_tool_use_response() {
        let provider = ClaudeProvider::new();
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [{
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "Tokyo"}
            }],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30}
        }"#;

        let result = provider.parse_response(json).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn test_parse_stream_text_delta() {
        let mut parser = ClaudeParser::new();

        // message_start
        let start =
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}"#;
        parser.parse_chunk(start).unwrap();

        // content_block_start
        let block_start =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        parser.parse_chunk(block_start).unwrap();

        // text_delta
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;
        let chunk = parser.parse_chunk(delta).unwrap().unwrap();
        assert_eq!(chunk.text().unwrap().as_ref(), "Hello");
    }

    #[test]
    fn test_parse_stream_tool_delta() {
        let mut parser = ClaudeParser::new();

        // message_start
        let start =
            r#"{"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}"#;
        parser.parse_chunk(start).unwrap();

        // content_block_start for tool_use
        let block_start = r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{}}}"#;
        parser.parse_chunk(block_start).unwrap();

        // input_json_delta
        let delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}"#;
        let chunk = parser.parse_chunk(delta).unwrap().unwrap();
        assert_eq!(chunk.kind, ChunkKind::ToolDelta);
        let tool_delta = chunk.tool_call_delta.unwrap();
        assert_eq!(tool_delta.function_name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_build_body_with_tools() {
        let provider = ClaudeProvider::new();
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
            .build_stream_body("claude-3-haiku", &messages, &config)
            .unwrap();

        assert!(body["tools"].is_array());
        assert!(body["stream"].as_bool().unwrap());
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn test_headers() {
        let provider = ClaudeProvider::new();
        let headers = provider.headers("test-key");
        assert!(headers.contains_key("x-api-key"));
        assert!(headers.contains_key("anthropic-version"));
    }
}
