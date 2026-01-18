//! OpenAI Responses API provider implementation.
//!
//! OpenAI Responses API is a newer API format with:
//! - Input as string or messages array
//! - Streaming with named events: response.created, response.output_text.delta, etc.
//! - Rich metadata including billing, reasoning, and service tier

use crate::error::Error;
use crate::providers::{Provider, RequestConfig, ToolChoice};
use crate::stream::ProviderParser;
use crate::types::*;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::Value;

/// OpenAI Responses API provider.
pub struct OpenAIProvider {
    base_url: String,
}

impl OpenAIProvider {
    pub fn new() -> Self {
        Self {
            base_url: "https://api.openai.com".to_string(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl Provider for OpenAIProvider {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn headers(&self, api_key: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if let Ok(auth) = HeaderValue::from_str(&format!("Bearer {api_key}")) {
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
        Box::new(OpenAIParser::new())
    }

    fn parse_response(&self, body: &str) -> Result<CompletionResult, Error> {
        let resp: ResponsesResponse =
            serde_json::from_str(body).map_err(|e| Error::parse(e.to_string()))?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for item in &resp.output {
            match item {
                OutputItem::Message { content: parts, .. } => {
                    for part in parts {
                        if part.content_type == "output_text" {
                            content.push_str(&part.text);
                        }
                    }
                }
                OutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    tool_calls.push(ToolCall {
                        id: call_id.clone(),
                        tool_type: "function".to_string(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    });
                }
            }
        }

        let finish_reason = if tool_calls.is_empty() {
            match resp.status.as_str() {
                "completed" => FinishReason::Stop,
                "incomplete" => FinishReason::Length,
                _ => FinishReason::Unknown,
            }
        } else {
            FinishReason::ToolCalls
        };

        Ok(CompletionResult {
            content,
            usage: Usage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
                cache_read_input_tokens: resp.usage.input_tokens_details.cached_tokens,
                ..Default::default()
            },
            model: resp.model,
            finish_reason,
            tool_calls,
        })
    }

    fn chat_endpoint(&self) -> &'static str {
        "/v1/responses"
    }
}

impl OpenAIProvider {
    fn build_base_body(
        &self,
        model: &str,
        messages: &[Message],
        config: &RequestConfig,
    ) -> Result<Value, Error> {
        // Extract system as instructions
        let instructions = messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.content.as_text())
            .or(config.system.as_deref());

        // Convert messages to input format
        let input = self.convert_messages(messages);

        let mut body = serde_json::json!({
            "model": model,
            "input": input,
        });

        if let Some(instructions) = instructions {
            body["instructions"] = Value::String(instructions.to_string());
        }

        if let Some(max_tokens) = config.max_tokens {
            body["max_output_tokens"] = Value::Number(max_tokens.into());
        }
        if let Some(temp) = config.temperature {
            body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = config.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }

        // Tools
        if let Some(tools) = &config.tools {
            let openai_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters
                    })
                })
                .collect();
            body["tools"] = Value::Array(openai_tools);
        }

        // Tool choice
        if let Some(tool_choice) = &config.tool_choice {
            body["tool_choice"] = match tool_choice {
                ToolChoice::Auto => Value::String("auto".to_string()),
                ToolChoice::None => Value::String("none".to_string()),
                ToolChoice::Required => Value::String("required".to_string()),
                ToolChoice::Function(name) => serde_json::json!({
                    "type": "function",
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

    fn convert_messages(&self, messages: &[Message]) -> Value {
        let msgs: Vec<Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => "tool",
                };

                let content = match &m.content {
                    MessageContent::Text(text) => Value::String(text.clone()),
                    MessageContent::Parts(parts) => {
                        let arr: Vec<Value> = parts
                            .iter()
                            .map(|p| match p {
                                ContentPart::Text { text } => {
                                    serde_json::json!({"type": "text", "text": text})
                                }
                                ContentPart::ImageUrl { image_url } => {
                                    serde_json::json!({
                                        "type": "image_url",
                                        "image_url": {"url": image_url.url}
                                    })
                                }
                            })
                            .collect();
                        Value::Array(arr)
                    }
                };

                serde_json::json!({
                    "role": role,
                    "content": content
                })
            })
            .collect();

        Value::Array(msgs)
    }
}

/// Streaming response parser for OpenAI Responses API.
pub struct OpenAIParser {
    #[allow(dead_code)]
    current_usage: Option<Usage>,
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    tool_index: usize,
}

impl OpenAIParser {
    pub fn new() -> Self {
        Self {
            current_usage: None,
            current_tool_id: None,
            current_tool_name: None,
            tool_index: 0,
        }
    }
}

impl Default for OpenAIParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderParser for OpenAIParser {
    fn parse_chunk(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
        let event: OpenAIStreamEvent =
            serde_json::from_str(data).map_err(|e| Error::parse(e.to_string()))?;

        match event {
            OpenAIStreamEvent::ResponseCreated { .. }
            | OpenAIStreamEvent::ResponseInProgress { .. } => Ok(None),

            OpenAIStreamEvent::OutputItemAdded { item, .. } => {
                if let Some(StreamOutputItem::FunctionCall { call_id, name, .. }) = item {
                    self.current_tool_id = Some(call_id);
                    self.current_tool_name = Some(name);
                }
                Ok(None)
            }

            OpenAIStreamEvent::ContentPartAdded { .. } => Ok(None),

            OpenAIStreamEvent::OutputTextDelta { delta, .. } => {
                Ok(Some(StreamChunk::text_owned(delta)))
            }

            OpenAIStreamEvent::FunctionCallArgumentsDelta { delta, .. } => {
                let mut chunk = StreamChunk::empty(ChunkKind::ToolDelta);
                chunk.tool_call_delta = Some(ToolCallDelta {
                    index: self.tool_index,
                    id: self.current_tool_id.clone(),
                    function_name: self.current_tool_name.clone(),
                    function_arguments: Some(delta),
                });
                Ok(Some(chunk))
            }

            OpenAIStreamEvent::FunctionCallArgumentsDone { .. } => {
                self.tool_index += 1;
                self.current_tool_id = None;
                self.current_tool_name = None;
                Ok(None)
            }

            OpenAIStreamEvent::OutputTextDone { .. }
            | OpenAIStreamEvent::ContentPartDone { .. }
            | OpenAIStreamEvent::OutputItemDone { .. } => Ok(None),

            OpenAIStreamEvent::ResponseCompleted { response } => {
                let usage = response.usage.map(|u| Usage {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    cache_read_input_tokens: u.input_tokens_details.cached_tokens,
                    ..Default::default()
                });

                let finish_reason = match response.status.as_str() {
                    "completed" => FinishReason::Stop,
                    "incomplete" => FinishReason::Length,
                    _ => FinishReason::Unknown,
                };

                let mut chunk = StreamChunk::empty(ChunkKind::Unknown);
                chunk.usage = usage;
                chunk.finish_reason = Some(finish_reason);
                Ok(Some(chunk))
            }

            OpenAIStreamEvent::Error { error } => Err(Error::Api {
                status: 0,
                message: error.message,
            }),

            OpenAIStreamEvent::Unknown => Ok(None),
        }
    }

    fn is_end_of_stream(&self, _data: &str) -> bool {
        false
    }
}

// --- Serde types for OpenAI Responses API ---

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    model: String,
    status: String,
    output: Vec<OutputItem>,
    usage: ResponsesUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum OutputItem {
    #[serde(rename = "message")]
    Message {
        #[allow(dead_code)]
        id: String,
        content: Vec<OutputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[allow(dead_code)]
        id: String,
        name: String,
        arguments: String,
        call_id: String,
    },
}

#[derive(Debug, Deserialize)]
struct OutputContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    input_tokens: u32,
    output_tokens: u32,
    input_tokens_details: TokenDetails,
}

#[derive(Debug, Deserialize, Default)]
struct TokenDetails {
    #[serde(default)]
    cached_tokens: u32,
}

// Stream event types
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum OpenAIStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated {
        #[allow(dead_code)]
        response: StreamResponse,
    },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        #[allow(dead_code)]
        response: StreamResponse,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        #[allow(dead_code)]
        output_index: usize,
        item: Option<StreamOutputItem>,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        #[allow(dead_code)]
        content_index: usize,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        #[allow(dead_code)]
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        #[allow(dead_code)]
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        #[allow(dead_code)]
        item_id: String,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        #[allow(dead_code)]
        arguments: String,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {},
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {},
    #[serde(rename = "response.completed")]
    ResponseCompleted { response: CompletedResponse },
    #[serde(rename = "error")]
    Error { error: OpenAIError },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
struct StreamResponse {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamOutputItem {
    #[serde(rename = "message")]
    Message {},
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
    },
}

#[derive(Debug, Deserialize)]
struct CompletedResponse {
    status: String,
    usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_message_response() {
        let provider = OpenAIProvider::new();
        let json = r#"{
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_123",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "Hello!"
                }],
                "role": "assistant"
            }],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}
            }
        }"#;

        let result = provider.parse_response(json).unwrap();
        assert_eq!(result.content, "Hello!");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
        assert_eq!(result.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn test_parse_function_call_response() {
        let provider = OpenAIProvider::new();
        let json = r#"{
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [{
                "type": "function_call",
                "id": "fc_123",
                "status": "completed",
                "name": "get_weather",
                "arguments": "{\"location\":\"Tokyo\"}",
                "call_id": "call_123"
            }],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 15,
                "total_tokens": 65,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0}
            }
        }"#;

        let result = provider.parse_response(json).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].function.name, "get_weather");
        assert_eq!(result.finish_reason, FinishReason::ToolCalls);
    }

    #[test]
    fn test_parse_stream_text_delta() {
        let mut parser = OpenAIParser::new();

        // response.created
        let created = r#"{"type":"response.created","sequence_number":0,"response":{"id":"resp_123","status":"in_progress"}}"#;
        parser.parse_chunk(created).unwrap();

        // output_text.delta
        let delta = r#"{"type":"response.output_text.delta","sequence_number":5,"output_index":0,"content_index":0,"delta":"Hello"}"#;
        let chunk = parser.parse_chunk(delta).unwrap().unwrap();
        assert_eq!(chunk.text().unwrap().as_ref(), "Hello");
    }

    #[test]
    fn test_parse_stream_function_call() {
        let mut parser = OpenAIParser::new();

        // output_item.added with function_call
        let added = r#"{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"type":"function_call","call_id":"call_123","name":"get_weather","arguments":""}}"#;
        parser.parse_chunk(added).unwrap();

        // function_call_arguments.delta
        let delta = r#"{"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc_123","output_index":0,"delta":"{\"loc"}"#;
        let chunk = parser.parse_chunk(delta).unwrap().unwrap();
        assert_eq!(chunk.kind, ChunkKind::ToolDelta);
        let tool_delta = chunk.tool_call_delta.unwrap();
        assert_eq!(tool_delta.function_name, Some("get_weather".to_string()));
    }

    #[test]
    fn test_build_body_with_tools() {
        let provider = OpenAIProvider::new();
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
            .build_stream_body("gpt-4o-mini", &messages, &config)
            .unwrap();

        assert!(body["tools"].is_array());
        assert!(body["stream"].as_bool().unwrap());
        assert_eq!(body["max_output_tokens"], 100);
    }

    #[test]
    fn test_headers() {
        let provider = OpenAIProvider::new();
        let headers = provider.headers("test-key");
        assert!(headers.contains_key("authorization"));
        let auth = headers.get("authorization").unwrap().to_str().unwrap();
        assert!(auth.starts_with("Bearer "));
    }
}
