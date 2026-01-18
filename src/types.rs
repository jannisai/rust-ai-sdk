use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::borrow::Cow;

/// Message role in conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A conversation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }
}

/// Message content - either plain text or structured parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Get text content if available.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            MessageContent::Text(s) => Some(s),
            MessageContent::Parts(_) => None,
        }
    }
}

/// Content part for multi-modal messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens read from cache (Anthropic).
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Tokens written to cache (Anthropic).
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
}

impl Usage {
    /// Total tokens used.
    #[inline]
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }

    /// Merge with another usage, taking max of each field.
    pub fn merge(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.max(other.input_tokens);
        self.output_tokens = self.output_tokens.max(other.output_tokens);
        self.cache_read_input_tokens = self
            .cache_read_input_tokens
            .max(other.cache_read_input_tokens);
        self.cache_creation_input_tokens = self
            .cache_creation_input_tokens
            .max(other.cache_creation_input_tokens);
    }
}

/// Reason the completion finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    #[serde(other)]
    Unknown,
}

/// Result of a completed request.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub content: String,
    pub usage: Usage,
    pub model: String,
    pub finish_reason: FinishReason,
    pub tool_calls: Vec<ToolCall>,
}

/// Kind of streaming chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    Text,
    UsageOnly,
    Ping,
    ToolDelta,
    Thinking,
    Unknown,
}

/// A single streaming chunk.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub kind: ChunkKind,
    /// Text content - may be borrowed (zero-copy) or owned (after unescape).
    text_data: TextData,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub tool_call_delta: Option<ToolCallDelta>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // Borrowed reserved for future zero-copy optimization
enum TextData {
    Empty,
    /// Zero-copy view into the original buffer (future optimization).
    Borrowed {
        start: usize,
        len: usize,
    },
    /// Owned after JSON unescape.
    Owned(String),
}

impl StreamChunk {
    /// Create an empty chunk.
    pub fn empty(kind: ChunkKind) -> Self {
        Self {
            kind,
            text_data: TextData::Empty,
            finish_reason: None,
            usage: None,
            tool_call_delta: None,
        }
    }

    /// Create a text chunk with owned data.
    pub fn text_owned(text: String) -> Self {
        Self {
            kind: ChunkKind::Text,
            text_data: if text.is_empty() {
                TextData::Empty
            } else {
                TextData::Owned(text)
            },
            finish_reason: None,
            usage: None,
            tool_call_delta: None,
        }
    }

    /// Create a usage-only chunk.
    pub fn usage(usage: Usage) -> Self {
        Self {
            kind: ChunkKind::UsageOnly,
            text_data: TextData::Empty,
            finish_reason: None,
            usage: Some(usage),
            tool_call_delta: None,
        }
    }

    /// Get text content if any. Returns Cow for zero-copy when possible.
    #[inline]
    pub fn text(&self) -> Option<Cow<'_, str>> {
        match &self.text_data {
            TextData::Empty => None,
            TextData::Borrowed { .. } => None, // Would need buffer reference
            TextData::Owned(s) if s.is_empty() => None,
            TextData::Owned(s) => Some(Cow::Borrowed(s.as_str())),
        }
    }

    /// Get owned text, consuming the chunk.
    pub fn into_text(self) -> Option<String> {
        match self.text_data {
            TextData::Empty => None,
            TextData::Borrowed { .. } => None,
            TextData::Owned(s) if s.is_empty() => None,
            TextData::Owned(s) => Some(s),
        }
    }

    /// Set finish reason.
    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = Some(reason);
        self
    }

    /// Set usage.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }
}

/// Tool/function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

impl Tool {
    /// Create a function tool.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
            },
        }
    }
}

/// Function definition for tool calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool call in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionCall,
}

/// Function call details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

impl FunctionCall {
    /// Parse arguments as JSON.
    pub fn parse_arguments<T: serde::de::DeserializeOwned>(&self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// Delta for streaming tool calls.
#[derive(Debug, Clone)]
pub struct ToolCallDelta {
    pub index: usize,
    pub id: Option<String>,
    pub function_name: Option<String>,
    pub function_arguments: Option<String>,
}

/// Accumulator for building tool calls from deltas.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    calls: SmallVec<[ToolCallBuilder; 4]>,
}

#[derive(Debug, Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    /// Apply a delta to the accumulator.
    pub fn apply(&mut self, delta: &ToolCallDelta) {
        // Ensure we have enough slots
        while self.calls.len() <= delta.index {
            self.calls.push(ToolCallBuilder::default());
        }

        let builder = &mut self.calls[delta.index];
        if let Some(id) = &delta.id {
            builder.id.push_str(id);
        }
        if let Some(name) = &delta.function_name {
            builder.name.push_str(name);
        }
        if let Some(args) = &delta.function_arguments {
            builder.arguments.push_str(args);
        }
    }

    /// Finalize into completed tool calls.
    pub fn finalize(self) -> Vec<ToolCall> {
        self.calls
            .into_iter()
            .filter(|b| !b.id.is_empty())
            .map(|b| ToolCall {
                id: b.id,
                tool_type: "function".to_string(),
                function: FunctionCall {
                    name: b.name,
                    arguments: b.arguments,
                },
            })
            .collect()
    }
}

/// Parsed provider and model from a model string.
#[derive(Debug, Clone)]
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

impl ModelId {
    /// Parse a model string like "cerebras/llama3.1-70b".
    pub fn parse(s: &str) -> Result<Self, crate::Error> {
        let (provider, model) = s
            .split_once('/')
            .ok_or_else(|| crate::Error::InvalidModel(s.to_string()))?;

        if provider.is_empty() || model.is_empty() {
            return Err(crate::Error::InvalidModel(s.to_string()));
        }

        Ok(Self {
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_parse() {
        let id = ModelId::parse("cerebras/llama3.1-70b").unwrap();
        assert_eq!(id.provider, "cerebras");
        assert_eq!(id.model, "llama3.1-70b");

        assert!(ModelId::parse("invalid").is_err());
        assert!(ModelId::parse("/model").is_err());
        assert!(ModelId::parse("provider/").is_err());
    }

    #[test]
    fn test_usage_merge() {
        let mut a = Usage {
            input_tokens: 10,
            output_tokens: 5,
            ..Default::default()
        };
        let b = Usage {
            input_tokens: 8,
            output_tokens: 20,
            ..Default::default()
        };
        a.merge(&b);
        assert_eq!(a.input_tokens, 10);
        assert_eq!(a.output_tokens, 20);
    }
}
