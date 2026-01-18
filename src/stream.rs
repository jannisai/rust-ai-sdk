//! Streaming completion handler with usage accumulation.

use crate::error::Error;
use crate::sse::SseParser;
use crate::types::*;
use bytes::Bytes;
use futures::Stream;
use pin_project_lite::pin_project;

pin_project! {
    /// A streaming completion response.
    ///
    /// Yields `StreamChunk` items and accumulates content/usage for finalization.
    pub struct CompletionStream<S> {
        #[pin]
        inner: S,
        parser: SseParser,
        provider_parser: Box<dyn ProviderParser + Send>,
        // Accumulation state
        content: String,
        usage: Usage,
        finish_reason: Option<FinishReason>,
        tool_calls: ToolCallAccumulator,
        model: String,
        // Stream state
        done: bool,
        finalized: bool,
    }
}

/// Trait for provider-specific chunk parsing.
pub trait ProviderParser: Send {
    /// Parse an SSE data payload into a StreamChunk.
    fn parse_chunk(&mut self, data: &str) -> Result<Option<StreamChunk>, Error>;

    /// Check if this data indicates end of stream.
    fn is_end_of_stream(&self, data: &str) -> bool;
}

impl<S> CompletionStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    /// Create a new completion stream.
    pub fn new(inner: S, parser: Box<dyn ProviderParser + Send>, model: String) -> Self {
        Self {
            inner,
            parser: SseParser::new(),
            provider_parser: parser,
            content: String::with_capacity(4096),
            usage: Usage::default(),
            finish_reason: None,
            tool_calls: ToolCallAccumulator::default(),
            model,
            done: false,
            finalized: false,
        }
    }

    /// Get the next chunk from the stream.
    pub async fn next(&mut self) -> Option<Result<StreamChunk, Error>> {
        use futures::StreamExt;

        if self.done {
            return None;
        }

        loop {
            // First, try to get an event from buffered data
            if let Some(event) = self.parser.next_event() {
                if self.provider_parser.is_end_of_stream(event.data) {
                    self.done = true;
                    return None;
                }

                match self.provider_parser.parse_chunk(event.data) {
                    Ok(Some(chunk)) => {
                        self.accumulate(&chunk);
                        return Some(Ok(chunk));
                    }
                    Ok(None) => continue, // Skip empty chunks
                    Err(e) => return Some(Err(e)),
                }
            }

            // Need more data from the stream
            match self.inner.next().await {
                Some(Ok(bytes)) => {
                    self.parser.feed(&bytes);
                }
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(Error::Http(e)));
                }
                None => {
                    // Stream ended - check for any remaining buffered data
                    if let Some(event) = self.parser.next_event() {
                        if !self.provider_parser.is_end_of_stream(event.data) {
                            if let Ok(Some(chunk)) = self.provider_parser.parse_chunk(event.data) {
                                self.accumulate(&chunk);
                                self.done = true;
                                return Some(Ok(chunk));
                            }
                        }
                    }
                    self.done = true;
                    return None;
                }
            }
        }
    }

    /// Accumulate chunk data for final result.
    fn accumulate(&mut self, chunk: &StreamChunk) {
        // Accumulate text
        if let Some(text) = chunk.text() {
            self.content.push_str(&text);
        }

        // Update usage (keep latest/max)
        if let Some(usage) = &chunk.usage {
            self.usage.merge(usage);
        }

        // Update finish reason
        if chunk.finish_reason.is_some() {
            self.finish_reason = chunk.finish_reason;
        }

        // Accumulate tool calls
        if let Some(delta) = &chunk.tool_call_delta {
            self.tool_calls.apply(delta);
        }
    }

    /// Finalize the stream and get the accumulated result.
    ///
    /// Must be called after the stream is exhausted.
    pub fn finalize(mut self) -> Result<CompletionResult, Error> {
        if self.finalized {
            return Err(Error::StreamConsumed);
        }
        self.finalized = true;

        Ok(CompletionResult {
            content: std::mem::take(&mut self.content),
            usage: std::mem::take(&mut self.usage),
            model: std::mem::take(&mut self.model),
            finish_reason: self.finish_reason.unwrap_or(FinishReason::Stop),
            tool_calls: self.tool_calls.finalize(),
        })
    }

    /// Get current accumulated content without finalizing.
    pub fn current_content(&self) -> &str {
        &self.content
    }

    /// Get current accumulated usage without finalizing.
    pub fn current_usage(&self) -> &Usage {
        &self.usage
    }

    /// Check if stream is done.
    pub fn is_done(&self) -> bool {
        self.done
    }
}

/// Builder for CompletionStream that allows custom configuration.
pub struct StreamBuilder<S> {
    inner: S,
    parser: Box<dyn ProviderParser + Send>,
    model: String,
    content_capacity: usize,
}

impl<S> StreamBuilder<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    /// Create a new stream builder.
    pub fn new(inner: S, parser: Box<dyn ProviderParser + Send>, model: String) -> Self {
        Self {
            inner,
            parser,
            model,
            content_capacity: 4096,
        }
    }

    /// Set initial content buffer capacity.
    pub fn content_capacity(mut self, cap: usize) -> Self {
        self.content_capacity = cap;
        self
    }

    /// Build the CompletionStream.
    pub fn build(self) -> CompletionStream<S> {
        let mut stream = CompletionStream::new(self.inner, self.parser, self.model);
        stream.content = String::with_capacity(self.content_capacity);
        stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestParser;

    impl ProviderParser for TestParser {
        fn parse_chunk(&mut self, data: &str) -> Result<Option<StreamChunk>, Error> {
            if let Some(text) = data.strip_prefix("text:") {
                Ok(Some(StreamChunk::text_owned(text.to_string())))
            } else if data == "usage" {
                Ok(Some(StreamChunk::usage(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                })))
            } else {
                Ok(None)
            }
        }

        fn is_end_of_stream(&self, data: &str) -> bool {
            data == "[DONE]"
        }
    }

    #[tokio::test]
    async fn test_stream_accumulation() {
        let chunks = vec![
            Ok(Bytes::from("data: text:Hello\n\n")),
            Ok(Bytes::from("data: text: World\n\n")),
            Ok(Bytes::from("data: usage\n\n")),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let stream = futures::stream::iter(chunks);

        let mut completion =
            CompletionStream::new(stream, Box::new(TestParser), "test-model".to_string());

        let mut texts = Vec::new();
        while let Some(chunk) = completion.next().await {
            let chunk = chunk.unwrap();
            if let Some(text) = chunk.text() {
                texts.push(text.to_string());
            }
        }

        assert_eq!(texts, vec!["Hello", " World"]);

        let result = completion.finalize().unwrap();
        assert_eq!(result.content, "Hello World");
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }
}
