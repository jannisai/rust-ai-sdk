//! Zero-copy SSE (Server-Sent Events) parser.
//!
//! Handles:
//! - Partial frames across TCP chunks
//! - Multi-line data fields
//! - CRLF and LF line endings
//! - Buffer compaction to prevent unbounded growth

use bytes::{Buf, BytesMut};
use memchr::memchr;

/// A parsed SSE event with zero-copy views into the buffer.
#[derive(Debug)]
pub struct SseEvent<'a> {
    pub event: Option<&'a str>,
    pub data: &'a str,
    pub id: Option<&'a str>,
}

/// Line-based SSE parser with minimal allocations.
pub struct SseParser {
    buffer: BytesMut,
    /// Scratch space for multi-line data concatenation.
    data_scratch: String,
    /// Current event type being built.
    event_scratch: String,
    /// Current id being built.
    id_scratch: String,
    /// Offset of unconsumed data in buffer.
    consumed: usize,
}

impl SseParser {
    /// Create a new parser with default buffer capacity.
    pub fn new() -> Self {
        Self::with_capacity(8192)
    }

    /// Create a new parser with specified initial capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buffer: BytesMut::with_capacity(cap),
            data_scratch: String::with_capacity(1024),
            event_scratch: String::new(),
            id_scratch: String::new(),
            consumed: 0,
        }
    }

    /// Feed bytes into the parser.
    #[inline]
    pub fn feed(&mut self, data: &[u8]) {
        // Compact buffer if we've consumed more than half
        if self.consumed > self.buffer.len() / 2 && self.consumed > 4096 {
            self.compact();
        }
        self.buffer.extend_from_slice(data);
    }

    /// Compact buffer by removing consumed bytes.
    fn compact(&mut self) {
        if self.consumed > 0 {
            self.buffer.advance(self.consumed);
            self.consumed = 0;
        }
    }

    /// Try to parse the next complete event.
    /// Returns `None` if more data is needed.
    pub fn next_event(&mut self) -> Option<SseEvent<'_>> {
        // Clear scratch buffers
        self.data_scratch.clear();
        self.event_scratch.clear();
        self.id_scratch.clear();

        let buf = &self.buffer[self.consumed..];
        let mut pos = 0;
        let mut found_blank = false;
        let mut event_end = 0;

        // Process lines until we hit a blank line
        while pos < buf.len() {
            // Find end of line
            let line_end = match memchr(b'\n', &buf[pos..]) {
                Some(i) => pos + i,
                None => return None, // Need more data
            };

            let line = &buf[pos..line_end];
            // Handle CRLF
            let line = if line.ends_with(b"\r") {
                &line[..line.len() - 1]
            } else {
                line
            };

            // Check for blank line (event boundary)
            if line.is_empty() {
                found_blank = true;
                event_end = line_end + 1;
                break;
            }

            // Parse field
            if let Some(colon_pos) = memchr(b':', line) {
                let field = &line[..colon_pos];
                // Value starts after colon, skip optional space
                let value_start = if colon_pos + 1 < line.len() && line[colon_pos + 1] == b' ' {
                    colon_pos + 2
                } else {
                    colon_pos + 1
                };
                let value = &line[value_start..];

                // Safe to convert to str - SSE spec requires UTF-8
                if let Ok(value_str) = std::str::from_utf8(value) {
                    match field {
                        b"data" => {
                            if !self.data_scratch.is_empty() {
                                self.data_scratch.push('\n');
                            }
                            self.data_scratch.push_str(value_str);
                        }
                        b"event" => {
                            self.event_scratch.clear();
                            self.event_scratch.push_str(value_str);
                        }
                        b"id" => {
                            self.id_scratch.clear();
                            self.id_scratch.push_str(value_str);
                        }
                        _ => {} // Ignore unknown fields
                    }
                }
            }
            // Lines starting with ':' are comments, ignore them

            pos = line_end + 1;
        }

        if !found_blank {
            return None; // Need more data for complete event
        }

        // Update consumed position
        self.consumed += event_end;

        // Only return if we have data
        if self.data_scratch.is_empty() {
            // Empty event, try next
            return self.next_event();
        }

        // SAFETY: We're returning references to scratch buffers that live in `self`.
        // The returned SseEvent borrows from these scratch buffers which are cleared
        // at the start of each next_event() call. The lifetime 'a in SseEvent<'a>
        // is tied to the borrow of `self`, ensuring the references remain valid.
        // The pointer casts extend the borrow to match the return lifetime.
        #[allow(unsafe_code)]
        Some(SseEvent {
            event: if self.event_scratch.is_empty() {
                None
            } else {
                // SAFETY: event_scratch lives in self and won't be modified until next call
                Some(unsafe { &*(self.event_scratch.as_str() as *const str) })
            },
            // SAFETY: data_scratch lives in self and won't be modified until next call
            data: unsafe { &*(self.data_scratch.as_str() as *const str) },
            id: if self.id_scratch.is_empty() {
                None
            } else {
                // SAFETY: id_scratch lives in self and won't be modified until next call
                Some(unsafe { &*(self.id_scratch.as_str() as *const str) })
            },
        })
    }

    /// Check if the data indicates end of stream (e.g., `[DONE]`).
    #[inline]
    pub fn is_done(data: &str) -> bool {
        data == "[DONE]"
    }

    /// Reset parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.data_scratch.clear();
        self.event_scratch.clear();
        self.id_scratch.clear();
        self.consumed = 0;
    }

    /// Current buffer size.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len() - self.consumed
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_event() {
        let mut parser = SseParser::new();
        parser.feed(b"data: hello world\n\n");

        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "hello world");
        assert!(event.event.is_none());
    }

    #[test]
    fn test_multiline_data() {
        let mut parser = SseParser::new();
        parser.feed(b"data: line1\ndata: line2\ndata: line3\n\n");

        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "line1\nline2\nline3");
    }

    #[test]
    fn test_event_type() {
        let mut parser = SseParser::new();
        parser.feed(b"event: message\ndata: payload\n\n");

        let event = parser.next_event().unwrap();
        assert_eq!(event.event, Some("message"));
        assert_eq!(event.data, "payload");
    }

    #[test]
    fn test_crlf() {
        let mut parser = SseParser::new();
        parser.feed(b"data: hello\r\n\r\n");

        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "hello");
    }

    #[test]
    fn test_partial_event() {
        let mut parser = SseParser::new();
        parser.feed(b"data: hel");
        assert!(parser.next_event().is_none());

        parser.feed(b"lo\n\n");
        let event = parser.next_event().unwrap();
        assert_eq!(event.data, "hello");
    }

    #[test]
    fn test_multiple_events() {
        let mut parser = SseParser::new();
        parser.feed(b"data: first\n\ndata: second\n\n");

        let event1 = parser.next_event().unwrap();
        assert_eq!(event1.data, "first");

        let event2 = parser.next_event().unwrap();
        assert_eq!(event2.data, "second");
    }

    #[test]
    fn test_coalesced_frames() {
        let mut parser = SseParser::new();
        // Multiple events in one TCP frame
        parser.feed(b"data: a\n\ndata: b\n\ndata: c\n\n");

        assert_eq!(parser.next_event().unwrap().data, "a");
        assert_eq!(parser.next_event().unwrap().data, "b");
        assert_eq!(parser.next_event().unwrap().data, "c");
        assert!(parser.next_event().is_none());
    }

    #[test]
    fn test_done_marker() {
        assert!(SseParser::is_done("[DONE]"));
        assert!(!SseParser::is_done("data"));
    }

    #[test]
    fn test_json_data() {
        let mut parser = SseParser::new();
        parser.feed(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n");

        let event = parser.next_event().unwrap();
        assert!(event.data.starts_with('{'));
        assert!(event.data.ends_with('}'));
    }
}
