# rust-ai-sdk

High-performance streaming LLM client for Rust with zero-copy parsing and usage tracking.

## Features

- **Zero-copy SSE parsing** - Efficient streaming with minimal allocations
- **Multi-provider support** - Cerebras, Claude, Gemini, OpenAI
- **Function/tool calling** - Full support for tool use with streaming
- **Usage tracking** - Input/output/cache token counting
- **Cost tracking** - Built-in pricing for cost estimation
- **Async streaming** - Native tokio/futures support
- **Retry logic** - Exponential backoff with configurable retries

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rust-ai-sdk = { path = "." }
tokio = { version = "1", features = ["full"] }
```

## Quick Start

```rust
use rust_ai_sdk::{Client, Message};

#[tokio::main]
async fn main() -> Result<(), rust_ai_sdk::Error> {
    let client = Client::from_env()?;

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
    ];

    let mut stream = client
        .stream("cerebras/llama-3.3-70b", &messages)
        .max_tokens(256)
        .send()
        .await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(text) = chunk.text() {
            print!("{text}");
        }
    }

    let result = stream.finalize()?;
    println!("\nTokens: {} in, {} out",
        result.usage.input_tokens,
        result.usage.output_tokens
    );

    Ok(())
}
```

## Environment Variables

| Variable | Provider | Description |
|----------|----------|-------------|
| `CEREBRAS_API_KEY` | Cerebras | API key for Cerebras |
| `ANTHROPIC_API_KEY` | Claude | API key for Anthropic Claude |
| `GEMINI_API_KEY` | Gemini | API key for Google Gemini |
| `OPENAI_API_KEY` | OpenAI | API key for OpenAI |

## Providers

### Cerebras

```rust
// Model format: "cerebras/{model}"
let mut stream = client
    .stream("cerebras/llama-3.3-70b", &messages)
    .max_tokens(256)
    .temperature(0.7)
    .send()
    .await?;
```

Available models:
- `cerebras/llama-3.3-70b`
- `cerebras/llama3.1-8b`

### Claude

```rust
// Model format: "claude/{model}"
let mut stream = client
    .stream("claude/claude-3-5-sonnet-20241022", &messages)
    .max_tokens(256)
    .send()
    .await?;
```

Available models:
- `claude/claude-3-5-sonnet-20241022`
- `claude/claude-3-5-haiku-20241022`
- `claude/claude-3-opus-20240229`
- `claude/claude-3-haiku-20240307`

### Gemini

```rust
// Model format: "gemini/{model}"
let mut stream = client
    .stream("gemini/gemini-2.0-flash", &messages)
    .max_tokens(256)
    .send()
    .await?;
```

Available models:
- `gemini/gemini-2.0-flash`
- `gemini/gemini-1.5-pro`
- `gemini/gemini-1.5-flash`

### OpenAI

```rust
// Model format: "openai/{model}"
let mut stream = client
    .stream("openai/gpt-4o", &messages)
    .max_tokens(256)
    .send()
    .await?;
```

Available models:
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/o1`
- `openai/o1-mini`

## Function Calling

```rust
use rust_ai_sdk::{Client, Message, Tool};
use serde_json::json;

let tools = vec![
    Tool::function(
        "get_weather",
        "Get weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }),
    ),
];

let mut stream = client
    .stream("cerebras/llama-3.3-70b", &messages)
    .tools(tools)
    .send()
    .await?;

// ... stream chunks ...

let result = stream.finalize()?;
for tc in &result.tool_calls {
    println!("Call: {} with {}", tc.function.name, tc.function.arguments);
}
```

## Cost Tracking

```rust
use rust_ai_sdk::{CostTracker, PricingRegistry};

let registry = PricingRegistry::with_defaults();
let mut tracker = CostTracker::new(registry);

// After each completion
let result = stream.finalize()?;
if let Some(cost) = tracker.track(&result.model, &result.usage) {
    println!("Cost: ${:.6}", cost.total());
}

// Get totals
let total = tracker.total_cost();
println!("Session total: ${:.4}", total.total());
```

## Request Configuration

```rust
client
    .stream("cerebras/llama-3.3-70b", &messages)
    .max_tokens(256)          // Max output tokens
    .temperature(0.7)         // Randomness (0.0-2.0)
    .top_p(0.9)              // Nucleus sampling
    .tools(tools)            // Function definitions
    .send()
    .await?;
```

## Client Configuration

```rust
use rust_ai_sdk::ClientBuilder;
use std::time::Duration;

let client = ClientBuilder::new()
    .api_key("cerebras", "your-key")
    .api_key("claude", "your-key")
    .api_key("gemini", "your-key")
    .api_key("openai", "your-key")
    .timeout(Duration::from_secs(60))
    .max_retries(3)
    .build()?;
```

## Examples

```bash
# Streaming
CEREBRAS_API_KEY=... cargo run --example stream

# Function calling
CEREBRAS_API_KEY=... cargo run --example function_calling

# Gemini
GEMINI_API_KEY=... cargo run --example gemini
```

## Architecture

```
src/
├── lib.rs           # Public API exports
├── client.rs        # Client and request builders
├── stream.rs        # CompletionStream implementation
├── sse.rs           # Zero-copy SSE parser
├── types.rs         # Message, Tool, Usage types
├── error.rs         # Error types
├── cost.rs          # Pricing and cost tracking
└── providers/
    ├── mod.rs       # Provider trait
    ├── cerebras.rs  # Cerebras (OpenAI-compatible)
    ├── claude.rs    # Anthropic Claude Messages API
    ├── gemini.rs    # Google Gemini
    └── openai.rs    # OpenAI Responses API
```

## Testing

```bash
# Unit tests (42 tests)
cargo test --lib

# Integration tests with TokenIpsum mock server (10 tests)
cargo test --test tokenipsum_integration

# All tests
cargo test

# Clippy
cargo clippy --all-targets
```

Integration tests use [TokenIpsum](https://github.com/jannisai/tokenipsum) as a mock server, enabling testing without real API keys.

## Related

- **[tokenipsum](./tokenipsum/)** - Mock LLM API server for testing

## License

MIT OR Apache-2.0
