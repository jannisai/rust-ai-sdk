//! Integration tests using TokenIpsum mock server.
//!
//! These tests spin up a local TokenIpsum server and run the rust-ai-sdk client against it.

use rust_ai_sdk::{Client, ClientBuilder, Message};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tokenipsum::{create_router, Config, RuntimeState};

/// Start a TokenIpsum mock server on a random available port.
async fn start_mock_server() -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let config = Config::default();
    let state = RuntimeState::new(config);
    let app = create_router(state);

    // Bind to port 0 to get a random available port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    (addr, handle)
}

/// Create a client configured to use the mock server.
fn create_test_client(mock_addr: SocketAddr) -> Client {
    ClientBuilder::new()
        .api_key("cerebras", "test-key")
        .api_key("claude", "test-key")
        .api_key("gemini", "test-key")
        .api_key("openai", "test-key")
        .base_url("cerebras", format!("http://{}/v1", mock_addr))
        .base_url("claude", format!("http://{}", mock_addr))
        .base_url("gemini", format!("http://{}/v1beta", mock_addr))
        .base_url("openai", format!("http://{}", mock_addr))
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_cerebras_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Hello, how are you?")];

    let mut stream = client
        .stream("cerebras/llama-3.3-70b", &messages)
        .max_tokens(100)
        .send()
        .await
        .unwrap();

    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.text() {
            content.push_str(text.as_ref());
        }
    }

    let result = stream.finalize().unwrap();

    // TokenIpsum generates fake content
    assert!(!content.is_empty(), "Should receive content");
    assert!(result.usage.input_tokens > 0, "Should have input tokens");
    assert!(result.usage.output_tokens > 0, "Should have output tokens");
}

#[tokio::test]
async fn test_cerebras_non_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Say hello")];

    let result = client
        .complete("cerebras/llama-3.3-70b", &messages)
        .max_tokens(50)
        .send_complete()
        .await
        .unwrap();

    assert!(!result.content.is_empty(), "Should receive content");
    assert!(result.usage.input_tokens > 0);
    assert!(result.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_gemini_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Tell me a joke")];

    let mut stream = client
        .stream("gemini/gemini-2.0-flash", &messages)
        .max_tokens(100)
        .send()
        .await
        .unwrap();

    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.text() {
            content.push_str(text.as_ref());
        }
    }

    let result = stream.finalize().unwrap();

    assert!(!content.is_empty(), "Should receive content");
    assert!(result.usage.input_tokens > 0);
    assert!(result.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_tool_calling() {
    use rust_ai_sdk::Tool;

    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("What's the weather in Paris?")];

    let tools = vec![Tool::function(
        "get_weather",
        "Get the current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["location"]
        }),
    )];

    let mut stream = client
        .stream("cerebras/llama-3.3-70b", &messages)
        .max_tokens(100)
        .tools(tools)
        .send()
        .await
        .unwrap();

    while let Some(chunk) = stream.next().await {
        let _ = chunk.unwrap();
    }

    let result = stream.finalize().unwrap();

    // TokenIpsum detects "weather" keyword and generates tool calls
    assert!(
        !result.tool_calls.is_empty() || !result.content.is_empty(),
        "Should receive either tool calls or content"
    );
}

#[tokio::test]
async fn test_multiple_messages() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello!"),
        Message::assistant("Hi there! How can I help you today?"),
        Message::user("What's 2+2?"),
    ];

    let mut stream = client
        .stream("cerebras/llama-3.3-70b", &messages)
        .max_tokens(50)
        .send()
        .await
        .unwrap();

    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.text() {
            content.push_str(text.as_ref());
        }
    }

    assert!(!content.is_empty());
}

#[tokio::test]
async fn test_with_temperature() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Generate something random")];

    let result = client
        .complete("cerebras/llama-3.3-70b", &messages)
        .max_tokens(50)
        .temperature(0.9)
        .top_p(0.95)
        .send_complete()
        .await
        .unwrap();

    assert!(!result.content.is_empty());
}

#[tokio::test]
async fn test_claude_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Hello from Claude test")];

    let mut stream = client
        .stream("claude/claude-3-haiku-20240307", &messages)
        .max_tokens(100)
        .send()
        .await
        .unwrap();

    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.text() {
            content.push_str(text.as_ref());
        }
    }

    let result = stream.finalize().unwrap();

    assert!(!content.is_empty(), "Should receive content from Claude");
    assert!(result.usage.input_tokens > 0);
    assert!(result.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_claude_non_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Say hello")];

    let result = client
        .complete("claude/claude-3-haiku-20240307", &messages)
        .max_tokens(50)
        .send_complete()
        .await
        .unwrap();

    assert!(!result.content.is_empty());
    assert!(result.usage.input_tokens > 0);
    assert!(result.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_openai_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Hello from OpenAI test")];

    let mut stream = client
        .stream("openai/gpt-4o-mini", &messages)
        .max_tokens(100)
        .send()
        .await
        .unwrap();

    let mut content = String::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        if let Some(text) = chunk.text() {
            content.push_str(text.as_ref());
        }
    }

    let result = stream.finalize().unwrap();

    assert!(!content.is_empty(), "Should receive content from OpenAI");
    assert!(result.usage.input_tokens > 0);
    assert!(result.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_openai_non_streaming() {
    let (addr, _handle) = start_mock_server().await;
    let client = create_test_client(addr);

    let messages = vec![Message::user("Say hello")];

    let result = client
        .complete("openai/gpt-4o-mini", &messages)
        .max_tokens(50)
        .send_complete()
        .await
        .unwrap();

    assert!(!result.content.is_empty());
    assert!(result.usage.input_tokens > 0);
    assert!(result.usage.output_tokens > 0);
}
