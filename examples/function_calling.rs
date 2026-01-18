//! Function calling example with Cerebras.
//!
//! Run with: CEREBRAS_API_KEY=... cargo run --example function_calling

use rust_ai_sdk::{Client, Message, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::from_env()?;

    // Define available tools
    let tools = vec![
        Tool::function(
            "get_weather",
            "Get the current weather for a location",
            json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country, e.g. 'London, UK'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }),
        ),
        Tool::function(
            "search_web",
            "Search the web for information",
            json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }),
        ),
    ];

    let messages = vec![Message::user(
        "What's the weather like in Tokyo and Paris right now?",
    )];

    println!("Requesting with function calling...\n");

    let mut stream = client
        .stream("cerebras/llama-3.3-70b", &messages)
        .max_tokens(1024)
        .tools(tools)
        .send()
        .await?;

    // Collect response
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(text) = chunk.text() {
            print!("{}", text);
        }
    }

    let result = stream.finalize()?;

    // Check for tool calls
    if !result.tool_calls.is_empty() {
        println!("\n\n--- Tool Calls ---");
        for tc in &result.tool_calls {
            println!("Function: {}", tc.function.name);
            println!("Arguments: {}", tc.function.arguments);
            println!();
        }
    }

    println!("--- Stats ---");
    println!("Finish reason: {:?}", result.finish_reason);
    println!(
        "Tokens: {} in, {} out",
        result.usage.input_tokens, result.usage.output_tokens
    );

    Ok(())
}
