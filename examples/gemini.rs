//! Gemini streaming example with cost tracking.
//!
//! Run with: GEMINI_API_KEY=... cargo run --example gemini

use rust_ai_sdk::{Client, CostTracker, Message, PricingRegistry, Tool};
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::from_env()?;
    let pricing = PricingRegistry::new();
    let mut cost_tracker = CostTracker::new();

    // Example 1: Simple streaming
    println!("=== Simple Streaming ===\n");

    let messages = vec![Message::user("Explain quantum computing in 2 sentences.")];

    let mut stream = client
        .stream("gemini/gemini-2.0-flash", &messages)
        .max_tokens(256)
        .temperature(0.7)
        .send()
        .await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(text) = chunk.text() {
            print!("{}", text);
        }
    }

    let result = stream.finalize()?;
    println!("\n");

    // Calculate and track cost
    let model_key = "gemini/gemini-2.0-flash";
    if let Some(cost) = pricing.calculate_cost(model_key, &result.usage) {
        cost_tracker.record(&result.usage, Some(&cost));
        println!(
            "Usage: {} in, {} out | Cost: ${:.6}",
            result.usage.input_tokens,
            result.usage.output_tokens,
            cost.total()
        );
    }

    // Example 2: Function calling
    println!("\n=== Function Calling ===\n");

    let tools = vec![Tool::function(
        "get_weather",
        "Get current weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }),
    )];

    let messages = vec![Message::user("What's the weather in San Francisco?")];

    let mut stream = client
        .stream("gemini/gemini-2.0-flash", &messages)
        .max_tokens(256)
        .tools(tools)
        .send()
        .await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(text) = chunk.text() {
            print!("{}", text);
        }
    }

    let result = stream.finalize()?;

    if !result.tool_calls.is_empty() {
        println!("\nTool calls:");
        for tc in &result.tool_calls {
            println!("  - {}({})", tc.function.name, tc.function.arguments);
        }
    }

    // Track this request too
    if let Some(cost) = pricing.calculate_cost(model_key, &result.usage) {
        cost_tracker.record(&result.usage, Some(&cost));
    }

    // Summary
    println!("\n=== Session Summary ===");
    println!("Total requests: {}", cost_tracker.request_count());
    println!(
        "Total tokens: {} in, {} out",
        cost_tracker.input_tokens(),
        cost_tracker.output_tokens()
    );
    println!("Total cost: ${:.6}", cost_tracker.total_cost());

    Ok(())
}
