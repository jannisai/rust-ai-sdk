//! Streaming example with Cerebras.
//!
//! Run with: CEREBRAS_API_KEY=... cargo run --example stream

use rust_ai_sdk::{Client, Message};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create client from environment
    let client = Client::from_env()?;

    // Build messages
    let messages = vec![
        Message::system("You are a helpful assistant. Be concise."),
        Message::user("Write a haiku about Rust programming."),
    ];

    // Stream the response
    println!("Streaming from Cerebras...\n");

    let mut stream = client
        .stream("cerebras/llama-3.3-70b", &messages)
        .max_tokens(256)
        .temperature(0.7)
        .send()
        .await?;

    // Print chunks as they arrive
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if let Some(text) = chunk.text() {
            print!("{}", text);
        }
    }

    // Get final result with usage stats
    let result = stream.finalize()?;

    println!("\n\n--- Stats ---");
    println!("Model: {}", result.model);
    println!("Input tokens: {}", result.usage.input_tokens);
    println!("Output tokens: {}", result.usage.output_tokens);
    println!("Finish reason: {:?}", result.finish_reason);

    Ok(())
}
