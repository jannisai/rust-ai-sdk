//! Cost tracking for LLM API usage.
//!
//! Provides pricing information and cost calculation for different providers and models.

use crate::types::Usage;
use std::collections::HashMap;

/// Cost in USD for token usage.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cost {
    /// Cost for input tokens in USD.
    pub input_cost: f64,
    /// Cost for output tokens in USD.
    pub output_cost: f64,
    /// Cost for cached input tokens in USD (if applicable).
    pub cache_read_cost: f64,
    /// Cost for cache creation in USD (if applicable).
    pub cache_write_cost: f64,
}

impl Cost {
    /// Total cost in USD.
    pub fn total(&self) -> f64 {
        self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost
    }
}

/// Pricing per 1M tokens for a model.
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Cost per 1M input tokens.
    pub input_per_million: f64,
    /// Cost per 1M output tokens.
    pub output_per_million: f64,
    /// Cost per 1M cached input tokens (if supported).
    pub cache_read_per_million: Option<f64>,
    /// Cost per 1M tokens for cache creation (if supported).
    pub cache_write_per_million: Option<f64>,
}

impl ModelPricing {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
            cache_read_per_million: None,
            cache_write_per_million: None,
        }
    }

    pub const fn with_cache(mut self, read: f64, write: f64) -> Self {
        self.cache_read_per_million = Some(read);
        self.cache_write_per_million = Some(write);
        self
    }

    /// Calculate cost for given usage.
    pub fn calculate(&self, usage: &Usage) -> Cost {
        let input_cost = (f64::from(usage.input_tokens) / 1_000_000.0) * self.input_per_million;
        let output_cost = (f64::from(usage.output_tokens) / 1_000_000.0) * self.output_per_million;

        let cache_read_cost = self.cache_read_per_million.map_or(0.0, |rate| {
            (f64::from(usage.cache_read_input_tokens) / 1_000_000.0) * rate
        });

        let cache_write_cost = self.cache_write_per_million.map_or(0.0, |rate| {
            (f64::from(usage.cache_creation_input_tokens) / 1_000_000.0) * rate
        });

        Cost {
            input_cost,
            output_cost,
            cache_read_cost,
            cache_write_cost,
        }
    }
}

/// Registry of model pricing.
pub struct PricingRegistry {
    prices: HashMap<String, ModelPricing>,
}

impl PricingRegistry {
    /// Create a new registry with default pricing.
    pub fn new() -> Self {
        let mut prices = HashMap::new();

        // Cerebras pricing (as of 2024)
        prices.insert(
            "cerebras/llama3.1-8b".to_string(),
            ModelPricing::new(0.10, 0.10),
        );
        prices.insert(
            "cerebras/llama3.1-70b".to_string(),
            ModelPricing::new(0.60, 0.60),
        );
        prices.insert(
            "cerebras/llama-3.3-70b".to_string(),
            ModelPricing::new(0.60, 0.60),
        );

        // Gemini pricing (as of 2024)
        // Gemini 1.5 Flash
        prices.insert(
            "gemini/gemini-1.5-flash".to_string(),
            ModelPricing::new(0.075, 0.30).with_cache(0.01875, 0.075),
        );
        // Gemini 1.5 Pro
        prices.insert(
            "gemini/gemini-1.5-pro".to_string(),
            ModelPricing::new(1.25, 5.00).with_cache(0.3125, 1.25),
        );
        // Gemini 2.0 Flash
        prices.insert(
            "gemini/gemini-2.0-flash".to_string(),
            ModelPricing::new(0.10, 0.40),
        );
        // Gemini 2.0 Flash-Lite (free tier available)
        prices.insert(
            "gemini/gemini-2.0-flash-lite".to_string(),
            ModelPricing::new(0.075, 0.30),
        );

        // Claude pricing (as of 2024)
        prices.insert(
            "claude/claude-3-5-sonnet-20241022".to_string(),
            ModelPricing::new(3.00, 15.00).with_cache(0.30, 3.75),
        );
        prices.insert(
            "claude/claude-3-5-haiku-20241022".to_string(),
            ModelPricing::new(0.80, 4.00).with_cache(0.08, 1.00),
        );
        prices.insert(
            "claude/claude-3-opus-20240229".to_string(),
            ModelPricing::new(15.00, 75.00).with_cache(1.50, 18.75),
        );
        prices.insert(
            "claude/claude-3-haiku-20240307".to_string(),
            ModelPricing::new(0.25, 1.25).with_cache(0.03, 0.30),
        );

        // OpenAI pricing (as of 2024)
        prices.insert(
            "openai/gpt-4o".to_string(),
            ModelPricing::new(2.50, 10.00).with_cache(1.25, 2.50),
        );
        prices.insert(
            "openai/gpt-4o-mini".to_string(),
            ModelPricing::new(0.15, 0.60).with_cache(0.075, 0.15),
        );
        prices.insert(
            "openai/o1".to_string(),
            ModelPricing::new(15.00, 60.00).with_cache(7.50, 15.00),
        );
        prices.insert(
            "openai/o1-mini".to_string(),
            ModelPricing::new(3.00, 12.00).with_cache(1.50, 3.00),
        );

        Self { prices }
    }

    /// Get pricing for a model.
    pub fn get(&self, model: &str) -> Option<&ModelPricing> {
        self.prices.get(model)
    }

    /// Calculate cost for a model and usage.
    pub fn calculate_cost(&self, model: &str, usage: &Usage) -> Option<Cost> {
        self.get(model).map(|p| p.calculate(usage))
    }

    /// Add or update pricing for a model.
    pub fn set(&mut self, model: impl Into<String>, pricing: ModelPricing) {
        self.prices.insert(model.into(), pricing);
    }
}

impl Default for PricingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Track cumulative costs across multiple requests.
#[derive(Debug, Clone, Default)]
pub struct CostTracker {
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cache_read_tokens: u64,
    total_cache_write_tokens: u64,
    total_cost: f64,
    request_count: u32,
}

impl CostTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record usage and cost from a request.
    pub fn record(&mut self, usage: &Usage, cost: Option<&Cost>) {
        self.total_input_tokens += u64::from(usage.input_tokens);
        self.total_output_tokens += u64::from(usage.output_tokens);
        self.total_cache_read_tokens += u64::from(usage.cache_read_input_tokens);
        self.total_cache_write_tokens += u64::from(usage.cache_creation_input_tokens);
        if let Some(c) = cost {
            self.total_cost += c.total();
        }
        self.request_count += 1;
    }

    /// Get total input tokens.
    pub fn input_tokens(&self) -> u64 {
        self.total_input_tokens
    }

    /// Get total output tokens.
    pub fn output_tokens(&self) -> u64 {
        self.total_output_tokens
    }

    /// Get total cost in USD.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }

    /// Get number of requests tracked.
    pub fn request_count(&self) -> u32 {
        self.request_count
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_calculation() {
        let pricing = ModelPricing::new(1.0, 2.0); // $1/M input, $2/M output
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            ..Default::default()
        };

        let cost = pricing.calculate(&usage);
        assert!((cost.input_cost - 0.001).abs() < 1e-10);
        assert!((cost.output_cost - 0.001).abs() < 1e-10);
        assert!((cost.total() - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_cache_cost() {
        let pricing = ModelPricing::new(1.0, 2.0).with_cache(0.25, 1.0);
        let usage = Usage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_read_input_tokens: 2000,
            cache_creation_input_tokens: 500,
        };

        let cost = pricing.calculate(&usage);
        assert!(cost.cache_read_cost > 0.0);
        assert!(cost.cache_write_cost > 0.0);
    }

    #[test]
    fn test_registry() {
        let registry = PricingRegistry::new();
        assert!(registry.get("cerebras/llama3.1-70b").is_some());
        assert!(registry.get("gemini/gemini-1.5-pro").is_some());
    }

    #[test]
    fn test_cost_tracker() {
        let mut tracker = CostTracker::new();
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        let cost = Cost {
            input_cost: 0.001,
            output_cost: 0.002,
            ..Default::default()
        };

        tracker.record(&usage, Some(&cost));
        assert_eq!(tracker.input_tokens(), 100);
        assert_eq!(tracker.output_tokens(), 50);
        assert!((tracker.total_cost() - 0.003).abs() < 1e-10);
        assert_eq!(tracker.request_count(), 1);
    }
}
