//! Grammar correction module
//!
//! Provides grammar correction that preserves domain-specific vocabulary.

mod llm_corrector;
mod noop;

pub use llm_corrector::LLMGrammarCorrector;
pub use noop::NoopCorrector;

use voice_agent_core::{GrammarCorrector, LanguageModel};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Grammar correction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarConfig {
    /// Which provider to use
    pub provider: GrammarProvider,
    /// Domain for vocabulary
    pub domain: String,
    /// LLM temperature
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Max tokens for correction
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_temperature() -> f32 {
    0.1
}

fn default_max_tokens() -> u32 {
    256
}

/// Grammar correction providers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum GrammarProvider {
    /// Use LLM for correction (P1 FIX: Now default when LLM available)
    #[default]
    Llm,
    /// Disabled (pass-through)
    Disabled,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        Self {
            // P1 FIX: Default to LLM-based grammar correction
            provider: GrammarProvider::Llm,
            domain: "gold_loan".to_string(),
            temperature: 0.1,
            max_tokens: 256,
        }
    }
}

/// Create grammar corrector based on config
pub fn create_corrector(
    config: &GrammarConfig,
    llm: Option<Arc<dyn LanguageModel>>,
) -> Arc<dyn GrammarCorrector> {
    match config.provider {
        GrammarProvider::Llm => {
            if let Some(llm) = llm {
                Arc::new(LLMGrammarCorrector::new(llm, &config.domain, config.temperature))
            } else {
                tracing::warn!("LLM not available, using noop corrector");
                Arc::new(NoopCorrector)
            }
        }
        GrammarProvider::Disabled => Arc::new(NoopCorrector),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GrammarConfig::default();
        // P1 FIX: Grammar is now enabled by default
        assert!(matches!(config.provider, GrammarProvider::Llm));
        assert_eq!(config.domain, "gold_loan");
    }
}
