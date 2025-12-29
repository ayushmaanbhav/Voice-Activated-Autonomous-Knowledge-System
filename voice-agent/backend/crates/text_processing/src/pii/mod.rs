//! PII detection and redaction module
//!
//! Supports India-specific PII types: Aadhaar, PAN, IFSC, etc.
//! P3 FIX: Added NER-based detection for names and addresses.

mod patterns;
mod detector;
mod ner;

pub use patterns::IndianPIIPatterns;
pub use detector::HybridPIIDetector;
pub use ner::NameAddressDetector;

use voice_agent_core::{PIIRedactor, PIIType, RedactionStrategy};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// PII detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIIConfig {
    /// Which provider to use
    pub provider: PIIProvider,
    /// Entity types to detect
    #[serde(default = "default_entities")]
    pub entities: Vec<String>,
    /// Redaction strategy
    #[serde(default)]
    pub strategy: RedactionStrategyConfig,
}

fn default_entities() -> Vec<String> {
    vec![
        "Aadhaar".to_string(),
        "PAN".to_string(),
        "PhoneNumber".to_string(),
        "Email".to_string(),
        "BankAccount".to_string(),
        "IFSC".to_string(),
    ]
}

/// Redaction strategy configuration (serializable wrapper)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RedactionStrategyConfig {
    Mask,
    TypeMask,
    PartialMask {
        #[serde(default = "default_visible_start")]
        visible_start: usize,
        #[serde(default = "default_visible_end")]
        visible_end: usize,
    },
    Remove,
    Hash,
}

impl Default for RedactionStrategyConfig {
    fn default() -> Self {
        Self::PartialMask {
            visible_start: 2,
            visible_end: 2,
        }
    }
}

fn default_visible_start() -> usize {
    2
}

fn default_visible_end() -> usize {
    2
}

impl From<RedactionStrategyConfig> for RedactionStrategy {
    fn from(config: RedactionStrategyConfig) -> Self {
        match config {
            RedactionStrategyConfig::Mask => RedactionStrategy::Mask,
            RedactionStrategyConfig::TypeMask => RedactionStrategy::TypeMask,
            RedactionStrategyConfig::PartialMask { visible_start, visible_end } => {
                RedactionStrategy::PartialMask { visible_start, visible_end }
            }
            RedactionStrategyConfig::Remove => RedactionStrategy::Remove,
            RedactionStrategyConfig::Hash => RedactionStrategy::Hash,
        }
    }
}

/// PII detection providers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PIIProvider {
    /// Hybrid (regex + NER)
    Hybrid,
    /// Regex only (faster)
    #[default]
    Regex,
    /// Disabled
    Disabled,
}

impl Default for PIIConfig {
    fn default() -> Self {
        Self {
            provider: PIIProvider::Regex,
            entities: default_entities(),
            strategy: RedactionStrategyConfig::default(),
        }
    }
}

/// Create PII detector based on config
pub fn create_detector(config: &PIIConfig) -> Arc<dyn PIIRedactor> {
    match config.provider {
        PIIProvider::Hybrid => Arc::new(HybridPIIDetector::new(&config.entities, true)),
        PIIProvider::Regex => Arc::new(HybridPIIDetector::new(&config.entities, false)),
        PIIProvider::Disabled => Arc::new(NoopDetector),
    }
}

/// No-op detector
struct NoopDetector;

#[async_trait::async_trait]
impl PIIRedactor for NoopDetector {
    async fn detect(&self, _text: &str) -> voice_agent_core::Result<Vec<voice_agent_core::PIIEntity>> {
        Ok(vec![])
    }

    async fn redact(&self, text: &str, _strategy: &RedactionStrategy) -> voice_agent_core::Result<String> {
        Ok(text.to_string())
    }

    fn supported_types(&self) -> &[PIIType] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PIIConfig::default();
        assert!(matches!(config.provider, PIIProvider::Regex));
        assert!(config.entities.contains(&"Aadhaar".to_string()));
    }
}
