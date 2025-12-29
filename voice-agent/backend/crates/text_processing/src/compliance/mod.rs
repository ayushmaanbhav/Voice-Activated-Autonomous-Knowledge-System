//! Compliance checking module
//!
//! Ensures banking regulatory compliance for agent responses.

mod checker;
mod rules;

pub use checker::RuleBasedComplianceChecker;
pub use rules::{ComplianceRules, load_rules, default_rules};

use voice_agent_core::ComplianceChecker;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Compliance checking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Which provider to use
    pub provider: ComplianceProvider,
    /// Rules file path (optional)
    #[serde(default)]
    pub rules_file: Option<String>,
    /// Strict mode (block on any violation)
    #[serde(default)]
    pub strict_mode: bool,
}

/// Compliance checking providers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ComplianceProvider {
    /// Rule-based checking
    #[default]
    RuleBased,
    /// Disabled
    Disabled,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            provider: ComplianceProvider::RuleBased,
            rules_file: None,
            strict_mode: false,
        }
    }
}

/// Create compliance checker based on config
pub fn create_checker(config: &ComplianceConfig) -> Arc<dyn ComplianceChecker> {
    match config.provider {
        ComplianceProvider::RuleBased => {
            let rules = if let Some(path) = &config.rules_file {
                match load_rules(path) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!("Failed to load rules from {}: {}, using defaults", path, e);
                        default_rules()
                    }
                }
            } else {
                default_rules()
            };
            Arc::new(RuleBasedComplianceChecker::new(rules, config.strict_mode))
        }
        ComplianceProvider::Disabled => Arc::new(NoopChecker),
    }
}

/// No-op checker
struct NoopChecker;

#[async_trait::async_trait]
impl ComplianceChecker for NoopChecker {
    async fn check(&self, _text: &str) -> voice_agent_core::Result<voice_agent_core::ComplianceResult> {
        Ok(voice_agent_core::ComplianceResult::compliant())
    }

    async fn make_compliant(&self, text: &str) -> voice_agent_core::Result<String> {
        Ok(text.to_string())
    }

    fn rules_version(&self) -> &str {
        "noop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ComplianceConfig::default();
        assert!(matches!(config.provider, ComplianceProvider::RuleBased));
        assert!(!config.strict_mode);
    }
}
