//! Intent Configuration
//!
//! Defines config-driven intent definitions for the voice agent.
//! Intents are loaded from domain config files instead of being hardcoded.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Intents configuration loaded from intents.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentsConfig {
    /// Intent definitions
    #[serde(default)]
    pub intents: Vec<IntentDefinition>,
    /// Default intent when none matches
    #[serde(default = "default_intent")]
    pub default_intent: String,
    /// Minimum confidence threshold
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
}

fn default_intent() -> String {
    "unknown".to_string()
}

fn default_min_confidence() -> f32 {
    0.3
}

impl Default for IntentsConfig {
    fn default() -> Self {
        Self {
            intents: Vec::new(),
            default_intent: default_intent(),
            min_confidence: default_min_confidence(),
        }
    }
}

impl IntentsConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, IntentsConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            IntentsConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content).map_err(|e| IntentsConfigError::ParseError(e.to_string()))
    }

    /// Get an intent definition by name
    pub fn get_intent(&self, name: &str) -> Option<&IntentDefinition> {
        self.intents.iter().find(|i| i.name == name)
    }

    /// Get all intent names
    pub fn intent_names(&self) -> Vec<&str> {
        self.intents.iter().map(|i| i.name.as_str()).collect()
    }

    /// Check if an intent exists
    pub fn has_intent(&self, name: &str) -> bool {
        self.intents.iter().any(|i| i.name == name)
    }

    /// Get intents that require specific slots
    pub fn intents_requiring_slot(&self, slot: &str) -> Vec<&str> {
        self.intents
            .iter()
            .filter(|i| i.required_slots.iter().any(|s| s == slot))
            .map(|i| i.name.as_str())
            .collect()
    }
}

/// Single intent definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDefinition {
    /// Intent name (identifier)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Required slots for this intent
    #[serde(default)]
    pub required_slots: Vec<String>,
    /// Optional slots for this intent
    #[serde(default)]
    pub optional_slots: Vec<String>,
    /// Example utterances for training/matching
    #[serde(default)]
    pub examples: Vec<String>,
}

impl IntentDefinition {
    /// Check if all required slots are present
    pub fn has_required_slots(&self, available_slots: &[&str]) -> bool {
        self.required_slots.iter().all(|s| available_slots.contains(&s.as_str()))
    }

    /// Get all slots (required + optional)
    pub fn all_slots(&self) -> Vec<&str> {
        self.required_slots
            .iter()
            .chain(self.optional_slots.iter())
            .map(|s| s.as_str())
            .collect()
    }
}

/// Errors when loading intents configuration
#[derive(Debug)]
pub enum IntentsConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for IntentsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Intents config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse intents config: {}", err),
        }
    }
}

impl std::error::Error for IntentsConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_config_deserialization() {
        let yaml = r#"
intents:
  - name: eligibility_check
    description: "Check eligibility"
    required_slots:
      - asset_quantity
    optional_slots:
      - asset_quality
    examples:
      - "Am I eligible"
      - "Can I get approved"
default_intent: unknown
min_confidence: 0.4
"#;
        let config: IntentsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.intents.len(), 1);
        assert_eq!(config.default_intent, "unknown");
        assert_eq!(config.min_confidence, 0.4);

        let intent = config.get_intent("eligibility_check").unwrap();
        assert_eq!(intent.required_slots, vec!["asset_quantity"]);
        assert_eq!(intent.examples.len(), 2);
    }

    #[test]
    fn test_has_required_slots() {
        let intent = IntentDefinition {
            name: "test".to_string(),
            description: "Test intent".to_string(),
            required_slots: vec!["slot_a".to_string(), "slot_b".to_string()],
            optional_slots: vec![],
            examples: vec![],
        };

        assert!(intent.has_required_slots(&["slot_a", "slot_b", "slot_c"]));
        assert!(!intent.has_required_slots(&["slot_a"]));
        assert!(!intent.has_required_slots(&[]));
    }
}
