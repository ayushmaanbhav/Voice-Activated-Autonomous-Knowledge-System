//! Slot Schema Configuration
//!
//! Defines config-driven slot definitions for dialogue state tracking.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Slot schema loaded from slots.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotsConfig {
    /// Slot definitions keyed by slot name
    #[serde(default)]
    pub slots: HashMap<String, SlotDefinition>,
    /// Goal definitions keyed by goal name
    #[serde(default)]
    pub goals: HashMap<String, GoalDefinition>,
    /// Intent to goal mapping
    #[serde(default)]
    pub intent_mapping: HashMap<String, Vec<String>>,
}

impl Default for SlotsConfig {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            goals: HashMap::new(),
            intent_mapping: HashMap::new(),
        }
    }
}

impl SlotsConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, SlotsConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            SlotsConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content).map_err(|e| SlotsConfigError::ParseError(e.to_string()))
    }

    /// Get a slot definition by name
    pub fn get_slot(&self, name: &str) -> Option<&SlotDefinition> {
        self.slots.get(name)
    }

    /// Get a goal definition by name
    pub fn get_goal(&self, name: &str) -> Option<&GoalDefinition> {
        self.goals.get(name)
    }

    /// Map an intent to a goal
    pub fn goal_for_intent(&self, intent: &str) -> Option<&str> {
        for (goal, intents) in &self.intent_mapping {
            if intents.iter().any(|i| i == intent) {
                return Some(goal);
            }
        }
        None
    }

    /// Get extraction patterns for a slot
    pub fn extraction_patterns(&self, slot_name: &str, language: &str) -> Vec<&str> {
        self.slots
            .get(slot_name)
            .and_then(|s| s.extraction_patterns.as_ref())
            .and_then(|p| p.get(language))
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get enum values for a slot
    pub fn enum_values(&self, slot_name: &str) -> Vec<&EnumValue> {
        self.slots
            .get(slot_name)
            .and_then(|s| s.values.as_ref())
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Get purity factor for a gold purity value
    pub fn purity_factor(&self, purity_id: &str) -> f64 {
        self.slots
            .get("gold_purity")
            .and_then(|s| s.values.as_ref())
            .and_then(|values| values.iter().find(|v| v.id == purity_id))
            .and_then(|v| v.purity_factor)
            .unwrap_or(1.0)
    }

    /// Get typical rate for a lender
    pub fn lender_rate(&self, lender_id: &str) -> Option<f64> {
        self.slots
            .get("current_lender")
            .and_then(|s| s.values.as_ref())
            .and_then(|values| values.iter().find(|v| v.id == lender_id))
            .and_then(|v| v.typical_rate)
    }

    /// Get unit conversion factor
    pub fn unit_conversion(&self, slot_name: &str, unit: &str) -> Option<f64> {
        self.slots
            .get(slot_name)
            .and_then(|s| s.unit_conversions.as_ref())
            .and_then(|c| c.get(unit))
            .copied()
    }
}

/// Definition for a single slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotDefinition {
    /// Slot type: string, number, enum, date
    #[serde(rename = "type")]
    pub slot_type: SlotType,
    /// Human-readable description
    #[serde(default)]
    pub description: String,
    /// Validation regex (for string/number types)
    #[serde(default)]
    pub validation: Option<String>,
    /// Minimum value (for number types)
    #[serde(default)]
    pub min: Option<f64>,
    /// Maximum value (for number types)
    #[serde(default)]
    pub max: Option<f64>,
    /// Enum values (for enum types)
    #[serde(default)]
    pub values: Option<Vec<EnumValue>>,
    /// Default value
    #[serde(default)]
    pub default: Option<String>,
    /// Extraction patterns by language
    #[serde(default)]
    pub extraction_patterns: Option<HashMap<String, Vec<String>>>,
    /// Unit conversions (e.g., tola -> grams)
    #[serde(default)]
    pub unit_conversions: Option<HashMap<String, f64>>,
}

/// Slot type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SlotType {
    String,
    Number,
    Enum,
    Date,
}

/// Enum value definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumValue {
    /// Unique identifier
    pub id: String,
    /// Display name
    pub display: String,
    /// Pattern strings for extraction
    #[serde(default)]
    pub patterns: Vec<String>,
    /// Purity factor (for gold_purity)
    #[serde(default)]
    pub purity_factor: Option<f64>,
    /// Typical rate (for lenders)
    #[serde(default)]
    pub typical_rate: Option<f64>,
}

/// Goal definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalDefinition {
    /// Description of the goal
    #[serde(default)]
    pub description: String,
    /// Required slots for this goal
    #[serde(default)]
    pub required_slots: Vec<String>,
    /// Optional slots that enhance the goal
    #[serde(default)]
    pub optional_slots: Vec<String>,
    /// Action to take when goal is complete
    #[serde(default)]
    pub completion_action: Option<String>,
}

/// Errors when loading slot configuration
#[derive(Debug)]
pub enum SlotsConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for SlotsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => write!(f, "Slots config not found at {}: {}", path, err),
            Self::ParseError(err) => write!(f, "Failed to parse slots config: {}", err),
        }
    }
}

impl std::error::Error for SlotsConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slot_type_deserialization() {
        let yaml = r#"
slots:
  test_string:
    type: string
    description: "A test string slot"
  test_number:
    type: number
    min: 0
    max: 100
  test_enum:
    type: enum
    values:
      - id: option1
        display: "Option 1"
        patterns: ["one", "first"]
"#;
        let config: SlotsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.slots.len(), 3);
        assert_eq!(config.slots["test_string"].slot_type, SlotType::String);
        assert_eq!(config.slots["test_number"].slot_type, SlotType::Number);
        assert_eq!(config.slots["test_enum"].slot_type, SlotType::Enum);
    }

    #[test]
    fn test_goal_deserialization() {
        let yaml = r#"
goals:
  test_goal:
    description: "A test goal"
    required_slots:
      - slot1
      - slot2
    optional_slots:
      - slot3
    completion_action: test_action
"#;
        let config: SlotsConfig = serde_yaml::from_str(yaml).unwrap();
        let goal = config.get_goal("test_goal").unwrap();
        assert_eq!(goal.required_slots, vec!["slot1", "slot2"]);
        assert_eq!(goal.optional_slots, vec!["slot3"]);
        assert_eq!(goal.completion_action, Some("test_action".to_string()));
    }

    #[test]
    fn test_intent_mapping() {
        let yaml = r#"
intent_mapping:
  goal_a:
    - intent1
    - intent2
  goal_b:
    - intent3
"#;
        let config: SlotsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.goal_for_intent("intent1"), Some("goal_a"));
        assert_eq!(config.goal_for_intent("intent2"), Some("goal_a"));
        assert_eq!(config.goal_for_intent("intent3"), Some("goal_b"));
        assert_eq!(config.goal_for_intent("unknown"), None);
    }

    #[test]
    fn test_unit_conversion() {
        let yaml = r#"
slots:
  weight:
    type: number
    unit_conversions:
      tola: 11.66
      oz: 31.1
"#;
        let config: SlotsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.unit_conversion("weight", "tola"), Some(11.66));
        assert_eq!(config.unit_conversion("weight", "oz"), Some(31.1));
        assert_eq!(config.unit_conversion("weight", "unknown"), None);
    }
}
