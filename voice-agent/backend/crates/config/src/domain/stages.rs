//! Conversation Stages Configuration
//!
//! Defines config-driven stage definitions for conversation flow.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Stages configuration loaded from stages.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagesConfig {
    /// Initial stage when conversation begins
    #[serde(default = "default_initial_stage")]
    pub initial_stage: String,
    /// Stage definitions keyed by stage ID
    #[serde(default)]
    pub stages: HashMap<String, StageDefinition>,
    /// Transition triggers - patterns that suggest moving to a stage
    #[serde(default)]
    pub transition_triggers: HashMap<String, TransitionTrigger>,
    /// P16 FIX: Intent-based stage transitions
    /// Maps intent → current_stage → target_stage (or IntentTransition)
    #[serde(default)]
    pub intent_transitions: HashMap<String, HashMap<String, IntentTransitionTarget>>,
}

fn default_initial_stage() -> String {
    "greeting".to_string()
}

impl Default for StagesConfig {
    fn default() -> Self {
        Self {
            initial_stage: default_initial_stage(),
            stages: HashMap::new(),
            transition_triggers: HashMap::new(),
            intent_transitions: HashMap::new(),
        }
    }
}

/// P16 FIX: Target for intent-based transition
/// Can be either a simple stage name or a complex transition with conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IntentTransitionTarget {
    /// Simple target stage name
    Simple(String),
    /// Complex transition with conditions
    Complex(IntentTransition),
}

impl IntentTransitionTarget {
    /// Get the target stage name
    pub fn target(&self) -> &str {
        match self {
            Self::Simple(s) => s.as_str(),
            Self::Complex(t) => t.target.as_str(),
        }
    }

    /// Get minimum turns required (0 if not specified)
    pub fn min_turns(&self) -> usize {
        match self {
            Self::Simple(_) => 0,
            Self::Complex(t) => t.min_turns,
        }
    }
}

/// Complex intent transition with conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentTransition {
    /// Target stage
    pub target: String,
    /// Minimum turns in current stage before transition allowed
    #[serde(default)]
    pub min_turns: usize,
}

impl StagesConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, StagesConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            StagesConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content).map_err(|e| StagesConfigError::ParseError(e.to_string()))
    }

    /// Get a stage definition by ID
    pub fn get_stage(&self, stage_id: &str) -> Option<&StageDefinition> {
        self.stages.get(stage_id)
    }

    /// Get the initial stage definition
    pub fn get_initial_stage(&self) -> Option<&StageDefinition> {
        self.stages.get(&self.initial_stage)
    }

    /// Get valid transitions from a stage
    pub fn get_transitions(&self, stage_id: &str) -> Vec<&str> {
        self.stages
            .get(stage_id)
            .map(|s| s.transitions.iter().map(|t| t.as_str()).collect())
            .unwrap_or_default()
    }

    /// Check if a transition is valid
    pub fn is_valid_transition(&self, from: &str, to: &str) -> bool {
        self.stages
            .get(from)
            .map(|s| s.transitions.contains(&to.to_string()))
            .unwrap_or(false)
    }

    /// Get transition trigger for a stage
    pub fn get_trigger(&self, stage_id: &str) -> Option<&TransitionTrigger> {
        self.transition_triggers.get(stage_id)
    }

    /// Get all stage IDs
    pub fn stage_ids(&self) -> Vec<&str> {
        self.stages.keys().map(|s| s.as_str()).collect()
    }

    /// Get stage guidance text
    pub fn get_guidance(&self, stage_id: &str) -> Option<&str> {
        self.stages.get(stage_id).map(|s| s.guidance.as_str())
    }

    /// Get suggested questions for a stage
    pub fn get_suggested_questions(&self, stage_id: &str) -> Vec<&str> {
        self.stages
            .get(stage_id)
            .map(|s| s.suggested_questions.iter().map(|q| q.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get context budget for a stage
    pub fn get_context_budget(&self, stage_id: &str) -> usize {
        self.stages
            .get(stage_id)
            .map(|s| s.context_budget_tokens)
            .unwrap_or(2048)
    }

    /// Get RAG context fraction for a stage
    pub fn get_rag_fraction(&self, stage_id: &str) -> f32 {
        self.stages
            .get(stage_id)
            .map(|s| s.rag_context_fraction)
            .unwrap_or(0.0)
    }

    /// P16 FIX: Get intent-based transition target
    ///
    /// Returns the target stage for a given intent and current stage, if defined.
    /// Also returns the minimum turns required before transition is allowed.
    pub fn get_intent_transition(
        &self,
        intent: &str,
        current_stage: &str,
    ) -> Option<(&str, usize)> {
        self.intent_transitions
            .get(intent)
            .and_then(|stage_map| stage_map.get(current_stage))
            .map(|target| (target.target(), target.min_turns()))
    }

    /// P16 FIX: Check if an intent transition is valid
    ///
    /// Returns true if the intent can trigger a transition from current_stage,
    /// and the min_turns requirement is satisfied.
    pub fn can_transition_on_intent(
        &self,
        intent: &str,
        current_stage: &str,
        current_turns: usize,
    ) -> Option<&str> {
        self.get_intent_transition(intent, current_stage)
            .and_then(|(target, min_turns)| {
                if current_turns >= min_turns {
                    Some(target)
                } else {
                    None
                }
            })
    }
}

/// Definition for a single conversation stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageDefinition {
    /// Human-readable display name
    #[serde(default)]
    pub display_name: String,
    /// Description of the stage purpose
    #[serde(default)]
    pub description: String,
    /// Agent guidance for this stage
    #[serde(default)]
    pub guidance: String,
    /// Suggested questions to ask
    #[serde(default)]
    pub suggested_questions: Vec<String>,
    /// Token budget for context in this stage
    #[serde(default = "default_context_budget")]
    pub context_budget_tokens: usize,
    /// Fraction of context budget for RAG (0.0-1.0)
    #[serde(default)]
    pub rag_context_fraction: f32,
    /// Number of conversation history turns to keep
    #[serde(default = "default_history_turns")]
    pub history_turns_to_keep: usize,
    /// Valid next stages (transitions)
    #[serde(default)]
    pub transitions: Vec<String>,
    /// Requirements to stay in or leave this stage
    #[serde(default)]
    pub requirements: StageRequirements,
}

fn default_context_budget() -> usize {
    2048
}

fn default_history_turns() -> usize {
    3
}

/// Requirements for stage transitions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageRequirements {
    /// Minimum turns before transitioning
    #[serde(default)]
    pub min_turns: usize,
    /// Required info slots to collect
    #[serde(default)]
    pub required_info: Vec<String>,
    /// Required intents to detect
    #[serde(default)]
    pub required_intents: Vec<String>,
}

/// Transition trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransitionTrigger {
    /// Intents that trigger transition to this stage
    #[serde(default)]
    pub intents: Vec<String>,
    /// Regex patterns that trigger transition
    #[serde(default)]
    pub patterns: Vec<String>,
}

/// Errors when loading stages configuration
#[derive(Debug)]
pub enum StagesConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for StagesConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Stages config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse stages config: {}", err),
        }
    }
}

impl std::error::Error for StagesConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_deserialization() {
        let yaml = r#"
initial_stage: greeting

stages:
  greeting:
    display_name: "Greeting"
    description: "Initial greeting"
    guidance: "Warmly greet the customer"
    suggested_questions:
      - "How are you today?"
    context_budget_tokens: 1024
    rag_context_fraction: 0.0
    transitions:
      - discovery
      - farewell
    requirements:
      min_turns: 1

  discovery:
    display_name: "Discovery"
    guidance: "Understand customer needs"
    transitions:
      - qualification
"#;
        let config: StagesConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.initial_stage, "greeting");
        assert_eq!(config.stages.len(), 2);

        let greeting = config.get_stage("greeting").unwrap();
        assert_eq!(greeting.display_name, "Greeting");
        assert_eq!(greeting.context_budget_tokens, 1024);
        assert_eq!(greeting.transitions, vec!["discovery", "farewell"]);
    }

    #[test]
    fn test_transition_validation() {
        let yaml = r#"
stages:
  a:
    transitions: [b, c]
  b:
    transitions: [a]
  c:
    transitions: []
"#;
        let config: StagesConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(config.is_valid_transition("a", "b"));
        assert!(config.is_valid_transition("a", "c"));
        assert!(!config.is_valid_transition("a", "d"));
        assert!(config.is_valid_transition("b", "a"));
        assert!(!config.is_valid_transition("c", "a"));
    }

    #[test]
    fn test_transition_triggers() {
        let yaml = r#"
transition_triggers:
  discovery:
    intents:
      - loan_inquiry
    patterns:
      - "(?i)current.*loan"
"#;
        let config: StagesConfig = serde_yaml::from_str(yaml).unwrap();
        let trigger = config.get_trigger("discovery").unwrap();
        assert_eq!(trigger.intents, vec!["loan_inquiry"]);
        assert_eq!(trigger.patterns.len(), 1);
    }

    #[test]
    fn test_intent_transitions() {
        let yaml = r#"
intent_transitions:
  greeting:
    greeting:
      target: discovery
      min_turns: 1
  loan_inquiry:
    greeting: discovery
    discovery: qualification
  affirmative:
    greeting: discovery
    discovery: qualification
"#;
        let config: StagesConfig = serde_yaml::from_str(yaml).unwrap();

        // Test simple transition
        let (target, min_turns) = config.get_intent_transition("loan_inquiry", "greeting").unwrap();
        assert_eq!(target, "discovery");
        assert_eq!(min_turns, 0);

        // Test complex transition with min_turns
        let (target, min_turns) = config.get_intent_transition("greeting", "greeting").unwrap();
        assert_eq!(target, "discovery");
        assert_eq!(min_turns, 1);

        // Test can_transition_on_intent with min_turns check
        assert!(config.can_transition_on_intent("greeting", "greeting", 0).is_none());
        assert_eq!(config.can_transition_on_intent("greeting", "greeting", 1), Some("discovery"));

        // Test non-existent transition
        assert!(config.get_intent_transition("loan_inquiry", "closing").is_none());
    }
}
