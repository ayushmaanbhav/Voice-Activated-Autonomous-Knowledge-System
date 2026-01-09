//! Lead Scoring Configuration
//!
//! Defines config-driven scoring rules for lead qualification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Scoring configuration loaded from scoring.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Qualification thresholds (score ranges)
    #[serde(default)]
    pub qualification_thresholds: QualificationThresholds,
    /// Trust level scores
    #[serde(default)]
    pub trust_scores: TrustScores,
    /// Escalation trigger thresholds
    #[serde(default)]
    pub escalation: EscalationConfig,
    /// Category weights
    #[serde(default)]
    pub weights: CategoryWeights,
    /// Urgency scoring config
    #[serde(default)]
    pub urgency: UrgencyScoringConfig,
    /// Engagement scoring config
    #[serde(default)]
    pub engagement: EngagementScoringConfig,
    /// Information completeness scoring config
    #[serde(default)]
    pub information: InformationScoringConfig,
    /// Intent strength scoring config
    #[serde(default)]
    pub intent: IntentScoringConfig,
    /// Penalty scores
    #[serde(default)]
    pub penalties: PenaltyConfig,
    /// Conversion probability multipliers
    #[serde(default)]
    pub conversion_multipliers: ConversionMultipliers,
    /// P16 FIX: Intent to signal mappings (domain-agnostic)
    #[serde(default)]
    pub intent_signal_mappings: HashMap<String, IntentSignalMapping>,
    /// P16 FIX: Slot to signal mappings (domain-agnostic)
    #[serde(default)]
    pub slot_signal_mappings: HashMap<String, SlotSignalMapping>,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            qualification_thresholds: QualificationThresholds::default(),
            trust_scores: TrustScores::default(),
            escalation: EscalationConfig::default(),
            weights: CategoryWeights::default(),
            urgency: UrgencyScoringConfig::default(),
            engagement: EngagementScoringConfig::default(),
            information: InformationScoringConfig::default(),
            intent: IntentScoringConfig::default(),
            penalties: PenaltyConfig::default(),
            conversion_multipliers: ConversionMultipliers::default(),
            intent_signal_mappings: HashMap::new(),
            slot_signal_mappings: HashMap::new(),
        }
    }
}

impl ScoringConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ScoringConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ScoringConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content).map_err(|e| ScoringConfigError::ParseError(e.to_string()))
    }

    /// Get qualification level from score
    pub fn qualification_level(&self, score: u32) -> &'static str {
        if score >= self.qualification_thresholds.qualified {
            "qualified"
        } else if score >= self.qualification_thresholds.hot {
            "hot"
        } else if score >= self.qualification_thresholds.warm {
            "warm"
        } else {
            "cold"
        }
    }

    /// Get urgency keywords for a language
    pub fn urgency_keywords(&self, language: &str) -> Vec<&str> {
        self.urgency
            .keywords
            .get(language)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get trust score for a level
    pub fn trust_score(&self, level: &str) -> u32 {
        match level.to_lowercase().as_str() {
            "unknown" => self.trust_scores.unknown,
            "low" => self.trust_scores.low,
            "medium" => self.trust_scores.medium,
            "high" => self.trust_scores.high,
            _ => self.trust_scores.unknown,
        }
    }

    /// P16 FIX: Get signal mapping for an intent
    pub fn get_intent_signals(&self, intent: &str) -> Option<&IntentSignalMapping> {
        self.intent_signal_mappings.get(intent)
    }

    /// P16 FIX: Get signal for a slot
    pub fn get_slot_signal(&self, slot: &str) -> Option<&str> {
        self.slot_signal_mappings.get(slot).map(|m| m.signal.as_str())
    }

    /// P16 FIX: Get all urgency keywords (all languages combined)
    pub fn all_urgency_keywords(&self) -> Vec<&str> {
        self.urgency
            .keywords
            .values()
            .flat_map(|v| v.iter().map(|s| s.as_str()))
            .collect()
    }
}

/// Qualification level thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualificationThresholds {
    pub cold: u32,
    pub warm: u32,
    pub hot: u32,
    pub qualified: u32,
}

impl Default for QualificationThresholds {
    fn default() -> Self {
        Self {
            cold: 0,
            warm: 30,
            hot: 60,
            qualified: 80,
        }
    }
}

/// Trust level scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScores {
    pub unknown: u32,
    pub low: u32,
    pub medium: u32,
    pub high: u32,
}

impl Default for TrustScores {
    fn default() -> Self {
        Self {
            unknown: 0,
            low: 5,
            medium: 10,
            high: 15,
        }
    }
}

/// Escalation trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationConfig {
    pub max_objections: u32,
    pub max_stalled_turns: u32,
    pub high_value_threshold: f64,
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            max_objections: 3,
            max_stalled_turns: 5,
            high_value_threshold: 1_000_000.0,
        }
    }
}

/// Category weights for scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryWeights {
    pub urgency: f32,
    pub engagement: f32,
    pub information: f32,
    pub intent: f32,
}

impl Default for CategoryWeights {
    fn default() -> Self {
        Self {
            urgency: 1.0,
            engagement: 1.0,
            information: 1.0,
            intent: 1.0,
        }
    }
}

/// Urgency scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrgencyScoringConfig {
    pub has_signal_score: u32,
    pub keyword_score: u32,
    pub max_keywords: u32,
    pub max_score: u32,
    #[serde(default)]
    pub keywords: HashMap<String, Vec<String>>,
}

impl Default for UrgencyScoringConfig {
    fn default() -> Self {
        let mut keywords = HashMap::new();
        keywords.insert(
            "en".to_string(),
            vec![
                "urgent".to_string(),
                "urgently".to_string(),
                "immediately".to_string(),
                "today".to_string(),
                "now".to_string(),
                "asap".to_string(),
                "emergency".to_string(),
            ],
        );
        keywords.insert(
            "hi".to_string(),
            vec![
                "jaldi".to_string(),
                "abhi".to_string(),
                "turant".to_string(),
                "aaj".to_string(),
                "foran".to_string(),
            ],
        );

        Self {
            has_signal_score: 10,
            keyword_score: 5,
            max_keywords: 3,
            max_score: 25,
            keywords,
        }
    }
}

/// Engagement scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngagementScoringConfig {
    pub max_score: u32,
    pub per_turn_score: u32,
    pub max_turns: u32,
    pub per_question_score: u32,
    pub max_questions: u32,
    pub rates_inquiry_score: u32,
    pub comparison_score: u32,
}

impl Default for EngagementScoringConfig {
    fn default() -> Self {
        Self {
            max_score: 25,
            per_turn_score: 3,
            max_turns: 5,
            per_question_score: 2,
            max_questions: 3,
            rates_inquiry_score: 3,
            comparison_score: 3,
        }
    }
}

/// Information completeness scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationScoringConfig {
    pub max_score: u32,
    pub contact_info_score: u32,
    pub gold_details_score: u32,
    pub loan_amount_score: u32,
    pub specific_requirements_score: u32,
}

impl Default for InformationScoringConfig {
    fn default() -> Self {
        Self {
            max_score: 25,
            contact_info_score: 8,
            gold_details_score: 8,
            loan_amount_score: 5,
            specific_requirements_score: 4,
        }
    }
}

/// Intent strength scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentScoringConfig {
    pub max_score: u32,
    pub intent_to_proceed_score: u32,
    pub callback_request_score: u32,
    pub branch_visit_score: u32,
}

impl Default for IntentScoringConfig {
    fn default() -> Self {
        Self {
            max_score: 25,
            intent_to_proceed_score: 15,
            callback_request_score: 5,
            branch_visit_score: 8,
        }
    }
}

/// Penalty scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyConfig {
    pub disinterest: i32,
    pub competitor_preference: i32,
    pub human_request: i32,
    pub per_unresolved_objection: i32,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            disinterest: -15,
            competitor_preference: -10,
            human_request: -5,
            per_unresolved_objection: -3,
        }
    }
}

/// Conversion probability multipliers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionMultipliers {
    pub unqualified: f32,
    pub mql: f32,
    pub sql: f32,
    pub intent_to_proceed: f32,
    pub disinterest: f32,
    pub max_probability: f32,
}

impl Default for ConversionMultipliers {
    fn default() -> Self {
        Self {
            unqualified: 0.5,
            mql: 0.8,
            sql: 1.2,
            intent_to_proceed: 1.2,
            disinterest: 0.3,
            max_probability: 0.95,
        }
    }
}

// =============================================================================
// P16 FIX: Signal Mapping Types (Domain-Agnostic)
// =============================================================================

/// P16 FIX: Intent to signal mapping
///
/// Maps a detected intent to the signals that should be updated.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntentSignalMapping {
    /// Signals to set/increment when this intent is detected
    #[serde(default)]
    pub signals: Vec<String>,
    /// Optional slot checks - if specific slots are present, set additional signals
    #[serde(default)]
    pub slot_checks: Vec<SlotCheck>,
}

/// P16 FIX: Slot check within intent mapping
///
/// If any of the specified slots are present, set the specified signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotCheck {
    /// Slot names to check for (any match triggers)
    pub slots: Vec<String>,
    /// Signals to set if any slot is present
    pub signals: Vec<String>,
}

/// P16 FIX: Slot to signal mapping
///
/// Maps a slot name to the signal that should be set when the slot is present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotSignalMapping {
    /// Signal to set when this slot is present
    pub signal: String,
}

/// Errors when loading scoring configuration
#[derive(Debug)]
pub enum ScoringConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for ScoringConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Scoring config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse scoring config: {}", err),
        }
    }
}

impl std::error::Error for ScoringConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thresholds() {
        let config = ScoringConfig::default();
        assert_eq!(config.qualification_level(25), "cold");
        assert_eq!(config.qualification_level(45), "warm");
        assert_eq!(config.qualification_level(70), "hot");
        assert_eq!(config.qualification_level(90), "qualified");
    }

    #[test]
    fn test_scoring_config_deserialization() {
        let yaml = r#"
qualification_thresholds:
  cold: 0
  warm: 25
  hot: 50
  qualified: 75

weights:
  urgency: 1.5
  engagement: 1.0
  information: 1.0
  intent: 1.2

urgency:
  has_signal_score: 15
  keyword_score: 5
  max_keywords: 5
  max_score: 30
  keywords:
    en:
      - urgent
      - now
"#;
        let config: ScoringConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.qualification_thresholds.warm, 25);
        assert_eq!(config.weights.urgency, 1.5);
        assert_eq!(config.urgency.has_signal_score, 15);
        assert_eq!(config.urgency_keywords("en").len(), 2);
    }

    #[test]
    fn test_trust_score() {
        let config = ScoringConfig::default();
        assert_eq!(config.trust_score("unknown"), 0);
        assert_eq!(config.trust_score("low"), 5);
        assert_eq!(config.trust_score("medium"), 10);
        assert_eq!(config.trust_score("high"), 15);
        assert_eq!(config.trust_score("invalid"), 0);
    }
}
