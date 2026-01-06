//! Domain Abstraction Traits
//!
//! Generic traits for domain-agnostic agent behavior.
//! These traits define interfaces without any specific domain knowledge
//! (e.g., no "gold loan" terminology in core).
//!
//! Domain-specific implementations are provided by the config crate
//! based on YAML configuration files.

use std::collections::{HashMap, HashSet};

/// Generic identifiers used across domains
pub type StageId = String;
pub type SegmentId = String;
pub type ObjectionId = String;
pub type FeatureId = String;
pub type SlotId = String;
pub type ToolId = String;

/// Generic customer signals for segment matching
/// Domain config defines which keys are relevant (e.g., "gold_weight_grams")
#[derive(Debug, Clone, Default)]
pub struct CustomerSignals {
    /// Numeric values extracted from conversation
    pub numeric_values: HashMap<String, f64>,
    /// Text values extracted from conversation
    pub text_values: HashMap<String, String>,
    /// Boolean flags detected
    pub flags: HashSet<String>,
}

impl CustomerSignals {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_numeric(&mut self, key: impl Into<String>, value: f64) {
        self.numeric_values.insert(key.into(), value);
    }

    pub fn set_text(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.text_values.insert(key.into(), value.into());
    }

    pub fn set_flag(&mut self, flag: impl Into<String>) {
        self.flags.insert(flag.into());
    }

    pub fn get_numeric(&self, key: &str) -> Option<f64> {
        self.numeric_values.get(key).copied()
    }

    pub fn get_text(&self, key: &str) -> Option<&str> {
        self.text_values.get(key).map(|s| s.as_str())
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }
}

/// Pattern type for matching text
#[derive(Debug, Clone)]
pub enum PatternType {
    /// Regular expression pattern
    Regex,
    /// Exact keyword match
    Keyword,
    /// Fuzzy/phonetic match
    Fuzzy,
}

/// Generic pattern for matching user input
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Pattern value (regex string, keyword, etc.)
    pub value: String,
    /// Language code (e.g., "en", "hi")
    pub language: Option<String>,
}

/// Response template with variable placeholders
#[derive(Debug, Clone)]
pub struct ResponseTemplate {
    /// Template text with {placeholders}
    pub template: String,
    /// Language code
    pub language: String,
}

/// Generic customer segment trait
/// Implementations define segment-specific matching logic
pub trait CustomerSegment: Send + Sync {
    /// Unique segment identifier
    fn segment_id(&self) -> &str;

    /// Check if customer signals match this segment
    fn matches(&self, signals: &CustomerSignals) -> bool;

    /// Features to highlight for this segment
    fn priority_features(&self) -> &[FeatureId];

    /// Value propositions for this segment
    fn value_propositions(&self) -> &[String];
}

/// Generic objection handler trait
/// Implementations define objection-specific responses
pub trait ObjectionHandler: Send + Sync {
    /// Unique objection identifier
    fn objection_id(&self) -> &str;

    /// Patterns to detect this objection
    fn patterns(&self) -> &[Pattern];

    /// Response template for this objection
    fn response_template(&self) -> &ResponseTemplate;

    /// Check if text matches this objection
    fn matches(&self, text: &str) -> bool;
}

/// Generic conversation stage trait
/// Implementations define stage-specific behavior
pub trait ConversationStage: Send + Sync {
    /// Unique stage identifier
    fn stage_id(&self) -> &str;

    /// Stage guidance/instructions for the agent
    fn guidance(&self) -> &str;

    /// Stages that can be transitioned to from this stage
    fn allowed_transitions(&self) -> &[StageId];

    /// Context budget for this stage (tokens)
    fn context_budget(&self) -> usize;

    /// RAG retrieval fraction for this stage
    fn rag_fraction(&self) -> f32;
}

/// Generic slot definition for dialogue state tracking
#[derive(Debug, Clone)]
pub struct SlotDefinition {
    /// Slot identifier
    pub slot_id: SlotId,
    /// Slot type
    pub slot_type: SlotType,
    /// Extraction patterns
    pub patterns: Vec<Pattern>,
    /// Validation rules
    pub validation: Option<SlotValidation>,
    /// Unit conversions (e.g., "tola" -> grams multiplier)
    pub unit_conversions: HashMap<String, f64>,
}

/// Slot types for DST
#[derive(Debug, Clone)]
pub enum SlotType {
    /// String value
    String,
    /// Numeric value
    Number { min: Option<f64>, max: Option<f64> },
    /// Enumerated value
    Enum { values: Vec<String> },
    /// Date value
    Date,
    /// Boolean value
    Boolean,
}

/// Slot validation rules
#[derive(Debug, Clone)]
pub struct SlotValidation {
    /// Required field
    pub required: bool,
    /// Regex pattern for validation
    pub pattern: Option<String>,
    /// Custom validation function name
    pub validator: Option<String>,
}

/// Goal definition for DST
#[derive(Debug, Clone)]
pub struct GoalDefinition {
    /// Goal identifier
    pub goal_id: String,
    /// Required slots for this goal
    pub required_slots: Vec<SlotId>,
    /// Optional slots for this goal
    pub optional_slots: Vec<SlotId>,
    /// Action to take when goal is complete
    pub completion_action: Option<String>,
}

/// Scoring weights for lead qualification
#[derive(Debug, Clone)]
pub struct ScoringWeights {
    /// Weight for urgency signals
    pub urgency: f32,
    /// Weight for engagement level
    pub engagement: f32,
    /// Weight for information provided
    pub information: f32,
    /// Weight for intent signals
    pub intent: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            urgency: 25.0,
            engagement: 25.0,
            information: 25.0,
            intent: 25.0,
        }
    }
}

/// Scoring thresholds
#[derive(Debug, Clone)]
pub struct ScoringThresholds {
    /// High-value amount threshold
    pub high_value_amount: f64,
    /// Minimum engagement turns for MQL
    pub min_engagement_turns: usize,
}

/// Domain configuration view for agents
/// Trait for domain-specific configuration access
pub trait DomainView: Send + Sync {
    /// Domain identifier
    fn domain_id(&self) -> &str;

    /// Display name for the domain
    fn display_name(&self) -> &str;

    /// Get a constant value by key path (e.g., "interest_rates.base_rate")
    fn get_constant(&self, key: &str) -> Option<serde_json::Value>;

    /// Get brand configuration
    fn brand_name(&self) -> &str;
    fn agent_name(&self) -> &str;
    fn helpline(&self) -> &str;
}
