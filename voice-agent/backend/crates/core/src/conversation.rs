//! Conversation types including stages and turns

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

/// Conversation stages for sales flow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum ConversationStage {
    /// Initial greeting and introduction
    #[default]
    Greeting,
    /// Understanding customer needs
    Discovery,
    /// Assessing eligibility and fit
    Qualification,
    /// Presenting the offer
    Presentation,
    /// Handling objections
    ObjectionHandling,
    /// Closing the deal
    Closing,
    /// Final wrap-up
    Farewell,
}

/// P2 FIX: Static transition map using once_cell::Lazy for O(1) lookup.
/// Previously, allowed_transitions() created a new Vec on every call.
static STAGE_TRANSITIONS: Lazy<HashMap<ConversationStage, &'static [ConversationStage]>> = Lazy::new(|| {
    use ConversationStage::*;
    let mut map = HashMap::new();
    map.insert(Greeting, &[Discovery, Farewell] as &[_]);
    map.insert(Discovery, &[Qualification, Presentation, Farewell] as &[_]);
    map.insert(Qualification, &[Presentation, Discovery, Farewell] as &[_]);
    map.insert(Presentation, &[ObjectionHandling, Closing, Farewell] as &[_]);
    map.insert(ObjectionHandling, &[Presentation, Closing, Farewell] as &[_]);
    map.insert(Closing, &[ObjectionHandling, Farewell] as &[_]);
    map.insert(Farewell, &[] as &[_]);
    map
});

impl ConversationStage {
    /// Get allowed transitions from current stage
    ///
    /// P2 FIX: Now uses static lookup table instead of creating Vec on each call
    pub fn allowed_transitions(&self) -> &'static [ConversationStage] {
        STAGE_TRANSITIONS.get(self).copied().unwrap_or(&[])
    }

    /// Check if transition to target stage is allowed
    pub fn can_transition_to(&self, target: ConversationStage) -> bool {
        self.allowed_transitions().contains(&target)
    }

    /// Get stage-specific prompt guidance
    pub fn prompt_guidance(&self) -> &'static str {
        match self {
            ConversationStage::Greeting => {
                "Introduce yourself warmly. Acknowledge any previous relationship with the bank. \
                 Ask an open question to understand their current situation."
            }
            ConversationStage::Discovery => {
                "Ask about their current gold loan situation. Understand pain points with \
                 current lender. Identify gold quantity and purpose of loan."
            }
            ConversationStage::Qualification => {
                "Assess eligibility based on gold quantity and purity. Understand loan amount \
                 needs. Check for any documentation requirements."
            }
            ConversationStage::Presentation => {
                "Present personalized benefits. Show savings calculator results. Emphasize \
                 trust and safety of Kotak. Mention the Switch & Save program."
            }
            ConversationStage::ObjectionHandling => {
                "Listen empathetically. Address specific concerns. Provide evidence and \
                 testimonials. Never be pushy or dismissive."
            }
            ConversationStage::Closing => {
                "Summarize benefits. Ask for commitment. Offer to schedule appointment. \
                 Create appropriate urgency without pressure."
            }
            ConversationStage::Farewell => {
                "Thank them for their time. Provide clear next steps. Leave door open \
                 for future contact. End on a positive note."
            }
        }
    }

    /// Get suggested maximum duration for stage (in seconds)
    pub fn suggested_duration_seconds(&self) -> u32 {
        match self {
            ConversationStage::Greeting => 30,
            ConversationStage::Discovery => 120,
            ConversationStage::Qualification => 60,
            ConversationStage::Presentation => 90,
            ConversationStage::ObjectionHandling => 60,
            ConversationStage::Closing => 60,
            ConversationStage::Farewell => 30,
        }
    }

    /// Get default next stage
    pub fn default_next(&self) -> Option<ConversationStage> {
        match self {
            ConversationStage::Greeting => Some(ConversationStage::Discovery),
            ConversationStage::Discovery => Some(ConversationStage::Qualification),
            ConversationStage::Qualification => Some(ConversationStage::Presentation),
            ConversationStage::Presentation => Some(ConversationStage::Closing),
            ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
            ConversationStage::Closing => Some(ConversationStage::Farewell),
            ConversationStage::Farewell => None,
        }
    }
}


impl std::fmt::Display for ConversationStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversationStage::Greeting => write!(f, "Greeting"),
            ConversationStage::Discovery => write!(f, "Discovery"),
            ConversationStage::Qualification => write!(f, "Qualification"),
            ConversationStage::Presentation => write!(f, "Presentation"),
            ConversationStage::ObjectionHandling => write!(f, "Objection Handling"),
            ConversationStage::Closing => write!(f, "Closing"),
            ConversationStage::Farewell => write!(f, "Farewell"),
        }
    }
}

/// Role in a conversation turn
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    /// User/customer message
    User,
    /// Assistant/agent message
    Assistant,
    /// System message (instructions)
    System,
}

impl TurnRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            TurnRole::User => "user",
            TurnRole::Assistant => "assistant",
            TurnRole::System => "system",
        }
    }
}

impl std::fmt::Display for TurnRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A single turn in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Role of the speaker
    pub role: TurnRole,
    /// Content of the turn
    pub content: String,
    /// When the turn occurred
    pub timestamp: DateTime<Utc>,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<TurnMetadata>,
}

impl Turn {
    /// Create a new turn
    pub fn new(role: TurnRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp: Utc::now(),
            metadata: None,
        }
    }

    /// Create a user turn
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(TurnRole::User, content)
    }

    /// Create an assistant turn
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(TurnRole::Assistant, content)
    }

    /// Create a system turn
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(TurnRole::System, content)
    }

    /// Add metadata to the turn
    pub fn with_metadata(mut self, metadata: TurnMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get word count
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Estimate token count (rough: words * 1.3)
    pub fn estimated_tokens(&self) -> usize {
        (self.word_count() as f32 * 1.3) as usize
    }
}

/// Metadata for a conversation turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnMetadata {
    /// Detected intent
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Extracted entities
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
    /// Stage when this turn occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<ConversationStage>,
    /// Confidence of the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Latency in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// P2 FIX: Speaker ID for diarization support.
    /// Used to identify different speakers when processing multi-speaker audio.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
}

impl TurnMetadata {
    pub fn new() -> Self {
        Self {
            intent: None,
            entities: Vec::new(),
            stage: None,
            confidence: None,
            latency_ms: None,
            speaker_id: None,
        }
    }

    pub fn with_intent(mut self, intent: impl Into<String>) -> Self {
        self.intent = Some(intent.into());
        self
    }

    pub fn with_stage(mut self, stage: ConversationStage) -> Self {
        self.stage = Some(stage);
        self
    }

    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = Some(latency_ms);
        self
    }

    /// P2 FIX: Set speaker ID for diarization
    pub fn with_speaker_id(mut self, speaker_id: impl Into<String>) -> Self {
        self.speaker_id = Some(speaker_id.into());
        self
    }
}

impl Default for TurnMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Intent detected from user input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Intent name
    pub name: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Extracted slots/parameters
    #[serde(default)]
    pub slots: std::collections::HashMap<String, String>,
}

impl Intent {
    pub fn new(name: impl Into<String>, confidence: f32) -> Self {
        Self {
            name: name.into(),
            confidence,
            slots: std::collections::HashMap::new(),
        }
    }

    pub fn with_slot(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.slots.insert(key.into(), value.into());
        self
    }
}

/// Entity extracted from conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity type/name
    pub name: String,
    /// Entity value
    pub value: serde_json::Value,
    /// Confidence score
    pub confidence: f32,
    /// Source text span
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_text: Option<String>,
}

impl Entity {
    pub fn new(name: impl Into<String>, value: serde_json::Value, confidence: f32) -> Self {
        Self {
            name: name.into(),
            value,
            confidence,
            source_text: None,
        }
    }

    pub fn string(name: impl Into<String>, value: impl Into<String>, confidence: f32) -> Self {
        Self::new(name, serde_json::Value::String(value.into()), confidence)
    }

    pub fn number(name: impl Into<String>, value: f64, confidence: f32) -> Self {
        Self::new(
            name,
            serde_json::Number::from_f64(value)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            confidence,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_transitions() {
        let stage = ConversationStage::Greeting;
        assert!(stage.can_transition_to(ConversationStage::Discovery));
        assert!(stage.can_transition_to(ConversationStage::Farewell));
        assert!(!stage.can_transition_to(ConversationStage::Closing));
    }

    #[test]
    fn test_turn_creation() {
        let turn = Turn::user("Hello, I need a gold loan");
        assert_eq!(turn.role, TurnRole::User);
        assert!(turn.word_count() > 0);

        let turn = Turn::assistant("I'd be happy to help!")
            .with_metadata(TurnMetadata::new().with_stage(ConversationStage::Greeting));
        assert!(turn.metadata.is_some());
    }

    #[test]
    fn test_intent() {
        let intent = Intent::new("request_loan", 0.95)
            .with_slot("loan_type", "gold_loan")
            .with_slot("amount", "100000");

        assert_eq!(intent.slots.len(), 2);
        assert_eq!(intent.slots.get("loan_type"), Some(&"gold_loan".to_string()));
    }
}
