//! Stage-Based Dialog Management
//!
//! Manages conversation stages and transitions for sales conversations.
//!
//! ## Domain Agnosticism
//!
//! This module provides generic stage management for any sales domain.
//! Domain-specific guidance and questions come from config via `StageConfigProvider`.
//! The enum values (Greeting, Discovery, etc.) are generic sales stages.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// P4 FIX: RAG timing strategy for prefetch behavior
///
/// Different strategies trade off between latency and resource usage:
/// - **Eager**: Prefetch on any partial transcript (lowest latency, highest resource usage)
/// - **Conservative**: Only prefetch on high-confidence partials (balanced)
/// - **StageAware**: Only prefetch during stages that heavily use RAG (efficient)
/// - **Disabled**: No prefetching (lowest resource usage, highest latency)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RagTimingStrategy {
    /// Prefetch on any partial transcript (confidence > 0.5)
    /// Best for: Low-latency requirements, powerful hardware
    Eager,
    /// Only prefetch on high-confidence partials (confidence > 0.8)
    /// Best for: Balanced latency/resource trade-off
    #[default]
    Conservative,
    /// Only prefetch during high-RAG stages (Presentation, ObjectionHandling, Discovery)
    /// Best for: Resource-constrained environments
    StageAware,
    /// No prefetching - only search when response is needed
    /// Best for: Minimal resource usage, batch processing
    Disabled,
}

impl RagTimingStrategy {
    /// Check if prefetch should be triggered given the parameters
    pub fn should_prefetch(&self, confidence: f32, stage: ConversationStage) -> bool {
        match self {
            RagTimingStrategy::Eager => confidence > 0.5,
            RagTimingStrategy::Conservative => confidence > 0.8,
            RagTimingStrategy::StageAware => confidence > 0.7 && stage.rag_context_fraction() > 0.1,
            RagTimingStrategy::Disabled => false,
        }
    }

    /// Get minimum word count for prefetch trigger
    pub fn min_words(&self) -> usize {
        match self {
            RagTimingStrategy::Eager => 2,
            RagTimingStrategy::Conservative => 3,
            RagTimingStrategy::StageAware => 3,
            RagTimingStrategy::Disabled => usize::MAX, // Never trigger
        }
    }

    /// Get cache TTL in seconds
    pub fn cache_ttl_secs(&self) -> u64 {
        match self {
            RagTimingStrategy::Eager => 5,
            RagTimingStrategy::Conservative => 10,
            RagTimingStrategy::StageAware => 15,
            RagTimingStrategy::Disabled => 0,
        }
    }
}

/// Conversation stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ConversationStage {
    /// Initial greeting and rapport building
    #[default]
    Greeting,
    /// Understanding customer needs
    Discovery,
    /// Assessing eligibility and readiness
    Qualification,
    /// Presenting product and benefits
    Presentation,
    /// Handling concerns and objections
    ObjectionHandling,
    /// Moving towards commitment
    Closing,
    /// Wrapping up the conversation
    Farewell,
}

impl ConversationStage {
    /// P16 FIX: Get stage ID as string (for config lookups)
    ///
    /// Returns the lowercase snake_case ID matching config keys.
    pub fn as_str(&self) -> &'static str {
        match self {
            ConversationStage::Greeting => "greeting",
            ConversationStage::Discovery => "discovery",
            ConversationStage::Qualification => "qualification",
            ConversationStage::Presentation => "presentation",
            ConversationStage::ObjectionHandling => "objection_handling",
            ConversationStage::Closing => "closing",
            ConversationStage::Farewell => "farewell",
        }
    }

    /// P16 FIX: Parse stage from config string
    ///
    /// Accepts lowercase snake_case IDs from config.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "greeting" => Some(ConversationStage::Greeting),
            "discovery" => Some(ConversationStage::Discovery),
            "qualification" => Some(ConversationStage::Qualification),
            "presentation" => Some(ConversationStage::Presentation),
            "objection_handling" => Some(ConversationStage::ObjectionHandling),
            "closing" => Some(ConversationStage::Closing),
            "farewell" => Some(ConversationStage::Farewell),
            _ => None,
        }
    }

    /// Get stage display name
    pub fn display_name(&self) -> &'static str {
        match self {
            ConversationStage::Greeting => "Greeting",
            ConversationStage::Discovery => "Discovery",
            ConversationStage::Qualification => "Qualification",
            ConversationStage::Presentation => "Presentation",
            ConversationStage::ObjectionHandling => "Objection Handling",
            ConversationStage::Closing => "Closing",
            ConversationStage::Farewell => "Farewell",
        }
    }

    /// P2 FIX: Get stage-aware context budget in tokens
    ///
    /// Different stages require different amounts of context:
    /// - **Greeting**: Minimal - just system prompt and basic info
    /// - **Discovery**: Moderate - conversation history to understand needs
    /// - **Qualification**: Moderate - collected info and history
    /// - **Presentation**: High - full context + RAG for product info
    /// - **ObjectionHandling**: High - full context to address all concerns
    /// - **Closing**: Moderate - key points and summary
    /// - **Farewell**: Low - just wrap up
    pub fn context_budget_tokens(&self) -> usize {
        match self {
            ConversationStage::Greeting => 1024,
            ConversationStage::Discovery => 2048,
            ConversationStage::Qualification => 2048,
            ConversationStage::Presentation => 3584, // Room for RAG context
            ConversationStage::ObjectionHandling => 3584, // Need full context
            ConversationStage::Closing => 2560,
            ConversationStage::Farewell => 1024,
        }
    }

    /// P2 FIX: Get the fraction of context budget to reserve for RAG results
    ///
    /// Returns the fraction (0.0 - 0.5) of the context budget that should
    /// be reserved for RAG-retrieved information.
    pub fn rag_context_fraction(&self) -> f32 {
        match self {
            ConversationStage::Greeting => 0.0,           // No RAG needed
            ConversationStage::Discovery => 0.15,         // Some background info
            ConversationStage::Qualification => 0.2,      // Product details
            ConversationStage::Presentation => 0.4,       // Heavy RAG usage
            ConversationStage::ObjectionHandling => 0.35, // Need facts for objections
            ConversationStage::Closing => 0.2,            // Summary info
            ConversationStage::Farewell => 0.0,           // No RAG needed
        }
    }

    /// P2 FIX: Get recommended number of conversation history turns to keep
    ///
    /// Returns the number of most recent user+assistant turn pairs to include.
    pub fn history_turns_to_keep(&self) -> usize {
        match self {
            ConversationStage::Greeting => 0,          // Fresh start
            ConversationStage::Discovery => 3,         // Recent context
            ConversationStage::Qualification => 4,     // More context
            ConversationStage::Presentation => 5,      // Full history
            ConversationStage::ObjectionHandling => 6, // Need all context
            ConversationStage::Closing => 4,           // Key exchanges
            ConversationStage::Farewell => 2,          // Just recent
        }
    }

    /// Get generic guidance for this stage
    ///
    /// Note: For domain-specific guidance, use `StageConfigProvider` which loads
    /// guidance from config. This method provides generic fallback guidance.
    pub fn guidance(&self) -> &'static str {
        match self {
            ConversationStage::Greeting =>
                "Warmly greet the customer. Introduce yourself. Build initial rapport before discussing products.",
            ConversationStage::Discovery =>
                "Ask open questions to understand their needs. Learn about their current situation and pain points.",
            ConversationStage::Qualification =>
                "Assess eligibility and readiness. Understand timeline and decision-making process.",
            ConversationStage::Presentation =>
                "Present tailored benefits based on their needs. Focus on value and address their specific situation.",
            ConversationStage::ObjectionHandling =>
                "Address concerns with empathy. Use evidence and social proof. Don't be pushy.",
            ConversationStage::Closing =>
                "Summarize benefits and guide to next steps. Schedule appointment or capture contact info.",
            ConversationStage::Farewell =>
                "Thank warmly and confirm next steps. Leave door open for future conversations.",
        }
    }

    /// Get generic suggested questions for this stage
    ///
    /// Note: For domain-specific questions, use `StageConfigProvider` which loads
    /// questions from config. This method provides generic fallback questions.
    pub fn suggested_questions(&self) -> Vec<&'static str> {
        match self {
            ConversationStage::Greeting => {
                vec!["How are you doing today?", "Is this a good time to talk?"]
            },
            ConversationStage::Discovery => vec![
                "Can you tell me about your current situation?",
                "What brought you to us today?",
                "What are you looking for?",
                "What would make you consider switching?",
            ],
            ConversationStage::Qualification => vec![
                "What timeline are you working with?",
                "Are you the primary decision maker?",
                "What are your key requirements?",
            ],
            ConversationStage::Presentation => vec![
                "Would you like to know more about our offering?",
                "Can I show you how this could benefit you?",
            ],
            ConversationStage::ObjectionHandling => vec![
                "What concerns do you have?",
                "Is there anything holding you back?",
            ],
            ConversationStage::Closing => vec![
                "Would you like to proceed?",
                "Can I schedule a follow-up for you?",
            ],
            ConversationStage::Farewell => vec![
                "Is there anything else I can help with?",
                "Do you have any other questions?",
            ],
        }
    }

    /// Get all valid transitions from this stage
    pub fn valid_transitions(&self) -> Vec<ConversationStage> {
        match self {
            ConversationStage::Greeting => {
                vec![ConversationStage::Discovery, ConversationStage::Farewell]
            },
            ConversationStage::Discovery => vec![
                ConversationStage::Qualification,
                ConversationStage::Presentation,
                ConversationStage::ObjectionHandling, // P1 FIX: Customer may object early
                ConversationStage::Farewell,
            ],
            ConversationStage::Qualification => vec![
                ConversationStage::Presentation,
                ConversationStage::Discovery,
                ConversationStage::Farewell,
            ],
            ConversationStage::Presentation => vec![
                ConversationStage::ObjectionHandling,
                ConversationStage::Closing,
                ConversationStage::Farewell,
            ],
            ConversationStage::ObjectionHandling => vec![
                ConversationStage::Presentation,
                ConversationStage::Discovery, // P1 FIX: May need to revisit needs
                ConversationStage::Closing,
                ConversationStage::Farewell,
            ],
            ConversationStage::Closing => vec![
                ConversationStage::ObjectionHandling,
                ConversationStage::Farewell,
            ],
            ConversationStage::Farewell => vec![],
        }
    }
}

/// Stage transition
#[derive(Debug, Clone)]
pub struct StageTransition {
    /// From stage
    pub from: ConversationStage,
    /// To stage
    pub to: ConversationStage,
    /// Reason for transition
    pub reason: TransitionReason,
    /// Confidence in the transition
    pub confidence: f32,
}

/// Reason for stage transition
#[derive(Debug, Clone)]
pub enum TransitionReason {
    /// Intent detected that triggers transition
    IntentDetected(String),
    /// Minimum requirements met for current stage
    StageCompleted,
    /// Customer explicitly requested
    CustomerRequest,
    /// Natural conversation flow
    NaturalFlow,
    /// Timeout or stall in current stage
    Timeout,
    /// Manual override
    Manual,
}

/// Stage requirements for completion
#[derive(Debug, Clone)]
pub struct StageRequirements {
    /// Minimum turns in this stage
    pub min_turns: usize,
    /// Required information collected
    pub required_info: Vec<String>,
    /// Required intents detected
    pub required_intents: Vec<String>,
}

/// Stage manager for tracking and transitioning conversation stages
pub struct StageManager {
    current_stage: Mutex<ConversationStage>,
    stage_history: Mutex<Vec<StageTransition>>,
    stage_turns: Mutex<HashMap<ConversationStage, usize>>,
    collected_info: Mutex<HashMap<String, String>>,
    /// P0 FIX: Track detected intents for stage requirement validation
    detected_intents: Mutex<Vec<String>>,
    requirements: HashMap<ConversationStage, StageRequirements>,
}

impl StageManager {
    /// Create a new stage manager
    pub fn new() -> Self {
        Self {
            current_stage: Mutex::new(ConversationStage::Greeting),
            stage_history: Mutex::new(Vec::new()),
            stage_turns: Mutex::new(HashMap::new()),
            collected_info: Mutex::new(HashMap::new()),
            detected_intents: Mutex::new(Vec::new()),
            requirements: Self::default_requirements(),
        }
    }

    /// P16 FIX: Create a stage manager with requirements loaded from config
    ///
    /// Loads stage requirements from the provided StagesConfig, falling back
    /// to defaults for any stages not defined in config.
    pub fn from_config(config: &voice_agent_config::domain::StagesConfig) -> Self {
        let mut requirements = Self::default_requirements();

        // Override with config values
        for (stage_id, stage_def) in &config.stages {
            if let Some(stage) = ConversationStage::from_str(stage_id) {
                requirements.insert(
                    stage,
                    StageRequirements {
                        min_turns: stage_def.requirements.min_turns,
                        required_info: stage_def.requirements.required_info.clone(),
                        required_intents: stage_def.requirements.required_intents.clone(),
                    },
                );
            }
        }

        Self {
            current_stage: Mutex::new(ConversationStage::Greeting),
            stage_history: Mutex::new(Vec::new()),
            stage_turns: Mutex::new(HashMap::new()),
            collected_info: Mutex::new(HashMap::new()),
            detected_intents: Mutex::new(Vec::new()),
            requirements,
        }
    }

    /// Get default stage requirements
    ///
    /// P16 FIX: Use generic slot names with domain-specific aliases
    fn default_requirements() -> HashMap<ConversationStage, StageRequirements> {
        let mut req = HashMap::new();

        req.insert(
            ConversationStage::Greeting,
            StageRequirements {
                min_turns: 1,
                required_info: vec![],
                required_intents: vec![],
            },
        );

        req.insert(
            ConversationStage::Discovery,
            StageRequirements {
                min_turns: 2,
                // P16 FIX: Generic name (competitor) with domain alias (current_lender)
                required_info: vec!["competitor".into()],
                required_intents: vec![],
            },
        );

        req.insert(
            ConversationStage::Qualification,
            StageRequirements {
                min_turns: 1,
                // P16 FIX: Generic name (asset_quantity) with domain alias (gold_weight)
                required_info: vec!["asset_quantity".into()],
                required_intents: vec![],
            },
        );

        req.insert(
            ConversationStage::Presentation,
            StageRequirements {
                min_turns: 1,
                required_info: vec![],
                required_intents: vec![],
            },
        );

        req.insert(
            ConversationStage::ObjectionHandling,
            StageRequirements {
                min_turns: 1,
                required_info: vec![],
                required_intents: vec!["objection".into()],
            },
        );

        req.insert(
            ConversationStage::Closing,
            StageRequirements {
                min_turns: 1,
                required_info: vec![],
                required_intents: vec![],
            },
        );

        req.insert(
            ConversationStage::Farewell,
            StageRequirements {
                min_turns: 1,
                required_info: vec![],
                required_intents: vec![],
            },
        );

        req
    }

    /// Get current stage
    pub fn current(&self) -> ConversationStage {
        *self.current_stage.lock()
    }

    /// Record a turn in the current stage
    pub fn record_turn(&self) {
        let stage = self.current();
        let mut turns = self.stage_turns.lock();
        *turns.entry(stage).or_insert(0) += 1;
    }

    /// Record collected information
    pub fn record_info(&self, key: &str, value: &str) {
        self.collected_info
            .lock()
            .insert(key.to_string(), value.to_string());
    }

    /// Record a detected intent
    ///
    /// P0 FIX: Tracks intents for stage requirement validation.
    pub fn record_intent(&self, intent: &str) {
        let mut intents = self.detected_intents.lock();
        if !intents.contains(&intent.to_string()) {
            intents.push(intent.to_string());
            tracing::debug!("Recorded intent: {}", intent);
        }
    }

    /// Check if a specific intent has been detected
    pub fn has_intent(&self, intent: &str) -> bool {
        self.detected_intents.lock().contains(&intent.to_string())
    }

    /// Check if current stage requirements are met
    ///
    /// P0 FIX: Now validates required_intents in addition to min_turns and required_info.
    /// P16 FIX: Accepts both generic and domain-specific slot names.
    pub fn stage_completed(&self) -> bool {
        let stage = self.current();
        let turns = self.stage_turns.lock();
        let info = self.collected_info.lock();
        let intents = self.detected_intents.lock();

        if let Some(req) = self.requirements.get(&stage) {
            // Check minimum turns
            let stage_turns = turns.get(&stage).copied().unwrap_or(0);
            if stage_turns < req.min_turns {
                return false;
            }

            // Check required info (P16 FIX: check aliases too)
            for key in &req.required_info {
                if !self.has_info_or_alias(&info, key) {
                    return false;
                }
            }

            // P0 FIX: Check required intents
            for intent in &req.required_intents {
                if !intents.contains(intent) {
                    tracing::debug!(
                        "Stage {:?} incomplete: missing required intent '{}'",
                        stage,
                        intent
                    );
                    return false;
                }
            }

            true
        } else {
            true // No requirements, always completed
        }
    }

    /// P16 FIX: Check if info contains key or any of its aliases
    fn has_info_or_alias(&self, info: &HashMap<String, String>, key: &str) -> bool {
        // Check direct key
        if info.contains_key(key) {
            return true;
        }

        // Check known aliases (generic ↔ domain-specific mappings)
        let aliases: &[&str] = match key {
            // Generic → domain-specific
            "asset_quantity" => &["gold_weight", "gold_weight_grams", "weight"],
            "asset_quality" => &["gold_purity", "purity", "karat"],
            "competitor" => &["current_lender", "other_bank"],
            "amount" => &["loan_amount", "requested_amount"],
            // Domain-specific → generic
            "gold_weight" | "gold_weight_grams" => &["asset_quantity"],
            "gold_purity" | "purity" => &["asset_quality"],
            "current_lender" => &["competitor"],
            "loan_amount" => &["amount"],
            _ => &[],
        };

        aliases.iter().any(|alias| info.contains_key(*alias))
    }

    /// Transition to a new stage
    pub fn transition(
        &self,
        to: ConversationStage,
        reason: TransitionReason,
    ) -> Result<StageTransition, String> {
        let from = self.current();

        // Check if transition is valid
        if !from.valid_transitions().contains(&to) && to != from {
            return Err(format!("Invalid transition from {:?} to {:?}", from, to));
        }

        let transition = StageTransition {
            from,
            to,
            reason,
            confidence: 1.0,
        };

        // Update state
        *self.current_stage.lock() = to;
        self.stage_history.lock().push(transition.clone());

        Ok(transition)
    }

    /// Force set stage without validation (for restore/checkpoint operations)
    ///
    /// Unlike `transition()`, this method bypasses the valid_transitions check.
    /// Use with caution - only for checkpoint restore or testing.
    pub fn set_stage(&self, stage: ConversationStage) {
        let from = self.current();
        *self.current_stage.lock() = stage;

        // Record the transition for history
        let transition = StageTransition {
            from,
            to: stage,
            reason: TransitionReason::Manual,
            confidence: 1.0,
        };
        self.stage_history.lock().push(transition);
    }

    /// Suggest next stage based on current state
    pub fn suggest_next(&self) -> Option<ConversationStage> {
        let current = self.current();

        if self.stage_completed() {
            // Suggest natural next stage
            match current {
                ConversationStage::Greeting => Some(ConversationStage::Discovery),
                ConversationStage::Discovery => Some(ConversationStage::Qualification),
                ConversationStage::Qualification => Some(ConversationStage::Presentation),
                ConversationStage::Presentation => Some(ConversationStage::Closing),
                ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
                ConversationStage::Closing => Some(ConversationStage::Farewell),
                ConversationStage::Farewell => None,
            }
        } else {
            None // Stay in current stage
        }
    }

    /// Get stage history
    pub fn history(&self) -> Vec<StageTransition> {
        self.stage_history.lock().clone()
    }

    /// Get turns in current stage
    pub fn current_stage_turns(&self) -> usize {
        let stage = self.current();
        self.stage_turns.lock().get(&stage).copied().unwrap_or(0)
    }

    /// Reset manager
    pub fn reset(&self) {
        *self.current_stage.lock() = ConversationStage::Greeting;
        self.stage_history.lock().clear();
        self.stage_turns.lock().clear();
        self.collected_info.lock().clear();
    }
}

impl Default for StageManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_transitions() {
        let manager = StageManager::new();

        assert_eq!(manager.current(), ConversationStage::Greeting);

        // Valid transition
        let result =
            manager.transition(ConversationStage::Discovery, TransitionReason::NaturalFlow);
        assert!(result.is_ok());
        assert_eq!(manager.current(), ConversationStage::Discovery);
    }

    #[test]
    fn test_invalid_transition() {
        let manager = StageManager::new();

        // Invalid: can't go from Greeting to Closing
        let result = manager.transition(ConversationStage::Closing, TransitionReason::Manual);
        assert!(result.is_err());
    }

    #[test]
    fn test_stage_completion() {
        let manager = StageManager::new();

        // Not completed initially
        assert!(!manager.stage_completed());

        // Record a turn
        manager.record_turn();

        // Now completed (Greeting only needs 1 turn)
        assert!(manager.stage_completed());
    }

    #[test]
    fn test_suggest_next() {
        let manager = StageManager::new();
        manager.record_turn();

        let next = manager.suggest_next();
        assert_eq!(next, Some(ConversationStage::Discovery));
    }

    #[test]
    fn test_context_budget_tokens() {
        // P2 FIX: Test stage-aware context budgets
        // Greeting should have lowest budget
        assert!(ConversationStage::Greeting.context_budget_tokens() <= 1024);

        // Presentation should have high budget for RAG
        assert!(ConversationStage::Presentation.context_budget_tokens() >= 3000);

        // ObjectionHandling should also have high budget
        assert!(ConversationStage::ObjectionHandling.context_budget_tokens() >= 3000);

        // Farewell should have low budget
        assert!(ConversationStage::Farewell.context_budget_tokens() <= 1500);
    }

    #[test]
    fn test_rag_context_fraction() {
        // P2 FIX: Test RAG context fractions
        // Greeting should have no RAG
        assert_eq!(ConversationStage::Greeting.rag_context_fraction(), 0.0);

        // Farewell should have no RAG
        assert_eq!(ConversationStage::Farewell.rag_context_fraction(), 0.0);

        // Presentation should have highest RAG fraction
        let presentation_rag = ConversationStage::Presentation.rag_context_fraction();
        assert!(
            presentation_rag >= 0.3,
            "Presentation RAG fraction should be >= 0.3"
        );

        // ObjectionHandling should also have high RAG fraction
        let objection_rag = ConversationStage::ObjectionHandling.rag_context_fraction();
        assert!(
            objection_rag >= 0.3,
            "ObjectionHandling RAG fraction should be >= 0.3"
        );
    }

    #[test]
    fn test_history_turns_to_keep() {
        // P2 FIX: Test history turns recommendations
        // Greeting should keep no history (fresh start)
        assert_eq!(ConversationStage::Greeting.history_turns_to_keep(), 0);

        // ObjectionHandling should keep the most history
        assert!(ConversationStage::ObjectionHandling.history_turns_to_keep() >= 5);

        // Farewell should keep minimal history
        assert!(ConversationStage::Farewell.history_turns_to_keep() <= 3);
    }

    // P4 FIX: RAG timing strategy tests

    #[test]
    fn test_rag_timing_strategy_eager() {
        let strategy = RagTimingStrategy::Eager;

        // Should prefetch on any stage with medium confidence
        assert!(strategy.should_prefetch(0.6, ConversationStage::Greeting));
        assert!(strategy.should_prefetch(0.6, ConversationStage::Presentation));

        // Should not prefetch with very low confidence
        assert!(!strategy.should_prefetch(0.4, ConversationStage::Greeting));

        // Should have low min words
        assert_eq!(strategy.min_words(), 2);
    }

    #[test]
    fn test_rag_timing_strategy_conservative() {
        let strategy = RagTimingStrategy::Conservative;

        // Should only prefetch on high confidence
        assert!(!strategy.should_prefetch(0.7, ConversationStage::Presentation));
        assert!(strategy.should_prefetch(0.85, ConversationStage::Presentation));

        // Should require more words
        assert_eq!(strategy.min_words(), 3);
    }

    #[test]
    fn test_rag_timing_strategy_stage_aware() {
        let strategy = RagTimingStrategy::StageAware;

        // Should not prefetch during Greeting (no RAG needed)
        assert!(!strategy.should_prefetch(0.9, ConversationStage::Greeting));
        assert!(!strategy.should_prefetch(0.9, ConversationStage::Farewell));

        // Should prefetch during Presentation (high RAG)
        assert!(strategy.should_prefetch(0.8, ConversationStage::Presentation));

        // Should prefetch during ObjectionHandling (high RAG)
        assert!(strategy.should_prefetch(0.8, ConversationStage::ObjectionHandling));
    }

    #[test]
    fn test_rag_timing_strategy_disabled() {
        let strategy = RagTimingStrategy::Disabled;

        // Should never prefetch
        assert!(!strategy.should_prefetch(1.0, ConversationStage::Presentation));
        assert!(!strategy.should_prefetch(1.0, ConversationStage::ObjectionHandling));

        // Min words should be very high (effectively never)
        assert_eq!(strategy.min_words(), usize::MAX);
    }

    #[test]
    fn test_rag_timing_strategy_default() {
        // Default should be Conservative
        let strategy = RagTimingStrategy::default();
        assert_eq!(strategy, RagTimingStrategy::Conservative);
    }
}
