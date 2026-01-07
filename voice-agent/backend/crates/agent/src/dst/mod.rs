//! Dialogue State Tracking (DST) for Conversations
//!
//! Implements domain-agnostic dialogue state tracking based on LDST and ACL 2024 research.
//!
//! ## Phase 2 (Domain-Agnosticism): DialogueStateTracking Trait
//!
//! The `DialogueStateTracking` trait abstracts DST operations, allowing domain-agnostic
//! agents to work with any dialogue state implementation. This enables:
//! - Domain-specific state structs (e.g., GoldLoanDialogueState, InsuranceDialogueState)
//! - Testing with mock state trackers
//! - Alternative slot extraction strategies
//!
//! # Features
//!
//! - Domain-slot based state tracking
//! - Multi-turn slot value tracking
//! - Slot correction handling ("actually, it's 50 grams, not 40")
//! - Slot confirmation tracking
//! - Confidence-based slot updates
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_agent::dst::{DialogueStateTracker, GoldLoanDialogueState, DialogueStateTracking};
//! use voice_agent_text_processing::intent::IntentDetector;
//!
//! let detector = IntentDetector::new();
//! let mut tracker = DialogueStateTracker::new();
//!
//! // User: "I want a gold loan of 5 lakh"
//! let intent = detector.detect("I want a gold loan of 5 lakh");
//! tracker.update(&intent);
//!
//! // State now contains loan_amount = 500000
//! assert_eq!(tracker.state().loan_amount(), Some(500000.0));
//! ```

pub mod slots;
pub mod extractor;

pub use slots::{
    GoldLoanDialogueState, SlotValue, UrgencyLevel, GoalId, NextBestAction, DEFAULT_GOAL,
    // Config-driven purity types
    PurityId, purity_ids, parse_purity_id, format_purity_display,
};
pub use extractor::SlotExtractor;
// Phase 2: Re-export DialogueState trait (implemented for GoldLoanDialogueState in slots.rs)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use voice_agent_text_processing::intent::{DetectedIntent, Slot};
// P13 FIX: Import AgentDomainView for config-driven instructions
use voice_agent_config::domain::AgentDomainView;

// =============================================================================
// Phase 2: DialogueStateTracking Trait (Domain-Agnostic Abstraction)
// =============================================================================

/// Trait for dialogue state abstraction
///
/// This trait allows domain-agnostic access to dialogue state properties.
/// Different domains can implement their own state structs while conforming
/// to this interface.
pub trait DialogueState: Send + Sync {
    /// Get the primary detected intent
    fn primary_intent(&self) -> Option<&str>;

    /// Get a slot value by name
    fn get_slot_value(&self, slot_name: &str) -> Option<String>;

    /// Set a slot value with confidence
    fn set_slot_value(&mut self, slot_name: &str, value: &str, confidence: f32);

    /// Clear a slot value
    fn clear_slot(&mut self, slot_name: &str);

    /// Get all filled slot names
    fn filled_slots(&self) -> Vec<&str>;

    /// Get pending (unconfirmed) slot names
    fn pending_slots(&self) -> &HashSet<String>;

    /// Get confirmed slot names
    fn confirmed_slots(&self) -> &HashSet<String>;

    /// Mark a slot as pending confirmation
    fn mark_pending(&mut self, slot_name: &str);

    /// Mark a slot as confirmed
    fn mark_confirmed(&mut self, slot_name: &str);

    /// Get current goal ID
    fn goal_id(&self) -> &str;

    /// Set current goal
    fn set_goal(&mut self, goal_id: &str, turn: usize);

    /// Confirm goal (user explicitly stated it)
    fn confirm_goal(&mut self, goal_id: &str, turn: usize);

    /// Check if we should auto-capture lead
    fn should_auto_capture_lead(&self) -> bool;

    /// Generate context string for prompts
    fn to_context_string(&self) -> String;

    /// Generate full context including goal information
    fn to_full_context_string(&self) -> String;

    /// Update intent with confidence
    fn update_intent(&mut self, intent: &str, confidence: f32);

    /// Get slot value with confidence
    fn get_slot_with_confidence(&self, slot_name: &str) -> Option<&SlotValue>;

    /// Get next best action for current state
    fn next_best_action(&self) -> NextBestAction;
}

/// Trait for dialogue state tracking operations
///
/// This trait abstracts the dialogue state tracker, allowing domain-agnostic
/// agents to work with any DST implementation.
///
/// # Example
/// ```ignore
/// // Agent uses trait bound instead of concrete DialogueStateTracker
/// pub struct DomainAgent<D: DialogueStateTracking> {
///     dst: Arc<parking_lot::RwLock<D>>,
///     // ...
/// }
/// ```
pub trait DialogueStateTracking: Send + Sync {
    /// Type of dialogue state managed by this tracker
    type State: DialogueState;

    /// Get current dialogue state (immutable)
    fn state(&self) -> &Self::State;

    /// Get current dialogue state (mutable)
    fn state_mut(&mut self) -> &mut Self::State;

    /// Get state change history
    fn history(&self) -> &[StateChange];

    /// Update state from detected intent
    fn update(&mut self, intent: &DetectedIntent);

    /// Update a specific slot
    fn update_slot(
        &mut self,
        slot_name: &str,
        value: &str,
        confidence: f32,
        source: ChangeSource,
        turn_index: usize,
    );

    /// Confirm a slot value
    fn confirm_slot(&mut self, slot_name: &str);

    /// Clear a slot value
    fn clear_slot(&mut self, slot_name: &str);

    /// Get slots needing confirmation
    fn slots_needing_confirmation(&self) -> Vec<&str>;

    /// Get confirmed slots
    fn confirmed_slots(&self) -> Vec<&str>;

    /// Check if all required slots for an intent are filled
    fn is_intent_complete(&self, intent: &str) -> bool;

    /// Get missing required slots for an intent
    fn missing_slots_for_intent(&self, intent: &str) -> Vec<&str>;

    /// Generate prompt context from current state
    fn state_context(&self) -> String;

    /// Generate full context including goal information
    fn full_context(&self) -> String;

    /// Get current conversation goal ID
    fn goal_id(&self) -> &str;

    /// Update goal from detected intent
    fn update_goal_from_intent(&mut self, intent: &str, turn: usize);

    /// Set goal explicitly
    fn set_goal(&mut self, goal_id: &str, turn: usize);

    /// Confirm goal
    fn confirm_goal(&mut self, goal_id: &str, turn: usize);

    /// Check if we should auto-capture lead
    fn should_auto_capture_lead(&self) -> bool;

    /// Reset the tracker
    fn reset(&mut self);

    /// Get instruction for an action (config-driven if domain view available)
    fn instruction_for_action(&self, action: &NextBestAction, language: &str) -> String;
}

/// Dialogue State Tracker for Gold Loan conversations
pub struct DialogueStateTracker {
    /// Current dialogue state
    state: GoldLoanDialogueState,
    /// History of state changes
    history: Vec<StateChange>,
    /// Configuration
    config: DstConfig,
    /// P13 FIX: Domain view for config-driven instructions (optional for backward compat)
    domain_view: Option<Arc<AgentDomainView>>,
}

/// Configuration for DST
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DstConfig {
    /// Minimum confidence to accept a slot value
    pub min_slot_confidence: f32,
    /// Confidence threshold for auto-confirmation
    pub auto_confirm_confidence: f32,
    /// Enable correction detection
    pub enable_corrections: bool,
    /// Maximum turns to look back for corrections
    pub correction_lookback: usize,
}

impl Default for DstConfig {
    fn default() -> Self {
        Self {
            min_slot_confidence: 0.5,
            auto_confirm_confidence: 0.9,
            enable_corrections: true,
            correction_lookback: 3,
        }
    }
}

/// Record of a state change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateChange {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Slot that changed
    pub slot_name: String,
    /// Old value
    pub old_value: Option<String>,
    /// New value
    pub new_value: Option<String>,
    /// Confidence of the change
    pub confidence: f32,
    /// Source of the change
    pub source: ChangeSource,
    /// Turn index
    pub turn_index: usize,
}

/// Source of a state change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeSource {
    /// Extracted from user utterance
    UserUtterance,
    /// User correction ("actually, it's...")
    Correction,
    /// System confirmation
    SystemConfirmation,
    /// External data (CRM, etc.)
    External,
}

impl DialogueStateTracker {
    /// Create a new dialogue state tracker
    pub fn new() -> Self {
        Self {
            state: GoldLoanDialogueState::new(),
            history: Vec::new(),
            config: DstConfig::default(),
            domain_view: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DstConfig) -> Self {
        Self {
            state: GoldLoanDialogueState::new(),
            history: Vec::new(),
            config,
            domain_view: None,
        }
    }

    /// P13 FIX: Set domain view for config-driven instructions
    pub fn with_domain_view(mut self, view: Arc<AgentDomainView>) -> Self {
        self.domain_view = Some(view);
        self
    }

    /// P13 FIX: Set domain view (mutable reference version)
    pub fn set_domain_view(&mut self, view: Arc<AgentDomainView>) {
        self.domain_view = Some(view);
    }

    /// P13 FIX: Get instruction for an action, using config if available
    /// Falls back to hardcoded instructions if domain view is not set
    pub fn instruction_for_action(&self, action: &NextBestAction, language: &str) -> String {
        if let Some(ref view) = self.domain_view {
            // Try to get config-driven instruction
            let action_type = match action {
                NextBestAction::ExplainProcess => "explain_process",
                NextBestAction::DiscoverIntent => "discover_intent",
                NextBestAction::OfferAppointment => "offer_appointment",
                NextBestAction::CaptureLead => "capture_lead",
                _ => "", // CallTool and AskFor are dynamic, use fallback
            };

            if !action_type.is_empty() {
                if let Some(instruction) = view.dst_instruction(action_type, language) {
                    return instruction.to_string();
                }
            }
        }

        // Fallback to hardcoded instructions
        action.to_instruction()
    }

    /// Get current dialogue state
    pub fn state(&self) -> &GoldLoanDialogueState {
        &self.state
    }

    /// Get mutable state
    pub fn state_mut(&mut self) -> &mut GoldLoanDialogueState {
        &mut self.state
    }

    /// Get state change history
    pub fn history(&self) -> &[StateChange] {
        &self.history
    }

    /// Update state from detected intent
    pub fn update(&mut self, intent: &DetectedIntent) {
        let turn_index = self.history.len();

        // Check for corrections first
        if self.config.enable_corrections {
            self.detect_and_apply_corrections(&intent.slots, turn_index);
        }

        // Update from extracted slots
        for (slot_name, slot) in &intent.slots {
            if slot.confidence >= self.config.min_slot_confidence {
                if let Some(ref value) = slot.value {
                    self.update_slot(slot_name, value, slot.confidence, ChangeSource::UserUtterance, turn_index);
                }
            }
        }

        // Update primary intent
        self.state.update_intent(&intent.intent, intent.confidence);

        // Check for auto-confirmation
        self.check_auto_confirmations();
    }

    /// Update a specific slot
    pub fn update_slot(
        &mut self,
        slot_name: &str,
        value: &str,
        confidence: f32,
        source: ChangeSource,
        turn_index: usize,
    ) {
        let old_value = self.state.get_slot_value(slot_name);

        // Skip if value unchanged
        if old_value.as_ref().map(|v| v.as_str()) == Some(value) {
            return;
        }

        // Record change
        self.history.push(StateChange {
            timestamp: Utc::now(),
            slot_name: slot_name.to_string(),
            old_value: old_value.clone(),
            new_value: Some(value.to_string()),
            confidence,
            source,
            turn_index,
        });

        // Apply change to state
        self.state.set_slot_value(slot_name, value, confidence);

        // Mark as pending confirmation if not auto-confirmed
        if confidence < self.config.auto_confirm_confidence {
            self.state.mark_pending(slot_name);
        } else {
            self.state.mark_confirmed(slot_name);
        }

        tracing::debug!(
            slot = slot_name,
            old_value = ?old_value,
            new_value = value,
            confidence = confidence,
            "Slot updated"
        );
    }

    /// Confirm a slot value
    pub fn confirm_slot(&mut self, slot_name: &str) {
        self.state.mark_confirmed(slot_name);

        self.history.push(StateChange {
            timestamp: Utc::now(),
            slot_name: slot_name.to_string(),
            old_value: self.state.get_slot_value(slot_name),
            new_value: self.state.get_slot_value(slot_name),
            confidence: 1.0,
            source: ChangeSource::SystemConfirmation,
            turn_index: self.history.len(),
        });
    }

    /// Clear a slot value
    pub fn clear_slot(&mut self, slot_name: &str) {
        let old_value = self.state.get_slot_value(slot_name);
        self.state.clear_slot(slot_name);

        self.history.push(StateChange {
            timestamp: Utc::now(),
            slot_name: slot_name.to_string(),
            old_value,
            new_value: None,
            confidence: 1.0,
            source: ChangeSource::UserUtterance,
            turn_index: self.history.len(),
        });
    }

    /// Detect and apply corrections
    fn detect_and_apply_corrections(
        &mut self,
        new_slots: &HashMap<String, Slot>,
        turn_index: usize,
    ) {
        // Look for correction patterns in the new slots
        for (slot_name, new_slot) in new_slots {
            if let Some(ref new_value) = new_slot.value {
                // Check if this slot was recently set with a different value
                let recent_changes: Vec<_> = self.history
                    .iter()
                    .rev()
                    .take(self.config.correction_lookback)
                    .filter(|c| c.slot_name == *slot_name)
                    .collect();

                if let Some(recent) = recent_changes.first() {
                    if recent.new_value.as_ref() != Some(new_value) {
                        // This looks like a correction
                        tracing::debug!(
                            slot = slot_name,
                            old = ?recent.new_value,
                            new = new_value,
                            "Detected slot correction"
                        );

                        // Apply with correction source (higher priority)
                        self.update_slot(
                            slot_name,
                            new_value,
                            new_slot.confidence.max(0.9), // Boost confidence for corrections
                            ChangeSource::Correction,
                            turn_index,
                        );
                    }
                }
            }
        }
    }

    /// Check and apply auto-confirmations
    fn check_auto_confirmations(&mut self) {
        let pending: Vec<String> = self.state.pending_slots().iter().cloned().collect();

        for slot_name in pending {
            // Check if we have high confidence
            if let Some(slot_value) = self.state.get_slot_with_confidence(&slot_name) {
                if slot_value.confidence >= self.config.auto_confirm_confidence {
                    self.state.mark_confirmed(&slot_name);
                }
            }
        }
    }

    /// Get slots that need confirmation
    pub fn slots_needing_confirmation(&self) -> Vec<&str> {
        self.state.pending_slots().iter().map(|s| s.as_str()).collect()
    }

    /// Get confirmed slots
    pub fn confirmed_slots(&self) -> Vec<&str> {
        self.state.confirmed_slots().iter().map(|s| s.as_str()).collect()
    }

    /// Check if all required slots for an intent are filled
    /// Uses config-driven mappings if domain view is set, falls back to hardcoded
    pub fn is_intent_complete(&self, intent: &str) -> bool {
        // P13 FIX: Use config-driven required slots via domain view
        if let Some(ref view) = self.domain_view {
            // Map intent to goal, then check required slots for that goal
            let goal_id = view.goal_for_intent(intent).unwrap_or(intent);
            let required = view.required_slots_for_goal(goal_id);

            if !required.is_empty() {
                return required.iter().all(|slot| self.state.get_slot_value(slot).is_some());
            }
            // If no required slots defined, intent is complete
            return true;
        }

        // Fallback: hardcoded mappings for backward compatibility
        match intent {
            "eligibility_check" => self.state.gold_weight_grams().is_some(),
            "switch_lender" => self.state.current_lender().is_some(),
            "schedule_visit" => self.state.location().is_some(),
            "send_sms" => self.state.phone_number().is_some(),
            _ => true, // Most intents don't have required slots
        }
    }

    /// Get missing required slots for an intent
    /// Uses config-driven mappings if domain view is set, falls back to hardcoded
    pub fn missing_slots_for_intent(&self, intent: &str) -> Vec<&str> {
        // P13 FIX: Use config-driven required slots via domain view
        if let Some(ref view) = self.domain_view {
            // Map intent to goal, then get required slots for that goal
            let goal_id = view.goal_for_intent(intent).unwrap_or(intent);
            let required = view.required_slots_for_goal(goal_id);

            // Filter to only slots that are missing
            // Note: We return &str from the config, but the caller expects &str
            // This is safe because the config lives for the duration of the view
            return required
                .into_iter()
                .filter(|slot| self.state.get_slot_value(slot).is_none())
                .collect();
        }

        // Fallback: hardcoded mappings for backward compatibility
        match intent {
            "eligibility_check" => {
                let mut missing = Vec::new();
                if self.state.gold_weight_grams().is_none() {
                    missing.push("gold_weight");
                }
                missing
            }
            "switch_lender" => {
                let mut missing = Vec::new();
                if self.state.current_lender().is_none() {
                    missing.push("current_lender");
                }
                missing
            }
            "schedule_visit" => {
                let mut missing = Vec::new();
                if self.state.location().is_none() {
                    missing.push("location");
                }
                missing
            }
            "send_sms" => {
                let mut missing = Vec::new();
                if self.state.phone_number().is_none() {
                    missing.push("phone_number");
                }
                missing
            }
            _ => Vec::new(),
        }
    }

    /// Generate a prompt context from current state
    pub fn state_context(&self) -> String {
        self.state.to_context_string()
    }

    /// Generate full context including goal information
    pub fn full_context(&self) -> String {
        self.state.to_full_context_string()
    }

    /// Get current conversation goal ID
    pub fn goal_id(&self) -> &str {
        self.state.goal_id()
    }

    /// Update goal from detected intent using domain view
    /// Falls back to using intent as goal ID if domain view not available
    pub fn update_goal_from_intent(&mut self, intent: &str, turn: usize) {
        // Use domain view to map intent to goal if available
        if let Some(ref view) = self.domain_view {
            if let Some(goal_id) = view.goal_for_intent(intent) {
                self.state.set_goal(goal_id, turn);
                return;
            }
        }
        // Fallback: use intent as goal ID if it's not exploration
        if intent != "unknown" && intent != "exploration" {
            self.state.set_goal(intent, turn);
        }
    }

    /// Set goal explicitly
    pub fn set_goal(&mut self, goal_id: &str, turn: usize) {
        self.state.set_goal(goal_id, turn);
    }

    /// Confirm goal (user explicitly stated it)
    pub fn confirm_goal(&mut self, goal_id: &str, turn: usize) {
        self.state.confirm_goal(goal_id, turn);
    }

    /// Check if we should auto-capture lead (when we have contact info during any goal)
    pub fn should_auto_capture_lead(&self) -> bool {
        self.state.should_auto_capture_lead()
    }

    /// P13 FIX: Get completion tool/action for current goal
    /// Returns the tool to call when all required slots are filled
    pub fn completion_action_for_goal(&self, goal_id: &str) -> Option<&str> {
        self.domain_view
            .as_ref()
            .and_then(|view| view.completion_action_for_goal(goal_id))
    }

    /// P13 FIX: Get prompt to ask for a missing slot
    /// Returns a localized prompt from config, or generates a default
    pub fn slot_prompt(&self, slot_name: &str, language: &str) -> String {
        // Try to get from config first
        if let Some(ref view) = self.domain_view {
            // Check if slot definition has a description we can use
            if let Some(slot_def) = view.get_slot(slot_name) {
                if !slot_def.description.is_empty() {
                    let prefix = if language == "hi" { "कृपया बताएं" } else { "Please provide" };
                    return format!("{} {}.", prefix, slot_def.description.to_lowercase());
                }
            }
        }

        // Fallback: generate default prompt
        let slot_display = slot_name.replace('_', " ");
        if language == "hi" {
            format!("कृपया अपना {} बताएं।", slot_display)
        } else {
            format!("Please provide your {}.", slot_display)
        }
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.state = GoldLoanDialogueState::new();
        self.history.clear();
    }
}

impl Default for DialogueStateTracker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Phase 2: DialogueStateTracking Implementation for DialogueStateTracker
// =============================================================================

impl DialogueStateTracking for DialogueStateTracker {
    type State = GoldLoanDialogueState;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn history(&self) -> &[StateChange] {
        &self.history
    }

    fn update(&mut self, intent: &DetectedIntent) {
        DialogueStateTracker::update(self, intent)
    }

    fn update_slot(
        &mut self,
        slot_name: &str,
        value: &str,
        confidence: f32,
        source: ChangeSource,
        turn_index: usize,
    ) {
        DialogueStateTracker::update_slot(self, slot_name, value, confidence, source, turn_index)
    }

    fn confirm_slot(&mut self, slot_name: &str) {
        DialogueStateTracker::confirm_slot(self, slot_name)
    }

    fn clear_slot(&mut self, slot_name: &str) {
        DialogueStateTracker::clear_slot(self, slot_name)
    }

    fn slots_needing_confirmation(&self) -> Vec<&str> {
        DialogueStateTracker::slots_needing_confirmation(self)
    }

    fn confirmed_slots(&self) -> Vec<&str> {
        DialogueStateTracker::confirmed_slots(self)
    }

    fn is_intent_complete(&self, intent: &str) -> bool {
        DialogueStateTracker::is_intent_complete(self, intent)
    }

    fn missing_slots_for_intent(&self, intent: &str) -> Vec<&str> {
        DialogueStateTracker::missing_slots_for_intent(self, intent)
    }

    fn state_context(&self) -> String {
        DialogueStateTracker::state_context(self)
    }

    fn full_context(&self) -> String {
        DialogueStateTracker::full_context(self)
    }

    fn goal_id(&self) -> &str {
        DialogueStateTracker::goal_id(self)
    }

    fn update_goal_from_intent(&mut self, intent: &str, turn: usize) {
        DialogueStateTracker::update_goal_from_intent(self, intent, turn)
    }

    fn set_goal(&mut self, goal_id: &str, turn: usize) {
        DialogueStateTracker::set_goal(self, goal_id, turn)
    }

    fn confirm_goal(&mut self, goal_id: &str, turn: usize) {
        DialogueStateTracker::confirm_goal(self, goal_id, turn)
    }

    fn should_auto_capture_lead(&self) -> bool {
        DialogueStateTracker::should_auto_capture_lead(self)
    }

    fn reset(&mut self) {
        DialogueStateTracker::reset(self)
    }

    fn instruction_for_action(&self, action: &NextBestAction, language: &str) -> String {
        DialogueStateTracker::instruction_for_action(self, action, language)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voice_agent_text_processing::intent::IntentDetector;

    #[test]
    fn test_tracker_creation() {
        let tracker = DialogueStateTracker::new();
        assert!(tracker.state().customer_name().is_none());
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_slot_update() {
        let mut tracker = DialogueStateTracker::new();

        tracker.update_slot("customer_name", "Rahul", 0.9, ChangeSource::UserUtterance, 0);

        assert_eq!(tracker.state().customer_name(), Some("Rahul"));
        assert_eq!(tracker.history().len(), 1);
    }

    #[test]
    fn test_intent_update() {
        let mut tracker = DialogueStateTracker::new();
        let detector = IntentDetector::new();

        let intent = detector.detect("I want a gold loan of 5 lakh");
        tracker.update(&intent);

        assert_eq!(tracker.state().primary_intent(), Some("loan_inquiry"));
        assert!(tracker.state().loan_amount().is_some());
    }

    #[test]
    fn test_slot_correction() {
        let mut tracker = DialogueStateTracker::with_config(DstConfig {
            enable_corrections: true,
            ..Default::default()
        });

        // Initial value
        tracker.update_slot("gold_weight", "40", 0.8, ChangeSource::UserUtterance, 0);
        assert_eq!(tracker.state().get_slot_value("gold_weight"), Some("40".to_string()));

        // Correction
        tracker.update_slot("gold_weight", "50", 0.9, ChangeSource::Correction, 1);
        assert_eq!(tracker.state().get_slot_value("gold_weight"), Some("50".to_string()));
        assert_eq!(tracker.history().len(), 2);
    }

    #[test]
    fn test_confirmation_tracking() {
        let mut tracker = DialogueStateTracker::with_config(DstConfig {
            auto_confirm_confidence: 0.95,
            ..Default::default()
        });

        // Low confidence - should be pending
        tracker.update_slot("loan_amount", "500000", 0.8, ChangeSource::UserUtterance, 0);
        assert!(tracker.state().pending_slots().contains(&"loan_amount".to_string()));

        // Confirm
        tracker.confirm_slot("loan_amount");
        assert!(tracker.state().confirmed_slots().contains(&"loan_amount".to_string()));
    }

    #[test]
    fn test_auto_confirmation() {
        let mut tracker = DialogueStateTracker::with_config(DstConfig {
            auto_confirm_confidence: 0.9,
            ..Default::default()
        });

        // High confidence - should auto-confirm
        tracker.update_slot("loan_amount", "500000", 0.95, ChangeSource::UserUtterance, 0);
        assert!(tracker.state().confirmed_slots().contains(&"loan_amount".to_string()));
    }

    #[test]
    fn test_missing_slots_detection() {
        let tracker = DialogueStateTracker::new();

        let missing = tracker.missing_slots_for_intent("eligibility_check");
        assert!(missing.contains(&"gold_weight"));
    }

    #[test]
    fn test_intent_completeness() {
        let mut tracker = DialogueStateTracker::new();

        assert!(!tracker.is_intent_complete("eligibility_check"));

        tracker.update_slot("gold_weight", "50", 0.9, ChangeSource::UserUtterance, 0);
        assert!(tracker.is_intent_complete("eligibility_check"));
    }

    #[test]
    fn test_state_context() {
        let mut tracker = DialogueStateTracker::new();

        tracker.update_slot("customer_name", "Rahul", 0.9, ChangeSource::UserUtterance, 0);
        tracker.update_slot("loan_amount", "500000", 0.9, ChangeSource::UserUtterance, 1);

        let context = tracker.state_context();
        assert!(context.contains("Rahul"));
        // Loan amount is formatted as "5.0 lakh" in context
        assert!(context.contains("5.0 lakh") || context.contains("500000"));
    }
}
