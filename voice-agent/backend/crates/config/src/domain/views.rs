//! Crate-Specific Domain Views
//!
//! Each crate accesses domain configuration through a "view" that provides
//! only the information that crate needs, in terminology appropriate for that crate.

use std::sync::Arc;

use super::scoring::{CategoryWeights, EscalationConfig, ScoringConfig};
use super::slots::{GoalDefinition, SlotDefinition, SlotsConfig};
use super::stages::{StageDefinition, StagesConfig, TransitionTrigger};
use super::MasterDomainConfig;

/// View for the agent crate
/// Provides access to conversation stages, DST slots, scoring, objections
pub struct AgentDomainView {
    config: Arc<MasterDomainConfig>,
}

impl AgentDomainView {
    pub fn new(config: Arc<MasterDomainConfig>) -> Self {
        Self { config }
    }

    /// Get high-value thresholds for lead scoring
    pub fn high_value_amount_threshold(&self) -> f64 {
        self.config.high_value.amount_threshold
    }

    pub fn high_value_weight_threshold(&self) -> f64 {
        self.config.high_value.weight_threshold_grams
    }

    /// Check if signals indicate high-value customer
    pub fn is_high_value(&self, amount: Option<f64>, weight: Option<f64>) -> bool {
        self.config.is_high_value(amount, weight)
    }

    /// Get high-value features to highlight
    pub fn high_value_features(&self) -> &[String] {
        &self.config.high_value.features
    }

    /// Get competitor by name for comparison
    pub fn get_competitor_rate(&self, name: &str) -> Option<f64> {
        self.config.get_competitor(name).map(|c| c.typical_rate)
    }

    /// Get our rate for comparison
    pub fn our_rate_for_amount(&self, amount: f64) -> f64 {
        self.config.get_rate_for_amount(amount)
    }

    // ====== Slot Configuration ======

    /// Get the full slots configuration
    pub fn slots_config(&self) -> &SlotsConfig {
        &self.config.slots
    }

    /// Get a slot definition by name
    pub fn get_slot(&self, name: &str) -> Option<&SlotDefinition> {
        self.config.slots.get_slot(name)
    }

    /// Get a goal definition by name
    pub fn get_goal(&self, name: &str) -> Option<&GoalDefinition> {
        self.config.slots.get_goal(name)
    }

    /// Map an intent to a goal
    pub fn goal_for_intent(&self, intent: &str) -> Option<&str> {
        self.config.slots.goal_for_intent(intent)
    }

    /// Get extraction patterns for a slot in a specific language
    pub fn extraction_patterns(&self, slot_name: &str, language: &str) -> Vec<&str> {
        self.config.slots.extraction_patterns(slot_name, language)
    }

    /// Get purity factor for a gold purity value
    pub fn purity_factor(&self, purity_id: &str) -> f64 {
        self.config.slots.purity_factor(purity_id)
    }

    /// Get typical rate for a lender (from slot enum values)
    pub fn lender_rate(&self, lender_id: &str) -> Option<f64> {
        self.config.slots.lender_rate(lender_id)
    }

    /// Get unit conversion factor (e.g., tola -> grams)
    pub fn unit_conversion(&self, slot_name: &str, unit: &str) -> Option<f64> {
        self.config.slots.unit_conversion(slot_name, unit)
    }

    /// Get required slots for a goal
    pub fn required_slots_for_goal(&self, goal_name: &str) -> Vec<&str> {
        self.config
            .slots
            .get_goal(goal_name)
            .map(|g| g.required_slots.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get optional slots for a goal
    pub fn optional_slots_for_goal(&self, goal_name: &str) -> Vec<&str> {
        self.config
            .slots
            .get_goal(goal_name)
            .map(|g| g.optional_slots.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get completion action for a goal
    pub fn completion_action_for_goal(&self, goal_name: &str) -> Option<&str> {
        self.config
            .slots
            .get_goal(goal_name)
            .and_then(|g| g.completion_action.as_deref())
    }

    // ====== Stage Configuration ======

    /// Get the full stages configuration
    pub fn stages_config(&self) -> &StagesConfig {
        &self.config.stages
    }

    /// Get a stage definition by ID
    pub fn get_stage(&self, stage_id: &str) -> Option<&StageDefinition> {
        self.config.stages.get_stage(stage_id)
    }

    /// Get the initial stage ID
    pub fn initial_stage_id(&self) -> &str {
        &self.config.stages.initial_stage
    }

    /// Get the initial stage definition
    pub fn get_initial_stage(&self) -> Option<&StageDefinition> {
        self.config.stages.get_initial_stage()
    }

    /// Get valid transitions from a stage
    pub fn stage_transitions(&self, stage_id: &str) -> Vec<&str> {
        self.config.stages.get_transitions(stage_id)
    }

    /// Check if a transition is valid
    pub fn is_valid_stage_transition(&self, from: &str, to: &str) -> bool {
        self.config.stages.is_valid_transition(from, to)
    }

    /// Get stage guidance text
    pub fn stage_guidance(&self, stage_id: &str) -> Option<&str> {
        self.config.stages.get_guidance(stage_id)
    }

    /// Get suggested questions for a stage
    pub fn stage_questions(&self, stage_id: &str) -> Vec<&str> {
        self.config.stages.get_suggested_questions(stage_id)
    }

    /// Get context budget for a stage (in tokens)
    pub fn stage_context_budget(&self, stage_id: &str) -> usize {
        self.config.stages.get_context_budget(stage_id)
    }

    /// Get RAG context fraction for a stage (0.0-1.0)
    pub fn stage_rag_fraction(&self, stage_id: &str) -> f32 {
        self.config.stages.get_rag_fraction(stage_id)
    }

    /// Get transition trigger for a stage
    pub fn stage_trigger(&self, stage_id: &str) -> Option<&TransitionTrigger> {
        self.config.stages.get_trigger(stage_id)
    }

    /// Get all stage IDs
    pub fn all_stage_ids(&self) -> Vec<&str> {
        self.config.stages.stage_ids()
    }

    // ====== Lead Scoring Configuration ======

    /// Get the full scoring configuration
    pub fn scoring_config(&self) -> &ScoringConfig {
        &self.config.scoring
    }

    /// Get qualification level from score
    pub fn qualification_level(&self, score: u32) -> &'static str {
        self.config.scoring.qualification_level(score)
    }

    /// Get escalation configuration
    pub fn escalation_config(&self) -> &EscalationConfig {
        &self.config.scoring.escalation
    }

    /// Get max objections before escalation
    pub fn max_objections_before_escalate(&self) -> u32 {
        self.config.scoring.escalation.max_objections
    }

    /// Get max stalled turns before escalation
    pub fn max_stalled_turns(&self) -> u32 {
        self.config.scoring.escalation.max_stalled_turns
    }

    /// Get high-value loan threshold
    pub fn high_value_loan_threshold(&self) -> f64 {
        self.config.scoring.escalation.high_value_threshold
    }

    /// Get category weights for scoring
    pub fn scoring_weights(&self) -> &CategoryWeights {
        &self.config.scoring.weights
    }

    /// Get urgency keywords for a language
    pub fn urgency_keywords(&self, language: &str) -> Vec<&str> {
        self.config.scoring.urgency_keywords(language)
    }

    /// Get trust score for a level
    pub fn trust_score(&self, level: &str) -> u32 {
        self.config.scoring.trust_score(level)
    }

    /// Get qualification thresholds
    pub fn qualification_thresholds(&self) -> (u32, u32, u32, u32) {
        let t = &self.config.scoring.qualification_thresholds;
        (t.cold, t.warm, t.hot, t.qualified)
    }
}

/// View for the llm crate
/// Provides access to prompts, tool schemas, brand info
pub struct LlmDomainView {
    config: Arc<MasterDomainConfig>,
}

impl LlmDomainView {
    pub fn new(config: Arc<MasterDomainConfig>) -> Self {
        Self { config }
    }

    /// Get domain display name for prompts
    pub fn domain_name(&self) -> &str {
        &self.config.display_name
    }

    /// Get brand information for prompts
    pub fn bank_name(&self) -> &str {
        &self.config.brand.bank_name
    }

    pub fn agent_name(&self) -> &str {
        &self.config.brand.agent_name
    }

    pub fn helpline(&self) -> &str {
        &self.config.brand.helpline
    }

    /// Get key facts for system prompt
    pub fn key_facts(&self) -> Vec<String> {
        let mut facts = Vec::new();

        // Best interest rate
        if let Some(best_tier) = self.config.constants.interest_rates.tiers.last() {
            facts.push(format!("Interest rates: Starting from {}% p.a.", best_tier.rate));
        }

        // LTV
        facts.push(format!("LTV: Up to {}% of gold value", self.config.constants.ltv_percent));

        // Loan range
        let min = self.config.constants.loan_limits.min;
        let max = self.config.constants.loan_limits.max;
        facts.push(format!("Loan range: ₹{} to ₹{}", format_amount(min), format_amount(max)));

        facts
    }

    /// Get product variants for tool responses
    pub fn product_names(&self) -> Vec<&str> {
        self.config.products.values().map(|p| p.name.as_str()).collect()
    }
}

/// View for the tools crate
/// Provides access to tool configs, branch data, SMS templates, constants
pub struct ToolsDomainView {
    config: Arc<MasterDomainConfig>,
}

impl ToolsDomainView {
    pub fn new(config: Arc<MasterDomainConfig>) -> Self {
        Self { config }
    }

    /// Get interest rate for eligibility calculations
    pub fn get_rate_for_amount(&self, amount: f64) -> f64 {
        self.config.get_rate_for_amount(amount)
    }

    /// Get LTV percentage
    pub fn ltv_percent(&self) -> f64 {
        self.config.constants.ltv_percent
    }

    /// Get purity factor for gold type
    pub fn purity_factor(&self, purity: &str) -> f64 {
        self.config.constants.purity_factors
            .get(purity)
            .copied()
            .unwrap_or(1.0)
    }

    /// Get gold price per gram
    pub fn gold_price_per_gram(&self) -> f64 {
        self.config.constants.gold_price_per_gram
    }

    /// Get loan limits
    pub fn min_loan_amount(&self) -> f64 {
        self.config.constants.loan_limits.min
    }

    pub fn max_loan_amount(&self) -> f64 {
        self.config.constants.loan_limits.max
    }

    /// Get processing fee percentage
    pub fn processing_fee_percent(&self) -> f64 {
        self.config.constants.processing_fee_percent
    }

    /// Get competitor info for savings calculations
    pub fn get_competitor(&self, name: &str) -> Option<CompetitorInfo> {
        self.config.get_competitor(name).map(|c| CompetitorInfo {
            name: c.display_name.clone(),
            rate: c.typical_rate,
            ltv: c.ltv_percent,
        })
    }

    /// Get brand info for SMS/responses
    pub fn bank_name(&self) -> &str {
        &self.config.brand.bank_name
    }

    pub fn helpline(&self) -> &str {
        &self.config.brand.helpline
    }
}

/// Simplified competitor info for tools
#[derive(Debug, Clone)]
pub struct CompetitorInfo {
    pub name: String,
    pub rate: f64,
    pub ltv: f64,
}

/// Format amount in Indian style (lakhs/crores)
fn format_amount(amount: f64) -> String {
    if amount >= 10_000_000.0 {
        format!("{:.1} Cr", amount / 10_000_000.0)
    } else if amount >= 100_000.0 {
        format!("{:.1} L", amount / 100_000.0)
    } else {
        format!("{:.0}", amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_amount() {
        assert_eq!(format_amount(10000.0), "10000");
        assert_eq!(format_amount(100000.0), "1.0 L");
        assert_eq!(format_amount(2500000.0), "25.0 L");
        assert_eq!(format_amount(10000000.0), "1.0 Cr");
        assert_eq!(format_amount(25000000.0), "2.5 Cr");
    }
}
