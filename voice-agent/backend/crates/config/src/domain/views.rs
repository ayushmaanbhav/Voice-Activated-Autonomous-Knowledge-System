//! Crate-Specific Domain Views
//!
//! Each crate accesses domain configuration through a "view" that provides
//! only the information that crate needs, in terminology appropriate for that crate.

use std::collections::HashMap;
use std::sync::Arc;

use super::branches::{BranchEntry, BranchesConfig};
use super::competitors::{CompetitorEntry as ExtCompetitorEntry, CompetitorsConfig};
use super::objections::{ObjectionResponse, ObjectionsConfig};
use super::prompts::PromptsConfig;
use super::scoring::{CategoryWeights, EscalationConfig, ScoringConfig};
use super::segments::{SegmentDefinition, SegmentsConfig};
use super::slots::{GoalDefinition, SlotDefinition, SlotsConfig};
use super::sms_templates::SmsTemplatesConfig;
use super::stages::{StageDefinition, StagesConfig, TransitionTrigger};
use super::tools::{ToolSchema, ToolsConfig};
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

    // ====== Brand Information ======

    /// P13 FIX: Get bank name for persona goal
    pub fn bank_name(&self) -> &str {
        &self.config.brand.bank_name
    }

    /// P13 FIX: Get agent name
    pub fn agent_name(&self) -> &str {
        &self.config.brand.agent_name
    }

    /// P13 FIX: Get agent role for persona goal (e.g., "Gold Loan Advisor")
    /// Falls back to domain display name if not set
    pub fn agent_role(&self) -> &str {
        if self.config.brand.agent_role.is_empty() {
            // Default fallback based on domain
            "Advisor"
        } else {
            &self.config.brand.agent_role
        }
    }

    /// Get helpline number
    pub fn helpline(&self) -> &str {
        &self.config.brand.helpline
    }

    // ====== DST Instructions ======

    /// P13 FIX: Get DST instruction for an action type
    /// Falls back to English if the language-specific instruction is not found
    pub fn dst_instruction(&self, action_type: &str, language: &str) -> Option<&str> {
        self.config.prompts.dst_instruction(action_type, language)
    }

    // ====== High-Value Customer Detection ======

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

    // ====== Objection Handling Configuration ======

    /// Get the full objections configuration
    pub fn objections_config(&self) -> &ObjectionsConfig {
        &self.config.objections
    }

    /// Detect objection type from text
    pub fn detect_objection(&self, text: &str, language: &str) -> Option<&str> {
        self.config.objections.detect_objection(text, language)
    }

    /// Get objection response for a type and language
    pub fn objection_response(&self, objection_type: &str, language: &str) -> Option<&ObjectionResponse> {
        self.config.objections.get_response(objection_type, language)
    }

    /// Get default objection response for unrecognized concerns
    pub fn default_objection_response(&self, language: &str) -> Option<&ObjectionResponse> {
        self.config.objections.get_default_response(language)
    }

    /// Get all objection type names
    pub fn objection_types(&self) -> Vec<&str> {
        self.config.objections.objection_types()
    }

    /// Build full response text for an objection
    pub fn build_objection_response(&self, objection_type: &str, language: &str) -> Option<String> {
        self.config.objections.build_full_response(objection_type, language)
    }

    // ====== Feature Configuration ======

    /// Get display name for a feature in a specific language
    pub fn feature_display_name(&self, feature_id: &str, language: &str) -> Option<&str> {
        self.config.features.display_name(feature_id, language)
    }

    /// Get features for a segment (from features config)
    pub fn features_for_segment(&self, segment_id: &str) -> Vec<&str> {
        self.config.features.features_for_segment(segment_id)
    }

    /// Get top N features for a segment
    pub fn top_features_for_segment(&self, segment_id: &str, n: usize) -> Vec<&str> {
        self.config.features.top_features_for_segment(segment_id, n)
    }

    /// Get value propositions for a segment (from features config)
    pub fn value_propositions_for_segment(&self, segment_id: &str) -> Vec<&str> {
        self.config.features.value_propositions_for_segment(segment_id)
    }

    /// Get value propositions with rate substitution
    pub fn value_propositions_with_rate(&self, segment_id: &str, rate: f64) -> Vec<String> {
        self.config.features.value_propositions_with_rate(segment_id, rate)
    }

    /// Check if a feature exists
    pub fn has_feature(&self, feature_id: &str) -> bool {
        self.config.features.has_feature(feature_id)
    }

    /// Get all feature IDs
    pub fn all_feature_ids(&self) -> Vec<&str> {
        self.config.features.feature_ids()
    }

    // ====== Customer Segment Configuration ======

    /// Get the full segments configuration
    pub fn segments_config(&self) -> &SegmentsConfig {
        &self.config.segments
    }

    /// Get a segment definition by ID
    pub fn get_segment(&self, segment_id: &str) -> Option<&SegmentDefinition> {
        self.config.segments.get_segment(segment_id)
    }

    /// Detect customer segments from text and signals
    pub fn detect_segments(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> Vec<&str> {
        self.config.segments.detect_segments(text, language, numeric_values, text_values)
    }

    /// Get value propositions for a segment
    pub fn segment_value_props(&self, segment_id: &str, language: &str) -> Vec<&str> {
        self.config.segments.get_value_props(segment_id, language)
    }

    /// Get features to highlight for a segment
    pub fn segment_features(&self, segment_id: &str) -> Vec<&str> {
        self.config.segments.get_features(segment_id)
    }

    /// Get the default segment ID
    pub fn default_segment(&self) -> &str {
        &self.config.segments.default_segment
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

    // ====== Tool Schema Configuration ======

    /// Get the full tools configuration
    pub fn tools_config(&self) -> &ToolsConfig {
        &self.config.tools
    }

    /// Get a tool schema by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolSchema> {
        self.config.tools.get_tool(name)
    }

    /// Get all tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.config.tools.tool_names()
    }

    /// Get enabled tool names
    pub fn enabled_tool_names(&self) -> Vec<&str> {
        self.config.tools.enabled_tool_names()
    }

    /// Get all tool schemas as JSON for LLM consumption
    pub fn tool_schemas_json(&self) -> Vec<serde_json::Value> {
        self.config.tools.to_json_schemas()
    }

    /// Get tool usage guideline
    pub fn tool_guideline(&self, key: &str) -> Option<&str> {
        self.config.tools.get_guideline(key)
    }

    /// Get general tool usage guideline
    pub fn general_tool_guideline(&self) -> Option<&str> {
        self.config.tools.get_guideline("general")
    }

    // ====== Prompts Configuration ======

    /// Get the full prompts configuration
    pub fn prompts_config(&self) -> &PromptsConfig {
        &self.config.prompts
    }

    /// Get system prompt template
    pub fn system_prompt_template(&self) -> &str {
        &self.config.prompts.system_prompt
    }

    /// Get language style description
    pub fn language_style(&self, language: &str) -> &str {
        self.config.prompts.language_style(language)
    }

    /// Build persona traits string
    pub fn build_persona_traits(&self, warmth: f32, empathy: f32, formality: f32, urgency: f32) -> String {
        self.config.prompts.build_persona_traits(warmth, empathy, formality, urgency)
    }

    /// Build system prompt with brand and persona
    pub fn build_system_prompt(
        &self,
        persona_traits: &str,
        language: &str,
        key_facts: &str,
    ) -> String {
        self.config.prompts.build_system_prompt(
            &self.config.brand.agent_name,
            &self.config.brand.bank_name,
            persona_traits,
            language,
            key_facts,
            &self.config.brand.helpline,
        )
    }

    /// Build RAG context message
    pub fn build_rag_context(&self, context: &str) -> String {
        self.config.prompts.build_rag_context(context)
    }

    /// Build stage guidance message
    pub fn build_stage_guidance(&self, guidance: &str) -> String {
        self.config.prompts.build_stage_guidance(guidance)
    }

    /// Get response template for scenario and language
    pub fn response_template(&self, scenario: &str, language: &str) -> Option<&str> {
        self.config.prompts.response_template(scenario, language)
    }

    /// Get error template for scenario and language
    pub fn error_template(&self, scenario: &str, language: &str) -> Option<&str> {
        self.config.prompts.error_template(scenario, language)
    }

    // ====== P7 FIX: Methods migrated from DomainConfigManager ======

    /// Get greeting for language (from response templates)
    /// Falls back to English if language not found
    pub fn get_greeting(&self, language: &str) -> String {
        self.config.prompts.response_template("greeting", language)
            .or_else(|| self.config.prompts.response_template("greeting", "en"))
            .map(|template| {
                template
                    .replace("{agent_name}", &self.config.brand.agent_name)
                    .replace("{bank_name}", &self.config.brand.bank_name)
            })
            .unwrap_or_else(|| {
                format!(
                    "Hello! I'm {} from {}. How can I help you today?",
                    self.config.brand.agent_name,
                    self.config.brand.bank_name
                )
            })
    }

    /// Get greeting with time-based prefix (morning/afternoon/evening)
    pub fn get_greeting_with_time(&self, language: &str, hour: u32) -> String {
        let time_greeting = match hour {
            5..=11 => "Good morning",
            12..=16 => "Good afternoon",
            17..=20 => "Good evening",
            _ => "Hello",
        };

        let base_greeting = self.get_greeting(language);
        format!("{}! {}", time_greeting, base_greeting.trim_start_matches("Hello! "))
    }

    /// Get farewell message for language
    pub fn get_farewell(&self, language: &str) -> String {
        self.config.prompts.response_template("farewell", language)
            .or_else(|| self.config.prompts.response_template("farewell", "en"))
            .map(|template| {
                template
                    .replace("{helpline}", &self.config.brand.helpline)
            })
            .unwrap_or_else(|| {
                format!(
                    "Thank you for speaking with me! Call our helpline at {} if you have any questions.",
                    self.config.brand.helpline
                )
            })
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

    // ====== Branch Configuration ======

    /// Get the full branches configuration
    pub fn branches_config(&self) -> &BranchesConfig {
        &self.config.branches
    }

    /// Get all branches
    pub fn all_branches(&self) -> &[BranchEntry] {
        &self.config.branches.branches
    }

    /// Find branches by city
    pub fn find_branches_by_city(&self, city: &str) -> Vec<&BranchEntry> {
        self.config.branches.find_by_city(city)
    }

    /// Find branches by pincode
    pub fn find_branches_by_pincode(&self, pincode: &str) -> Vec<&BranchEntry> {
        self.config.branches.find_by_pincode(pincode)
    }

    /// Get branch by ID
    pub fn get_branch(&self, branch_id: &str) -> Option<&BranchEntry> {
        self.config.branches.get_branch(branch_id)
    }

    /// Get branches with gold loan service
    pub fn gold_loan_branches(&self) -> Vec<&BranchEntry> {
        self.config.branches.gold_loan_branches()
    }

    /// Get default max results for branch search
    pub fn branch_search_max_results(&self) -> usize {
        self.config.branches.defaults.max_results
    }

    // ====== SMS Templates Configuration ======

    /// Get the full SMS templates configuration
    pub fn sms_templates_config(&self) -> &SmsTemplatesConfig {
        &self.config.sms_templates
    }

    /// Get SMS template by type and language
    pub fn sms_template(&self, template_type: &str, language: &str) -> Option<&str> {
        self.config.sms_templates.get_template(template_type, language)
    }

    /// Build SMS message from template with placeholders
    pub fn build_sms_message(
        &self,
        template_type: &str,
        language: &str,
        placeholders: &HashMap<String, String>,
    ) -> Option<String> {
        self.config.sms_templates.build_message(template_type, language, placeholders)
    }

    /// Get all SMS template types
    pub fn sms_template_types(&self) -> Vec<&str> {
        self.config.sms_templates.template_types()
    }

    /// Check if SMS type is transactional
    pub fn is_transactional_sms(&self, template_type: &str) -> bool {
        self.config.sms_templates.is_transactional(template_type)
    }

    // ====== Extended Competitors Configuration ======

    /// Get the full competitors configuration
    pub fn competitors_config(&self) -> &CompetitorsConfig {
        &self.config.competitors_config
    }

    /// Get extended competitor info by ID
    pub fn get_competitor_extended(&self, id: &str) -> Option<&ExtCompetitorEntry> {
        self.config.competitors_config.get_competitor(id)
    }

    /// Find competitor by name or alias
    pub fn find_competitor_by_name(&self, name: &str) -> Option<(&str, &ExtCompetitorEntry)> {
        self.config.competitors_config.find_by_name(name)
    }

    /// Get all NBFCs
    pub fn nbfc_competitors(&self) -> Vec<(&str, &ExtCompetitorEntry)> {
        self.config.competitors_config.nbfcs()
    }

    /// Get all bank competitors
    pub fn bank_competitors(&self) -> Vec<(&str, &ExtCompetitorEntry)> {
        self.config.competitors_config.banks()
    }

    /// Get default rate for competitor type
    pub fn default_competitor_rate(&self, competitor_type: &str) -> f64 {
        self.config.competitors_config.default_rate_for_type(competitor_type)
    }

    /// Get highlighted comparison points
    pub fn highlighted_comparison_points(&self) -> Vec<(&str, &str)> {
        self.config.competitors_config.highlighted_points()
            .into_iter()
            .map(|p| (p.category.as_str(), p.our_advantage.as_str()))
            .collect()
    }

    /// P14 FIX: Get our features for comparison
    pub fn our_features(&self) -> &[String] {
        self.config.competitors_config.our_features()
    }

    /// P14 FIX: Get all competitor IDs
    pub fn all_competitor_ids(&self) -> Vec<&str> {
        self.config.competitors_config.competitor_ids()
    }

    /// P14 FIX: Get competitor data tuple (id, display_name, rate, ltv, strengths)
    /// Returns all competitors as tuples for comparison tool
    pub fn all_competitors_data(&self) -> Vec<(&str, &str, f64, f64, Vec<&str>)> {
        self.config.competitors_config.competitors.iter()
            .map(|(id, entry)| {
                (
                    id.as_str(),
                    entry.display_name.as_str(),
                    entry.typical_rate,
                    entry.ltv_percent,
                    entry.strengths.iter().map(|s| s.as_str()).collect(),
                )
            })
            .collect()
    }

    // ====== P7 FIX: Methods migrated from DomainConfigManager ======

    /// Check if doorstep service is available in a city
    pub fn doorstep_available(&self, city: &str) -> bool {
        self.config.branches.doorstep_available(city)
    }

    /// Calculate monthly savings vs competitor
    /// Returns savings info, or None if competitor not found
    pub fn calculate_competitor_savings(
        &self,
        competitor: &str,
        loan_amount: f64,
    ) -> Option<MonthlySavings> {
        // Try to get competitor rate from extended config first, then basic config
        let their_rate = self.config.competitors_config.find_by_name(competitor)
            .map(|(_, entry)| entry.typical_rate)
            .or_else(|| {
                // Fallback to basic competitor list in domain.yaml
                self.config.get_competitor(competitor).map(|c| c.typical_rate)
            })?;

        let our_rate = self.config.get_rate_for_amount(loan_amount);

        if their_rate <= our_rate {
            // No savings if competitor is cheaper
            return Some(MonthlySavings {
                monthly: 0.0,
                annual: 0.0,
                our_rate,
                their_rate,
            });
        }

        // Monthly savings = loan_amount * (rate_diff / 12 / 100)
        let rate_diff = their_rate - our_rate;
        let monthly_savings = loan_amount * rate_diff / 12.0 / 100.0;
        let annual_savings = monthly_savings * 12.0;

        Some(MonthlySavings {
            monthly: monthly_savings,
            annual: annual_savings,
            our_rate,
            their_rate,
        })
    }

    // ====== P12: Methods replacing GoldLoanConfig ======

    /// Calculate gold value given weight and purity
    pub fn calculate_gold_value(&self, weight_grams: f64, purity: &str) -> f64 {
        let purity_factor = self.purity_factor(purity);
        weight_grams * self.gold_price_per_gram() * purity_factor
    }

    /// Calculate maximum loan amount based on gold value
    pub fn calculate_max_loan(&self, gold_value: f64) -> f64 {
        let max_from_ltv = gold_value * (self.ltv_percent() / 100.0);
        max_from_ltv.min(self.max_loan_amount())
    }

    /// Get competitor rate by name (convenience method)
    /// Falls back to default NBFC rate if competitor not found
    pub fn get_competitor_rate(&self, lender: &str) -> f64 {
        self.config.competitors_config.find_by_name(lender)
            .map(|(_, entry)| entry.typical_rate)
            .or_else(|| {
                self.config.get_competitor(lender).map(|c| c.typical_rate)
            })
            .unwrap_or_else(|| self.default_competitor_rate("nbfc"))
    }

    /// Calculate monthly savings when switching from competitor
    /// Uses the competitor's rate and our tiered rate
    pub fn calculate_monthly_savings(&self, loan_amount: f64, current_rate: f64) -> f64 {
        let our_rate = self.get_rate_for_amount(loan_amount);
        let current_monthly = loan_amount * (current_rate / 100.0 / 12.0);
        let our_monthly = loan_amount * (our_rate / 100.0 / 12.0);
        current_monthly - our_monthly
    }

    /// Get the base/headline interest rate (for marketing purposes)
    pub fn base_interest_rate(&self) -> f64 {
        self.config.constants.interest_rates.base_rate
    }

    /// P15 FIX: Get rate tier name for an amount
    /// Returns the tier name (e.g., "Standard", "Premium", "Elite")
    /// If tier has no name, derives from tier index
    pub fn get_rate_tier_name(&self, amount: f64) -> &str {
        for (idx, tier) in self.config.constants.interest_rates.tiers.iter().enumerate() {
            let threshold = tier.max_amount.unwrap_or(f64::MAX);
            if amount <= threshold {
                // Return tier name if set, otherwise derive from index
                if !tier.name.is_empty() {
                    return &tier.name;
                }
                // Derive tier name from index
                return match idx {
                    0 => "Standard",
                    1 => "Premium",
                    2 => "Elite",
                    _ => "Special",
                };
            }
        }
        // Return the last tier name for amounts above all thresholds
        self.config.constants.interest_rates.tiers
            .last()
            .and_then(|t| {
                if !t.name.is_empty() {
                    Some(t.name.as_str())
                } else {
                    None
                }
            })
            .unwrap_or("Elite")
    }

    /// P15 FIX: Get competitor IDs for building dynamic schema enums
    /// Returns owned strings for use in tool schemas
    pub fn competitor_ids(&self) -> Vec<String> {
        self.config.competitors_config.competitor_ids()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    // ====== P15 FIX: Domain Context / Vocabulary ======

    /// Get vocabulary terms for text processing
    pub fn vocabulary_terms(&self) -> &[String] {
        &self.config.vocabulary.terms
    }

    /// Get phrases for text processing
    pub fn vocabulary_phrases(&self) -> &[String] {
        &self.config.vocabulary.phrases
    }

    /// Get abbreviations as (short, full) pairs
    pub fn vocabulary_abbreviations(&self) -> Vec<(String, String)> {
        self.config.vocabulary.abbreviations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Get entity types to preserve
    pub fn vocabulary_entities(&self) -> &[String] {
        &self.config.vocabulary.preserve_entities
    }

    /// Get all competitor names for text processing
    pub fn all_competitor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        // From extended competitors config
        for (_, entry) in &self.config.competitors_config.competitors {
            names.push(entry.display_name.clone());
            names.extend(entry.aliases.clone());
        }
        // From basic competitors in domain.yaml
        for (_, entry) in &self.config.competitors {
            names.push(entry.display_name.clone());
            names.extend(entry.aliases.clone());
        }
        names.sort();
        names.dedup();
        names
    }

    /// P15 FIX: Create DomainContext from config
    /// Returns a DomainContext populated from vocabulary config
    pub fn create_domain_context(&self) -> voice_agent_core::DomainContext {
        voice_agent_core::DomainContext::from_config(
            &self.config.domain_id,
            self.vocabulary_terms().to_vec(),
            self.vocabulary_phrases().to_vec(),
            self.vocabulary_abbreviations(),
            self.vocabulary_entities().to_vec(),
            self.all_competitor_names(),
        )
    }
}

/// Simplified competitor info for tools
#[derive(Debug, Clone)]
pub struct CompetitorInfo {
    pub name: String,
    pub rate: f64,
    pub ltv: f64,
}

/// Monthly savings calculation result
#[derive(Debug, Clone)]
pub struct MonthlySavings {
    pub monthly: f64,
    pub annual: f64,
    pub our_rate: f64,
    pub their_rate: f64,
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
