//! Lead Scoring Module for Gold Loan Voice Agent
//!
//! Implements predictive lead scoring based on research best practices:
//! - Signal-based scoring (urgency, price sensitivity, trust, engagement)
//! - MQL (Marketing Qualified Lead) vs SQL (Sales Qualified Lead) classification
//! - Conversion probability estimation
//! - Auto-escalation triggers
//!
//! # Research References
//! - Section 10: Sales Conversion & Customer Support from SMALL_MODEL_AGENT_RESEARCH.md
//! - Lead scoring models (MQL → SQL → Opportunity)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lead qualification level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeadQualification {
    /// Cold lead - just exploring, low intent
    Cold,
    /// Warm lead - showing interest, gathering information
    Warm,
    /// Hot lead - high intent, ready to act
    Hot,
    /// Qualified lead - all criteria met, ready for conversion
    Qualified,
}

impl LeadQualification {
    /// Get the minimum score threshold for this qualification level
    pub fn min_score(&self) -> u32 {
        match self {
            LeadQualification::Cold => 0,
            LeadQualification::Warm => 30,
            LeadQualification::Hot => 60,
            LeadQualification::Qualified => 80,
        }
    }

    /// Create from score
    pub fn from_score(score: u32) -> Self {
        match score {
            0..=29 => LeadQualification::Cold,
            30..=59 => LeadQualification::Warm,
            60..=79 => LeadQualification::Hot,
            _ => LeadQualification::Qualified,
        }
    }
}

/// Lead classification (MQL vs SQL)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeadClassification {
    /// Not yet qualified
    Unqualified,
    /// Marketing Qualified Lead - meets marketing criteria
    MQL,
    /// Sales Qualified Lead - meets sales criteria, ready for conversion
    SQL,
}

impl LeadClassification {
    /// Determine classification from signals
    pub fn from_signals(signals: &LeadSignals) -> Self {
        // SQL criteria: High urgency + provided contact info + specific loan requirements
        if signals.has_urgency_signal
            && signals.provided_contact_info
            && signals.has_specific_requirements
        {
            return LeadClassification::SQL;
        }

        // MQL criteria: Engaged + showed interest in rates/comparison
        if signals.engagement_turns >= 3
            && (signals.asked_about_rates || signals.asked_for_comparison)
        {
            return LeadClassification::MQL;
        }

        LeadClassification::Unqualified
    }
}

/// Signals collected during conversation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LeadSignals {
    // Urgency indicators
    pub has_urgency_signal: bool,
    pub urgency_keywords_count: u32,

    // Price sensitivity
    pub price_sensitive: bool,
    pub asked_about_rates: bool,
    pub asked_for_comparison: bool,
    pub objected_to_price: bool,

    // Trust indicators
    pub trust_level: TrustLevel,
    pub has_existing_relationship: bool,
    pub mentioned_other_banks: bool,

    // Engagement metrics
    pub engagement_turns: u32,
    pub questions_asked: u32,
    pub objections_raised: u32,
    pub objections_resolved: u32,

    // Information provided
    pub provided_contact_info: bool,
    pub provided_asset_details: bool,
    pub provided_loan_amount: bool,
    pub has_specific_requirements: bool,

    // Intent signals
    pub expressed_intent_to_proceed: bool,
    pub requested_callback: bool,
    pub requested_branch_visit: bool,
    pub requested_human_agent: bool,

    // Negative signals
    pub expressed_disinterest: bool,
    pub mentioned_competitor_preference: bool,
    pub conversation_stalled_turns: u32,
}

/// Trust level indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TrustLevel {
    #[default]
    Unknown,
    Low,
    Medium,
    High,
}

impl TrustLevel {
    pub fn score(&self) -> u32 {
        match self {
            TrustLevel::Unknown => 0,
            TrustLevel::Low => 5,
            TrustLevel::Medium => 10,
            TrustLevel::High => 15,
        }
    }
}

/// Auto-escalation trigger reasons
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EscalationTrigger {
    /// Too many objections detected
    ExcessiveObjections { count: u32, threshold: u32 },
    /// Conversation stalled without progress
    ConversationStalled { turns: u32, threshold: u32 },
    /// High-value loan amount requires human attention
    HighValueLoan { amount: f64, threshold: f64 },
    /// Customer explicitly frustrated
    CustomerFrustration,
    /// Customer requested human agent
    CustomerRequested,
    /// Complex query beyond AI capability
    ComplexQuery,
    /// Compliance-sensitive topic
    ComplianceSensitive,
}

/// Lead score with breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadScore {
    /// Total score (0-100)
    pub total: u32,
    /// Qualification level
    pub qualification: LeadQualification,
    /// MQL/SQL classification
    pub classification: LeadClassification,
    /// Estimated conversion probability (0.0-1.0)
    pub conversion_probability: f32,
    /// Score breakdown by category
    pub breakdown: ScoreBreakdown,
    /// Active escalation triggers
    pub escalation_triggers: Vec<EscalationTrigger>,
    /// Recommendation for next action
    pub recommendation: LeadRecommendation,
}

/// Score breakdown by category
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Urgency score (0-25)
    pub urgency: u32,
    /// Engagement score (0-25)
    pub engagement: u32,
    /// Information completeness score (0-25)
    pub information: u32,
    /// Intent strength score (0-25)
    pub intent: u32,
    /// Penalty from negative signals
    pub penalty: i32,
}

/// Recommended next action
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeadRecommendation {
    /// Continue AI conversation
    ContinueConversation,
    /// Push for appointment scheduling
    PushForAppointment,
    /// Offer callback from human agent
    OfferCallback,
    /// Escalate to human immediately
    EscalateNow { reason: String },
    /// Send follow-up SMS/email
    SendFollowUp,
    /// Mark as low priority
    LowPriority,
}

/// Lead Scoring Engine
pub struct LeadScoringEngine {
    /// Configuration
    config: LeadScoringConfig,
    /// Current signals
    signals: LeadSignals,
    /// Score history (for trend analysis)
    score_history: Vec<u32>,
}

/// Configuration for lead scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadScoringConfig {
    /// Objection threshold for auto-escalation
    pub max_objections_before_escalate: u32,
    /// Stalled turns threshold for auto-escalation
    pub max_stalled_turns: u32,
    /// High-value loan threshold (in rupees)
    pub high_value_loan_threshold: f64,
    /// Weight multipliers for score categories
    pub weights: ScoreWeights,
}

/// Weights for different score categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreWeights {
    pub urgency: f32,
    pub engagement: f32,
    pub information: f32,
    pub intent: f32,
}

impl Default for LeadScoringConfig {
    fn default() -> Self {
        Self {
            max_objections_before_escalate: 3,
            max_stalled_turns: 5,
            high_value_loan_threshold: 1_000_000.0, // 10 lakh
            weights: ScoreWeights {
                urgency: 1.0,
                engagement: 1.0,
                information: 1.0,
                intent: 1.0,
            },
        }
    }
}

impl LeadScoringEngine {
    /// Create a new lead scoring engine
    pub fn new() -> Self {
        Self {
            config: LeadScoringConfig::default(),
            signals: LeadSignals::default(),
            score_history: Vec::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: LeadScoringConfig) -> Self {
        Self {
            config,
            signals: LeadSignals::default(),
            score_history: Vec::new(),
        }
    }

    /// Get current signals (read-only)
    pub fn signals(&self) -> &LeadSignals {
        &self.signals
    }

    /// Get mutable signals for updates
    pub fn signals_mut(&mut self) -> &mut LeadSignals {
        &mut self.signals
    }

    /// Update signals from detected intent
    pub fn update_from_intent(&mut self, intent: &str, slots: &HashMap<String, String>) {
        match intent {
            "loan_inquiry" | "eligibility_query" => {
                self.signals.engagement_turns += 1;
                // P16 FIX: Accept both domain-specific and generic slot names
                if slots.contains_key("loan_amount") || slots.contains_key("offer_amount") || slots.contains_key("requested_amount") {
                    self.signals.provided_loan_amount = true;
                    self.signals.has_specific_requirements = true;
                }
            }
            "interest_rate_query" => {
                self.signals.asked_about_rates = true;
                self.signals.engagement_turns += 1;
            }
            "switch_lender" | "balance_transfer" => {
                self.signals.mentioned_other_banks = true;
                self.signals.asked_for_comparison = true;
                self.signals.engagement_turns += 1;
            }
            "schedule_visit" | "schedule_callback" => {
                self.signals.expressed_intent_to_proceed = true;
                self.signals.requested_branch_visit = intent == "schedule_visit";
                self.signals.requested_callback = intent == "schedule_callback";
            }
            "objection" => {
                self.signals.objections_raised += 1;
            }
            "affirmative" | "agreement" => {
                if self.signals.objections_raised > self.signals.objections_resolved {
                    self.signals.objections_resolved += 1;
                }
            }
            "negative" | "rejection" => {
                self.signals.expressed_disinterest = true;
            }
            "escalate" | "human_agent" => {
                self.signals.requested_human_agent = true;
            }
            "question" => {
                self.signals.questions_asked += 1;
                self.signals.engagement_turns += 1;
            }
            _ => {
                self.signals.engagement_turns += 1;
            }
        }

        // Check slots for additional signals
        if slots.contains_key("phone_number") || slots.contains_key("customer_name") || slots.contains_key("phone") {
            self.signals.provided_contact_info = true;
        }
        // P16 FIX: Accept both domain-specific and generic slot names
        // Domain-specific: gold_weight, gold_purity (gold loan)
        // Generic: asset_quantity, asset_quality (from slots.yaml aliases)
        if slots.contains_key("gold_weight") || slots.contains_key("gold_purity")
            || slots.contains_key("asset_quantity") || slots.contains_key("asset_quality") {
            self.signals.provided_asset_details = true;
        }
    }

    /// Update urgency signal from text analysis
    pub fn update_urgency(&mut self, text: &str) {
        let urgency_keywords = [
            "urgent",
            "urgently",
            "immediately",
            "today",
            "now",
            "asap",
            "emergency",
            "jaldi",
            "abhi",
            "turant",
            "aaj",
            "foran",
        ];

        let text_lower = text.to_lowercase();
        for keyword in &urgency_keywords {
            if text_lower.contains(keyword) {
                self.signals.has_urgency_signal = true;
                self.signals.urgency_keywords_count += 1;
            }
        }
    }

    /// Update trust level based on conversation
    pub fn update_trust(&mut self, positive_signal: bool) {
        self.signals.trust_level = match (self.signals.trust_level, positive_signal) {
            (TrustLevel::Unknown, true) => TrustLevel::Low,
            (TrustLevel::Unknown, false) => TrustLevel::Unknown,
            (TrustLevel::Low, true) => TrustLevel::Medium,
            (TrustLevel::Low, false) => TrustLevel::Low,
            (TrustLevel::Medium, true) => TrustLevel::High,
            (TrustLevel::Medium, false) => TrustLevel::Low,
            (TrustLevel::High, true) => TrustLevel::High,
            (TrustLevel::High, false) => TrustLevel::Medium,
        };
    }

    /// Mark conversation as stalled (no meaningful progress)
    pub fn mark_stalled(&mut self) {
        self.signals.conversation_stalled_turns += 1;
    }

    /// Reset stall counter (progress made)
    pub fn reset_stall(&mut self) {
        self.signals.conversation_stalled_turns = 0;
    }

    /// Calculate current lead score
    pub fn calculate_score(&mut self) -> LeadScore {
        let breakdown = self.calculate_breakdown();
        let total = self.calculate_total(&breakdown);

        // Track score history
        self.score_history.push(total);

        let qualification = LeadQualification::from_score(total);
        let classification = LeadClassification::from_signals(&self.signals);
        let conversion_probability = self.estimate_conversion_probability(total, &classification);
        let escalation_triggers = self.check_escalation_triggers();
        let recommendation = self.generate_recommendation(&qualification, &escalation_triggers);

        LeadScore {
            total,
            qualification,
            classification,
            conversion_probability,
            breakdown,
            escalation_triggers,
            recommendation,
        }
    }

    /// Calculate score breakdown
    fn calculate_breakdown(&self) -> ScoreBreakdown {
        let signals = &self.signals;

        // Urgency score (0-25)
        let urgency = {
            let mut score = 0u32;
            if signals.has_urgency_signal {
                score += 10;
            }
            score += signals.urgency_keywords_count.min(3) * 5;
            score.min(25)
        };

        // Engagement score (0-25)
        let engagement = {
            let mut score = 0u32;
            score += signals.engagement_turns.min(5) * 3;
            score += signals.questions_asked.min(3) * 2;
            if signals.asked_about_rates {
                score += 3;
            }
            if signals.asked_for_comparison {
                score += 3;
            }
            score.min(25)
        };

        // Information completeness score (0-25)
        let information = {
            let mut score = 0u32;
            if signals.provided_contact_info {
                score += 8;
            }
            if signals.provided_asset_details {
                score += 8;
            }
            if signals.provided_loan_amount {
                score += 5;
            }
            if signals.has_specific_requirements {
                score += 4;
            }
            score.min(25)
        };

        // Intent strength score (0-25)
        let intent = {
            let mut score = 0u32;
            if signals.expressed_intent_to_proceed {
                score += 15;
            }
            if signals.requested_callback {
                score += 5;
            }
            if signals.requested_branch_visit {
                score += 8;
            }
            score += signals.trust_level.score();
            score.min(25)
        };

        // Penalty from negative signals
        let penalty = {
            let mut p = 0i32;
            if signals.expressed_disinterest {
                p -= 15;
            }
            if signals.mentioned_competitor_preference {
                p -= 10;
            }
            if signals.requested_human_agent {
                p -= 5; // Slight penalty, might indicate frustration
            }
            // Unresolved objections
            let unresolved = signals.objections_raised.saturating_sub(signals.objections_resolved);
            p -= (unresolved * 3) as i32;
            p
        };

        ScoreBreakdown {
            urgency,
            engagement,
            information,
            intent,
            penalty,
        }
    }

    /// Calculate total score from breakdown
    fn calculate_total(&self, breakdown: &ScoreBreakdown) -> u32 {
        let weights = &self.config.weights;

        let weighted_sum = (breakdown.urgency as f32 * weights.urgency)
            + (breakdown.engagement as f32 * weights.engagement)
            + (breakdown.information as f32 * weights.information)
            + (breakdown.intent as f32 * weights.intent);

        let total_with_penalty = weighted_sum as i32 + breakdown.penalty;
        total_with_penalty.max(0).min(100) as u32
    }

    /// Estimate conversion probability
    fn estimate_conversion_probability(&self, score: u32, classification: &LeadClassification) -> f32 {
        // Base probability from score
        let base = (score as f32) / 100.0;

        // Adjust based on classification
        let classification_multiplier = match classification {
            LeadClassification::Unqualified => 0.5,
            LeadClassification::MQL => 0.8,
            LeadClassification::SQL => 1.2,
        };

        // Adjust for positive/negative signals
        let signal_adjustment = if self.signals.expressed_intent_to_proceed {
            1.2
        } else if self.signals.expressed_disinterest {
            0.3
        } else {
            1.0
        };

        (base * classification_multiplier * signal_adjustment).min(0.95)
    }

    /// Check for auto-escalation triggers
    fn check_escalation_triggers(&self) -> Vec<EscalationTrigger> {
        let mut triggers = Vec::new();
        let signals = &self.signals;

        // Check objection threshold
        let unresolved_objections = signals.objections_raised.saturating_sub(signals.objections_resolved);
        if unresolved_objections >= self.config.max_objections_before_escalate {
            triggers.push(EscalationTrigger::ExcessiveObjections {
                count: unresolved_objections,
                threshold: self.config.max_objections_before_escalate,
            });
        }

        // Check stalled conversation
        if signals.conversation_stalled_turns >= self.config.max_stalled_turns {
            triggers.push(EscalationTrigger::ConversationStalled {
                turns: signals.conversation_stalled_turns,
                threshold: self.config.max_stalled_turns,
            });
        }

        // Check for customer request
        if signals.requested_human_agent {
            triggers.push(EscalationTrigger::CustomerRequested);
        }

        triggers
    }

    /// Check if loan amount triggers high-value escalation
    pub fn check_high_value_loan(&mut self, amount: f64) -> Option<EscalationTrigger> {
        if amount >= self.config.high_value_loan_threshold {
            Some(EscalationTrigger::HighValueLoan {
                amount,
                threshold: self.config.high_value_loan_threshold,
            })
        } else {
            None
        }
    }

    /// Generate recommendation based on current state
    fn generate_recommendation(
        &self,
        qualification: &LeadQualification,
        triggers: &[EscalationTrigger],
    ) -> LeadRecommendation {
        // Check escalation triggers first
        for trigger in triggers {
            match trigger {
                EscalationTrigger::CustomerRequested => {
                    return LeadRecommendation::EscalateNow {
                        reason: "Customer requested human agent".to_string(),
                    };
                }
                EscalationTrigger::ExcessiveObjections { count, .. } => {
                    return LeadRecommendation::EscalateNow {
                        reason: format!("{} unresolved objections", count),
                    };
                }
                EscalationTrigger::ConversationStalled { turns, .. } => {
                    return LeadRecommendation::OfferCallback;
                }
                EscalationTrigger::HighValueLoan { amount, .. } => {
                    return LeadRecommendation::EscalateNow {
                        reason: format!("High-value loan: ₹{:.0}", amount),
                    };
                }
                _ => {}
            }
        }

        // Recommendation based on qualification level
        match qualification {
            LeadQualification::Cold => {
                if self.signals.expressed_disinterest {
                    LeadRecommendation::LowPriority
                } else {
                    LeadRecommendation::ContinueConversation
                }
            }
            LeadQualification::Warm => {
                if self.signals.asked_about_rates || self.signals.asked_for_comparison {
                    LeadRecommendation::ContinueConversation
                } else {
                    LeadRecommendation::SendFollowUp
                }
            }
            LeadQualification::Hot => LeadRecommendation::PushForAppointment,
            LeadQualification::Qualified => {
                if self.signals.requested_callback || self.signals.requested_branch_visit {
                    LeadRecommendation::PushForAppointment
                } else {
                    LeadRecommendation::OfferCallback
                }
            }
        }
    }

    /// Get score trend (positive = improving, negative = declining)
    pub fn score_trend(&self) -> i32 {
        if self.score_history.len() < 2 {
            return 0;
        }

        let recent = self.score_history.iter().rev().take(3);
        let scores: Vec<_> = recent.cloned().collect();

        if scores.len() >= 2 {
            scores[0] as i32 - scores[scores.len() - 1] as i32
        } else {
            0
        }
    }

    /// Reset for new conversation
    pub fn reset(&mut self) {
        self.signals = LeadSignals::default();
        self.score_history.clear();
    }
}

impl Default for LeadScoringEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lead_qualification_from_score() {
        assert_eq!(LeadQualification::from_score(0), LeadQualification::Cold);
        assert_eq!(LeadQualification::from_score(29), LeadQualification::Cold);
        assert_eq!(LeadQualification::from_score(30), LeadQualification::Warm);
        assert_eq!(LeadQualification::from_score(59), LeadQualification::Warm);
        assert_eq!(LeadQualification::from_score(60), LeadQualification::Hot);
        assert_eq!(LeadQualification::from_score(79), LeadQualification::Hot);
        assert_eq!(LeadQualification::from_score(80), LeadQualification::Qualified);
        assert_eq!(LeadQualification::from_score(100), LeadQualification::Qualified);
    }

    #[test]
    fn test_lead_scoring_basic() {
        let mut engine = LeadScoringEngine::new();

        // Initial score should be low
        let score = engine.calculate_score();
        assert_eq!(score.qualification, LeadQualification::Cold);
        assert_eq!(score.classification, LeadClassification::Unqualified);
    }

    #[test]
    fn test_lead_scoring_with_engagement() {
        let mut engine = LeadScoringEngine::new();

        // Simulate strong engagement
        engine.signals_mut().engagement_turns = 5;
        engine.signals_mut().asked_about_rates = true;
        engine.signals_mut().asked_for_comparison = true;
        engine.signals_mut().provided_contact_info = true;
        engine.signals_mut().questions_asked = 3;

        let score = engine.calculate_score();
        // engagement: min(5*3 + 3*2 + 3 + 3, 25) = 25
        // information: 8 (contact info)
        // Total: 33
        assert!(score.total >= 30, "Score {} should be >= 30", score.total);
        assert_eq!(score.classification, LeadClassification::MQL);
    }

    #[test]
    fn test_lead_scoring_sql_classification() {
        let mut engine = LeadScoringEngine::new();

        // SQL criteria: urgency + contact + specific requirements + engagement
        engine.signals_mut().has_urgency_signal = true;
        engine.signals_mut().urgency_keywords_count = 2;
        engine.signals_mut().provided_contact_info = true;
        engine.signals_mut().has_specific_requirements = true;
        engine.signals_mut().provided_loan_amount = true;
        engine.signals_mut().provided_asset_details = true;
        engine.signals_mut().expressed_intent_to_proceed = true;
        engine.signals_mut().requested_branch_visit = true;
        engine.signals_mut().engagement_turns = 5;
        engine.signals_mut().trust_level = TrustLevel::High;

        let score = engine.calculate_score();
        assert_eq!(score.classification, LeadClassification::SQL);
        // urgency: 10 + 2*5 = 20
        // engagement: min(5*3, 25) = 15
        // information: 8 + 8 + 5 + 4 = 25
        // intent: 15 + 8 + 15 = 38 -> capped at 25
        // Total: 20 + 15 + 25 + 25 = 85
        assert!(score.total >= 60, "Score {} should be >= 60", score.total);
    }

    #[test]
    fn test_auto_escalation_objections() {
        let mut engine = LeadScoringEngine::new();

        // Raise objections without resolving
        engine.signals_mut().objections_raised = 4;
        engine.signals_mut().objections_resolved = 1;

        let score = engine.calculate_score();
        assert!(score.escalation_triggers.iter().any(|t| {
            matches!(t, EscalationTrigger::ExcessiveObjections { .. })
        }));
    }

    #[test]
    fn test_auto_escalation_stalled() {
        let mut engine = LeadScoringEngine::new();

        // Stalled conversation
        for _ in 0..5 {
            engine.mark_stalled();
        }

        let score = engine.calculate_score();
        assert!(score.escalation_triggers.iter().any(|t| {
            matches!(t, EscalationTrigger::ConversationStalled { .. })
        }));
    }

    #[test]
    fn test_high_value_loan_escalation() {
        let mut engine = LeadScoringEngine::new();

        // High-value loan
        let trigger = engine.check_high_value_loan(1_500_000.0);
        assert!(trigger.is_some());
        assert!(matches!(trigger, Some(EscalationTrigger::HighValueLoan { .. })));
    }

    #[test]
    fn test_urgency_detection() {
        let mut engine = LeadScoringEngine::new();

        engine.update_urgency("I need the loan urgently today");
        assert!(engine.signals().has_urgency_signal);
        assert!(engine.signals().urgency_keywords_count >= 2);
    }

    #[test]
    fn test_intent_updates() {
        let mut engine = LeadScoringEngine::new();
        let mut slots = HashMap::new();
        slots.insert("loan_amount".to_string(), "500000".to_string());

        engine.update_from_intent("loan_inquiry", &slots);
        assert!(engine.signals().provided_loan_amount);
        assert!(engine.signals().has_specific_requirements);
    }

    #[test]
    fn test_negative_signals_penalty() {
        let mut engine = LeadScoringEngine::new();

        // Add positive signals first
        engine.signals_mut().engagement_turns = 5;
        engine.signals_mut().provided_contact_info = true;
        let positive_score = engine.calculate_score().total;

        // Add negative signals
        engine.signals_mut().expressed_disinterest = true;
        engine.signals_mut().objections_raised = 2;
        let negative_score = engine.calculate_score().total;

        assert!(negative_score < positive_score);
    }

    #[test]
    fn test_score_trend() {
        let mut engine = LeadScoringEngine::new();

        // Build up engagement over multiple calculations
        engine.calculate_score(); // Low score

        engine.signals_mut().engagement_turns = 3;
        engine.calculate_score(); // Medium score

        engine.signals_mut().provided_contact_info = true;
        engine.signals_mut().expressed_intent_to_proceed = true;
        engine.calculate_score(); // Higher score

        let trend = engine.score_trend();
        assert!(trend > 0, "Score trend should be positive");
    }
}
