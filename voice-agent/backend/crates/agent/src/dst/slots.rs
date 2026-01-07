//! Gold Loan Dialogue State Slot Definitions
//!
//! Domain-specific slot schema based on LDST and ACL 2024 research.
//! Implements structured dialogue state for gold loan conversations.
//!
//! NOTE: Purity values are now config-driven. Use AgentDomainView.purity_factor()
//! to get the purity factor for a given purity ID.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Purity ID - string-based identifier for gold purity (config-driven)
/// Common IDs: "24k", "22k", "18k", "14k"
/// Purity factors are defined in config/domains/{domain}/slots.yaml
pub type PurityId = String;

/// Common purity IDs (for reference - actual purity factors come from config)
pub mod purity_ids {
    pub const K24: &str = "24k";
    pub const K22: &str = "22k";
    pub const K18: &str = "18k";
    pub const K14: &str = "14k";
    pub const UNKNOWN: &str = "unknown";
}

/// Parse purity ID from free text
pub fn parse_purity_id(text: &str) -> &'static str {
    let lower = text.to_lowercase();
    if lower.contains("24") {
        purity_ids::K24
    } else if lower.contains("22") {
        purity_ids::K22
    } else if lower.contains("18") {
        purity_ids::K18
    } else if lower.contains("14") {
        purity_ids::K14
    } else {
        purity_ids::UNKNOWN
    }
}

/// Format purity ID for display
pub fn format_purity_display(purity_id: &str) -> &'static str {
    match purity_id {
        "24k" => "24 karat",
        "22k" => "22 karat",
        "18k" => "18 karat",
        "14k" => "14 karat",
        _ => "unknown purity",
    }
}

/// Goal ID - string-based goal identifier (config-driven)
///
/// Goals are defined in config/domains/{domain}/goals.yaml
/// Common goal IDs: "exploration", "balance_transfer", "new_loan",
/// "eligibility_check", "branch_visit", "lead_capture"
pub type GoalId = String;

/// Default goal ID
pub const DEFAULT_GOAL: &str = "exploration";

/// Next best action for the agent
#[derive(Debug, Clone, PartialEq)]
pub enum NextBestAction {
    /// Call a specific tool
    CallTool(String),
    /// Ask for a specific slot
    AskFor(String),
    /// Offer to schedule an appointment
    OfferAppointment,
    /// Explain the process (e.g., balance transfer process)
    ExplainProcess,
    /// Discover customer intent first
    DiscoverIntent,
    /// Capture lead now
    CaptureLead,
}

impl NextBestAction {
    /// Convert to instruction for LLM
    pub fn to_instruction(&self) -> String {
        match self {
            NextBestAction::CallTool(tool) => format!("CALL the {} tool now with available information", tool),
            NextBestAction::AskFor(slot) => format!("ASK customer for their {} (required to proceed)", slot.replace("_", " ")),
            NextBestAction::OfferAppointment => "OFFER to schedule a branch visit appointment".to_string(),
            NextBestAction::ExplainProcess => "EXPLAIN the balance transfer process: Kotak pays off current lender directly, no cash needed from customer".to_string(),
            NextBestAction::DiscoverIntent => "ASK what brings them to Kotak Gold Loan today".to_string(),
            NextBestAction::CaptureLead => "CAPTURE customer details for follow-up (name and phone)".to_string(),
        }
    }
}

/// Urgency level for loan requirement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UrgencyLevel {
    /// Immediate need (today/tomorrow)
    Immediate,
    /// Soon (within a week)
    Soon,
    /// Planning ahead (no specific timeline)
    Planning,
    /// Just exploring options
    Exploring,
}

impl UrgencyLevel {
    /// Parse from utterance context
    pub fn from_utterance(text: &str) -> Option<Self> {
        let lower = text.to_lowercase();

        // Immediate indicators
        if lower.contains("urgent") || lower.contains("today") || lower.contains("now")
            || lower.contains("immediately") || lower.contains("abhi") || lower.contains("turant")
            || lower.contains("aaj") || lower.contains("emergency")
        {
            return Some(UrgencyLevel::Immediate);
        }

        // Soon indicators
        if lower.contains("this week") || lower.contains("few days") || lower.contains("jaldi")
            || lower.contains("soon") || lower.contains("is hafte")
        {
            return Some(UrgencyLevel::Soon);
        }

        // Planning indicators
        if lower.contains("next month") || lower.contains("planning") || lower.contains("thinking")
            || lower.contains("soch") || lower.contains("agle mahine")
        {
            return Some(UrgencyLevel::Planning);
        }

        // Exploring indicators
        if lower.contains("just checking") || lower.contains("exploring") || lower.contains("options")
            || lower.contains("jaankari") || lower.contains("information")
        {
            return Some(UrgencyLevel::Exploring);
        }

        None
    }
}

impl std::fmt::Display for UrgencyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UrgencyLevel::Immediate => write!(f, "immediate"),
            UrgencyLevel::Soon => write!(f, "soon"),
            UrgencyLevel::Planning => write!(f, "planning"),
            UrgencyLevel::Exploring => write!(f, "exploring"),
        }
    }
}

/// A slot value with confidence and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotValue {
    /// The value as a string
    pub value: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Turn index when this was set
    pub turn_set: usize,
    /// Whether user confirmed this value
    pub confirmed: bool,
}

impl SlotValue {
    /// Create a new slot value
    pub fn new(value: impl Into<String>, confidence: f32, turn: usize) -> Self {
        Self {
            value: value.into(),
            confidence,
            turn_set: turn,
            confirmed: false,
        }
    }

    /// Mark as confirmed
    pub fn confirm(&mut self) {
        self.confirmed = true;
        self.confidence = 1.0;
    }
}

/// Gold Loan Dialogue State
///
/// Tracks all slot values relevant to a gold loan conversation.
/// Implements domain-specific slot schema based on gold loan business logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldLoanDialogueState {
    // ====== Customer Information ======
    /// Customer name
    customer_name: Option<SlotValue>,
    /// Phone number
    phone_number: Option<SlotValue>,
    /// Location/city
    location: Option<SlotValue>,
    /// Pincode
    pincode: Option<SlotValue>,

    // ====== Gold Details ======
    /// Gold weight in grams
    gold_weight_grams: Option<SlotValue>,
    /// Gold purity (karat)
    gold_purity: Option<SlotValue>,
    /// Type of gold item (jewelry, coins, bars)
    gold_item_type: Option<SlotValue>,

    // ====== Loan Requirements ======
    /// Desired loan amount
    loan_amount: Option<SlotValue>,
    /// Loan purpose
    loan_purpose: Option<SlotValue>,
    /// Preferred tenure (months)
    loan_tenure: Option<SlotValue>,
    /// Urgency level
    urgency: Option<SlotValue>,

    // ====== Existing Loan (for balance transfer) ======
    /// Current lender
    current_lender: Option<SlotValue>,
    /// Current outstanding amount
    current_outstanding: Option<SlotValue>,
    /// Current interest rate
    current_interest_rate: Option<SlotValue>,

    // ====== Scheduling ======
    /// Preferred visit date
    preferred_date: Option<SlotValue>,
    /// Preferred time slot
    preferred_time: Option<SlotValue>,
    /// Branch preference
    preferred_branch: Option<SlotValue>,

    // ====== Intent Tracking ======
    /// Primary detected intent
    primary_intent: Option<String>,
    /// Intent confidence
    intent_confidence: f32,
    /// Secondary intents detected
    secondary_intents: Vec<String>,

    // ====== Goal Tracking ======
    /// Current conversation goal ID (config-driven)
    conversation_goal: GoalId,
    /// Whether goal has been explicitly set (vs inferred)
    goal_confirmed: bool,
    /// Turn at which goal was set
    goal_set_turn: usize,

    // ====== State Management ======
    /// Slots pending confirmation
    pending_slots: HashSet<String>,
    /// Confirmed slots
    confirmed_slots: HashSet<String>,
    /// Custom/dynamic slots
    custom_slots: HashMap<String, SlotValue>,
}

impl Default for GoldLoanDialogueState {
    fn default() -> Self {
        Self {
            customer_name: None,
            phone_number: None,
            location: None,
            pincode: None,
            gold_weight_grams: None,
            gold_purity: None,
            gold_item_type: None,
            loan_amount: None,
            loan_purpose: None,
            loan_tenure: None,
            urgency: None,
            current_lender: None,
            current_outstanding: None,
            current_interest_rate: None,
            preferred_date: None,
            preferred_time: None,
            preferred_branch: None,
            primary_intent: None,
            intent_confidence: 0.0,
            secondary_intents: Vec::new(),
            // Phase 2 fix: Set default goal to exploration
            conversation_goal: DEFAULT_GOAL.to_string(),
            goal_confirmed: false,
            goal_set_turn: 0,
            pending_slots: HashSet::new(),
            confirmed_slots: HashSet::new(),
            custom_slots: HashMap::new(),
        }
    }
}

impl GoldLoanDialogueState {
    /// Create a new empty state
    pub fn new() -> Self {
        Self::default()
    }

    // ====== Customer Information Accessors ======

    /// Get customer name
    pub fn customer_name(&self) -> Option<&str> {
        self.customer_name.as_ref().map(|v| v.value.as_str())
    }

    /// Get phone number
    pub fn phone_number(&self) -> Option<&str> {
        self.phone_number.as_ref().map(|v| v.value.as_str())
    }

    /// Get location
    pub fn location(&self) -> Option<&str> {
        self.location.as_ref().map(|v| v.value.as_str())
    }

    /// Get pincode
    pub fn pincode(&self) -> Option<&str> {
        self.pincode.as_ref().map(|v| v.value.as_str())
    }

    // ====== Gold Details Accessors ======

    /// Get gold weight in grams
    pub fn gold_weight_grams(&self) -> Option<f64> {
        self.gold_weight_grams
            .as_ref()
            .and_then(|v| v.value.parse().ok())
    }

    /// Get gold purity ID (config-driven)
    /// Use AgentDomainView.purity_factor() to get the actual factor
    pub fn gold_purity(&self) -> Option<&'static str> {
        self.gold_purity
            .as_ref()
            .map(|v| parse_purity_id(&v.value))
    }

    /// Get gold purity as raw slot value
    pub fn gold_purity_raw(&self) -> Option<&str> {
        self.gold_purity.as_ref().map(|v| v.value.as_str())
    }

    /// Get gold item type
    pub fn gold_item_type(&self) -> Option<&str> {
        self.gold_item_type.as_ref().map(|v| v.value.as_str())
    }

    // ====== Loan Requirements Accessors ======

    /// Get loan amount
    pub fn loan_amount(&self) -> Option<f64> {
        self.loan_amount
            .as_ref()
            .and_then(|v| v.value.parse().ok())
    }

    /// Get loan purpose
    pub fn loan_purpose(&self) -> Option<&str> {
        self.loan_purpose.as_ref().map(|v| v.value.as_str())
    }

    /// Get loan tenure in months
    pub fn loan_tenure(&self) -> Option<u32> {
        self.loan_tenure
            .as_ref()
            .and_then(|v| v.value.parse().ok())
    }

    /// Get urgency level
    pub fn urgency(&self) -> Option<UrgencyLevel> {
        self.urgency.as_ref().and_then(|v| {
            match v.value.to_lowercase().as_str() {
                "immediate" => Some(UrgencyLevel::Immediate),
                "soon" => Some(UrgencyLevel::Soon),
                "planning" => Some(UrgencyLevel::Planning),
                "exploring" => Some(UrgencyLevel::Exploring),
                _ => None,
            }
        })
    }

    // ====== Existing Loan Accessors ======

    /// Get current lender
    pub fn current_lender(&self) -> Option<&str> {
        self.current_lender.as_ref().map(|v| v.value.as_str())
    }

    /// Get current outstanding amount
    pub fn current_outstanding(&self) -> Option<f64> {
        self.current_outstanding
            .as_ref()
            .and_then(|v| v.value.parse().ok())
    }

    /// Get current interest rate
    pub fn current_interest_rate(&self) -> Option<f32> {
        self.current_interest_rate
            .as_ref()
            .and_then(|v| v.value.parse().ok())
    }

    // ====== Scheduling Accessors ======

    /// Get preferred date
    pub fn preferred_date(&self) -> Option<&str> {
        self.preferred_date.as_ref().map(|v| v.value.as_str())
    }

    /// Get preferred time
    pub fn preferred_time(&self) -> Option<&str> {
        self.preferred_time.as_ref().map(|v| v.value.as_str())
    }

    /// Get preferred branch
    pub fn preferred_branch(&self) -> Option<&str> {
        self.preferred_branch.as_ref().map(|v| v.value.as_str())
    }

    // ====== Intent Accessors ======

    /// Get primary intent
    pub fn primary_intent(&self) -> Option<&str> {
        self.primary_intent.as_deref()
    }

    /// Get intent confidence
    pub fn intent_confidence(&self) -> f32 {
        self.intent_confidence
    }

    /// Get secondary intents
    pub fn secondary_intents(&self) -> &[String] {
        &self.secondary_intents
    }

    /// Update primary intent
    pub fn update_intent(&mut self, intent: &str, confidence: f32) {
        // If we already have this intent, just update confidence
        if self.primary_intent.as_deref() == Some(intent) {
            self.intent_confidence = confidence;
            return;
        }

        // Move current primary to secondary if exists
        if let Some(ref prev) = self.primary_intent {
            if !self.secondary_intents.contains(prev) {
                self.secondary_intents.push(prev.clone());
            }
        }

        self.primary_intent = Some(intent.to_string());
        self.intent_confidence = confidence;
    }

    // ====== Goal Tracking Methods ======

    /// Get the current conversation goal ID
    pub fn goal_id(&self) -> &str {
        &self.conversation_goal
    }

    /// Check if goal is confirmed (explicit) vs inferred
    pub fn is_goal_confirmed(&self) -> bool {
        self.goal_confirmed
    }

    /// Set the conversation goal by ID
    ///
    /// Use the goal schema to map intents to goals:
    /// ```ignore
    /// if let Some(goal_id) = schema.goal_for_intent(intent) {
    ///     state.set_goal(goal_id, turn);
    /// }
    /// ```
    pub fn set_goal(&mut self, goal_id: &str, turn: usize) {
        // Only update if it's a meaningful change (not downgrading to exploration)
        if goal_id != DEFAULT_GOAL || self.conversation_goal == DEFAULT_GOAL {
            self.conversation_goal = goal_id.to_string();
            self.goal_set_turn = turn;
        }
    }

    /// Set the conversation goal explicitly (confirmed by user)
    pub fn confirm_goal(&mut self, goal_id: &str, turn: usize) {
        self.conversation_goal = goal_id.to_string();
        self.goal_confirmed = true;
        self.goal_set_turn = turn;
    }

    /// Get the turn at which the goal was set
    pub fn goal_set_turn(&self) -> usize {
        self.goal_set_turn
    }

    /// Check if we should auto-capture lead (when we have contact info during any goal)
    /// Returns true if lead capture should be triggered as a secondary action
    pub fn should_auto_capture_lead(&self) -> bool {
        // Don't duplicate if already in lead capture mode
        if self.conversation_goal == "lead_capture" {
            return false;
        }

        // Capture lead if we have both name and phone collected
        self.customer_name.is_some() && self.phone_number.is_some()
    }

    /// Check if we have complete contact info
    pub fn has_complete_contact(&self) -> bool {
        self.customer_name.is_some() && self.phone_number.is_some()
    }

    // ====== State Management ======

    /// Get slots pending confirmation
    pub fn pending_slots(&self) -> &HashSet<String> {
        &self.pending_slots
    }

    /// Get confirmed slots
    pub fn confirmed_slots(&self) -> &HashSet<String> {
        &self.confirmed_slots
    }

    /// Get slots pending confirmation with their values
    /// Returns a list of (slot_name, slot_value) pairs
    pub fn slots_needing_confirmation(&self) -> Vec<(&str, String)> {
        self.pending_slots
            .iter()
            .filter_map(|slot_name| {
                self.get_slot_value(slot_name)
                    .map(|value| (slot_name.as_str(), value))
            })
            .collect()
    }

    /// Generate a confirmation prompt for pending slots
    ///
    /// Returns None if no slots need confirmation.
    /// For small models, this helps ensure critical values are verified.
    ///
    /// Example output: "Please confirm: loan amount ₹5 lakh, gold weight 50g"
    pub fn pending_confirmation_prompt(&self) -> Option<String> {
        let pending = self.slots_needing_confirmation();
        if pending.is_empty() {
            return None;
        }

        let formatted: Vec<String> = pending
            .iter()
            .map(|(slot, value)| {
                let display_name = slot.replace("_", " ");
                let formatted_value = Self::format_slot_value_for_display(slot, value);
                format!("{}: {}", display_name, formatted_value)
            })
            .collect();

        Some(format!("Please confirm: {}", formatted.join(", ")))
    }

    /// Format a slot value for human-readable display
    fn format_slot_value_for_display(slot_name: &str, value: &str) -> String {
        match slot_name {
            "loan_amount" | "current_outstanding" => {
                if let Ok(amount) = value.parse::<f64>() {
                    if amount >= 10_000_000.0 {
                        format!("₹{:.1} crore", amount / 10_000_000.0)
                    } else if amount >= 100_000.0 {
                        format!("₹{:.1} lakh", amount / 100_000.0)
                    } else {
                        format!("₹{:.0}", amount)
                    }
                } else {
                    value.to_string()
                }
            }
            "gold_weight" => {
                if let Ok(weight) = value.parse::<f64>() {
                    format!("{}g", weight)
                } else {
                    value.to_string()
                }
            }
            "current_interest_rate" => {
                if let Ok(rate) = value.parse::<f32>() {
                    format!("{}%", rate)
                } else {
                    value.to_string()
                }
            }
            _ => value.to_string(),
        }
    }

    /// Get key slots that should always be confirmed before tool invocation
    /// These are the critical values that affect loan calculations
    pub fn critical_slots_for_confirmation(&self) -> Vec<(&str, String)> {
        let critical = ["loan_amount", "gold_weight", "current_outstanding", "current_interest_rate"];

        critical
            .iter()
            .filter_map(|slot| {
                self.get_slot_value(slot)
                    .map(|value| (*slot, value))
            })
            .filter(|(slot, _)| !self.confirmed_slots.contains(*slot))
            .collect()
    }

    /// Generate a prompt for critical value confirmation
    /// Used especially for small models before tool invocation
    pub fn critical_confirmation_prompt(&self) -> Option<String> {
        let critical = self.critical_slots_for_confirmation();
        if critical.is_empty() {
            return None;
        }

        let formatted: Vec<String> = critical
            .iter()
            .map(|(slot, value)| {
                let display_name = slot.replace("_", " ");
                let formatted_value = Self::format_slot_value_for_display(slot, value);
                format!("{}: {}", display_name, formatted_value)
            })
            .collect();

        Some(format!(
            "Before proceeding, please confirm these values are correct: {}",
            formatted.join(", ")
        ))
    }

    /// Mark a slot as pending confirmation
    pub fn mark_pending(&mut self, slot_name: &str) {
        self.confirmed_slots.remove(slot_name);
        self.pending_slots.insert(slot_name.to_string());
    }

    /// Mark a slot as confirmed
    pub fn mark_confirmed(&mut self, slot_name: &str) {
        self.pending_slots.remove(slot_name);
        self.confirmed_slots.insert(slot_name.to_string());

        // Also update the slot value's confirmed flag
        match slot_name {
            "customer_name" => { if let Some(ref mut v) = self.customer_name { v.confirm(); } }
            "phone_number" => { if let Some(ref mut v) = self.phone_number { v.confirm(); } }
            "location" => { if let Some(ref mut v) = self.location { v.confirm(); } }
            "pincode" => { if let Some(ref mut v) = self.pincode { v.confirm(); } }
            "gold_weight" => { if let Some(ref mut v) = self.gold_weight_grams { v.confirm(); } }
            "gold_purity" => { if let Some(ref mut v) = self.gold_purity { v.confirm(); } }
            "gold_item_type" => { if let Some(ref mut v) = self.gold_item_type { v.confirm(); } }
            "loan_amount" => { if let Some(ref mut v) = self.loan_amount { v.confirm(); } }
            "loan_purpose" => { if let Some(ref mut v) = self.loan_purpose { v.confirm(); } }
            "loan_tenure" => { if let Some(ref mut v) = self.loan_tenure { v.confirm(); } }
            "urgency" => { if let Some(ref mut v) = self.urgency { v.confirm(); } }
            "current_lender" => { if let Some(ref mut v) = self.current_lender { v.confirm(); } }
            "current_outstanding" => { if let Some(ref mut v) = self.current_outstanding { v.confirm(); } }
            "current_interest_rate" => { if let Some(ref mut v) = self.current_interest_rate { v.confirm(); } }
            "preferred_date" => { if let Some(ref mut v) = self.preferred_date { v.confirm(); } }
            "preferred_time" => { if let Some(ref mut v) = self.preferred_time { v.confirm(); } }
            "preferred_branch" => { if let Some(ref mut v) = self.preferred_branch { v.confirm(); } }
            _ => {
                if let Some(v) = self.custom_slots.get_mut(slot_name) {
                    v.confirm();
                }
            }
        }
    }

    // ====== Generic Slot Access ======

    /// Get slot value by name
    pub fn get_slot_value(&self, slot_name: &str) -> Option<String> {
        match slot_name {
            "customer_name" => self.customer_name.as_ref().map(|v| v.value.clone()),
            "phone_number" => self.phone_number.as_ref().map(|v| v.value.clone()),
            "location" => self.location.as_ref().map(|v| v.value.clone()),
            "pincode" => self.pincode.as_ref().map(|v| v.value.clone()),
            "gold_weight" => self.gold_weight_grams.as_ref().map(|v| v.value.clone()),
            "gold_purity" => self.gold_purity.as_ref().map(|v| v.value.clone()),
            "gold_item_type" => self.gold_item_type.as_ref().map(|v| v.value.clone()),
            "loan_amount" => self.loan_amount.as_ref().map(|v| v.value.clone()),
            "loan_purpose" => self.loan_purpose.as_ref().map(|v| v.value.clone()),
            "loan_tenure" => self.loan_tenure.as_ref().map(|v| v.value.clone()),
            "urgency" => self.urgency.as_ref().map(|v| v.value.clone()),
            "current_lender" => self.current_lender.as_ref().map(|v| v.value.clone()),
            "current_outstanding" => self.current_outstanding.as_ref().map(|v| v.value.clone()),
            "current_interest_rate" => self.current_interest_rate.as_ref().map(|v| v.value.clone()),
            "preferred_date" => self.preferred_date.as_ref().map(|v| v.value.clone()),
            "preferred_time" => self.preferred_time.as_ref().map(|v| v.value.clone()),
            "preferred_branch" => self.preferred_branch.as_ref().map(|v| v.value.clone()),
            _ => self.custom_slots.get(slot_name).map(|v| v.value.clone()),
        }
    }

    /// Get slot with confidence
    pub fn get_slot_with_confidence(&self, slot_name: &str) -> Option<&SlotValue> {
        match slot_name {
            "customer_name" => self.customer_name.as_ref(),
            "phone_number" => self.phone_number.as_ref(),
            "location" => self.location.as_ref(),
            "pincode" => self.pincode.as_ref(),
            "gold_weight" => self.gold_weight_grams.as_ref(),
            "gold_purity" => self.gold_purity.as_ref(),
            "gold_item_type" => self.gold_item_type.as_ref(),
            "loan_amount" => self.loan_amount.as_ref(),
            "loan_purpose" => self.loan_purpose.as_ref(),
            "loan_tenure" => self.loan_tenure.as_ref(),
            "urgency" => self.urgency.as_ref(),
            "current_lender" => self.current_lender.as_ref(),
            "current_outstanding" => self.current_outstanding.as_ref(),
            "current_interest_rate" => self.current_interest_rate.as_ref(),
            "preferred_date" => self.preferred_date.as_ref(),
            "preferred_time" => self.preferred_time.as_ref(),
            "preferred_branch" => self.preferred_branch.as_ref(),
            _ => self.custom_slots.get(slot_name),
        }
    }

    /// Set slot value by name
    pub fn set_slot_value(&mut self, slot_name: &str, value: &str, confidence: f32) {
        let slot_value = SlotValue::new(value, confidence, 0);

        match slot_name {
            "customer_name" => self.customer_name = Some(slot_value),
            "phone_number" => self.phone_number = Some(slot_value),
            "location" => self.location = Some(slot_value),
            "pincode" => self.pincode = Some(slot_value),
            "gold_weight" => self.gold_weight_grams = Some(slot_value),
            "gold_purity" => self.gold_purity = Some(slot_value),
            "gold_item_type" => self.gold_item_type = Some(slot_value),
            "loan_amount" => self.loan_amount = Some(slot_value),
            "loan_purpose" => self.loan_purpose = Some(slot_value),
            "loan_tenure" => self.loan_tenure = Some(slot_value),
            "urgency" => self.urgency = Some(slot_value),
            "current_lender" => self.current_lender = Some(slot_value),
            "current_outstanding" => self.current_outstanding = Some(slot_value),
            "current_interest_rate" => self.current_interest_rate = Some(slot_value),
            "preferred_date" => self.preferred_date = Some(slot_value),
            "preferred_time" => self.preferred_time = Some(slot_value),
            "preferred_branch" => self.preferred_branch = Some(slot_value),
            _ => {
                self.custom_slots.insert(slot_name.to_string(), slot_value);
            }
        }
    }

    /// Clear a slot
    pub fn clear_slot(&mut self, slot_name: &str) {
        self.pending_slots.remove(slot_name);
        self.confirmed_slots.remove(slot_name);

        match slot_name {
            "customer_name" => self.customer_name = None,
            "phone_number" => self.phone_number = None,
            "location" => self.location = None,
            "pincode" => self.pincode = None,
            "gold_weight" => self.gold_weight_grams = None,
            "gold_purity" => self.gold_purity = None,
            "gold_item_type" => self.gold_item_type = None,
            "loan_amount" => self.loan_amount = None,
            "loan_purpose" => self.loan_purpose = None,
            "loan_tenure" => self.loan_tenure = None,
            "urgency" => self.urgency = None,
            "current_lender" => self.current_lender = None,
            "current_outstanding" => self.current_outstanding = None,
            "current_interest_rate" => self.current_interest_rate = None,
            "preferred_date" => self.preferred_date = None,
            "preferred_time" => self.preferred_time = None,
            "preferred_branch" => self.preferred_branch = None,
            _ => {
                self.custom_slots.remove(slot_name);
            }
        }
    }

    /// Convert state to context string for LLM prompts
    pub fn to_context_string(&self) -> String {
        let mut parts = Vec::new();

        // Customer info
        if let Some(name) = self.customer_name() {
            parts.push(format!("Customer: {}", name));
        }
        if let Some(phone) = self.phone_number() {
            parts.push(format!("Phone: {}", phone));
        }
        if let Some(loc) = self.location() {
            parts.push(format!("Location: {}", loc));
        }

        // Gold details
        if let Some(weight) = self.gold_weight_grams() {
            parts.push(format!("Gold weight: {}g", weight));
        }
        if let Some(purity) = self.gold_purity() {
            parts.push(format!("Purity: {}", purity));
        }

        // Loan requirements
        if let Some(amount) = self.loan_amount() {
            let formatted = if amount >= 100_000.0 {
                format!("₹{:.1} lakh", amount / 100_000.0)
            } else {
                format!("₹{:.0}", amount)
            };
            parts.push(format!("Loan amount: {}", formatted));
        }
        if let Some(purpose) = self.loan_purpose() {
            parts.push(format!("Purpose: {}", purpose));
        }

        // Existing loan (for balance transfer)
        if let Some(lender) = self.current_lender() {
            parts.push(format!("Current lender: {}", lender));
        }
        if let Some(outstanding) = self.current_outstanding() {
            parts.push(format!("Outstanding: ₹{:.0}", outstanding));
        }
        if let Some(rate) = self.current_interest_rate() {
            parts.push(format!("Current rate: {}%", rate));
        }

        // Intent
        if let Some(intent) = self.primary_intent() {
            parts.push(format!("Intent: {}", intent));
        }

        if parts.is_empty() {
            "No information collected yet.".to_string()
        } else {
            parts.join("\n")
        }
    }

    /// Convert state to full context string including goal information
    pub fn to_full_context_string(&self) -> String {
        let mut output = String::new();

        // Collected information
        output.push_str("# Customer Information\n");
        output.push_str(&self.to_context_string());
        output.push_str("\n\n");

        // Goal info
        output.push_str(&format!("# Current Goal: {}\n", self.conversation_goal));

        output
    }

    /// Get all filled slot names
    pub fn filled_slots(&self) -> Vec<&str> {
        let mut slots = Vec::new();

        if self.customer_name.is_some() { slots.push("customer_name"); }
        if self.phone_number.is_some() { slots.push("phone_number"); }
        if self.location.is_some() { slots.push("location"); }
        if self.pincode.is_some() { slots.push("pincode"); }
        if self.gold_weight_grams.is_some() { slots.push("gold_weight"); }
        if self.gold_purity.is_some() { slots.push("gold_purity"); }
        if self.gold_item_type.is_some() { slots.push("gold_item_type"); }
        if self.loan_amount.is_some() { slots.push("loan_amount"); }
        if self.loan_purpose.is_some() { slots.push("loan_purpose"); }
        if self.loan_tenure.is_some() { slots.push("loan_tenure"); }
        if self.urgency.is_some() { slots.push("urgency"); }
        if self.current_lender.is_some() { slots.push("current_lender"); }
        if self.current_outstanding.is_some() { slots.push("current_outstanding"); }
        if self.current_interest_rate.is_some() { slots.push("current_interest_rate"); }
        if self.preferred_date.is_some() { slots.push("preferred_date"); }
        if self.preferred_time.is_some() { slots.push("preferred_time"); }
        if self.preferred_branch.is_some() { slots.push("preferred_branch"); }

        for key in self.custom_slots.keys() {
            slots.push(key.as_str());
        }

        slots
    }

    /// Get the next best action for the agent based on current state
    ///
    /// This method analyzes the current goal and filled slots to recommend
    /// what the agent should do next.
    pub fn next_best_action(&self) -> NextBestAction {
        // Based on current goal, determine what to do next
        match self.conversation_goal.as_str() {
            "exploration" | "" => {
                // Still exploring - discover intent first
                NextBestAction::DiscoverIntent
            }
            "balance_transfer" => {
                // Need current lender info to calculate savings
                if self.current_lender.is_none() {
                    return NextBestAction::AskFor("current_lender".to_string());
                }
                if self.current_outstanding.is_none() {
                    return NextBestAction::AskFor("current_outstanding".to_string());
                }
                if self.current_interest_rate.is_none() {
                    return NextBestAction::AskFor("current_interest_rate".to_string());
                }
                // Have all info - calculate savings
                NextBestAction::CallTool("calculate_savings".to_string())
            }
            "eligibility_check" | "new_loan" => {
                // Need gold details
                if self.gold_weight_grams.is_none() {
                    return NextBestAction::AskFor("gold_weight".to_string());
                }
                // Can check eligibility
                NextBestAction::CallTool("check_eligibility".to_string())
            }
            "branch_visit" => {
                // Need location
                if self.location.is_none() {
                    return NextBestAction::AskFor("location".to_string());
                }
                // Offer appointment
                NextBestAction::OfferAppointment
            }
            "lead_capture" => {
                // Need contact info
                if self.customer_name.is_none() {
                    return NextBestAction::AskFor("customer_name".to_string());
                }
                if self.phone_number.is_none() {
                    return NextBestAction::AskFor("phone_number".to_string());
                }
                // Have contact info - capture lead
                NextBestAction::CaptureLead
            }
            _ => {
                // Unknown goal - discover intent
                NextBestAction::DiscoverIntent
            }
        }
    }

    /// Calculate completion percentage for a given intent
    pub fn completion_for_intent(&self, intent: &str) -> f32 {
        let (filled, required) = match intent {
            "eligibility_check" => {
                let required = ["gold_weight"];
                let filled = required.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                (filled, required.len())
            }
            "loan_inquiry" => {
                let required = ["loan_amount"];
                let optional = ["gold_weight", "gold_purity", "loan_tenure"];
                let filled_req = required.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                let filled_opt = optional.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                // Weight: required slots 70%, optional 30%
                let score = (filled_req as f32 / required.len() as f32) * 0.7
                    + (filled_opt as f32 / optional.len() as f32) * 0.3;
                return score;
            }
            "switch_lender" | "balance_transfer" => {
                let required = ["current_lender"];
                let filled = required.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                (filled, required.len())
            }
            "schedule_visit" => {
                let required = ["location"];
                let optional = ["preferred_date", "preferred_time"];
                let filled_req = required.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                let filled_opt = optional.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                let score = (filled_req as f32 / required.len() as f32) * 0.6
                    + (filled_opt as f32 / optional.len() as f32) * 0.4;
                return score;
            }
            "send_sms" | "contact_callback" => {
                let required = ["phone_number"];
                let filled = required.iter().filter(|s| self.get_slot_value(s).is_some()).count();
                (filled, required.len())
            }
            _ => return 1.0, // Unknown intents are always "complete"
        };

        if required == 0 {
            1.0
        } else {
            filled as f32 / required as f32
        }
    }
}

// =============================================================================
// Phase 2: DialogueState Trait Implementation
// =============================================================================

impl super::DialogueState for GoldLoanDialogueState {
    fn primary_intent(&self) -> Option<&str> {
        self.primary_intent.as_deref()
    }

    fn get_slot_value(&self, slot_name: &str) -> Option<String> {
        GoldLoanDialogueState::get_slot_value(self, slot_name)
    }

    fn set_slot_value(&mut self, slot_name: &str, value: &str, confidence: f32) {
        GoldLoanDialogueState::set_slot_value(self, slot_name, value, confidence)
    }

    fn clear_slot(&mut self, slot_name: &str) {
        GoldLoanDialogueState::clear_slot(self, slot_name)
    }

    fn filled_slots(&self) -> Vec<&str> {
        GoldLoanDialogueState::filled_slots(self)
    }

    fn pending_slots(&self) -> &HashSet<String> {
        &self.pending_slots
    }

    fn confirmed_slots(&self) -> &HashSet<String> {
        &self.confirmed_slots
    }

    fn mark_pending(&mut self, slot_name: &str) {
        GoldLoanDialogueState::mark_pending(self, slot_name)
    }

    fn mark_confirmed(&mut self, slot_name: &str) {
        GoldLoanDialogueState::mark_confirmed(self, slot_name)
    }

    fn goal_id(&self) -> &str {
        &self.conversation_goal
    }

    fn set_goal(&mut self, goal_id: &str, turn: usize) {
        GoldLoanDialogueState::set_goal(self, goal_id, turn)
    }

    fn confirm_goal(&mut self, goal_id: &str, turn: usize) {
        GoldLoanDialogueState::confirm_goal(self, goal_id, turn)
    }

    fn should_auto_capture_lead(&self) -> bool {
        GoldLoanDialogueState::should_auto_capture_lead(self)
    }

    fn to_context_string(&self) -> String {
        GoldLoanDialogueState::to_context_string(self)
    }

    fn to_full_context_string(&self) -> String {
        GoldLoanDialogueState::to_full_context_string(self)
    }

    fn update_intent(&mut self, intent: &str, confidence: f32) {
        GoldLoanDialogueState::update_intent(self, intent, confidence)
    }

    fn get_slot_with_confidence(&self, slot_name: &str) -> Option<&SlotValue> {
        GoldLoanDialogueState::get_slot_with_confidence(self, slot_name)
    }

    fn next_best_action(&self) -> NextBestAction {
        GoldLoanDialogueState::next_best_action(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gold_purity_parsing() {
        // Now uses string-based purity IDs (config-driven)
        assert_eq!(parse_purity_id("24k gold"), purity_ids::K24);
        assert_eq!(parse_purity_id("22 karat"), purity_ids::K22);
        assert_eq!(parse_purity_id("18kt"), purity_ids::K18);
        assert_eq!(parse_purity_id("pure gold"), purity_ids::UNKNOWN);
    }

    #[test]
    fn test_gold_purity_display() {
        // Display formatting
        assert_eq!(format_purity_display(purity_ids::K24), "24 karat");
        assert_eq!(format_purity_display(purity_ids::K22), "22 karat");
        // NOTE: Actual purity factors should come from config via AgentDomainView.purity_factor()
    }

    #[test]
    fn test_urgency_detection() {
        assert_eq!(UrgencyLevel::from_utterance("I need it today"), Some(UrgencyLevel::Immediate));
        assert_eq!(UrgencyLevel::from_utterance("mujhe abhi chahiye"), Some(UrgencyLevel::Immediate));
        assert_eq!(UrgencyLevel::from_utterance("this week sometime"), Some(UrgencyLevel::Soon));
        assert_eq!(UrgencyLevel::from_utterance("just exploring options"), Some(UrgencyLevel::Exploring));
    }

    #[test]
    fn test_state_creation() {
        let state = GoldLoanDialogueState::new();
        assert!(state.customer_name().is_none());
        assert!(state.loan_amount().is_none());
        assert!(state.filled_slots().is_empty());
    }

    #[test]
    fn test_slot_set_and_get() {
        let mut state = GoldLoanDialogueState::new();

        state.set_slot_value("customer_name", "Rahul", 0.9);
        state.set_slot_value("loan_amount", "500000", 0.85);

        assert_eq!(state.customer_name(), Some("Rahul"));
        assert_eq!(state.loan_amount(), Some(500000.0));
    }

    #[test]
    fn test_slot_confirmation() {
        let mut state = GoldLoanDialogueState::new();

        state.set_slot_value("gold_weight", "50", 0.8);
        state.mark_pending("gold_weight");

        assert!(state.pending_slots().contains("gold_weight"));
        assert!(!state.confirmed_slots().contains("gold_weight"));

        state.mark_confirmed("gold_weight");

        assert!(!state.pending_slots().contains("gold_weight"));
        assert!(state.confirmed_slots().contains("gold_weight"));
    }

    #[test]
    fn test_custom_slots() {
        let mut state = GoldLoanDialogueState::new();

        state.set_slot_value("custom_field", "custom_value", 0.9);

        assert_eq!(state.get_slot_value("custom_field"), Some("custom_value".to_string()));
    }

    #[test]
    fn test_context_string() {
        let mut state = GoldLoanDialogueState::new();

        state.set_slot_value("customer_name", "Rahul", 0.9);
        state.set_slot_value("loan_amount", "500000", 0.9);
        state.set_slot_value("gold_weight", "50", 0.9);

        let context = state.to_context_string();
        assert!(context.contains("Rahul"));
        assert!(context.contains("5.0 lakh"));
        assert!(context.contains("50g"));
    }

    #[test]
    fn test_intent_completion() {
        let mut state = GoldLoanDialogueState::new();

        // Eligibility check requires gold_weight
        assert_eq!(state.completion_for_intent("eligibility_check"), 0.0);

        state.set_slot_value("gold_weight", "50", 0.9);
        assert_eq!(state.completion_for_intent("eligibility_check"), 1.0);
    }

    #[test]
    fn test_clear_slot() {
        let mut state = GoldLoanDialogueState::new();

        state.set_slot_value("customer_name", "Rahul", 0.9);
        state.mark_confirmed("customer_name");

        assert!(state.customer_name().is_some());
        assert!(state.confirmed_slots().contains("customer_name"));

        state.clear_slot("customer_name");

        assert!(state.customer_name().is_none());
        assert!(!state.confirmed_slots().contains("customer_name"));
    }

    #[test]
    fn test_intent_tracking() {
        let mut state = GoldLoanDialogueState::new();

        state.update_intent("loan_inquiry", 0.9);
        assert_eq!(state.primary_intent(), Some("loan_inquiry"));

        state.update_intent("eligibility_check", 0.85);
        assert_eq!(state.primary_intent(), Some("eligibility_check"));
        assert!(state.secondary_intents().contains(&"loan_inquiry".to_string()));
    }

    // ====== Goal Tracking Tests ======

    #[test]
    fn test_default_goal() {
        let state = GoldLoanDialogueState::new();
        assert_eq!(state.goal_id(), DEFAULT_GOAL);
    }

    #[test]
    fn test_set_goal() {
        let mut state = GoldLoanDialogueState::new();
        assert_eq!(state.goal_id(), "exploration");

        state.set_goal("balance_transfer", 1);
        assert_eq!(state.goal_id(), "balance_transfer");
        assert_eq!(state.goal_set_turn(), 1);
    }

    #[test]
    fn test_confirm_goal() {
        let mut state = GoldLoanDialogueState::new();
        state.confirm_goal("new_loan", 2);

        assert_eq!(state.goal_id(), "new_loan");
        assert!(state.is_goal_confirmed());
    }

    #[test]
    fn test_goal_not_downgraded_to_exploration() {
        let mut state = GoldLoanDialogueState::new();
        state.set_goal("balance_transfer", 1);

        // Setting exploration should not overwrite when already have a goal
        state.set_goal(DEFAULT_GOAL, 2);
        assert_eq!(state.goal_id(), "balance_transfer");
    }

    #[test]
    fn test_next_action_instruction() {
        let action = NextBestAction::CallTool("calculate_savings".to_string());
        assert!(action.to_instruction().contains("CALL"));
        assert!(action.to_instruction().contains("calculate_savings"));

        let action = NextBestAction::AskFor("loan_amount".to_string());
        assert!(action.to_instruction().contains("ASK"));
        assert!(action.to_instruction().contains("loan amount"));

        let action = NextBestAction::ExplainProcess;
        assert!(action.to_instruction().contains("EXPLAIN"));
        assert!(action.to_instruction().contains("balance transfer"));
    }

    #[test]
    fn test_full_context_string() {
        let mut state = GoldLoanDialogueState::new();
        state.set_goal("balance_transfer", 0);
        state.set_slot_value("customer_name", "Rahul", 0.9);
        state.set_slot_value("current_lender", "Muthoot", 0.9);
        state.set_slot_value("loan_amount", "1000000", 0.9);

        let context = state.to_full_context_string();
        assert!(context.contains("Customer Information"));
        assert!(context.contains("Rahul"));
        assert!(context.contains("Muthoot"));
        assert!(context.contains("Current Goal: balance_transfer"));
    }

    // ====== Confirmation Prompt Tests ======

    #[test]
    fn test_pending_confirmation_prompt_empty() {
        let state = GoldLoanDialogueState::new();
        assert!(state.pending_confirmation_prompt().is_none());
    }

    #[test]
    fn test_pending_confirmation_prompt_with_values() {
        let mut state = GoldLoanDialogueState::new();
        state.set_slot_value("loan_amount", "500000", 0.7);
        state.mark_pending("loan_amount");
        state.set_slot_value("gold_weight", "50", 0.7);
        state.mark_pending("gold_weight");

        let prompt = state.pending_confirmation_prompt();
        assert!(prompt.is_some());
        let prompt = prompt.unwrap();
        assert!(prompt.contains("Please confirm"));
        assert!(prompt.contains("loan amount"));
        assert!(prompt.contains("gold weight"));
    }

    #[test]
    fn test_format_slot_value_for_display() {
        // Test amount formatting
        assert_eq!(
            GoldLoanDialogueState::format_slot_value_for_display("loan_amount", "500000"),
            "₹5.0 lakh"
        );
        assert_eq!(
            GoldLoanDialogueState::format_slot_value_for_display("loan_amount", "10000000"),
            "₹1.0 crore"
        );
        assert_eq!(
            GoldLoanDialogueState::format_slot_value_for_display("loan_amount", "50000"),
            "₹50000"
        );

        // Test weight formatting
        assert_eq!(
            GoldLoanDialogueState::format_slot_value_for_display("gold_weight", "50"),
            "50g"
        );

        // Test rate formatting
        assert_eq!(
            GoldLoanDialogueState::format_slot_value_for_display("current_interest_rate", "18.5"),
            "18.5%"
        );

        // Test default (no formatting)
        assert_eq!(
            GoldLoanDialogueState::format_slot_value_for_display("customer_name", "Rahul"),
            "Rahul"
        );
    }

    #[test]
    fn test_critical_slots_for_confirmation() {
        let mut state = GoldLoanDialogueState::new();

        // No critical slots set
        assert!(state.critical_slots_for_confirmation().is_empty());

        // Set critical slots but mark as confirmed
        state.set_slot_value("loan_amount", "500000", 0.9);
        state.mark_confirmed("loan_amount");
        assert!(state.critical_slots_for_confirmation().is_empty());

        // Set critical slot without confirmation
        state.set_slot_value("gold_weight", "50", 0.8);
        let critical = state.critical_slots_for_confirmation();
        assert_eq!(critical.len(), 1);
        assert!(critical.iter().any(|(slot, _)| *slot == "gold_weight"));
    }

    #[test]
    fn test_critical_confirmation_prompt() {
        let mut state = GoldLoanDialogueState::new();

        // No critical values
        assert!(state.critical_confirmation_prompt().is_none());

        // Add unconfirmed critical values
        state.set_slot_value("loan_amount", "500000", 0.8);
        state.set_slot_value("gold_weight", "50", 0.8);

        let prompt = state.critical_confirmation_prompt();
        assert!(prompt.is_some());
        let prompt = prompt.unwrap();
        assert!(prompt.contains("Before proceeding"));
        assert!(prompt.contains("loan amount"));
        assert!(prompt.contains("₹5.0 lakh"));
        assert!(prompt.contains("gold weight"));
        assert!(prompt.contains("50g"));
    }

    #[test]
    fn test_slots_needing_confirmation() {
        let mut state = GoldLoanDialogueState::new();

        // No pending slots
        assert!(state.slots_needing_confirmation().is_empty());

        // Add pending slots
        state.set_slot_value("loan_amount", "500000", 0.7);
        state.mark_pending("loan_amount");

        let needing = state.slots_needing_confirmation();
        assert_eq!(needing.len(), 1);
        assert_eq!(needing[0].0, "loan_amount");
        assert_eq!(needing[0].1, "500000");
    }
}
