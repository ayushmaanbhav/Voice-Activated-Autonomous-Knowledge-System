//! Gold Loan Dialogue State Slot Definitions
//!
//! Domain-specific slot schema based on LDST and ACL 2024 research.
//! Implements structured dialogue state for gold loan conversations.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Gold purity levels (in karats)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GoldPurity {
    /// 24 karat (99.9% pure)
    K24,
    /// 22 karat (91.6% pure)
    K22,
    /// 18 karat (75% pure)
    K18,
    /// 14 karat (58.3% pure)
    K14,
    /// Unknown purity
    Unknown,
}

impl GoldPurity {
    /// Get purity percentage
    pub fn percentage(&self) -> f32 {
        match self {
            GoldPurity::K24 => 99.9,
            GoldPurity::K22 => 91.6,
            GoldPurity::K18 => 75.0,
            GoldPurity::K14 => 58.3,
            GoldPurity::Unknown => 0.0,
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        if lower.contains("24") {
            GoldPurity::K24
        } else if lower.contains("22") {
            GoldPurity::K22
        } else if lower.contains("18") {
            GoldPurity::K18
        } else if lower.contains("14") {
            GoldPurity::K14
        } else {
            GoldPurity::Unknown
        }
    }
}

impl std::fmt::Display for GoldPurity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GoldPurity::K24 => write!(f, "24 karat"),
            GoldPurity::K22 => write!(f, "22 karat"),
            GoldPurity::K18 => write!(f, "18 karat"),
            GoldPurity::K14 => write!(f, "14 karat"),
            GoldPurity::Unknown => write!(f, "unknown purity"),
        }
    }
}

/// Conversation Goal - tracks the primary journey the customer is on
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ConversationGoal {
    /// Just exploring/gathering information
    #[default]
    Exploration,
    /// Balance transfer from another lender (Muthoot, Manappuram, etc.)
    BalanceTransfer,
    /// New gold loan application
    NewLoan,
    /// Checking eligibility
    EligibilityCheck,
    /// Looking for branch/appointment
    BranchVisit,
    /// Wants callback/lead capture
    LeadCapture,
}

impl ConversationGoal {
    /// Get the required slots for this goal
    pub fn required_slots(&self) -> &'static [&'static str] {
        match self {
            ConversationGoal::Exploration => &[],
            ConversationGoal::BalanceTransfer => &["current_lender", "loan_amount"],
            ConversationGoal::NewLoan => &["gold_weight", "loan_amount"],
            ConversationGoal::EligibilityCheck => &["gold_weight"],
            ConversationGoal::BranchVisit => &["location"],
            ConversationGoal::LeadCapture => &["customer_name", "phone_number"],
        }
    }

    /// Get optional slots that enhance the goal
    pub fn optional_slots(&self) -> &'static [&'static str] {
        match self {
            ConversationGoal::Exploration => &[],
            ConversationGoal::BalanceTransfer => &["current_interest_rate", "gold_weight", "gold_purity"],
            ConversationGoal::NewLoan => &["gold_purity", "loan_purpose", "loan_tenure"],
            ConversationGoal::EligibilityCheck => &["gold_purity", "loan_amount"],
            ConversationGoal::BranchVisit => &["preferred_date", "preferred_time", "customer_name", "phone_number"],
            ConversationGoal::LeadCapture => &["location", "loan_amount"],
        }
    }

    /// Get the next best action based on filled slots
    pub fn next_action(&self, filled_slots: &[&str]) -> NextBestAction {
        match self {
            ConversationGoal::BalanceTransfer => {
                let has_lender = filled_slots.contains(&"current_lender");
                let has_amount = filled_slots.contains(&"loan_amount");
                let has_rate = filled_slots.contains(&"current_interest_rate");
                let has_location = filled_slots.contains(&"location");
                let has_contact = filled_slots.contains(&"phone_number") || filled_slots.contains(&"customer_name");

                if has_lender && has_amount && has_rate {
                    // Ready to calculate savings
                    NextBestAction::CallTool("calculate_savings".to_string())
                } else if has_lender && has_amount && !has_rate {
                    NextBestAction::AskFor("current_interest_rate".to_string())
                } else if has_lender && !has_amount {
                    NextBestAction::AskFor("loan_amount".to_string())
                } else if has_lender && has_amount && has_location && !has_contact {
                    // Have location, offer appointment
                    NextBestAction::OfferAppointment
                } else if has_lender && has_amount && !has_location {
                    NextBestAction::AskFor("location".to_string())
                } else {
                    NextBestAction::ExplainProcess
                }
            }
            ConversationGoal::NewLoan => {
                let has_weight = filled_slots.contains(&"gold_weight");
                let has_amount = filled_slots.contains(&"loan_amount");

                if has_weight {
                    NextBestAction::CallTool("check_eligibility".to_string())
                } else if has_amount && !has_weight {
                    NextBestAction::AskFor("gold_weight".to_string())
                } else {
                    NextBestAction::AskFor("loan_amount".to_string())
                }
            }
            ConversationGoal::EligibilityCheck => {
                if filled_slots.contains(&"gold_weight") {
                    NextBestAction::CallTool("check_eligibility".to_string())
                } else {
                    NextBestAction::AskFor("gold_weight".to_string())
                }
            }
            ConversationGoal::BranchVisit => {
                if filled_slots.contains(&"location") {
                    NextBestAction::CallTool("find_branches".to_string())
                } else {
                    NextBestAction::AskFor("location".to_string())
                }
            }
            ConversationGoal::LeadCapture => {
                let has_name = filled_slots.contains(&"customer_name");
                let has_phone = filled_slots.contains(&"phone_number");

                if has_name && has_phone {
                    NextBestAction::CallTool("capture_lead".to_string())
                } else if has_name && !has_phone {
                    NextBestAction::AskFor("phone_number".to_string())
                } else {
                    NextBestAction::AskFor("customer_name".to_string())
                }
            }
            ConversationGoal::Exploration => NextBestAction::DiscoverIntent,
        }
    }

    /// Detect goal from intent string
    pub fn from_intent(intent: &str) -> Self {
        match intent {
            "balance_transfer" | "switch_lender" | "loan_transfer" => ConversationGoal::BalanceTransfer,
            "eligibility_inquiry" | "eligibility_check" => ConversationGoal::EligibilityCheck,
            "new_loan" | "loan_inquiry" | "gold_loan" => ConversationGoal::NewLoan,
            "branch_inquiry" | "find_branch" | "schedule_visit" | "appointment_request" => ConversationGoal::BranchVisit,
            "callback_request" | "capture_lead" | "interested" | "sms_request" => ConversationGoal::LeadCapture,
            _ => ConversationGoal::Exploration,
        }
    }
}

impl std::fmt::Display for ConversationGoal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversationGoal::Exploration => write!(f, "exploration"),
            ConversationGoal::BalanceTransfer => write!(f, "balance_transfer"),
            ConversationGoal::NewLoan => write!(f, "new_loan"),
            ConversationGoal::EligibilityCheck => write!(f, "eligibility_check"),
            ConversationGoal::BranchVisit => write!(f, "branch_visit"),
            ConversationGoal::LeadCapture => write!(f, "lead_capture"),
        }
    }
}

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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
    /// Current conversation goal
    conversation_goal: ConversationGoal,
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

    /// Get gold purity
    pub fn gold_purity(&self) -> Option<GoldPurity> {
        self.gold_purity
            .as_ref()
            .map(|v| GoldPurity::from_str(&v.value))
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

    /// Get the current conversation goal
    pub fn conversation_goal(&self) -> ConversationGoal {
        self.conversation_goal
    }

    /// Check if goal is confirmed (explicit) vs inferred
    pub fn is_goal_confirmed(&self) -> bool {
        self.goal_confirmed
    }

    /// Update the conversation goal based on detected intent
    pub fn update_goal_from_intent(&mut self, intent: &str, turn: usize) {
        let new_goal = ConversationGoal::from_intent(intent);

        // Only upgrade goal, don't downgrade (e.g., don't go from BalanceTransfer to Exploration)
        let should_update = match (&self.conversation_goal, &new_goal) {
            (ConversationGoal::Exploration, _) => true, // Always upgrade from exploration
            (_, ConversationGoal::Exploration) => false, // Never downgrade to exploration
            (ConversationGoal::BalanceTransfer, ConversationGoal::LeadCapture) => true, // BT can lead to lead capture
            (ConversationGoal::BalanceTransfer, ConversationGoal::BranchVisit) => true, // BT can lead to branch visit
            (ConversationGoal::NewLoan, ConversationGoal::LeadCapture) => true,
            (ConversationGoal::NewLoan, ConversationGoal::BranchVisit) => true,
            (ConversationGoal::EligibilityCheck, ConversationGoal::NewLoan) => true, // Eligibility often leads to new loan
            (ConversationGoal::EligibilityCheck, ConversationGoal::LeadCapture) => true,
            _ => false, // Don't change for other transitions
        };

        if should_update && new_goal != ConversationGoal::Exploration {
            self.conversation_goal = new_goal;
            self.goal_set_turn = turn;
        }
    }

    /// Set the conversation goal explicitly (e.g., user confirmed it)
    pub fn set_goal(&mut self, goal: ConversationGoal, turn: usize) {
        self.conversation_goal = goal;
        self.goal_confirmed = true;
        self.goal_set_turn = turn;
    }

    /// Get the next best action based on current goal and filled slots
    pub fn get_next_action(&self) -> NextBestAction {
        let filled = self.filled_slots();
        self.conversation_goal.next_action(&filled)
    }

    /// Get missing required slots for current goal
    pub fn missing_required_slots(&self) -> Vec<&'static str> {
        let filled = self.filled_slots();
        self.conversation_goal
            .required_slots()
            .iter()
            .filter(|s| !filled.contains(*s))
            .copied()
            .collect()
    }

    /// Get missing optional slots for current goal
    pub fn missing_optional_slots(&self) -> Vec<&'static str> {
        let filled = self.filled_slots();
        self.conversation_goal
            .optional_slots()
            .iter()
            .filter(|s| !filled.contains(*s))
            .copied()
            .collect()
    }

    /// Calculate goal completion percentage
    pub fn goal_completion(&self) -> f32 {
        let required = self.conversation_goal.required_slots();
        if required.is_empty() {
            return 1.0;
        }

        let filled = self.filled_slots();
        let filled_required = required.iter().filter(|s| filled.contains(*s)).count();
        filled_required as f32 / required.len() as f32
    }

    /// Check if we should proactively trigger a tool
    pub fn should_trigger_tool(&self) -> Option<String> {
        let completion = self.goal_completion();

        match self.conversation_goal {
            ConversationGoal::BalanceTransfer => {
                // If we have lender + amount + rate, trigger calculate_savings
                if self.current_lender.is_some()
                    && self.loan_amount.is_some()
                    && self.current_interest_rate.is_some()
                {
                    return Some("calculate_savings".to_string());
                }
            }
            ConversationGoal::EligibilityCheck => {
                if self.gold_weight_grams.is_some() {
                    return Some("check_eligibility".to_string());
                }
            }
            ConversationGoal::BranchVisit => {
                if self.location.is_some() {
                    return Some("find_branches".to_string());
                }
            }
            ConversationGoal::LeadCapture => {
                if self.customer_name.is_some() && self.phone_number.is_some() {
                    return Some("capture_lead".to_string());
                }
            }
            ConversationGoal::NewLoan => {
                if self.gold_weight_grams.is_some() && completion >= 0.5 {
                    return Some("check_eligibility".to_string());
                }
            }
            ConversationGoal::Exploration => {}
        }

        None
    }

    /// Check if we should auto-capture lead (when we have contact info during any goal)
    /// Returns true if lead capture should be triggered as a secondary action
    pub fn should_auto_capture_lead(&self) -> bool {
        // Don't duplicate if already in lead capture mode
        if self.conversation_goal == ConversationGoal::LeadCapture {
            return false;
        }

        // Capture lead if we have both name and phone collected
        // and we're progressing well on another goal (showing engagement)
        let has_contact = self.customer_name.is_some() && self.phone_number.is_some();
        let has_partial_contact = self.customer_name.is_some() || self.phone_number.is_some();

        // Full contact info = definitely capture
        if has_contact {
            return true;
        }

        // Partial contact + high goal completion = capture
        let completion = self.goal_completion();
        if has_partial_contact && completion >= 0.75 {
            return true;
        }

        false
    }

    /// Generate goal-aware context for LLM prompt
    pub fn goal_context(&self) -> String {
        let mut parts = Vec::new();

        // Current goal
        parts.push(format!("## Current Goal: {}", self.conversation_goal));

        // Goal completion
        let completion = self.goal_completion();
        if completion < 1.0 {
            let missing = self.missing_required_slots();
            if !missing.is_empty() {
                parts.push(format!("Missing required info: {}", missing.join(", ")));
            }
        } else {
            parts.push("All required information collected.".to_string());
        }

        // Next action
        let next_action = self.get_next_action();
        parts.push(format!("## Next Action: {}", next_action.to_instruction()));

        // For balance transfer, add key process info
        if self.conversation_goal == ConversationGoal::BalanceTransfer {
            parts.push("\n## Balance Transfer Key Points:".to_string());
            parts.push("- Kotak pays off existing lender directly (no cash needed from customer)".to_string());
            parts.push("- Customer keeps same gold, just transfers to Kotak vault".to_string());
            parts.push("- Lower interest rate: 10.5% vs competitor's higher rates".to_string());
            parts.push("- Process: Verify gold → Pay off old loan → New Kotak loan same day".to_string());
        }

        parts.join("\n")
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

        // Goal context
        output.push_str(&self.goal_context());

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gold_purity_parsing() {
        assert_eq!(GoldPurity::from_str("24k gold"), GoldPurity::K24);
        assert_eq!(GoldPurity::from_str("22 karat"), GoldPurity::K22);
        assert_eq!(GoldPurity::from_str("18kt"), GoldPurity::K18);
        assert_eq!(GoldPurity::from_str("pure gold"), GoldPurity::Unknown);
    }

    #[test]
    fn test_gold_purity_percentage() {
        assert!((GoldPurity::K24.percentage() - 99.9).abs() < 0.1);
        assert!((GoldPurity::K22.percentage() - 91.6).abs() < 0.1);
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
    fn test_conversation_goal_from_intent() {
        assert_eq!(ConversationGoal::from_intent("balance_transfer"), ConversationGoal::BalanceTransfer);
        assert_eq!(ConversationGoal::from_intent("switch_lender"), ConversationGoal::BalanceTransfer);
        assert_eq!(ConversationGoal::from_intent("loan_transfer"), ConversationGoal::BalanceTransfer);
        assert_eq!(ConversationGoal::from_intent("eligibility_inquiry"), ConversationGoal::EligibilityCheck);
        assert_eq!(ConversationGoal::from_intent("new_loan"), ConversationGoal::NewLoan);
        assert_eq!(ConversationGoal::from_intent("branch_inquiry"), ConversationGoal::BranchVisit);
        assert_eq!(ConversationGoal::from_intent("callback_request"), ConversationGoal::LeadCapture);
        assert_eq!(ConversationGoal::from_intent("unknown"), ConversationGoal::Exploration);
    }

    #[test]
    fn test_goal_required_slots() {
        assert_eq!(ConversationGoal::BalanceTransfer.required_slots(), &["current_lender", "loan_amount"]);
        assert_eq!(ConversationGoal::EligibilityCheck.required_slots(), &["gold_weight"]);
        assert_eq!(ConversationGoal::BranchVisit.required_slots(), &["location"]);
        assert_eq!(ConversationGoal::LeadCapture.required_slots(), &["customer_name", "phone_number"]);
    }

    #[test]
    fn test_goal_next_action_balance_transfer() {
        // No slots -> explain process
        let action = ConversationGoal::BalanceTransfer.next_action(&[]);
        assert!(matches!(action, NextBestAction::ExplainProcess));

        // Has lender only -> ask for amount
        let action = ConversationGoal::BalanceTransfer.next_action(&["current_lender"]);
        assert!(matches!(action, NextBestAction::AskFor(ref s) if s == "loan_amount"));

        // Has lender + amount -> ask for rate
        let action = ConversationGoal::BalanceTransfer.next_action(&["current_lender", "loan_amount"]);
        assert!(matches!(action, NextBestAction::AskFor(ref s) if s == "current_interest_rate"));

        // Has all -> call tool
        let action = ConversationGoal::BalanceTransfer.next_action(&["current_lender", "loan_amount", "current_interest_rate"]);
        assert!(matches!(action, NextBestAction::CallTool(ref s) if s == "calculate_savings"));
    }

    #[test]
    fn test_goal_next_action_branch_visit() {
        // No location -> ask for it
        let action = ConversationGoal::BranchVisit.next_action(&[]);
        assert!(matches!(action, NextBestAction::AskFor(ref s) if s == "location"));

        // Has location -> call find_branches
        let action = ConversationGoal::BranchVisit.next_action(&["location"]);
        assert!(matches!(action, NextBestAction::CallTool(ref s) if s == "find_branches"));
    }

    #[test]
    fn test_goal_update_from_intent() {
        let mut state = GoldLoanDialogueState::new();
        assert_eq!(state.conversation_goal(), ConversationGoal::Exploration);

        // Upgrade from exploration to balance transfer
        state.update_goal_from_intent("balance_transfer", 1);
        assert_eq!(state.conversation_goal(), ConversationGoal::BalanceTransfer);

        // Should not downgrade back to exploration
        state.update_goal_from_intent("unknown", 2);
        assert_eq!(state.conversation_goal(), ConversationGoal::BalanceTransfer);

        // Can upgrade to lead capture
        state.update_goal_from_intent("callback_request", 3);
        assert_eq!(state.conversation_goal(), ConversationGoal::LeadCapture);
    }

    #[test]
    fn test_goal_completion() {
        let mut state = GoldLoanDialogueState::new();
        state.set_goal(ConversationGoal::BalanceTransfer, 0);

        // 0% complete
        assert_eq!(state.goal_completion(), 0.0);

        // 50% complete (1 of 2 required slots)
        state.set_slot_value("current_lender", "Muthoot", 0.9);
        assert_eq!(state.goal_completion(), 0.5);

        // 100% complete
        state.set_slot_value("loan_amount", "1000000", 0.9);
        assert_eq!(state.goal_completion(), 1.0);
    }

    #[test]
    fn test_missing_required_slots() {
        let mut state = GoldLoanDialogueState::new();
        state.set_goal(ConversationGoal::BalanceTransfer, 0);

        let missing = state.missing_required_slots();
        assert!(missing.contains(&"current_lender"));
        assert!(missing.contains(&"loan_amount"));

        state.set_slot_value("current_lender", "Muthoot", 0.9);
        let missing = state.missing_required_slots();
        assert!(!missing.contains(&"current_lender"));
        assert!(missing.contains(&"loan_amount"));
    }

    #[test]
    fn test_should_trigger_tool() {
        let mut state = GoldLoanDialogueState::new();
        state.set_goal(ConversationGoal::BalanceTransfer, 0);

        // Not enough info -> no tool
        assert!(state.should_trigger_tool().is_none());

        state.set_slot_value("current_lender", "Muthoot", 0.9);
        state.set_slot_value("loan_amount", "1000000", 0.9);
        assert!(state.should_trigger_tool().is_none()); // Still missing rate

        state.set_slot_value("current_interest_rate", "18", 0.9);
        assert_eq!(state.should_trigger_tool(), Some("calculate_savings".to_string()));
    }

    #[test]
    fn test_goal_context_generation() {
        let mut state = GoldLoanDialogueState::new();
        state.set_goal(ConversationGoal::BalanceTransfer, 0);
        state.set_slot_value("current_lender", "Muthoot", 0.9);

        let context = state.goal_context();
        assert!(context.contains("balance_transfer"));
        assert!(context.contains("Missing required info"));
        assert!(context.contains("loan_amount"));
        assert!(context.contains("Balance Transfer Key Points"));
        assert!(context.contains("Kotak pays off existing lender directly"));
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
        state.set_goal(ConversationGoal::BalanceTransfer, 0);
        state.set_slot_value("customer_name", "Rahul", 0.9);
        state.set_slot_value("current_lender", "Muthoot", 0.9);
        state.set_slot_value("loan_amount", "1000000", 0.9);

        let context = state.to_full_context_string();
        assert!(context.contains("Customer Information"));
        assert!(context.contains("Rahul"));
        assert!(context.contains("Muthoot"));
        assert!(context.contains("Current Goal: balance_transfer"));
        assert!(context.contains("Next Action"));
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
