//! Conversation Management
//!
//! Manages the overall conversation flow and state.

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;

use crate::intent::{DetectedIntent, IntentDetector};
use crate::memory::MemoryConfig;
use crate::memory_legacy::{ConversationMemory, MemoryEntry};
use crate::stage::{ConversationStage, StageManager, TransitionReason};
use crate::AgentError;
use voice_agent_core::{Turn, TurnRole};

/// Conversation configuration
#[derive(Debug, Clone)]
pub struct ConversationConfig {
    /// Maximum duration in seconds
    pub max_duration_seconds: u32,
    /// Session timeout in seconds
    pub session_timeout_seconds: u32,
    /// Memory config
    pub memory: MemoryConfig,
    /// Enable intent detection
    pub intent_detection: bool,
    /// Default language
    pub language: String,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_duration_seconds: 600, // 10 minutes
            session_timeout_seconds: 60,
            memory: MemoryConfig::default(),
            intent_detection: true,
            language: "en".to_string(),
        }
    }
}

/// Conversation event
#[derive(Debug, Clone)]
pub enum ConversationEvent {
    /// Conversation started
    Started { session_id: String },
    /// Turn added
    TurnAdded { role: TurnRole, content: String },
    /// Intent detected
    IntentDetected(DetectedIntent),
    /// Stage changed
    StageChanged {
        from: ConversationStage,
        to: ConversationStage,
    },
    /// Fact learned
    FactLearned { key: String, value: String },
    /// Tool called
    ToolCalled { name: String, success: bool },
    /// Conversation ended
    Ended { reason: EndReason },
    /// Error occurred
    Error(String),
}

/// Reason for conversation end
#[derive(Debug, Clone)]
pub enum EndReason {
    UserEnded,
    AgentEnded,
    Timeout,
    MaxDuration,
    Error(String),
}

/// Conversation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    Active,
    Paused,
    Ended,
}

// =============================================================================
// P0 FIX: AI Disclosure and Consent Tracking (RBI Compliance)
// =============================================================================

/// P0 FIX: AI Disclosure tracking for RBI compliance
///
/// RBI requires that customers are informed they are speaking with an AI,
/// and must be given the option to speak with a human agent.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AiDisclosure {
    /// Whether AI disclosure has been given to the customer
    pub given: bool,
    /// When the disclosure was given
    pub timestamp: Option<DateTime<Utc>>,
    /// Language in which disclosure was given
    pub language: Option<String>,
    /// The actual disclosure text that was used
    pub disclosure_text: Option<String>,
    /// Whether human escalation option was offered
    pub human_option_offered: bool,
}

impl AiDisclosure {
    /// Mark AI disclosure as given
    pub fn mark_disclosed(&mut self, language: &str, text: &str, human_option: bool) {
        self.given = true;
        self.timestamp = Some(Utc::now());
        self.language = Some(language.to_string());
        self.disclosure_text = Some(text.to_string());
        self.human_option_offered = human_option;
    }

    /// Get localized AI disclosure message
    pub fn get_disclosure_message(language: &str) -> &'static str {
        match language {
            "hi" | "hindi" => "यह एक AI सहायक है। आप किसी भी समय 'एजेंट से बात करें' कहकर मानव एजेंट से बात कर सकते हैं।",
            "mr" | "marathi" => "हे एक AI सहाय्यक आहे. तुम्ही कधीही 'एजंटशी बोला' असे सांगून मानवी एजंटशी बोलू शकता.",
            "ta" | "tamil" => "இது ஒரு AI உதவியாளர். நீங்கள் எந்த நேரத்திலும் 'ஏஜெண்டுடன் பேசுங்கள்' என்று சொல்லி மனித ஏஜெண்டுடன் பேசலாம்.",
            "te" | "telugu" => "ఇది AI సహాయకుడు. మీరు ఎప్పుడైనా 'ఏజెంట్‌తో మాట్లాడండి' అని చెప్పి మానవ ఏజెంట్‌తో మాట్లాడవచ్చు.",
            "bn" | "bengali" => "এটি একটি AI সহায়ক। আপনি যেকোনো সময় 'এজেন্টের সাথে কথা বলুন' বলে একজন মানব এজেন্টের সাথে কথা বলতে পারেন।",
            "gu" | "gujarati" => "આ એક AI સહાયક છે. તમે ગમે ત્યારે 'એજન્ટ સાથે વાત કરો' કહીને માનવ એજન્ટ સાથે વાત કરી શકો છો.",
            "kn" | "kannada" => "ಇದು AI ಸಹಾಯಕ. ನೀವು ಯಾವುದೇ ಸಮಯದಲ್ಲಿ 'ಏಜೆಂಟ್‌ನೊಂದಿಗೆ ಮಾತನಾಡಿ' ಎಂದು ಹೇಳುವ ಮೂಲಕ ಮಾನವ ಏಜೆಂಟ್‌ನೊಂದಿಗೆ ಮಾತನಾಡಬಹುದು.",
            "ml" | "malayalam" => "ഇത് ഒരു AI അസിസ്റ്റന്റ് ആണ്. 'ഏജന്റിനോട് സംസാരിക്കുക' എന്ന് പറഞ്ഞ് നിങ്ങൾക്ക് എപ്പോൾ വേണമെങ്കിലും ഒരു മനുഷ്യ ഏജന്റുമായി സംസാരിക്കാം.",
            "pa" | "punjabi" => "ਇਹ ਇੱਕ AI ਸਹਾਇਕ ਹੈ। ਤੁਸੀਂ ਕਿਸੇ ਵੀ ਸਮੇਂ 'ਏਜੰਟ ਨਾਲ ਗੱਲ ਕਰੋ' ਕਹਿ ਕੇ ਮਨੁੱਖੀ ਏਜੰਟ ਨਾਲ ਗੱਲ ਕਰ ਸਕਦੇ ਹੋ।",
            "or" | "odia" => "ଏହା ଏକ AI ସହାୟକ। ଆପଣ ଯେକୌଣସି ସମୟରେ 'ଏଜେଣ୍ଟଙ୍କ ସହ କଥା ହୁଅନ୍ତୁ' କହି ମାନବ ଏଜେଣ୍ଟଙ୍କ ସହ କଥା ହୋଇପାରିବେ।",
            _ => "This is an AI assistant. You can speak with a human agent at any time by saying 'speak to agent'.",
        }
    }
}

/// P0 FIX: Consent tracking for RBI compliance
///
/// Tracks various consents required for voice banking:
/// - Recording consent: Customer agrees to call recording
/// - PII processing consent: Customer agrees to personal data processing
/// - Marketing consent: Optional consent for promotional communications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Consent for call recording
    pub recording_consent: bool,
    /// When recording consent was given
    pub recording_consent_timestamp: Option<DateTime<Utc>>,
    /// Consent for PII processing
    pub pii_processing_consent: bool,
    /// When PII consent was given
    pub pii_consent_timestamp: Option<DateTime<Utc>>,
    /// Optional marketing consent
    pub marketing_consent: Option<bool>,
    /// When marketing consent was given/denied
    pub marketing_consent_timestamp: Option<DateTime<Utc>>,
    /// Method by which consent was obtained
    pub consent_method: ConsentMethod,
    /// Language of consent
    pub consent_language: String,
}

/// Method by which consent was obtained
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentMethod {
    /// Customer explicitly said "yes" or equivalent
    Voice,
    /// Customer typed confirmation
    Text,
    /// Implied by continuing after disclosure
    Implied,
    /// Customer explicitly clicked/tapped consent
    Explicit,
}

impl Default for ConsentMethod {
    fn default() -> Self {
        Self::Implied
    }
}

impl Default for ConsentRecord {
    fn default() -> Self {
        Self {
            recording_consent: false,
            recording_consent_timestamp: None,
            pii_processing_consent: false,
            pii_consent_timestamp: None,
            marketing_consent: None,
            marketing_consent_timestamp: None,
            consent_method: ConsentMethod::Implied,
            consent_language: "en".to_string(),
        }
    }
}

impl ConsentRecord {
    /// Record recording consent
    pub fn record_recording_consent(&mut self, given: bool, method: ConsentMethod) {
        self.recording_consent = given;
        self.recording_consent_timestamp = Some(Utc::now());
        self.consent_method = method;
    }

    /// Record PII processing consent
    pub fn record_pii_consent(&mut self, given: bool, method: ConsentMethod) {
        self.pii_processing_consent = given;
        self.pii_consent_timestamp = Some(Utc::now());
        self.consent_method = method;
    }

    /// Record marketing consent
    pub fn record_marketing_consent(&mut self, given: Option<bool>) {
        self.marketing_consent = given;
        self.marketing_consent_timestamp = Some(Utc::now());
    }

    /// Check if minimum required consents are given
    pub fn has_minimum_consent(&self) -> bool {
        // For voice banking, recording consent is essential
        // PII consent is required before processing personal data
        self.recording_consent
    }

    /// Check if all consents are given
    pub fn has_full_consent(&self) -> bool {
        self.recording_consent && self.pii_processing_consent
    }
}

/// P0 FIX: Compliance status for the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    /// AI disclosure tracking
    pub ai_disclosure: AiDisclosure,
    /// Consent tracking
    pub consent: ConsentRecord,
    /// Whether compliance requirements are met
    pub compliant: bool,
    /// List of pending compliance requirements
    pub pending_requirements: Vec<String>,
}

impl Default for ComplianceStatus {
    fn default() -> Self {
        Self {
            ai_disclosure: AiDisclosure::default(),
            consent: ConsentRecord::default(),
            compliant: false,
            pending_requirements: vec![
                "ai_disclosure".to_string(),
                "recording_consent".to_string(),
            ],
        }
    }
}

impl ComplianceStatus {
    /// Update compliance status based on current state
    pub fn update(&mut self) {
        self.pending_requirements.clear();

        if !self.ai_disclosure.given {
            self.pending_requirements.push("ai_disclosure".to_string());
        }

        if !self.consent.recording_consent {
            self.pending_requirements
                .push("recording_consent".to_string());
        }

        self.compliant = self.pending_requirements.is_empty();
    }

    /// Check if ready for PII processing
    pub fn can_process_pii(&self) -> bool {
        self.ai_disclosure.given && self.consent.pii_processing_consent
    }
}

/// Conversation manager
pub struct Conversation {
    /// Session ID
    session_id: String,
    /// Configuration
    config: ConversationConfig,
    /// Start time
    start_time: Instant,
    /// Last activity time
    last_activity: Mutex<Instant>,
    /// Current state
    state: Mutex<ConversationState>,
    /// Stage manager
    stage_manager: Arc<StageManager>,
    /// Memory
    memory: Arc<ConversationMemory>,
    /// Intent detector
    intent_detector: Arc<IntentDetector>,
    /// Event sender
    event_tx: broadcast::Sender<ConversationEvent>,
    /// Turn counter
    turn_count: Mutex<usize>,
    /// P0 FIX: Compliance status (AI disclosure, consent)
    compliance: Mutex<ComplianceStatus>,
}

impl Conversation {
    /// Create a new conversation
    pub fn new(session_id: impl Into<String>, config: ConversationConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        Self {
            session_id: session_id.into(),
            config: config.clone(),
            start_time: Instant::now(),
            last_activity: Mutex::new(Instant::now()),
            state: Mutex::new(ConversationState::Active),
            stage_manager: Arc::new(StageManager::new()),
            memory: Arc::new(ConversationMemory::new(config.memory)),
            intent_detector: Arc::new(IntentDetector::new()),
            event_tx,
            turn_count: Mutex::new(0),
            compliance: Mutex::new(ComplianceStatus::default()),
        }
    }

    /// Subscribe to conversation events
    pub fn subscribe(&self) -> broadcast::Receiver<ConversationEvent> {
        self.event_tx.subscribe()
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get current state
    pub fn state(&self) -> ConversationState {
        *self.state.lock()
    }

    /// Get current stage
    pub fn stage(&self) -> ConversationStage {
        self.stage_manager.current()
    }

    /// Get duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get turn count
    pub fn turn_count(&self) -> usize {
        *self.turn_count.lock()
    }

    /// Check if conversation is active
    pub fn is_active(&self) -> bool {
        *self.state.lock() == ConversationState::Active
    }

    /// Add user turn
    pub fn add_user_turn(&self, content: &str) -> Result<DetectedIntent, AgentError> {
        self.check_active()?;
        self.update_activity();

        // Create turn
        let turn = Turn {
            role: TurnRole::User,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        // Add to memory
        let mut entry = MemoryEntry::from(&turn);
        entry.stage = Some(self.stage().display_name().to_string());

        // Detect intent
        let detected = if self.config.intent_detection {
            self.intent_detector.detect(content)
        } else {
            DetectedIntent {
                intent: "unknown".to_string(),
                confidence: 0.0,
                slots: std::collections::HashMap::new(),
                alternatives: vec![],
            }
        };

        entry.intents = vec![detected.intent.clone()];

        // Extract and store entities
        for (key, slot) in &detected.slots {
            if let Some(ref value) = slot.value {
                entry.entities.insert(key.clone(), value.clone());
                self.memory.add_fact(key, value, slot.confidence);

                let _ = self.event_tx.send(ConversationEvent::FactLearned {
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }

        self.memory.add(entry);
        self.stage_manager.record_turn();
        *self.turn_count.lock() += 1;

        // Emit events
        let _ = self.event_tx.send(ConversationEvent::TurnAdded {
            role: TurnRole::User,
            content: content.to_string(),
        });

        let _ = self
            .event_tx
            .send(ConversationEvent::IntentDetected(detected.clone()));

        // Check for stage transition triggers
        self.check_stage_transitions(&detected);

        Ok(detected)
    }

    /// Add assistant turn
    pub fn add_assistant_turn(&self, content: &str) -> Result<(), AgentError> {
        self.check_active()?;
        self.update_activity();

        let turn = Turn {
            role: TurnRole::Assistant,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        let mut entry = MemoryEntry::from(&turn);
        entry.stage = Some(self.stage().display_name().to_string());

        self.memory.add(entry);
        *self.turn_count.lock() += 1;

        let _ = self.event_tx.send(ConversationEvent::TurnAdded {
            role: TurnRole::Assistant,
            content: content.to_string(),
        });

        Ok(())
    }

    /// Transition to a new stage
    pub fn transition_stage(&self, to: ConversationStage) -> Result<(), AgentError> {
        let from = self.stage();

        match self
            .stage_manager
            .transition(to, TransitionReason::NaturalFlow)
        {
            Ok(_) => {
                let _ = self
                    .event_tx
                    .send(ConversationEvent::StageChanged { from, to });
                Ok(())
            },
            Err(e) => Err(AgentError::Stage(e)),
        }
    }

    /// Check and perform stage transitions based on intent
    ///
    /// P0 FIX: Added comprehensive intent→stage mappings for gold loan sales flow.
    /// Covers all major intents: greeting, loan_inquiry, eligibility, interest_rate,
    /// branch_locator, schedule_visit, objection, affirmative, negative, farewell, thank_you.
    fn check_stage_transitions(&self, intent: &DetectedIntent) {
        let current = self.stage();

        // Record intent for stage requirement tracking
        self.stage_manager.record_intent(&intent.intent);

        // P0 FIX: Comprehensive intent-based transitions for gold loan sales flow
        let new_stage = match intent.intent.as_str() {
            // Greeting -> Discovery: After initial greeting, move to discovery
            "greeting" if current == ConversationStage::Greeting => {
                // Only transition if we've had at least 1 turn (rapport built)
                if self.stage_manager.current_stage_turns() >= 1 {
                    Some(ConversationStage::Discovery)
                } else {
                    None
                }
            },

            // Loan inquiry / eligibility query: Move to relevant stage
            "loan_inquiry" | "eligibility_query" => match current {
                ConversationStage::Greeting => Some(ConversationStage::Discovery),
                ConversationStage::Discovery => Some(ConversationStage::Qualification),
                _ => None,
            },

            // Interest rate query: Customer interested in rates -> Presentation
            "interest_rate_query" => match current {
                ConversationStage::Greeting | ConversationStage::Discovery => {
                    Some(ConversationStage::Presentation)
                },
                ConversationStage::Qualification => Some(ConversationStage::Presentation),
                _ => None,
            },

            // Competitor reference: Need to understand their situation
            "competitor_reference" => {
                match current {
                    ConversationStage::Greeting => Some(ConversationStage::Discovery),
                    _ => None, // Stay in current stage but note the competitor info
                }
            },

            // Branch locator: Ready to visit -> Closing
            "branch_locator" => match current {
                ConversationStage::Presentation => Some(ConversationStage::Closing),
                ConversationStage::ObjectionHandling => Some(ConversationStage::Closing),
                _ => None,
            },

            // Schedule visit: Ready to book appointment -> Closing
            "schedule_visit" | "schedule_appointment" | "book_appointment" => {
                match current {
                    ConversationStage::Presentation => Some(ConversationStage::Closing),
                    ConversationStage::ObjectionHandling => Some(ConversationStage::Closing),
                    ConversationStage::Discovery => Some(ConversationStage::Closing), // Fast track
                    _ => None,
                }
            },

            // Objection: Handle objection (from any sales stage)
            "objection" if current != ConversationStage::ObjectionHandling => match current {
                ConversationStage::Discovery
                | ConversationStage::Qualification
                | ConversationStage::Presentation
                | ConversationStage::Closing => Some(ConversationStage::ObjectionHandling),
                _ => None,
            },

            // Affirmative: Agreement to proceed to next stage
            "affirmative" => match current {
                ConversationStage::Greeting => Some(ConversationStage::Discovery),
                ConversationStage::Discovery => Some(ConversationStage::Qualification),
                ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
                ConversationStage::Presentation => Some(ConversationStage::Closing),
                ConversationStage::Closing => Some(ConversationStage::Farewell),
                _ => None,
            },

            // Negative: Might need objection handling or early exit
            "negative" => {
                match current {
                    ConversationStage::Closing => Some(ConversationStage::ObjectionHandling),
                    ConversationStage::Presentation => Some(ConversationStage::ObjectionHandling),
                    _ => None, // Don't force exit on negative, handle gracefully
                }
            },

            // Thank you: Positive signal, often near end
            "thank_you" => match current {
                ConversationStage::Closing => Some(ConversationStage::Farewell),
                ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
                _ => None,
            },

            // Farewell: End conversation (from any stage)
            "farewell" => Some(ConversationStage::Farewell),

            // Confusion: May need to revisit discovery
            "confusion" => match current {
                ConversationStage::Presentation => Some(ConversationStage::Discovery),
                _ => None,
            },

            _ => None,
        };

        if let Some(to) = new_stage {
            if current.valid_transitions().contains(&to) {
                let _ = self
                    .stage_manager
                    .transition(to, TransitionReason::IntentDetected(intent.intent.clone()));
                let _ = self
                    .event_tx
                    .send(ConversationEvent::StageChanged { from: current, to });
            }
        }

        // Check if we should suggest next stage
        if let Some(suggested) = self.stage_manager.suggest_next() {
            // Don't auto-transition, just note it
            tracing::debug!("Suggested next stage: {:?}", suggested);
        }
    }

    /// Get memory context
    pub fn get_context(&self) -> String {
        self.memory.get_context()
    }

    /// Get recent messages for LLM
    pub fn get_messages(&self) -> Vec<(String, String)> {
        self.memory.get_recent_messages()
    }

    /// Get stage guidance
    pub fn get_stage_guidance(&self) -> &'static str {
        self.stage().guidance()
    }

    /// Get suggested questions
    pub fn get_suggested_questions(&self) -> Vec<&'static str> {
        self.stage().suggested_questions()
    }

    /// Record a fact
    pub fn record_fact(&self, key: &str, value: &str, confidence: f32) {
        self.memory.add_fact(key, value, confidence);
        self.stage_manager.record_info(key, value);

        let _ = self.event_tx.send(ConversationEvent::FactLearned {
            key: key.to_string(),
            value: value.to_string(),
        });
    }

    /// End the conversation
    pub fn end(&self, reason: EndReason) {
        *self.state.lock() = ConversationState::Ended;
        let _ = self.event_tx.send(ConversationEvent::Ended { reason });
    }

    /// Pause the conversation
    pub fn pause(&self) {
        *self.state.lock() = ConversationState::Paused;
    }

    /// Resume the conversation
    pub fn resume(&self) {
        let mut state = self.state.lock();
        if *state == ConversationState::Paused {
            *state = ConversationState::Active;
            self.update_activity();
        }
    }

    /// Check if active
    fn check_active(&self) -> Result<(), AgentError> {
        let state = *self.state.lock();
        match state {
            ConversationState::Active => {
                // Check timeouts
                if self.duration() > Duration::from_secs(self.config.max_duration_seconds as u64) {
                    self.end(EndReason::MaxDuration);
                    return Err(AgentError::Conversation(
                        "Max duration exceeded".to_string(),
                    ));
                }

                let last = *self.last_activity.lock();
                if last.elapsed() > Duration::from_secs(self.config.session_timeout_seconds as u64)
                {
                    self.end(EndReason::Timeout);
                    return Err(AgentError::Conversation("Session timeout".to_string()));
                }

                Ok(())
            },
            ConversationState::Paused => Err(AgentError::Conversation(
                "Conversation is paused".to_string(),
            )),
            ConversationState::Ended => Err(AgentError::Conversation(
                "Conversation has ended".to_string(),
            )),
        }
    }

    /// Update last activity time
    fn update_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    /// Get memory reference
    pub fn memory(&self) -> &ConversationMemory {
        &self.memory
    }

    /// P1 FIX: Get memory Arc for async operations
    pub fn memory_arc(&self) -> Arc<ConversationMemory> {
        Arc::clone(&self.memory)
    }

    /// Get stage manager reference
    pub fn stage_manager(&self) -> &StageManager {
        &self.stage_manager
    }

    // =========================================================================
    // P0 FIX: Compliance Methods (AI Disclosure, Consent)
    // =========================================================================

    /// Get compliance status
    pub fn compliance(&self) -> ComplianceStatus {
        self.compliance.lock().clone()
    }

    /// Check if AI disclosure has been given
    pub fn ai_disclosure_given(&self) -> bool {
        self.compliance.lock().ai_disclosure.given
    }

    /// Mark AI disclosure as given
    ///
    /// Should be called at the start of conversation after the greeting.
    /// Returns the disclosure message that should be spoken to the customer.
    pub fn mark_ai_disclosed(&self) -> String {
        let language = &self.config.language;
        let disclosure_text = AiDisclosure::get_disclosure_message(language);

        let mut compliance = self.compliance.lock();
        compliance
            .ai_disclosure
            .mark_disclosed(language, disclosure_text, true);
        compliance.update();

        disclosure_text.to_string()
    }

    /// Mark AI disclosure with custom text
    pub fn mark_ai_disclosed_with_text(&self, language: &str, text: &str, human_option: bool) {
        let mut compliance = self.compliance.lock();
        compliance
            .ai_disclosure
            .mark_disclosed(language, text, human_option);
        compliance.update();
    }

    /// Record recording consent
    pub fn record_recording_consent(&self, given: bool, method: ConsentMethod) {
        let mut compliance = self.compliance.lock();
        compliance.consent.record_recording_consent(given, method);
        compliance.consent.consent_language = self.config.language.clone();
        compliance.update();
    }

    /// Record PII processing consent
    pub fn record_pii_consent(&self, given: bool, method: ConsentMethod) {
        let mut compliance = self.compliance.lock();
        compliance.consent.record_pii_consent(given, method);
        compliance.update();
    }

    /// Record marketing consent
    pub fn record_marketing_consent(&self, given: Option<bool>) {
        let mut compliance = self.compliance.lock();
        compliance.consent.record_marketing_consent(given);
    }

    /// Check if conversation is compliant (AI disclosure given, minimum consent obtained)
    pub fn is_compliant(&self) -> bool {
        self.compliance.lock().compliant
    }

    /// Check if PII processing is allowed
    pub fn can_process_pii(&self) -> bool {
        self.compliance.lock().can_process_pii()
    }

    /// Get pending compliance requirements
    pub fn pending_compliance(&self) -> Vec<String> {
        self.compliance.lock().pending_requirements.clone()
    }

    /// Get AI disclosure message for the configured language
    pub fn get_ai_disclosure_message(&self) -> String {
        AiDisclosure::get_disclosure_message(&self.config.language).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_creation() {
        let conv = Conversation::new("test-session", ConversationConfig::default());

        assert_eq!(conv.session_id(), "test-session");
        assert!(conv.is_active());
        assert_eq!(conv.stage(), ConversationStage::Greeting);
    }

    #[test]
    fn test_add_turns() {
        let conv = Conversation::new("test", ConversationConfig::default());

        let intent = conv.add_user_turn("Hello").unwrap();
        assert_eq!(intent.intent, "greeting");

        conv.add_assistant_turn("Hello! How can I help you?")
            .unwrap();

        assert_eq!(conv.turn_count(), 2);
    }

    #[test]
    fn test_stage_transition() {
        let conv = Conversation::new("test", ConversationConfig::default());

        conv.transition_stage(ConversationStage::Discovery).unwrap();
        assert_eq!(conv.stage(), ConversationStage::Discovery);
    }

    #[test]
    fn test_fact_recording() {
        let conv = Conversation::new("test", ConversationConfig::default());

        conv.record_fact("customer_name", "Rajesh", 0.9);

        let fact = conv.memory().get_fact("customer_name");
        assert!(fact.is_some());
        assert_eq!(fact.unwrap().value, "Rajesh");
    }
}
