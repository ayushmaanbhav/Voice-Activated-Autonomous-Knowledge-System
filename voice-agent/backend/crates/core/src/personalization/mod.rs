//! Personalization Engine
//!
//! Comprehensive personalization system for adapting agent responses:
//! - Persona management (tone, warmth, style)
//! - Segment-aware response adaptation
//! - Behavior signal detection
//! - Dynamic response customization
//!
//! # Architecture
//!
//! ```text
//! CustomerProfile → SegmentAdapter → Persona Selection
//!                                           ↓
//! User Input → SignalDetector → PersonalizationEngine → Response Adaptation
//!                                           ↓
//!                              System Prompt + Context
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use voice_agent_core::personalization::*;
//! use voice_agent_core::{CustomerProfile, CustomerSegment};
//!
//! let engine = PersonalizationEngine::new();
//!
//! // Create profile
//! let profile = CustomerProfile::new()
//!     .segment(CustomerSegment::TrustSeeker);
//!
//! // Create context
//! let mut ctx = PersonalizationContext::for_profile(&profile);
//!
//! // Detect signals and adapt
//! if let Some(signal) = engine.detect_signal("Is my gold safe?") {
//!     ctx = ctx.with_signal(signal);
//! }
//!
//! // Get personalized instructions
//! let instructions = engine.generate_instructions(&ctx);
//! ```

pub mod persona;
pub mod adaptation;
pub mod signals;

pub use persona::{
    Persona, PersonaTemplates, Tone, LanguageComplexity, ResponseUrgency,
};
pub use adaptation::{
    SegmentAdapter, Feature, Objection, ObjectionResponse,
};
pub use signals::{
    SignalDetector, BehaviorSignal, SignalDetection, TrendAnalysis,
};

use crate::{CustomerProfile, CustomerSegment};
use serde::{Deserialize, Serialize};

/// Personalization context for a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationContext {
    /// Customer segment
    pub segment: Option<CustomerSegment>,
    /// Active persona
    pub persona: Persona,
    /// Detected behavior signals
    pub signals: Vec<BehaviorSignal>,
    /// Current sentiment (-1.0 to 1.0)
    pub sentiment: f32,
    /// Conversation turn count
    pub turn_count: usize,
    /// Whether objection was detected
    pub has_objection: bool,
    /// Detected objection type
    pub current_objection: Option<Objection>,
    /// Customer name for personalization
    pub customer_name: Option<String>,
    /// Preferred language
    pub preferred_language: String,
}

impl PersonalizationContext {
    /// Create context for a customer profile
    pub fn for_profile(profile: &CustomerProfile) -> Self {
        let segment = profile.infer_segment();
        let persona = segment
            .map(Persona::for_segment)
            .unwrap_or_default();

        Self {
            segment,
            persona,
            signals: Vec::new(),
            sentiment: 0.0,
            turn_count: 0,
            has_objection: false,
            current_objection: None,
            customer_name: profile.name.clone(),
            preferred_language: profile.preferred_language.clone(),
        }
    }

    /// Create default context
    pub fn new() -> Self {
        Self {
            segment: None,
            persona: Persona::default(),
            signals: Vec::new(),
            sentiment: 0.0,
            turn_count: 0,
            has_objection: false,
            current_objection: None,
            customer_name: None,
            preferred_language: "en".to_string(),
        }
    }

    /// Set segment
    pub fn with_segment(mut self, segment: CustomerSegment) -> Self {
        self.segment = Some(segment);
        self.persona = Persona::for_segment(segment);
        self
    }

    /// Set persona
    pub fn with_persona(mut self, persona: Persona) -> Self {
        self.persona = persona;
        self
    }

    /// Add behavior signal
    pub fn with_signal(mut self, signal: BehaviorSignal) -> Self {
        self.signals.push(signal);
        self
    }

    /// Set customer name
    pub fn with_customer_name(mut self, name: impl Into<String>) -> Self {
        self.customer_name = Some(name.into());
        self
    }

    /// Update from signal detection
    pub fn update_from_detection(&mut self, detection: &SignalDetection) {
        self.signals.push(detection.primary);
        self.sentiment = (self.sentiment + detection.sentiment()) / 2.0;

        for (signal, _) in &detection.secondary {
            self.signals.push(*signal);
        }
    }

    /// Record objection
    pub fn record_objection(&mut self, objection: Objection) {
        self.has_objection = true;
        self.current_objection = Some(objection);
    }

    /// Clear objection after handling
    pub fn clear_objection(&mut self) {
        self.current_objection = None;
    }

    /// Increment turn count
    pub fn next_turn(&mut self) {
        self.turn_count += 1;
    }

    /// Get recent signals (last N)
    pub fn recent_signals(&self, n: usize) -> &[BehaviorSignal] {
        let start = self.signals.len().saturating_sub(n);
        &self.signals[start..]
    }

    /// Check if sentiment is positive
    pub fn is_positive(&self) -> bool {
        self.sentiment > 0.2
    }

    /// Check if sentiment is negative
    pub fn is_negative(&self) -> bool {
        self.sentiment < -0.2
    }
}

impl Default for PersonalizationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Main personalization engine
pub struct PersonalizationEngine {
    /// Signal detector
    signal_detector: SignalDetector,
    /// Segment adapter
    segment_adapter: SegmentAdapter,
    /// Enable adaptive persona
    adaptive_persona: bool,
}

impl PersonalizationEngine {
    /// Create a new personalization engine
    pub fn new() -> Self {
        Self {
            signal_detector: SignalDetector::new(),
            segment_adapter: SegmentAdapter::new(),
            adaptive_persona: true,
        }
    }

    /// Create with custom signal detector
    pub fn with_signal_detector(mut self, detector: SignalDetector) -> Self {
        self.signal_detector = detector;
        self
    }

    /// Create with custom segment adapter
    pub fn with_segment_adapter(mut self, adapter: SegmentAdapter) -> Self {
        self.segment_adapter = adapter;
        self
    }

    /// Enable/disable adaptive persona
    pub fn with_adaptive_persona(mut self, enabled: bool) -> Self {
        self.adaptive_persona = enabled;
        self
    }

    /// Detect signal from user input
    pub fn detect_signal(&self, text: &str) -> Option<SignalDetection> {
        self.signal_detector.detect(text)
    }

    /// Detect signal with timing information
    pub fn detect_signal_with_timing(
        &self,
        text: &str,
        pause_ms: Option<u64>,
        speech_rate: Option<f32>,
    ) -> Option<SignalDetection> {
        self.signal_detector.detect_with_timing(text, pause_ms, speech_rate)
    }

    /// Detect objection from user input
    pub fn detect_objection(&self, text: &str) -> Option<Objection> {
        Objection::detect(text)
    }

    /// Handle objection and return response
    pub fn handle_objection(
        &self,
        ctx: &PersonalizationContext,
        objection: Objection,
    ) -> Option<String> {
        let segment = ctx.segment.unwrap_or(CustomerSegment::FirstTime);
        self.segment_adapter.handle_objection(
            segment,
            objection,
            ctx.customer_name.as_deref(),
        )
    }

    /// Get priority features for segment
    pub fn get_features(&self, segment: CustomerSegment) -> Vec<Feature> {
        self.segment_adapter.get_features(segment)
    }

    /// Get value propositions for segment
    pub fn get_value_propositions(&self, segment: CustomerSegment) -> Vec<String> {
        self.segment_adapter.get_value_propositions(segment)
    }

    /// Analyze conversation trend
    pub fn analyze_trend(&self, ctx: &PersonalizationContext) -> TrendAnalysis {
        self.signal_detector.analyze_trend(&ctx.signals)
    }

    /// Get adapted persona based on context
    pub fn get_adapted_persona(&self, ctx: &PersonalizationContext) -> Persona {
        if !self.adaptive_persona {
            return ctx.persona.clone();
        }

        let mut persona = ctx.persona.clone();

        // Adjust based on recent signals
        let recent = ctx.recent_signals(3);

        // Increase empathy if frustration or confusion detected
        if recent.iter().any(|s| matches!(s, BehaviorSignal::Frustration | BehaviorSignal::Confusion)) {
            persona.empathy = (persona.empathy + 0.2).min(1.0);
            persona.warmth = (persona.warmth + 0.1).min(1.0);
        }

        // Increase urgency if interest detected
        if recent.iter().any(|s| matches!(s, BehaviorSignal::StrongInterest | BehaviorSignal::Commitment)) {
            persona.urgency = ResponseUrgency::Efficient;
        }

        // Slow down if hesitation detected
        if recent.iter().filter(|s| matches!(s, BehaviorSignal::Hesitation)).count() >= 2 {
            persona.urgency = ResponseUrgency::Relaxed;
        }

        // If exit intent, maximize empathy
        if recent.iter().any(|s| matches!(s, BehaviorSignal::ExitIntent)) {
            persona.empathy = 0.95;
            persona.warmth = 0.9;
            persona.acknowledge_emotions = true;
        }

        persona
    }

    /// Generate system prompt instructions
    pub fn generate_instructions(&self, ctx: &PersonalizationContext) -> String {
        let persona = self.get_adapted_persona(ctx);
        let mut instructions = persona.system_prompt_instructions();

        // Add segment-specific guidance
        if let Some(segment) = ctx.segment {
            let features = self.segment_adapter.get_top_features(segment, 3);
            if !features.is_empty() {
                instructions.push_str(&format!(
                    " Emphasize these features: {}.",
                    features
                        .iter()
                        .map(|f| f.display_name())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        // Add signal-based guidance
        let trend = self.analyze_trend(ctx);
        instructions.push_str(&format!(" Current strategy: {}", trend.recommended_action()));

        // Add customer name guidance
        if let Some(ref name) = ctx.customer_name {
            instructions.push_str(&format!(" Customer name is {}.", name));
        }

        // Add language guidance
        if ctx.preferred_language != "en" {
            instructions.push_str(&format!(
                " Customer prefers {}. Use code-switching if appropriate.",
                ctx.preferred_language
            ));
        }

        // Add objection guidance
        if ctx.has_objection {
            instructions.push_str(" Customer has raised an objection. Address it empathetically before proceeding.");
        }

        instructions
    }

    /// Process user input and update context
    pub fn process_input(&self, ctx: &mut PersonalizationContext, text: &str) {
        ctx.next_turn();

        // Detect signals
        if let Some(detection) = self.detect_signal(text) {
            ctx.update_from_detection(&detection);
        }

        // Detect objections
        if let Some(objection) = self.detect_objection(text) {
            ctx.record_objection(objection);
        }
    }

    /// Get segment adapter
    pub fn segment_adapter(&self) -> &SegmentAdapter {
        &self.segment_adapter
    }

    /// Get mutable segment adapter
    pub fn segment_adapter_mut(&mut self) -> &mut SegmentAdapter {
        &mut self.segment_adapter
    }

    /// Get signal detector
    pub fn signal_detector(&self) -> &SignalDetector {
        &self.signal_detector
    }
}

impl Default for PersonalizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_from_profile() {
        let profile = CustomerProfile::new()
            .name("Raj Kumar")
            .current_lender("Muthoot");

        let ctx = PersonalizationContext::for_profile(&profile);

        assert_eq!(ctx.customer_name, Some("Raj Kumar".to_string()));
        assert_eq!(ctx.segment, Some(CustomerSegment::TrustSeeker));
    }

    #[test]
    fn test_context_signals() {
        let mut ctx = PersonalizationContext::new()
            .with_signal(BehaviorSignal::Interest);

        ctx.update_from_detection(&SignalDetection::new(BehaviorSignal::StrongInterest, 0.9));

        assert_eq!(ctx.signals.len(), 2);
        assert!(ctx.sentiment > 0.0);
    }

    #[test]
    fn test_engine_detect_signal() {
        let engine = PersonalizationEngine::new();

        let detection = engine.detect_signal("Tell me more about this").unwrap();
        assert_eq!(detection.primary, BehaviorSignal::Interest);
    }

    #[test]
    fn test_engine_detect_objection() {
        let engine = PersonalizationEngine::new();

        let objection = engine.detect_objection("Is my gold safe with you?").unwrap();
        assert_eq!(objection, Objection::GoldSafety);
    }

    #[test]
    fn test_engine_handle_objection() {
        let engine = PersonalizationEngine::new();
        let ctx = PersonalizationContext::new()
            .with_segment(CustomerSegment::TrustSeeker)
            .with_customer_name("Raj");

        let response = engine.handle_objection(&ctx, Objection::GoldSafety);
        assert!(response.is_some());
        assert!(response.unwrap().contains("RBI"));
    }

    #[test]
    fn test_adapted_persona() {
        let engine = PersonalizationEngine::new();

        // Context with frustration
        let ctx = PersonalizationContext::new()
            .with_segment(CustomerSegment::Professional)
            .with_signal(BehaviorSignal::Frustration);

        let persona = engine.get_adapted_persona(&ctx);
        // Empathy should be increased
        assert!(persona.empathy > Persona::for_segment(CustomerSegment::Professional).empathy);
    }

    #[test]
    fn test_generate_instructions() {
        let engine = PersonalizationEngine::new();
        let ctx = PersonalizationContext::new()
            .with_segment(CustomerSegment::TrustSeeker)
            .with_customer_name("Priya");

        let instructions = engine.generate_instructions(&ctx);

        assert!(instructions.contains("empathy") || instructions.contains("Acknowledge"));
        assert!(instructions.contains("Priya"));
        assert!(instructions.contains("RBI") || instructions.contains("Security"));
    }

    #[test]
    fn test_process_input() {
        let engine = PersonalizationEngine::new();
        let mut ctx = PersonalizationContext::new();

        engine.process_input(&mut ctx, "Tell me more about this, sounds interesting");

        assert_eq!(ctx.turn_count, 1);
        assert!(!ctx.signals.is_empty());
    }

    #[test]
    fn test_trend_analysis() {
        let engine = PersonalizationEngine::new();
        let ctx = PersonalizationContext::new()
            .with_signal(BehaviorSignal::Interest)
            .with_signal(BehaviorSignal::StrongInterest);

        let trend = engine.analyze_trend(&ctx);
        assert_eq!(trend, TrendAnalysis::Improving);
    }

    #[test]
    fn test_get_features() {
        let engine = PersonalizationEngine::new();
        let features = engine.get_features(CustomerSegment::PriceSensitive);

        assert!(features.contains(&Feature::LowRates));
        assert!(features.contains(&Feature::ZeroForeclosure));
    }
}
