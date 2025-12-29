//! Behavior signal detection
//!
//! Detects customer behavior patterns from conversation:
//! - Hesitation signals
//! - Interest indicators
//! - Urgency markers
//! - Emotional state
//! - Readiness to proceed

use serde::{Deserialize, Serialize};

/// Detected behavior signal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BehaviorSignal {
    /// Customer is hesitating or uncertain
    Hesitation,
    /// Customer shows interest
    Interest,
    /// Customer shows strong interest / ready to proceed
    StrongInterest,
    /// Customer is in a hurry
    Urgency,
    /// Customer is frustrated
    Frustration,
    /// Customer is confused
    Confusion,
    /// Customer is comparing options
    Comparison,
    /// Customer is skeptical
    Skepticism,
    /// Customer is satisfied
    Satisfaction,
    /// Customer wants to exit
    ExitIntent,
    /// Customer is ready to commit
    Commitment,
    /// Customer needs reassurance
    NeedsReassurance,
}

impl BehaviorSignal {
    /// Get signal display name
    pub fn display_name(&self) -> &'static str {
        match self {
            BehaviorSignal::Hesitation => "Hesitation",
            BehaviorSignal::Interest => "Interest",
            BehaviorSignal::StrongInterest => "Strong Interest",
            BehaviorSignal::Urgency => "Urgency",
            BehaviorSignal::Frustration => "Frustration",
            BehaviorSignal::Confusion => "Confusion",
            BehaviorSignal::Comparison => "Comparison",
            BehaviorSignal::Skepticism => "Skepticism",
            BehaviorSignal::Satisfaction => "Satisfaction",
            BehaviorSignal::ExitIntent => "Exit Intent",
            BehaviorSignal::Commitment => "Commitment",
            BehaviorSignal::NeedsReassurance => "Needs Reassurance",
        }
    }

    /// Check if this is a positive signal
    pub fn is_positive(&self) -> bool {
        matches!(
            self,
            BehaviorSignal::Interest
                | BehaviorSignal::StrongInterest
                | BehaviorSignal::Satisfaction
                | BehaviorSignal::Commitment
        )
    }

    /// Check if this is a negative signal
    pub fn is_negative(&self) -> bool {
        matches!(
            self,
            BehaviorSignal::Frustration
                | BehaviorSignal::ExitIntent
                | BehaviorSignal::Skepticism
        )
    }

    /// Check if this signal needs immediate response adjustment
    pub fn needs_adjustment(&self) -> bool {
        matches!(
            self,
            BehaviorSignal::Frustration
                | BehaviorSignal::Confusion
                | BehaviorSignal::ExitIntent
                | BehaviorSignal::NeedsReassurance
        )
    }

    /// Suggested response strategy for this signal
    pub fn response_strategy(&self) -> &'static str {
        match self {
            BehaviorSignal::Hesitation => "Provide reassurance and address unspoken concerns",
            BehaviorSignal::Interest => "Provide more details and move towards next step",
            BehaviorSignal::StrongInterest => "Offer to proceed with application",
            BehaviorSignal::Urgency => "Emphasize quick processing and offer immediate help",
            BehaviorSignal::Frustration => "Acknowledge frustration, simplify, offer alternative",
            BehaviorSignal::Confusion => "Clarify with simpler explanation, offer examples",
            BehaviorSignal::Comparison => "Highlight unique differentiators, offer comparison",
            BehaviorSignal::Skepticism => "Provide proof points, testimonials, guarantees",
            BehaviorSignal::Satisfaction => "Reinforce positive experience, suggest next step",
            BehaviorSignal::ExitIntent => "Ask what would help, offer alternatives",
            BehaviorSignal::Commitment => "Make it easy to proceed, confirm details",
            BehaviorSignal::NeedsReassurance => "Provide guarantees, success stories, support info",
        }
    }
}

/// Signal detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDetection {
    /// Primary detected signal
    pub primary: BehaviorSignal,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Secondary signals detected
    pub secondary: Vec<(BehaviorSignal, f32)>,
    /// Text patterns that triggered detection
    pub triggers: Vec<String>,
}

impl SignalDetection {
    /// Create a new detection result
    pub fn new(signal: BehaviorSignal, confidence: f32) -> Self {
        Self {
            primary: signal,
            confidence: confidence.clamp(0.0, 1.0),
            secondary: Vec::new(),
            triggers: Vec::new(),
        }
    }

    /// Add secondary signal
    pub fn with_secondary(mut self, signal: BehaviorSignal, confidence: f32) -> Self {
        self.secondary.push((signal, confidence.clamp(0.0, 1.0)));
        self
    }

    /// Add trigger text
    pub fn with_trigger(mut self, trigger: impl Into<String>) -> Self {
        self.triggers.push(trigger.into());
        self
    }

    /// Get overall sentiment (-1.0 to 1.0)
    pub fn sentiment(&self) -> f32 {
        let mut score = 0.0;

        // Primary signal contribution
        if self.primary.is_positive() {
            score += self.confidence;
        } else if self.primary.is_negative() {
            score -= self.confidence;
        }

        // Secondary signals contribution (weighted less)
        for (signal, conf) in &self.secondary {
            if signal.is_positive() {
                score += conf * 0.3;
            } else if signal.is_negative() {
                score -= conf * 0.3;
            }
        }

        score.clamp(-1.0, 1.0)
    }
}

/// Behavior signal detector
pub struct SignalDetector {
    /// Minimum confidence threshold
    min_confidence: f32,
    /// Enable pause analysis
    analyze_pauses: bool,
}

impl SignalDetector {
    /// Create a new detector
    pub fn new() -> Self {
        Self {
            min_confidence: 0.6,
            analyze_pauses: true,
        }
    }

    /// Set minimum confidence threshold
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable pause analysis
    pub fn with_pause_analysis(mut self, enabled: bool) -> Self {
        self.analyze_pauses = enabled;
        self
    }

    /// Detect signals from text
    pub fn detect(&self, text: &str) -> Option<SignalDetection> {
        let lower = text.to_lowercase();
        let mut detections: Vec<(BehaviorSignal, f32, String)> = Vec::new();

        // Hesitation patterns
        let hesitation_patterns = [
            ("hmm", 0.7),
            ("umm", 0.7),
            ("let me think", 0.8),
            ("sochna", 0.75),
            ("not sure", 0.8),
            ("pata nahi", 0.8),
            ("maybe", 0.6),
            ("shayad", 0.65),
            ("i don't know", 0.85),
            ("mujhe nahi pata", 0.85),
            ("need to check", 0.7),
            ("...", 0.6),
        ];

        for (pattern, conf) in hesitation_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Hesitation, conf, pattern.to_string()));
            }
        }

        // Interest patterns
        let interest_patterns = [
            ("tell me more", 0.85),
            ("aur batao", 0.85),
            ("interesting", 0.8),
            ("accha", 0.7),
            ("sounds good", 0.85),
            ("how do i", 0.8),
            ("kaise karu", 0.8),
            ("what's the process", 0.85),
            ("apply", 0.75),
            ("next step", 0.9),
            ("aage kya", 0.85),
        ];

        for (pattern, conf) in interest_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Interest, conf, pattern.to_string()));
            }
        }

        // Strong interest patterns
        let strong_interest_patterns = [
            ("let's do it", 0.95),
            ("i want to", 0.9),
            ("sign me up", 0.95),
            ("mujhe chahiye", 0.9),
            ("abhi karo", 0.9),
            ("ready", 0.85),
            ("proceed", 0.9),
            ("let's proceed", 0.95),
            ("aage badho", 0.9),
        ];

        for (pattern, conf) in strong_interest_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::StrongInterest, conf, pattern.to_string()));
            }
        }

        // Urgency patterns
        let urgency_patterns = [
            ("urgent", 0.95),
            ("jaldi", 0.9),
            ("quickly", 0.85),
            ("today", 0.75),
            ("aaj hi", 0.9),
            ("asap", 0.95),
            ("right now", 0.9),
            ("abhi", 0.85),
            ("emergency", 0.95),
        ];

        for (pattern, conf) in urgency_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Urgency, conf, pattern.to_string()));
            }
        }

        // Frustration patterns
        let frustration_patterns = [
            ("frustrated", 0.95),
            ("this is taking", 0.8),
            ("why so", 0.75),
            ("not working", 0.85),
            ("problem", 0.7),
            ("dikkat", 0.8),
            ("pareshan", 0.85),
            ("waste of time", 0.95),
            ("bekar", 0.8),
        ];

        for (pattern, conf) in frustration_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Frustration, conf, pattern.to_string()));
            }
        }

        // Confusion patterns
        let confusion_patterns = [
            ("confused", 0.9),
            ("don't understand", 0.9),
            ("samajh nahi", 0.9),
            ("what do you mean", 0.85),
            ("kya matlab", 0.85),
            ("can you explain", 0.8),
            ("phir se batao", 0.85),
            ("sorry?", 0.7),
            ("huh", 0.65),
        ];

        for (pattern, conf) in confusion_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Confusion, conf, pattern.to_string()));
            }
        }

        // Comparison patterns
        let comparison_patterns = [
            ("compared to", 0.9),
            ("vs", 0.8),
            ("better than", 0.85),
            ("muthoot", 0.8),
            ("manappuram", 0.8),
            ("iifl", 0.8),
            ("other bank", 0.85),
            ("difference", 0.8),
            ("farak kya hai", 0.85),
        ];

        for (pattern, conf) in comparison_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Comparison, conf, pattern.to_string()));
            }
        }

        // Skepticism patterns
        let skepticism_patterns = [
            ("really", 0.6),
            ("sure about", 0.75),
            ("hard to believe", 0.9),
            ("yakeen nahi", 0.9),
            ("seems too good", 0.85),
            ("catch", 0.8),
            ("hidden", 0.8),
            ("chhupa", 0.8),
            ("is this true", 0.85),
            ("sach mein", 0.8),
        ];

        for (pattern, conf) in skepticism_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Skepticism, conf, pattern.to_string()));
            }
        }

        // Satisfaction patterns
        let satisfaction_patterns = [
            ("great", 0.8),
            ("perfect", 0.9),
            ("bahut accha", 0.9),
            ("wonderful", 0.9),
            ("excellent", 0.9),
            ("satisfied", 0.95),
            ("happy", 0.85),
            ("khush", 0.85),
            ("thanks", 0.7),
            ("dhanyawad", 0.75),
        ];

        for (pattern, conf) in satisfaction_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Satisfaction, conf, pattern.to_string()));
            }
        }

        // Exit intent patterns
        let exit_patterns = [
            ("not interested", 0.95),
            ("nahi chahiye", 0.95),
            ("no thanks", 0.9),
            ("bye", 0.85),
            ("alvida", 0.85),
            ("later", 0.75),
            ("call back", 0.7),
            ("phir kabhi", 0.8),
            ("don't call", 0.95),
            ("stop", 0.85),
        ];

        for (pattern, conf) in exit_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::ExitIntent, conf, pattern.to_string()));
            }
        }

        // Commitment patterns
        let commitment_patterns = [
            ("yes", 0.6),
            ("haan", 0.65),
            ("ok let's", 0.85),
            ("i agree", 0.9),
            ("done", 0.85),
            ("theek hai", 0.8),
            ("confirm", 0.9),
            ("book", 0.8),
            ("schedule", 0.8),
        ];

        for (pattern, conf) in commitment_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::Commitment, conf, pattern.to_string()));
            }
        }

        // Needs reassurance patterns
        let reassurance_patterns = [
            ("are you sure", 0.9),
            ("guarantee", 0.85),
            ("promise", 0.85),
            ("pakka", 0.9),
            ("safe", 0.75),
            ("surakshit", 0.8),
            ("trust", 0.8),
            ("bharosa", 0.85),
            ("what if", 0.75),
            ("agar", 0.7),
        ];

        for (pattern, conf) in reassurance_patterns {
            if lower.contains(pattern) {
                detections.push((BehaviorSignal::NeedsReassurance, conf, pattern.to_string()));
            }
        }

        // Filter by confidence threshold
        detections.retain(|(_, conf, _)| *conf >= self.min_confidence);

        if detections.is_empty() {
            return None;
        }

        // Sort by confidence
        detections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Build result
        let (primary, primary_conf, primary_trigger) = detections.remove(0);
        let mut result = SignalDetection::new(primary, primary_conf)
            .with_trigger(primary_trigger);

        // Add secondary signals
        for (signal, conf, trigger) in detections.into_iter().take(3) {
            result = result.with_secondary(signal, conf).with_trigger(trigger);
        }

        Some(result)
    }

    /// Detect signals from transcript with timing info
    pub fn detect_with_timing(
        &self,
        text: &str,
        pause_duration_ms: Option<u64>,
        speech_rate_wpm: Option<f32>,
    ) -> Option<SignalDetection> {
        let mut detection = self.detect(text)?;

        if self.analyze_pauses {
            // Long pause indicates hesitation
            if let Some(pause) = pause_duration_ms {
                if pause > 3000 {
                    detection = detection
                        .with_secondary(BehaviorSignal::Hesitation, 0.8)
                        .with_trigger("long_pause");
                } else if pause > 1500 {
                    detection = detection
                        .with_secondary(BehaviorSignal::Hesitation, 0.5)
                        .with_trigger("medium_pause");
                }
            }

            // Speech rate analysis
            if let Some(rate) = speech_rate_wpm {
                if rate > 180.0 {
                    detection = detection
                        .with_secondary(BehaviorSignal::Urgency, 0.6)
                        .with_trigger("fast_speech");
                } else if rate < 80.0 {
                    detection = detection
                        .with_secondary(BehaviorSignal::Hesitation, 0.5)
                        .with_trigger("slow_speech");
                }
            }
        }

        Some(detection)
    }

    /// Analyze a sequence of signals for trend
    pub fn analyze_trend(&self, signals: &[BehaviorSignal]) -> TrendAnalysis {
        if signals.is_empty() {
            return TrendAnalysis::Neutral;
        }

        let recent = signals.iter().rev().take(3);
        let positive_count = recent.clone().filter(|s| s.is_positive()).count();
        let negative_count = recent.filter(|s| s.is_negative()).count();

        if positive_count >= 2 {
            TrendAnalysis::Improving
        } else if negative_count >= 2 {
            TrendAnalysis::Declining
        } else if signals.last().map(|s| s.is_positive()).unwrap_or(false) {
            TrendAnalysis::SlightlyImproving
        } else if signals.last().map(|s| s.is_negative()).unwrap_or(false) {
            TrendAnalysis::SlightlyDeclining
        } else {
            TrendAnalysis::Neutral
        }
    }
}

impl Default for SignalDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Trend analysis result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrendAnalysis {
    /// Signals are improving
    Improving,
    /// Slight improvement
    SlightlyImproving,
    /// Neutral/stable
    Neutral,
    /// Slight decline
    SlightlyDeclining,
    /// Signals are declining
    Declining,
}

impl TrendAnalysis {
    /// Get recommended action
    pub fn recommended_action(&self) -> &'static str {
        match self {
            TrendAnalysis::Improving => "Continue current approach, move towards close",
            TrendAnalysis::SlightlyImproving => "Maintain momentum, reinforce positive points",
            TrendAnalysis::Neutral => "Try a new angle or ask discovery question",
            TrendAnalysis::SlightlyDeclining => "Address potential concern, offer reassurance",
            TrendAnalysis::Declining => "Pause, acknowledge concern, try different approach",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_display() {
        assert_eq!(BehaviorSignal::Interest.display_name(), "Interest");
        assert!(BehaviorSignal::Interest.is_positive());
        assert!(BehaviorSignal::Frustration.is_negative());
    }

    #[test]
    fn test_detect_hesitation() {
        let detector = SignalDetector::new();
        let result = detector.detect("Hmm, let me think about this").unwrap();

        assert_eq!(result.primary, BehaviorSignal::Hesitation);
        assert!(result.confidence >= 0.6);
    }

    #[test]
    fn test_detect_interest() {
        let detector = SignalDetector::new();
        let result = detector.detect("Tell me more about the process").unwrap();

        assert_eq!(result.primary, BehaviorSignal::Interest);
    }

    #[test]
    fn test_detect_strong_interest() {
        let detector = SignalDetector::new();
        let result = detector.detect("Let's do it, I want to apply").unwrap();

        assert_eq!(result.primary, BehaviorSignal::StrongInterest);
    }

    #[test]
    fn test_detect_frustration() {
        let detector = SignalDetector::new();
        let result = detector.detect("I'm frustrated, this is taking too long").unwrap();

        assert_eq!(result.primary, BehaviorSignal::Frustration);
        assert!(result.primary.is_negative());
        assert!(result.primary.needs_adjustment());
    }

    #[test]
    fn test_detect_hindi() {
        let detector = SignalDetector::new();

        let result = detector.detect("Mujhe sochna padega").unwrap();
        assert_eq!(result.primary, BehaviorSignal::Hesitation);

        let result = detector.detect("Bahut accha!").unwrap();
        assert_eq!(result.primary, BehaviorSignal::Satisfaction);
    }

    #[test]
    fn test_detect_comparison() {
        let detector = SignalDetector::new();
        let result = detector.detect("How does this compare to Muthoot?").unwrap();

        assert_eq!(result.primary, BehaviorSignal::Comparison);
    }

    #[test]
    fn test_no_signal() {
        let detector = SignalDetector::new();
        // Use neutral text that doesn't match any patterns
        let result = detector.detect("The loan amount is fifty thousand rupees");

        assert!(result.is_none());
    }

    #[test]
    fn test_secondary_signals() {
        let detector = SignalDetector::new();
        let result = detector
            .detect("Let me think... but tell me more about the rates")
            .unwrap();

        assert!(!result.secondary.is_empty());
    }

    #[test]
    fn test_sentiment() {
        let detector = SignalDetector::new();

        let positive = detector.detect("Great, sounds perfect!").unwrap();
        assert!(positive.sentiment() > 0.0);

        let negative = detector.detect("I'm frustrated with this").unwrap();
        assert!(negative.sentiment() < 0.0);
    }

    #[test]
    fn test_with_timing() {
        let detector = SignalDetector::new();
        let result = detector
            .detect_with_timing("Let me think", Some(4000), None)
            .unwrap();

        // Should have hesitation from both text and pause
        assert_eq!(result.primary, BehaviorSignal::Hesitation);
        assert!(result.triggers.iter().any(|t| t.contains("pause")));
    }

    #[test]
    fn test_trend_analysis() {
        let detector = SignalDetector::new();

        let signals = vec![
            BehaviorSignal::Hesitation,
            BehaviorSignal::Interest,
            BehaviorSignal::StrongInterest,
        ];
        assert_eq!(detector.analyze_trend(&signals), TrendAnalysis::Improving);

        let signals = vec![
            BehaviorSignal::Interest,
            BehaviorSignal::Frustration,
            BehaviorSignal::ExitIntent,
        ];
        assert_eq!(detector.analyze_trend(&signals), TrendAnalysis::Declining);
    }

    #[test]
    fn test_response_strategy() {
        let strategy = BehaviorSignal::Confusion.response_strategy();
        assert!(strategy.contains("Clarify") || strategy.contains("simpl"));
    }
}
