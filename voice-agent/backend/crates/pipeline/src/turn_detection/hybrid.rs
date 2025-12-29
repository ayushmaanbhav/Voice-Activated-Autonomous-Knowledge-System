//! Hybrid Turn Detector
//!
//! Combines VAD silence detection with semantic completeness analysis.
//! Dynamically adjusts silence threshold based on utterance type.

use std::time::{Duration, Instant};
use parking_lot::Mutex;

use super::semantic::{SemanticTurnDetector, SemanticConfig, CompletenessClass};
use crate::vad::VadState;
use crate::PipelineError;

/// Turn detection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum TurnState {
    /// Waiting for user to speak
    #[default]
    Idle,
    /// User is speaking
    UserSpeaking,
    /// User paused, evaluating turn completion
    Evaluating,
    /// Turn complete, ready for response
    TurnComplete,
    /// Agent is responding
    AgentSpeaking,
}


/// Turn detection result
#[derive(Debug, Clone)]
pub struct TurnDetectionResult {
    /// Current state
    pub state: TurnState,
    /// Whether turn is complete
    pub is_turn_complete: bool,
    /// Semantic completeness if analyzed
    pub semantic_class: Option<CompletenessClass>,
    /// Confidence in turn completion
    pub confidence: f32,
    /// Elapsed silence duration
    pub silence_duration: Duration,
    /// Dynamic silence threshold being used
    pub silence_threshold: Duration,
}

/// Configuration for hybrid turn detection
#[derive(Debug, Clone)]
pub struct TurnDetectionConfig {
    /// Base silence threshold (before semantic adjustment)
    pub base_silence_ms: u32,
    /// Minimum silence threshold
    pub min_silence_ms: u32,
    /// Maximum silence threshold
    pub max_silence_ms: u32,
    /// Minimum speech duration to consider (avoid false triggers)
    pub min_speech_ms: u32,
    /// Enable semantic analysis
    pub semantic_enabled: bool,
    /// Semantic config
    pub semantic_config: SemanticConfig,
    /// Weight for semantic vs VAD decision
    pub semantic_weight: f32,
}

impl Default for TurnDetectionConfig {
    fn default() -> Self {
        Self {
            base_silence_ms: 500,
            min_silence_ms: 200,
            max_silence_ms: 1000,
            min_speech_ms: 200,
            semantic_enabled: true,
            semantic_config: SemanticConfig::default(),
            semantic_weight: 0.6,
        }
    }
}

/// Internal state for tracking
struct InternalState {
    state: TurnState,
    speech_start: Option<Instant>,
    silence_start: Option<Instant>,
    current_transcript: String,
    last_semantic_class: Option<CompletenessClass>,
    last_semantic_confidence: f32,
    dynamic_threshold: Duration,
}

/// Hybrid Turn Detector
pub struct HybridTurnDetector {
    config: TurnDetectionConfig,
    semantic: Option<SemanticTurnDetector>,
    internal: Mutex<InternalState>,
}

impl HybridTurnDetector {
    /// Create a new hybrid turn detector
    pub fn new(config: TurnDetectionConfig) -> Self {
        let semantic = if config.semantic_enabled {
            SemanticTurnDetector::simple(config.semantic_config.clone()).ok()
        } else {
            None
        };

        Self {
            internal: Mutex::new(InternalState {
                state: TurnState::Idle,
                speech_start: None,
                silence_start: None,
                current_transcript: String::new(),
                last_semantic_class: None,
                last_semantic_confidence: 0.0,
                dynamic_threshold: Duration::from_millis(config.base_silence_ms as u64),
            }),
            config,
            semantic,
        }
    }

    /// Create with custom semantic detector
    pub fn with_semantic(
        config: TurnDetectionConfig,
        semantic: SemanticTurnDetector,
    ) -> Self {
        Self {
            internal: Mutex::new(InternalState {
                state: TurnState::Idle,
                speech_start: None,
                silence_start: None,
                current_transcript: String::new(),
                last_semantic_class: None,
                last_semantic_confidence: 0.0,
                dynamic_threshold: Duration::from_millis(config.base_silence_ms as u64),
            }),
            config,
            semantic: Some(semantic),
        }
    }

    /// Process VAD result and optional transcript update
    pub fn process(
        &self,
        vad_state: VadState,
        transcript: Option<&str>,
    ) -> Result<TurnDetectionResult, PipelineError> {
        // P1 FIX: Get timestamp before acquiring lock to avoid syscall while holding mutex
        let now = Instant::now();
        let mut internal = self.internal.lock();

        // Update transcript if provided
        if let Some(text) = transcript {
            if !text.is_empty() {
                internal.current_transcript = text.to_string();

                // Run semantic analysis
                if let Some(ref semantic) = self.semantic {
                    if let Ok((class, conf)) = semantic.classify(text) {
                        internal.last_semantic_class = Some(class);
                        internal.last_semantic_confidence = conf;

                        // Update dynamic threshold based on semantic class
                        let suggested = class.suggested_silence_ms();
                        internal.dynamic_threshold = Duration::from_millis(
                            suggested.clamp(
                                self.config.min_silence_ms,
                                self.config.max_silence_ms,
                            ) as u64
                        );
                    }
                }
            }
        }

        // State machine transitions
        let (new_state, is_turn_complete) = match (internal.state, vad_state) {
            // Idle -> UserSpeaking when speech starts
            (TurnState::Idle, VadState::Speech) |
            (TurnState::Idle, VadState::SpeechStart) => {
                internal.speech_start = Some(now);
                internal.silence_start = None;
                (TurnState::UserSpeaking, false)
            }

            // UserSpeaking continues
            (TurnState::UserSpeaking, VadState::Speech) => {
                internal.silence_start = None;
                (TurnState::UserSpeaking, false)
            }

            // UserSpeaking -> Evaluating when speech ends
            (TurnState::UserSpeaking, VadState::SpeechEnd) |
            (TurnState::UserSpeaking, VadState::Silence) => {
                internal.silence_start = Some(now);
                (TurnState::Evaluating, false)
            }

            // Evaluating -> check silence duration
            (TurnState::Evaluating, VadState::Silence) => {
                let silence_duration = internal.silence_start
                    .map(|s| now.duration_since(s))
                    .unwrap_or_default();

                // Check if we've had enough speech
                let speech_duration = internal.speech_start
                    .map(|s| {
                        let end = internal.silence_start.unwrap_or(now);
                        end.duration_since(s)
                    })
                    .unwrap_or_default();

                let min_speech = Duration::from_millis(self.config.min_speech_ms as u64);

                if speech_duration < min_speech {
                    // Not enough speech, keep waiting
                    (TurnState::Evaluating, false)
                } else if silence_duration >= internal.dynamic_threshold {
                    // Turn complete
                    (TurnState::TurnComplete, true)
                } else {
                    // Keep evaluating
                    (TurnState::Evaluating, false)
                }
            }

            // Evaluating -> UserSpeaking if speech resumes
            (TurnState::Evaluating, VadState::Speech) |
            (TurnState::Evaluating, VadState::SpeechStart) => {
                internal.silence_start = None;
                (TurnState::UserSpeaking, false)
            }

            // TurnComplete -> stays complete until reset
            (TurnState::TurnComplete, _) => {
                (TurnState::TurnComplete, true)
            }

            // AgentSpeaking -> can transition based on barge-in
            (TurnState::AgentSpeaking, VadState::Speech) |
            (TurnState::AgentSpeaking, VadState::SpeechStart) => {
                // Potential barge-in - handled by orchestrator
                internal.speech_start = Some(now);
                (TurnState::UserSpeaking, false)
            }

            (TurnState::AgentSpeaking, _) => {
                (TurnState::AgentSpeaking, false)
            }

            // Default: stay in current state
            (state, _) => (state, false)
        };

        internal.state = new_state;

        let silence_duration = internal.silence_start
            .map(|s| now.duration_since(s))
            .unwrap_or_default();

        // Calculate confidence
        let confidence = if is_turn_complete {
            self.calculate_confidence(&internal, silence_duration)
        } else {
            0.0
        };

        Ok(TurnDetectionResult {
            state: new_state,
            is_turn_complete,
            semantic_class: internal.last_semantic_class,
            confidence,
            silence_duration,
            silence_threshold: internal.dynamic_threshold,
        })
    }

    /// Calculate turn completion confidence
    fn calculate_confidence(&self, internal: &InternalState, silence: Duration) -> f32 {
        let mut confidence = 0.0;

        // Silence-based confidence (0.4 weight)
        let silence_ratio = silence.as_millis() as f32 /
            internal.dynamic_threshold.as_millis() as f32;
        confidence += 0.4 * silence_ratio.min(1.5);

        // Semantic-based confidence (0.6 weight if available)
        if let Some(class) = internal.last_semantic_class {
            let semantic_conf = internal.last_semantic_confidence;
            let class_weight = match class {
                CompletenessClass::Complete => 1.0,
                CompletenessClass::Question => 0.95,
                CompletenessClass::PossiblyComplete => 0.7,
                CompletenessClass::Backchannel => 0.3,
                CompletenessClass::Incomplete => 0.2,
            };
            confidence += self.config.semantic_weight * semantic_conf * class_weight;
        } else {
            // No semantic analysis, boost silence weight
            confidence += 0.3 * silence_ratio.min(1.0);
        }

        confidence.min(1.0)
    }

    /// Reset turn detector state
    pub fn reset(&self) {
        let mut internal = self.internal.lock();
        internal.state = TurnState::Idle;
        internal.speech_start = None;
        internal.silence_start = None;
        internal.current_transcript.clear();
        internal.last_semantic_class = None;
        internal.last_semantic_confidence = 0.0;
        internal.dynamic_threshold = Duration::from_millis(self.config.base_silence_ms as u64);

        if let Some(ref semantic) = self.semantic {
            semantic.reset();
        }
    }

    /// Get current state
    pub fn state(&self) -> TurnState {
        self.internal.lock().state
    }

    /// Set agent speaking state (called when TTS starts)
    pub fn set_agent_speaking(&self) {
        let mut internal = self.internal.lock();
        internal.state = TurnState::AgentSpeaking;
        internal.speech_start = None;
        internal.silence_start = None;
    }

    /// Get current transcript
    pub fn current_transcript(&self) -> String {
        self.internal.lock().current_transcript.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_detection_basic() {
        let detector = HybridTurnDetector::new(TurnDetectionConfig::default());

        // Start with idle
        assert_eq!(detector.state(), TurnState::Idle);

        // Speech starts
        let result = detector.process(VadState::Speech, None).unwrap();
        assert_eq!(result.state, TurnState::UserSpeaking);

        // Speech continues
        let result = detector.process(VadState::Speech, Some("Hello")).unwrap();
        assert_eq!(result.state, TurnState::UserSpeaking);
    }

    #[test]
    fn test_turn_detection_with_question() {
        let detector = HybridTurnDetector::new(TurnDetectionConfig::default());

        // Simulate speech with question
        let _ = detector.process(VadState::Speech, None);
        let result = detector.process(VadState::Speech, Some("What is the rate?")).unwrap();

        // Should detect question and lower threshold
        assert_eq!(result.semantic_class, Some(CompletenessClass::Question));
    }

    #[test]
    fn test_reset() {
        let detector = HybridTurnDetector::new(TurnDetectionConfig::default());

        let _ = detector.process(VadState::Speech, Some("Hello"));
        detector.reset();

        assert_eq!(detector.state(), TurnState::Idle);
        assert!(detector.current_transcript().is_empty());
    }
}
