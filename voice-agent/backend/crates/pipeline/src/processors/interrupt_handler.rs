//! Interrupt handler for barge-in processing
//!
//! Handles user interruptions during TTS playback with configurable modes:
//! - Immediate: Stop TTS immediately
//! - SentenceBoundary: Finish current sentence before stopping
//! - WordBoundary: Finish current word before stopping

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use voice_agent_core::{Frame, FrameProcessor, ProcessorContext, Result};

/// Interrupt mode determines how quickly TTS stops on barge-in
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InterruptMode {
    /// Stop immediately on barge-in (lowest latency)
    #[default]
    Immediate,
    /// Finish current sentence before stopping (better prosody)
    SentenceBoundary,
    /// Finish current word before stopping (balance)
    WordBoundary,
    /// Don't interrupt (ignore barge-in)
    Disabled,
}

/// Interrupt handler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptHandlerConfig {
    /// Interrupt mode
    pub mode: InterruptMode,
    /// Minimum speech duration (ms) to trigger interrupt
    pub min_speech_duration_ms: u32,
    /// Minimum energy level (dB) for interrupt detection
    pub min_energy_db: f32,
    /// Grace period after TTS starts (ms) - ignore interrupts during this
    pub grace_period_ms: u32,
    /// Fade out duration (ms) for smooth audio transition
    pub fade_out_ms: u32,
}

impl Default for InterruptHandlerConfig {
    fn default() -> Self {
        Self {
            mode: InterruptMode::Immediate,
            min_speech_duration_ms: 150,
            min_energy_db: -40.0,
            grace_period_ms: 200,
            fade_out_ms: 50,
        }
    }
}

/// Handler state for tracking interruption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HandlerState {
    /// Idle, not tracking
    Idle,
    /// TTS is playing
    Speaking,
    /// Interrupt detected, waiting for boundary
    PendingInterrupt,
    /// Interrupted, blocking output
    Interrupted,
}

/// Interrupt handler processor
pub struct InterruptHandler {
    config: InterruptHandlerConfig,
    /// Current state
    state: Mutex<HandlerState>,
    /// Current sentence index (for sentence boundary mode)
    current_sentence: Mutex<usize>,
    /// Target sentence to stop at
    target_sentence: Mutex<Option<usize>>,
    /// Accumulated speech duration for interrupt detection
    speech_duration_ms: Mutex<u32>,
    /// TTS start timestamp (frame index for grace period)
    tts_start_frame: Mutex<u64>,
    /// Current frame counter
    frame_counter: Mutex<u64>,
}

impl InterruptHandler {
    /// Create a new interrupt handler
    pub fn new(config: InterruptHandlerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(HandlerState::Idle),
            current_sentence: Mutex::new(0),
            target_sentence: Mutex::new(None),
            speech_duration_ms: Mutex::new(0),
            tts_start_frame: Mutex::new(0),
            frame_counter: Mutex::new(0),
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(InterruptHandlerConfig::default())
    }

    /// Handle a barge-in event
    fn handle_barge_in(&self, audio_position_ms: u64) -> Vec<Frame> {
        let state = *self.state.lock();

        if state != HandlerState::Speaking {
            return vec![];
        }

        // Check grace period
        let frame = *self.frame_counter.lock();
        let start = *self.tts_start_frame.lock();
        let elapsed_frames = frame.saturating_sub(start);

        // Approximate: 20ms per frame (50 fps)
        let elapsed_ms = elapsed_frames * 20;
        if elapsed_ms < self.config.grace_period_ms as u64 {
            // Still in grace period, ignore
            return vec![];
        }

        match self.config.mode {
            InterruptMode::Disabled => vec![],

            InterruptMode::Immediate => {
                *self.state.lock() = HandlerState::Interrupted;
                vec![Frame::BargeIn {
                    audio_position_ms,
                    transcript: None,
                }]
            }

            InterruptMode::SentenceBoundary => {
                *self.state.lock() = HandlerState::PendingInterrupt;
                let current = *self.current_sentence.lock();
                *self.target_sentence.lock() = Some(current);
                // Don't emit barge-in yet, wait for sentence end
                vec![]
            }

            InterruptMode::WordBoundary => {
                // For word boundary, we set pending and let TTS finish current word
                *self.state.lock() = HandlerState::PendingInterrupt;
                // TTS layer will handle word boundary
                vec![Frame::BargeIn {
                    audio_position_ms,
                    transcript: None,
                }]
            }
        }
    }

    /// Check if we should emit frames or block them
    fn should_pass(&self, frame: &Frame) -> bool {
        let state = *self.state.lock();

        match state {
            HandlerState::Idle | HandlerState::Speaking => true,

            HandlerState::PendingInterrupt => {
                // In pending, we block audio output but allow other frames
                !matches!(frame, Frame::AudioOutput(_))
            }

            HandlerState::Interrupted => {
                // When interrupted, block TTS audio
                !matches!(frame, Frame::AudioOutput(_) | Frame::Sentence { .. })
            }
        }
    }

    /// Process a sentence frame
    fn process_sentence(&self, index: usize) -> bool {
        *self.current_sentence.lock() = index;

        // Check if this is the target sentence for pending interrupt
        if *self.state.lock() == HandlerState::PendingInterrupt {
            if let Some(target) = *self.target_sentence.lock() {
                if index > target {
                    // Past target sentence, interrupt now
                    *self.state.lock() = HandlerState::Interrupted;
                    return false; // Block this sentence
                }
            }
        }

        true
    }

    /// Handle voice activity for interrupt detection
    fn process_voice_activity(&self, is_speech: bool, energy_db: f32) {
        if *self.state.lock() != HandlerState::Speaking {
            return;
        }

        if is_speech && energy_db >= self.config.min_energy_db {
            let mut duration = self.speech_duration_ms.lock();
            *duration += 20; // Assume 20ms frame
        } else {
            *self.speech_duration_ms.lock() = 0;
        }
    }

    /// Start speaking state
    fn start_speaking(&self) {
        *self.state.lock() = HandlerState::Speaking;
        *self.tts_start_frame.lock() = *self.frame_counter.lock();
        *self.speech_duration_ms.lock() = 0;
    }

    /// Reset to idle
    pub fn reset(&self) {
        *self.state.lock() = HandlerState::Idle;
        *self.current_sentence.lock() = 0;
        *self.target_sentence.lock() = None;
        *self.speech_duration_ms.lock() = 0;
    }

    /// Get current mode
    pub fn mode(&self) -> InterruptMode {
        self.config.mode
    }

    /// Set interrupt mode
    pub fn set_mode(&self, mode: InterruptMode) {
        // Note: We can't mutate config directly, but we could use interior mutability
        // For now, create a new handler if mode needs to change
        tracing::warn!("InterruptMode cannot be changed at runtime, create new handler");
        let _ = mode;
    }

    /// Check if currently interrupted
    pub fn is_interrupted(&self) -> bool {
        *self.state.lock() == HandlerState::Interrupted
    }
}

#[async_trait]
impl FrameProcessor for InterruptHandler {
    async fn process(
        &self,
        frame: Frame,
        _context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        // Increment frame counter
        *self.frame_counter.lock() += 1;

        match &frame {
            // Handle barge-in event
            Frame::BargeIn { audio_position_ms, .. } => {
                let additional = self.handle_barge_in(*audio_position_ms);
                if additional.is_empty() && self.config.mode == InterruptMode::Disabled {
                    // Pass through the original barge-in if disabled
                    return Ok(vec![frame]);
                }
                if additional.is_empty() {
                    // Consumed for pending interrupt
                    return Ok(vec![]);
                }
                return Ok(additional);
            }

            // Track sentence progress
            Frame::Sentence { index, .. } => {
                if !self.process_sentence(*index) {
                    // Block sentence after interrupt
                    return Ok(vec![]);
                }
            }

            // Start speaking when audio output begins
            Frame::AudioOutput(_) => {
                if *self.state.lock() == HandlerState::Idle {
                    self.start_speaking();
                }
            }

            // Voice activity for interrupt detection
            Frame::VoiceStart => {
                self.process_voice_activity(true, 0.0);
            }

            Frame::VoiceEnd { .. } => {
                self.process_voice_activity(false, -60.0);
            }

            // Reset on end of stream or control
            Frame::EndOfStream => {
                self.reset();
            }

            Frame::Control(voice_agent_core::ControlFrame::Reset) => {
                self.reset();
            }

            _ => {}
        }

        // Filter based on current state
        if self.should_pass(&frame) {
            Ok(vec![frame])
        } else {
            Ok(vec![])
        }
    }

    fn name(&self) -> &'static str {
        "interrupt_handler"
    }

    fn description(&self) -> &str {
        "Handles barge-in with configurable interrupt modes"
    }

    async fn on_start(&self, _context: &mut ProcessorContext) -> Result<()> {
        self.reset();
        Ok(())
    }

    async fn on_stop(&self, _context: &mut ProcessorContext) -> Result<()> {
        self.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_immediate_interrupt() {
        let handler = InterruptHandler::new(InterruptHandlerConfig {
            mode: InterruptMode::Immediate,
            grace_period_ms: 0,
            ..Default::default()
        });

        let mut ctx = ProcessorContext::default();

        // Start speaking
        handler.start_speaking();

        // Trigger barge-in
        let frames = handler
            .process(
                Frame::BargeIn {
                    audio_position_ms: 1000,
                    transcript: None,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should emit barge-in immediately
        assert!(frames.iter().any(|f| matches!(f, Frame::BargeIn { .. })));
        assert!(handler.is_interrupted());
    }

    #[tokio::test]
    async fn test_sentence_boundary_interrupt() {
        let handler = InterruptHandler::new(InterruptHandlerConfig {
            mode: InterruptMode::SentenceBoundary,
            grace_period_ms: 0,
            ..Default::default()
        });

        let mut ctx = ProcessorContext::default();

        // Start speaking
        handler.start_speaking();

        // Set current sentence
        handler
            .process(
                Frame::Sentence {
                    text: "First sentence.".into(),
                    language: voice_agent_core::Language::English,
                    index: 0,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Trigger barge-in
        let frames = handler
            .process(
                Frame::BargeIn {
                    audio_position_ms: 1000,
                    transcript: None,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should NOT emit barge-in yet (pending)
        assert!(frames.is_empty());
        assert!(!handler.is_interrupted());

        // Process next sentence - should trigger interrupt
        let frames = handler
            .process(
                Frame::Sentence {
                    text: "Second sentence.".into(),
                    language: voice_agent_core::Language::English,
                    index: 1,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Now interrupted (sentence blocked)
        assert!(frames.is_empty());
        assert!(handler.is_interrupted());
    }

    #[tokio::test]
    async fn test_disabled_mode() {
        let handler = InterruptHandler::new(InterruptHandlerConfig {
            mode: InterruptMode::Disabled,
            ..Default::default()
        });

        let mut ctx = ProcessorContext::default();

        // Start speaking
        handler.start_speaking();

        // Barge-in should pass through unchanged
        let frames = handler
            .process(
                Frame::BargeIn {
                    audio_position_ms: 1000,
                    transcript: None,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        assert_eq!(frames.len(), 1);
        assert!(!handler.is_interrupted());
    }

    #[tokio::test]
    async fn test_grace_period() {
        let handler = InterruptHandler::new(InterruptHandlerConfig {
            mode: InterruptMode::Immediate,
            grace_period_ms: 500, // Long grace period
            ..Default::default()
        });

        let mut ctx = ProcessorContext::default();

        // Start speaking
        handler.start_speaking();

        // Barge-in during grace period should be ignored
        let frames = handler
            .process(
                Frame::BargeIn {
                    audio_position_ms: 100,
                    transcript: None,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Grace period blocks interrupt
        assert!(frames.is_empty());
        assert!(!handler.is_interrupted());
    }

    #[tokio::test]
    async fn test_reset() {
        let handler = InterruptHandler::new(InterruptHandlerConfig {
            mode: InterruptMode::Immediate,
            grace_period_ms: 0,
            ..Default::default()
        });

        let mut ctx = ProcessorContext::default();

        // Start and interrupt
        handler.start_speaking();
        handler
            .process(
                Frame::BargeIn {
                    audio_position_ms: 1000,
                    transcript: None,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        assert!(handler.is_interrupted());

        // Reset
        handler.reset();
        assert!(!handler.is_interrupted());
    }

    #[tokio::test]
    async fn test_audio_blocked_when_interrupted() {
        let handler = InterruptHandler::new(InterruptHandlerConfig {
            mode: InterruptMode::Immediate,
            grace_period_ms: 0,
            ..Default::default()
        });

        let mut ctx = ProcessorContext::default();

        // Interrupt
        handler.start_speaking();
        handler
            .process(
                Frame::BargeIn {
                    audio_position_ms: 1000,
                    transcript: None,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Audio should be blocked
        let audio_frame = voice_agent_core::AudioFrame::new(
            vec![0.0; 160],
            voice_agent_core::SampleRate::Hz16000,
            voice_agent_core::Channels::Mono,
            0,
        );

        let frames = handler
            .process(Frame::AudioOutput(audio_frame), &mut ctx)
            .await
            .unwrap();

        assert!(frames.is_empty());
    }
}
