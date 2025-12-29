//! Voice Pipeline Orchestrator
//!
//! Coordinates VAD, STT, TTS, and turn detection for real-time conversation.

use std::sync::Arc;
use std::time::Instant;
use parking_lot::Mutex;
use tokio::sync::{mpsc, broadcast};

use crate::vad::{VoiceActivityDetector, VadState, VadConfig};
use crate::turn_detection::{HybridTurnDetector, TurnDetectionConfig, TurnDetectionResult};
use crate::stt::{StreamingStt, SttConfig};
use crate::tts::{StreamingTts, TtsConfig, TtsEvent};
use crate::PipelineError;
use voice_agent_core::{AudioFrame, TranscriptResult};

/// Pipeline events
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// VAD state changed
    VadStateChanged(VadState),
    /// Turn state changed
    TurnStateChanged(TurnDetectionResult),
    /// Partial transcript available
    PartialTranscript(TranscriptResult),
    /// Final transcript available
    FinalTranscript(TranscriptResult),
    /// TTS audio chunk ready
    TtsAudio {
        samples: Arc<[f32]>,
        text: String,
        is_final: bool,
    },
    /// Barge-in detected
    BargeIn {
        /// Word index where user interrupted
        at_word: usize,
    },
    /// Error occurred
    Error(String),
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// VAD configuration
    pub vad: VadConfig,
    /// Turn detection configuration
    pub turn_detection: TurnDetectionConfig,
    /// STT configuration
    pub stt: SttConfig,
    /// TTS configuration
    pub tts: TtsConfig,
    /// Barge-in settings
    pub barge_in: BargeInConfig,
    /// Latency budget in milliseconds
    pub latency_budget_ms: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            turn_detection: TurnDetectionConfig::default(),
            stt: SttConfig::default(),
            tts: TtsConfig::default(),
            barge_in: BargeInConfig::default(),
            latency_budget_ms: 500,
        }
    }
}

/// Barge-in configuration
#[derive(Debug, Clone)]
pub struct BargeInConfig {
    /// Enable barge-in detection
    pub enabled: bool,
    /// Minimum speech duration to trigger barge-in (ms)
    pub min_speech_ms: u32,
    /// Minimum energy level for barge-in (dB)
    pub min_energy_db: f32,
    /// Action on barge-in
    pub action: BargeInAction,
}

impl Default for BargeInConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_speech_ms: 150,
            min_energy_db: -40.0,
            action: BargeInAction::StopAndListen,
        }
    }
}

/// Barge-in action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BargeInAction {
    /// Stop TTS and switch to listening
    StopAndListen,
    /// Fade out TTS audio
    FadeOut,
    /// Continue TTS (ignore barge-in)
    Ignore,
}

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    /// Idle, waiting for audio
    Idle,
    /// Listening to user
    Listening,
    /// Processing turn
    Processing,
    /// Speaking response
    Speaking,
    /// Paused
    Paused,
}

/// Voice Pipeline orchestrator
pub struct VoicePipeline {
    config: PipelineConfig,
    vad: Arc<VoiceActivityDetector>,
    turn_detector: Arc<HybridTurnDetector>,
    stt: Arc<Mutex<StreamingStt>>,
    tts: Arc<StreamingTts>,
    state: Mutex<PipelineState>,
    /// Event broadcaster
    event_tx: broadcast::Sender<PipelineEvent>,
    /// Barge-in speech accumulator
    barge_in_speech_ms: Mutex<u32>,
    /// Last audio timestamp
    last_audio_time: Mutex<Instant>,
}

impl VoicePipeline {
    /// Create a new voice pipeline with simple components (for testing)
    pub fn simple(config: PipelineConfig) -> Result<Self, PipelineError> {
        let vad = Arc::new(VoiceActivityDetector::simple(config.vad.clone())?);
        let turn_detector = Arc::new(HybridTurnDetector::new(config.turn_detection.clone()));
        let stt = Arc::new(Mutex::new(StreamingStt::simple(config.stt.clone())));
        let tts = Arc::new(StreamingTts::simple(config.tts.clone()));

        let (event_tx, _) = broadcast::channel(100);

        Ok(Self {
            config,
            vad,
            turn_detector,
            stt,
            tts,
            state: Mutex::new(PipelineState::Idle),
            event_tx,
            barge_in_speech_ms: Mutex::new(0),
            last_audio_time: Mutex::new(Instant::now()),
        })
    }

    /// Subscribe to pipeline events
    pub fn subscribe(&self) -> broadcast::Receiver<PipelineEvent> {
        self.event_tx.subscribe()
    }

    /// Process an audio frame
    pub async fn process_audio(&self, mut frame: AudioFrame) -> Result<(), PipelineError> {
        let now = Instant::now();
        *self.last_audio_time.lock() = now;

        // 1. Run VAD
        let (vad_state, _vad_prob, _vad_result) = self.vad.process_frame(&mut frame)?;

        // Emit VAD event on state change
        let _ = self.event_tx.send(PipelineEvent::VadStateChanged(vad_state));

        // 2. Check for barge-in if speaking
        if *self.state.lock() == PipelineState::Speaking
            && self.check_barge_in(&frame, vad_state).await? {
                return Ok(());
            }

        // 3. Process based on state
        match *self.state.lock() {
            PipelineState::Idle => {
                if vad_state == VadState::Speech || vad_state == VadState::SpeechStart {
                    *self.state.lock() = PipelineState::Listening;
                    self.stt.lock().reset();
                }
            }

            PipelineState::Listening => {
                // Feed audio to STT
                // Note: True parallelization with spawn_blocking isn't possible because
                // ort::Session contains raw pointers that aren't Send. The ONNX runtime
                // handles threading internally, so this is acceptable for now.
                if let Some(partial) = self.stt.lock().process(&frame.samples)? {
                    let _ = self.event_tx.send(PipelineEvent::PartialTranscript(partial.clone()));

                    // Update turn detector with transcript
                    let turn_result = self.turn_detector.process(
                        vad_state,
                        Some(&partial.text),
                    )?;

                    let _ = self.event_tx.send(PipelineEvent::TurnStateChanged(turn_result.clone()));

                    // Check for turn completion
                    if turn_result.is_turn_complete {
                        let final_transcript = self.stt.lock().finalize();
                        let _ = self.event_tx.send(PipelineEvent::FinalTranscript(final_transcript));
                        *self.state.lock() = PipelineState::Processing;
                    }
                } else {
                    // No transcript yet, just update turn detector with VAD
                    let turn_result = self.turn_detector.process(vad_state, None)?;
                    let _ = self.event_tx.send(PipelineEvent::TurnStateChanged(turn_result));
                }
            }

            PipelineState::Processing => {
                // Waiting for agent response
                // Audio is still monitored for barge-in
            }

            PipelineState::Speaking => {
                // Handled above in barge-in check
            }

            PipelineState::Paused => {
                // Do nothing
            }
        }

        Ok(())
    }

    /// Check for barge-in during TTS
    async fn check_barge_in(&self, frame: &AudioFrame, vad_state: VadState) -> Result<bool, PipelineError> {
        if !self.config.barge_in.enabled {
            return Ok(false);
        }

        if self.config.barge_in.action == BargeInAction::Ignore {
            return Ok(false);
        }

        // Check if user is speaking
        let is_speech = vad_state == VadState::Speech || vad_state == VadState::SpeechStart;
        let sufficient_energy = frame.energy_db >= self.config.barge_in.min_energy_db;

        if is_speech && sufficient_energy {
            let mut speech_ms = self.barge_in_speech_ms.lock();
            *speech_ms += self.config.vad.frame_ms;

            if *speech_ms >= self.config.barge_in.min_speech_ms {
                // Barge-in triggered!
                let word_index = self.tts.current_word_index();

                // Stop TTS
                self.tts.barge_in();

                // Emit event
                let _ = self.event_tx.send(PipelineEvent::BargeIn { at_word: word_index });

                // Switch to listening
                *self.state.lock() = PipelineState::Listening;
                *speech_ms = 0;

                // Reset turn detector
                self.turn_detector.reset();
                self.stt.lock().reset();

                return Ok(true);
            }
        } else {
            *self.barge_in_speech_ms.lock() = 0;
        }

        Ok(false)
    }

    /// Start speaking a response
    pub async fn speak(&self, text: &str) -> Result<(), PipelineError> {
        // Set state
        *self.state.lock() = PipelineState::Speaking;
        self.turn_detector.set_agent_speaking();
        *self.barge_in_speech_ms.lock() = 0;

        // Create channel for TTS events
        let (tx, mut rx) = mpsc::channel::<TtsEvent>(100);

        // Start TTS
        self.tts.start(text, tx);

        // Process TTS events
        while let Some(event) = rx.recv().await {
            match event {
                TtsEvent::Audio { samples, text, is_final, .. } => {
                    let _ = self.event_tx.send(PipelineEvent::TtsAudio {
                        samples,
                        text,
                        is_final,
                    });
                }
                TtsEvent::Complete => {
                    *self.state.lock() = PipelineState::Idle;
                    self.turn_detector.reset();
                    break;
                }
                TtsEvent::BargedIn { word_index } => {
                    let _ = self.event_tx.send(PipelineEvent::BargeIn { at_word: word_index });
                    break;
                }
                TtsEvent::Error(e) => {
                    let _ = self.event_tx.send(PipelineEvent::Error(e));
                    *self.state.lock() = PipelineState::Idle;
                    break;
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Get current pipeline state
    pub fn state(&self) -> PipelineState {
        *self.state.lock()
    }

    /// Pause pipeline
    pub fn pause(&self) {
        *self.state.lock() = PipelineState::Paused;
    }

    /// Resume pipeline
    pub fn resume(&self) {
        let mut state = self.state.lock();
        if *state == PipelineState::Paused {
            *state = PipelineState::Idle;
        }
    }

    /// Reset pipeline
    pub fn reset(&self) {
        *self.state.lock() = PipelineState::Idle;
        self.vad.reset();
        self.turn_detector.reset();
        self.stt.lock().reset();
        self.tts.reset();
        *self.barge_in_speech_ms.lock() = 0;
    }

    /// Get current transcript
    pub fn current_transcript(&self) -> String {
        self.turn_detector.current_transcript()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voice_agent_core::{Channels, SampleRate};

    #[allow(dead_code)]
    fn create_test_frame(samples: Vec<f32>) -> AudioFrame {
        AudioFrame::new(
            samples,
            SampleRate::Hz16000,
            Channels::Mono,
            0,
        )
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let pipeline = VoicePipeline::simple(PipelineConfig::default()).unwrap();
        assert_eq!(pipeline.state(), PipelineState::Idle);
    }

    #[tokio::test]
    async fn test_pipeline_state_transitions() {
        let pipeline = VoicePipeline::simple(PipelineConfig::default()).unwrap();

        pipeline.pause();
        assert_eq!(pipeline.state(), PipelineState::Paused);

        pipeline.resume();
        assert_eq!(pipeline.state(), PipelineState::Idle);
    }

    #[tokio::test]
    async fn test_pipeline_reset() {
        let pipeline = VoicePipeline::simple(PipelineConfig::default()).unwrap();
        pipeline.reset();
        assert_eq!(pipeline.state(), PipelineState::Idle);
    }
}
