//! Voice Session Handler
//!
//! Integrates WebRTC transport with STT/TTS pipeline for end-to-end voice conversations.

use std::sync::Arc;
use tokio::sync::{mpsc, broadcast, RwLock};

use voice_agent_pipeline::{
    stt::{StreamingStt, SttConfig, SttEngine},
    tts::{StreamingTts, TtsConfig, TtsEngine, TtsEvent, create_hindi_g2p},
};

use crate::{GoldLoanAgent, AgentConfig, AgentEvent, AgentError};

/// Voice session configuration
#[derive(Debug, Clone)]
pub struct VoiceSessionConfig {
    /// Agent configuration
    pub agent: AgentConfig,
    /// STT configuration
    pub stt: SttConfig,
    /// TTS configuration
    pub tts: TtsConfig,
    /// Enable barge-in
    pub barge_in_enabled: bool,
    /// Silence timeout for turn detection (ms)
    pub silence_timeout_ms: u64,
    /// Maximum turn duration (ms)
    pub max_turn_duration_ms: u64,
}

impl Default for VoiceSessionConfig {
    fn default() -> Self {
        Self {
            agent: AgentConfig::default(),
            stt: SttConfig {
                engine: SttEngine::IndicConformer,
                language: Some("hi".to_string()),
                ..Default::default()
            },
            tts: TtsConfig {
                engine: TtsEngine::Piper,
                ..Default::default()
            },
            barge_in_enabled: true,
            silence_timeout_ms: 800,
            max_turn_duration_ms: 30000,
        }
    }
}

/// Voice session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceSessionState {
    /// Session not started
    Idle,
    /// Listening for user speech
    Listening,
    /// Processing user input
    Processing,
    /// Speaking response
    Speaking,
    /// Session ended
    Ended,
}

/// Voice session events
#[derive(Debug, Clone)]
pub enum VoiceSessionEvent {
    /// Session started
    Started { session_id: String },
    /// State changed
    StateChanged { old: VoiceSessionState, new: VoiceSessionState },
    /// Partial transcript available
    PartialTranscript { text: String },
    /// Final transcript available
    FinalTranscript { text: String },
    /// Agent response being spoken
    Speaking { text: String },
    /// Audio chunk available for playback
    AudioChunk { samples: Vec<f32>, sample_rate: u32 },
    /// Barge-in detected
    BargedIn,
    /// Agent event
    Agent(AgentEvent),
    /// Error occurred
    Error(String),
    /// Session ended
    Ended { reason: String },
}

/// Voice session for a single conversation
pub struct VoiceSession {
    session_id: String,
    config: VoiceSessionConfig,
    state: Arc<RwLock<VoiceSessionState>>,
    agent: Arc<GoldLoanAgent>,
    stt: Arc<StreamingStt>,
    tts: Arc<StreamingTts>,
    event_tx: broadcast::Sender<VoiceSessionEvent>,
    #[allow(dead_code)] // Reserved for future transport integration
    audio_tx: Option<mpsc::Sender<Vec<f32>>>,
}

impl VoiceSession {
    /// Create a new voice session
    pub fn new(session_id: impl Into<String>, config: VoiceSessionConfig) -> Result<Self, AgentError> {
        let session_id = session_id.into();
        let (event_tx, _) = broadcast::channel(100);

        // Create agent
        let agent = Arc::new(GoldLoanAgent::without_llm(
            session_id.clone(),
            config.agent.clone(),
        ));

        // Create STT
        let stt = Arc::new(StreamingStt::simple(config.stt.clone()));

        // Add domain vocabulary for entity boosting
        stt.add_entities([
            "kotak", "mahindra", "muthoot", "manappuram", "iifl",
            "gold loan", "interest rate", "processing fee",
            "lakh", "rupees", "percent",
        ]);

        // Create TTS
        let tts = Arc::new(StreamingTts::simple(config.tts.clone()));

        Ok(Self {
            session_id,
            config,
            state: Arc::new(RwLock::new(VoiceSessionState::Idle)),
            agent,
            stt,
            tts,
            event_tx,
            audio_tx: None,
        })
    }

    /// Start the voice session
    pub async fn start(&self) -> Result<(), AgentError> {
        self.set_state(VoiceSessionState::Listening).await;

        let _ = self.event_tx.send(VoiceSessionEvent::Started {
            session_id: self.session_id.clone(),
        });

        // Play greeting
        let greeting = self.agent.process("").await?;
        self.speak(&greeting).await?;

        Ok(())
    }

    /// Process incoming audio from transport
    pub async fn process_audio(&self, samples: &[f32]) -> Result<(), AgentError> {
        let state = *self.state.read().await;

        match state {
            VoiceSessionState::Listening => {
                // Process through STT
                if let Some(result) = self.stt.process(samples)
                    .map_err(|e| AgentError::Pipeline(e.to_string()))?
                {
                    let _ = self.event_tx.send(VoiceSessionEvent::PartialTranscript {
                        text: result.text.clone(),
                    });
                }
            }
            VoiceSessionState::Speaking if self.config.barge_in_enabled => {
                // Check for barge-in (voice activity during TTS)
                let energy: f32 = samples.iter().map(|s| s.powi(2)).sum::<f32>() / samples.len() as f32;
                if energy > 0.01 {  // Energy threshold for barge-in
                    self.handle_barge_in().await?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Handle end of user turn (silence detected)
    pub async fn end_user_turn(&self) -> Result<(), AgentError> {
        let state = *self.state.read().await;
        if state != VoiceSessionState::Listening {
            return Ok(());
        }

        self.set_state(VoiceSessionState::Processing).await;

        // Finalize STT
        let transcript = self.stt.finalize();

        if transcript.text.is_empty() {
            // No speech detected, go back to listening
            self.set_state(VoiceSessionState::Listening).await;
            return Ok(());
        }

        let _ = self.event_tx.send(VoiceSessionEvent::FinalTranscript {
            text: transcript.text.clone(),
        });

        // Process through agent
        let response = self.agent.process(&transcript.text).await?;

        // Speak response
        self.speak(&response).await?;

        // Reset STT for next turn
        self.stt.reset();

        Ok(())
    }

    /// Speak text using TTS
    async fn speak(&self, text: &str) -> Result<(), AgentError> {
        self.set_state(VoiceSessionState::Speaking).await;

        let _ = self.event_tx.send(VoiceSessionEvent::Speaking {
            text: text.to_string(),
        });

        // Convert to phonemes for Indian language support
        let g2p = create_hindi_g2p();
        let _phonemes = g2p.convert(text)
            .map_err(|e| AgentError::Pipeline(e.to_string()))?;

        // Start TTS
        let (tts_tx, mut tts_rx) = mpsc::channel::<TtsEvent>(10);
        self.tts.start(text, tts_tx);

        // Process TTS chunks
        loop {
            match self.tts.process_next()
                .map_err(|e| AgentError::Pipeline(e.to_string()))?
            {
                Some(TtsEvent::Audio { samples, is_final, .. }) => {
                    let _ = self.event_tx.send(VoiceSessionEvent::AudioChunk {
                        samples: samples.to_vec(),
                        sample_rate: self.tts.sample_rate(),
                    });

                    if is_final {
                        break;
                    }
                }
                Some(TtsEvent::Complete) => break,
                Some(TtsEvent::BargedIn { .. }) => {
                    let _ = self.event_tx.send(VoiceSessionEvent::BargedIn);
                    break;
                }
                Some(TtsEvent::Error(e)) => {
                    return Err(AgentError::Pipeline(e));
                }
                _ => {}
            }

            // Check for external events
            if let Ok(event) = tts_rx.try_recv() {
                if matches!(event, TtsEvent::BargedIn { .. }) {
                    break;
                }
            }
        }

        self.set_state(VoiceSessionState::Listening).await;
        Ok(())
    }

    /// Handle barge-in during TTS
    async fn handle_barge_in(&self) -> Result<(), AgentError> {
        self.tts.barge_in();

        let _ = self.event_tx.send(VoiceSessionEvent::BargedIn);

        // Reset and start listening
        self.tts.reset();
        self.set_state(VoiceSessionState::Listening).await;

        Ok(())
    }

    /// End the voice session
    pub async fn end(&self, reason: impl Into<String>) {
        self.set_state(VoiceSessionState::Ended).await;

        let _ = self.event_tx.send(VoiceSessionEvent::Ended {
            reason: reason.into(),
        });
    }

    /// Subscribe to session events
    pub fn subscribe(&self) -> broadcast::Receiver<VoiceSessionEvent> {
        self.event_tx.subscribe()
    }

    /// Get current state
    pub async fn state(&self) -> VoiceSessionState {
        *self.state.read().await
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get agent reference
    pub fn agent(&self) -> &GoldLoanAgent {
        &self.agent
    }

    /// Set state and emit event
    async fn set_state(&self, new_state: VoiceSessionState) {
        let old_state = {
            let mut state = self.state.write().await;
            let old = *state;
            *state = new_state;
            old
        };

        if old_state != new_state {
            let _ = self.event_tx.send(VoiceSessionEvent::StateChanged {
                old: old_state,
                new: new_state,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_voice_session_creation() {
        let session = VoiceSession::new("test-session", VoiceSessionConfig::default());
        assert!(session.is_ok());

        let session = session.unwrap();
        assert_eq!(session.session_id(), "test-session");
    }

    #[tokio::test]
    async fn test_voice_session_state() {
        let session = VoiceSession::new("test", VoiceSessionConfig::default()).unwrap();

        assert_eq!(session.state().await, VoiceSessionState::Idle);
    }

    #[tokio::test]
    async fn test_voice_session_start() {
        let session = VoiceSession::new("test", VoiceSessionConfig::default()).unwrap();

        let result = session.start().await;
        assert!(result.is_ok());

        assert_eq!(session.state().await, VoiceSessionState::Listening);
    }
}
