//! Voice Session Handler
//!
//! Integrates WebRTC transport with STT/TTS pipeline for end-to-end voice conversations.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  Transport  │────▶│     STT     │────▶│    Agent    │────▶│     TTS     │
//! │  (WebRTC)   │     │ (streaming) │     │ (reasoning) │     │ (streaming) │
//! └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
//!       ▲                                                            │
//!       │                                                            │
//!       └────────────────── Audio Playback ◀─────────────────────────┘
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, broadcast, RwLock};
use tokio::time::interval;

use voice_agent_pipeline::{
    stt::{StreamingStt, SttConfig, SttEngine, IndicConformerConfig},
    tts::{StreamingTts, TtsConfig, TtsEngine, TtsEvent, create_hindi_g2p},
    vad::{SileroVad, SileroConfig, VadState, VadResult},
};
use voice_agent_core::AudioFrame;
use voice_agent_transport::{
    TransportSession, SessionConfig, TransportEvent,
};

use crate::{GoldLoanAgent, AgentConfig, AgentEvent, AgentError};

/// Voice session configuration
#[derive(Debug, Clone)]
pub struct VoiceSessionConfig {
    /// Agent configuration
    pub agent: AgentConfig,
    /// STT configuration
    pub stt: SttConfig,
    /// IndicConformer configuration (if using IndicConformer engine)
    pub indicconformer: Option<IndicConformerConfig>,
    /// TTS configuration
    pub tts: TtsConfig,
    /// Transport configuration
    pub transport: SessionConfig,
    /// VAD configuration (Silero)
    pub vad: SileroConfig,
    /// Enable barge-in
    pub barge_in_enabled: bool,
    /// Silence timeout for turn detection (ms)
    pub silence_timeout_ms: u64,
    /// Maximum turn duration (ms)
    pub max_turn_duration_ms: u64,
    /// Audio processing interval (ms) - how often to poll for audio
    pub audio_poll_interval_ms: u64,
    /// Energy threshold for voice activity detection (0.0 - 1.0)
    pub vad_energy_threshold: f32,
    /// Use Silero VAD instead of energy-based detection
    pub use_silero_vad: bool,
    /// Path to Silero VAD model
    pub vad_model_path: Option<std::path::PathBuf>,
    /// Path to IndicConformer model directory
    pub stt_model_path: Option<std::path::PathBuf>,
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
            indicconformer: Some(IndicConformerConfig::default()),
            tts: TtsConfig {
                engine: TtsEngine::Piper,
                ..Default::default()
            },
            transport: SessionConfig::default(),
            vad: SileroConfig::default(),
            barge_in_enabled: true,
            silence_timeout_ms: 800,
            max_turn_duration_ms: 30000,
            audio_poll_interval_ms: 20, // 20ms = 50Hz polling (matches Opus frame size)
            vad_energy_threshold: 0.01,
            use_silero_vad: false, // Default to energy-based (simpler, no model needed)
            vad_model_path: None,
            stt_model_path: None,
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
    /// Silero VAD (optional, if enabled)
    vad: Option<Arc<parking_lot::Mutex<SileroVad>>>,
    event_tx: broadcast::Sender<VoiceSessionEvent>,
    /// Transport session for WebRTC/WebSocket communication
    transport: Arc<RwLock<Option<TransportSession>>>,
    /// Channel to send audio to transport
    audio_out_tx: mpsc::Sender<Vec<f32>>,
    audio_out_rx: Arc<RwLock<Option<mpsc::Receiver<Vec<f32>>>>>,
    /// Transport event receiver
    transport_event_tx: mpsc::Sender<TransportEvent>,
    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
    /// Last voice activity timestamp for silence detection
    last_voice_activity: Arc<RwLock<Option<Instant>>>,
    /// VAD state for speech detection
    vad_state: Arc<RwLock<VadState>>,
}

impl VoiceSession {
    /// Create a new voice session
    pub fn new(session_id: impl Into<String>, config: VoiceSessionConfig) -> Result<Self, AgentError> {
        let session_id = session_id.into();
        let (event_tx, _) = broadcast::channel(100);
        let (shutdown_tx, _) = broadcast::channel(1);
        let (audio_out_tx, audio_out_rx) = mpsc::channel(100);
        let (transport_event_tx, _transport_event_rx) = mpsc::channel(100);

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

        // Create VAD if enabled
        let vad = if config.use_silero_vad {
            if let Some(ref model_path) = config.vad_model_path {
                match SileroVad::new(model_path, config.vad.clone()) {
                    Ok(v) => Some(Arc::new(parking_lot::Mutex::new(v))),
                    Err(e) => {
                        tracing::warn!("Failed to load Silero VAD: {}, falling back to energy-based", e);
                        None
                    }
                }
            } else {
                // Create energy-based fallback VAD
                match SileroVad::simple(config.vad.clone()) {
                    Ok(v) => Some(Arc::new(parking_lot::Mutex::new(v))),
                    Err(_) => None,
                }
            }
        } else {
            None
        };

        Ok(Self {
            session_id,
            config,
            state: Arc::new(RwLock::new(VoiceSessionState::Idle)),
            agent,
            stt,
            tts,
            vad,
            event_tx,
            transport: Arc::new(RwLock::new(None)),
            audio_out_tx,
            audio_out_rx: Arc::new(RwLock::new(Some(audio_out_rx))),
            transport_event_tx,
            shutdown_tx,
            last_voice_activity: Arc::new(RwLock::new(None)),
            vad_state: Arc::new(RwLock::new(VadState::Silence)),
        })
    }

    /// Attach a transport session for WebRTC/WebSocket communication
    pub async fn attach_transport(&self, mut transport: TransportSession) {
        // Set up event callback for transport events
        transport.set_event_callback(self.transport_event_tx.clone());
        *self.transport.write().await = Some(transport);
    }

    /// Connect transport with SDP offer and return answer
    pub async fn connect_transport(&self, offer: &str) -> Result<String, AgentError> {
        let mut transport_guard = self.transport.write().await;
        let transport = transport_guard.as_mut()
            .ok_or_else(|| AgentError::Pipeline("No transport attached".to_string()))?;

        transport.connect(offer).await
            .map_err(|e| AgentError::Pipeline(format!("Transport connection failed: {}", e)))
    }

    /// Start the voice session
    ///
    /// This starts the main processing loop that:
    /// 1. Receives audio from transport
    /// 2. Processes through STT
    /// 3. Detects end of turn (silence)
    /// 4. Gets agent response
    /// 5. Synthesizes with TTS
    /// 6. Sends audio back through transport
    pub async fn start(&self) -> Result<(), AgentError> {
        self.set_state(VoiceSessionState::Listening).await;

        let _ = self.event_tx.send(VoiceSessionEvent::Started {
            session_id: self.session_id.clone(),
        });

        // Spawn the transport event handler
        self.spawn_transport_event_handler();

        // Spawn the audio output handler (TTS → Transport)
        self.spawn_audio_output_handler();

        // Play greeting
        let greeting = self.agent.process("").await?;
        self.speak(&greeting).await?;

        Ok(())
    }

    /// Spawn task to handle transport events (incoming audio)
    fn spawn_transport_event_handler(&self) {
        let state = Arc::clone(&self.state);
        let stt = Arc::clone(&self.stt);
        let event_tx = self.event_tx.clone();
        let config = self.config.clone();
        let last_voice_activity = Arc::clone(&self.last_voice_activity);
        let _transport_event_tx = self.transport_event_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        // Create a receiver for transport events
        let (internal_tx, mut internal_rx) = mpsc::channel::<TransportEvent>(100);

        // Spawn a task that forwards transport events
        let transport = Arc::clone(&self.transport);
        tokio::spawn(async move {
            // Set up the transport callback
            if let Some(ref mut t) = *transport.write().await {
                t.set_event_callback(internal_tx);
            }
        });

        // Session reference for processing
        let session_id = self.session_id.clone();
        let agent = Arc::clone(&self.agent);
        let tts = Arc::clone(&self.tts);
        let audio_out_tx = self.audio_out_tx.clone();

        tokio::spawn(async move {
            let mut silence_timer = interval(Duration::from_millis(100));

            loop {
                tokio::select! {
                    // Handle shutdown
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Transport event handler shutting down for session {}", session_id);
                        break;
                    }

                    // Handle incoming transport events
                    Some(event) = internal_rx.recv() => {
                        match event {
                            TransportEvent::AudioReceived { samples, timestamp_ms: _ } => {
                                let current_state = *state.read().await;

                                match current_state {
                                    VoiceSessionState::Listening => {
                                        // Check for voice activity
                                        let energy = calculate_energy(&samples);

                                        if energy > config.vad_energy_threshold {
                                            *last_voice_activity.write().await = Some(Instant::now());

                                            // Process through STT
                                            if let Some(result) = stt.process(&samples)
                                                .map_err(|e| tracing::error!("STT error: {}", e))
                                                .ok()
                                                .flatten()
                                            {
                                                let _ = event_tx.send(VoiceSessionEvent::PartialTranscript {
                                                    text: result.text,
                                                });
                                            }
                                        }
                                    }

                                    VoiceSessionState::Speaking => {
                                        // Check for barge-in
                                        if config.barge_in_enabled {
                                            let energy = calculate_energy(&samples);
                                            if energy > config.vad_energy_threshold * 2.0 {
                                                // Barge-in detected
                                                let _ = event_tx.send(VoiceSessionEvent::BargedIn);
                                                tts.barge_in();
                                                *state.write().await = VoiceSessionState::Listening;
                                            }
                                        }
                                    }

                                    _ => {}
                                }
                            }

                            TransportEvent::Disconnected { reason } => {
                                let _ = event_tx.send(VoiceSessionEvent::Ended { reason });
                                break;
                            }

                            TransportEvent::Error { message } => {
                                let _ = event_tx.send(VoiceSessionEvent::Error(message));
                            }

                            _ => {}
                        }
                    }

                    // Check for silence timeout (end of user turn)
                    _ = silence_timer.tick() => {
                        let current_state = *state.read().await;
                        if current_state != VoiceSessionState::Listening {
                            continue;
                        }

                        let should_end_turn = {
                            let last_activity = last_voice_activity.read().await;
                            if let Some(last) = *last_activity {
                                last.elapsed() > Duration::from_millis(config.silence_timeout_ms)
                            } else {
                                false
                            }
                        };

                        if should_end_turn {
                            // End user turn and process
                            *state.write().await = VoiceSessionState::Processing;

                            let transcript = stt.finalize();

                            if !transcript.text.is_empty() {
                                let _ = event_tx.send(VoiceSessionEvent::FinalTranscript {
                                    text: transcript.text.clone(),
                                });

                                // Process through agent
                                if let Ok(response) = agent.process(&transcript.text).await {
                                    let _ = event_tx.send(VoiceSessionEvent::Speaking {
                                        text: response.clone(),
                                    });

                                    // Synthesize and send audio
                                    *state.write().await = VoiceSessionState::Speaking;

                                    let g2p = create_hindi_g2p();
                                    if let Ok(_phonemes) = g2p.convert(&response) {
                                        let (tts_tx, mut tts_rx) = mpsc::channel::<TtsEvent>(10);
                                        tts.start(&response, tts_tx);

                                        // Process TTS chunks
                                        while let Some(tts_event) = tts_rx.recv().await {
                                            match tts_event {
                                                TtsEvent::Audio { samples, is_final, .. } => {
                                                    let _ = audio_out_tx.send(samples.to_vec()).await;
                                                    if is_final {
                                                        break;
                                                    }
                                                }
                                                TtsEvent::Complete => break,
                                                TtsEvent::BargedIn { .. } => break,
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }

                            // Reset for next turn
                            stt.reset();
                            *last_voice_activity.write().await = None;
                            *state.write().await = VoiceSessionState::Listening;
                        }
                    }
                }
            }
        });
    }

    /// Spawn task to handle audio output (send TTS audio to transport)
    fn spawn_audio_output_handler(&self) {
        let transport = Arc::clone(&self.transport);
        let audio_out_rx = Arc::clone(&self.audio_out_rx);
        let event_tx = self.event_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let session_id = self.session_id.clone();

        tokio::spawn(async move {
            // Take ownership of the receiver
            let mut rx = match audio_out_rx.write().await.take() {
                Some(rx) => rx,
                None => {
                    tracing::error!("Audio output receiver already taken");
                    return;
                }
            };

            let mut timestamp_ms: u64 = 0;

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        tracing::info!("Audio output handler shutting down for session {}", session_id);
                        break;
                    }

                    Some(samples) = rx.recv() => {
                        // Send to event subscribers for local playback
                        let _ = event_tx.send(VoiceSessionEvent::AudioChunk {
                            samples: samples.clone(),
                            sample_rate: 16000,
                        });

                        // Send through transport if connected (using the new send_audio method)
                        let transport_guard = transport.read().await;
                        if let Some(ref transport_session) = *transport_guard {
                            if transport_session.is_connected() {
                                // Release guard before async operation
                                drop(transport_guard);

                                // Use the convenience method that handles guard lifetimes
                                let guard = transport.read().await;
                                if let Some(ref ts) = *guard {
                                    if let Err(e) = ts.send_audio(&samples, timestamp_ms).await {
                                        tracing::debug!("Transport send: {}", e);
                                    }
                                }
                            }
                        }

                        // Update timestamp (20ms per frame at 16kHz)
                        timestamp_ms += 20;
                    }
                }
            }
        });
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
        // Signal shutdown to all spawned tasks
        let _ = self.shutdown_tx.send(());

        // Close transport if connected
        if let Some(ref mut transport) = *self.transport.write().await {
            let _ = transport.close().await;
        }

        self.set_state(VoiceSessionState::Ended).await;

        let _ = self.event_tx.send(VoiceSessionEvent::Ended {
            reason: reason.into(),
        });
    }

    /// Check if transport is connected
    pub async fn is_transport_connected(&self) -> bool {
        if let Some(ref transport) = *self.transport.read().await {
            transport.is_connected()
        } else {
            false
        }
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

    /// Process audio through VAD and return whether speech is detected
    ///
    /// Uses Silero VAD if enabled, otherwise falls back to energy-based detection.
    pub fn detect_voice_activity(&self, samples: &[f32]) -> (bool, VadResult) {
        if let Some(ref vad) = self.vad {
            // Use Silero VAD
            use voice_agent_core::{SampleRate, Channels};
            let mut frame = AudioFrame::new(
                samples.to_vec(),
                SampleRate::Hz16000,
                Channels::Mono,
                0,
            );

            let vad_guard = vad.lock();
            match vad_guard.process(&mut frame) {
                Ok((_state, _prob, result)) => {
                    let is_speech = matches!(
                        result,
                        VadResult::SpeechConfirmed | VadResult::SpeechContinue | VadResult::PotentialSpeechStart
                    );
                    (is_speech, result)
                }
                Err(e) => {
                    tracing::warn!("VAD error: {}, falling back to energy", e);
                    let energy = calculate_energy(samples);
                    let is_speech = energy > self.config.vad_energy_threshold;
                    (is_speech, if is_speech { VadResult::SpeechContinue } else { VadResult::Silence })
                }
            }
        } else {
            // Use simple energy-based detection
            let energy = calculate_energy(samples);
            let is_speech = energy > self.config.vad_energy_threshold;
            (is_speech, if is_speech { VadResult::SpeechContinue } else { VadResult::Silence })
        }
    }

    /// Reset VAD state
    pub fn reset_vad(&self) {
        if let Some(ref vad) = self.vad {
            vad.lock().reset();
        }
    }

    /// Get current VAD state
    pub async fn get_vad_state(&self) -> VadState {
        *self.vad_state.read().await
    }
}

/// Calculate RMS energy of audio samples
fn calculate_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
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

    #[tokio::test]
    async fn test_voice_session_no_transport() {
        let session = VoiceSession::new("test", VoiceSessionConfig::default()).unwrap();

        // No transport attached
        assert!(!session.is_transport_connected().await);

        // Connect should fail without transport
        let result = session.connect_transport("offer").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_voice_session_attach_transport() {
        let session = VoiceSession::new("test", VoiceSessionConfig::default()).unwrap();

        // Attach a transport session
        let transport = TransportSession::new(SessionConfig::default());
        session.attach_transport(transport).await;

        // Transport attached but not connected yet
        assert!(!session.is_transport_connected().await);
    }

    #[test]
    fn test_calculate_energy() {
        // Silence should have zero energy
        let silence = vec![0.0f32; 100];
        assert!(calculate_energy(&silence) < 0.001);

        // Loud signal should have high energy
        let loud = vec![0.5f32; 100];
        assert!(calculate_energy(&loud) > 0.4);

        // Empty should return 0
        assert_eq!(calculate_energy(&[]), 0.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = VoiceSessionConfig::default();
        assert!(config.barge_in_enabled);
        assert_eq!(config.silence_timeout_ms, 800);
        assert_eq!(config.audio_poll_interval_ms, 20);
        assert!(config.vad_energy_threshold > 0.0);
    }
}
