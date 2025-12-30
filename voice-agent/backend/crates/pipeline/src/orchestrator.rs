//! Voice Pipeline Orchestrator
//!
//! Coordinates VAD, STT, TTS, and turn detection for real-time conversation.
//!
//! ## P0-3 FIX: LLM Integration
//!
//! The orchestrator now includes automatic LLM integration:
//! - When a FinalTranscript is received, the LLM is automatically called
//! - LLM response is streamed through the TTS processor chain
//! - Barge-in handling works throughout the entire flow

use futures::StreamExt;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{broadcast, mpsc};

use crate::stt::{StreamingStt, SttConfig};
use crate::tts::{StreamingTts, TtsConfig, TtsEvent};
use crate::turn_detection::{HybridTurnDetector, TurnDetectionConfig, TurnDetectionResult};
use crate::vad::{VadConfig, VadState, VoiceActivityDetector};
use crate::PipelineError;
use voice_agent_core::{
    AudioFrame, ControlFrame, Frame, GenerateRequest, Language, LanguageModel, ProcessorContext,
    TranscriptResult,
};

// P1 FIX: Import processors for streaming LLM → TTS pipeline
use crate::processors::{
    InterruptHandler, InterruptHandlerConfig, ProcessorChain, SentenceDetector,
    SentenceDetectorConfig, TtsProcessor, TtsProcessorConfig,
};

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
    /// P1 FIX: Processor chain configuration for streaming LLM output
    pub processors: ProcessorChainConfig,
    /// P0-3 FIX: LLM configuration for automatic response generation
    pub llm: LlmConfig,
}

/// P0-3 FIX: LLM configuration for the pipeline
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Enable automatic LLM response generation
    pub enabled: bool,
    /// System prompt for the LLM
    pub system_prompt: String,
    /// Language for responses
    pub language: Language,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for generation (0.0 - 1.0)
    pub temperature: f32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            system_prompt: "You are a helpful voice assistant for Kotak Gold Loan services. \
                Respond concisely and naturally in Hindi or Hinglish. Keep responses brief \
                as they will be spoken aloud."
                .to_string(),
            language: Language::Hindi,
            max_tokens: 256,
            temperature: 0.7,
        }
    }
}

/// P1 FIX: Configuration for the processor chain
#[derive(Debug, Clone)]
pub struct ProcessorChainConfig {
    /// Enable processor chain for streaming LLM output
    pub enabled: bool,
    /// Sentence detector configuration
    pub sentence_detector: SentenceDetectorConfig,
    /// TTS processor configuration
    pub tts_processor: TtsProcessorConfig,
    /// Interrupt handler configuration
    pub interrupt_handler: InterruptHandlerConfig,
}

impl Default for ProcessorChainConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sentence_detector: SentenceDetectorConfig::default(),
            tts_processor: TtsProcessorConfig::default(),
            interrupt_handler: InterruptHandlerConfig::default(),
        }
    }
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
            processors: ProcessorChainConfig::default(),
            llm: LlmConfig::default(),
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
    /// P1 FIX: Processor chain for streaming LLM → TTS
    /// Contains: SentenceDetector → TtsProcessor → InterruptHandler
    processor_chain: Option<ProcessorChain>,
    /// P0-3 FIX: LLM for automatic response generation
    llm: Option<Arc<dyn LanguageModel>>,
    /// P0-3 FIX: Pending transcript waiting for LLM processing
    pending_transcript: Mutex<Option<TranscriptResult>>,
}

impl VoicePipeline {
    /// Create a new voice pipeline with simple components (for testing)
    pub fn simple(config: PipelineConfig) -> Result<Self, PipelineError> {
        let vad = Arc::new(VoiceActivityDetector::simple(config.vad.clone())?);
        let turn_detector = Arc::new(HybridTurnDetector::new(config.turn_detection.clone()));
        let stt = Arc::new(Mutex::new(StreamingStt::simple(config.stt.clone())));
        let tts = Arc::new(StreamingTts::simple(config.tts.clone()));

        let (event_tx, _) = broadcast::channel(100);

        // P1 FIX: Build processor chain if enabled
        let processor_chain = if config.processors.enabled {
            Some(Self::build_processor_chain(&config.processors, tts.clone()))
        } else {
            None
        };

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
            processor_chain,
            llm: None, // P0-3 FIX: LLM not set by default, use with_llm()
            pending_transcript: Mutex::new(None),
        })
    }

    /// P0-3 FIX: Set the LLM for automatic response generation
    ///
    /// When set, the pipeline will automatically call the LLM when a
    /// final transcript is received, and stream the response through TTS.
    ///
    /// # Example
    /// ```ignore
    /// let llm = Arc::new(OllamaBackend::new(config));
    /// let pipeline = VoicePipeline::simple(config)?
    ///     .with_llm(llm);
    /// ```
    pub fn with_llm(mut self, llm: Arc<dyn LanguageModel>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// P0-3 FIX: Check if LLM is configured
    pub fn has_llm(&self) -> bool {
        self.llm.is_some()
    }

    /// P0-3 FIX: Handle a final transcript by calling LLM and streaming to TTS
    ///
    /// This is the core auto-response logic that connects STT → LLM → TTS.
    ///
    /// # Arguments
    /// * `transcript` - The final transcript from STT
    ///
    /// # Returns
    /// Ok(()) on success, or error if LLM/TTS fails
    async fn handle_final_transcript(&self, transcript: &TranscriptResult) -> Result<(), PipelineError> {
        // Check if LLM is configured and enabled
        let llm = match &self.llm {
            Some(llm) if self.config.llm.enabled => llm.clone(),
            _ => {
                tracing::debug!("LLM not configured or disabled, skipping auto-response");
                return Ok(());
            }
        };

        tracing::info!(
            transcript = %transcript.text,
            confidence = %transcript.confidence,
            "Processing final transcript through LLM"
        );

        // Build the LLM request
        let request = GenerateRequest::new(&self.config.llm.system_prompt)
            .with_user_message(&transcript.text)
            .with_temperature(self.config.llm.temperature)
            .with_max_tokens(self.config.llm.max_tokens);

        // Stream the LLM response
        let mut stream = llm.generate_stream(request);

        // Create channel for TTS input
        let (tx, rx) = mpsc::channel::<String>(100);

        // Start TTS streaming in background
        let tts_handle = {
            let pipeline_event_tx = self.event_tx.clone();
            let language = self.config.llm.language;

            // Use processor chain if available, otherwise fall back to simple speak
            if self.has_processor_chain() {
                // Stream through processor chain
                let output_rx = self.speak_streaming(rx, language).await?;

                // Spawn task to forward TTS audio frames to event channel
                tokio::spawn(async move {
                    let mut output_rx = output_rx;
                    while let Some(frame) = output_rx.recv().await {
                        if let Frame::AudioOutput(audio) = frame {
                            let _ = pipeline_event_tx.send(PipelineEvent::TtsAudio {
                                samples: audio.samples.into(),
                                text: String::new(), // Word text not available in this path
                                is_final: false,
                            });
                        }
                    }
                })
            } else {
                // Fall back to collecting full response then speaking
                tokio::spawn(async move {
                    // This path doesn't stream - handled below
                })
            }
        };

        // Stream LLM chunks to TTS
        let mut full_response = String::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    full_response.push_str(&chunk.delta);

                    // Send chunk to TTS channel
                    if tx.send(chunk.delta).await.is_err() {
                        tracing::warn!("TTS channel closed while streaming LLM response");
                        break;
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "LLM streaming error");
                    let _ = self.event_tx.send(PipelineEvent::Error(format!("LLM error: {}", e)));
                    break;
                }
            }
        }

        // Drop sender to signal completion
        drop(tx);

        // If no processor chain, use simple speak with full response
        if !self.has_processor_chain() && !full_response.is_empty() {
            self.speak(&full_response).await?;
        }

        // Wait for TTS to complete
        let _ = tts_handle.await;

        // Transition back to Idle state
        *self.state.lock() = PipelineState::Idle;
        self.turn_detector.reset();

        tracing::info!(
            response_length = full_response.len(),
            "LLM response completed"
        );

        Ok(())
    }

    /// P0-3 FIX: Process pending transcript if in Processing state
    ///
    /// This should be called periodically or after state transitions
    /// to check if there's a pending transcript to process.
    pub async fn process_pending(&self) -> Result<(), PipelineError> {
        // Check if we're in Processing state with a pending transcript
        if *self.state.lock() != PipelineState::Processing {
            return Ok(());
        }

        // Take the pending transcript
        let transcript = self.pending_transcript.lock().take();

        if let Some(transcript) = transcript {
            self.handle_final_transcript(&transcript).await?;
        }

        Ok(())
    }

    /// P1 FIX: Build the processor chain for LLM streaming output
    ///
    /// Chain: SentenceDetector → TtsProcessor → InterruptHandler
    ///
    /// This pipeline:
    /// 1. Buffers LLM text chunks until sentence boundary
    /// 2. Sends complete sentences to TTS for synthesis
    /// 3. Handles barge-in interrupts during audio playback
    fn build_processor_chain(
        config: &ProcessorChainConfig,
        tts: Arc<StreamingTts>,
    ) -> ProcessorChain {
        let mut chain = ProcessorChain::new("llm-to-audio");

        // 1. Sentence detector: buffers LLM chunks, emits sentences
        chain.add(SentenceDetector::new(config.sentence_detector.clone()));

        // 2. TTS processor: converts sentences to audio frames
        // Share the TTS instance with the main pipeline for barge-in coordination
        let mut tts_config = config.tts_processor.clone();
        tts_config.tts = TtsConfig::default(); // Will use shared instance
        chain.add(TtsProcessor::with_tts(tts_config, tts));

        // 3. Interrupt handler: manages barge-in during audio output
        chain.add(InterruptHandler::new(config.interrupt_handler.clone()));

        tracing::info!(
            chain_name = chain.name(),
            processor_count = chain.len(),
            "Built LLM → Audio processor chain"
        );

        chain
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
        let _ = self
            .event_tx
            .send(PipelineEvent::VadStateChanged(vad_state));

        // 2. Check for barge-in if speaking
        if *self.state.lock() == PipelineState::Speaking
            && self.check_barge_in(&frame, vad_state).await?
        {
            return Ok(());
        }

        // 3. Process based on state
        // NOTE: We copy the state to avoid holding MutexGuard across await points
        let current_state = *self.state.lock();

        match current_state {
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
                    let _ = self
                        .event_tx
                        .send(PipelineEvent::PartialTranscript(partial.clone()));

                    // Update turn detector with transcript
                    let turn_result = self.turn_detector.process(vad_state, Some(&partial.text))?;

                    let _ = self
                        .event_tx
                        .send(PipelineEvent::TurnStateChanged(turn_result.clone()));

                    // Check for turn completion
                    if turn_result.is_turn_complete {
                        let final_transcript = self.stt.lock().finalize();
                        let _ = self
                            .event_tx
                            .send(PipelineEvent::FinalTranscript(final_transcript.clone()));

                        // P0-3 FIX: Store transcript and transition to Processing
                        *self.pending_transcript.lock() = Some(final_transcript);
                        *self.state.lock() = PipelineState::Processing;
                    }
                } else {
                    // No transcript yet, just update turn detector with VAD
                    let turn_result = self.turn_detector.process(vad_state, None)?;
                    let _ = self
                        .event_tx
                        .send(PipelineEvent::TurnStateChanged(turn_result));
                }
            }

            PipelineState::Processing => {
                // P0-3 FIX: Auto-process pending transcript through LLM
                // This is triggered when we have an LLM configured
                if self.has_llm() {
                    // Take transcript before await (releases lock)
                    let transcript = self.pending_transcript.lock().take();

                    if let Some(transcript) = transcript {
                        // Process transcript asynchronously - errors are logged, not propagated
                        // This keeps the audio processing loop responsive
                        if let Err(e) = self.handle_final_transcript(&transcript).await {
                            tracing::error!(error = %e, "Failed to process transcript through LLM");
                            let _ = self.event_tx.send(PipelineEvent::Error(e.to_string()));
                            *self.state.lock() = PipelineState::Idle;
                        }
                    }
                }
                // Audio is still monitored for barge-in during processing
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
    async fn check_barge_in(
        &self,
        frame: &AudioFrame,
        vad_state: VadState,
    ) -> Result<bool, PipelineError> {
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
                let _ = self.event_tx.send(PipelineEvent::BargeIn {
                    at_word: word_index,
                });

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
                TtsEvent::Audio {
                    samples,
                    text,
                    is_final,
                    ..
                } => {
                    let _ = self.event_tx.send(PipelineEvent::TtsAudio {
                        samples,
                        text,
                        is_final,
                    });
                },
                TtsEvent::Complete => {
                    *self.state.lock() = PipelineState::Idle;
                    self.turn_detector.reset();
                    break;
                },
                TtsEvent::BargedIn { word_index } => {
                    let _ = self.event_tx.send(PipelineEvent::BargeIn {
                        at_word: word_index,
                    });
                    break;
                },
                TtsEvent::Error(e) => {
                    let _ = self.event_tx.send(PipelineEvent::Error(e));
                    *self.state.lock() = PipelineState::Idle;
                    break;
                },
                _ => {},
            }
        }

        Ok(())
    }

    /// P1 FIX: Speak using streaming LLM output through the processor chain
    ///
    /// This method processes LLM text chunks through:
    /// 1. SentenceDetector - buffers chunks until sentence boundary
    /// 2. TtsProcessor - converts sentences to audio
    /// 3. InterruptHandler - handles barge-in during playback
    ///
    /// This provides lower latency than `speak()` for streaming LLM output
    /// because TTS starts as soon as the first sentence is complete.
    ///
    /// # Arguments
    /// * `chunk_rx` - Receiver for LLM text chunks
    /// * `language` - Language for sentence detection and TTS
    ///
    /// # Returns
    /// Receiver for output audio frames
    pub async fn speak_streaming(
        &self,
        mut chunk_rx: mpsc::Receiver<String>,
        language: Language,
    ) -> Result<mpsc::Receiver<Frame>, PipelineError> {
        // Check if processor chain is available
        let chain = self
            .processor_chain
            .as_ref()
            .ok_or(PipelineError::NotInitialized)?;

        // Set state
        *self.state.lock() = PipelineState::Speaking;
        self.turn_detector.set_agent_speaking();
        *self.barge_in_speech_ms.lock() = 0;

        // Start the processor chain with session context
        let context = ProcessorContext::new("streaming-session").with_language(language);

        let (input_tx, output_rx) = chain.run(context);

        // Spawn task to feed LLM chunks into the processor chain
        tokio::spawn(async move {
            while let Some(chunk) = chunk_rx.recv().await {
                // Create LLM chunk frame
                let frame = Frame::LLMChunk {
                    text: chunk,
                    is_final: false,
                };

                if input_tx.send(frame).await.is_err() {
                    tracing::warn!("Processor chain input channel closed");
                    break;
                }
            }

            // Send final LLM chunk to signal completion
            let _ = input_tx
                .send(Frame::LLMChunk {
                    text: String::new(),
                    is_final: true,
                })
                .await;

            // Send flush control frame
            let _ = input_tx.send(Frame::Control(ControlFrame::Flush)).await;

            tracing::debug!("LLM streaming complete, sent flush to processor chain");
        });

        // Note: We return the output_rx for the caller to process audio frames
        // The caller is responsible for sending audio to the transport layer

        Ok(output_rx)
    }

    /// P1 FIX: Check if processor chain is enabled and available
    pub fn has_processor_chain(&self) -> bool {
        self.processor_chain.is_some()
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
        AudioFrame::new(samples, SampleRate::Hz16000, Channels::Mono, 0)
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
