//! TTS Processor
//!
//! Bridges Frame::Sentence to Frame::AudioOutput via StreamingTts.
//! Wires the SentenceDetector output directly to TTS synthesis.

use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;

use voice_agent_core::{Frame, FrameProcessor, ProcessorContext, Result, Language};

use crate::tts::{StreamingTts, TtsConfig, TtsEvent};

/// TTS processor configuration
#[derive(Debug, Clone)]
pub struct TtsProcessorConfig {
    /// TTS configuration
    pub tts: TtsConfig,
    /// Enable parallel synthesis (queue sentences)
    pub parallel_synthesis: bool,
    /// Maximum queued sentences
    pub max_queue_size: usize,
    /// Sample rate for output audio
    pub sample_rate: u32,
}

impl Default for TtsProcessorConfig {
    fn default() -> Self {
        Self {
            tts: TtsConfig::default(),
            parallel_synthesis: false,
            max_queue_size: 5,
            sample_rate: 22050,
        }
    }
}

/// TTS processor that converts sentences to audio frames
pub struct TtsProcessor {
    config: TtsProcessorConfig,
    tts: Arc<StreamingTts>,
    /// Current sentence index being processed
    current_sentence: Mutex<usize>,
    /// Whether synthesis is active
    active: Mutex<bool>,
    /// Barge-in requested
    barge_in: Mutex<bool>,
}

impl TtsProcessor {
    /// Create a new TTS processor
    pub fn new(config: TtsProcessorConfig) -> Self {
        let tts = Arc::new(StreamingTts::simple(config.tts.clone()));
        Self {
            config,
            tts,
            current_sentence: Mutex::new(0),
            active: Mutex::new(false),
            barge_in: Mutex::new(false),
        }
    }

    /// Create with a shared TTS instance
    pub fn with_tts(config: TtsProcessorConfig, tts: Arc<StreamingTts>) -> Self {
        Self {
            config,
            tts,
            current_sentence: Mutex::new(0),
            active: Mutex::new(false),
            barge_in: Mutex::new(false),
        }
    }

    /// Synthesize a sentence and return audio frames
    async fn synthesize_sentence(
        &self,
        text: &str,
        _language: Language, // May be used for language-specific TTS voices in future
        sentence_index: usize,
    ) -> Result<Vec<Frame>> {
        // Check for barge-in before starting
        if *self.barge_in.lock() {
            return Ok(vec![Frame::BargeIn {
                audio_position_ms: 0,
                transcript: None,
            }]);
        }

        *self.active.lock() = true;
        *self.current_sentence.lock() = sentence_index;

        // Create channel for TTS events
        let (tx, mut rx) = mpsc::channel::<TtsEvent>(32);

        // Start TTS synthesis
        self.tts.start(text, tx);

        let mut frames = Vec::new();

        // Process TTS events synchronously by polling
        loop {
            // Check for barge-in during synthesis
            if *self.barge_in.lock() {
                self.tts.barge_in();
                frames.push(Frame::BargeIn {
                    audio_position_ms: frames.len() as u64 * 20, // Approximate ms position
                    transcript: None,
                });
                break;
            }

            // Process next chunk
            match self.tts.process_next() {
                Ok(Some(TtsEvent::Audio { samples, text: chunk_text, is_final, word_indices })) => {
                    frames.push(Frame::AudioOutput(voice_agent_core::AudioFrame::new(
                        samples.to_vec(),
                        voice_agent_core::SampleRate::Hz16000, // Will be resampled if needed
                        voice_agent_core::Channels::Mono,
                        frames.len() as u64,
                    )));

                    tracing::trace!(
                        sentence = sentence_index,
                        chunk = chunk_text,
                        words = ?word_indices,
                        is_final = is_final,
                        "TTS chunk synthesized"
                    );

                    if is_final {
                        break;
                    }
                }
                Ok(Some(TtsEvent::Complete)) => {
                    tracing::debug!(sentence = sentence_index, "TTS synthesis complete");
                    break;
                }
                Ok(Some(TtsEvent::BargedIn { word_index })) => {
                    frames.push(Frame::BargeIn {
                        audio_position_ms: word_index as u64 * 100, // Approximate word to ms
                        transcript: None,
                    });
                    break;
                }
                Ok(Some(TtsEvent::Error(e))) => {
                    tracing::error!("TTS error: {}", e);
                    return Err(voice_agent_core::Error::Pipeline(
                        voice_agent_core::error::PipelineError::Tts(e),
                    ));
                }
                Ok(Some(TtsEvent::Started)) => {
                    tracing::trace!(sentence = sentence_index, "TTS started");
                }
                Ok(None) => {
                    // No more events
                    break;
                }
                Err(e) => {
                    return Err(voice_agent_core::Error::Pipeline(
                        voice_agent_core::error::PipelineError::Tts(e.to_string()),
                    ));
                }
            }

            // Also check the async channel in case start() sent events there
            if let Ok(event) = rx.try_recv() {
                match event {
                    TtsEvent::Started => {}
                    TtsEvent::Complete => break,
                    TtsEvent::BargedIn { word_index } => {
                        frames.push(Frame::BargeIn {
                            audio_position_ms: word_index as u64 * 100,
                            transcript: None,
                        });
                        break;
                    }
                    TtsEvent::Error(e) => {
                        return Err(voice_agent_core::Error::Pipeline(
                            voice_agent_core::error::PipelineError::Tts(e),
                        ));
                    }
                    _ => {}
                }
            }
        }

        *self.active.lock() = false;
        Ok(frames)
    }

    /// Request barge-in (stop synthesis)
    pub fn barge_in(&self) {
        *self.barge_in.lock() = true;
        self.tts.barge_in();
    }

    /// Check if currently synthesizing
    pub fn is_active(&self) -> bool {
        *self.active.lock()
    }

    /// Get current sentence index
    pub fn current_sentence(&self) -> usize {
        *self.current_sentence.lock()
    }

    /// Reset processor state
    pub fn reset(&self) {
        *self.current_sentence.lock() = 0;
        *self.active.lock() = false;
        *self.barge_in.lock() = false;
        self.tts.reset();
    }
}

#[async_trait]
impl FrameProcessor for TtsProcessor {
    async fn process(
        &self,
        frame: Frame,
        _context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        match frame {
            Frame::Sentence { text, language, index } => {
                tracing::debug!(
                    sentence = index,
                    text = %text,
                    language = ?language,
                    "Processing sentence for TTS"
                );

                // Synthesize the sentence
                let audio_frames = self.synthesize_sentence(&text, language, index).await?;

                Ok(audio_frames)
            }

            Frame::Control(voice_agent_core::ControlFrame::Reset) => {
                self.reset();
                Ok(vec![frame])
            }

            Frame::Control(voice_agent_core::ControlFrame::Flush) => {
                // On flush, finish any pending synthesis
                if self.is_active() {
                    self.tts.finalize_text();
                }
                Ok(vec![frame])
            }

            Frame::BargeIn { .. } => {
                // Propagate barge-in and stop synthesis
                self.barge_in();
                Ok(vec![frame])
            }

            Frame::EndOfStream => {
                // Finish any pending synthesis
                if self.is_active() {
                    self.tts.finalize_text();
                }
                self.reset();
                Ok(vec![frame])
            }

            // Pass through other frames
            _ => Ok(vec![frame]),
        }
    }

    fn name(&self) -> &'static str {
        "tts_processor"
    }

    fn description(&self) -> &str {
        "Converts sentences to synthesized audio via streaming TTS"
    }

    async fn on_start(&self, _context: &mut ProcessorContext) -> Result<()> {
        self.reset();
        Ok(())
    }

    async fn on_stop(&self, _context: &mut ProcessorContext) -> Result<()> {
        self.reset();
        Ok(())
    }

    fn can_handle(&self, frame: &Frame) -> bool {
        matches!(
            frame,
            Frame::Sentence { .. }
                | Frame::Control(_)
                | Frame::BargeIn { .. }
                | Frame::EndOfStream
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_processor() -> TtsProcessor {
        TtsProcessor::new(TtsProcessorConfig::default())
    }

    #[tokio::test]
    async fn test_processor_creation() {
        let processor = create_processor();
        assert!(!processor.is_active());
        assert_eq!(processor.current_sentence(), 0);
    }

    #[tokio::test]
    async fn test_sentence_processing() {
        let processor = create_processor();
        let mut ctx = ProcessorContext::default();

        let frames = processor
            .process(
                Frame::Sentence {
                    text: "Hello world.".to_string(),
                    language: Language::English,
                    index: 0,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should produce audio output frames
        let audio_count = frames
            .iter()
            .filter(|f| matches!(f, Frame::AudioOutput(_)))
            .count();

        assert!(audio_count > 0, "Should produce audio frames");
    }

    #[tokio::test]
    async fn test_passthrough() {
        let processor = create_processor();
        let mut ctx = ProcessorContext::default();

        // Non-sentence frames should pass through
        let frames = processor
            .process(Frame::VoiceStart, &mut ctx)
            .await
            .unwrap();

        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::VoiceStart));
    }

    #[tokio::test]
    async fn test_reset() {
        let processor = create_processor();
        let mut ctx = ProcessorContext::default();

        // Process a sentence
        let _ = processor
            .process(
                Frame::Sentence {
                    text: "Test".to_string(),
                    language: Language::English,
                    index: 0,
                },
                &mut ctx,
            )
            .await;

        // Reset
        processor.reset();

        assert!(!processor.is_active());
        assert_eq!(processor.current_sentence(), 0);
    }

    #[tokio::test]
    async fn test_barge_in() {
        let processor = create_processor();

        // Request barge-in
        processor.barge_in();

        let mut ctx = ProcessorContext::default();
        let frames = processor
            .process(
                Frame::Sentence {
                    text: "Hello".to_string(),
                    language: Language::English,
                    index: 0,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should produce barge-in frame
        assert!(frames.iter().any(|f| matches!(f, Frame::BargeIn { .. })));
    }
}
