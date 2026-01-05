//! Streaming Speech-to-Text
//!
//! Supports multiple STT backends with enhanced decoding:
//! - Whisper (via ONNX)
//! - IndicConformer (for Indian languages)
//!
//! ## P0-2 FIX: Engine Routing
//!
//! The STT system now properly routes to the selected engine:
//! - `SttEngine::IndicConformer` uses the native IndicConformerStt
//! - `SttEngine::Whisper` uses ONNX-based Whisper
//! - `SttEngine::Wav2Vec2` uses ONNX-based Wav2Vec2

mod decoder;
mod indicconformer;
mod streaming;
mod vocab;

pub use decoder::{DecoderConfig, EnhancedDecoder};
pub use indicconformer::{IndicConformerConfig, IndicConformerStt, MelFilterbank};
pub use streaming::{StreamingStt, SttConfig, SttEngine};
pub use vocab::{load_domain_vocab, load_vocabulary, Vocabulary};

use crate::PipelineError;
use std::sync::Arc;
use voice_agent_core::TranscriptResult;

/// STT backend trait
#[async_trait::async_trait]
pub trait SttBackend: Send + Sync {
    /// Process audio chunk and return partial transcript
    async fn process_chunk(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<TranscriptResult>, PipelineError>;

    /// Finalize and return final transcript
    async fn finalize(&mut self) -> Result<TranscriptResult, PipelineError>;

    /// Reset state
    fn reset(&mut self);

    /// Get current partial transcript
    fn partial(&self) -> Option<&TranscriptResult>;

    /// Synchronous process for use in non-async contexts
    /// Default implementation panics - override for sync backends
    fn process(&mut self, _audio: &[f32]) -> Result<Option<TranscriptResult>, PipelineError> {
        Err(PipelineError::Stt(
            "Sync process not implemented for this backend".to_string(),
        ))
    }

    /// Synchronous finalize for use in non-async contexts
    /// Default implementation returns empty transcript - override for sync backends
    fn finalize_sync(&mut self) -> TranscriptResult {
        TranscriptResult::default()
    }
}

// ============================================================================
// P0-2 FIX: Backend Implementations for Engine Routing
// ============================================================================

/// IndicConformer STT Backend wrapper (for Arc<dyn SttBackend>)
///
/// This wraps IndicConformerStt to provide interior mutability via Mutex,
/// allowing it to be used as Arc<dyn SttBackend>.
pub struct IndicConformerBackend {
    inner: parking_lot::Mutex<IndicConformerStt>,
}

impl IndicConformerBackend {
    /// Create a new IndicConformer backend
    ///
    /// # Arguments
    /// * `model_dir` - Path to the model directory containing assets/
    /// * `config` - IndicConformer configuration
    pub fn new(
        model_dir: impl AsRef<std::path::Path>,
        config: IndicConformerConfig,
    ) -> Result<Self, PipelineError> {
        let stt = IndicConformerStt::new(model_dir, config)?;
        tracing::info!("IndicConformer STT backend loaded successfully");
        Ok(Self {
            inner: parking_lot::Mutex::new(stt),
        })
    }

    /// Create with default Hindi config
    pub fn new_hindi(model_dir: impl AsRef<std::path::Path>) -> Result<Self, PipelineError> {
        Self::new(model_dir, IndicConformerConfig::default())
    }

    /// Add entities to boost in decoder
    pub fn add_entities(&self, entities: impl IntoIterator<Item = impl AsRef<str>>) {
        self.inner.lock().add_entities(entities);
    }

    /// Set start time for timestamps
    pub fn set_start_time(&self, time_ms: u64) {
        self.inner.lock().set_start_time(time_ms);
    }
}

#[async_trait::async_trait]
impl SttBackend for IndicConformerBackend {
    async fn process_chunk(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<TranscriptResult>, PipelineError> {
        self.inner.lock().process(audio)
    }

    async fn finalize(&mut self) -> Result<TranscriptResult, PipelineError> {
        Ok(self.inner.lock().finalize())
    }

    fn reset(&mut self) {
        self.inner.lock().reset();
    }

    fn partial(&self) -> Option<&TranscriptResult> {
        None // Partials returned through process_chunk
    }
}

/// Stub STT backend for testing or when models are unavailable
pub struct StubSttBackend {
    language: String,
    partial_text: parking_lot::Mutex<String>,
}

impl StubSttBackend {
    pub fn new(language: impl Into<String>) -> Self {
        tracing::warn!("Using stub STT backend - no transcription will occur");
        Self {
            language: language.into(),
            partial_text: parking_lot::Mutex::new(String::new()),
        }
    }
}

#[async_trait::async_trait]
impl SttBackend for StubSttBackend {
    async fn process_chunk(
        &mut self,
        _audio: &[f32],
    ) -> Result<Option<TranscriptResult>, PipelineError> {
        // Return empty partial - no actual transcription
        Ok(None)
    }

    async fn finalize(&mut self) -> Result<TranscriptResult, PipelineError> {
        let text = std::mem::take(&mut *self.partial_text.lock());
        Ok(TranscriptResult {
            text,
            is_final: true,
            confidence: 0.0,
            start_time_ms: 0,
            end_time_ms: 0,
            language: Some(self.language.clone()),
            words: vec![],
        })
    }

    fn reset(&mut self) {
        self.partial_text.lock().clear();
    }

    fn partial(&self) -> Option<&TranscriptResult> {
        None
    }
}

// ============================================================================
// P0-2 FIX: Factory function for creating backends
// ============================================================================

/// Create an STT backend based on engine selection
///
/// # Arguments
/// * `engine` - Which STT engine to use
/// * `model_dir` - Path to the model directory
/// * `language` - Language code (e.g., "hi" for Hindi)
#[allow(unused_variables)] // model_dir unused for stub backends
pub fn create_stt_backend(
    engine: SttEngine,
    model_dir: Option<&std::path::Path>,
    language: &str,
) -> Result<Arc<parking_lot::Mutex<dyn SttBackend>>, PipelineError> {
    match engine {
        SttEngine::IndicConformer => {
            let path = model_dir.ok_or_else(|| {
                PipelineError::Model("IndicConformer requires model_dir".to_string())
            })?;

            let config = IndicConformerConfig {
                language: language.to_string(),
                ..Default::default()
            };

            let backend = IndicConformerBackend::new(path, config)?;
            Ok(Arc::new(parking_lot::Mutex::new(backend)))
        },

        SttEngine::Whisper => {
            // Whisper uses StreamingStt which already exists
            if let Some(path) = model_dir {
                let config = SttConfig {
                    engine: SttEngine::Whisper,
                    language: Some(language.to_string()),
                    ..Default::default()
                };

                let backend = StreamingStt::new(path, config)?;
                Ok(Arc::new(parking_lot::Mutex::new(backend)))
            } else {
                tracing::warn!("Whisper requested but no model_dir, using stub");
                Ok(Arc::new(parking_lot::Mutex::new(StubSttBackend::new(
                    language,
                ))))
            }
        },

        SttEngine::Wav2Vec2 => {
            // TODO: Implement Wav2Vec2 backend
            tracing::warn!("Wav2Vec2 STT not yet implemented, using stub backend");
            Ok(Arc::new(parking_lot::Mutex::new(StubSttBackend::new(
                language,
            ))))
        },
    }
}

/// Create an IndicConformer backend directly (convenience function)
///
/// # Arguments
/// * `model_dir` - Path to the IndicConformer model directory
/// * `language` - Language code (e.g., "hi" for Hindi)
pub fn create_indicconformer(
    model_dir: impl AsRef<std::path::Path>,
    language: &str,
) -> Result<IndicConformerBackend, PipelineError> {
    let config = IndicConformerConfig {
        language: language.to_string(),
        ..Default::default()
    };
    IndicConformerBackend::new(model_dir, config)
}
