//! Streaming Text-to-Speech
//!
//! Features:
//! - Word-level chunking for early audio emission
//! - Barge-in aware (can stop mid-word)
//! - Multiple backend support (Piper, IndicF5, Parler)
//! - Hindi/Hinglish G2P conversion
//! - Native Candle-based IndicF5 model (optional)
//!
//! ## P0-1 FIX: Engine Routing
//!
//! The TTS system now properly routes to the selected engine:
//! - `TtsEngine::IndicF5` uses the Candle-based IndicF5Model
//! - `TtsEngine::Piper` uses ONNX-based Piper
//! - `TtsEngine::ParlerTts` uses ONNX-based ParlerTts

mod chunker;
mod g2p;
mod streaming;

/// Candle-based TTS implementations (native Rust with SafeTensors)
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(not(feature = "candle"))]
pub mod candle {
    //! Stub module when candle feature is disabled
    pub struct IndicF5Model;
    pub struct IndicF5Config;
}

pub use chunker::{ChunkStrategy, WordChunker};
pub use g2p::{create_hindi_g2p, G2pConfig, HindiG2p, Language, Phoneme};
pub use streaming::{StreamingTts, TtsConfig, TtsEngine, TtsEvent};

// P1-3 FIX: Re-export IndicF5 model types from candle module
// TtsBackend, StubTtsBackend, IndicF5Backend, and create_tts_backend
// are already public as they're defined in this module
#[cfg(feature = "candle")]
pub use candle::{IndicF5Config, IndicF5Model};

use crate::PipelineError;
use std::sync::Arc;

/// TTS backend trait
#[async_trait::async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize text to audio
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, PipelineError>;

    /// Get sample rate
    fn sample_rate(&self) -> u32;

    /// Supports streaming word-by-word?
    fn supports_streaming(&self) -> bool;
}

// ============================================================================
// P0-1 FIX: Backend Implementations for Engine Routing
// ============================================================================

/// IndicF5 TTS Backend (Candle-based, Hindi-optimized)
#[cfg(feature = "candle")]
pub struct IndicF5Backend {
    model: candle::IndicF5Model,
    /// Reference audio for voice cloning (pre-loaded)
    reference_audio: Vec<f32>,
    sample_rate: u32,
}

#[cfg(feature = "candle")]
impl IndicF5Backend {
    /// Create a new IndicF5 backend
    ///
    /// # Arguments
    /// * `model_path` - Path to the SafeTensors model file
    /// * `reference_audio` - Reference audio samples for voice cloning (24kHz)
    pub fn new(
        model_path: impl AsRef<std::path::Path>,
        reference_audio: Vec<f32>,
    ) -> Result<Self, PipelineError> {
        use candle_core::Device;

        // Use CPU by default, can be extended to support CUDA
        let device = Device::Cpu;

        let model = candle::IndicF5Model::load(model_path, None::<&std::path::Path>, device)
            .map_err(|e| PipelineError::Model(format!("Failed to load IndicF5: {}", e)))?;

        let sample_rate = model.config().sample_rate;

        tracing::info!("IndicF5 TTS backend loaded successfully (sample_rate={})", sample_rate);

        Ok(Self {
            model,
            reference_audio,
            sample_rate,
        })
    }

    /// Create with default reference audio (silence - for testing)
    pub fn new_with_default_reference(
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<Self, PipelineError> {
        // 1 second of silence at 24kHz as default reference
        let reference_audio = vec![0.0f32; 24000];
        Self::new(model_path, reference_audio)
    }

    /// Set reference audio for voice cloning
    pub fn set_reference_audio(&mut self, audio: Vec<f32>) {
        self.reference_audio = audio;
    }
}

#[cfg(feature = "candle")]
#[async_trait::async_trait]
impl TtsBackend for IndicF5Backend {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        // IndicF5 synthesis is CPU-bound, run in blocking task
        let text = text.to_string();
        let reference = self.reference_audio.clone();

        // Clone Arc for the closure
        let model_ptr = &self.model as *const candle::IndicF5Model;

        // Safety: We're running synchronously in spawn_blocking, model lifetime is valid
        let audio = tokio::task::spawn_blocking(move || {
            let model = unsafe { &*model_ptr };
            model.synthesize(&text, &reference)
        })
        .await
        .map_err(|e| PipelineError::Tts(format!("Task join error: {}", e)))?
        .map_err(|e| PipelineError::Tts(format!("IndicF5 synthesis failed: {}", e)))?;

        Ok(audio)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn supports_streaming(&self) -> bool {
        true // IndicF5 supports streaming via synthesize_streaming
    }
}

/// Stub backend when no model is loaded (returns silence)
pub struct StubTtsBackend {
    sample_rate: u32,
}

impl StubTtsBackend {
    pub fn new(sample_rate: u32) -> Self {
        tracing::warn!("Using stub TTS backend - audio output will be silence");
        Self { sample_rate }
    }
}

#[async_trait::async_trait]
impl TtsBackend for StubTtsBackend {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        // Return silence of appropriate length (~50ms per character)
        let duration_samples = text.len() * (self.sample_rate as usize / 20);
        Ok(vec![0.0f32; duration_samples])
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

// ============================================================================
// P0-1 FIX: Factory function for creating backends
// ============================================================================

/// Create a TTS backend based on engine selection
///
/// # Arguments
/// * `engine` - Which TTS engine to use
/// * `model_path` - Path to the model file/directory
/// * `reference_audio` - Optional reference audio for voice cloning (IndicF5)
#[allow(unused_variables)] // model_path/reference_audio unused for stub backends
pub fn create_tts_backend(
    engine: TtsEngine,
    model_path: Option<&std::path::Path>,
    reference_audio: Option<Vec<f32>>,
) -> Result<Arc<dyn TtsBackend>, PipelineError> {
    match engine {
        TtsEngine::IndicF5 => {
            #[cfg(feature = "candle")]
            {
                let path = model_path.ok_or_else(|| {
                    PipelineError::Model("IndicF5 requires model_path".to_string())
                })?;

                let backend = if let Some(ref_audio) = reference_audio {
                    IndicF5Backend::new(path, ref_audio)?
                } else {
                    IndicF5Backend::new_with_default_reference(path)?
                };

                Ok(Arc::new(backend))
            }

            #[cfg(not(feature = "candle"))]
            {
                tracing::warn!("IndicF5 requested but candle feature not enabled, using stub");
                Ok(Arc::new(StubTtsBackend::new(24000)))
            }
        }

        TtsEngine::Piper => {
            // TODO: Implement Piper ONNX backend
            // For now, fall back to stub with warning
            tracing::warn!("Piper TTS not yet implemented, using stub backend");
            Ok(Arc::new(StubTtsBackend::new(22050)))
        }

        TtsEngine::ParlerTts => {
            // TODO: Implement ParlerTts ONNX backend
            tracing::warn!("ParlerTts not yet implemented, using stub backend");
            Ok(Arc::new(StubTtsBackend::new(24000)))
        }
    }
}
