//! Streaming TTS Engine
//!
//! Word-level streaming with barge-in support.
//!
//! ## P0-1 FIX: Engine Routing
//!
//! StreamingTts now supports multiple backends via the `TtsBackend` trait:
//! - Use `StreamingTts::with_backend()` for production with real TTS
//! - Use `StreamingTts::simple()` for testing with silence output

use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};

use super::chunker::{ChunkStrategy, ChunkerConfig, TextChunk, WordChunker};
use super::{create_tts_backend, TtsBackend};
use crate::PipelineError;

/// TTS engine selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsEngine {
    /// Piper (fast, lightweight)
    Piper,
    /// IndicF5 (Indian languages)
    IndicF5,
    /// Parler TTS (expressive)
    ParlerTts,
}

/// TTS configuration
#[derive(Debug, Clone)]
pub struct TtsConfig {
    /// Engine to use
    pub engine: TtsEngine,
    /// Sample rate
    pub sample_rate: u32,
    /// Voice/speaker ID
    pub voice_id: Option<String>,
    /// Speaking rate (1.0 = normal)
    pub speaking_rate: f32,
    /// Pitch adjustment (1.0 = normal)
    pub pitch: f32,
    /// Chunking strategy
    pub chunk_strategy: ChunkStrategy,
    /// Enable prosody hints
    pub prosody_hints: bool,
    /// P0-1 FIX: Path to the TTS model (required for IndicF5, Piper, etc.)
    pub model_path: Option<std::path::PathBuf>,
    /// P0-1 FIX: Path to reference audio for voice cloning (IndicF5)
    pub reference_audio_path: Option<std::path::PathBuf>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            engine: TtsEngine::Piper,
            sample_rate: 22050,
            voice_id: None,
            speaking_rate: 1.0,
            pitch: 1.0,
            chunk_strategy: ChunkStrategy::Adaptive,
            prosody_hints: true,
            model_path: None,
            reference_audio_path: None,
        }
    }
}

impl TtsConfig {
    /// P0-1 FIX: Create config for IndicF5 engine
    pub fn indicf5(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            engine: TtsEngine::IndicF5,
            sample_rate: 24000, // IndicF5 uses 24kHz
            model_path: Some(model_path.into()),
            ..Default::default()
        }
    }

    /// P0-1 FIX: Create config for IndicF5 with reference audio
    pub fn indicf5_with_reference(
        model_path: impl Into<std::path::PathBuf>,
        reference_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            engine: TtsEngine::IndicF5,
            sample_rate: 24000,
            model_path: Some(model_path.into()),
            reference_audio_path: Some(reference_path.into()),
            ..Default::default()
        }
    }
}

/// TTS event for streaming output
#[derive(Debug, Clone)]
pub enum TtsEvent {
    /// Audio chunk ready
    Audio {
        /// Audio samples
        samples: Arc<[f32]>,
        /// Text that was synthesized
        text: String,
        /// Word indices
        word_indices: Vec<usize>,
        /// Is final chunk
        is_final: bool,
    },
    /// Synthesis started
    Started,
    /// Synthesis complete
    Complete,
    /// Barge-in occurred, synthesis stopped
    BargedIn {
        /// Word index where barge-in occurred
        word_index: usize,
    },
    /// Error occurred
    Error(String),
}

/// Streaming TTS processor
pub struct StreamingTts {
    /// ONNX session (None for simple/testing mode) - legacy, prefer backend
    #[cfg(feature = "onnx")]
    session: Option<Session>,
    /// P0-1 FIX: TTS backend for actual synthesis
    backend: Option<Arc<dyn TtsBackend>>,
    config: TtsConfig,
    chunker: Mutex<WordChunker>,
    /// Is currently synthesizing?
    synthesizing: Mutex<bool>,
    /// Barge-in requested?
    barge_in: Mutex<bool>,
    /// Current word index
    current_word: Mutex<usize>,
}

impl StreamingTts {
    /// Create a new streaming TTS (legacy ONNX mode)
    #[cfg(feature = "onnx")]
    pub fn new(model_path: impl AsRef<Path>, config: TtsConfig) -> Result<Self, PipelineError> {
        let session = Session::builder()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let chunker_config = ChunkerConfig {
            strategy: config.chunk_strategy,
            ..Default::default()
        };

        Ok(Self {
            session: Some(session),
            backend: None,
            config,
            chunker: Mutex::new(WordChunker::new(chunker_config)),
            synthesizing: Mutex::new(false),
            barge_in: Mutex::new(false),
            current_word: Mutex::new(0),
        })
    }

    /// Create a new streaming TTS (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn new(_model_path: impl AsRef<Path>, config: TtsConfig) -> Result<Self, PipelineError> {
        Ok(Self::simple(config))
    }

    /// P0-1 FIX: Create streaming TTS with a specific backend
    ///
    /// This is the recommended constructor for production use.
    /// Use `create_tts_backend()` to create the appropriate backend.
    pub fn with_backend(backend: Arc<dyn TtsBackend>, config: TtsConfig) -> Self {
        let chunker_config = ChunkerConfig {
            strategy: config.chunk_strategy,
            ..Default::default()
        };

        let sample_rate = backend.sample_rate();
        let mut config = config;
        config.sample_rate = sample_rate;

        Self {
            #[cfg(feature = "onnx")]
            session: None,
            backend: Some(backend),
            config,
            chunker: Mutex::new(WordChunker::new(chunker_config)),
            synthesizing: Mutex::new(false),
            barge_in: Mutex::new(false),
            current_word: Mutex::new(0),
        }
    }

    /// P0-1 FIX: Create streaming TTS from config (auto-creates backend)
    ///
    /// Automatically creates the appropriate backend based on TtsConfig.engine
    pub fn from_config(config: TtsConfig) -> Result<Self, PipelineError> {
        // Load reference audio if specified
        let reference_audio = if let Some(ref path) = config.reference_audio_path {
            Some(load_reference_audio(path)?)
        } else {
            None
        };

        let backend =
            create_tts_backend(config.engine, config.model_path.as_deref(), reference_audio)?;

        Ok(Self::with_backend(backend, config))
    }

    /// Create a simple TTS for testing (no model required, returns silence)
    pub fn simple(config: TtsConfig) -> Self {
        let chunker_config = ChunkerConfig {
            strategy: config.chunk_strategy,
            ..Default::default()
        };

        Self {
            #[cfg(feature = "onnx")]
            session: None, // No model - will use stub synthesis
            backend: None,
            config,
            chunker: Mutex::new(WordChunker::new(chunker_config)),
            synthesizing: Mutex::new(false),
            barge_in: Mutex::new(false),
            current_word: Mutex::new(0),
        }
    }

    /// Start streaming synthesis
    pub fn start(&self, text: &str, tx: mpsc::Sender<TtsEvent>) {
        let mut chunker = self.chunker.lock();
        chunker.reset();
        chunker.add_text(text);
        chunker.finalize();

        *self.synthesizing.lock() = true;
        *self.barge_in.lock() = false;
        *self.current_word.lock() = 0;

        let _ = tx.try_send(TtsEvent::Started);
    }

    /// Process next chunk (call in a loop)
    pub fn process_next(&self) -> Result<Option<TtsEvent>, PipelineError> {
        if *self.barge_in.lock() {
            *self.synthesizing.lock() = false;
            let word_idx = *self.current_word.lock();
            return Ok(Some(TtsEvent::BargedIn {
                word_index: word_idx,
            }));
        }

        if !*self.synthesizing.lock() {
            return Ok(None);
        }

        let chunk = {
            let mut chunker = self.chunker.lock();
            chunker.next_chunk()
        };

        match chunk {
            Some(text_chunk) => {
                let audio = self.synthesize_chunk(&text_chunk)?;

                if let Some(&last_idx) = text_chunk.word_indices.last() {
                    *self.current_word.lock() = last_idx + 1;
                }

                Ok(Some(TtsEvent::Audio {
                    samples: audio.into(),
                    text: text_chunk.text,
                    word_indices: text_chunk.word_indices,
                    is_final: text_chunk.is_final,
                }))
            },
            None => {
                *self.synthesizing.lock() = false;
                Ok(Some(TtsEvent::Complete))
            },
        }
    }

    /// Synthesize a single chunk
    ///
    /// P0-1 FIX: Now routes to the configured backend if available
    #[cfg(feature = "onnx")]
    fn synthesize_chunk(&self, chunk: &TextChunk) -> Result<Vec<f32>, PipelineError> {
        // P0-1 FIX: Use backend if available (preferred path)
        if let Some(ref backend) = self.backend {
            // Backend synthesis is async, but we're in a sync context
            // Use block_in_place to run async code
            let text = chunk.text.clone();
            let backend = backend.clone();

            let audio = std::thread::scope(|_| {
                tokio::runtime::Handle::try_current()
                    .map(|handle| handle.block_on(backend.synthesize(&text)))
                    .unwrap_or_else(|_| {
                        // No runtime, use blocking
                        Err(PipelineError::Tts("No tokio runtime available".to_string()))
                    })
            })?;

            return Ok(audio);
        }

        // Legacy ONNX path: If no backend but ONNX session exists, use it
        let session = match &self.session {
            Some(s) => s,
            None => {
                // Return silence of appropriate length (sample_rate samples per second)
                let duration_samples = chunk.text.len() * (self.config.sample_rate as usize / 20); // ~50ms per char
                return Ok(vec![0.0f32; duration_samples]);
            },
        };

        let text_ids: Vec<i64> = chunk.text.chars().map(|c| c as i64).collect();

        let input = Array2::from_shape_vec((1, text_ids.len()), text_ids)
            .map_err(|e| PipelineError::Tts(e.to_string()))?;

        let input_lengths = Array2::from_shape_vec((1, 1), vec![chunk.text.len() as i64])
            .map_err(|e| PipelineError::Tts(e.to_string()))?;

        let scales = Array2::from_shape_vec((1, 3), vec![0.667, self.config.speaking_rate, 0.8])
            .map_err(|e| PipelineError::Tts(e.to_string()))?;

        let outputs = session
            .run(
                ort::inputs![
                    "input" => input.view(),
                    "input_lengths" => input_lengths.view(),
                    "scales" => scales.view(),
                ]
                .map_err(|e| PipelineError::Model(e.to_string()))?,
            )
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let audio = outputs
            .get("output")
            .ok_or_else(|| PipelineError::Model("Missing output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        Ok(audio.view().iter().copied().collect())
    }

    /// Synthesize a single chunk (stub when ONNX disabled)
    ///
    /// P0-1 FIX: Now routes to the configured backend if available
    #[cfg(not(feature = "onnx"))]
    fn synthesize_chunk(&self, chunk: &TextChunk) -> Result<Vec<f32>, PipelineError> {
        // P0-1 FIX: Use backend if available
        if let Some(ref backend) = self.backend {
            let text = chunk.text.clone();
            let backend = backend.clone();

            let audio = std::thread::scope(|_| {
                tokio::runtime::Handle::try_current()
                    .map(|handle| handle.block_on(backend.synthesize(&text)))
                    .unwrap_or_else(|_| {
                        Err(PipelineError::Tts("No tokio runtime available".to_string()))
                    })
            })?;

            return Ok(audio);
        }

        // Return silence of appropriate length (22050 samples per second)
        let duration_samples = chunk.text.len() * 2000; // ~50ms per char
        Ok(vec![0.0f32; duration_samples])
    }

    /// Request barge-in (stop synthesis)
    pub fn barge_in(&self) {
        *self.barge_in.lock() = true;
    }

    /// Check if currently synthesizing
    pub fn is_synthesizing(&self) -> bool {
        *self.synthesizing.lock()
    }

    /// Get current word index
    pub fn current_word_index(&self) -> usize {
        *self.current_word.lock()
    }

    /// Add more text (for streaming input)
    pub fn add_text(&self, text: &str) {
        let mut chunker = self.chunker.lock();
        chunker.add_text(text);
    }

    /// Finalize text input
    pub fn finalize_text(&self) {
        let mut chunker = self.chunker.lock();
        chunker.finalize();
    }

    /// Reset TTS state
    pub fn reset(&self) {
        let mut chunker = self.chunker.lock();
        chunker.reset();
        *self.synthesizing.lock() = false;
        *self.barge_in.lock() = false;
        *self.current_word.lock() = 0;
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

#[async_trait::async_trait]
impl TtsBackend for StreamingTts {
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, PipelineError> {
        let chunk = TextChunk {
            text: text.to_string(),
            word_indices: vec![0],
            is_final: true,
            can_pause: true,
        };
        self.synthesize_chunk(&chunk)
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ============================================================================
// P0-1 FIX: Helper functions
// ============================================================================

/// Load reference audio from a WAV file
///
/// Returns the audio samples as f32 normalized to [-1.0, 1.0]
fn load_reference_audio(path: &std::path::Path) -> Result<Vec<f32>, PipelineError> {
    use hound::WavReader;

    let reader = WavReader::open(path)
        .map_err(|e| PipelineError::Audio(format!("Failed to open reference audio: {}", e)))?;

    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_val)
                .collect()
        },
    };

    // If stereo, convert to mono by averaging channels
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    tracing::debug!(
        "Loaded reference audio: {} samples at {} Hz",
        samples.len(),
        spec.sample_rate
    );

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_config_default() {
        let config = TtsConfig::default();
        assert_eq!(config.engine, TtsEngine::Piper);
        assert_eq!(config.speaking_rate, 1.0);
    }

    #[test]
    fn test_tts_config_indicf5() {
        let config = TtsConfig::indicf5("/path/to/model");
        assert_eq!(config.engine, TtsEngine::IndicF5);
        assert_eq!(config.sample_rate, 24000);
        assert!(config.model_path.is_some());
    }

    #[test]
    fn test_barge_in() {
        let tts = StreamingTts::simple(TtsConfig::default());
        let (tx, _rx) = mpsc::channel(10);

        tts.start("Hello world", tx);
        assert!(tts.is_synthesizing());

        tts.barge_in();
        let event = tts.process_next().unwrap();
        assert!(matches!(event, Some(TtsEvent::BargedIn { .. })));
    }

    #[test]
    fn test_reset() {
        let tts = StreamingTts::simple(TtsConfig::default());
        let (tx, _rx) = mpsc::channel(10);

        tts.start("Hello", tx);
        tts.reset();

        assert!(!tts.is_synthesizing());
    }
}
