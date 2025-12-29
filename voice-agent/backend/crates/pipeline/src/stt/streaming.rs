//! Streaming Speech-to-Text Engine
//!
//! Supports chunked audio processing with partial results.

use std::path::Path;
use parking_lot::Mutex;

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};

use super::decoder::{EnhancedDecoder, DecoderConfig};
use super::SttBackend;
use crate::PipelineError;
use voice_agent_core::{TranscriptResult, WordTimestamp, SampleRate};

/// STT engine selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttEngine {
    /// Whisper (multilingual, accurate)
    Whisper,
    /// IndicConformer (optimized for Indian languages)
    IndicConformer,
    /// Wav2Vec2 (general purpose)
    Wav2Vec2,
}

/// STT configuration
#[derive(Debug, Clone)]
pub struct SttConfig {
    /// Engine to use
    pub engine: SttEngine,
    /// Sample rate (must match model)
    pub sample_rate: SampleRate,
    /// Chunk size in milliseconds
    pub chunk_ms: u32,
    /// Language hint
    pub language: Option<String>,
    /// Enable partial results
    pub enable_partials: bool,
    /// Partial emission interval (frames)
    pub partial_interval: usize,
    /// Decoder configuration
    pub decoder: DecoderConfig,
    /// Model directory for vocab loading
    pub model_dir: Option<std::path::PathBuf>,
    /// Domain vocabulary file (entity boosting)
    pub domain_vocab_path: Option<std::path::PathBuf>,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            engine: SttEngine::Whisper,
            sample_rate: SampleRate::Hz16000,
            chunk_ms: 100,
            language: Some("hi".to_string()),
            enable_partials: true,
            partial_interval: 10,
            decoder: DecoderConfig::default(),
            model_dir: None,
            domain_vocab_path: None,
        }
    }
}

/// Streaming STT processor
pub struct StreamingStt {
    #[cfg(feature = "onnx")]
    session: Session,
    config: SttConfig,
    decoder: EnhancedDecoder,
    /// Audio buffer for chunking
    audio_buffer: Mutex<Vec<f32>>,
    /// Frame counter for partial emission
    frame_count: Mutex<usize>,
    /// Current partial result
    current_partial: Mutex<Option<TranscriptResult>>,
    /// Start timestamp
    start_time_ms: Mutex<u64>,
    /// Words detected
    words: Mutex<Vec<WordTimestamp>>,
}

impl StreamingStt {
    /// Create a new streaming STT
    #[cfg(feature = "onnx")]
    pub fn new(model_path: impl AsRef<Path>, config: SttConfig) -> Result<Self, PipelineError> {
        let session = Session::builder()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let vocab = Self::load_vocab(&config.engine)?;
        let decoder = EnhancedDecoder::new(vocab, config.decoder.clone());

        Ok(Self {
            session,
            config,
            decoder,
            audio_buffer: Mutex::new(Vec::new()),
            frame_count: Mutex::new(0),
            current_partial: Mutex::new(None),
            start_time_ms: Mutex::new(0),
            words: Mutex::new(Vec::new()),
        })
    }

    /// Create a new streaming STT (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn new(_model_path: impl AsRef<Path>, config: SttConfig) -> Result<Self, PipelineError> {
        Ok(Self::simple(config))
    }

    /// Create a simple STT for testing (only available when ONNX is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn simple(config: SttConfig) -> Self {
        Self {
            config: config.clone(),
            decoder: EnhancedDecoder::simple(config.decoder),
            audio_buffer: Mutex::new(Vec::new()),
            frame_count: Mutex::new(0),
            current_partial: Mutex::new(None),
            start_time_ms: Mutex::new(0),
            words: Mutex::new(Vec::new()),
        }
    }

    /// Create a simple STT for testing (ONNX enabled - returns error)
    #[cfg(feature = "onnx")]
    pub fn simple(_config: SttConfig) -> Self {
        panic!("StreamingStt::simple() is not available when ONNX feature is enabled. Use new() instead.")
    }

    /// Load vocabulary for engine
    #[allow(dead_code)]
    fn load_vocab(engine: &SttEngine) -> Result<Vec<String>, PipelineError> {
        // Use the proper vocabulary loader
        super::vocab::load_vocabulary(engine, None)
            .map(|v| v.into_tokens())
    }

    /// Load vocabulary from model directory
    #[allow(dead_code)]
    fn load_vocab_from_dir(engine: &SttEngine, model_dir: &std::path::Path) -> Result<Vec<String>, PipelineError> {
        super::vocab::load_vocabulary(engine, Some(model_dir))
            .map(|v| v.into_tokens())
    }

    /// Get chunk size in samples
    fn chunk_samples(&self) -> usize {
        self.config.sample_rate.as_u32() as usize * self.config.chunk_ms as usize / 1000
    }

    /// Process an audio chunk
    pub fn process(&self, audio: &[f32]) -> Result<Option<TranscriptResult>, PipelineError> {
        let mut buffer = self.audio_buffer.lock();
        buffer.extend_from_slice(audio);

        let chunk_size = self.chunk_samples();
        if buffer.len() < chunk_size {
            return Ok(None);
        }

        while buffer.len() >= chunk_size {
            let chunk: Vec<f32> = buffer.drain(..chunk_size).collect();
            self.process_chunk_internal(&chunk)?;
        }

        if self.config.enable_partials {
            let mut frame_count = self.frame_count.lock();
            if *frame_count >= self.config.partial_interval {
                *frame_count = 0;
                return Ok(self.get_partial());
            }
        }

        Ok(None)
    }

    /// Process a single chunk
    #[cfg(feature = "onnx")]
    fn process_chunk_internal(&self, chunk: &[f32]) -> Result<(), PipelineError> {
        let input = Array2::from_shape_vec(
            (1, chunk.len()),
            chunk.to_vec(),
        ).map_err(|e| PipelineError::Stt(e.to_string()))?;

        let outputs = self.session.run(ort::inputs![
            "audio" => input.view(),
        ].map_err(|e| PipelineError::Model(e.to_string()))?)
        .map_err(|e| PipelineError::Model(e.to_string()))?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| PipelineError::Model("Missing logits output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let logits_view = logits.view();
        let shape = logits_view.shape();

        if shape.len() >= 2 {
            let n_frames = shape[1];
            let vocab_size = shape[2];

            for frame_idx in 0..n_frames {
                let frame_logits: Vec<f32> = (0..vocab_size)
                    .map(|v| logits_view[[0, frame_idx, v]])
                    .collect();

                if let Some(partial_text) = self.decoder.process_frame(&frame_logits)? {
                    self.add_word(&partial_text);
                }
            }
        }

        *self.frame_count.lock() += 1;
        Ok(())
    }

    /// Process a single chunk (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    fn process_chunk_internal(&self, _chunk: &[f32]) -> Result<(), PipelineError> {
        *self.frame_count.lock() += 1;
        Ok(())
    }

    /// Add a word to the word list
    #[allow(dead_code)]
    fn add_word(&self, word: &str) {
        let mut words = self.words.lock();
        let start_ms = *self.start_time_ms.lock();

        let total_chars: usize = words.iter().map(|w| w.word.len()).sum();
        let char_ms = 50;

        let word_start = start_ms + (total_chars * char_ms) as u64;
        let word_end = word_start + (word.len() * char_ms) as u64;

        words.push(WordTimestamp {
            word: word.trim().to_string(),
            start_ms: word_start,
            end_ms: word_end,
            confidence: 0.9,
        });
    }

    /// Get current partial result
    fn get_partial(&self) -> Option<TranscriptResult> {
        let text = self.decoder.current_best();
        if text.is_empty() {
            return None;
        }

        let words = self.words.lock().clone();
        let start_ms = *self.start_time_ms.lock();
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(start_ms);

        Some(TranscriptResult {
            text,
            is_final: false,
            confidence: 0.8,
            start_time_ms: start_ms,
            end_time_ms: end_ms,
            language: self.config.language.clone(),
            words,
        })
    }

    /// Finalize and get final result
    pub fn finalize(&self) -> TranscriptResult {
        let remaining: Vec<f32> = {
            let mut buffer = self.audio_buffer.lock();
            buffer.drain(..).collect()
        };

        if !remaining.is_empty() {
            let chunk_size = self.chunk_samples();
            let mut padded = remaining;
            padded.resize(chunk_size, 0.0);
            let _ = self.process_chunk_internal(&padded);
        }

        let text = self.decoder.finalize();
        let words = self.words.lock().clone();
        let start_ms = *self.start_time_ms.lock();
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(start_ms);

        TranscriptResult {
            text,
            is_final: true,
            confidence: 0.9,
            start_time_ms: start_ms,
            end_time_ms: end_ms,
            language: self.config.language.clone(),
            words,
        }
    }

    /// Reset STT state
    pub fn reset(&self) {
        self.audio_buffer.lock().clear();
        *self.frame_count.lock() = 0;
        *self.current_partial.lock() = None;
        *self.start_time_ms.lock() = 0;
        self.words.lock().clear();
        self.decoder.reset();
    }

    /// Set start time for word timestamps
    pub fn set_start_time(&self, time_ms: u64) {
        *self.start_time_ms.lock() = time_ms;
    }

    /// Add entities to boost in decoder
    pub fn add_entities(&self, entities: impl IntoIterator<Item = impl AsRef<str>>) {
        self.decoder.add_entities(entities);
    }
}

#[async_trait::async_trait]
impl SttBackend for StreamingStt {
    async fn process_chunk(&mut self, audio: &[f32]) -> Result<Option<TranscriptResult>, PipelineError> {
        self.process(audio)
    }

    async fn finalize(&mut self) -> Result<TranscriptResult, PipelineError> {
        Ok(StreamingStt::finalize(self))
    }

    fn reset(&mut self) {
        StreamingStt::reset(self);
    }

    fn partial(&self) -> Option<&TranscriptResult> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_config_default() {
        let config = SttConfig::default();
        assert_eq!(config.engine, SttEngine::Whisper);
        assert!(config.enable_partials);
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_chunk_samples() {
        let stt = StreamingStt::simple(SttConfig::default());
        assert_eq!(stt.chunk_samples(), 1600);
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_reset() {
        let stt = StreamingStt::simple(SttConfig::default());
        stt.reset();
        assert!(stt.audio_buffer.lock().is_empty());
    }
}
