//! IndicConformer STT - Speech-to-Text for Indian Languages
//!
//! Implementation of AI4Bharat's IndicConformer 600M multilingual model.
//! Optimized for Hindi, Marathi, and other Indian languages.
//!
//! Model architecture:
//! - Mel spectrogram preprocessing (80 mel bins, 16kHz)
//! - Conformer encoder (encoder.onnx)
//! - CTC decoder (ctc_decoder.onnx)
//! - Language-specific post-net (joint_post_net_hi.onnx for Hindi)

use std::collections::HashMap;
use std::path::Path;
use parking_lot::Mutex;
use ndarray::Array2;

#[cfg(feature = "onnx")]
use ndarray::Array3;

#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};

use super::decoder::{EnhancedDecoder, DecoderConfig};
use super::vocab::Vocabulary;
use super::SttBackend;
use crate::PipelineError;
use voice_agent_core::{TranscriptResult, WordTimestamp, SampleRate};

/// IndicConformer configuration
#[derive(Debug, Clone)]
pub struct IndicConformerConfig {
    /// Language code (hi, mr, bn, etc.)
    pub language: String,
    /// Sample rate (must be 16000)
    pub sample_rate: SampleRate,
    /// Number of mel frequency bins
    pub n_mels: usize,
    /// FFT window size in samples
    pub n_fft: usize,
    /// Hop length in samples
    pub hop_length: usize,
    /// Window length in samples
    pub win_length: usize,
    /// Chunk size in milliseconds for streaming
    pub chunk_ms: u32,
    /// Enable partial results
    pub enable_partials: bool,
    /// Partial emission interval (frames)
    pub partial_interval: usize,
    /// Decoder configuration
    pub decoder: DecoderConfig,
}

impl Default for IndicConformerConfig {
    fn default() -> Self {
        Self {
            language: "hi".to_string(),
            sample_rate: SampleRate::Hz16000,
            n_mels: 80,
            n_fft: 512,
            hop_length: 160,  // 10ms at 16kHz
            win_length: 400,  // 25ms at 16kHz
            chunk_ms: 100,
            enable_partials: true,
            partial_interval: 10,
            decoder: DecoderConfig::default(),
        }
    }
}

/// Mutable state for IndicConformer
struct IndicConformerState {
    /// Audio buffer for accumulating samples
    audio_buffer: Vec<f32>,
    /// Frame counter for partial emission
    frame_count: usize,
    /// Words detected with timestamps
    words: Vec<WordTimestamp>,
    /// Start timestamp
    start_time_ms: u64,
    /// Previous encoder hidden state for streaming (if using RNN-T)
    encoder_state: Option<Array2<f32>>,
    /// P0 FIX: Running sum of frame confidences for averaging
    confidence_sum: f32,
    /// P0 FIX: Number of frames processed (for averaging)
    confidence_count: usize,
    /// P0 FIX: Per-word confidence accumulator
    current_word_confidence: f32,
    /// P0 FIX: Frame count for current word
    current_word_frames: usize,
    /// P0 FIX: Total audio frames processed (for timestamp calculation)
    total_audio_frames: usize,
}

/// IndicConformer STT implementation
pub struct IndicConformerStt {
    #[cfg(feature = "onnx")]
    encoder_session: Session,
    #[cfg(feature = "onnx")]
    decoder_session: Session,
    #[cfg(feature = "onnx")]
    post_net_session: Option<Session>,

    config: IndicConformerConfig,
    vocabulary: Vocabulary,
    decoder: EnhancedDecoder,
    mel_filterbank: MelFilterbank,
    state: Mutex<IndicConformerState>,
}

impl IndicConformerStt {
    /// Create a new IndicConformer STT from model directory
    ///
    /// Expected directory structure:
    /// - assets/encoder.onnx
    /// - assets/ctc_decoder.onnx
    /// - assets/joint_post_net_{lang}.onnx
    /// - assets/vocab.json
    #[cfg(feature = "onnx")]
    pub fn new(model_dir: impl AsRef<Path>, config: IndicConformerConfig) -> Result<Self, PipelineError> {
        let model_dir = model_dir.as_ref();
        let assets_dir = model_dir.join("assets");

        // Load encoder
        let encoder_path = assets_dir.join("encoder.onnx");
        let encoder_session = Self::load_session(&encoder_path)?;

        // Load CTC decoder
        let decoder_path = assets_dir.join("ctc_decoder.onnx");
        let decoder_session = Self::load_session(&decoder_path)?;

        // Load language-specific post-net (optional)
        let post_net_path = assets_dir.join(format!("joint_post_net_{}.onnx", config.language));
        let post_net_session = if post_net_path.exists() {
            Some(Self::load_session(&post_net_path)?)
        } else {
            None
        };

        // Load vocabulary
        let vocab_path = assets_dir.join("vocab.json");
        let vocabulary = Self::load_vocab(&vocab_path, &config.language)?;

        // Create decoder with vocabulary
        let decoder = EnhancedDecoder::new(vocabulary.clone().into_tokens(), config.decoder.clone());

        // Create mel filterbank
        let mel_filterbank = MelFilterbank::new(
            config.sample_rate.as_u32() as usize,
            config.n_fft,
            config.n_mels,
        );

        Ok(Self {
            encoder_session,
            decoder_session,
            post_net_session,
            config,
            vocabulary,
            decoder,
            mel_filterbank,
            state: Mutex::new(IndicConformerState {
                audio_buffer: Vec::new(),
                frame_count: 0,
                words: Vec::new(),
                start_time_ms: 0,
                encoder_state: None,
                confidence_sum: 0.0,
                confidence_count: 0,
                current_word_confidence: 0.0,
                current_word_frames: 0,
                total_audio_frames: 0,
            }),
        })
    }

    /// Create IndicConformer without ONNX (stub for testing)
    #[cfg(not(feature = "onnx"))]
    pub fn new(_model_dir: impl AsRef<Path>, config: IndicConformerConfig) -> Result<Self, PipelineError> {
        Self::simple(config)
    }

    /// Create a simple stub for testing
    #[cfg(not(feature = "onnx"))]
    pub fn simple(config: IndicConformerConfig) -> Result<Self, PipelineError> {
        let vocabulary = Vocabulary::default_indicconformer();
        let decoder = EnhancedDecoder::simple(config.decoder.clone());
        let mel_filterbank = MelFilterbank::new(
            config.sample_rate.as_u32() as usize,
            config.n_fft,
            config.n_mels,
        );

        Ok(Self {
            config,
            vocabulary,
            decoder,
            mel_filterbank,
            state: Mutex::new(IndicConformerState {
                audio_buffer: Vec::new(),
                frame_count: 0,
                words: Vec::new(),
                start_time_ms: 0,
                encoder_state: None,
                confidence_sum: 0.0,
                confidence_count: 0,
                current_word_confidence: 0.0,
                current_word_frames: 0,
                total_audio_frames: 0,
            }),
        })
    }

    #[cfg(feature = "onnx")]
    fn load_session(path: &Path) -> Result<Session, PipelineError> {
        Session::builder()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| PipelineError::Model(format!("Failed to load {}: {}", path.display(), e)))
    }

    fn load_vocab(path: &Path, language: &str) -> Result<Vocabulary, PipelineError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PipelineError::Io(format!("Failed to read vocab: {}", e)))?;

        let vocab_map: HashMap<String, Vec<String>> = serde_json::from_str(&content)
            .map_err(|e| PipelineError::Stt(format!("Failed to parse vocab: {}", e)))?;

        let tokens = vocab_map.get(language)
            .ok_or_else(|| PipelineError::Stt(format!("Language '{}' not found in vocab", language)))?
            .clone();

        Ok(Vocabulary::from_tokens(tokens))
    }

    /// P0 FIX: Extract confidence from model logits using softmax
    ///
    /// Computes the probability of the predicted token by applying softmax
    /// to the logits and returning the max probability.
    fn extract_confidence_from_logits(logits: &[f32]) -> f32 {
        if logits.is_empty() {
            return 0.5; // Default for empty logits
        }

        // Find max logit for numerical stability (log-sum-exp trick)
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute softmax denominator: sum(exp(logit - max))
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();

        if exp_sum == 0.0 {
            return 0.5; // Avoid division by zero
        }

        // Find the maximum probability (confidence of best prediction)
        let max_prob = logits
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .fold(0.0f32, f32::max);

        // Clamp to valid probability range
        max_prob.clamp(0.0, 1.0)
    }

    /// P0 FIX: Calculate word timestamp based on frame timing
    ///
    /// Uses actual frame-level timing instead of naive character-based heuristic.
    fn calculate_word_timestamp(
        &self,
        word: &str,
        start_frame: usize,
        end_frame: usize,
    ) -> (u64, u64) {
        // Each frame represents hop_length samples at sample_rate Hz
        let frame_duration_ms = (self.config.hop_length as f64 / self.config.sample_rate.as_u32() as f64 * 1000.0) as u64;

        let start_ms = start_frame as u64 * frame_duration_ms;
        let end_ms = end_frame as u64 * frame_duration_ms;

        // Ensure end is at least start + minimum word duration
        let min_word_duration = (word.chars().count() as u64 * 30).max(100); // At least 30ms per char, min 100ms
        let adjusted_end = end_ms.max(start_ms + min_word_duration);

        (start_ms, adjusted_end)
    }

    /// Get chunk size in samples
    fn chunk_samples(&self) -> usize {
        self.config.sample_rate.as_u32() as usize * self.config.chunk_ms as usize / 1000
    }

    /// Process audio samples
    pub fn process(&self, audio: &[f32]) -> Result<Option<TranscriptResult>, PipelineError> {
        let mut state = self.state.lock();
        state.audio_buffer.extend_from_slice(audio);

        let chunk_size = self.chunk_samples();
        if state.audio_buffer.len() < chunk_size {
            return Ok(None);
        }

        // Process full chunks
        while state.audio_buffer.len() >= chunk_size {
            let chunk: Vec<f32> = state.audio_buffer.drain(..chunk_size).collect();
            drop(state);

            self.process_chunk_internal(&chunk)?;

            state = self.state.lock();
        }

        if self.config.enable_partials {
            if state.frame_count >= self.config.partial_interval {
                state.frame_count = 0;
                drop(state);
                return Ok(self.get_partial());
            }
        }

        Ok(None)
    }

    /// Process a single audio chunk
    #[cfg(feature = "onnx")]
    fn process_chunk_internal(&self, audio: &[f32]) -> Result<(), PipelineError> {
        // Extract mel spectrogram
        let mel = self.mel_filterbank.extract(audio);

        // Prepare input tensor [batch, time, n_mels]
        let mel_input = Array3::from_shape_vec(
            (1, mel.len() / self.config.n_mels, self.config.n_mels),
            mel,
        ).map_err(|e| PipelineError::Stt(format!("Failed to reshape mel: {}", e)))?;

        // Run encoder
        let encoder_outputs = self.encoder_session.run(ort::inputs![
            "audio_signal" => mel_input.view(),
        ].map_err(|e| PipelineError::Model(e.to_string()))?)
        .map_err(|e| PipelineError::Model(format!("Encoder failed: {}", e)))?;

        // Get encoder output
        let encoded = encoder_outputs
            .get("encoded")
            .ok_or_else(|| PipelineError::Model("Missing encoded output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        // Run CTC decoder
        let decoder_outputs = self.decoder_session.run(ort::inputs![
            "encoder_output" => encoded.view(),
        ].map_err(|e| PipelineError::Model(e.to_string()))?)
        .map_err(|e| PipelineError::Model(format!("Decoder failed: {}", e)))?;

        // Get logits
        let logits = decoder_outputs
            .get("logits")
            .or_else(|| decoder_outputs.get("log_probs"))
            .ok_or_else(|| PipelineError::Model("Missing logits output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let logits_view = logits.view();
        let shape = logits_view.shape();

        // Process each frame through enhanced decoder
        if shape.len() >= 2 {
            let n_frames = shape[1];
            let vocab_size = if shape.len() > 2 { shape[2] } else { shape[1] };

            for frame_idx in 0..n_frames {
                let frame_logits: Vec<f32> = if shape.len() > 2 {
                    (0..vocab_size).map(|v| logits_view[[0, frame_idx, v]]).collect()
                } else {
                    (0..vocab_size).map(|v| logits_view[[frame_idx, v]]).collect()
                };

                // P0 FIX: Extract actual confidence from logits
                let frame_confidence = Self::extract_confidence_from_logits(&frame_logits);

                // Update running confidence stats
                {
                    let mut state = self.state.lock();
                    state.confidence_sum += frame_confidence;
                    state.confidence_count += 1;
                    state.current_word_confidence += frame_confidence;
                    state.current_word_frames += 1;
                    state.total_audio_frames += 1;
                }

                if let Some(word) = self.decoder.process_frame(&frame_logits)? {
                    self.add_word_with_confidence(&word, frame_confidence);
                }
            }
        }

        let mut state = self.state.lock();
        state.frame_count += 1;
        Ok(())
    }

    /// Process a single audio chunk (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    fn process_chunk_internal(&self, _audio: &[f32]) -> Result<(), PipelineError> {
        let mut state = self.state.lock();
        state.frame_count += 1;
        Ok(())
    }

    /// P0 FIX: Add a detected word with actual confidence from model
    fn add_word_with_confidence(&self, word: &str, last_frame_confidence: f32) {
        let mut state = self.state.lock();

        // P0 FIX: Calculate word confidence as average of frames for this word
        let word_confidence = if state.current_word_frames > 0 {
            (state.current_word_confidence / state.current_word_frames as f32).clamp(0.0, 1.0)
        } else {
            last_frame_confidence.clamp(0.0, 1.0)
        };

        // P0 FIX: Calculate timestamps based on actual frame positions
        let frame_duration_ms = (self.config.hop_length as f64 / self.config.sample_rate.as_u32() as f64 * 1000.0) as u64;

        // Calculate word boundaries based on frames processed
        let word_end_frame = state.total_audio_frames;
        let word_start_frame = word_end_frame.saturating_sub(state.current_word_frames);

        let word_start = state.start_time_ms + (word_start_frame as u64 * frame_duration_ms);
        let word_end = state.start_time_ms + (word_end_frame as u64 * frame_duration_ms);

        // Ensure minimum duration
        let min_duration = (word.chars().count() as u64 * 30).max(100);
        let adjusted_end = word_end.max(word_start + min_duration);

        state.words.push(WordTimestamp {
            word: word.trim().to_string(),
            start_ms: word_start,
            end_ms: adjusted_end,
            confidence: word_confidence,
        });

        // Reset per-word accumulators
        state.current_word_confidence = 0.0;
        state.current_word_frames = 0;
    }

    /// Legacy add_word for backward compatibility (uses default confidence)
    #[allow(dead_code)]
    fn add_word(&self, word: &str) {
        self.add_word_with_confidence(word, 0.5)
    }

    /// Get current partial result
    fn get_partial(&self) -> Option<TranscriptResult> {
        let text = self.decoder.current_best();
        if text.is_empty() {
            return None;
        }

        let state = self.state.lock();
        let words = state.words.clone();
        let start_ms = state.start_time_ms;
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(start_ms);

        // P0 FIX: Calculate actual confidence as average of all frames
        let confidence = if state.confidence_count > 0 {
            (state.confidence_sum / state.confidence_count as f32).clamp(0.0, 1.0)
        } else {
            // Fallback: average word confidences if available
            if !words.is_empty() {
                words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
            } else {
                0.5 // Neutral confidence when no data
            }
        };

        Some(TranscriptResult {
            text,
            is_final: false,
            confidence,
            start_time_ms: start_ms,
            end_time_ms: end_ms,
            language: Some(self.config.language.clone()),
            words,
        })
    }

    /// Finalize and get final result
    pub fn finalize(&self) -> TranscriptResult {
        // Process remaining audio
        let remaining: Vec<f32> = {
            let mut state = self.state.lock();
            state.audio_buffer.drain(..).collect()
        };

        if !remaining.is_empty() {
            let chunk_size = self.chunk_samples();
            let mut padded = remaining;
            padded.resize(chunk_size, 0.0);
            let _ = self.process_chunk_internal(&padded);
        }

        let text = self.decoder.finalize();
        let state = self.state.lock();
        let words = state.words.clone();
        let start_ms = state.start_time_ms;
        let end_ms = words.last().map(|w| w.end_ms).unwrap_or(start_ms);

        // P0 FIX: Calculate actual final confidence as average of all frames
        let confidence = if state.confidence_count > 0 {
            (state.confidence_sum / state.confidence_count as f32).clamp(0.0, 1.0)
        } else {
            // Fallback: average word confidences if available
            if !words.is_empty() {
                words.iter().map(|w| w.confidence).sum::<f32>() / words.len() as f32
            } else {
                0.5 // Neutral confidence when no data
            }
        };

        TranscriptResult {
            text,
            is_final: true,
            confidence,
            start_time_ms: start_ms,
            end_time_ms: end_ms,
            language: Some(self.config.language.clone()),
            words,
        }
    }

    /// Reset STT state
    pub fn reset(&self) {
        let mut state = self.state.lock();
        state.audio_buffer.clear();
        state.frame_count = 0;
        state.words.clear();
        state.start_time_ms = 0;
        state.encoder_state = None;
        // P0 FIX: Reset confidence tracking state
        state.confidence_sum = 0.0;
        state.confidence_count = 0;
        state.current_word_confidence = 0.0;
        state.current_word_frames = 0;
        state.total_audio_frames = 0;
        self.decoder.reset();
    }

    /// Set start time for word timestamps
    pub fn set_start_time(&self, time_ms: u64) {
        self.state.lock().start_time_ms = time_ms;
    }

    /// Add entities to boost in decoder
    pub fn add_entities(&self, entities: impl IntoIterator<Item = impl AsRef<str>>) {
        self.decoder.add_entities(entities);
    }

    /// Get vocabulary
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Get supported languages
    pub fn supported_languages() -> Vec<&'static str> {
        vec![
            "as",   // Assamese
            "bn",   // Bengali
            "brx",  // Bodo
            "doi",  // Dogri
            "gu",   // Gujarati
            "hi",   // Hindi
            "kn",   // Kannada
            "kok",  // Konkani
            "ks",   // Kashmiri
            "mai",  // Maithili
            "ml",   // Malayalam
            "mni",  // Manipuri
            "mr",   // Marathi
            "ne",   // Nepali
            "or",   // Odia
            "pa",   // Punjabi
            "sa",   // Sanskrit
            "sat",  // Santali
            "sd",   // Sindhi
            "ta",   // Tamil
            "te",   // Telugu
            "ur",   // Urdu
        ]
    }
}

#[async_trait::async_trait]
impl SttBackend for IndicConformerStt {
    async fn process_chunk(&mut self, audio: &[f32]) -> Result<Option<TranscriptResult>, PipelineError> {
        self.process(audio)
    }

    async fn finalize(&mut self) -> Result<TranscriptResult, PipelineError> {
        Ok(IndicConformerStt::finalize(self))
    }

    fn reset(&mut self) {
        IndicConformerStt::reset(self);
    }

    fn partial(&self) -> Option<&TranscriptResult> {
        None // Partials are returned through process()
    }
}

/// Mel filterbank for audio preprocessing with sliding-window FFT
///
/// Uses realfft for efficient real-signal FFT computation.
/// Supports streaming mode with audio buffer for sliding window.
pub struct MelFilterbank {
    n_fft: usize,
    n_mels: usize,
    hop_length: usize,
    mel_filters: Vec<Vec<f32>>,
    hann_window: Vec<f32>,
    /// Reusable FFT planner
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    /// Sliding window buffer for streaming
    audio_buffer: parking_lot::Mutex<Vec<f32>>,
}

impl MelFilterbank {
    pub fn new(sample_rate: usize, n_fft: usize, n_mels: usize) -> Self {
        // Create Hann window
        let hann_window: Vec<f32> = (0..n_fft)
            .map(|i| {
                let x = std::f32::consts::PI * i as f32 / (n_fft - 1) as f32;
                0.5 * (1.0 - (2.0 * x).cos())
            })
            .collect();

        // Create mel filterbank
        let mel_filters = Self::create_mel_filters(sample_rate, n_fft, n_mels);

        // Create FFT planner
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // 10ms hop at 16kHz
        let hop_length = sample_rate / 100;

        Self {
            n_fft,
            n_mels,
            hop_length,
            mel_filters,
            hann_window,
            fft,
            audio_buffer: parking_lot::Mutex::new(Vec::new()),
        }
    }

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn create_mel_filters(sample_rate: usize, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
        let fmin = 0.0;
        let fmax = sample_rate as f32 / 2.0;

        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // Mel points
        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Hz points
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft + 1) as f32 * hz / sample_rate as f32).floor() as usize)
            .collect();

        // Create triangular filters
        let n_bins = n_fft / 2 + 1;
        let mut filters = vec![vec![0.0f32; n_bins]; n_mels];

        for i in 0..n_mels {
            let start = bin_points[i];
            let center = bin_points[i + 1];
            let end = bin_points[i + 2];

            // Rising slope
            for j in start..center {
                if center > start && j < n_bins {
                    filters[i][j] = (j - start) as f32 / (center - start) as f32;
                }
            }

            // Falling slope
            for j in center..end {
                if end > center && j < n_bins {
                    filters[i][j] = (end - j) as f32 / (end - center) as f32;
                }
            }
        }

        filters
    }

    /// Compute FFT magnitude spectrum for a single frame using realfft
    fn compute_fft_frame(&self, windowed: &mut [f32]) -> Vec<f32> {
        use realfft::num_complex::Complex;

        let n_bins = self.n_fft / 2 + 1;
        let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); n_bins];

        // Perform FFT
        if self.fft.process(windowed, &mut spectrum).is_ok() {
            spectrum.iter().map(|c| c.norm()).collect()
        } else {
            // Fallback to zeros on error
            vec![0.0f32; n_bins]
        }
    }

    /// Extract mel spectrogram from audio (batch mode)
    pub fn extract(&self, audio: &[f32]) -> Vec<f32> {
        let n_frames = (audio.len().saturating_sub(self.n_fft)) / self.hop_length + 1;

        if n_frames == 0 {
            return vec![0.0; self.n_mels];
        }

        let mut mel_spec = Vec::with_capacity(n_frames * self.n_mels);

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            let end = (start + self.n_fft).min(audio.len());

            // Apply window
            let mut windowed = vec![0.0f32; self.n_fft];
            for (i, sample) in audio[start..end].iter().enumerate() {
                windowed[i] = sample * self.hann_window[i];
            }

            // Compute FFT magnitudes
            let magnitudes = self.compute_fft_frame(&mut windowed);

            // Apply mel filterbank
            for filter in &self.mel_filters {
                let mut mel_energy = 0.0f32;
                for (j, &mag) in magnitudes.iter().enumerate() {
                    mel_energy += mag * filter[j];
                }
                // Log mel
                mel_spec.push((mel_energy + 1e-10).ln());
            }
        }

        mel_spec
    }

    /// Streaming mel extraction - add audio and get new mel frames
    ///
    /// Returns only the NEW mel frames since last call.
    /// Maintains internal buffer for sliding window.
    pub fn extract_streaming(&self, audio: &[f32]) -> Vec<f32> {
        let mut buffer = self.audio_buffer.lock();
        buffer.extend_from_slice(audio);

        let mut mel_frames = Vec::new();

        // Process complete frames
        while buffer.len() >= self.n_fft {
            // Apply window to current frame
            let mut windowed = vec![0.0f32; self.n_fft];
            for i in 0..self.n_fft {
                windowed[i] = buffer[i] * self.hann_window[i];
            }

            // Compute FFT magnitudes
            let magnitudes = self.compute_fft_frame(&mut windowed);

            // Apply mel filterbank
            for filter in &self.mel_filters {
                let mut mel_energy = 0.0f32;
                for (j, &mag) in magnitudes.iter().enumerate() {
                    mel_energy += mag * filter[j];
                }
                mel_frames.push((mel_energy + 1e-10).ln());
            }

            // Slide window by hop_length
            buffer.drain(..self.hop_length);
        }

        mel_frames
    }

    /// Reset streaming buffer
    pub fn reset_streaming(&self) {
        self.audio_buffer.lock().clear();
    }

    /// Get pending samples in buffer
    pub fn pending_samples(&self) -> usize {
        self.audio_buffer.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicconformer_config_default() {
        let config = IndicConformerConfig::default();
        assert_eq!(config.language, "hi");
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.sample_rate, SampleRate::Hz16000);
    }

    #[test]
    fn test_mel_filterbank() {
        let mel = MelFilterbank::new(16000, 512, 80);
        assert_eq!(mel.mel_filters.len(), 80);
        assert_eq!(mel.hann_window.len(), 512);
    }

    #[test]
    fn test_mel_extract() {
        let mel = MelFilterbank::new(16000, 512, 80);

        // Generate 100ms of audio (1600 samples at 16kHz)
        let audio: Vec<f32> = (0..1600)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();

        let features = mel.extract(&audio);

        // Should have multiple frames, each with 80 mel bins
        assert!(features.len() >= 80);
        assert_eq!(features.len() % 80, 0);
    }

    #[test]
    fn test_supported_languages() {
        let languages = IndicConformerStt::supported_languages();
        assert!(languages.contains(&"hi"));
        assert!(languages.contains(&"mr"));
        assert!(languages.contains(&"bn"));
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_indicconformer_simple() {
        let config = IndicConformerConfig::default();
        let stt = IndicConformerStt::simple(config).unwrap();
        // Default vocabulary has 8000 tokens (placeholder)
        assert_eq!(stt.vocabulary().len(), 8000);
    }
}
