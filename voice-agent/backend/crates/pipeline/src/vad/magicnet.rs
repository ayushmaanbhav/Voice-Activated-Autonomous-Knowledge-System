//! MagicNet-inspired Voice Activity Detector
//!
//! Based on "MagicNet: Semi-supervised Voice Activity Detection"
//! Features:
//! - 10ms frame size for low latency (<15ms detection)
//! - Causal depth-separable convolutions (no future lookahead)
//! - GRU for temporal modeling
//! - Stateful for streaming

use async_trait::async_trait;
use futures::Stream;
use parking_lot::Mutex;
use std::path::Path;
use std::pin::Pin;
use voice_agent_core::AudioFrame;
// P0 FIX: Import core VAD trait with alias to avoid naming collision
use voice_agent_core::traits::{
    VADConfig as CoreVADConfig, VADEvent as CoreVADEvent, VADState as CoreVADState,
    VoiceActivityDetector as CoreVadTrait,
};

use crate::PipelineError;

#[cfg(feature = "onnx")]
use ndarray::{Array2, Array3};

#[cfg(not(feature = "onnx"))]
use ndarray::Array2;

#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};

/// VAD configuration
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Speech probability threshold (0.0 - 1.0)
    pub threshold: f32,
    /// Frame size in milliseconds
    pub frame_ms: u32,
    /// Minimum speech frames to confirm speech
    pub min_speech_frames: usize,
    /// Minimum silence frames to confirm silence
    pub min_silence_frames: usize,
    /// Number of mel filterbank bins
    pub n_mels: usize,
    /// Sample rate (must be 16kHz)
    pub sample_rate: u32,
    /// GRU hidden size
    pub gru_hidden_size: usize,
    /// Energy floor in dB for quick silence detection
    pub energy_floor_db: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            frame_ms: 10,
            min_speech_frames: 25,  // 250ms
            min_silence_frames: 30, // 300ms
            n_mels: 40,
            sample_rate: 16000,
            gru_hidden_size: 64,
            energy_floor_db: -50.0,
        }
    }
}

/// VAD state machine states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VadState {
    /// No speech detected
    #[default]
    Silence,
    /// Potential speech start (accumulating)
    SpeechStart,
    /// Active speech confirmed
    Speech,
    /// Potential speech end (accumulating silence)
    SpeechEnd,
}

/// VAD processing result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadResult {
    /// Silence detected
    Silence,
    /// Potential speech start (below threshold duration)
    PotentialSpeechStart,
    /// Speech confirmed (above threshold duration)
    SpeechConfirmed,
    /// Speech continuing
    SpeechContinue,
    /// Potential speech end (accumulating silence)
    PotentialSpeechEnd,
    /// Speech ended (silence threshold met)
    SpeechEnd,
}

/// Mutable state for VAD (consolidated into single struct for single lock)
///
/// P0 FIX: Previously 4 separate Mutex fields caused unnecessary lock contention
/// at 100 frames/sec. Now consolidated into single lock for 75% reduction in
/// lock operations per frame.
struct VadMutableState {
    /// GRU hidden state
    gru_state: Array2<f32>,
    /// Current VAD state
    state: VadState,
    /// Accumulated speech frames
    speech_frames: usize,
    /// Accumulated silence frames
    silence_frames: usize,
}

/// MagicNet-inspired Voice Activity Detector
pub struct VoiceActivityDetector {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(feature = "onnx")]
    mel_filterbank: MelFilterbank,
    config: VadConfig,
    /// P0 FIX: Single lock for all mutable state (was 4 separate locks)
    mutable: Mutex<VadMutableState>,
}

impl VoiceActivityDetector {
    /// Create a new VAD from ONNX model
    #[cfg(feature = "onnx")]
    pub fn new(model_path: impl AsRef<Path>, config: VadConfig) -> Result<Self, PipelineError> {
        let session = Session::builder()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let gru_state = Array2::zeros((1, config.gru_hidden_size));
        let frame_samples = config.sample_rate as usize * config.frame_ms as usize / 1000;
        let mel_filterbank = MelFilterbank::new(config.sample_rate, frame_samples, config.n_mels)?;

        Ok(Self {
            session,
            mel_filterbank,
            config,
            mutable: Mutex::new(VadMutableState {
                gru_state,
                state: VadState::Silence,
                speech_frames: 0,
                silence_frames: 0,
            }),
        })
    }

    /// Create a new VAD (without ONNX - uses energy-based detection)
    #[cfg(not(feature = "onnx"))]
    pub fn new(_model_path: impl AsRef<Path>, config: VadConfig) -> Result<Self, PipelineError> {
        Self::simple(config)
    }

    /// Create a simple energy-based VAD (no model required)
    #[cfg(feature = "onnx")]
    pub fn simple(_config: VadConfig) -> Result<Self, PipelineError> {
        Err(PipelineError::Model(
            "Simple VAD requires ONNX feature disabled".to_string(),
        ))
    }

    /// Create a simple energy-based VAD (no model required)
    #[cfg(not(feature = "onnx"))]
    pub fn simple(config: VadConfig) -> Result<Self, PipelineError> {
        Ok(Self {
            config,
            mutable: Mutex::new(VadMutableState {
                gru_state: Array2::zeros((1, 64)),
                state: VadState::Silence,
                speech_frames: 0,
                silence_frames: 0,
            }),
        })
    }

    /// Process a 10ms audio frame
    ///
    /// Returns (VadState, probability, VadResult) where VadResult provides
    /// detailed transition information (e.g., SpeechConfirmed, PotentialSpeechEnd).
    ///
    /// P0 FIX: Now uses single lock for all mutable state access.
    /// P1 FIX: Now returns VadResult for detailed transition info.
    pub fn process_frame(
        &self,
        frame: &mut AudioFrame,
    ) -> Result<(VadState, f32, VadResult), PipelineError> {
        // Quick energy check for obvious silence (no lock needed)
        if frame.energy_db < self.config.energy_floor_db {
            frame.vad_probability = Some(0.0);
            frame.is_speech = false;

            // Single lock for state update
            let mut state = self.mutable.lock();
            return self.update_state_inner(&mut state, false, 0.0);
        }

        // Compute speech probability (acquires lock internally for GRU state)
        let speech_prob = self.compute_probability(frame)?;

        frame.vad_probability = Some(speech_prob);
        let is_speech = speech_prob >= self.config.threshold;
        frame.is_speech = is_speech;

        // Single lock for state update
        let mut state = self.mutable.lock();
        self.update_state_inner(&mut state, is_speech, speech_prob)
    }

    /// Compute speech probability using ONNX model
    #[cfg(feature = "onnx")]
    fn compute_probability(&self, frame: &AudioFrame) -> Result<f32, PipelineError> {
        let mel_features = self.mel_filterbank.compute(&frame.samples)?;
        self.infer(&mel_features)
    }

    /// Compute speech probability (energy-based fallback)
    #[cfg(not(feature = "onnx"))]
    fn compute_probability(&self, frame: &AudioFrame) -> Result<f32, PipelineError> {
        // Simple energy-based VAD
        let energy_threshold = self.config.energy_floor_db + 10.0;
        let prob = if frame.energy_db > energy_threshold {
            ((frame.energy_db - energy_threshold) / 30.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Ok(prob)
    }

    /// Run ONNX inference
    ///
    /// P0 FIX: Now uses single consolidated lock for GRU state.
    #[cfg(feature = "onnx")]
    fn infer(&self, mel_features: &[f32]) -> Result<f32, PipelineError> {
        use ndarray::ArrayView2;

        let mut state = self.mutable.lock();

        let input = Array3::from_shape_vec((1, 1, self.config.n_mels), mel_features.to_vec())
            .map_err(|e| PipelineError::Vad(e.to_string()))?;

        let outputs = self
            .session
            .run(
                ort::inputs![
                    "mel_input" => input.view(),
                    "gru_state_in" => state.gru_state.view(),
                ]
                .map_err(|e| PipelineError::Model(e.to_string()))?,
            )
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let speech_prob: f32 = outputs
            .get("speech_prob")
            .ok_or_else(|| PipelineError::Model("Missing speech_prob output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .view()
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0);

        if let Some(new_state) = outputs.get("gru_state_out") {
            let new_state: ArrayView2<f32> = new_state
                .try_extract_tensor()
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            state.gru_state.assign(&new_state);
        }

        Ok(speech_prob)
    }

    /// Update state machine based on detection result (inner implementation)
    ///
    /// P0 FIX: Takes mutable reference to consolidated state struct.
    /// P1 FIX: Now returns VadResult for detailed transition information.
    fn update_state_inner(
        &self,
        state: &mut VadMutableState,
        is_speech: bool,
        probability: f32,
    ) -> Result<(VadState, f32, VadResult), PipelineError> {
        let result = match (state.state, is_speech) {
            (VadState::Silence, true) => {
                state.state = VadState::SpeechStart;
                state.speech_frames = 1;
                state.silence_frames = 0;
                VadResult::PotentialSpeechStart
            },

            (VadState::SpeechStart, true) => {
                state.speech_frames += 1;
                if state.speech_frames >= self.config.min_speech_frames {
                    state.state = VadState::Speech;
                    VadResult::SpeechConfirmed
                } else {
                    VadResult::PotentialSpeechStart
                }
            },

            (VadState::SpeechStart, false) => {
                state.state = VadState::Silence;
                state.speech_frames = 0;
                VadResult::Silence
            },

            (VadState::Speech, true) => {
                state.silence_frames = 0;
                VadResult::SpeechContinue
            },

            (VadState::Speech, false) => {
                state.state = VadState::SpeechEnd;
                state.silence_frames = 1;
                VadResult::PotentialSpeechEnd
            },

            (VadState::SpeechEnd, true) => {
                state.state = VadState::Speech;
                state.silence_frames = 0;
                VadResult::SpeechContinue
            },

            (VadState::SpeechEnd, false) => {
                state.silence_frames += 1;
                if state.silence_frames >= self.config.min_silence_frames {
                    state.state = VadState::Silence;
                    state.speech_frames = 0;
                    state.silence_frames = 0;
                    VadResult::SpeechEnd
                } else {
                    VadResult::PotentialSpeechEnd
                }
            },

            (VadState::Silence, false) => VadResult::Silence,
        };

        Ok((state.state, probability, result))
    }

    /// Reset VAD state
    ///
    /// P0 FIX: Now uses single lock.
    pub fn reset(&self) {
        let mut state = self.mutable.lock();
        state.state = VadState::Silence;
        state.speech_frames = 0;
        state.silence_frames = 0;
        state.gru_state.fill(0.0);
    }

    /// Get current state
    pub fn state(&self) -> VadState {
        self.mutable.lock().state
    }

    /// Get accumulated speech duration in frames
    pub fn speech_frames(&self) -> usize {
        self.mutable.lock().speech_frames
    }

    /// Get accumulated silence duration in frames
    pub fn silence_frames(&self) -> usize {
        self.mutable.lock().silence_frames
    }
}

// ============================================================================
// ONNX-only: Mel Filterbank for feature extraction
// ============================================================================

#[cfg(feature = "onnx")]
/// Mel filterbank for feature extraction (ONNX mode only)
pub struct MelFilterbank {
    n_mels: usize,
    filterbank: Array2<f32>,
    window: Vec<f32>,
    n_fft: usize,
}

#[cfg(feature = "onnx")]
impl MelFilterbank {
    pub fn new(sample_rate: u32, frame_size: usize, n_mels: usize) -> Result<Self, PipelineError> {
        let n_fft = frame_size.next_power_of_two();

        let window: Vec<f32> = (0..frame_size)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos())
            })
            .collect();

        let filterbank = create_mel_filterbank(sample_rate, n_fft, n_mels)?;

        Ok(Self {
            n_mels,
            filterbank,
            window,
            n_fft,
        })
    }

    pub fn compute(&self, audio: &[f32]) -> Result<Vec<f32>, PipelineError> {
        if audio.len() < self.window.len() {
            return Err(PipelineError::Vad(format!(
                "Audio too short: {} < {}",
                audio.len(),
                self.window.len()
            )));
        }

        let windowed: Vec<f32> = audio.iter().zip(&self.window).map(|(a, w)| a * w).collect();

        let mut padded = vec![0.0f32; self.n_fft];
        padded[..windowed.len()].copy_from_slice(&windowed);

        let spectrum = self.simple_magnitude_spectrum(&padded);

        let mel_energies: Vec<f32> = (0..self.n_mels)
            .map(|i| {
                let row = self.filterbank.row(i);
                row.iter()
                    .zip(&spectrum)
                    .map(|(f, s)| f * s)
                    .sum::<f32>()
                    .max(1e-10)
            })
            .collect();

        let log_mel: Vec<f32> = mel_energies.iter().map(|e| e.ln()).collect();

        Ok(log_mel)
    }

    fn simple_magnitude_spectrum(&self, signal: &[f32]) -> Vec<f32> {
        let n_bins = self.n_fft / 2 + 1;
        let mut spectrum = vec![0.0f32; n_bins];

        let band_size = signal.len() / n_bins;
        for (i, bin) in spectrum.iter_mut().enumerate() {
            let start = i * band_size;
            let end = ((i + 1) * band_size).min(signal.len());
            *bin = signal[start..end].iter().map(|s| s * s).sum::<f32>().sqrt();
        }

        spectrum
    }
}

#[cfg(feature = "onnx")]
fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
) -> Result<Array2<f32>, PipelineError> {
    let n_bins = n_fft / 2 + 1;

    let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
    let mel_to_hz = |mel: f32| 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0);

    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let bin_points: Vec<usize> = mel_points
        .iter()
        .map(|&mel| {
            let hz = mel_to_hz(mel);
            ((n_fft as f32 + 1.0) * hz / sample_rate as f32) as usize
        })
        .collect();

    let mut filterbank = Array2::zeros((n_mels, n_bins));

    for i in 0..n_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for j in left..center {
            if j < n_bins {
                filterbank[[i, j]] = (j - left) as f32 / (center - left).max(1) as f32;
            }
        }

        for j in center..right {
            if j < n_bins {
                filterbank[[i, j]] = (right - j) as f32 / (right - center).max(1) as f32;
            }
        }
    }

    Ok(filterbank)
}

// =============================================================================
// P0 FIX: Implement core VoiceActivityDetector trait for polymorphic usage
// =============================================================================

/// Map local VadState to core VADState
impl From<VadState> for CoreVADState {
    fn from(state: VadState) -> Self {
        match state {
            VadState::Silence => CoreVADState::Idle,
            VadState::SpeechStart => CoreVADState::PendingSpeech,
            VadState::Speech => CoreVADState::InSpeech,
            VadState::SpeechEnd => CoreVADState::PendingSilence,
        }
    }
}

#[async_trait]
impl CoreVadTrait for VoiceActivityDetector {
    async fn detect(&self, audio: &AudioFrame, sensitivity: f32) -> bool {
        // Clone frame since process_frame takes &mut
        let mut frame = audio.clone();
        match self.process_frame(&mut frame) {
            Ok((_, prob, _)) => prob >= sensitivity,
            Err(_) => false,
        }
    }

    async fn speech_probability(&self, audio: &AudioFrame) -> f32 {
        let mut frame = audio.clone();
        match self.process_frame(&mut frame) {
            Ok((_, prob, _)) => prob,
            Err(_) => 0.0,
        }
    }

    fn process_stream<'a>(
        &'a self,
        audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
        _config: &'a CoreVADConfig,
    ) -> Pin<Box<dyn Stream<Item = CoreVADEvent> + Send + 'a>> {
        use futures::StreamExt;

        // Transform audio frames to VAD events
        let vad_stream = audio_stream.map(move |frame| {
            let mut frame = frame;
            match self.process_frame(&mut frame) {
                Ok((state, prob, result)) => {
                    // Map VadResult to CoreVADEvent
                    match result {
                        VadResult::SpeechConfirmed => CoreVADEvent::SpeechStart,
                        VadResult::SpeechContinue => {
                            CoreVADEvent::SpeechContinue { probability: prob }
                        },
                        VadResult::SpeechEnd => CoreVADEvent::SpeechEnd,
                        VadResult::Silence
                        | VadResult::PotentialSpeechStart
                        | VadResult::PotentialSpeechEnd => {
                            // During pending states, report based on current state
                            match state {
                                VadState::Speech | VadState::SpeechEnd => {
                                    CoreVADEvent::SpeechContinue { probability: prob }
                                },
                                _ => CoreVADEvent::Silence,
                            }
                        },
                    }
                },
                Err(_) => CoreVADEvent::Silence,
            }
        });

        Box::pin(vad_stream)
    }

    fn reset(&self) {
        // Call existing reset method
        VoiceActivityDetector::reset(self);
    }

    fn current_state(&self) -> CoreVADState {
        self.state().into()
    }

    fn model_info(&self) -> &str {
        "MagicNet VAD (10ms frame, GRU-based)"
    }

    fn is_neural(&self) -> bool {
        #[cfg(feature = "onnx")]
        {
            true
        }
        #[cfg(not(feature = "onnx"))]
        {
            false // Energy-based fallback
        }
    }

    fn recommended_frame_size(&self) -> usize {
        // 10ms at 16kHz = 160 samples
        (self.config.sample_rate as usize * self.config.frame_ms as usize) / 1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.frame_ms, 10);
    }

    #[cfg(feature = "onnx")]
    #[test]
    fn test_mel_filterbank() {
        let fb = MelFilterbank::new(16000, 160, 40).unwrap();
        let audio = vec![0.1f32; 160];
        let mels = fb.compute(&audio).unwrap();
        assert_eq!(mels.len(), 40);
    }
}
