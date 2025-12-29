//! Silero VAD - Voice Activity Detection
//!
//! Implementation of Silero VAD v5 using ONNX Runtime.
//! Features:
//! - Takes raw audio input (16kHz, 512 samples = 32ms chunks)
//! - LSTM-based architecture with stateful inference
//! - No mel filterbank required - works directly on waveform

use parking_lot::Mutex;
use std::path::Path;
use voice_agent_core::AudioFrame;

use crate::PipelineError;

use ndarray::Array2;

#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};

use super::{VadConfig, VadEngine, VadResult, VadState};

/// Silero VAD configuration
#[derive(Debug, Clone)]
pub struct SileroConfig {
    /// Speech probability threshold (0.0 - 1.0)
    pub threshold: f32,
    /// Chunk size in samples (512 for 16kHz = 32ms)
    pub chunk_size: usize,
    /// Sample rate (must be 16000)
    pub sample_rate: u32,
    /// Minimum speech frames to confirm speech
    pub min_speech_frames: usize,
    /// Minimum silence frames to confirm silence
    pub min_silence_frames: usize,
    /// Energy floor in dB for quick silence detection
    pub energy_floor_db: f32,
}

impl Default for SileroConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            chunk_size: 512, // 32ms at 16kHz
            sample_rate: 16000,
            min_speech_frames: 8,   // ~256ms
            min_silence_frames: 10, // ~320ms
            energy_floor_db: -50.0,
        }
    }
}

impl From<SileroConfig> for VadConfig {
    fn from(config: SileroConfig) -> Self {
        VadConfig {
            threshold: config.threshold,
            frame_ms: (config.chunk_size as u32 * 1000) / config.sample_rate,
            min_speech_frames: config.min_speech_frames,
            min_silence_frames: config.min_silence_frames,
            n_mels: 0, // Not used by Silero
            sample_rate: config.sample_rate,
            gru_hidden_size: 64, // Silero uses 64-dim LSTM states
            energy_floor_db: config.energy_floor_db,
        }
    }
}

/// Mutable state for Silero VAD
struct SileroMutableState {
    /// LSTM hidden state [2, 64]
    h_state: Array2<f32>,
    /// LSTM cell state [2, 64]
    c_state: Array2<f32>,
    /// Current VAD state
    state: VadState,
    /// Accumulated speech frames
    speech_frames: usize,
    /// Accumulated silence frames
    silence_frames: usize,
    /// Audio buffer for accumulating samples to chunk_size
    audio_buffer: Vec<f32>,
}

/// Silero VAD v5 implementation
pub struct SileroVad {
    #[cfg(feature = "onnx")]
    session: Session,
    config: SileroConfig,
    /// Single lock for all mutable state
    mutable: Mutex<SileroMutableState>,
}

impl SileroVad {
    /// Create a new Silero VAD from ONNX model
    ///
    /// Model should be silero_vad.onnx from:
    /// https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx
    #[cfg(feature = "onnx")]
    pub fn new(model_path: impl AsRef<Path>, config: SileroConfig) -> Result<Self, PipelineError> {
        let session = Session::builder()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        // Silero VAD v5 uses LSTM with [2, 64] states (2 layers, 64 hidden)
        let h_state = Array2::zeros((2, 64));
        let c_state = Array2::zeros((2, 64));

        Ok(Self {
            session,
            config,
            mutable: Mutex::new(SileroMutableState {
                h_state,
                c_state,
                state: VadState::Silence,
                speech_frames: 0,
                silence_frames: 0,
                audio_buffer: Vec::with_capacity(config.chunk_size),
            }),
        })
    }

    /// Create a new Silero VAD (without ONNX - uses energy-based detection)
    #[cfg(not(feature = "onnx"))]
    pub fn new(_model_path: impl AsRef<Path>, config: SileroConfig) -> Result<Self, PipelineError> {
        Self::simple(config)
    }

    /// Create a simple energy-based VAD (no model required)
    #[cfg(not(feature = "onnx"))]
    pub fn simple(config: SileroConfig) -> Result<Self, PipelineError> {
        let chunk_size = config.chunk_size;
        Ok(Self {
            config,
            mutable: Mutex::new(SileroMutableState {
                h_state: Array2::zeros((2, 64)),
                c_state: Array2::zeros((2, 64)),
                state: VadState::Silence,
                speech_frames: 0,
                silence_frames: 0,
                audio_buffer: Vec::with_capacity(chunk_size),
            }),
        })
    }

    /// Process audio samples and return VAD result
    ///
    /// Buffers samples until chunk_size is reached, then runs inference.
    ///
    /// # Thread Safety
    /// P0 FIX: Lock is held throughout the entire process to prevent race conditions.
    /// The previous implementation released the lock before compute_probability(),
    /// which could allow another thread to modify state between inference and update.
    pub fn process(&self, frame: &mut AudioFrame) -> Result<(VadState, f32, VadResult), PipelineError> {
        // Quick energy check for obvious silence
        if frame.energy_db < self.config.energy_floor_db {
            frame.vad_probability = Some(0.0);
            frame.is_speech = false;

            let mut state = self.mutable.lock();
            return self.update_state_inner(&mut state, false, 0.0);
        }

        // P0 FIX: Hold lock throughout entire process to prevent race conditions
        let mut state = self.mutable.lock();
        state.audio_buffer.extend_from_slice(&frame.samples);

        // If we have enough samples, run inference
        if state.audio_buffer.len() >= self.config.chunk_size {
            let chunk: Vec<f32> = state.audio_buffer.drain(..self.config.chunk_size).collect();

            // P0 FIX: Compute probability while holding lock, pass mutable state
            let speech_prob = self.compute_probability_locked(&mut state, &chunk)?;

            frame.vad_probability = Some(speech_prob);
            let is_speech = speech_prob >= self.config.threshold;
            frame.is_speech = is_speech;

            // State update within same lock scope
            self.update_state_inner(&mut state, is_speech, speech_prob)
        } else {
            // Not enough samples yet, return current state
            let current_state = state.state;
            let result = match current_state {
                VadState::Speech | VadState::SpeechStart => VadResult::SpeechContinue,
                VadState::SpeechEnd => VadResult::PotentialSpeechEnd,
                VadState::Silence => VadResult::Silence,
            };
            Ok((current_state, 0.0, result))
        }
    }

    /// Compute speech probability using ONNX model (lock-free version)
    ///
    /// P0 FIX: Takes mutable state as parameter to avoid double-locking.
    /// Caller must hold the lock.
    #[cfg(feature = "onnx")]
    fn compute_probability_locked(
        &self,
        state: &mut SileroMutableState,
        audio_chunk: &[f32],
    ) -> Result<f32, PipelineError> {
        use ndarray::{Array1, ArrayView2};

        // Prepare input tensor [1, chunk_size]
        let input = Array1::from_vec(audio_chunk.to_vec())
            .into_shape((1, audio_chunk.len()))
            .map_err(|e| PipelineError::Vad(e.to_string()))?;

        // Sample rate tensor [1]
        let sr = ndarray::arr1(&[self.config.sample_rate as i64]);

        // Run inference
        // Silero VAD v5 inputs: input, sr, h, c
        // Silero VAD v5 outputs: output, hn, cn
        let outputs = self.session.run(ort::inputs![
            "input" => input.view(),
            "sr" => sr.view(),
            "h" => state.h_state.view(),
            "c" => state.c_state.view(),
        ].map_err(|e| PipelineError::Model(e.to_string()))?)
        .map_err(|e| PipelineError::Model(e.to_string()))?;

        // Extract speech probability
        let speech_prob: f32 = outputs
            .get("output")
            .ok_or_else(|| PipelineError::Model("Missing output tensor".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .view()
            .iter()
            .next()
            .copied()
            .unwrap_or(0.0);

        // Update LSTM states
        if let Some(hn) = outputs.get("hn") {
            let new_h: ArrayView2<f32> = hn
                .try_extract_tensor()
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            state.h_state.assign(&new_h);
        }

        if let Some(cn) = outputs.get("cn") {
            let new_c: ArrayView2<f32> = cn
                .try_extract_tensor()
                .map_err(|e| PipelineError::Model(e.to_string()))?;
            state.c_state.assign(&new_c);
        }

        Ok(speech_prob)
    }

    /// Compute speech probability (energy-based fallback, lock-free version)
    ///
    /// P0 FIX: Takes mutable state as parameter for consistency with ONNX version.
    #[cfg(not(feature = "onnx"))]
    fn compute_probability_locked(
        &self,
        _state: &mut SileroMutableState,
        audio_chunk: &[f32],
    ) -> Result<f32, PipelineError> {
        // Simple energy-based VAD
        let energy: f32 = audio_chunk.iter().map(|s| s * s).sum::<f32>() / audio_chunk.len() as f32;
        let energy_db = 10.0 * energy.max(1e-10).log10();

        let threshold_db = self.config.energy_floor_db + 10.0;
        let prob = if energy_db > threshold_db {
            ((energy_db - threshold_db) / 30.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Ok(prob)
    }

    /// Update state machine based on detection result
    fn update_state_inner(
        &self,
        state: &mut SileroMutableState,
        is_speech: bool,
        probability: f32,
    ) -> Result<(VadState, f32, VadResult), PipelineError> {
        let result = match (state.state, is_speech) {
            (VadState::Silence, true) => {
                state.state = VadState::SpeechStart;
                state.speech_frames = 1;
                state.silence_frames = 0;
                VadResult::PotentialSpeechStart
            }

            (VadState::SpeechStart, true) => {
                state.speech_frames += 1;
                if state.speech_frames >= self.config.min_speech_frames {
                    state.state = VadState::Speech;
                    VadResult::SpeechConfirmed
                } else {
                    VadResult::PotentialSpeechStart
                }
            }

            (VadState::SpeechStart, false) => {
                state.state = VadState::Silence;
                state.speech_frames = 0;
                VadResult::Silence
            }

            (VadState::Speech, true) => {
                state.silence_frames = 0;
                VadResult::SpeechContinue
            }

            (VadState::Speech, false) => {
                state.state = VadState::SpeechEnd;
                state.silence_frames = 1;
                VadResult::PotentialSpeechEnd
            }

            (VadState::SpeechEnd, true) => {
                state.state = VadState::Speech;
                state.silence_frames = 0;
                VadResult::SpeechContinue
            }

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
            }

            (VadState::Silence, false) => {
                VadResult::Silence
            }
        };

        Ok((state.state, probability, result))
    }

    /// Reset VAD state
    pub fn reset(&self) {
        let mut state = self.mutable.lock();
        state.state = VadState::Silence;
        state.speech_frames = 0;
        state.silence_frames = 0;
        state.h_state.fill(0.0);
        state.c_state.fill(0.0);
        state.audio_buffer.clear();
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


/// Implement VadEngine trait for SileroVad
#[async_trait::async_trait]
impl VadEngine for SileroVad {
    fn process_frame(&mut self, frame: &mut AudioFrame) -> Result<VadResult, PipelineError> {
        let (_, _, result) = self.process(frame)?;
        Ok(result)
    }

    fn reset(&mut self) {
        SileroVad::reset(self);
    }

    fn state(&self) -> VadState {
        SileroVad::state(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silero_config_default() {
        let config = SileroConfig::default();
        assert_eq!(config.threshold, 0.5);
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_silero_config_to_vad_config() {
        let silero = SileroConfig::default();
        let vad: VadConfig = silero.into();
        assert_eq!(vad.threshold, 0.5);
        assert_eq!(vad.frame_ms, 32); // 512 samples at 16kHz = 32ms
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_silero_simple_vad() {
        let config = SileroConfig::default();
        let vad = SileroVad::simple(config).unwrap();
        assert_eq!(vad.state(), VadState::Silence);
    }

    #[cfg(not(feature = "onnx"))]
    #[test]
    fn test_silero_energy_detection() {
        use voice_agent_core::{SampleRate, Channels};

        let config = SileroConfig::default();
        let vad = SileroVad::simple(config).unwrap();

        // Silent audio
        let silence = vec![0.0f32; 512];
        let mut frame = AudioFrame::new(
            silence,
            SampleRate::Hz16000,
            Channels::Mono,
            0,
        );

        let (state, prob, result) = vad.process(&mut frame).unwrap();
        assert_eq!(state, VadState::Silence);
        assert!(prob < 0.5);
        assert_eq!(result, VadResult::Silence);

        // "Speech" audio (high energy)
        let speech: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let mut frame = AudioFrame::new(
            speech,
            SampleRate::Hz16000,
            Channels::Mono,
            1,
        );

        let (_state, prob, _result) = vad.process(&mut frame).unwrap();
        // Should detect potential speech start
        assert!(prob > 0.0);
    }
}
