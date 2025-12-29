//! Audio frame types and utilities

use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Supported audio sample rates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum SampleRate {
    /// 8kHz - Telephony
    Hz8000,
    /// 16kHz - Standard speech recognition
    #[default]
    Hz16000,
    /// 22.05kHz - TTS output
    Hz22050,
    /// 44.1kHz - CD quality
    Hz44100,
    /// 48kHz - Professional audio
    Hz48000,
}

impl SampleRate {
    /// Get sample rate as u32
    pub fn as_u32(&self) -> u32 {
        match self {
            SampleRate::Hz8000 => 8000,
            SampleRate::Hz16000 => 16000,
            SampleRate::Hz22050 => 22050,
            SampleRate::Hz44100 => 44100,
            SampleRate::Hz48000 => 48000,
        }
    }

    /// Get frame size for 20ms chunk
    pub fn frame_size_20ms(&self) -> usize {
        (self.as_u32() as usize * 20) / 1000
    }

    /// Get frame size for 10ms chunk
    pub fn frame_size_10ms(&self) -> usize {
        (self.as_u32() as usize * 10) / 1000
    }

    /// Get samples per millisecond
    pub fn samples_per_ms(&self) -> usize {
        self.as_u32() as usize / 1000
    }
}


/// Audio encoding formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum AudioEncoding {
    /// 16-bit signed PCM (little-endian)
    Pcm16,
    /// 32-bit float PCM
    #[default]
    PcmF32,
    /// Opus codec (WebRTC)
    Opus,
    /// μ-law (telephony)
    Mulaw,
    /// A-law (telephony)
    Alaw,
}


/// Audio channel configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum Channels {
    #[default]
    Mono,
    Stereo,
}


impl Channels {
    pub fn count(&self) -> usize {
        match self {
            Channels::Mono => 1,
            Channels::Stereo => 2,
        }
    }
}

/// Audio frame with metadata
///
/// Internally stores samples as f32 for processing efficiency.
#[derive(Clone)]
pub struct AudioFrame {
    /// Raw audio samples (f32, normalized to [-1.0, 1.0])
    pub samples: Arc<[f32]>,
    /// Sample rate
    pub sample_rate: SampleRate,
    /// Number of channels
    pub channels: Channels,
    /// Frame sequence number for ordering
    pub sequence: u64,
    /// Timestamp when frame was captured/generated
    pub timestamp: Instant,
    /// Duration of this frame
    pub duration: Duration,
    /// Voice activity probability (0.0 - 1.0), set by VAD
    pub vad_probability: Option<f32>,
    /// Is this frame during active speech?
    pub is_speech: bool,
    /// Energy level in dB
    pub energy_db: f32,
}

impl std::fmt::Debug for AudioFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioFrame")
            .field("samples_len", &self.samples.len())
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("sequence", &self.sequence)
            .field("duration", &self.duration)
            .field("vad_probability", &self.vad_probability)
            .field("is_speech", &self.is_speech)
            .field("energy_db", &self.energy_db)
            .finish()
    }
}

impl AudioFrame {
    /// Create a new audio frame from f32 samples
    pub fn new(
        samples: Vec<f32>,
        sample_rate: SampleRate,
        channels: Channels,
        sequence: u64,
    ) -> Self {
        let duration = Duration::from_secs_f64(
            samples.len() as f64 / (sample_rate.as_u32() as f64 * channels.count() as f64)
        );
        let energy_db = Self::calculate_energy_db(&samples);

        Self {
            samples: samples.into(),
            sample_rate,
            channels,
            sequence,
            timestamp: Instant::now(),
            duration,
            vad_probability: None,
            is_speech: false,
            energy_db,
        }
    }

    /// Create audio frame with explicit timestamp
    pub fn with_timestamp(
        samples: Vec<f32>,
        sample_rate: SampleRate,
        channels: Channels,
        sequence: u64,
        timestamp: Instant,
    ) -> Self {
        let mut frame = Self::new(samples, sample_rate, channels, sequence);
        frame.timestamp = timestamp;
        frame
    }

    /// Calculate RMS energy in decibels
    fn calculate_energy_db(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return -96.0; // Minimum dB (silence)
        }

        let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
        let rms = (sum_squares / samples.len() as f32).sqrt();

        if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            -96.0
        }
    }

    /// Convert from PCM16 bytes (little-endian)
    pub fn from_pcm16(
        bytes: &[u8],
        sample_rate: SampleRate,
        channels: Channels,
        sequence: u64,
    ) -> Self {
        let samples: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        Self::new(samples, sample_rate, channels, sequence)
    }

    /// Convert to PCM16 bytes (little-endian)
    pub fn to_pcm16(&self) -> Vec<u8> {
        self.samples
            .iter()
            .flat_map(|&sample| {
                let clamped = sample.clamp(-1.0, 1.0);
                let pcm16 = (clamped * 32767.0) as i16;
                pcm16.to_le_bytes()
            })
            .collect()
    }

    /// P5 FIX: High-quality resampling using Rubato (sinc interpolation)
    ///
    /// Uses `FastFixedIn` resampler for efficient, high-quality conversion.
    /// Falls back to linear interpolation if Rubato fails (e.g., for very short frames).
    pub fn resample(&self, target_rate: SampleRate) -> Self {
        use rubato::{FftFixedIn, Resampler};

        if self.sample_rate == target_rate {
            return self.clone();
        }

        let from_rate = self.sample_rate.as_u32() as usize;
        let to_rate = target_rate.as_u32() as usize;

        // Convert f32 samples to f64 for Rubato (higher precision)
        let samples_f64: Vec<f64> = self.samples.iter().map(|&s| s as f64).collect();

        // For very short frames or edge cases, use linear fallback
        if self.samples.len() < 64 {
            return self.resample_linear(target_rate);
        }

        // Create FFT-based resampler (high quality, efficient for batch processing)
        // chunk_size should divide input evenly for best results
        let chunk_size = self.samples.len().min(1024);

        match FftFixedIn::<f64>::new(from_rate, to_rate, chunk_size, 2, 1) {
            Ok(mut resampler) => {
                // Rubato expects Vec<Vec<f64>> for multi-channel, we have mono
                let input_frames = vec![samples_f64];

                match resampler.process(&input_frames, None) {
                    Ok(output_frames) => {
                        // Convert back to f32
                        let resampled: Vec<f32> = output_frames[0]
                            .iter()
                            .map(|&s| s as f32)
                            .collect();

                        Self::new(resampled, target_rate, self.channels, self.sequence)
                    }
                    Err(e) => {
                        tracing::warn!("Rubato processing failed, using linear fallback: {}", e);
                        self.resample_linear(target_rate)
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Rubato init failed, using linear fallback: {}", e);
                self.resample_linear(target_rate)
            }
        }
    }

    /// Linear interpolation fallback for edge cases
    fn resample_linear(&self, target_rate: SampleRate) -> Self {
        let ratio = target_rate.as_u32() as f64 / self.sample_rate.as_u32() as f64;
        let new_len = (self.samples.len() as f64 * ratio) as usize;

        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let idx_floor = src_idx.floor() as usize;
            let idx_ceil = (idx_floor + 1).min(self.samples.len().saturating_sub(1));
            let frac = src_idx - idx_floor as f64;

            let sample = self.samples[idx_floor] * (1.0 - frac as f32)
                + self.samples[idx_ceil] * frac as f32;
            resampled.push(sample);
        }

        Self::new(resampled, target_rate, self.channels, self.sequence)
    }

    /// Convert stereo to mono by averaging channels
    pub fn to_mono(&self) -> Self {
        if self.channels == Channels::Mono {
            return self.clone();
        }

        let mono_samples: Vec<f32> = self.samples
            .chunks_exact(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect();

        Self::new(mono_samples, self.sample_rate, Channels::Mono, self.sequence)
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.duration.as_millis() as u64
    }

    /// Check if frame is likely silence based on energy
    pub fn is_likely_silence(&self, threshold_db: f32) -> bool {
        self.energy_db < threshold_db
    }

    /// Split frame into smaller chunks
    pub fn split(&self, chunk_samples: usize) -> Vec<AudioFrame> {
        let mut chunks = Vec::new();
        let mut seq = self.sequence;

        for chunk in self.samples.chunks(chunk_samples) {
            chunks.push(AudioFrame::new(
                chunk.to_vec(),
                self.sample_rate,
                self.channels,
                seq,
            ));
            seq += 1;
        }

        chunks
    }
}

/// Audio buffer for accumulating frames
#[derive(Debug)]
pub struct AudioBuffer {
    samples: Vec<f32>,
    sample_rate: SampleRate,
    channels: Channels,
    max_duration: Duration,
}

impl AudioBuffer {
    pub fn new(sample_rate: SampleRate, channels: Channels, max_duration: Duration) -> Self {
        let max_samples = (sample_rate.as_u32() as f64
            * channels.count() as f64
            * max_duration.as_secs_f64()) as usize;

        Self {
            samples: Vec::with_capacity(max_samples),
            sample_rate,
            channels,
            max_duration,
        }
    }

    /// Push audio frame to buffer
    pub fn push(&mut self, frame: &AudioFrame) {
        // Resample if needed
        let frame = if frame.sample_rate != self.sample_rate {
            frame.resample(self.sample_rate)
        } else {
            frame.clone()
        };

        // Convert to mono if needed
        let frame = if frame.channels != self.channels {
            frame.to_mono()
        } else {
            frame
        };

        self.samples.extend(frame.samples.iter());

        // Trim if exceeds max duration
        let max_samples = (self.sample_rate.as_u32() as f64
            * self.channels.count() as f64
            * self.max_duration.as_secs_f64()) as usize;

        if self.samples.len() > max_samples {
            let excess = self.samples.len() - max_samples;
            self.samples.drain(0..excess);
        }
    }

    /// Get all samples
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Get buffer duration
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(
            self.samples.len() as f64
            / (self.sample_rate.as_u32() as f64 * self.channels.count() as f64)
        )
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Drain samples from front
    pub fn drain(&mut self, count: usize) -> Vec<f32> {
        let count = count.min(self.samples.len());
        self.samples.drain(0..count).collect()
    }

    /// Check if buffer has at least specified duration
    pub fn has_duration(&self, duration: Duration) -> bool {
        self.duration() >= duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_rate_conversions() {
        assert_eq!(SampleRate::Hz16000.as_u32(), 16000);
        assert_eq!(SampleRate::Hz16000.frame_size_10ms(), 160);
        assert_eq!(SampleRate::Hz16000.frame_size_20ms(), 320);
    }

    #[test]
    fn test_audio_frame_from_pcm16() {
        let pcm16: Vec<u8> = vec![0x00, 0x40, 0x00, 0xC0]; // Two samples
        let frame = AudioFrame::from_pcm16(&pcm16, SampleRate::Hz16000, Channels::Mono, 0);

        assert_eq!(frame.samples.len(), 2);
        assert!(frame.samples[0] > 0.0); // Positive sample
        assert!(frame.samples[1] < 0.0); // Negative sample
    }

    #[test]
    fn test_audio_frame_resample() {
        let samples = vec![0.0f32; 160]; // 10ms at 16kHz
        let frame = AudioFrame::new(samples, SampleRate::Hz16000, Channels::Mono, 0);

        let resampled = frame.resample(SampleRate::Hz8000);
        assert_eq!(resampled.samples.len(), 80); // 10ms at 8kHz
    }

    #[test]
    fn test_energy_calculation() {
        // Silence
        let silent = AudioFrame::new(vec![0.0; 160], SampleRate::Hz16000, Channels::Mono, 0);
        assert!(silent.energy_db < -90.0);

        // Full scale sine-ish
        let loud = AudioFrame::new(vec![0.5; 160], SampleRate::Hz16000, Channels::Mono, 0);
        assert!(loud.energy_db > -10.0);
    }

    #[test]
    fn test_audio_buffer() {
        let mut buffer = AudioBuffer::new(
            SampleRate::Hz16000,
            Channels::Mono,
            Duration::from_secs(1),
        );

        let frame = AudioFrame::new(vec![0.1; 160], SampleRate::Hz16000, Channels::Mono, 0);
        buffer.push(&frame);

        assert_eq!(buffer.samples().len(), 160);
        assert!(buffer.duration() >= Duration::from_millis(9));
    }
}
