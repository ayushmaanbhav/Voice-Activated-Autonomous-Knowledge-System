//! Audio Codec Support
//!
//! Opus encoding/decoding for WebRTC audio transport.

use audiopus::{
    coder::{Encoder, Decoder},
    packet::Packet,
    Application, Channels, SampleRate as OpusSampleRate, MutSignals,
};
use parking_lot::Mutex;

use crate::TransportError;

/// Opus encoder wrapper
pub struct OpusEncoder {
    encoder: Mutex<Encoder>,
    sample_rate: u32,
    channels: u8,
    frame_size: usize,
}

impl OpusEncoder {
    /// Create a new Opus encoder
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate (8000, 12000, 16000, 24000, 48000)
    /// * `channels` - Number of channels (1 or 2)
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self, TransportError> {
        let opus_sample_rate = match sample_rate {
            8000 => OpusSampleRate::Hz8000,
            12000 => OpusSampleRate::Hz12000,
            16000 => OpusSampleRate::Hz16000,
            24000 => OpusSampleRate::Hz24000,
            48000 => OpusSampleRate::Hz48000,
            _ => return Err(TransportError::Internal(format!(
                "Unsupported sample rate: {}. Use 8000, 12000, 16000, 24000, or 48000",
                sample_rate
            ))),
        };

        let opus_channels = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(TransportError::Internal(format!(
                "Unsupported channel count: {}. Use 1 or 2",
                channels
            ))),
        };

        let encoder = Encoder::new(opus_sample_rate, opus_channels, Application::Voip)
            .map_err(|e| TransportError::Internal(format!("Failed to create Opus encoder: {}", e)))?;

        // Frame size: 20ms of audio
        let frame_size = (sample_rate as usize * 20) / 1000;

        Ok(Self {
            encoder: Mutex::new(encoder),
            sample_rate,
            channels,
            frame_size,
        })
    }

    /// Encode PCM samples to Opus
    ///
    /// # Arguments
    /// * `pcm` - PCM samples (f32, -1.0 to 1.0)
    ///
    /// # Returns
    /// Encoded Opus data
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<u8>, TransportError> {
        // Convert f32 to i16
        let pcm_i16: Vec<i16> = pcm.iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        let mut output = vec![0u8; 4000]; // Max Opus frame size

        let encoder = self.encoder.lock();
        let encoded_len = encoder.encode(&pcm_i16, &mut output)
            .map_err(|e| TransportError::Internal(format!("Opus encode error: {}", e)))?;

        output.truncate(encoded_len);
        Ok(output)
    }

    /// Encode a frame (20ms of audio)
    pub fn encode_frame(&self, pcm: &[f32]) -> Result<Vec<u8>, TransportError> {
        if pcm.len() != self.frame_size * self.channels as usize {
            return Err(TransportError::Internal(format!(
                "Invalid frame size: expected {}, got {}",
                self.frame_size * self.channels as usize,
                pcm.len()
            )));
        }
        self.encode(pcm)
    }

    /// Get frame size in samples
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u8 {
        self.channels
    }
}

/// Opus decoder wrapper
pub struct OpusDecoder {
    decoder: Mutex<Decoder>,
    sample_rate: u32,
    channels: u8,
    frame_size: usize,
}

impl OpusDecoder {
    /// Create a new Opus decoder
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate (8000, 12000, 16000, 24000, 48000)
    /// * `channels` - Number of channels (1 or 2)
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self, TransportError> {
        let opus_sample_rate = match sample_rate {
            8000 => OpusSampleRate::Hz8000,
            12000 => OpusSampleRate::Hz12000,
            16000 => OpusSampleRate::Hz16000,
            24000 => OpusSampleRate::Hz24000,
            48000 => OpusSampleRate::Hz48000,
            _ => return Err(TransportError::Internal(format!(
                "Unsupported sample rate: {}",
                sample_rate
            ))),
        };

        let opus_channels = match channels {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(TransportError::Internal(format!(
                "Unsupported channel count: {}",
                channels
            ))),
        };

        let decoder = Decoder::new(opus_sample_rate, opus_channels)
            .map_err(|e| TransportError::Internal(format!("Failed to create Opus decoder: {}", e)))?;

        // Frame size: 20ms of audio
        let frame_size = (sample_rate as usize * 20) / 1000;

        Ok(Self {
            decoder: Mutex::new(decoder),
            sample_rate,
            channels,
            frame_size,
        })
    }

    /// Decode Opus data to PCM samples
    ///
    /// # Arguments
    /// * `opus_data` - Encoded Opus data
    ///
    /// # Returns
    /// PCM samples (f32, -1.0 to 1.0)
    pub fn decode(&self, opus_data: &[u8]) -> Result<Vec<f32>, TransportError> {
        let max_samples = self.frame_size * self.channels as usize * 6; // Up to 120ms
        let mut output = vec![0i16; max_samples];

        let packet = Packet::try_from(opus_data)
            .map_err(|e| TransportError::Internal(format!("Invalid Opus packet: {}", e)))?;

        let mut decoder = self.decoder.lock();
        let mut_signals = MutSignals::try_from(&mut output[..])
            .map_err(|e| TransportError::Internal(format!("Signal buffer error: {}", e)))?;

        let decoded_len = decoder.decode(Some(packet), mut_signals, false)
            .map_err(|e| TransportError::Internal(format!("Opus decode error: {}", e)))?;

        // Convert i16 to f32
        let pcm_f32: Vec<f32> = output[..decoded_len]
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        Ok(pcm_f32)
    }

    /// Decode with packet loss concealment (PLC)
    ///
    /// Call this when a packet is lost to generate concealment audio.
    pub fn decode_plc(&self) -> Result<Vec<f32>, TransportError> {
        let max_samples = self.frame_size * self.channels as usize;
        let mut output = vec![0i16; max_samples];

        let mut decoder = self.decoder.lock();
        let mut_signals = MutSignals::try_from(&mut output[..])
            .map_err(|e| TransportError::Internal(format!("Signal buffer error: {}", e)))?;

        let decoded_len = decoder.decode(None::<Packet>, mut_signals, false)
            .map_err(|e| TransportError::Internal(format!("Opus PLC error: {}", e)))?;

        let pcm_f32: Vec<f32> = output[..decoded_len]
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        Ok(pcm_f32)
    }

    /// Get frame size in samples
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of channels
    pub fn channels(&self) -> u8 {
        self.channels
    }
}

/// P5 FIX: High-quality resampler using Rubato
pub struct Resampler {
    from_rate: u32,
    to_rate: u32,
}

impl Resampler {
    /// Create a new resampler
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self { from_rate, to_rate }
    }

    /// P5 FIX: Resample audio using Rubato (sinc interpolation)
    ///
    /// Uses FFT-based resampler for high-quality conversion.
    /// Falls back to linear interpolation for edge cases.
    pub fn resample(&self, input: &[f32]) -> Vec<f32> {
        use rubato::{FftFixedIn, Resampler as RubatoResampler};

        if self.from_rate == self.to_rate {
            return input.to_vec();
        }

        // For very short inputs, use linear fallback
        if input.len() < 64 {
            return self.resample_linear(input);
        }

        // Convert to f64 for higher precision
        let samples_f64: Vec<f64> = input.iter().map(|&s| s as f64).collect();
        let chunk_size = input.len().min(1024);

        match FftFixedIn::<f64>::new(
            self.from_rate as usize,
            self.to_rate as usize,
            chunk_size,
            2,  // sub_chunks
            1,  // channels
        ) {
            Ok(mut resampler) => {
                let input_frames = vec![samples_f64];
                match resampler.process(&input_frames, None) {
                    Ok(output_frames) => {
                        output_frames[0].iter().map(|&s| s as f32).collect()
                    }
                    Err(e) => {
                        tracing::warn!("Rubato processing failed: {}", e);
                        self.resample_linear(input)
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Rubato init failed: {}", e);
                self.resample_linear(input)
            }
        }
    }

    /// Linear interpolation fallback
    fn resample_linear(&self, input: &[f32]) -> Vec<f32> {
        let ratio = self.to_rate as f64 / self.from_rate as f64;
        let output_len = (input.len() as f64 * ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 / ratio;
            let idx_floor = src_idx.floor() as usize;
            let idx_ceil = (idx_floor + 1).min(input.len().saturating_sub(1));
            let frac = src_idx - idx_floor as f64;

            let sample = input[idx_floor] * (1.0 - frac as f32) + input[idx_ceil] * frac as f32;
            output.push(sample);
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_encoder_new() {
        let encoder = OpusEncoder::new(16000, 1);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.sample_rate(), 16000);
        assert_eq!(encoder.channels(), 1);
        assert_eq!(encoder.frame_size(), 320); // 20ms at 16kHz
    }

    #[test]
    fn test_opus_decoder_new() {
        let decoder = OpusDecoder::new(16000, 1);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let encoder = OpusEncoder::new(16000, 1).unwrap();
        let decoder = OpusDecoder::new(16000, 1).unwrap();

        // Create a simple sine wave
        let frame_size = encoder.frame_size();
        let pcm: Vec<f32> = (0..frame_size)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();

        // Encode
        let encoded = encoder.encode(&pcm).unwrap();
        assert!(!encoded.is_empty());
        assert!(encoded.len() < pcm.len() * 2); // Should be compressed

        // Decode
        let decoded = decoder.decode(&encoded).unwrap();
        assert_eq!(decoded.len(), frame_size);

        // Verify the decode produced output
        // Note: Opus is optimized for voice and may not preserve simple synthetic signals well
        // We just check that the output has similar properties (non-zero, bounded)
        let output_rms: f32 = (decoded.iter().map(|s| s.powi(2)).sum::<f32>() / decoded.len() as f32).sqrt();
        let input_rms: f32 = (pcm.iter().map(|s| s.powi(2)).sum::<f32>() / pcm.len() as f32).sqrt();

        // Output should have similar energy (within an order of magnitude)
        assert!(output_rms > 0.01, "Output too quiet: {}", output_rms);
        assert!(output_rms < input_rms * 5.0, "Output energy ratio too high: {} vs {}", output_rms, input_rms);
    }

    #[test]
    fn test_resampler() {
        let resampler = Resampler::new(16000, 48000);

        let input: Vec<f32> = (0..160).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = resampler.resample(&input);

        // 16kHz to 48kHz = 3x samples
        assert_eq!(output.len(), 480);
    }
}
