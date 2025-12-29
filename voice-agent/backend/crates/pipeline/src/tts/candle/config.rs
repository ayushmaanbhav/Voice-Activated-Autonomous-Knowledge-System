//! Configuration for IndicF5 TTS Model
//!
//! Defines hyperparameters for the F5-TTS architecture.

/// Quantization mode for TTS inference
#[derive(Debug, Clone, Copy, Default)]
pub enum TtsQuantization {
    /// Full precision (FP32) - most accurate
    #[default]
    F32,
    /// Half precision (FP16) - 2x memory reduction, faster on some hardware
    F16,
    /// Brain float (BF16) - good balance of range and precision
    BF16,
}

impl TtsQuantization {
    /// Get the Candle DType for this quantization mode
    #[cfg(feature = "candle")]
    pub fn to_dtype(&self) -> candle_core::DType {
        match self {
            TtsQuantization::F32 => candle_core::DType::F32,
            TtsQuantization::F16 => candle_core::DType::F16,
            TtsQuantization::BF16 => candle_core::DType::BF16,
        }
    }

    /// Memory reduction factor compared to F32
    pub fn memory_factor(&self) -> f32 {
        match self {
            TtsQuantization::F32 => 1.0,
            TtsQuantization::F16 | TtsQuantization::BF16 => 0.5,
        }
    }
}

/// Configuration for the IndicF5 TTS model
#[derive(Debug, Clone)]
pub struct IndicF5Config {
    /// Model dimension (hidden size)
    pub dim: usize,

    /// Number of transformer layers
    pub depth: usize,

    /// Number of attention heads
    pub heads: usize,

    /// Dimension per attention head
    pub head_dim: usize,

    /// Feedforward multiplier (hidden_dim = dim * ff_mult)
    pub ff_mult: f32,

    /// Number of mel spectrogram bins
    pub n_mels: usize,

    /// Vocabulary size for text/phoneme tokens
    pub vocab_size: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Dropout rate (0.0 for inference)
    pub dropout: f32,

    /// Use RoPE position embeddings
    pub use_rope: bool,

    /// RoPE base frequency
    pub rope_base: f32,

    /// Number of ConvNeXt blocks in each location
    pub conv_layers: usize,

    /// ConvNeXt kernel size
    pub conv_kernel_size: usize,

    /// Layer norm epsilon
    pub layer_norm_eps: f64,

    /// Time embedding dimension
    pub time_dim: usize,

    /// Sample rate for audio
    pub sample_rate: usize,

    /// Hop length for mel spectrogram
    pub hop_length: usize,

    /// Audio segment length for training
    pub audio_segment_length: usize,

    /// Quantization mode for inference
    pub quantization: TtsQuantization,
}

impl Default for IndicF5Config {
    fn default() -> Self {
        Self {
            // F5-TTS default configuration
            dim: 1024,
            depth: 22,
            heads: 16,
            head_dim: 64,
            ff_mult: 4.0,
            n_mels: 100,
            vocab_size: 256,  // Character-level for Hindi
            max_seq_len: 4096,
            dropout: 0.0,
            use_rope: true,
            rope_base: 10000.0,
            conv_layers: 4,
            conv_kernel_size: 7,
            layer_norm_eps: 1e-6,
            time_dim: 256,
            sample_rate: 24000,
            hop_length: 256,
            audio_segment_length: 24000 * 30,  // 30 seconds
            quantization: TtsQuantization::F32,
        }
    }
}

impl IndicF5Config {
    /// Create a smaller configuration for testing
    pub fn small() -> Self {
        Self {
            dim: 256,
            depth: 4,
            heads: 4,
            head_dim: 64,
            ff_mult: 2.0,
            n_mels: 80,
            vocab_size: 256,
            max_seq_len: 1024,
            dropout: 0.0,
            use_rope: true,
            rope_base: 10000.0,
            conv_layers: 2,
            conv_kernel_size: 7,
            layer_norm_eps: 1e-6,
            time_dim: 128,
            sample_rate: 24000,
            hop_length: 256,
            audio_segment_length: 24000 * 10,
            quantization: TtsQuantization::F32,
        }
    }

    /// Create configuration matching IndicF5 Hindi model
    pub fn indicf5_hindi() -> Self {
        Self {
            dim: 1024,
            depth: 22,
            heads: 16,
            head_dim: 64,
            ff_mult: 4.0,
            n_mels: 100,
            vocab_size: 2545,  // IndicF5 Hindi vocabulary
            max_seq_len: 4096,
            dropout: 0.0,
            use_rope: true,
            rope_base: 10000.0,
            conv_layers: 4,
            conv_kernel_size: 7,
            layer_norm_eps: 1e-6,
            time_dim: 256,
            sample_rate: 24000,
            hop_length: 256,
            audio_segment_length: 24000 * 30,
            quantization: TtsQuantization::F32,
        }
    }

    /// Create configuration with FP16 quantization for faster inference
    pub fn indicf5_hindi_fp16() -> Self {
        Self {
            quantization: TtsQuantization::F16,
            ..Self::indicf5_hindi()
        }
    }

    /// Enable FP16 quantization on this config
    pub fn with_fp16(mut self) -> Self {
        self.quantization = TtsQuantization::F16;
        self
    }

    /// Enable BF16 quantization on this config
    pub fn with_bf16(mut self) -> Self {
        self.quantization = TtsQuantization::BF16;
        self
    }

    /// Compute intermediate dimension for feedforward
    pub fn ff_dim(&self) -> usize {
        (self.dim as f32 * self.ff_mult) as usize
    }

    /// Compute number of audio frames for given duration in seconds
    pub fn frames_for_duration(&self, duration_secs: f32) -> usize {
        ((duration_secs * self.sample_rate as f32) / self.hop_length as f32).ceil() as usize
    }

    /// Estimated model memory in bytes
    pub fn estimated_memory(&self) -> usize {
        // Rough estimate: params * bytes_per_param
        let params = self.dim * self.dim * self.depth * 12; // Rough transformer param count
        let bytes_per_param = match self.quantization {
            TtsQuantization::F32 => 4,
            TtsQuantization::F16 | TtsQuantization::BF16 => 2,
        };
        params * bytes_per_param
    }
}

/// Configuration for the Vocos vocoder
#[derive(Debug, Clone)]
pub struct VocosConfig {
    /// Number of input mel bins
    pub n_mels: usize,

    /// Hidden dimension in ConvNeXt blocks
    pub hidden_dim: usize,

    /// Number of ConvNeXt blocks
    pub num_blocks: usize,

    /// Intermediate dimension multiplier
    pub intermediate_mult: f32,

    /// ConvNeXt kernel sizes
    pub kernel_sizes: Vec<usize>,

    /// STFT window size
    pub n_fft: usize,

    /// Hop length
    pub hop_length: usize,

    /// Sample rate
    pub sample_rate: usize,
}

impl Default for VocosConfig {
    fn default() -> Self {
        Self {
            n_mels: 100,
            hidden_dim: 512,
            num_blocks: 8,
            intermediate_mult: 4.0,
            kernel_sizes: vec![7, 7, 7, 7, 7, 7, 7, 7],
            n_fft: 1024,
            hop_length: 256,
            sample_rate: 24000,
        }
    }
}

/// Configuration for flow matching ODE solver
#[derive(Debug, Clone)]
pub struct FlowMatchingConfig {
    /// Number of ODE integration steps
    pub num_steps: usize,

    /// Classifier-free guidance strength
    pub cfg_strength: f32,

    /// Sway sampling coefficient
    pub sway_coef: f32,

    /// Minimum timestep
    pub t_min: f32,

    /// Maximum timestep
    pub t_max: f32,
}

impl Default for FlowMatchingConfig {
    fn default() -> Self {
        Self {
            num_steps: 32,
            cfg_strength: 2.0,
            sway_coef: -1.0,
            t_min: 0.0,
            t_max: 1.0,
        }
    }
}

impl FlowMatchingConfig {
    /// Create a faster configuration with fewer steps
    pub fn fast() -> Self {
        Self {
            num_steps: 16,
            cfg_strength: 2.0,
            sway_coef: -1.0,
            t_min: 0.0,
            t_max: 1.0,
        }
    }

    /// Create a higher quality configuration with more steps
    pub fn quality() -> Self {
        Self {
            num_steps: 64,
            cfg_strength: 2.0,
            sway_coef: -1.0,
            t_min: 0.0,
            t_max: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = IndicF5Config::default();
        assert_eq!(config.dim, 1024);
        assert_eq!(config.depth, 22);
        assert_eq!(config.heads, 16);
    }

    #[test]
    fn test_ff_dim() {
        let config = IndicF5Config::default();
        assert_eq!(config.ff_dim(), 4096);  // 1024 * 4.0
    }

    #[test]
    fn test_frames_for_duration() {
        let config = IndicF5Config::default();
        // 1 second at 24kHz with hop_length 256
        let frames = config.frames_for_duration(1.0);
        assert_eq!(frames, 94);  // ceil(24000 / 256)
    }
}
