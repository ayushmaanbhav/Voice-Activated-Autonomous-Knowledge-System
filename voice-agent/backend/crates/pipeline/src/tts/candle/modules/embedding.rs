//! Embedding Modules for IndicF5
//!
//! Provides:
//! - TextEmbedding: Character/phoneme embeddings
//! - InputEmbedding: Combined input embedding with mel projection
//! - SinusoidalPositionalEmbedding: Time-step position encoding

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Module, Result, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn::{embedding, linear, Embedding, Linear, VarBuilder};

/// Text/Phoneme Embedding
///
/// Embeds discrete tokens (characters or phonemes) into continuous vectors.
#[cfg(feature = "candle")]
pub struct TextEmbedding {
    embedding: Embedding,
    dim: usize,
    vocab_size: usize,
}

#[cfg(feature = "candle")]
impl TextEmbedding {
    pub fn new(vocab_size: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(vocab_size, dim, vb.pp("embedding"))?;
        Ok(Self {
            embedding,
            dim,
            vocab_size,
        })
    }

    pub fn load(vocab_size: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(vocab_size, dim, vb)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(feature = "candle")]
impl Module for TextEmbedding {
    fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embedding.forward(token_ids)
    }
}

/// Input Embedding for F5-TTS
///
/// Combines:
/// 1. Text embedding (from phoneme tokens)
/// 2. Mel spectrogram projection
/// 3. Positional encoding (optional)
#[cfg(feature = "candle")]
pub struct InputEmbedding {
    text_embed: TextEmbedding,
    mel_proj: Linear,
    pos_embed: Option<super::conv::ConvPositionEmbedding>,
    dim: usize,
    n_mels: usize,
}

#[cfg(feature = "candle")]
impl InputEmbedding {
    pub fn new(
        vocab_size: usize,
        dim: usize,
        n_mels: usize,
        use_conv_pos: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let text_embed = TextEmbedding::new(vocab_size, dim, vb.pp("text_embed"))?;
        let mel_proj = linear(n_mels, dim, vb.pp("mel_proj"))?;

        let pos_embed = if use_conv_pos {
            Some(super::conv::ConvPositionEmbedding::new(
                dim,
                31,  // kernel_size
                16,  // groups
                vb.pp("pos_embed"),
            )?)
        } else {
            None
        };

        Ok(Self {
            text_embed,
            mel_proj,
            pos_embed,
            dim,
            n_mels,
        })
    }

    pub fn load(
        vocab_size: usize,
        dim: usize,
        n_mels: usize,
        use_conv_pos: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(vocab_size, dim, n_mels, use_conv_pos, vb)
    }

    /// Forward pass
    ///
    /// Args:
    ///   text_tokens: [batch, text_len] - phoneme token IDs
    ///   mel_spec: [batch, mel_len, n_mels] - mel spectrogram (reference audio)
    ///   lens: Optional lengths for masking
    pub fn forward(
        &self,
        text_tokens: &Tensor,
        mel_spec: &Tensor,
    ) -> Result<Tensor> {
        // Embed text
        let text_emb = self.text_embed.forward(text_tokens)?;

        // Project mel spectrogram
        let mel_emb = self.mel_proj.forward(mel_spec)?;

        // Concatenate along sequence dimension
        let combined = Tensor::cat(&[text_emb, mel_emb], 1)?;

        // Apply positional embedding if enabled
        if let Some(ref pos_embed) = self.pos_embed {
            pos_embed.forward(&combined)
        } else {
            Ok(combined)
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Sinusoidal Positional Embedding
///
/// Used for encoding timestep information in diffusion models.
/// PE(pos, 2i) = sin(pos / 10000^(2i/d))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
#[cfg(feature = "candle")]
pub struct SinusoidalPositionalEmbedding {
    dim: usize,
    max_period: f32,
}

#[cfg(feature = "candle")]
impl SinusoidalPositionalEmbedding {
    pub fn new(dim: usize, max_period: f32) -> Self {
        Self { dim, max_period }
    }

    /// Compute sinusoidal embedding for given timesteps
    ///
    /// Args:
    ///   timesteps: [batch] - float timesteps in [0, 1]
    ///   device: target device
    pub fn forward(&self, timesteps: &Tensor, device: &Device) -> Result<Tensor> {
        let half_dim = self.dim / 2;

        // Compute frequencies
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| {
                let exp = -((i as f32) * (self.max_period.ln()) / (half_dim as f32 - 1.0));
                exp.exp()
            })
            .collect();
        let freqs = Tensor::from_vec(freqs, (1, half_dim), device)?;

        // timesteps: [batch] -> [batch, 1]
        let timesteps = timesteps.unsqueeze(D::Minus1)?;

        // args: [batch, half_dim]
        let args = timesteps.broadcast_mul(&freqs)?;

        // Compute sin and cos
        let sin_emb = args.sin()?;
        let cos_emb = args.cos()?;

        // Interleave sin and cos: [batch, dim]
        Tensor::cat(&[sin_emb, cos_emb], D::Minus1)
    }
}

/// Time Embedding (for diffusion timestep conditioning)
///
/// Embeds continuous timesteps using sinusoidal encoding followed by MLP.
#[cfg(feature = "candle")]
pub struct TimeEmbedding {
    sinusoidal: SinusoidalPositionalEmbedding,
    mlp: TimeMLP,
}

#[cfg(feature = "candle")]
struct TimeMLP {
    fc1: Linear,
    fc2: Linear,
}

#[cfg(feature = "candle")]
impl TimeEmbedding {
    pub fn new(dim: usize, time_dim: usize, vb: VarBuilder) -> Result<Self> {
        let sinusoidal = SinusoidalPositionalEmbedding::new(dim, 10000.0);
        let fc1 = linear(dim, time_dim, vb.pp("mlp.fc1"))?;
        let fc2 = linear(time_dim, time_dim, vb.pp("mlp.fc2"))?;

        Ok(Self {
            sinusoidal,
            mlp: TimeMLP { fc1, fc2 },
        })
    }

    pub fn load(dim: usize, time_dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, time_dim, vb)
    }

    pub fn forward(&self, timesteps: &Tensor, device: &Device) -> Result<Tensor> {
        let emb = self.sinusoidal.forward(timesteps, device)?;
        let emb = self.mlp.fc1.forward(&emb)?;
        let emb = emb.silu()?;
        self.mlp.fc2.forward(&emb)
    }
}

/// Duration Embedding
///
/// Embeds duration information (for duration prediction in TTS).
#[cfg(feature = "candle")]
pub struct DurationEmbedding {
    embedding: Embedding,
    max_duration: usize,
    dim: usize,
}

#[cfg(feature = "candle")]
impl DurationEmbedding {
    pub fn new(max_duration: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = embedding(max_duration + 1, dim, vb.pp("embedding"))?;
        Ok(Self {
            embedding,
            max_duration,
            dim,
        })
    }

    pub fn load(max_duration: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(max_duration, dim, vb)
    }

    /// Embed durations, clamping to max_duration
    pub fn forward(&self, durations: &Tensor) -> Result<Tensor> {
        // Clamp durations to valid range
        let max_tensor = Tensor::new(self.max_duration as i64, durations.device())?;
        let clamped = durations.minimum(&max_tensor)?;
        self.embedding.forward(&clamped)
    }
}

// Non-Candle stubs for compilation
#[cfg(not(feature = "candle"))]
pub struct TextEmbedding;

#[cfg(not(feature = "candle"))]
pub struct InputEmbedding;

#[cfg(not(feature = "candle"))]
pub struct SinusoidalPositionalEmbedding;

#[cfg(not(feature = "candle"))]
pub struct TimeEmbedding;

#[cfg(not(feature = "candle"))]
pub struct DurationEmbedding;

#[cfg(test)]
mod tests {
    #[test]
    fn test_embedding_module_exists() {
        // Just verify the module compiles
    }
}
