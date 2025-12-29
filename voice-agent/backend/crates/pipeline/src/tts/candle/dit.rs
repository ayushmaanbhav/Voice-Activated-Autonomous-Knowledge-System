//! DiT (Diffusion Transformer) Backbone for IndicF5
//!
//! The DiT architecture uses transformer blocks with adaptive layer normalization
//! for conditioning on time/noise level. This is the core of the F5-TTS model.

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Module, Result, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn::{linear, Linear, VarBuilder};

#[cfg(feature = "candle")]
use super::config::IndicF5Config;
#[cfg(feature = "candle")]
use super::modules::{
    AdaLayerNorm, AdaLayerNormOutput, FeedForward, SelfAttention,
    ConvNeXtV2Block, ConvPositionEmbedding, TimeEmbedding, InputEmbedding,
};

/// Single DiT Block with adaptive normalization
///
/// Architecture:
/// 1. AdaLayerNorm (produces modulation for attention and FFN)
/// 2. Self-Attention with optional RoPE
/// 3. Gate and residual for attention output
/// 4. LayerNorm
/// 5. FeedForward
/// 6. Gate and residual for FFN output
#[cfg(feature = "candle")]
pub struct DiTBlock {
    attn_norm: AdaLayerNorm,
    attn: SelfAttention,
    ff_norm: super::modules::LayerNorm,
    ff: FeedForward,
    dim: usize,
}

#[cfg(feature = "candle")]
impl DiTBlock {
    pub fn new(config: &IndicF5Config, vb: VarBuilder) -> Result<Self> {
        let attn_norm = AdaLayerNorm::new(config.dim, config.layer_norm_eps, vb.pp("attn_norm"))?;
        let attn = SelfAttention::new(
            config.dim,
            config.heads,
            config.use_rope,
            config.max_seq_len,
            config.rope_base,
            vb.pp("attn"),
        )?;
        let ff_norm = super::modules::LayerNorm::new(
            config.dim,
            config.layer_norm_eps,
            vb.pp("ff_norm"),
        )?;
        let ff = FeedForward::new(config.dim, config.ff_mult, config.dropout, vb.pp("ff"))?;

        Ok(Self {
            attn_norm,
            attn,
            ff_norm,
            ff,
            dim: config.dim,
        })
    }

    pub fn load(config: &IndicF5Config, vb: VarBuilder) -> Result<Self> {
        let attn_norm = AdaLayerNorm::load(config.dim, config.layer_norm_eps, vb.pp("attn_norm"))?;
        let attn = SelfAttention::load(
            config.dim,
            config.heads,
            config.use_rope,
            config.max_seq_len,
            config.rope_base,
            vb.pp("attn"),
        )?;
        let ff_norm = super::modules::LayerNorm::load(
            config.dim,
            config.layer_norm_eps,
            vb.pp("ff_norm"),
        )?;
        let ff = FeedForward::load(config.dim, config.ff_mult, config.dropout, vb.pp("ff"))?;

        Ok(Self {
            attn_norm,
            attn,
            ff_norm,
            ff,
            dim: config.dim,
        })
    }

    /// Forward pass through the DiT block
    ///
    /// Args:
    ///   x: [batch, seq_len, dim] - input features
    ///   time_cond: [batch, dim] - time conditioning
    ///   mask: Optional attention mask
    pub fn forward(
        &self,
        x: &Tensor,
        time_cond: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Get adaptive normalization outputs
        let AdaLayerNormOutput {
            x_norm,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        } = self.attn_norm.forward(x, time_cond)?;

        // Modulate normalized input for attention
        let x_modulated = AdaLayerNorm::modulate(&x_norm, &shift_msa, &scale_msa)?;

        // Self-attention
        let attn_out = self.attn.forward(&x_modulated, mask, 0)?;

        // Gate and residual
        let x = x.broadcast_add(&attn_out.broadcast_mul(&gate_msa)?)?;

        // FFN path
        let x_norm = self.ff_norm.forward(&x)?;
        let x_modulated = AdaLayerNorm::modulate(&x_norm, &shift_mlp, &scale_mlp)?;
        let ff_out = self.ff.forward(&x_modulated)?;

        // Gate and residual
        x.broadcast_add(&ff_out.broadcast_mul(&gate_mlp)?)
    }
}

/// Full DiT Backbone
///
/// Stack of DiT blocks with input/output projections and time conditioning.
#[cfg(feature = "candle")]
pub struct DiTBackbone {
    /// Input embedding (text + mel)
    input_embed: InputEmbedding,

    /// Time embedding
    time_embed: TimeEmbedding,

    /// Convolutional position embedding
    pos_embed: ConvPositionEmbedding,

    /// Pre-processing ConvNeXt blocks
    pre_conv_blocks: Vec<ConvNeXtV2Block>,

    /// Main transformer blocks
    transformer_blocks: Vec<DiTBlock>,

    /// Post-processing ConvNeXt blocks
    post_conv_blocks: Vec<ConvNeXtV2Block>,

    /// Output projection to mel dimension
    output_proj: Linear,

    /// Configuration
    config: IndicF5Config,
}

#[cfg(feature = "candle")]
impl DiTBackbone {
    pub fn new(config: IndicF5Config, vb: VarBuilder) -> Result<Self> {
        // Input embedding
        let input_embed = InputEmbedding::new(
            config.vocab_size,
            config.dim,
            config.n_mels,
            false, // pos embed handled separately
            vb.pp("input_embed"),
        )?;

        // Time embedding
        let time_embed = TimeEmbedding::new(config.dim, config.time_dim, vb.pp("time_embed"))?;

        // Convolutional position embedding
        let pos_embed = ConvPositionEmbedding::new(config.dim, 31, 16, vb.pp("pos_embed"))?;

        // Pre-processing ConvNeXt blocks
        let mut pre_conv_blocks = Vec::with_capacity(config.conv_layers);
        for i in 0..config.conv_layers {
            let block = ConvNeXtV2Block::new(
                config.dim,
                config.ff_dim(),
                config.conv_kernel_size,
                vb.pp(&format!("pre_conv.{}", i)),
            )?;
            pre_conv_blocks.push(block);
        }

        // Main transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let block = DiTBlock::new(&config, vb.pp(&format!("transformer_blocks.{}", i)))?;
            transformer_blocks.push(block);
        }

        // Post-processing ConvNeXt blocks
        let mut post_conv_blocks = Vec::with_capacity(config.conv_layers);
        for i in 0..config.conv_layers {
            let block = ConvNeXtV2Block::new(
                config.dim,
                config.ff_dim(),
                config.conv_kernel_size,
                vb.pp(&format!("post_conv.{}", i)),
            )?;
            post_conv_blocks.push(block);
        }

        // Output projection
        let output_proj = linear(config.dim, config.n_mels, vb.pp("output_proj"))?;

        Ok(Self {
            input_embed,
            time_embed,
            pos_embed,
            pre_conv_blocks,
            transformer_blocks,
            post_conv_blocks,
            output_proj,
            config,
        })
    }

    pub fn load(config: IndicF5Config, vb: VarBuilder) -> Result<Self> {
        // Input embedding
        let input_embed = InputEmbedding::load(
            config.vocab_size,
            config.dim,
            config.n_mels,
            false,
            vb.pp("input_embed"),
        )?;

        // Time embedding
        let time_embed = TimeEmbedding::load(config.dim, config.time_dim, vb.pp("time_embed"))?;

        // Convolutional position embedding
        let pos_embed = ConvPositionEmbedding::load(config.dim, 31, 16, vb.pp("pos_embed"))?;

        // Pre-processing ConvNeXt blocks
        let mut pre_conv_blocks = Vec::with_capacity(config.conv_layers);
        for i in 0..config.conv_layers {
            let block = ConvNeXtV2Block::load(
                config.dim,
                config.ff_dim(),
                config.conv_kernel_size,
                vb.pp(&format!("pre_conv.{}", i)),
            )?;
            pre_conv_blocks.push(block);
        }

        // Main transformer blocks
        let mut transformer_blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let block = DiTBlock::load(&config, vb.pp(&format!("transformer_blocks.{}", i)))?;
            transformer_blocks.push(block);
        }

        // Post-processing ConvNeXt blocks
        let mut post_conv_blocks = Vec::with_capacity(config.conv_layers);
        for i in 0..config.conv_layers {
            let block = ConvNeXtV2Block::load(
                config.dim,
                config.ff_dim(),
                config.conv_kernel_size,
                vb.pp(&format!("post_conv.{}", i)),
            )?;
            post_conv_blocks.push(block);
        }

        // Output projection
        let output_proj = linear(config.dim, config.n_mels, vb.pp("output_proj"))?;

        Ok(Self {
            input_embed,
            time_embed,
            pos_embed,
            pre_conv_blocks,
            transformer_blocks,
            post_conv_blocks,
            output_proj,
            config,
        })
    }

    /// Forward pass through the DiT backbone
    ///
    /// Args:
    ///   text_tokens: [batch, text_len] - phoneme token IDs
    ///   mel_spec: [batch, mel_len, n_mels] - noisy mel spectrogram
    ///   timestep: [batch] - diffusion timestep in [0, 1]
    ///   mask: Optional attention mask
    ///
    /// Returns:
    ///   [batch, mel_len, n_mels] - predicted velocity/noise
    pub fn forward(
        &self,
        text_tokens: &Tensor,
        mel_spec: &Tensor,
        timestep: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = text_tokens.device();

        // Embed inputs
        let x = self.input_embed.forward(text_tokens, mel_spec)?;

        // Add position embedding
        let x = self.pos_embed.forward(&x)?;

        // Time conditioning
        let time_cond = self.time_embed.forward(timestep, device)?;

        // Pre-processing convolutions
        let mut x = x;
        for block in &self.pre_conv_blocks {
            x = block.forward(&x)?;
        }

        // Main transformer blocks
        for block in &self.transformer_blocks {
            x = block.forward(&x, &time_cond, mask)?;
        }

        // Post-processing convolutions
        for block in &self.post_conv_blocks {
            x = block.forward(&x)?;
        }

        // Project to mel dimension
        self.output_proj.forward(&x)
    }

    /// Get the expected mel sequence length from input
    pub fn mel_length(&self, text_len: usize, ref_mel_len: usize) -> usize {
        text_len + ref_mel_len
    }

    /// Get configuration
    pub fn config(&self) -> &IndicF5Config {
        &self.config
    }
}

/// Simple DiT without ConvNeXt blocks (for testing)
#[cfg(feature = "candle")]
pub struct SimpleDiT {
    transformer_blocks: Vec<DiTBlock>,
    output_proj: Linear,
    config: IndicF5Config,
}

#[cfg(feature = "candle")]
impl SimpleDiT {
    pub fn new(config: IndicF5Config, vb: VarBuilder) -> Result<Self> {
        let mut transformer_blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let block = DiTBlock::new(&config, vb.pp(&format!("blocks.{}", i)))?;
            transformer_blocks.push(block);
        }

        let output_proj = linear(config.dim, config.n_mels, vb.pp("output_proj"))?;

        Ok(Self {
            transformer_blocks,
            output_proj,
            config,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        time_cond: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.transformer_blocks {
            x = block.forward(&x, time_cond, mask)?;
        }
        self.output_proj.forward(&x)
    }
}

// Non-Candle stubs
#[cfg(not(feature = "candle"))]
pub struct DiTBlock;

#[cfg(not(feature = "candle"))]
pub struct DiTBackbone;

#[cfg(not(feature = "candle"))]
pub struct SimpleDiT;

#[cfg(test)]
mod tests {
    #[test]
    fn test_dit_module_exists() {
        // Just verify the module compiles
    }
}
