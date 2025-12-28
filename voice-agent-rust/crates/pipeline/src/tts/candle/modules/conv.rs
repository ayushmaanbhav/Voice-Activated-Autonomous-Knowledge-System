//! Convolutional Modules for IndicF5
//!
//! Provides:
//! - ConvNeXtV2Block: Modern ConvNeXt V2 block with GRN
//! - ConvPositionEmbedding: 1D convolution for position encoding
//! - GRN: Global Response Normalization

#[cfg(feature = "candle")]
use candle_core::{DType, Module, Result, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, Linear, VarBuilder};

/// Global Response Normalization (GRN)
///
/// Introduced in ConvNeXt V2 to add global context and regularization.
/// GRN(x) = x + gamma * (x * norm(x)) / (mean(norm(x)) + eps) + beta
#[cfg(feature = "candle")]
pub struct GRN {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

#[cfg(feature = "candle")]
impl GRN {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get_with_hints(dim, "gamma", candle_nn::Init::Const(0.0))?;
        let beta = vb.get_with_hints(dim, "beta", candle_nn::Init::Const(0.0))?;
        Ok(Self {
            gamma,
            beta,
            eps: 1e-6,
        })
    }

    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        let beta = vb.get(dim, "beta")?;
        Ok(Self {
            gamma,
            beta,
            eps: 1e-6,
        })
    }
}

#[cfg(feature = "candle")]
impl Module for GRN {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, dim] or [batch, dim, seq_len]
        // Compute L2 norm across the spatial dimension
        let gx = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;

        // Normalize by mean of norms
        let norm_mean = gx.mean_keepdim(D::Minus2)?;
        let nx = gx.broadcast_div(&(norm_mean + self.eps)?)?;

        // Apply GRN: x + gamma * x * nx + beta
        let scaled = x.broadcast_mul(&nx)?;
        let weighted = scaled.broadcast_mul(&self.gamma)?;
        let biased = weighted.broadcast_add(&self.beta)?;
        x.broadcast_add(&biased)
    }
}

/// ConvNeXt V2 Block
///
/// Architecture:
/// 1. Depthwise Conv (kernel=7, groups=dim)
/// 2. LayerNorm
/// 3. Pointwise Conv (dim -> dim*4)
/// 4. GELU
/// 5. GRN
/// 6. Pointwise Conv (dim*4 -> dim)
/// 7. Residual connection
#[cfg(feature = "candle")]
pub struct ConvNeXtV2Block {
    dwconv: Conv1d,
    norm: super::norm::LayerNorm,
    pwconv1: Linear,
    grn: GRN,
    pwconv2: Linear,
    dim: usize,
}

#[cfg(feature = "candle")]
impl ConvNeXtV2Block {
    pub fn new(
        dim: usize,
        intermediate_dim: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Depthwise convolution
        let dwconv_config = Conv1dConfig {
            padding: kernel_size / 2,
            groups: dim,
            ..Default::default()
        };
        let dwconv = conv1d(dim, dim, kernel_size, dwconv_config, vb.pp("dwconv"))?;

        // LayerNorm
        let norm = super::norm::LayerNorm::new(dim, 1e-6, vb.pp("norm"))?;

        // Pointwise convolutions (implemented as linear)
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let grn = GRN::new(intermediate_dim, vb.pp("grn"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            grn,
            pwconv2,
            dim,
        })
    }

    pub fn load(
        dim: usize,
        intermediate_dim: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dwconv_config = Conv1dConfig {
            padding: kernel_size / 2,
            groups: dim,
            ..Default::default()
        };
        let dwconv = conv1d(dim, dim, kernel_size, dwconv_config, vb.pp("dwconv"))?;
        let norm = super::norm::LayerNorm::load(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let grn = GRN::load(intermediate_dim, vb.pp("grn"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            grn,
            pwconv2,
            dim,
        })
    }
}

#[cfg(feature = "candle")]
impl Module for ConvNeXtV2Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // x: [batch, seq_len, dim] -> [batch, dim, seq_len] for conv
        let x = x.transpose(1, 2)?;

        // Depthwise conv
        let x = self.dwconv.forward(&x)?;

        // Back to [batch, seq_len, dim]
        let x = x.transpose(1, 2)?;

        // LayerNorm
        let x = self.norm.forward(&x)?;

        // Pointwise conv 1 (expand)
        let x = self.pwconv1.forward(&x)?;

        // GELU
        let x = x.gelu_erf()?;

        // GRN
        let x = self.grn.forward(&x)?;

        // Pointwise conv 2 (project back)
        let x = self.pwconv2.forward(&x)?;

        // Residual
        x.add(&residual)
    }
}

/// Convolutional Position Embedding
///
/// Uses 1D convolution to inject position information into the sequence.
/// This is an alternative to sinusoidal or learned position embeddings.
#[cfg(feature = "candle")]
pub struct ConvPositionEmbedding {
    conv: Conv1d,
}

#[cfg(feature = "candle")]
impl ConvPositionEmbedding {
    pub fn new(dim: usize, kernel_size: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let config = Conv1dConfig {
            padding: kernel_size / 2,
            groups,
            ..Default::default()
        };
        let conv = conv1d(dim, dim, kernel_size, config, vb.pp("conv"))?;

        Ok(Self { conv })
    }

    pub fn load(dim: usize, kernel_size: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, kernel_size, groups, vb)
    }
}

#[cfg(feature = "candle")]
impl Module for ConvPositionEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq_len, dim]
        let x_transposed = x.transpose(1, 2)?; // [batch, dim, seq_len]
        let conv_out = self.conv.forward(&x_transposed)?;
        let conv_out = conv_out.transpose(1, 2)?; // [batch, seq_len, dim]

        // Add position embedding to input
        x.add(&conv_out)
    }
}

/// Causal Conv1D for autoregressive modeling
#[cfg(feature = "candle")]
pub struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
}

#[cfg(feature = "candle")]
impl CausalConv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // For causal convolution, we use left padding only
        let config = Conv1dConfig {
            padding: 0, // We'll do manual padding
            ..Default::default()
        };
        let conv = conv1d(in_channels, out_channels, kernel_size, config, vb.pp("conv"))?;

        Ok(Self { conv, kernel_size })
    }
}

#[cfg(feature = "candle")]
impl Module for CausalConv1d {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, channels, seq_len]
        // Pad left side only for causal convolution
        let padding = self.kernel_size - 1;
        let x = x.pad_with_zeros(D::Minus1, padding, 0)?;
        self.conv.forward(&x)
    }
}

// Non-Candle stubs for compilation
#[cfg(not(feature = "candle"))]
pub struct GRN;

#[cfg(not(feature = "candle"))]
pub struct ConvNeXtV2Block;

#[cfg(not(feature = "candle"))]
pub struct ConvPositionEmbedding;

#[cfg(not(feature = "candle"))]
pub struct CausalConv1d;

#[cfg(test)]
mod tests {
    #[test]
    fn test_conv_module_exists() {
        // Just verify the module compiles
    }
}
