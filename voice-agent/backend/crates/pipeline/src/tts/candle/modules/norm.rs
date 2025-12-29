//! Normalization Layers for IndicF5
//!
//! Provides:
//! - LayerNorm: Standard layer normalization
//! - RMSNorm: Root mean square normalization (used in LLaMA)
//! - AdaLayerNorm: Adaptive layer norm with time conditioning (for DiT)

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Module, Result, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{linear, Linear, VarBuilder};

/// Standard Layer Normalization
#[cfg(feature = "candle")]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

#[cfg(feature = "candle")]
impl LayerNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(dim, "bias", candle_nn::Init::Const(0.0))?;
        Ok(Self { weight, bias, eps })
    }

    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self { weight, bias, eps })
    }
}

#[cfg(feature = "candle")]
impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let out = x_norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?;
        out.to_dtype(x_dtype)
    }
}

/// RMS Normalization (Root Mean Square Layer Normalization)
#[cfg(feature = "candle")]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

#[cfg(feature = "candle")]
impl RMSNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?;
        Ok(Self { weight, eps })
    }

    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

#[cfg(feature = "candle")]
impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let rms = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?.sqrt()?;
        let x_norm = x.broadcast_div(&(rms + self.eps)?)?;
        let out = x_norm.broadcast_mul(&self.weight)?;
        out.to_dtype(x_dtype)
    }
}

/// Adaptive Layer Normalization (for DiT)
///
/// Uses time conditioning to modulate the normalization.
/// Outputs: shift, scale, gate for MSA and MLP branches
#[cfg(feature = "candle")]
pub struct AdaLayerNorm {
    linear: Linear,
    norm: LayerNorm,
    dim: usize,
}

#[cfg(feature = "candle")]
impl AdaLayerNorm {
    /// Create a new AdaLayerNorm
    ///
    /// The linear layer projects from dim to dim * 6 for:
    /// shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let linear = linear(dim, dim * 6, vb.pp("linear"))?;
        let norm = LayerNorm::new(dim, eps, vb.pp("norm"))?;
        Ok(Self { linear, norm, dim })
    }

    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let linear = linear(dim, dim * 6, vb.pp("linear"))?;
        let norm = LayerNorm::load(dim, eps, vb.pp("norm"))?;
        Ok(Self { linear, norm, dim })
    }

    /// Forward pass
    ///
    /// Returns (normalized_x, shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    pub fn forward(&self, x: &Tensor, conditioning: &Tensor) -> Result<AdaLayerNormOutput> {
        // Project conditioning to 6 * dim
        let modulation = self.linear.forward(conditioning)?;

        // Split into 6 parts
        let chunks = modulation.chunk(6, candle_core::D::Minus1)?;

        let shift_msa = chunks[0].clone();
        let scale_msa = chunks[1].clone();
        let gate_msa = chunks[2].clone();
        let shift_mlp = chunks[3].clone();
        let scale_mlp = chunks[4].clone();
        let gate_mlp = chunks[5].clone();

        // Apply normalization
        let x_norm = self.norm.forward(x)?;

        Ok(AdaLayerNormOutput {
            x_norm,
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        })
    }

    /// Apply modulation to input: x * (1 + scale) + shift
    pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        let ones = Tensor::ones_like(scale)?;
        x.broadcast_mul(&ones.broadcast_add(scale)?)?.broadcast_add(shift)
    }
}

/// Output of AdaLayerNorm forward pass
#[cfg(feature = "candle")]
pub struct AdaLayerNormOutput {
    pub x_norm: Tensor,
    pub shift_msa: Tensor,
    pub scale_msa: Tensor,
    pub gate_msa: Tensor,
    pub shift_mlp: Tensor,
    pub scale_mlp: Tensor,
    pub gate_mlp: Tensor,
}

// Non-Candle stubs for compilation
#[cfg(not(feature = "candle"))]
pub struct LayerNorm;

#[cfg(not(feature = "candle"))]
pub struct RMSNorm;

#[cfg(not(feature = "candle"))]
pub struct AdaLayerNorm;

#[cfg(not(feature = "candle"))]
pub struct AdaLayerNormOutput;

#[cfg(test)]
mod tests {
    #[test]
    fn test_norm_module_exists() {
        // Just verify the module compiles
    }
}
