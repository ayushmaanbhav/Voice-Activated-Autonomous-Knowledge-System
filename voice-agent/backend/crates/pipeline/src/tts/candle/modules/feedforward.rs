//! Feedforward Networks for IndicF5
//!
//! Provides:
//! - FeedForward: Standard MLP with GELU activation
//! - GatedFeedForward: GLU-style gated feedforward (SwiGLU variant)

#[cfg(feature = "candle")]
use candle_core::{Module, Result, Tensor};
#[cfg(feature = "candle")]
use candle_nn::{linear, Linear, VarBuilder};

/// Standard Feedforward Network with GELU activation
///
/// Architecture: Linear -> GELU -> Linear
#[cfg(feature = "candle")]
pub struct FeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout: f32,
}

#[cfg(feature = "candle")]
impl FeedForward {
    pub fn new(dim: usize, mult: f32, dropout: f32, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (dim as f32 * mult) as usize;
        let fc1 = linear(dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, dim, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2, dropout })
    }

    pub fn load(dim: usize, mult: f32, dropout: f32, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, mult, dropout, vb)
    }
}

#[cfg(feature = "candle")]
impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = gelu(&x)?;
        // Note: dropout is typically only applied during training
        self.fc2.forward(&x)
    }
}

/// Gated Feedforward with SwiGLU activation
///
/// Architecture: (Linear * SiLU(Linear)) -> Linear
/// Used in more recent transformer architectures
#[cfg(feature = "candle")]
pub struct GatedFeedForward {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

#[cfg(feature = "candle")]
impl GatedFeedForward {
    pub fn new(dim: usize, mult: f32, vb: VarBuilder) -> Result<Self> {
        let hidden_dim = (dim as f32 * mult) as usize;
        // Often hidden_dim is adjusted to be multiple of 256 for efficiency
        let hidden_dim = ((hidden_dim + 255) / 256) * 256;

        let gate_proj = linear(dim, hidden_dim, vb.pp("gate_proj"))?;
        let up_proj = linear(dim, hidden_dim, vb.pp("up_proj"))?;
        let down_proj = linear(hidden_dim, dim, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn load(dim: usize, mult: f32, vb: VarBuilder) -> Result<Self> {
        Self::new(dim, mult, vb)
    }
}

#[cfg(feature = "candle")]
impl Module for GatedFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = silu(&gate)?;
        let up = self.up_proj.forward(x)?;
        let x = gate.mul(&up)?;
        self.down_proj.forward(&x)
    }
}

/// GELU activation function (Gaussian Error Linear Unit)
///
/// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
/// Approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
#[cfg(feature = "candle")]
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    // Use Candle's built-in GELU
    x.gelu_erf()
}

/// SiLU (Swish) activation function
///
/// SiLU(x) = x * sigmoid(x)
#[cfg(feature = "candle")]
pub fn silu(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

/// ReLU activation function
#[cfg(feature = "candle")]
pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

/// Dropout layer (typically identity during inference)
#[cfg(feature = "candle")]
pub struct Dropout {
    prob: f32,
}

#[cfg(feature = "candle")]
impl Dropout {
    pub fn new(prob: f32) -> Self {
        Self { prob }
    }
}

#[cfg(feature = "candle")]
impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // During inference, dropout is identity
        // For training, we'd need to randomly zero elements and scale
        Ok(x.clone())
    }
}

// Non-Candle stubs for compilation
#[cfg(not(feature = "candle"))]
pub struct FeedForward;

#[cfg(not(feature = "candle"))]
pub struct GatedFeedForward;

#[cfg(not(feature = "candle"))]
pub struct Dropout;

#[cfg(test)]
mod tests {
    #[test]
    fn test_feedforward_module_exists() {
        // Just verify the module compiles
    }
}
