//! Attention Mechanisms for IndicF5
//!
//! Provides:
//! - SelfAttention: Multi-head self-attention with optional RoPE
//! - RoPE: Rotary Position Embeddings

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Module, Result, Tensor, D};
#[cfg(feature = "candle")]
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

/// Rotary Position Embeddings (RoPE)
///
/// Applies rotation to query and key vectors based on position,
/// enabling the model to use relative position information.
#[cfg(feature = "candle")]
pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    base: f32,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

#[cfg(feature = "candle")]
impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32, device: &Device) -> Result<Self> {
        // Compute inverse frequencies: 1 / (base^(2i/dim))
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half_dim,), device)?;

        // Create position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len,), device)?;

        // Compute freqs: positions outer inv_freq
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;

        // Cache cos and sin values
        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            dim,
            max_seq_len,
            base,
            cos_cache,
            sin_cache,
        })
    }

    /// Apply rotary embeddings to query and key tensors
    ///
    /// Args:
    ///   q: [batch, heads, seq_len, head_dim]
    ///   k: [batch, heads, seq_len, head_dim]
    ///   offset: position offset for incremental decoding
    pub fn forward(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let head_dim = q.dim(3)?;

        // Get cached cos/sin for the required positions
        let cos = self.cos_cache.narrow(0, offset, seq_len)?;
        let sin = self.sin_cache.narrow(0, offset, seq_len)?;

        // Reshape for broadcasting: [1, 1, seq_len, head_dim/2]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let q_rotated = self.apply_rotary(q, &cos, &sin, head_dim)?;
        let k_rotated = self.apply_rotary(k, &cos, &sin, head_dim)?;

        Ok((q_rotated, k_rotated))
    }

    fn apply_rotary(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        head_dim: usize,
    ) -> Result<Tensor> {
        let half = head_dim / 2;

        // Split x into first half and second half
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;

        // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        let rotated_x1 = x1.broadcast_mul(cos)?.broadcast_sub(&x2.broadcast_mul(sin)?)?;
        let rotated_x2 = x1.broadcast_mul(sin)?.broadcast_add(&x2.broadcast_mul(cos)?)?;

        // Concatenate back
        Tensor::cat(&[rotated_x1, rotated_x2], D::Minus1)
    }
}

/// Multi-Head Self-Attention with optional RoPE
#[cfg(feature = "candle")]
pub struct SelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: Option<RotaryEmbedding>,
}

#[cfg(feature = "candle")]
impl SelfAttention {
    pub fn new(
        dim: usize,
        num_heads: usize,
        use_rope: bool,
        max_seq_len: usize,
        rope_base: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;

        let q_proj = linear(dim, dim, vb.pp("q_proj"))?;
        let k_proj = linear(dim, dim, vb.pp("k_proj"))?;
        let v_proj = linear(dim, dim, vb.pp("v_proj"))?;
        let out_proj = linear(dim, dim, vb.pp("out_proj"))?;

        let rope = if use_rope {
            Some(RotaryEmbedding::new(
                head_dim,
                max_seq_len,
                rope_base,
                vb.device(),
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            rope,
        })
    }

    pub fn load(
        dim: usize,
        num_heads: usize,
        use_rope: bool,
        max_seq_len: usize,
        rope_base: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Same as new, just different initialization semantics
        Self::new(dim, num_heads, use_rope, max_seq_len, rope_base, vb)
    }

    /// Forward pass
    ///
    /// Args:
    ///   x: [batch, seq_len, dim]
    ///   mask: Optional attention mask [batch, 1, seq_len, seq_len]
    ///   position_offset: Offset for RoPE (for incremental decoding)
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        position_offset: usize,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _dim) = x.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply RoPE if enabled
        let (q, k) = if let Some(ref rope) = self.rope {
            rope.forward(&q, &k, position_offset)?
        } else {
            (q, k)
        };

        // Scaled dot-product attention
        let scale = Tensor::new(self.scale, x.device())?;
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = attn_weights.broadcast_mul(&scale)?;

        // Apply mask if provided
        let attn_weights = if let Some(mask) = mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply attention to values
        let output = attn_weights.matmul(&v)?;

        // Transpose back: [batch, seq_len, num_heads, head_dim]
        let output = output.transpose(1, 2)?;

        // Reshape: [batch, seq_len, dim]
        let output = output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.out_proj.forward(&output)
    }
}

/// Joint Query-Key-Value projection (for efficiency)
#[cfg(feature = "candle")]
pub struct QKVProjection {
    qkv_proj: Linear,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
}

#[cfg(feature = "candle")]
impl QKVProjection {
    pub fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let qkv_proj = linear(dim, dim * 3, vb.pp("qkv_proj"))?;

        Ok(Self {
            qkv_proj,
            dim,
            num_heads,
            head_dim,
        })
    }

    /// Project input to Q, K, V and split
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;

        let qkv = self.qkv_proj.forward(x)?;

        // Split into Q, K, V
        let q = qkv.narrow(D::Minus1, 0, self.dim)?;
        let k = qkv.narrow(D::Minus1, self.dim, self.dim)?;
        let v = qkv.narrow(D::Minus1, self.dim * 2, self.dim)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let reshape_and_transpose = |t: Tensor| -> Result<Tensor> {
            t.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)
        };

        Ok((
            reshape_and_transpose(q)?,
            reshape_and_transpose(k)?,
            reshape_and_transpose(v)?,
        ))
    }
}

// Non-Candle stubs for compilation
#[cfg(not(feature = "candle"))]
pub struct RotaryEmbedding;

#[cfg(not(feature = "candle"))]
pub struct SelfAttention;

#[cfg(not(feature = "candle"))]
pub struct QKVProjection;

#[cfg(test)]
mod tests {
    #[test]
    fn test_attention_module_exists() {
        // Just verify the module compiles
    }
}
