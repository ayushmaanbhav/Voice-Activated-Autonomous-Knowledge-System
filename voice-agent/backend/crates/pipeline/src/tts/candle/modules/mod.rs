//! Neural Network Modules for IndicF5 TTS
//!
//! This module provides all the building blocks needed for the F5-TTS architecture:
//!
//! - **norm**: Layer normalization variants (LayerNorm, RMSNorm, AdaLayerNorm)
//! - **attention**: Self-attention with RoPE position encoding
//! - **feedforward**: MLP with GELU/SwiGLU activation
//! - **conv**: ConvNeXt V2 blocks and convolutional position embedding
//! - **embedding**: Text, mel, and time embeddings

pub mod norm;
pub mod attention;
pub mod feedforward;
pub mod conv;
pub mod embedding;

// Re-export commonly used types
pub use norm::{LayerNorm, RMSNorm, AdaLayerNorm, AdaLayerNormOutput};
pub use attention::{SelfAttention, RotaryEmbedding, QKVProjection};
pub use feedforward::{FeedForward, GatedFeedForward, Dropout};
pub use conv::{ConvNeXtV2Block, ConvPositionEmbedding, GRN, CausalConv1d};
pub use embedding::{
    TextEmbedding, InputEmbedding, SinusoidalPositionalEmbedding,
    TimeEmbedding, DurationEmbedding,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_modules_export() {
        // Verify all modules are accessible
    }
}
