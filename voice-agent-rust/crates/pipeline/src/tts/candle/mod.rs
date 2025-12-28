//! Candle-based TTS Implementation for IndicF5
//!
//! This module provides a pure Rust implementation of the F5-TTS model using
//! HuggingFace Candle for neural network inference with SafeTensors weights.
//!
//! # Architecture
//!
//! The F5-TTS model consists of:
//! 1. **Input Embedding**: Text (phoneme) and mel spectrogram embeddings
//! 2. **DiT Backbone**: Transformer blocks with AdaLayerNorm for time conditioning
//! 3. **Flow Matching**: ODE-based generation using Sway sampling
//! 4. **Vocos Vocoder**: ConvNeXt-based mel-to-waveform synthesis
//!
//! # Usage
//!
//! ```rust,ignore
//! use voice_agent_pipeline::tts::candle::{IndicF5Model, IndicF5Config};
//!
//! let config = IndicF5Config::indicf5_hindi();
//! let model = IndicF5Model::load("models/tts/IndicF5/model.safetensors", config)?;
//!
//! let audio = model.synthesize("नमस्ते", &reference_audio)?;
//! ```

pub mod config;
pub mod modules;
pub mod dit;
pub mod flow_matching;
pub mod vocos;
pub mod mel;
pub mod indicf5;

// Re-export main types
pub use config::{IndicF5Config, VocosConfig, FlowMatchingConfig};
pub use modules::*;

#[cfg(feature = "candle")]
pub use dit::DiTBackbone;
#[cfg(feature = "candle")]
pub use flow_matching::FlowMatcher;
#[cfg(feature = "candle")]
pub use vocos::VocosVocoder;
#[cfg(feature = "candle")]
pub use indicf5::IndicF5Model;

// Non-Candle stubs
#[cfg(not(feature = "candle"))]
pub struct DiTBackbone;

#[cfg(not(feature = "candle"))]
pub struct FlowMatcher;

#[cfg(not(feature = "candle"))]
pub struct VocosVocoder;

#[cfg(not(feature = "candle"))]
pub struct IndicF5Model;
