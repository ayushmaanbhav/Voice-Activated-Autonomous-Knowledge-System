//! Streaming Text-to-Speech
//!
//! Features:
//! - Word-level chunking for early audio emission
//! - Barge-in aware (can stop mid-word)
//! - Multiple backend support (Piper, IndicF5, Parler)
//! - Hindi/Hinglish G2P conversion
//! - Native Candle-based IndicF5 model (optional)

mod streaming;
mod chunker;
mod g2p;

/// Candle-based TTS implementations (native Rust with SafeTensors)
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(not(feature = "candle"))]
pub mod candle {
    //! Stub module when candle feature is disabled
    pub struct IndicF5Model;
    pub struct IndicF5Config;
}

pub use streaming::{StreamingTts, TtsConfig, TtsEngine, TtsEvent};
pub use chunker::{WordChunker, ChunkStrategy};
pub use g2p::{HindiG2p, G2pConfig, Language, Phoneme, create_hindi_g2p};

use crate::PipelineError;

/// TTS backend trait
#[async_trait::async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize text to audio
    async fn synthesize(&self, text: &str) -> Result<Vec<f32>, PipelineError>;

    /// Get sample rate
    fn sample_rate(&self) -> u32;

    /// Supports streaming word-by-word?
    fn supports_streaming(&self) -> bool;
}
