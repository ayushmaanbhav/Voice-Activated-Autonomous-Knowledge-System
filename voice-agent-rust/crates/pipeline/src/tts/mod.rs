//! Streaming Text-to-Speech
//!
//! Features:
//! - Word-level chunking for early audio emission
//! - Barge-in aware (can stop mid-word)
//! - Multiple backend support (Piper, IndicF5, Parler)
//! - Hindi/Hinglish G2P conversion

mod streaming;
mod chunker;
mod g2p;

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
