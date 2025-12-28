//! Audio pipeline with VAD, STT, TTS, and turn detection
//!
//! This crate provides the core audio processing pipeline:
//! - Voice Activity Detection (MagicNet-inspired)
//! - Semantic Turn Detection (HybridTurnDetector)
//! - Streaming Speech-to-Text
//! - Streaming Text-to-Speech with word-level chunking
//! - Barge-in handling

pub mod vad;
pub mod turn_detection;
pub mod stt;
pub mod tts;
pub mod orchestrator;

// VAD exports
pub use vad::{VoiceActivityDetector, VadConfig, VadState, VadResult};

// Turn detection exports
pub use turn_detection::{
    HybridTurnDetector, TurnDetectionConfig, TurnState, TurnDetectionResult,
    SemanticTurnDetector,
};

// STT exports
pub use stt::{StreamingStt, SttConfig, SttEngine, EnhancedDecoder, DecoderConfig};

// TTS exports
pub use tts::{StreamingTts, TtsConfig, TtsEngine, TtsEvent, WordChunker, ChunkStrategy};

// Orchestrator exports
pub use orchestrator::{VoicePipeline, PipelineConfig, PipelineEvent, PipelineState, BargeInConfig, BargeInAction};

use thiserror::Error;

/// Pipeline errors
#[derive(Error, Debug, Clone)]
pub enum PipelineError {
    #[error("VAD error: {0}")]
    Vad(String),

    #[error("Turn detection error: {0}")]
    TurnDetection(String),

    #[error("STT error: {0}")]
    Stt(String),

    #[error("TTS error: {0}")]
    Tts(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Channel closed")]
    ChannelClosed,

    #[error("Timeout")]
    Timeout,

    #[error("Not initialized")]
    NotInitialized,

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("IO error: {0}")]
    Io(String),
}

/// P2 FIX: Properly map each pipeline error variant to its corresponding core variant.
/// Previously all errors were converted to Vad, losing type information.
impl From<PipelineError> for voice_agent_core::Error {
    fn from(err: PipelineError) -> Self {
        use voice_agent_core::error::PipelineError as CorePipelineError;

        let core_err = match err {
            PipelineError::Vad(msg) => CorePipelineError::Vad(msg),
            PipelineError::TurnDetection(msg) => CorePipelineError::TurnDetection(msg),
            PipelineError::Stt(msg) => CorePipelineError::Stt(msg),
            PipelineError::Tts(msg) => CorePipelineError::Tts(msg),
            PipelineError::Model(msg) => CorePipelineError::Stt(format!("Model: {}", msg)),
            PipelineError::ChannelClosed => CorePipelineError::ChannelClosed,
            PipelineError::Timeout => CorePipelineError::Timeout(0),
            PipelineError::NotInitialized => CorePipelineError::NotInitialized,
            PipelineError::Audio(msg) => CorePipelineError::Vad(format!("Audio: {}", msg)),
            PipelineError::Io(msg) => CorePipelineError::Stt(format!("IO: {}", msg)),
        };

        voice_agent_core::Error::Pipeline(core_err)
    }
}
