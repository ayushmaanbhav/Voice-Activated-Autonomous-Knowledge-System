//! Audio pipeline with VAD, STT, TTS, and turn detection
//!
//! This crate provides the core audio processing pipeline:
//! - Voice Activity Detection (MagicNet-inspired)
//! - Semantic Turn Detection (HybridTurnDetector)
//! - Streaming Speech-to-Text
//! - Streaming Text-to-Speech with word-level chunking
//! - Barge-in handling
//! - Frame processors (SentenceDetector, InterruptHandler)
//! - Channel-based processor chains

pub mod adapters;
pub mod orchestrator;
pub mod processors;
pub mod stt;
pub mod tts;
pub mod turn_detection;
pub mod vad;

// VAD exports
pub use vad::{VadConfig, VadResult, VadState, VoiceActivityDetector};

// Turn detection exports
pub use turn_detection::{
    HybridTurnDetector, SemanticTurnDetector, TurnDetectionConfig, TurnDetectionResult, TurnState,
};

// STT exports
pub use stt::{DecoderConfig, EnhancedDecoder, StreamingStt, SttConfig, SttEngine};

// TTS exports
pub use tts::{ChunkStrategy, StreamingTts, TtsConfig, TtsEngine, TtsEvent, WordChunker};
// P1-3 FIX: Export TTS backend types and factory
pub use tts::{create_tts_backend, StubTtsBackend, TtsBackend};
#[cfg(feature = "candle")]
pub use tts::{IndicF5Backend, IndicF5Config, IndicF5Model};

// Orchestrator exports
pub use orchestrator::{
    BargeInAction,
    BargeInConfig,
    PipelineConfig,
    PipelineEvent,
    PipelineState,
    // P1 FIX: Export processor chain config for external configuration
    ProcessorChainConfig,
    VoicePipeline,
};

// Processor exports
pub use processors::{
    InterruptHandler, InterruptHandlerConfig, InterruptMode, ProcessorChain, ProcessorChainBuilder,
    SentenceDetector, SentenceDetectorConfig, TtsProcessor, TtsProcessorConfig,
    // P2-2 FIX: Export generic processors for extensibility
    FilterProcessor, MapProcessor, PassthroughProcessor,
};

// P3 FIX: Trait adapter exports - bridge internal STT/TTS with core traits
pub use adapters::{
    create_passthrough_processor,
    create_stt_adapter,
    create_tts_adapter,
    // P2-2: Passthrough audio processor (placeholder for future AEC/NS/AGC)
    PassthroughAudioProcessor,
    SttAdapter,
    TtsAdapter,
    // P2-1 FIX: Noise suppression processor
    create_noise_suppressor,
};
#[cfg(feature = "noise-suppression")]
pub use adapters::NoiseSuppressorProcessor;

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
            // P2 FIX: Use proper variants now that core has Audio, Io, Model
            PipelineError::Model(msg) => CorePipelineError::Model(msg),
            PipelineError::ChannelClosed => CorePipelineError::ChannelClosed,
            PipelineError::Timeout => CorePipelineError::Timeout(0),
            PipelineError::NotInitialized => CorePipelineError::NotInitialized,
            PipelineError::Audio(msg) => CorePipelineError::Audio(msg),
            PipelineError::Io(msg) => CorePipelineError::Io(msg),
        };

        voice_agent_core::Error::Pipeline(core_err)
    }
}
