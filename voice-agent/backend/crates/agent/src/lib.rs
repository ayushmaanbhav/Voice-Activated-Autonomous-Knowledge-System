//! Conversational Agent Framework
//!
//! Features:
//! - Stage-based dialog management
//! - Intent detection and slot filling
//! - Conversation memory (hierarchical)
//! - Tool orchestration
//! - Persona-aware response generation
//! - Voice session integration with STT/TTS
//! - WebRTC/WebSocket transport integration
//! - P2 FIX: Persuasion engine for objection handling

pub mod conversation;
pub mod memory;
pub mod stage;
pub mod intent;
pub mod agent;
pub mod voice_session;
// P2 FIX: Persuasion engine for objection handling
pub mod persuasion;

pub use conversation::{Conversation, ConversationConfig, ConversationEvent};
pub use memory::{ConversationMemory, MemoryConfig, MemoryEntry};
pub use stage::{StageManager, ConversationStage, StageTransition, RagTimingStrategy, TransitionReason};
pub use intent::{IntentDetector, Intent, Slot, DetectedIntent};
// P2 FIX: Persuasion engine exports
pub use persuasion::{
    PersuasionEngine, ObjectionType, ObjectionResponse, ValueProposition,
    CompetitorComparison, SwitchSavings, PersuasionScript,
};
pub use agent::{GoldLoanAgent, AgentConfig, AgentEvent};
pub use voice_session::{VoiceSession, VoiceSessionConfig, VoiceSessionState, VoiceSessionEvent};

// Re-export transport types for convenience
pub use voice_agent_transport::{
    TransportSession, SessionConfig, TransportEvent,
    WebRtcConfig, WebSocketConfig, AudioFormat, AudioCodec,
};

// Re-export VAD and STT types for convenience
pub use voice_agent_pipeline::vad::{
    SileroVad, SileroConfig, VadState, VadResult, VadEngine, VadConfig,
};
pub use voice_agent_pipeline::stt::{
    IndicConformerStt, IndicConformerConfig, StreamingStt, SttConfig, SttEngine,
};

// Re-export vad module for use in tests
pub mod vad {
    pub use voice_agent_pipeline::vad::*;
}

use thiserror::Error;

/// Agent errors
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Conversation error: {0}")]
    Conversation(String),

    #[error("Stage error: {0}")]
    Stage(String),

    #[error("Intent error: {0}")]
    Intent(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Timeout")]
    Timeout,
}

impl From<voice_agent_pipeline::PipelineError> for AgentError {
    fn from(err: voice_agent_pipeline::PipelineError) -> Self {
        AgentError::Pipeline(err.to_string())
    }
}

impl From<voice_agent_llm::LlmError> for AgentError {
    fn from(err: voice_agent_llm::LlmError) -> Self {
        AgentError::Llm(err.to_string())
    }
}

/// P2 FIX: Use ToolError instead of removed ToolsError.
impl From<voice_agent_tools::ToolError> for AgentError {
    fn from(err: voice_agent_tools::ToolError) -> Self {
        AgentError::Tool(err.to_string())
    }
}

impl From<voice_agent_transport::TransportError> for AgentError {
    fn from(err: voice_agent_transport::TransportError) -> Self {
        AgentError::Pipeline(format!("Transport error: {}", err))
    }
}
