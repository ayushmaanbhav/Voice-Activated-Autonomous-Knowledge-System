//! Conversational Agent Framework
//!
//! Features:
//! - Stage-based dialog management
//! - Intent detection and slot filling
//! - Conversation memory (hierarchical)
//! - Tool orchestration
//! - Persona-aware response generation
//! - Voice session integration with STT/TTS

pub mod conversation;
pub mod memory;
pub mod stage;
pub mod intent;
pub mod agent;
pub mod voice_session;

pub use conversation::{Conversation, ConversationConfig, ConversationEvent};
pub use memory::{ConversationMemory, MemoryConfig, MemoryEntry};
pub use stage::{StageManager, ConversationStage, StageTransition};
pub use intent::{IntentDetector, Intent, Slot, DetectedIntent};
pub use agent::{GoldLoanAgent, AgentConfig, AgentEvent};
pub use voice_session::{VoiceSession, VoiceSessionConfig, VoiceSessionState, VoiceSessionEvent};

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
