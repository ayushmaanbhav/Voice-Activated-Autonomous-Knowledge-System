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
//! - P1-1 FIX: Agent trait abstraction for testability
//! - P1-2 FIX: Intent detection moved to text_processing crate

pub mod agent;
pub mod agent_config;
pub mod conversation;
pub mod memory;
// Legacy memory module for backward compatibility
pub mod memory_legacy;
pub mod stage;
pub mod voice_session;
// P2 FIX: Persuasion engine for objection handling
pub mod persuasion;
// P1-1 FIX: Agent trait abstraction
pub mod traits;
// P3 FIX: FSM adapter to bridge StageManager with core ConversationFSM trait
pub mod fsm_adapter;
// Phase 5: Dialogue State Tracking (DST)
pub mod dst;
// Phase 10: Lead Scoring for Sales Conversion
pub mod lead_scoring;

// P1-2 FIX: Re-export intent module from text_processing for backward compatibility
pub mod intent {
    //! Intent Detection and Slot Filling
    //!
    //! P1-2 FIX: This module re-exports from voice_agent_text_processing::intent.
    //! The canonical implementation is now in the text_processing crate.
    pub use voice_agent_text_processing::intent::*;
}

// Phase 2: Export ConversationContext trait for domain-agnostic agents
pub use conversation::{
    Conversation, ConversationConfig, ConversationContext, ConversationEvent,
    ConversationState, EndReason, ComplianceStatus, ConsentMethod, AiDisclosure, ConsentRecord,
};
pub use memory::MemoryConfig;
// Context compression types
pub use memory::{CompressionLevel, CompressionMethod, CompressionStats};
// Agentic memory types
pub use memory::{
    AgenticMemory, AgenticMemoryConfig, ArchivalMemory, ArchivalMemoryConfig,
    ConversationTurn, CoreMemory, MemoryNote, MemoryStats, MemoryType, RecallMemory, TurnRole,
};
pub use memory_legacy::{ConversationMemory, MemoryEntry};
pub use stage::{
    ConversationStage, RagTimingStrategy, StageManager, StageTransition, TransitionReason,
};
// P1-2 FIX: Re-export intent types from text_processing
pub use voice_agent_text_processing::intent::{
    DetectedIntent, Intent, IntentDetector, Slot, SlotType,
};
// P2 FIX: Persuasion engine exports
pub use agent::GoldLoanAgent;
/// P5 FIX: Type alias for domain-agnostic naming (GoldLoanAgent is now generic)
pub type DomainAgent = GoldLoanAgent;
// P1-SRP: Export agent config types
pub use agent_config::{
    AgentConfig, AgentEvent, PersonaTraits, SmallModelConfig, SpeculativeDecodingConfig,
    ToolDefaults, is_small_model,
};
// Phase 2: PersuasionStrategy trait for domain-agnostic persuasion handling
pub use persuasion::{
    CompetitorComparison, ObjectionResponse, ObjectionType, PersuasionEngine, PersuasionScript,
    PersuasionStrategy, SwitchSavings, ValueProposition,
};
pub use voice_session::{VoiceSession, VoiceSessionConfig, VoiceSessionEvent, VoiceSessionState};
// P1-1 FIX: Export Agent traits
pub use traits::{Agent, PersonalizableAgent, PrefetchingAgent};
// P3 FIX: Export FSM adapter
pub use fsm_adapter::{create_fsm_adapter, StageManagerAdapter};
// Phase 5: Export DST types
// Phase 2: DialogueState and DialogueStateTracking traits for domain-agnostic agents
pub use dst::{
    ChangeSource, DialogueStateTracker, DstConfig, GoldLoanDialogueState, SlotExtractor,
    SlotValue, StateChange, UrgencyLevel,
    // Phase 2: Domain-agnostic traits
    DialogueState, DialogueStateTracking,
    // Config-driven purity types
    PurityId, purity_ids, parse_purity_id, format_purity_display,
};
// Phase 10: Export Lead Scoring types
pub use lead_scoring::{
    EscalationTrigger, LeadClassification, LeadQualification, LeadRecommendation, LeadScore,
    LeadScoringConfig, LeadScoringEngine, LeadSignals, ScoreBreakdown, ScoreWeights, TrustLevel,
};

// Re-export transport types for convenience
pub use voice_agent_transport::{
    AudioCodec, AudioFormat, SessionConfig, TransportEvent, TransportSession, WebRtcConfig,
    WebSocketConfig,
};

// Re-export VAD and STT types for convenience
pub use voice_agent_pipeline::stt::{
    IndicConformerConfig, IndicConformerStt, StreamingStt, SttConfig, SttEngine,
};
pub use voice_agent_pipeline::vad::{
    SileroConfig, SileroVad, VadConfig, VadEngine, VadResult, VadState,
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

    /// P1-2 FIX: Initialization errors (e.g., speculative executor setup)
    #[error("Initialization error: {0}")]
    Initialization(String),
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
