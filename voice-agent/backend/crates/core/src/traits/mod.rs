//! Core traits for the voice agent system
//!
//! All major components implement these traits to enable:
//! - Pluggable backends (swap implementations without code changes)
//! - Testing with mocks
//! - Runtime switching based on configuration
//!
//! # Trait Hierarchy
//!
//! ```text
//! Speech Processing:
//!   - SpeechToText: Audio → Text transcription
//!   - TextToSpeech: Text → Audio synthesis
//!
//! Language Models:
//!   - LanguageModel: Text generation and tool calling
//!
//! Tools:
//!   - Tool: MCP-compatible tool interface
//!
//! Retrieval:
//!   - Retriever: Document retrieval for RAG
//!
//! Text Processing:
//!   - GrammarCorrector: Fix grammar while preserving domain terms
//!   - Translator: Translate between Indian languages
//!   - PIIRedactor: Detect and redact PII
//!   - ComplianceChecker: Check for regulatory compliance
//!
//! Conversation:
//!   - ConversationFSM: Finite state machine for conversation flow
//!
//! Pipeline:
//!   - FrameProcessor: Process frames in the pipeline
//! ```

mod fsm;
mod llm;
mod pipeline;
mod retriever;
mod speech;
mod text_processing;
mod tool;

pub use speech::{SpeechToText, TextToSpeech};
// P1 FIX: Export VoiceActivityDetector trait and types
pub use llm::LanguageModel;
pub use pipeline::{ControlFrame, Frame, FrameProcessor, MetricsEvent, ProcessorContext};
pub use retriever::{
    ConversationContext, ConversationTurn, Document, FilterOp, MetadataFilter, RetrieveOptions,
    Retriever,
};
pub use speech::{AudioProcessor, VADConfig, VADEvent, VADState, VoiceActivityDetector};
pub use text_processing::{
    ComplianceChecker, GrammarCorrector, PIIRedactor, TextProcessor, TextProcessorResult,
    Translator,
};
// P0 FIX: Export ConversationFSM trait and types
pub use fsm::{
    ConversationEvent, ConversationFSM, ConversationOutcome, FSMAction, FSMCheckpoint, FSMError,
    FSMMetrics, ObjectionType, TransitionRecord,
};
// P3 FIX: Export Tool trait and types
pub use tool::{
    validate_property, ContentBlock, ErrorCode, InputSchema, PropertySchema, Tool, ToolError,
    ToolInput, ToolOutput, ToolSchema,
};
