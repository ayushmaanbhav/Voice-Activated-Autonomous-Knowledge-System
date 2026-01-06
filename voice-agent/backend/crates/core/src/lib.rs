//! Core traits and types for the voice agent
//!
//! This crate provides foundational types used across all other crates:
//! - Core traits for pluggable backends (STT, TTS, LLM, etc.)
//! - Audio frame types and processing
//! - Language definitions (22 Indian languages)
//! - Text processing types (PII, compliance)
//! - Error types
//! - Conversation types

// Existing modules
pub mod audio;
pub mod conversation;
pub mod customer;
pub mod error;
pub mod transcript;

// New modules (Phase 1)
pub mod compliance;
pub mod domain;
pub mod domain_context;
pub mod language;
pub mod llm_types;
pub mod pii;
pub mod traits;
pub mod voice_config;

// Phase 5: Personalization
pub mod personalization;

// Re-exports from existing modules
pub use audio::{AudioEncoding, AudioFrame, Channels, SampleRate};
pub use conversation::{ConversationStage, Turn, TurnRole};
pub use customer::{CustomerProfile, CustomerSegment, SegmentDetector};
pub use error::{Error, Result};
pub use transcript::{TranscriptResult, WordTimestamp};

// Re-exports from new modules
pub use compliance::{
    AdditionPosition, AdditionType, ComplianceResult, ComplianceViolation, RequiredAddition,
    Severity, SuggestedRewrite, ViolationCategory,
};
pub use domain_context::{Abbreviation, DomainContext};
pub use language::{Language, Script};
pub use llm_types::{
    FinishReason, GenerateRequest, GenerateResponse, Message, Role, StreamChunk, TokenUsage,
    ToolCall, ToolDefinition,
};
pub use pii::{DetectionMethod, PIIEntity, PIISeverity, PIIType, RedactionStrategy};
pub use voice_config::{VoiceConfig, VoiceGender, VoiceInfo};

// Trait re-exports
pub use traits::{
    AudioProcessor,
    ComplianceChecker,
    ControlFrame,
    ConversationContext,
    ConversationEvent,
    // P0 FIX: ConversationFSM trait and types
    ConversationFSM,
    ConversationOutcome,
    ConversationTurn,
    Document,
    FSMAction,
    FSMCheckpoint,
    FSMError,
    FSMMetrics,
    FilterOp,
    Frame,
    // Pipeline
    FrameProcessor,
    // Text processing
    GrammarCorrector,
    // LLM
    LanguageModel,
    MetadataFilter,
    MetricsEvent,
    ObjectionType,
    PIIRedactor,
    ProcessorContext,
    RetrieveOptions,
    // Retrieval
    Retriever,
    // Speech
    SpeechToText,
    // P0 FIX: TextProcessor trait for pipeline integration
    TextProcessor,
    TextProcessorResult,
    TextToSpeech,
    TransitionRecord,
    Translator,
    VADConfig,
    VADEvent,
    VADState,
    // P1 FIX: VoiceActivityDetector trait
    VoiceActivityDetector,
};

// Personalization re-exports
pub use personalization::{
    BehaviorSignal,
    Feature,
    LanguageComplexity,
    Objection,
    ObjectionResponse,
    // Persona
    Persona,
    PersonaTemplates,
    PersonalizationContext,
    // Engine
    PersonalizationEngine,
    ResponseUrgency,
    // Adaptation
    SegmentAdapter,
    SignalDetection,
    // Signals
    SignalDetector,
    Tone,
    TrendAnalysis,
};
