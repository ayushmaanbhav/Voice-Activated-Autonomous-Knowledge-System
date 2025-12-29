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
pub mod error;
pub mod transcript;
pub mod conversation;
pub mod customer;

// New modules (Phase 1)
pub mod language;
pub mod voice_config;
pub mod pii;
pub mod compliance;
pub mod domain_context;
pub mod llm_types;
pub mod traits;

// Phase 5: Personalization
pub mod personalization;

// Re-exports from existing modules
pub use audio::{AudioFrame, AudioEncoding, Channels, SampleRate};
pub use error::{Error, Result};
pub use transcript::{TranscriptResult, WordTimestamp};
pub use conversation::{Turn, TurnRole, ConversationStage};
pub use customer::{CustomerProfile, CustomerSegment};

// Re-exports from new modules
pub use language::{Language, Script};
pub use voice_config::{VoiceConfig, VoiceInfo, VoiceGender};
pub use pii::{PIIType, PIIEntity, PIISeverity, RedactionStrategy, DetectionMethod};
pub use compliance::{
    ComplianceResult, ComplianceViolation, Severity,
    ViolationCategory, RequiredAddition, AdditionType, AdditionPosition,
    SuggestedRewrite,
};
pub use domain_context::{DomainContext, Abbreviation};
pub use llm_types::{
    GenerateRequest, GenerateResponse, Message, Role,
    StreamChunk, FinishReason, TokenUsage,
    ToolDefinition, ToolCall,
};

// Trait re-exports
pub use traits::{
    // Speech
    SpeechToText, TextToSpeech,
    // P1 FIX: VoiceActivityDetector trait
    VoiceActivityDetector, VADConfig, VADEvent, VADState, AudioProcessor,
    // LLM
    LanguageModel,
    // Retrieval
    Retriever, RetrieveOptions, Document, ConversationContext, ConversationTurn,
    MetadataFilter, FilterOp,
    // Text processing
    GrammarCorrector, Translator, PIIRedactor, ComplianceChecker,
    // Pipeline
    FrameProcessor, Frame, ProcessorContext, ControlFrame, MetricsEvent,
};

// Personalization re-exports
pub use personalization::{
    // Engine
    PersonalizationEngine, PersonalizationContext,
    // Persona
    Persona, PersonaTemplates, Tone, LanguageComplexity, ResponseUrgency,
    // Adaptation
    SegmentAdapter, Feature, Objection, ObjectionResponse,
    // Signals
    SignalDetector, BehaviorSignal, SignalDetection, TrendAnalysis,
};
