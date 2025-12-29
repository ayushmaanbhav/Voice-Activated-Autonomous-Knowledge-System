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
//! Retrieval:
//!   - Retriever: Document retrieval for RAG
//!
//! Text Processing:
//!   - GrammarCorrector: Fix grammar while preserving domain terms
//!   - Translator: Translate between Indian languages
//!   - PIIRedactor: Detect and redact PII
//!   - ComplianceChecker: Check for regulatory compliance
//!
//! Pipeline:
//!   - FrameProcessor: Process frames in the pipeline
//! ```

mod speech;
mod llm;
mod retriever;
mod text_processing;
mod pipeline;

pub use speech::{SpeechToText, TextToSpeech};
// P1 FIX: Export VoiceActivityDetector trait and types
pub use speech::{VoiceActivityDetector, VADConfig, VADEvent, VADState, AudioProcessor};
pub use llm::LanguageModel;
pub use retriever::{
    Retriever, RetrieveOptions, Document, ConversationContext,
    ConversationTurn, MetadataFilter, FilterOp,
};
pub use text_processing::{GrammarCorrector, Translator, PIIRedactor, ComplianceChecker};
pub use pipeline::{FrameProcessor, Frame, ProcessorContext, ControlFrame, MetricsEvent};
