//! LLM Integration with Speculative Execution
//!
//! Features:
//! - Multiple backend support (Ollama, Claude, OpenAI)
//! - Native tool calling (Claude tool_use, text-based for Ollama)
//! - Speculative execution (SLM-first, race parallel, hybrid streaming)
//! - Streaming token generation
//! - Context management

pub mod backend;
pub mod prompt;
pub mod speculative;
pub mod streaming;
// P0 FIX: Adapter bridging LlmBackend to core::LanguageModel
pub mod adapter;
// P0-3a: Claude backend with native tool_use support
pub mod claude;
// P0-3c: LLM factory with provider abstraction
pub mod factory;

pub use backend::{
    FinishReason, GenerationResult, LlmBackend, LlmConfig, OllamaBackend, OpenAIBackend,
    OpenAIConfig,
};
// P0 FIX: Export adapter for clean dependency injection
pub use adapter::LanguageModelAdapter;
// P0-3a: Export Claude backend
pub use claude::{ClaudeBackend, ClaudeConfig, ClaudeModel, ClaudeResponse, ClaudeStopReason};
// P0-3c: Export factory
pub use factory::{ClaudeLanguageModel, LlmFactory, LlmProvider, LlmProviderConfig};
pub use prompt::{
    gold_loan_tools, parse_tool_call, Message, ParsedToolCall, PersonaConfig, PromptBuilder, Role,
    ToolBuilder, ToolDefinition,
};
pub use speculative::{
    SpeculativeConfig, SpeculativeExecutor, SpeculativeMode, SpeculativeResult,
};
pub use streaming::{GenerationEvent, StreamingGenerator, TokenStream};

use thiserror::Error;

/// LLM errors
#[derive(Error, Debug)]
pub enum LlmError {
    #[error("Generation error: {0}")]
    Generation(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Timeout")]
    Timeout,

    #[error("Context too long: {0} > {1}")]
    ContextTooLong(usize, usize),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        LlmError::Network(err.to_string())
    }
}

impl From<LlmError> for voice_agent_core::Error {
    fn from(err: LlmError) -> Self {
        voice_agent_core::Error::Llm(err.to_string())
    }
}
