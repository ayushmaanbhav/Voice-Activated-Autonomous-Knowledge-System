//! LLM Integration with Speculative Execution
//!
//! Features:
//! - Multiple backend support (Ollama, Claude, OpenAI)
//! - Speculative execution (SLM-first, race parallel, hybrid streaming)
//! - Streaming token generation
//! - Context management

pub mod backend;
pub mod speculative;
pub mod streaming;
pub mod prompt;

pub use backend::{LlmBackend, OllamaBackend, LlmConfig, OpenAIBackend, OpenAIConfig};
pub use speculative::{SpeculativeExecutor, SpeculativeMode, SpeculativeResult};
pub use streaming::{StreamingGenerator, TokenStream, GenerationEvent};
pub use prompt::{PromptBuilder, Message, Role, PersonaConfig, ToolDefinition, ToolParameter, ParsedToolCall, parse_tool_call};

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
