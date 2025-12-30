//! Error types for the voice agent

use thiserror::Error;

/// Result type alias using our Error
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for the voice agent
#[derive(Error, Debug)]
pub enum Error {
    // Audio errors
    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioError),

    // Pipeline errors
    #[error("Pipeline error: {0}")]
    Pipeline(#[from] PipelineError),

    // Model errors
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    // Tool errors
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    // Agent errors
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    // LLM errors
    #[error("LLM error: {0}")]
    Llm(String),

    // RAG errors
    #[error("RAG error: {0}")]
    Rag(String),

    // P0 FIX: Text processing errors
    #[error("Text processing error: {0}")]
    TextProcessing(String),

    // Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    // IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    // Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    // Generic errors
    #[error("{0}")]
    Other(String),
}

/// Audio-specific errors
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Invalid audio format: {0}")]
    InvalidFormat(String),

    #[error("Unsupported sample rate: {0}")]
    UnsupportedSampleRate(u32),

    #[error("Buffer overflow")]
    BufferOverflow,

    #[error("Codec error: {0}")]
    Codec(String),

    #[error("Resampling error: {0}")]
    Resampling(String),
}

/// Pipeline processing errors
#[derive(Error, Debug)]
pub enum PipelineError {
    #[error("VAD error: {0}")]
    Vad(String),

    #[error("STT error: {0}")]
    Stt(String),

    #[error("TTS error: {0}")]
    Tts(String),

    #[error("Turn detection error: {0}")]
    TurnDetection(String),

    #[error("Channel closed")]
    ChannelClosed,

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("Pipeline not initialized")]
    NotInitialized,

    /// P2 FIX: Added Audio variant for proper error type preservation
    #[error("Audio processing error: {0}")]
    Audio(String),

    /// P2 FIX: Added Io variant for proper error type preservation
    #[error("IO error: {0}")]
    Io(String),

    /// P2 FIX: Added Model variant for model loading/inference errors
    #[error("Model error: {0}")]
    Model(String),
}

/// Model/inference errors
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model not found: {0}")]
    NotFound(String),

    #[error("Model load error: {0}")]
    LoadError(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("ONNX runtime error: {0}")]
    OnnxRuntime(String),
}

/// Tool execution errors
#[derive(Error, Debug)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Execution failed: {0}")]
    ExecutionFailed(String),

    #[error("Timeout")]
    Timeout,

    #[error("Rate limited")]
    RateLimited,

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// P3 FIX: Convert MCP ToolError (from traits module) to error::ToolError
impl From<crate::traits::ToolError> for ToolError {
    fn from(err: crate::traits::ToolError) -> Self {
        match err.code {
            crate::traits::ErrorCode::MethodNotFound => ToolError::NotFound(err.message),
            crate::traits::ErrorCode::InvalidParams => ToolError::InvalidInput(err.message),
            _ => ToolError::ExecutionFailed(err.message),
        }
    }
}

/// P3 FIX: Convert MCP ToolError to top-level Error
impl From<crate::traits::ToolError> for Error {
    fn from(err: crate::traits::ToolError) -> Self {
        Error::Tool(err.into())
    }
}

/// Agent errors
#[derive(Error, Debug)]
pub enum AgentError {
    #[error("Invalid stage transition: {from:?} -> {to:?}")]
    InvalidStageTransition {
        from: crate::ConversationStage,
        to: crate::ConversationStage,
    },

    #[error("LLM generation error: {0}")]
    LlmGeneration(String),

    #[error("Context overflow: {0} tokens exceeds {1}")]
    ContextOverflow(usize, usize),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("No response generated")]
    NoResponse,
}

impl Error {
    /// Create a generic error from a string
    pub fn other<S: Into<String>>(msg: S) -> Self {
        Error::Other(msg.into())
    }

    /// Create a config error
    pub fn config<S: Into<String>>(msg: S) -> Self {
        Error::Config(msg.into())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Other(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Other(s.to_string())
    }
}

/// Error code for MCP protocol compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    InvalidInput,
    ExecutionFailed,
    Timeout,
    RateLimited,
    Unauthorized,
    NotFound,
    InternalError,
}

impl From<&ToolError> for ErrorCode {
    fn from(err: &ToolError) -> Self {
        match err {
            ToolError::InvalidInput(_) => ErrorCode::InvalidInput,
            ToolError::NotFound(_) => ErrorCode::NotFound,
            ToolError::ExecutionFailed(_) => ErrorCode::ExecutionFailed,
            ToolError::Timeout => ErrorCode::Timeout,
            ToolError::RateLimited => ErrorCode::RateLimited,
            ToolError::Unauthorized => ErrorCode::Unauthorized,
            ToolError::Internal(_) => ErrorCode::InternalError,
        }
    }
}
