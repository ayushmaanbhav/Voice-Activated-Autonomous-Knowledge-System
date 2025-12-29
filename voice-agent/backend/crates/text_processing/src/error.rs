//! Error types for text processing

use thiserror::Error;

/// Text processing errors
#[derive(Debug, Error)]
pub enum TextProcessingError {
    /// Grammar correction failed
    #[error("Grammar correction failed: {0}")]
    GrammarError(String),

    /// Translation failed
    #[error("Translation failed: {0}")]
    TranslationError(String),

    /// Unsupported language pair
    #[error("Unsupported language pair: {from} -> {to}")]
    UnsupportedLanguagePair { from: String, to: String },

    /// PII detection failed
    #[error("PII detection failed: {0}")]
    PIIError(String),

    /// Compliance check failed
    #[error("Compliance check failed: {0}")]
    ComplianceError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Core error
    #[error("Core error: {0}")]
    CoreError(#[from] voice_agent_core::Error),
}

/// Result type for text processing
pub type Result<T> = std::result::Result<T, TextProcessingError>;
