//! Text Processing Pipeline for Voice Agent
//!
//! This crate provides text processing capabilities:
//! - **Grammar Correction**: Fix STT errors while preserving domain vocabulary
//! - **Translation**: Translate between Indian languages (Translate-Think-Translate)
//! - **PII Detection**: Detect and redact sensitive Indian data (Aadhaar, PAN, etc.)
//! - **Compliance Checking**: Ensure banking regulatory compliance
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_text_processing::{TextProcessingPipeline, TextProcessingConfig};
//!
//! let config = TextProcessingConfig::default();
//! let pipeline = TextProcessingPipeline::new(config)?;
//!
//! // Process text through the pipeline
//! let result = pipeline.process("mujhe gol lone chahiye").await?;
//! println!("Processed: {}", result.text);
//! ```

pub mod grammar;
pub mod translation;
pub mod pii;
pub mod compliance;

mod pipeline;
mod error;

pub use pipeline::{TextProcessingPipeline, TextProcessingConfig, ProcessedText};
pub use error::{TextProcessingError, Result};

// Re-export key types
pub use grammar::{GrammarConfig, GrammarProvider, LLMGrammarCorrector, NoopCorrector};
pub use translation::{TranslationConfig, TranslationProvider, ScriptDetector};
pub use pii::{PIIConfig, PIIProvider, HybridPIIDetector, IndianPIIPatterns};
pub use compliance::{ComplianceConfig, ComplianceProvider, RuleBasedComplianceChecker};
