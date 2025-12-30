//! Agent configuration
//!
//! P2-3 FIX: Consolidated duplicate config definitions.
//! - RagConfig now uses the detailed version from settings.rs
//! - MemoryConfig now includes P1 token limits with serde support

use serde::{Deserialize, Serialize};

use crate::constants::endpoints;
use crate::settings::RagConfig;

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name for introductions
    #[serde(default = "default_agent_name")]
    pub name: String,

    /// Default language
    #[serde(default = "default_agent_language")]
    pub language: String,

    /// Maximum conversation duration (seconds)
    #[serde(default = "default_max_duration")]
    pub max_duration_seconds: u32,

    /// Enable tools
    #[serde(default = "default_true")]
    pub tools_enabled: bool,

    /// Persona configuration
    #[serde(default)]
    pub persona: PersonaConfig,

    /// LLM configuration
    #[serde(default)]
    pub llm: LlmConfig,

    /// RAG configuration (uses detailed settings::RagConfig)
    #[serde(default)]
    pub rag: RagConfig,

    /// Memory configuration
    #[serde(default)]
    pub memory: MemoryConfig,
}

fn default_agent_name() -> String {
    "Priya".to_string()
}
fn default_agent_language() -> String {
    "en".to_string()
}
fn default_max_duration() -> u32 {
    600 // 10 minutes
}
fn default_true() -> bool {
    true
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: default_agent_name(),
            language: default_agent_language(),
            max_duration_seconds: default_max_duration(),
            tools_enabled: true,
            persona: PersonaConfig::default(),
            llm: LlmConfig::default(),
            rag: RagConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
}

/// Persona traits configuration
///
/// P0 FIX: Consolidated from 3 duplicate definitions (config, llm, agent).
/// This is now the single source of truth for persona configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    /// Agent persona name (e.g., "Priya")
    #[serde(default = "default_persona_name")]
    pub name: String,

    /// Warmth level (0.0 - 1.0)
    #[serde(default = "default_warmth")]
    pub warmth: f32,

    /// Formality level (0.0 - 1.0)
    #[serde(default = "default_formality")]
    pub formality: f32,

    /// Urgency level (0.0 - 1.0)
    #[serde(default = "default_urgency")]
    pub urgency: f32,

    /// Empathy level (0.0 - 1.0)
    #[serde(default = "default_empathy")]
    pub empathy: f32,
}

fn default_persona_name() -> String {
    "Priya".to_string()
}

fn default_warmth() -> f32 {
    0.8
}
fn default_formality() -> f32 {
    0.6
}
fn default_urgency() -> f32 {
    0.4
}
fn default_empathy() -> f32 {
    0.9
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            name: default_persona_name(),
            warmth: default_warmth(),
            formality: default_formality(),
            urgency: default_urgency(),
            empathy: default_empathy(),
        }
    }
}

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// LLM provider
    #[serde(default = "default_llm_provider")]
    pub provider: LlmProvider,

    /// Model name/ID
    #[serde(default = "default_llm_model")]
    pub model: String,

    /// SLM model for speculative execution
    #[serde(default = "default_slm_model")]
    pub slm_model: String,

    /// API endpoint (for Ollama)
    #[serde(default = "default_llm_endpoint")]
    pub endpoint: String,

    /// API key (for cloud providers)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Enable speculative execution
    #[serde(default = "default_true")]
    pub speculative_enabled: bool,

    /// Speculative mode
    #[serde(default = "default_speculative_mode")]
    pub speculative_mode: SpeculativeMode,
}

fn default_llm_provider() -> LlmProvider {
    LlmProvider::Ollama
}
fn default_llm_model() -> String {
    "qwen3:4b-instruct-2507-q4_K_M".to_string()
}
fn default_slm_model() -> String {
    "qwen2.5:1.5b-instruct-q4_K_M".to_string()
}
fn default_llm_endpoint() -> String {
    endpoints::OLLAMA_DEFAULT.to_string()
}
fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_speculative_mode() -> SpeculativeMode {
    SpeculativeMode::SlmFirst
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: default_llm_provider(),
            model: default_llm_model(),
            slm_model: default_slm_model(),
            endpoint: default_llm_endpoint(),
            api_key: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            speculative_enabled: true,
            speculative_mode: default_speculative_mode(),
        }
    }
}

/// LLM provider
///
/// P3-2 FIX: Removed unused Kalosm variant (no implementation exists)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    /// Local Ollama server
    Ollama,
    /// Anthropic Claude
    Claude,
    /// OpenAI
    OpenAI,
}

/// Speculative execution mode
///
/// P0 FIX: Removed DraftVerify mode which was mislabeled as "EAGLE-style" but
/// actually ran SLM then LLM sequentially, doubling latency instead of reducing it.
/// Use SlmFirst (recommended) or RaceParallel instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeculativeMode {
    /// SLM first, upgrade if complex (recommended for most use cases)
    SlmFirst,
    /// Race SLM and LLM in parallel, use first acceptable response
    RaceParallel,
    /// Hybrid streaming (start SLM, switch to LLM mid-stream if quality drops)
    HybridStreaming,
}

// P2-3 FIX: Removed duplicate RagConfig - now using settings::RagConfig
// See settings.rs for the canonical RagConfig with all P5 fields

/// Memory configuration
///
/// P2-3 FIX: Consolidated from config::agent and agent::memory.
/// Now includes both serde support AND P1 token limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Working memory size (recent turns)
    #[serde(default = "default_working_memory")]
    pub working_memory_size: usize,

    /// Summarization threshold (turns before summarizing)
    #[serde(default = "default_summarization_threshold")]
    pub summarization_threshold: usize,

    /// Maximum episodic summaries to keep
    #[serde(default = "default_max_summaries")]
    pub max_episodic_summaries: usize,

    /// Enable semantic memory (key facts)
    #[serde(default = "default_true")]
    pub semantic_memory_enabled: bool,

    /// P1 FIX: Maximum total tokens before aggressive truncation
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,

    /// P1 FIX: High watermark - trigger summarization when exceeded
    #[serde(default = "default_high_watermark_tokens")]
    pub high_watermark_tokens: usize,

    /// P1 FIX: Low watermark - target after truncation
    #[serde(default = "default_low_watermark_tokens")]
    pub low_watermark_tokens: usize,
}

fn default_working_memory() -> usize {
    8
}
fn default_summarization_threshold() -> usize {
    6
}
fn default_max_summaries() -> usize {
    10
}
fn default_max_context_tokens() -> usize {
    4096 // Hard limit
}
fn default_high_watermark_tokens() -> usize {
    3072 // 75% - trigger summarization
}
fn default_low_watermark_tokens() -> usize {
    2048 // 50% - target after cleanup
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_size: default_working_memory(),
            summarization_threshold: default_summarization_threshold(),
            max_episodic_summaries: default_max_summaries(),
            semantic_memory_enabled: true,
            max_context_tokens: default_max_context_tokens(),
            high_watermark_tokens: default_high_watermark_tokens(),
            low_watermark_tokens: default_low_watermark_tokens(),
        }
    }
}
