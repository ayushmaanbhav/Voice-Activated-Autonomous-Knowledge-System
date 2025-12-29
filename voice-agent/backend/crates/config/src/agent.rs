//! Agent configuration

use serde::{Deserialize, Serialize};

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

    /// RAG configuration
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
    "hi".to_string()
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
    "qwen2.5:7b-instruct-q4_K_M".to_string()
}
fn default_slm_model() -> String {
    "qwen2.5:1.5b-instruct-q4_K_M".to_string()
}
fn default_llm_endpoint() -> String {
    "http://localhost:11434".to_string()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProvider {
    /// Local Ollama server
    Ollama,
    /// Kalosm (native Rust)
    Kalosm,
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

/// RAG configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Enable RAG
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Qdrant endpoint
    #[serde(default = "default_qdrant_endpoint")]
    pub qdrant_endpoint: String,

    /// Collection name
    #[serde(default = "default_collection")]
    pub collection: String,

    /// Number of results to retrieve
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum relevance score
    #[serde(default = "default_min_score")]
    pub min_score: f32,

    /// Enable hybrid search (dense + BM25)
    #[serde(default = "default_true")]
    pub hybrid_search: bool,

    /// Enable reranking
    #[serde(default = "default_true")]
    pub reranking_enabled: bool,

    /// Early exit configuration
    #[serde(default)]
    pub early_exit: EarlyExitConfig,

    /// Enable prefetch on partial transcript
    #[serde(default = "default_true")]
    pub prefetch_enabled: bool,

    /// Minimum confidence for prefetch
    #[serde(default = "default_prefetch_confidence")]
    pub prefetch_min_confidence: f32,
}

fn default_qdrant_endpoint() -> String {
    "http://localhost:6333".to_string()
}
fn default_collection() -> String {
    "gold_loan_knowledge".to_string()
}
fn default_top_k() -> usize {
    5
}
fn default_min_score() -> f32 {
    0.5
}
fn default_prefetch_confidence() -> f32 {
    0.7
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            qdrant_endpoint: default_qdrant_endpoint(),
            collection: default_collection(),
            top_k: default_top_k(),
            min_score: default_min_score(),
            hybrid_search: true,
            reranking_enabled: true,
            early_exit: EarlyExitConfig::default(),
            prefetch_enabled: true,
            prefetch_min_confidence: default_prefetch_confidence(),
        }
    }
}

/// Early exit cross-encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyExitConfig {
    /// Exit strategy
    #[serde(default = "default_exit_strategy")]
    pub strategy: ExitStrategy,

    /// Confidence threshold for early exit
    #[serde(default = "default_confidence_threshold")]
    pub confidence_threshold: f32,

    /// Patience (consecutive agreeing layers)
    #[serde(default = "default_patience")]
    pub patience: usize,

    /// Minimum layer before allowing exit
    #[serde(default = "default_min_layer")]
    pub min_layer: usize,
}

fn default_exit_strategy() -> ExitStrategy {
    ExitStrategy::Hybrid
}
fn default_confidence_threshold() -> f32 {
    0.9
}
fn default_patience() -> usize {
    2
}
fn default_min_layer() -> usize {
    3
}

impl Default for EarlyExitConfig {
    fn default() -> Self {
        Self {
            strategy: default_exit_strategy(),
            confidence_threshold: default_confidence_threshold(),
            patience: default_patience(),
            min_layer: default_min_layer(),
        }
    }
}

/// Early exit strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitStrategy {
    /// Exit when confidence exceeds threshold
    Confidence,
    /// Exit when k consecutive layers agree
    Patience,
    /// Combination of confidence and patience
    Hybrid,
    /// Exit based on layer output similarity
    Similarity,
}

/// Memory configuration
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

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_size: default_working_memory(),
            summarization_threshold: default_summarization_threshold(),
            max_episodic_summaries: default_max_summaries(),
            semantic_memory_enabled: true,
        }
    }
}
