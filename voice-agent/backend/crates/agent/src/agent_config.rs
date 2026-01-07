//! Agent Configuration Types
//!
//! Configuration structs for the DomainAgent (formerly GoldLoanAgent).

use voice_agent_config::PersonaConfig;
use voice_agent_llm::{LlmProviderConfig, SpeculativeConfig, SpeculativeMode};
use voice_agent_rag::AgenticRagConfig;

use crate::conversation::ConversationConfig;
use crate::dst::DstConfig;
use crate::stage::RagTimingStrategy;

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Default language
    pub language: String,
    /// Conversation config
    pub conversation: ConversationConfig,
    /// Persona configuration (P0 FIX: now uses consolidated PersonaConfig)
    pub persona: PersonaConfig,
    /// Enable RAG
    pub rag_enabled: bool,
    /// Enable tools
    pub tools_enabled: bool,
    /// P1 FIX: Configurable tool defaults (no more hardcoded values)
    pub tool_defaults: ToolDefaults,
    /// P2 FIX: Context window size in tokens (for LLM prompt truncation)
    pub context_window_tokens: usize,
    /// P4 FIX: RAG timing strategy for prefetch behavior
    pub rag_timing_strategy: RagTimingStrategy,
    /// P1-1 FIX: LLM provider configuration (supports Claude, Ollama, OpenAI, Azure)
    pub llm_provider: LlmProviderConfig,
    /// P1-2 FIX: Speculative decoding configuration (SLM + LLM)
    pub speculative: SpeculativeDecodingConfig,
    /// Phase 5: Dialogue State Tracking configuration
    pub dst_config: DstConfig,
    /// Phase 11: Agentic RAG configuration for multi-step retrieval
    pub agentic_rag: AgenticRagConfig,
    /// Small model optimizations (auto-detected or manual)
    pub small_model: SmallModelConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        // Detect if default model is small
        let default_model = "qwen2.5:1.5b-instruct-q4_K_M";
        let is_small = is_small_model(default_model);

        // Create small model config based on detection
        let small_model = if is_small {
            SmallModelConfig::enabled()
        } else {
            SmallModelConfig::disabled()
        };

        // Adjust context window based on model size
        let context_tokens = if is_small {
            small_model.context_window_tokens
        } else {
            4096
        };

        // Configure agentic RAG based on model size
        // Small models use single-shot retrieval with rule-based expansion only
        let agentic_rag = if is_small {
            AgenticRagConfig::for_small_model()
        } else {
            AgenticRagConfig::default()
        };

        Self {
            language: "en".to_string(),
            conversation: ConversationConfig::default(),
            persona: PersonaConfig::default(),
            rag_enabled: true,
            tools_enabled: true,
            tool_defaults: ToolDefaults::default(),
            // Context window adjusted for small models (2500 vs 4096)
            // Research: Qwen2.5 Technical Report (arXiv:2412.15115)
            context_window_tokens: context_tokens,
            // P4 FIX: Default to conservative prefetch strategy
            rag_timing_strategy: RagTimingStrategy::default(),
            // P1-1 FIX: Default to Ollama for local development
            // P6 FIX: Use qwen2.5:1.5b-instruct-q4_K_M for better Hindi/English support
            llm_provider: LlmProviderConfig::ollama(default_model),
            // P1-2 FIX: Speculative decoding disabled by default
            speculative: SpeculativeDecodingConfig::default(),
            // Phase 5: DST configuration
            dst_config: DstConfig::default(),
            // Phase 11: Agentic RAG - single-shot for small models, iterative for large
            agentic_rag,
            // Small model config (auto-detected)
            small_model,
        }
    }
}

impl AgentConfig {
    /// Get agent name from persona
    pub fn name(&self) -> &str {
        &self.persona.name
    }

    /// Check if small model optimizations are enabled
    pub fn is_small_model(&self) -> bool {
        self.small_model.enabled
    }

    /// Create config with specific model, auto-detecting small model settings
    pub fn with_model(model_name: &str) -> Self {
        let is_small = is_small_model(model_name);
        let small_model = if is_small {
            SmallModelConfig::enabled()
        } else {
            SmallModelConfig::disabled()
        };
        let context_tokens = if is_small {
            small_model.context_window_tokens
        } else {
            4096
        };
        // Configure agentic RAG based on model size
        let agentic_rag = if is_small {
            AgenticRagConfig::for_small_model()
        } else {
            AgenticRagConfig::default()
        };

        Self {
            llm_provider: LlmProviderConfig::ollama(model_name),
            context_window_tokens: context_tokens,
            small_model,
            agentic_rag,
            ..Default::default()
        }
    }

    /// Apply small model optimizations to an existing config
    pub fn optimize_for_small_model(mut self) -> Self {
        self.small_model = SmallModelConfig::enabled();
        self.context_window_tokens = self.small_model.context_window_tokens;
        // Switch to single-shot retrieval with rule-based expansion
        self.agentic_rag = AgenticRagConfig::for_small_model();
        self
    }
}

/// P1 FIX: Configurable default values for tool calls
#[derive(Debug, Clone)]
pub struct ToolDefaults {
    /// Default city for branch searches
    pub default_city: String,
    /// Default gold purity for eligibility checks
    pub default_gold_purity: String,
    /// Default competitor interest rate (%)
    pub default_competitor_rate: f64,
    /// Default loan amount for savings calculations
    pub default_loan_amount: u64,
    /// Default remaining tenure (months)
    pub default_tenure_months: u32,
}

impl Default for ToolDefaults {
    fn default() -> Self {
        Self {
            default_city: "Mumbai".to_string(),
            default_gold_purity: "22K".to_string(),
            default_competitor_rate: 18.0,
            default_loan_amount: 100_000,
            default_tenure_months: 12,
        }
    }
}

/// P1-2 FIX: Speculative decoding configuration
///
/// Configures the small (SLM) and large (LLM) models for speculative execution.
/// The SLM drafts responses quickly, and the LLM verifies/improves them.
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Enable speculative decoding
    pub enabled: bool,
    /// Speculative execution mode
    pub mode: SpeculativeMode,
    /// Small model (SLM) configuration - fast, for drafting
    pub slm: LlmProviderConfig,
    /// Large model (LLM) configuration - accurate, for verification
    pub llm: LlmProviderConfig,
    /// Speculative execution parameters
    pub params: SpeculativeConfig,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default, requires explicit opt-in
            mode: SpeculativeMode::SlmFirst,
            // Small model for fast drafting (Ollama)
            slm: LlmProviderConfig::ollama("llama3.2:1b"),
            // Large model for verification (Ollama)
            llm: LlmProviderConfig::ollama("llama3.2:3b"),
            params: SpeculativeConfig::default(),
        }
    }
}

impl SpeculativeDecodingConfig {
    /// Create Ollama-based speculative config
    pub fn ollama(slm_model: impl Into<String>, llm_model: impl Into<String>) -> Self {
        Self {
            enabled: true,
            mode: SpeculativeMode::SlmFirst,
            slm: LlmProviderConfig::ollama(slm_model),
            llm: LlmProviderConfig::ollama(llm_model),
            params: SpeculativeConfig::default(),
        }
    }

    /// Set speculative mode
    pub fn with_mode(mut self, mode: SpeculativeMode) -> Self {
        self.mode = mode;
        self.params.mode = mode;
        self
    }

    /// Enable draft-verify mode (good balance of speed and quality)
    pub fn draft_verify(mut self) -> Self {
        self.mode = SpeculativeMode::DraftVerify;
        self.params.mode = SpeculativeMode::DraftVerify;
        self
    }

    /// Enable race-parallel mode (lowest latency, highest cost)
    pub fn race_parallel(mut self) -> Self {
        self.mode = SpeculativeMode::RaceParallel;
        self.params.mode = SpeculativeMode::RaceParallel;
        self
    }

    /// Enable hybrid streaming mode (adaptive quality)
    pub fn hybrid_streaming(mut self) -> Self {
        self.mode = SpeculativeMode::HybridStreaming;
        self.params.mode = SpeculativeMode::HybridStreaming;
        self
    }
}

/// Small model optimization configuration
///
/// Configures optimizations for small language models (< 3B parameters)
/// like Qwen2.5:1.5B, Llama3.2:1B, etc.
///
/// Research: Qwen2.5 Technical Report (arXiv:2412.15115) shows that small
/// models benefit from reduced context windows and extractive compression.
#[derive(Debug, Clone)]
pub struct SmallModelConfig {
    /// Enable small model optimizations (auto-detected from model name)
    pub enabled: bool,
    /// Context window size for small models (default: 2500)
    pub context_window_tokens: usize,
    /// High watermark for compression trigger (default: 2000)
    pub high_watermark_tokens: usize,
    /// Low watermark target after compression (default: 1500)
    pub low_watermark_tokens: usize,
    /// Use extractive compression instead of LLM summarization
    pub use_extractive_compression: bool,
    /// Disable LLM-based query rewriting in RAG
    pub disable_llm_query_rewriting: bool,
}

impl Default for SmallModelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            context_window_tokens: 2500,
            high_watermark_tokens: 2000,
            low_watermark_tokens: 1500,
            use_extractive_compression: true,
            disable_llm_query_rewriting: true,
        }
    }
}

impl SmallModelConfig {
    /// Create config for small models (enabled)
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Create config for large models (disabled)
    pub fn disabled() -> Self {
        Self::default()
    }
}

/// Detect if a model name indicates a small model (< 3B parameters)
///
/// Recognizes common naming patterns:
/// - Size suffixes: "1b", "1.5b", "2b", "3b"
/// - Ollama tags: ":1b", ":1.5b", ":3b"
/// - Full names: "qwen2.5:1.5b", "llama3.2:1b", "phi-3-mini"
pub fn is_small_model(model_name: &str) -> bool {
    let model_lower = model_name.to_lowercase();

    // First check for large model patterns (must be checked first to avoid false positives)
    // e.g., "72b" contains "2b", so we must check for large patterns first
    let large_patterns = [
        "7b", ":7b", "-7b", "8b", ":8b", "-8b", "13b", ":13b", "-13b", "14b", ":14b", "-14b",
        "32b", ":32b", "-32b", "70b", ":70b", "-70b", "72b", ":72b", "-72b", "large", "xl",
    ];
    for pattern in &large_patterns {
        if model_lower.contains(pattern) {
            return false;
        }
    }

    // Now check for small model patterns
    let small_patterns = [
        "0.5b", ":0.5b", "-0.5b", "1b", ":1b", "-1b", "1.5b", ":1.5b", "-1.5b", "2b", ":2b",
        "-2b", "3b", ":3b", "-3b", "mini", "tiny", "small", "phi-2", "phi-3-mini",
    ];

    for pattern in &small_patterns {
        if model_lower.contains(pattern) {
            return true;
        }
    }

    // Default: not a small model
    false
}

use crate::conversation::ConversationEvent;

/// Agent events
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Response ready
    Response(String),
    /// Thinking/processing
    Thinking,
    /// Tool being called
    ToolCall { name: String },
    /// Tool result
    ToolResult { name: String, success: bool },
    /// Conversation event
    Conversation(ConversationEvent),
    /// Error
    Error(String),
    /// Lead score updated (Phase 10)
    LeadScoreUpdated {
        score: u32,
        qualification: String,
        classification: String,
        conversion_probability: f32,
    },
    /// Escalation triggered (Phase 10)
    EscalationTriggered {
        trigger: String,
        recommendation: String,
    },
}

// Re-export for backwards compatibility
pub use voice_agent_config::PersonaConfig as PersonaTraits;
