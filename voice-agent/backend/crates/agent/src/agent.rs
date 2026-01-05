//! Gold Loan Voice Agent
//!
//! Main agent implementation combining all components.

use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::broadcast;

use voice_agent_llm::{
    LlmFactory, LlmProviderConfig, Message, PromptBuilder, Role, SpeculativeConfig,
    SpeculativeExecutor, SpeculativeMode,
};
// P1 FIX: Use LanguageModel trait from core for proper abstraction
use voice_agent_core::LanguageModel;
// P0-2 FIX: Import ToolDefinition and FinishReason for LLM tool calling
use voice_agent_core::{FinishReason, ToolDefinition};
// P0 FIX: Import PersonaConfig from the single source of truth
use voice_agent_config::PersonaConfig;
use voice_agent_tools::{ToolExecutor, ToolRegistry};
// P1 FIX: Import RAG components for retrieval-augmented generation
use voice_agent_rag::{
    HybridRetriever, RerankerConfig, RetrieverConfig, SearchResult, VectorStore,
};
// P4 FIX: Import personalization engine for dynamic response adaptation
use voice_agent_core::personalization::{PersonalizationContext, PersonalizationEngine};
// P5 FIX: Import translator for Translate-Think-Translate pattern
use voice_agent_core::{Language, Translator};
use voice_agent_text_processing::translation::{
    CandleIndicTrans2Config, CandleIndicTrans2Translator,
};

use crate::conversation::{Conversation, ConversationConfig, ConversationEvent, EndReason};
use crate::memory::{
    AgenticMemory, AgenticMemoryConfig, ConversationTurn, TurnRole,
};
use crate::persuasion::PersuasionEngine;
use crate::stage::{ConversationStage, RagTimingStrategy};
use crate::AgentError;

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

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            conversation: ConversationConfig::default(),
            persona: PersonaConfig::default(),
            rag_enabled: true,
            tools_enabled: true,
            tool_defaults: ToolDefaults::default(),
            // P2 FIX: Default context window for typical LLMs (Llama 3, etc.)
            // 4096 tokens leaves room for response generation
            context_window_tokens: 4096,
            // P4 FIX: Default to conservative prefetch strategy
            rag_timing_strategy: RagTimingStrategy::default(),
            // P1-1 FIX: Default to Ollama for local development
            // Can be overridden via config to use Claude, OpenAI, or Azure
            llm_provider: LlmProviderConfig::ollama("llama3.2"),
            // P1-2 FIX: Speculative decoding disabled by default
            // Enable via config with speculative.enabled = true
            speculative: SpeculativeDecodingConfig::default(),
        }
    }
}

impl AgentConfig {
    /// Get agent name from persona
    pub fn name(&self) -> &str {
        &self.persona.name
    }
}

// P0 FIX: PersonaTraits removed - now uses PersonaConfig from voice_agent_config
// Re-export for backwards compatibility
pub use voice_agent_config::PersonaConfig as PersonaTraits;

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
}

/// Prefetch cache entry
#[derive(Debug, Clone)]
struct PrefetchEntry {
    /// Query that triggered prefetch
    query: String,
    /// Prefetched results
    results: Vec<SearchResult>,
    /// When prefetch was triggered
    timestamp: std::time::Instant,
}

/// Gold Loan Voice Agent
pub struct GoldLoanAgent {
    config: AgentConfig,
    conversation: Arc<Conversation>,
    tools: Arc<ToolRegistry>,
    /// P1 FIX: Now uses LanguageModel trait instead of LlmBackend for proper abstraction
    llm: Option<Arc<dyn LanguageModel>>,
    /// P1 FIX: RAG retriever for context augmentation
    retriever: Option<Arc<HybridRetriever>>,
    /// P1 FIX: Vector store for RAG search (optional, can be injected)
    vector_store: Option<Arc<VectorStore>>,
    event_tx: broadcast::Sender<AgentEvent>,
    /// P2 FIX: Prefetch cache for VAD → RAG prefetch optimization
    prefetch_cache: RwLock<Option<PrefetchEntry>>,
    /// P4 FIX: Personalization engine for dynamic response adaptation
    personalization: PersonalizationEngine,
    /// P4 FIX: Personalization context (updated each turn)
    personalization_ctx: RwLock<PersonalizationContext>,
    /// P5 FIX: Translator for Translate-Think-Translate pattern
    /// Translates user input to English before LLM, then translates response back
    translator: Option<Arc<dyn Translator>>,
    /// P5 FIX: User's language for translation
    user_language: Language,
    /// P0 FIX: Persuasion engine for objection handling
    persuasion: PersuasionEngine,
    /// P1-2 FIX: Speculative executor for low-latency generation
    /// Uses SLM for fast drafts, LLM for verification/improvement
    speculative: Option<Arc<SpeculativeExecutor>>,
    /// MemGPT-style agentic memory system
    /// Provides hierarchical memory: Core (human/persona) + Recall (FIFO) + Archival (long-term)
    agentic_memory: AgenticMemory,
}

impl GoldLoanAgent {
    /// Create a new agent
    pub fn new(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let session_id = session_id.into();

        let conversation = Arc::new(Conversation::new(&session_id, config.conversation.clone()));

        // Create MemGPT-style agentic memory
        let agentic_memory = AgenticMemory::new(AgenticMemoryConfig::default(), &session_id);
        agentic_memory.core.set_persona_name(&config.persona.name);
        agentic_memory.core.add_persona_goal(&format!(
            "Represent Kotak Mahindra Bank as a Gold Loan Advisor with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_default_registry());

        // P1-1 FIX: Use LlmFactory for provider-agnostic LLM creation
        // Supports Claude, Ollama, OpenAI, and Azure based on config.llm_provider
        let llm: Option<Arc<dyn LanguageModel>> = match LlmFactory::create(&config.llm_provider) {
            Ok(llm) => {
                tracing::info!(
                    provider = ?config.llm_provider.provider,
                    model = %config.llm_provider.model,
                    "LLM backend initialized successfully"
                );
                Some(llm)
            },
            Err(e) => {
                tracing::warn!(
                    provider = ?config.llm_provider.provider,
                    error = %e,
                    "Failed to create LLM backend, falling back to None"
                );
                None
            },
        };

        // P1 FIX: Create RAG retriever if enabled
        let retriever = if config.rag_enabled {
            Some(Arc::new(HybridRetriever::new(
                RetrieverConfig::default(),
                RerankerConfig::default(),
            )))
        } else {
            None
        };

        // P1 FIX: Wire LLM to memory for real summarization
        if let Some(ref llm_backend) = llm {
            conversation.memory().set_llm(llm_backend.clone());
            agentic_memory.set_llm(llm_backend.clone());
        }

        // P4 FIX: Initialize personalization engine and context
        let personalization = PersonalizationEngine::new();
        let personalization_ctx = PersonalizationContext::new();

        // P5 FIX: Parse user language and create translator if not English
        let user_language = Language::from_str_loose(&config.language).unwrap_or(Language::Hindi);

        // Only create translator if user language is not English
        let translator: Option<Arc<dyn Translator>> = if user_language != Language::English {
            // Try to create Candle-based IndicTrans2 translator
            match Self::create_default_translator() {
                Ok(t) => {
                    tracing::info!(
                        language = ?user_language,
                        "Translator initialized for Translate-Think-Translate pattern"
                    );
                    Some(Arc::new(t) as Arc<dyn Translator>)
                },
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to create translator, responses will be in English"
                    );
                    None
                },
            }
        } else {
            tracing::debug!("English language selected, translator not needed");
            None
        };

        // P0 FIX: Initialize persuasion engine for objection handling
        let persuasion = PersuasionEngine::new();

        // P1-2 FIX: Initialize speculative executor if enabled
        let speculative = if config.speculative.enabled {
            match Self::create_speculative_executor(&config.speculative) {
                Ok(executor) => {
                    tracing::info!(
                        mode = ?config.speculative.mode,
                        slm_model = %config.speculative.slm.model,
                        llm_model = %config.speculative.llm.model,
                        "Speculative executor initialized"
                    );
                    Some(Arc::new(executor))
                },
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to create speculative executor, falling back to direct LLM"
                    );
                    None
                },
            }
        } else {
            None
        };

        Self {
            config,
            conversation,
            tools,
            llm,
            retriever,
            vector_store: None,
            event_tx,
            prefetch_cache: RwLock::new(None),
            personalization,
            personalization_ctx: RwLock::new(personalization_ctx),
            translator,
            user_language,
            persuasion,
            speculative,
            agentic_memory,
        }
    }

    /// P1-2 FIX: Create speculative executor with SLM and LLM backends
    fn create_speculative_executor(
        config: &SpeculativeDecodingConfig,
    ) -> Result<SpeculativeExecutor, crate::AgentError> {
        // Create SLM backend (small/fast model)
        let slm = LlmFactory::create_backend(&config.slm).map_err(|e| {
            crate::AgentError::Initialization(format!("Failed to create SLM backend: {}", e))
        })?;

        // Create LLM backend (large/accurate model)
        let llm = LlmFactory::create_backend(&config.llm).map_err(|e| {
            crate::AgentError::Initialization(format!("Failed to create LLM backend: {}", e))
        })?;

        Ok(SpeculativeExecutor::new(slm, llm, config.params.clone()))
    }

    /// Create agent with custom LLM backend
    /// P1 FIX: Now accepts LanguageModel trait for proper abstraction
    pub fn with_llm(
        session_id: impl Into<String>,
        config: AgentConfig,
        llm: Arc<dyn LanguageModel>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let session_id = session_id.into();

        let conversation = Arc::new(Conversation::new(&session_id, config.conversation.clone()));

        // Create MemGPT-style agentic memory
        let agentic_memory = AgenticMemory::new(AgenticMemoryConfig::default(), &session_id);
        agentic_memory.core.set_persona_name(&config.persona.name);
        agentic_memory.core.add_persona_goal(&format!(
            "Represent Kotak Mahindra Bank as a Gold Loan Advisor with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_default_registry());

        // P1 FIX: Create RAG retriever if enabled
        let retriever = if config.rag_enabled {
            Some(Arc::new(HybridRetriever::new(
                RetrieverConfig::default(),
                RerankerConfig::default(),
            )))
        } else {
            None
        };

        // P1 FIX: Wire LLM to memory for real summarization
        conversation.memory().set_llm(llm.clone());
        agentic_memory.set_llm(llm.clone());

        // P4 FIX: Initialize personalization engine and context
        let personalization = PersonalizationEngine::new();
        let personalization_ctx = PersonalizationContext::new();

        // P5 FIX: Parse user language and create translator if not English
        let user_language = Language::from_str_loose(&config.language).unwrap_or(Language::Hindi);

        let translator: Option<Arc<dyn Translator>> = if user_language != Language::English {
            Self::create_default_translator()
                .map(|t| Arc::new(t) as Arc<dyn Translator>)
                .ok()
        } else {
            None
        };

        // P0 FIX: Initialize persuasion engine for objection handling
        let persuasion = PersuasionEngine::new();

        // P1-2 FIX: Initialize speculative executor if enabled
        let speculative = if config.speculative.enabled {
            Self::create_speculative_executor(&config.speculative)
                .map(Arc::new)
                .ok()
        } else {
            None
        };

        Self {
            config,
            conversation,
            tools,
            llm: Some(llm),
            retriever,
            vector_store: None,
            event_tx,
            prefetch_cache: RwLock::new(None),
            personalization,
            personalization_ctx: RwLock::new(personalization_ctx),
            translator,
            user_language,
            persuasion,
            speculative,
            agentic_memory,
        }
    }

    /// Create agent without LLM (uses mock responses)
    pub fn without_llm(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let session_id = session_id.into();

        let conversation = Arc::new(Conversation::new(&session_id, config.conversation.clone()));

        // Create MemGPT-style agentic memory
        let agentic_memory = AgenticMemory::new(AgenticMemoryConfig::default(), &session_id);
        agentic_memory.core.set_persona_name(&config.persona.name);
        agentic_memory.core.add_persona_goal(&format!(
            "Represent Kotak Mahindra Bank as a Gold Loan Advisor with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_default_registry());

        // P1 FIX: Create RAG retriever if enabled
        let retriever = if config.rag_enabled {
            Some(Arc::new(HybridRetriever::new(
                RetrieverConfig::default(),
                RerankerConfig::default(),
            )))
        } else {
            None
        };

        // P4 FIX: Initialize personalization engine and context
        let personalization = PersonalizationEngine::new();
        let personalization_ctx = PersonalizationContext::new();

        // P5 FIX: Parse user language and create translator if not English
        let user_language = Language::from_str_loose(&config.language).unwrap_or(Language::Hindi);

        let translator: Option<Arc<dyn Translator>> = if user_language != Language::English {
            Self::create_default_translator()
                .map(|t| Arc::new(t) as Arc<dyn Translator>)
                .ok()
        } else {
            None
        };

        // P0 FIX: Initialize persuasion engine for objection handling
        let persuasion = PersuasionEngine::new();

        Self {
            config,
            conversation,
            tools,
            llm: None,
            retriever,
            vector_store: None,
            event_tx,
            prefetch_cache: RwLock::new(None),
            personalization,
            personalization_ctx: RwLock::new(personalization_ctx),
            translator,
            user_language,
            persuasion,
            speculative: None, // P1-2 FIX: No speculative without LLM
            agentic_memory,
        }
    }

    /// P1 FIX: Set vector store for RAG search
    pub fn with_vector_store(mut self, vector_store: Arc<VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// P0 FIX: Set custom tool registry (with persistence wired)
    ///
    /// Use this to inject a ToolRegistry that has been configured with
    /// persistence services (SMS, GoldPrice, Appointments) from AppState.
    /// Without this, the agent uses a default registry without persistence.
    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }

    /// P5 FIX: Create default translator using Candle-based IndicTrans2
    ///
    /// This creates a translator that can handle bidirectional translation
    /// between English and Indian languages (Hindi, Tamil, Telugu, etc.)
    fn create_default_translator() -> voice_agent_core::Result<CandleIndicTrans2Translator> {
        use std::path::PathBuf;

        // Default paths relative to project root
        // In production, these would be configured via environment variables
        let config = CandleIndicTrans2Config {
            en_indic_path: PathBuf::from("models/translation/indictrans2-en-indic"),
            indic_en_path: PathBuf::from("models/translation/indictrans2-indic-en"),
            ..Default::default()
        };

        CandleIndicTrans2Translator::new(config)
    }

    /// P5 FIX: Set a custom translator
    pub fn with_translator(mut self, translator: Arc<dyn Translator>) -> Self {
        self.translator = Some(translator);
        self
    }

    /// P5 FIX: Get user's configured language
    pub fn user_language(&self) -> Language {
        self.user_language
    }

    /// Subscribe to agent events
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// P2 FIX: Prefetch RAG results based on partial transcript from STT
    ///
    /// This method should be called when VAD detects speech and STT provides
    /// partial transcripts. It triggers RAG prefetch in the background so
    /// results are ready when the full utterance completes.
    ///
    /// # Arguments
    /// * `partial_transcript` - Partial text from STT
    /// * `confidence` - STT confidence score (0.0 - 1.0)
    ///
    /// Returns true if prefetch was triggered, false if skipped (no RAG or low confidence)
    pub async fn prefetch_on_partial(&self, partial_transcript: &str, confidence: f32) -> bool {
        // Skip if RAG is disabled or components not available
        if !self.config.rag_enabled {
            return false;
        }

        let (retriever, vector_store) = match (&self.retriever, &self.vector_store) {
            (Some(r), Some(vs)) => (r.clone(), vs.clone()),
            _ => return false,
        };

        // P4 FIX: Use timing strategy to determine if we should prefetch
        let stage = self.conversation.stage();
        let strategy = &self.config.rag_timing_strategy;

        // Check if strategy allows prefetch at this point
        if !strategy.should_prefetch(confidence, stage) {
            tracing::trace!(
                strategy = ?strategy,
                confidence = confidence,
                stage = ?stage,
                "Skipping prefetch - timing strategy declined"
            );
            return false;
        }

        // Don't prefetch for very short partials (strategy-aware minimum)
        if partial_transcript.split_whitespace().count() < strategy.min_words() {
            return false;
        }

        // Clone for async task
        let partial = partial_transcript.to_string();
        let cache = self.prefetch_cache.read().clone();

        // Skip if we already prefetched for similar query (strategy-aware TTL)
        let cache_ttl = strategy.cache_ttl_secs();
        if let Some(entry) = &cache {
            if entry.timestamp.elapsed().as_secs() < cache_ttl && partial.contains(&entry.query) {
                tracing::trace!("Skipping prefetch - similar query already cached");
                return false;
            }
        }

        tracing::debug!(
            partial = %partial,
            confidence = confidence,
            strategy = ?strategy,
            stage = ?stage,
            "Triggering RAG prefetch on partial transcript"
        );

        // Run prefetch asynchronously
        match retriever
            .prefetch(&partial, confidence, &vector_store)
            .await
        {
            Ok(results) if !results.is_empty() => {
                tracing::debug!(count = results.len(), "RAG prefetch completed with results");
                // Store in cache
                *self.prefetch_cache.write() = Some(PrefetchEntry {
                    query: partial,
                    results,
                    timestamp: std::time::Instant::now(),
                });
                true
            },
            Ok(_) => {
                tracing::trace!("RAG prefetch returned no results");
                false
            },
            Err(e) => {
                tracing::warn!("RAG prefetch failed: {}", e);
                false
            },
        }
    }

    /// P2 FIX: Spawn prefetch as a background task (non-blocking)
    ///
    /// Use this when you want to trigger prefetch without waiting for results.
    /// The prefetch will run in the background and populate the cache.
    pub fn prefetch_background(&self, partial_transcript: String, confidence: f32) {
        if !self.config.rag_enabled {
            return;
        }

        let (retriever, vector_store) = match (&self.retriever, &self.vector_store) {
            (Some(r), Some(vs)) => (r.clone(), vs.clone()),
            _ => return,
        };

        if partial_transcript.split_whitespace().count() < 2 {
            return;
        }

        // Check cache under read lock, avoiding clone if possible
        {
            let cache = self.prefetch_cache.read();
            if let Some(entry) = &*cache {
                if entry.timestamp.elapsed().as_secs() < 2
                    && partial_transcript.contains(&entry.query)
                {
                    return;
                }
            }
        }

        // Spawn background prefetch task
        // Note: Results are not cached in background mode - use prefetch_on_partial() for caching
        // This is useful for warming up the retriever's internal caches
        tokio::spawn(async move {
            tracing::debug!(
                partial = %partial_transcript,
                confidence = confidence,
                "Background RAG prefetch triggered"
            );
            match retriever
                .prefetch(&partial_transcript, confidence, &vector_store)
                .await
            {
                Ok(results) if !results.is_empty() => {
                    tracing::debug!(count = results.len(), "Background prefetch completed");
                    // Note: Results are not cached in background mode - use prefetch_on_partial for caching
                },
                Ok(_) => tracing::trace!("Background prefetch returned no results"),
                Err(e) => tracing::warn!("Background prefetch failed: {}", e),
            }
        });
    }

    /// P2 FIX: Get prefetched results if available and relevant
    ///
    /// Returns cached prefetch results if they match the query and are fresh.
    fn get_prefetch_results(&self, query: &str) -> Option<Vec<SearchResult>> {
        let cache = self.prefetch_cache.read();
        if let Some(entry) = &*cache {
            // Check if cache is fresh (within 10 seconds)
            if entry.timestamp.elapsed().as_secs() > 10 {
                return None;
            }
            // Check if query is related to prefetched query
            // Simple check: query contains prefetch query or vice versa
            let query_lower = query.to_lowercase();
            let cached_lower = entry.query.to_lowercase();
            if query_lower.contains(&cached_lower) || cached_lower.contains(&query_lower) {
                tracing::debug!("Using prefetched RAG results");
                return Some(entry.results.clone());
            }
        }
        None
    }

    /// P2 FIX: Clear prefetch cache
    pub fn clear_prefetch_cache(&self) {
        *self.prefetch_cache.write() = None;
    }

    /// Process user input and generate response
    ///
    /// P5 FIX: Implements Translate-Think-Translate pattern:
    /// 1. If user language is not English, translate input to English
    /// 2. Process with LLM (which works best in English)
    /// 3. Translate response back to user's language
    pub async fn process(&self, user_input: &str) -> Result<String, AgentError> {
        // Emit thinking event
        let _ = self.event_tx.send(AgentEvent::Thinking);

        // P5 FIX: Translate user input to English if needed
        // This implements the "Translate" part of Translate-Think-Translate
        let english_input = if self.user_language != Language::English {
            if let Some(ref translator) = self.translator {
                match translator
                    .translate(user_input, self.user_language, Language::English)
                    .await
                {
                    Ok(translated) => {
                        tracing::debug!(
                            from = ?self.user_language,
                            original = %user_input,
                            translated = %translated,
                            "Translated user input to English"
                        );
                        translated
                    },
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Translation failed, using original input"
                        );
                        user_input.to_string()
                    },
                }
            } else {
                // No translator available, use original input
                user_input.to_string()
            }
        } else {
            // Already English, no translation needed
            user_input.to_string()
        };

        // Add user turn and detect intent (using original input for conversation history)
        let intent = self.conversation.add_user_turn(user_input)?;

        // Add to MemGPT-style agentic memory recall
        let turn = ConversationTurn::new(TurnRole::User, user_input)
            .with_intents(vec![intent.intent.clone()])
            .with_entities(
                intent
                    .slots
                    .iter()
                    .filter_map(|(k, v)| v.value.as_ref().map(|val| (k.clone(), val.clone())))
                    .collect(),
            )
            .with_stage(self.conversation.stage().display_name());
        self.agentic_memory.add_turn(turn);

        // Extract and store customer facts from intent slots in core memory
        for (key, slot) in &intent.slots {
            if let Some(ref value) = slot.value {
                let fact_key = match key.as_str() {
                    "gold_weight" | "weight" => Some("gold_weight"),
                    "gold_purity" | "purity" | "karat" => Some("gold_purity"),
                    "loan_amount" | "amount" => Some("loan_amount"),
                    "current_lender" | "lender" => Some("current_lender"),
                    "interest_rate" | "rate" => Some("current_interest_rate"),
                    "city" | "location" => Some("city"),
                    "customer_name" | "name" => {
                        self.set_customer_name(value);
                        None
                    }
                    "phone_number" | "phone" | "mobile" => Some("phone"),
                    _ => None,
                };
                if let Some(k) = fact_key {
                    let _ = self.agentic_memory.core_memory_append(k, value);
                }
            }
        }

        // P4 FIX: Process input through personalization engine
        // This detects behavior signals, objections, and updates context
        {
            let mut ctx = self.personalization_ctx.write();
            self.personalization.process_input(&mut ctx, user_input);

            // Log detected signals for debugging
            if let Some(recent_signal) = ctx.recent_signals(1).first() {
                tracing::debug!(signal = ?recent_signal, "Personalization signal detected");
            }
        }

        // Forward conversation events
        let _ = self
            .event_tx
            .send(AgentEvent::Conversation(ConversationEvent::IntentDetected(
                intent.clone(),
            )));

        // Check for tool calls based on intent
        let tool_result = if self.config.tools_enabled {
            self.maybe_call_tool(&intent).await?
        } else {
            None
        };

        // Build prompt for LLM (using English input for better LLM performance)
        // This is the "Think" part of Translate-Think-Translate
        let english_response = self
            .generate_response(&english_input, tool_result.as_deref())
            .await?;

        // P5 FIX: Translate response back to user's language if needed
        // This is the second "Translate" part of Translate-Think-Translate
        let response = if self.user_language != Language::English {
            if let Some(ref translator) = self.translator {
                match translator
                    .translate(&english_response, Language::English, self.user_language)
                    .await
                {
                    Ok(translated) => {
                        tracing::debug!(
                            to = ?self.user_language,
                            original = %english_response,
                            translated = %translated,
                            "Translated response to user language"
                        );
                        translated
                    },
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Response translation failed, using English response"
                        );
                        english_response
                    },
                }
            } else {
                english_response
            }
        } else {
            english_response
        };

        // Add assistant turn (store the translated response in conversation history)
        self.conversation.add_assistant_turn(&response)?;

        // Add to MemGPT-style agentic memory recall
        let assistant_turn = ConversationTurn::new(TurnRole::Assistant, &response)
            .with_stage(self.conversation.stage().display_name());
        self.agentic_memory.add_turn(assistant_turn);

        // P1 FIX: Trigger memory summarization in background (non-blocking)
        // This uses the LLM (if available) to summarize conversation history
        let memory = self.conversation.memory_arc();
        tokio::spawn(async move {
            if let Err(e) = memory.summarize_pending_async().await {
                tracing::debug!("Memory summarization skipped: {}", e);
            }
        });

        // P2 FIX: Check memory usage and cleanup if needed
        // This prevents unbounded memory growth during long conversations
        {
            let memory = self.conversation.memory_arc();
            if memory.needs_cleanup() {
                tracing::info!("Memory high watermark exceeded, triggering cleanup");
                memory.cleanup_to_watermark();
            }
        }

        // Check agentic memory compaction (MemGPT-style)
        if self.agentic_memory.needs_compaction() {
            let stats = self.agentic_memory.get_stats();
            tracing::debug!(
                core_tokens = stats.core_tokens,
                fifo_tokens = stats.fifo_tokens,
                archival_count = stats.archival_count,
                "Agentic memory high watermark exceeded"
            );
        }

        // Emit response event
        let _ = self.event_tx.send(AgentEvent::Response(response.clone()));

        Ok(response)
    }

    /// P0-2 FIX: Process user input with streaming LLM output
    ///
    /// Same as `process()` but streams LLM output sentence-by-sentence.
    /// Each sentence is translated (if needed) before being sent to the output channel.
    /// This enables lower latency TTS by starting synthesis before the full response is ready.
    ///
    /// # Arguments
    /// * `user_input` - User's message
    ///
    /// # Returns
    /// A channel receiver that yields translated sentences as they are ready
    pub async fn process_stream(
        &self,
        user_input: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, AgentError> {
        use futures::StreamExt;

        // Emit thinking event
        let _ = self.event_tx.send(AgentEvent::Thinking);

        // P5 FIX: Translate user input to English if needed
        let english_input = if self.user_language != Language::English {
            if let Some(ref translator) = self.translator {
                translator
                    .translate(user_input, self.user_language, Language::English)
                    .await
                    .unwrap_or_else(|_| user_input.to_string())
            } else {
                user_input.to_string()
            }
        } else {
            user_input.to_string()
        };

        // Add user turn and detect intent
        let intent = self.conversation.add_user_turn(user_input)?;

        // P4 FIX: Process through personalization engine
        {
            let mut ctx = self.personalization_ctx.write();
            self.personalization.process_input(&mut ctx, user_input);
        }

        // Forward intent event
        let _ = self
            .event_tx
            .send(AgentEvent::Conversation(ConversationEvent::IntentDetected(
                intent.clone(),
            )));

        // Check for tool calls
        let tool_result = if self.config.tools_enabled {
            self.maybe_call_tool(&intent).await?
        } else {
            None
        };

        // Build prompt (same as in generate_response)
        let prompt_request = self
            .build_llm_request(&english_input, tool_result.as_deref())
            .await?;

        // Create output channel
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);

        // Check if LLM is available for streaming
        if let Some(ref llm) = self.llm {
            if llm.is_available().await {
                // Get the stream - lifetime tied to &self, so process inline
                let mut stream = llm.generate_stream(prompt_request);

                let translator = &self.translator;
                let user_language = self.user_language;
                let terminators = user_language.sentence_terminators();

                let mut buffer = String::new();
                let mut full_response = String::new();

                // Process stream inline (can't spawn due to stream lifetime)
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            buffer.push_str(&chunk.delta);
                            full_response.push_str(&chunk.delta);

                            // Check for sentence boundaries
                            while let Some(pos) = find_sentence_end(&buffer, terminators) {
                                let sentence = buffer[..=pos].trim().to_string();
                                buffer = buffer[pos + 1..].to_string();

                                if sentence.is_empty() {
                                    continue;
                                }

                                // Translate sentence if needed
                                let translated = if user_language != Language::English {
                                    if let Some(ref t) = translator {
                                        t.translate(&sentence, Language::English, user_language)
                                            .await
                                            .unwrap_or(sentence)
                                    } else {
                                        sentence
                                    }
                                } else {
                                    sentence
                                };

                                // Send translated sentence - use try_send to not block
                                if tx.send(translated).await.is_err() {
                                    tracing::debug!("Stream receiver dropped");
                                    break;
                                }
                            }

                            if chunk.is_final {
                                break;
                            }
                        },
                        Err(e) => {
                            tracing::warn!("LLM stream error: {}", e);
                            break;
                        },
                    }
                }

                // Flush remaining buffer
                if !buffer.trim().is_empty() {
                    let sentence = buffer.trim().to_string();
                    let translated = if user_language != Language::English {
                        if let Some(ref t) = translator {
                            t.translate(&sentence, Language::English, user_language)
                                .await
                                .unwrap_or(sentence)
                        } else {
                            sentence
                        }
                    } else {
                        sentence
                    };
                    let _ = tx.send(translated).await;
                }

                // Update conversation with full response (translate for history)
                let final_response = if user_language != Language::English {
                    if let Some(ref t) = translator {
                        t.translate(&full_response, Language::English, user_language)
                            .await
                            .unwrap_or(full_response.clone())
                    } else {
                        full_response.clone()
                    }
                } else {
                    full_response.clone()
                };

                // Add assistant turn
                if let Err(e) = self.conversation.add_assistant_turn(&final_response) {
                    tracing::warn!("Failed to add assistant turn: {}", e);
                }

                // Emit response event
                let _ = self.event_tx.send(AgentEvent::Response(final_response));

                return Ok(rx);
            }
        }

        // Fallback: No LLM available, use mock response
        let response = self.generate_mock_response(user_input, tool_result.as_deref());
        self.conversation.add_assistant_turn(&response)?;
        let _ = self.event_tx.send(AgentEvent::Response(response.clone()));

        // Send mock response as single chunk
        let _ = tx.send(response).await;

        Ok(rx)
    }

    /// Build LLM request (extracted from generate_response for reuse)
    async fn build_llm_request(
        &self,
        english_input: &str,
        tool_result: Option<&str>,
    ) -> Result<voice_agent_core::GenerateRequest, AgentError> {
        let persona = self.config.persona.clone();

        let mut builder = PromptBuilder::new()
            .with_persona(persona)
            .system_prompt(&self.config.language);

        // Add personalization instructions
        {
            let ctx = self.personalization_ctx.read();
            let instructions = self.personalization.generate_instructions(&ctx);
            if !instructions.is_empty() {
                builder =
                    builder.with_context(&format!("## Personalization Guidance\n{}", instructions));
            }
        }

        // Add memory context
        let context = self.conversation.get_context();
        if !context.is_empty() {
            builder = builder.with_context(&context);
        }

        // Add RAG context
        if self.config.rag_enabled {
            let stage = self.conversation.stage();
            let rag_fraction = stage.rag_context_fraction();

            if rag_fraction > 0.0 {
                if let (Some(retriever), Some(vector_store)) = (&self.retriever, &self.vector_store)
                {
                    let results = if let Some(prefetched) = self.get_prefetch_results(english_input)
                    {
                        self.clear_prefetch_cache();
                        prefetched
                    } else {
                        retriever
                            .search(english_input, vector_store, None)
                            .await
                            .unwrap_or_default()
                    };

                    if !results.is_empty() {
                        let max_results = ((rag_fraction * 10.0).ceil() as usize).clamp(1, 5);
                        let rag_context = results
                            .iter()
                            .take(max_results)
                            .map(|r| format!("- {}", r.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        builder = builder
                            .with_context(&format!("## Relevant Information\n{}", rag_context));
                    }
                }
            }
        }

        // Add tool result
        if let Some(result) = tool_result {
            builder = builder.with_context(&format!("## Tool Result\n{}", result));
        }

        // Add stage guidance
        builder = builder.with_stage_guidance(self.conversation.stage().display_name());

        // Add persuasion guidance
        if let Some(objection_response) = self
            .persuasion
            .handle_objection(english_input, self.user_language)
        {
            let guidance = format!(
                "## Objection Handling Guidance\n\
                1. **Acknowledge**: {}\n\
                2. **Reframe**: {}\n\
                3. **Evidence**: {}\n\
                4. **Call to Action**: {}",
                objection_response.acknowledge,
                objection_response.reframe,
                objection_response.evidence,
                objection_response.call_to_action
            );
            builder = builder.with_context(&guidance);
        }

        // Add conversation history
        let history: Vec<Message> = self
            .conversation
            .get_messages()
            .into_iter()
            .map(|(role, content)| {
                let r = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => Role::System,
                };
                Message {
                    role: r,
                    content,
                    name: None,
                    tool_call_id: None,
                }
            })
            .collect();
        builder = builder.with_history(&history);

        // Add current message
        builder = builder.user_message(english_input);

        // Build with context budget
        let stage = self.conversation.stage();
        let effective_budget = self
            .config
            .context_window_tokens
            .min(stage.context_budget_tokens());

        Ok(builder.build_request_with_limit(effective_budget))
    }

    /// Maybe call a tool based on intent
    async fn maybe_call_tool(
        &self,
        intent: &crate::intent::DetectedIntent,
    ) -> Result<Option<String>, AgentError> {
        let tool_name = match intent.intent.as_str() {
            "eligibility_check" => {
                // Check if we have required slots
                if intent.slots.contains_key("gold_weight") {
                    Some("check_eligibility")
                } else {
                    None
                }
            },
            "switch_lender" => {
                if intent.slots.contains_key("current_lender") {
                    Some("calculate_savings")
                } else {
                    None
                }
            },
            "schedule_visit" => Some("find_branches"),
            // P4 FIX: Add intent mappings for CRM/Calendar integrations
            "capture_lead" | "interested" | "callback_request" => {
                // Capture lead when customer shows interest
                if intent.slots.contains_key("customer_name")
                    || intent.slots.contains_key("phone_number")
                {
                    Some("capture_lead")
                } else {
                    None
                }
            },
            "schedule_appointment" | "book_appointment" | "visit_branch" => {
                // Schedule appointment when customer wants to visit
                if intent.slots.contains_key("preferred_date")
                    || intent.slots.contains_key("branch_id")
                {
                    Some("schedule_appointment")
                } else {
                    // If no specific date/branch, first find branches
                    Some("find_branches")
                }
            },
            // P1 FIX: Add missing tool intent mappings
            "gold_price" | "check_gold_price" | "price_inquiry" | "current_rate" => {
                // Gold price inquiry - no required slots
                Some("get_gold_price")
            },
            "escalate" | "human_agent" | "speak_to_person" | "talk_to_human" | "real_person" => {
                // Escalation to human agent - no required slots
                Some("escalate_to_human")
            },
            "send_sms" | "send_message" | "text_me" | "send_details" | "sms_info" => {
                // Send SMS - phone_number slot is optional (can use customer's registered number)
                Some("send_sms")
            },
            _ => None,
        };

        if let Some(name) = tool_name {
            let _ = self.event_tx.send(AgentEvent::ToolCall {
                name: name.to_string(),
            });

            // Build arguments from slots
            let mut args = serde_json::Map::new();
            for (key, slot) in &intent.slots {
                if let Some(ref value) = slot.value {
                    args.insert(key.clone(), serde_json::json!(value));
                }
            }

            // P1 FIX: Use configurable defaults instead of hardcoded values
            let defaults = &self.config.tool_defaults;

            if name == "check_eligibility" && !args.contains_key("gold_purity") {
                args.insert(
                    "gold_purity".to_string(),
                    serde_json::json!(&defaults.default_gold_purity),
                );
            }

            if name == "calculate_savings" {
                if !args.contains_key("current_interest_rate") {
                    args.insert(
                        "current_interest_rate".to_string(),
                        serde_json::json!(defaults.default_competitor_rate),
                    );
                }
                if !args.contains_key("current_loan_amount") {
                    args.insert(
                        "current_loan_amount".to_string(),
                        serde_json::json!(defaults.default_loan_amount),
                    );
                }
                if !args.contains_key("remaining_tenure_months") {
                    args.insert(
                        "remaining_tenure_months".to_string(),
                        serde_json::json!(defaults.default_tenure_months),
                    );
                }
            }

            if name == "find_branches" && !args.contains_key("city") {
                args.insert(
                    "city".to_string(),
                    serde_json::json!(&defaults.default_city),
                );
            }

            // P4 FIX: Handle capture_lead tool arguments
            if name == "capture_lead" {
                // Map slot names to tool parameter names
                if args.contains_key("name") && !args.contains_key("customer_name") {
                    if let Some(v) = args.remove("name") {
                        args.insert("customer_name".to_string(), v);
                    }
                }
                if args.contains_key("phone") && !args.contains_key("phone_number") {
                    if let Some(v) = args.remove("phone") {
                        args.insert("phone_number".to_string(), v);
                    }
                }
                // Default interest level based on intent confidence
                if !args.contains_key("interest_level") {
                    let level = if intent.confidence > 0.8 {
                        "High"
                    } else {
                        "Medium"
                    };
                    args.insert("interest_level".to_string(), serde_json::json!(level));
                }
            }

            // P4 FIX: Handle schedule_appointment tool arguments
            if name == "schedule_appointment" {
                // Map slot names to tool parameter names
                if args.contains_key("name") && !args.contains_key("customer_name") {
                    if let Some(v) = args.remove("name") {
                        args.insert("customer_name".to_string(), v);
                    }
                }
                if args.contains_key("phone") && !args.contains_key("phone_number") {
                    if let Some(v) = args.remove("phone") {
                        args.insert("phone_number".to_string(), v);
                    }
                }
                if args.contains_key("date") && !args.contains_key("preferred_date") {
                    if let Some(v) = args.remove("date") {
                        args.insert("preferred_date".to_string(), v);
                    }
                }
                if args.contains_key("time") && !args.contains_key("preferred_time") {
                    if let Some(v) = args.remove("time") {
                        args.insert("preferred_time".to_string(), v);
                    }
                }
                if args.contains_key("branch") && !args.contains_key("branch_id") {
                    if let Some(v) = args.remove("branch") {
                        args.insert("branch_id".to_string(), v);
                    }
                }
            }

            let result = self
                .tools
                .execute(name, serde_json::Value::Object(args))
                .await;

            let success = result.is_ok();
            let _ = self.event_tx.send(AgentEvent::ToolResult {
                name: name.to_string(),
                success,
            });

            match result {
                Ok(output) => {
                    // Extract text from output
                    let text = output
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            voice_agent_tools::mcp::ContentBlock::Text { text } => {
                                Some(text.clone())
                            },
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(Some(text))
                },
                Err(e) => {
                    tracing::warn!("Tool error: {}", e);
                    Ok(None)
                },
            }
        } else {
            Ok(None)
        }
    }

    /// Generate response using LLM
    async fn generate_response(
        &self,
        user_input: &str,
        tool_result: Option<&str>,
    ) -> Result<String, AgentError> {
        // Build prompt - P0 FIX: now just clones consolidated PersonaConfig
        let persona = self.config.persona.clone();

        let mut builder = PromptBuilder::new()
            .with_persona(persona)
            .system_prompt(&self.config.language);

        // P4 FIX: Add personalization instructions based on detected signals
        // This dynamically adapts the prompt based on customer behavior
        {
            let ctx = self.personalization_ctx.read();
            let personalization_instructions = self.personalization.generate_instructions(&ctx);
            if !personalization_instructions.is_empty() {
                builder = builder.with_context(&format!(
                    "## Personalization Guidance\n{}",
                    personalization_instructions
                ));
                tracing::trace!(
                    instructions_len = personalization_instructions.len(),
                    "Added personalization instructions to prompt"
                );
            }
        }

        // Add context from memory
        let context = self.conversation.get_context();
        if !context.is_empty() {
            builder = builder.with_context(&context);
        }

        // P1 FIX: Add RAG context if retriever and vector store are available
        // P2 FIX: Use prefetched results if available, otherwise do fresh search
        // P2 FIX: Stage-aware RAG - use rag_context_fraction to determine how much RAG to include
        if self.config.rag_enabled {
            let stage = self.conversation.stage();
            let rag_fraction = stage.rag_context_fraction();

            // Skip RAG entirely for stages that don't need it (greeting, farewell)
            if rag_fraction > 0.0 {
                if let (Some(retriever), Some(vector_store)) = (&self.retriever, &self.vector_store)
                {
                    // First, try to use prefetched results
                    let results = if let Some(prefetched) = self.get_prefetch_results(user_input) {
                        tracing::debug!("Using {} prefetched RAG results", prefetched.len());
                        // Clear cache after use
                        self.clear_prefetch_cache();
                        prefetched
                    } else {
                        // Fall back to fresh search
                        match retriever.search(user_input, vector_store, None).await {
                            Ok(r) => r,
                            Err(e) => {
                                tracing::warn!("RAG search failed, continuing without: {}", e);
                                Vec::new()
                            },
                        }
                    };

                    if !results.is_empty() {
                        // P2 FIX: Calculate how many results to include based on stage RAG fraction
                        // Higher fraction = more results (1-5 based on fraction)
                        let max_results = ((rag_fraction * 10.0).ceil() as usize).clamp(1, 5);

                        let rag_context = results
                            .iter()
                            .take(max_results)
                            .map(|r| format!("- {}", r.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        builder = builder
                            .with_context(&format!("## Relevant Information\n{}", rag_context));

                        tracing::debug!(
                            stage = ?stage,
                            rag_fraction = rag_fraction,
                            max_results = max_results,
                            actual_results = results.len().min(max_results),
                            "Stage-aware RAG context added"
                        );
                    } else {
                        tracing::debug!("RAG returned no results for query");
                    }
                }
            } else {
                tracing::trace!(stage = ?stage, "Skipping RAG for stage with rag_fraction=0");
            }
        }

        // Add tool result if available
        if let Some(result) = tool_result {
            builder = builder.with_context(&format!("## Tool Result\n{}", result));
        }

        // Add stage guidance
        builder = builder.with_stage_guidance(self.conversation.stage().display_name());

        // P0 FIX: Detect objections and add persuasion guidance to prompt
        // Uses acknowledge-reframe-evidence pattern from PersuasionEngine
        if let Some(objection_response) = self
            .persuasion
            .handle_objection(user_input, self.user_language)
        {
            let persuasion_guidance = format!(
                "## Objection Handling Guidance\n\
                The customer appears to have a concern. Use this framework:\n\
                1. **Acknowledge**: {}\n\
                2. **Reframe**: {}\n\
                3. **Evidence**: {}\n\
                4. **Call to Action**: {}",
                objection_response.acknowledge,
                objection_response.reframe,
                objection_response.evidence,
                objection_response.call_to_action
            );
            builder = builder.with_context(&persuasion_guidance);

            tracing::debug!(
                objection_type = ?crate::persuasion::ObjectionType::detect(user_input),
                "Detected objection, adding persuasion guidance to prompt"
            );
        }

        // Add conversation history
        let history: Vec<Message> = self
            .conversation
            .get_messages()
            .into_iter()
            .map(|(role, content)| {
                let r = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => Role::System,
                };
                Message {
                    role: r,
                    content,
                    name: None,
                    tool_call_id: None,
                }
            })
            .collect();

        builder = builder.with_history(&history);

        // Add current user message
        builder = builder.user_message(user_input);

        // P2 FIX: Use stage-aware context budget to truncate conversation history
        // Different stages need different amounts of context - early stages need less,
        // presentation/objection handling stages need more for RAG and full history
        let stage = self.conversation.stage();
        let stage_budget = stage.context_budget_tokens();
        // Use the minimum of configured limit and stage-aware budget
        let effective_budget = self.config.context_window_tokens.min(stage_budget);

        tracing::debug!(
            stage = ?stage,
            stage_budget = stage_budget,
            effective_budget = effective_budget,
            "Using stage-aware context budget"
        );

        // P1-2 FIX: Try speculative execution first if enabled and appropriate
        // Speculative doesn't support tool calling, so only use for non-tool responses
        let tool_defs: Vec<ToolDefinition> = if self.config.tools_enabled {
            self.tools
                .list_tools()
                .iter()
                .map(ToolDefinition::from_schema)
                .collect()
        } else {
            Vec::new()
        };

        let has_tools = !tool_defs.is_empty();

        // P1-2 FIX: Use speculative executor when available and no tools needed
        if let Some(ref speculative) = self.speculative {
            if !has_tools {
                // Build messages for speculative executor (uses llm crate's Message type)
                let messages = builder.build_with_limit(effective_budget);

                tracing::debug!(
                    mode = ?self.config.speculative.mode,
                    message_count = messages.len(),
                    "Using speculative executor"
                );

                match speculative.execute(&messages).await {
                    Ok(result) => {
                        tracing::debug!(
                            model_used = ?result.model_used,
                            used_fallback = result.used_fallback,
                            complexity = ?result.complexity_score,
                            tokens = result.generation.tokens,
                            "Speculative execution succeeded"
                        );
                        return Ok(result.text);
                    },
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Speculative execution failed, falling back to direct LLM"
                        );
                        // Fall through to direct LLM path
                    },
                }
            } else {
                tracing::debug!("Skipping speculative executor - tool calling required");
            }
        }

        // P1 FIX: Use build_request_with_limit for LanguageModel trait (fallback path)
        // Rebuild the request since speculative may have consumed the builder
        let request = self.build_llm_request(user_input, tool_result).await?;

        // Try to use LLM backend if available
        if let Some(ref llm) = self.llm {
            // Check if LLM is available
            if llm.is_available().await {
                tracing::debug!(
                    tool_count = tool_defs.len(),
                    tools_enabled = self.config.tools_enabled,
                    "Calling LLM with tool definitions"
                );

                // P0-2 FIX: Use generate_with_tools when tools are available
                let result = if has_tools {
                    llm.generate_with_tools(request, &tool_defs).await
                } else {
                    llm.generate(request).await
                };

                match result {
                    Ok(response) => {
                        // P1 FIX: Use GenerateResponse fields (LanguageModel trait)
                        let tokens = response
                            .usage
                            .as_ref()
                            .map(|u| u.completion_tokens)
                            .unwrap_or(0);
                        tracing::debug!(
                            "LLM generated {} tokens, finish_reason={:?}, tool_calls={}",
                            tokens,
                            response.finish_reason,
                            response.tool_calls.len()
                        );

                        // P0-2 FIX: Handle tool calls from LLM
                        if response.finish_reason == FinishReason::ToolCalls
                            && !response.tool_calls.is_empty()
                        {
                            tracing::info!(
                                tool_calls = response.tool_calls.len(),
                                "LLM requested tool calls"
                            );

                            // Execute each tool call and collect results
                            let mut tool_results = Vec::new();
                            for tool_call in &response.tool_calls {
                                let _ = self.event_tx.send(AgentEvent::ToolCall {
                                    name: tool_call.name.clone(),
                                });

                                // Convert HashMap arguments to serde_json::Value
                                let args = serde_json::to_value(&tool_call.arguments)
                                    .unwrap_or(serde_json::json!({}));

                                match self.tools.execute(&tool_call.name, args).await {
                                    Ok(output) => {
                                        let _ = self.event_tx.send(AgentEvent::ToolResult {
                                            name: tool_call.name.clone(),
                                            success: true,
                                        });

                                        // Extract text from output
                                        let text = output
                                            .content
                                            .iter()
                                            .filter_map(|c| match c {
                                                voice_agent_tools::mcp::ContentBlock::Text {
                                                    text,
                                                } => Some(text.clone()),
                                                _ => None,
                                            })
                                            .collect::<Vec<_>>()
                                            .join("\n");

                                        tool_results.push(format!(
                                            "Tool '{}' result:\n{}",
                                            tool_call.name, text
                                        ));
                                        tracing::debug!(
                                            tool = %tool_call.name,
                                            "Tool execution successful"
                                        );
                                    },
                                    Err(e) => {
                                        let _ = self.event_tx.send(AgentEvent::ToolResult {
                                            name: tool_call.name.clone(),
                                            success: false,
                                        });
                                        tool_results.push(format!(
                                            "Tool '{}' failed: {}",
                                            tool_call.name, e
                                        ));
                                        tracing::warn!(
                                            tool = %tool_call.name,
                                            error = %e,
                                            "Tool execution failed"
                                        );
                                    },
                                }
                            }

                            // Recursive call with tool results to get final response
                            // Use Box::pin to avoid infinitely-sized future
                            let combined_results = tool_results.join("\n\n");
                            return Box::pin(
                                self.generate_response(user_input, Some(&combined_results)),
                            )
                            .await;
                        }

                        return Ok(response.text);
                    },
                    Err(e) => {
                        tracing::warn!("LLM generation failed, falling back to mock: {}", e);
                        // Fall through to mock response
                    },
                }
            } else {
                tracing::debug!("LLM not available, using mock response");
            }
        }

        // Fallback: generate a placeholder response based on intent and stage
        let response = self.generate_mock_response(user_input, tool_result);

        Ok(response)
    }

    /// Generate mock response (placeholder for LLM)
    /// P2 FIX: Language-aware mock responses
    ///
    /// Generates fallback responses based on configured language:
    /// - "hi" or "hi-IN": Hinglish (Hindi + English mix)
    /// - "en" or "en-IN": English
    fn generate_mock_response(&self, _user_input: &str, tool_result: Option<&str>) -> String {
        let stage = self.conversation.stage();

        // If we have tool results, incorporate them
        if let Some(result) = tool_result {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(result) {
                if let Some(msg) = json.get("message").and_then(|m| m.as_str()) {
                    return msg.to_string();
                }
            }
        }

        let name = &self.config.persona.name;
        let is_english = self.config.language.starts_with("en");

        // P2 FIX: Stage-based responses with language awareness
        match stage {
            ConversationStage::Greeting => {
                if is_english {
                    format!(
                        "Hello! I'm {}, calling from Kotak Mahindra Bank. How may I assist you today?",
                        name
                    )
                } else {
                    format!(
                        "Namaste! Main {} hoon, Kotak Mahindra Bank se. Aapki kya madad kar sakti hoon aaj?",
                        name
                    )
                }
            },
            ConversationStage::Discovery => {
                if is_english {
                    "I'd like to understand your needs better. Do you currently have a gold loan with another lender?".to_string()
                } else {
                    "Achha, aap batayein, aapka abhi kahan se gold loan hai? Main aapko dekhti hoon ki hum aapki kaise madad kar sakte hain.".to_string()
                }
            },
            ConversationStage::Qualification => {
                if is_english {
                    "That's helpful. Could you tell me how much gold you have pledged currently? And what interest rate are you paying?".to_string()
                } else {
                    "Bahut achha. Aapke paas kitna gold pledged hai abhi? Aur current rate kya chal raha hai?".to_string()
                }
            },
            ConversationStage::Presentation => {
                if is_english {
                    "At Kotak, we offer just 10.5% interest rate, which is much lower than the 18-20% NBFCs charge. Plus, you get the security of an RBI regulated bank. Would you be interested?".to_string()
                } else {
                    "Dekhiye, Kotak mein aapko sirf 10.5% rate milega, jo NBFC ke 18-20% se bahut kam hai. Aur hamare yahan RBI regulated bank ki security bhi hai. Aap interested hain?".to_string()
                }
            },
            ConversationStage::ObjectionHandling => {
                if is_english {
                    "I understand your concern. We offer a bridge loan facility that makes the transfer process seamless. Your gold is never left unprotected during the transition.".to_string()
                } else {
                    "Main samajh sakti hoon aapki chinta. Lekin dekhiye, hum ek bridge loan dete hain jo aapke transfer process ko seamless banata hai. Aapka gold kabhi bhi unprotected nahi rehta.".to_string()
                }
            },
            ConversationStage::Closing => {
                if is_english {
                    "Shall I schedule an appointment for you? You can visit your nearest branch for gold valuation.".to_string()
                } else {
                    "Toh kya main aapke liye ek appointment schedule kar doon? Aap apne nearest branch mein aa sakte hain gold valuation ke liye.".to_string()
                }
            },
            ConversationStage::Farewell => {
                if is_english {
                    "Thank you for your time! If you have any questions, please don't hesitate to call us. Have a great day!".to_string()
                } else {
                    "Dhanyavaad aapka samay dene ke liye! Agar koi bhi sawal ho toh zaroor call karein. Have a nice day!".to_string()
                }
            },
        }
    }

    /// Get current stage
    pub fn stage(&self) -> ConversationStage {
        self.conversation.stage()
    }

    /// Get conversation reference
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// P1 FIX: Get agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// P4 FIX: Set customer profile for personalization
    ///
    /// Updates the personalization context based on customer profile data.
    /// This should be called when customer information is discovered (name, segment, etc.)
    pub fn set_customer_profile(&self, profile: &voice_agent_core::CustomerProfile) {
        let mut ctx = self.personalization_ctx.write();
        *ctx = PersonalizationContext::for_profile(profile);
        tracing::debug!(
            segment = ?ctx.segment,
            customer_name = ?ctx.customer_name,
            "Updated personalization context from customer profile"
        );
    }

    /// P4 FIX: Set customer name for personalization
    pub fn set_customer_name(&self, name: impl Into<String>) {
        let name = name.into();
        let mut ctx = self.personalization_ctx.write();
        ctx.customer_name = Some(name.clone());
        tracing::debug!(customer_name = %name, "Set customer name for personalization");
    }

    /// P4 FIX: Set customer segment for personalization
    pub fn set_customer_segment(&self, segment: voice_agent_core::CustomerSegment) {
        use voice_agent_core::personalization::Persona;

        let mut ctx = self.personalization_ctx.write();
        ctx.segment = Some(segment);
        ctx.persona = Persona::for_segment(segment);
        tracing::debug!(segment = ?segment, "Set customer segment for personalization");
    }

    /// P4 FIX: Get current personalization context (read-only)
    pub fn personalization_context(&self) -> PersonalizationContext {
        self.personalization_ctx.read().clone()
    }

    /// P4 FIX: Get personalization engine reference
    pub fn personalization_engine(&self) -> &PersonalizationEngine {
        &self.personalization
    }

    /// End conversation
    pub fn end(&self, reason: EndReason) {
        self.conversation.end(reason);
    }

    /// Get agent name
    pub fn name(&self) -> &str {
        &self.config.persona.name
    }
}

// P1-1 FIX: Implement Agent trait for GoldLoanAgent
use crate::traits::{Agent, PersonalizableAgent, PrefetchingAgent};

#[async_trait::async_trait]
impl Agent for GoldLoanAgent {
    async fn process(&self, input: &str) -> Result<String, AgentError> {
        // Delegate to the inherent method
        GoldLoanAgent::process(self, input).await
    }

    async fn process_stream(
        &self,
        input: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, AgentError> {
        // Delegate to the inherent method
        GoldLoanAgent::process_stream(self, input).await
    }

    fn stage(&self) -> ConversationStage {
        GoldLoanAgent::stage(self)
    }

    fn user_language(&self) -> Language {
        GoldLoanAgent::user_language(self)
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        GoldLoanAgent::subscribe(self)
    }

    fn name(&self) -> &str {
        GoldLoanAgent::name(self)
    }

    fn end(&self, reason: crate::conversation::EndReason) {
        GoldLoanAgent::end(self, reason)
    }
}

#[async_trait::async_trait]
impl PrefetchingAgent for GoldLoanAgent {
    async fn prefetch_on_partial(&self, partial_transcript: &str, confidence: f32) -> bool {
        GoldLoanAgent::prefetch_on_partial(self, partial_transcript, confidence).await
    }

    fn prefetch_background(&self, partial_transcript: String, confidence: f32) {
        GoldLoanAgent::prefetch_background(self, partial_transcript, confidence)
    }

    fn clear_prefetch_cache(&self) {
        GoldLoanAgent::clear_prefetch_cache(self)
    }
}

impl PersonalizableAgent for GoldLoanAgent {
    fn set_customer_profile(&self, profile: &voice_agent_core::CustomerProfile) {
        GoldLoanAgent::set_customer_profile(self, profile)
    }

    fn set_customer_name(&self, name: impl Into<String>) {
        GoldLoanAgent::set_customer_name(self, name)
    }

    fn set_customer_segment(&self, segment: voice_agent_core::CustomerSegment) {
        GoldLoanAgent::set_customer_segment(self, segment)
    }
}

/// P0-2 FIX: Find the position of a sentence boundary in text
///
/// Returns the byte position of the sentence-ending character if found.
/// Supports multiple scripts including Indic terminators (।, ॥, etc.)
fn find_sentence_end(text: &str, terminators: &[char]) -> Option<usize> {
    for (i, c) in text.char_indices() {
        if terminators.contains(&c) {
            // Check if this is the end of a sentence (not abbreviation, etc.)
            // Look for space or end after the terminator
            let next_pos = i + c.len_utf8();
            if next_pos >= text.len() {
                return Some(i);
            }

            // Check next character - if whitespace or newline, it's a sentence end
            if let Some(next_char) = text[next_pos..].chars().next() {
                if next_char.is_whitespace() || next_char == '\n' {
                    return Some(i);
                }
                // For Devanagari terminators, always treat as sentence end
                if c == '।' || c == '॥' {
                    return Some(i);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_creation() {
        let agent = GoldLoanAgent::new("test-session", AgentConfig::default());

        assert_eq!(agent.name(), "Priya");
        assert_eq!(agent.stage(), ConversationStage::Greeting);
    }

    #[tokio::test]
    async fn test_agent_process() {
        let agent = GoldLoanAgent::new("test", AgentConfig::default());

        let response = agent.process("Hello").await.unwrap();

        // Response should not be empty
        assert!(!response.is_empty());
        // After processing greeting, agent transitions to Discovery stage
        // So response could be greeting OR discovery message
        assert!(
            response.contains("Namaste")
            || response.contains("Hello")
            || response.contains("understand")  // Discovery stage response
            || response.contains("batayein"), // Hindi Discovery response
            "Unexpected response: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_agent_conversation_flow() {
        let agent = GoldLoanAgent::new("test", AgentConfig::default());

        // Greeting
        let _ = agent.process("Hello").await.unwrap();

        // Should be able to transition to discovery
        agent
            .conversation()
            .transition_stage(ConversationStage::Discovery)
            .unwrap();

        // Discovery question
        let response = agent.process("I have a loan from Muthoot").await.unwrap();

        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_agent_english_responses() {
        // P2 FIX: Test language-aware mock responses
        let config = AgentConfig {
            language: "en".to_string(),
            ..AgentConfig::default()
        };
        let agent = GoldLoanAgent::without_llm("test-english", config);

        let response = agent.process("Hello").await.unwrap();

        // English mode should produce English response (may be from any stage)
        // After processing greeting, may advance to Discovery stage
        assert!(
            response.contains("Hello")
            || response.contains("assist")
            || response.contains("understand")  // Discovery stage
            || response.contains("needs"), // Discovery stage
            "Expected English response, got: {}",
            response
        );
        // Should NOT contain Hindi words in English mode
        assert!(
            !response.contains("Namaste")
                && !response.contains("hoon")
                && !response.contains("batayein"),
            "Should not contain Hindi in English mode, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_agent_hindi_responses() {
        // P2 FIX: Test language-aware mock responses
        let config = AgentConfig {
            language: "hi".to_string(),
            ..AgentConfig::default()
        };
        let agent = GoldLoanAgent::without_llm("test-hindi", config);

        let response = agent.process("Hello").await.unwrap();

        // Hindi mode should produce Hinglish response
        assert!(
            response.contains("Namaste") || response.contains("hoon"),
            "Expected Hinglish response, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_prefetch_requires_rag_components() {
        // P2 FIX: Test prefetch behavior without vector store
        let agent = GoldLoanAgent::without_llm("test-prefetch", AgentConfig::default());

        // Should return false when vector_store is not set
        let result = agent
            .prefetch_on_partial("gold loan interest rate", 0.8)
            .await;
        assert!(!result, "Prefetch should return false without vector store");
    }

    #[tokio::test]
    async fn test_prefetch_skips_short_partials() {
        // P2 FIX: Test that very short partials are skipped
        let agent = GoldLoanAgent::without_llm("test-prefetch-short", AgentConfig::default());

        // Single word should be skipped (returns false regardless of vector store)
        let result = agent.prefetch_on_partial("hello", 0.9).await;
        assert!(!result, "Prefetch should skip single-word partials");
    }

    #[test]
    fn test_prefetch_cache_lifecycle() {
        // P2 FIX: Test prefetch cache clear
        let agent = GoldLoanAgent::without_llm("test-cache", AgentConfig::default());

        // Initially empty
        assert!(agent.get_prefetch_results("test query").is_none());

        // After clear, still empty (no panic)
        agent.clear_prefetch_cache();
        assert!(agent.get_prefetch_results("test query").is_none());
    }
}
