//! Gold Loan Voice Agent
//!
//! Main agent implementation combining all components.

use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::broadcast;

use voice_agent_llm::{LlmFactory, Message, PromptBuilder, Role, SpeculativeExecutor};
// P1 FIX: Use LanguageModel trait from core for proper abstraction
use voice_agent_core::LanguageModel;
// P8 FIX: Import AgentDomainView for config-driven domain abstraction
use voice_agent_config::domain::AgentDomainView;
// P0-2 FIX: Import ToolDefinition and FinishReason for LLM tool calling
use voice_agent_core::{FinishReason, ToolDefinition};
use voice_agent_tools::{ToolExecutor, ToolRegistry};
// P1 FIX: Import RAG components for retrieval-augmented generation
use voice_agent_rag::{AgenticRetriever, QueryContext, SearchResult, VectorStore};
// P4 FIX: Import personalization engine for dynamic response adaptation
use voice_agent_core::personalization::{PersonalizationContext, PersonalizationEngine};
// P5 FIX: Import translator for Translate-Think-Translate pattern
use voice_agent_core::{Language, Translator};
use voice_agent_text_processing::translation::{
    CandleIndicTrans2Config, CandleIndicTrans2Translator,
};

use crate::conversation::{Conversation, ConversationConfig, ConversationEvent, EndReason};
use crate::dst::DialogueStateTracker;
use crate::lead_scoring::{
    EscalationTrigger, LeadRecommendation, LeadScore, LeadScoringEngine,
};
use crate::memory::{ConversationTurn, TurnRole};
use crate::persuasion::PersuasionEngine;
use crate::stage::ConversationStage;
use crate::AgentError;

// Re-export config types for backwards compatibility
pub use crate::agent_config::{
    AgentConfig, AgentEvent, PersonaTraits, SmallModelConfig, SpeculativeDecodingConfig,
    ToolDefaults, is_small_model,
};

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
    /// Phase 11: Agentic RAG retriever for multi-step retrieval with query rewriting
    /// Replaces simple HybridRetriever with iterative retrieval flow
    agentic_retriever: Option<Arc<AgenticRetriever>>,
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
    // NOTE: Agentic memory is now owned by Conversation to avoid desync issues.
    // Use self.conversation.agentic_memory() to access it.
    /// Phase 5: Dialogue State Tracker for slot-based state management
    dialogue_state: RwLock<DialogueStateTracker>,
    /// Phase 10: Lead Scoring Engine for sales conversion optimization
    /// Tracks signals, calculates MQL/SQL, triggers auto-escalation
    lead_scoring: RwLock<LeadScoringEngine>,
    /// P8 FIX: Domain view for config-driven values (optional for backward compat)
    domain_view: Option<Arc<AgentDomainView>>,
}

impl GoldLoanAgent {
    /// Create a new agent
    pub fn new(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let session_id = session_id.into();

        let conversation = Arc::new(Conversation::new(&session_id, config.conversation.clone()));

        // P15 FIX: Create domain config first, used for tools and persona
        let domain_config = Arc::new(voice_agent_config::MasterDomainConfig::default());
        let agent_view = Arc::new(voice_agent_config::AgentDomainView::new(domain_config.clone()));
        let tools_view = Arc::new(voice_agent_config::ToolsDomainView::new(domain_config));

        // Configure the conversation's agentic memory with persona settings
        // NOTE: We use conversation.agentic_memory() to avoid having two separate memory instances
        conversation.agentic_memory().core.set_persona_name(&config.persona.name);
        // P15 FIX: Use domain config for bank name and role instead of hardcoded values
        conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            agent_view.bank_name(),
            agent_view.agent_role(),
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view));

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

        // Phase 11: Create Agentic RAG retriever if enabled
        // This replaces the simple HybridRetriever with multi-step retrieval
        let agentic_retriever = if config.rag_enabled {
            let retriever = AgenticRetriever::new(config.agentic_rag.clone());
            // Wire LLM backend for query rewriting if LLM is available
            let retriever = if llm.is_some() {
                // Get LLM backend for query rewriting
                if let Ok(backend) = LlmFactory::create_backend(&config.llm_provider) {
                    tracing::info!("AgenticRetriever initialized with LLM for query rewriting");
                    retriever.with_llm(backend)
                } else {
                    tracing::debug!("AgenticRetriever initialized without query rewriting");
                    retriever
                }
            } else {
                retriever
            };
            Some(Arc::new(retriever))
        } else {
            None
        };

        // P1 FIX: Wire LLM to memory for real summarization
        if let Some(ref llm_backend) = llm {
            conversation.memory().set_llm(llm_backend.clone());
            conversation.agentic_memory().set_llm(llm_backend.clone());
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

        // Extract DST config before moving config into struct
        let dst_config = config.dst_config.clone();

        // Phase 10: Initialize lead scoring engine
        let lead_scoring = LeadScoringEngine::new();

        Self {
            config,
            conversation,
            tools,
            llm,
            agentic_retriever,
            vector_store: None,
            event_tx,
            prefetch_cache: RwLock::new(None),
            personalization,
            personalization_ctx: RwLock::new(personalization_ctx),
            translator,
            user_language,
            persuasion,
            speculative,
            dialogue_state: RwLock::new(DialogueStateTracker::with_config(dst_config)),
            lead_scoring: RwLock::new(lead_scoring),
            domain_view: None,
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

        // P15 FIX: Create domain config first, used for tools and persona
        let domain_config = Arc::new(voice_agent_config::MasterDomainConfig::default());
        let agent_view = Arc::new(voice_agent_config::AgentDomainView::new(domain_config.clone()));
        let tools_view = Arc::new(voice_agent_config::ToolsDomainView::new(domain_config));

        // Configure the conversation's agentic memory with persona settings
        // NOTE: We use conversation.agentic_memory() to avoid having two separate memory instances
        conversation.agentic_memory().core.set_persona_name(&config.persona.name);
        // P15 FIX: Use domain config for bank name and role instead of hardcoded values
        conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            agent_view.bank_name(),
            agent_view.agent_role(),
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view));

        // Phase 11: Create Agentic RAG retriever if enabled
        let agentic_retriever = if config.rag_enabled {
            let retriever = AgenticRetriever::new(config.agentic_rag.clone());
            // Wire LLM backend for query rewriting
            let retriever = if let Ok(backend) = LlmFactory::create_backend(&config.llm_provider) {
                tracing::info!("AgenticRetriever initialized with LLM for query rewriting");
                retriever.with_llm(backend)
            } else {
                retriever
            };
            Some(Arc::new(retriever))
        } else {
            None
        };

        // P1 FIX: Wire LLM to memory for real summarization
        conversation.memory().set_llm(llm.clone());
        conversation.agentic_memory().set_llm(llm.clone());

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

        // Phase 10: Initialize lead scoring engine
        let lead_scoring = LeadScoringEngine::new();

        Self {
            config: config.clone(),
            conversation,
            tools,
            llm: Some(llm),
            agentic_retriever,
            vector_store: None,
            event_tx,
            prefetch_cache: RwLock::new(None),
            personalization,
            personalization_ctx: RwLock::new(personalization_ctx),
            translator,
            user_language,
            persuasion,
            speculative,
            dialogue_state: RwLock::new(DialogueStateTracker::with_config(config.dst_config)),
            lead_scoring: RwLock::new(lead_scoring),
            domain_view: None,
        }
    }

    /// Create agent without LLM (uses mock responses)
    pub fn without_llm(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let session_id = session_id.into();

        let conversation = Arc::new(Conversation::new(&session_id, config.conversation.clone()));

        // P15 FIX: Create domain config first, used for tools and persona
        let domain_config = Arc::new(voice_agent_config::MasterDomainConfig::default());
        let agent_view = Arc::new(voice_agent_config::AgentDomainView::new(domain_config.clone()));
        let tools_view = Arc::new(voice_agent_config::ToolsDomainView::new(domain_config));

        // Configure the conversation's agentic memory with persona settings
        // NOTE: We use conversation.agentic_memory() to avoid having two separate memory instances
        conversation.agentic_memory().core.set_persona_name(&config.persona.name);
        // P15 FIX: Use domain config for bank name and role instead of hardcoded values
        conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            agent_view.bank_name(),
            agent_view.agent_role(),
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view));

        // Phase 11: Create Agentic RAG retriever if enabled
        // Without LLM, agentic retriever works but without query rewriting
        let agentic_retriever = if config.rag_enabled {
            Some(Arc::new(AgenticRetriever::new(config.agentic_rag.clone())))
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

        // Phase 10: Initialize lead scoring engine
        let lead_scoring = LeadScoringEngine::new();

        Self {
            config: config.clone(),
            conversation,
            tools,
            llm: None,
            agentic_retriever,
            vector_store: None,
            event_tx,
            prefetch_cache: RwLock::new(None),
            personalization,
            personalization_ctx: RwLock::new(personalization_ctx),
            translator,
            user_language,
            persuasion,
            speculative: None, // P1-2 FIX: No speculative without LLM
            dialogue_state: RwLock::new(DialogueStateTracker::with_config(config.dst_config)),
            lead_scoring: RwLock::new(lead_scoring),
            domain_view: None,
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

    /// P8 FIX: Set domain view for config-driven values
    ///
    /// P13 FIX: Also reinitializes PersuasionEngine to use config-driven responses,
    /// updates persona goal with brand names from config, and wires DST for instructions.
    pub fn with_domain_view(mut self, view: Arc<AgentDomainView>) -> Self {
        // P13 FIX: Reinitialize persuasion engine with config-driven responses
        self.persuasion = PersuasionEngine::from_view(&view);

        // P13 FIX: Update persona goal with brand names from config
        // This overrides the default hardcoded "Kotak Mahindra Bank" in constructors
        let bank_name = view.bank_name();
        let agent_role = view.agent_role();
        self.conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            bank_name,
            agent_role,
            self.config.persona.warmth * 100.0,
            self.config.persona.formality * 100.0,
            self.config.persona.empathy * 100.0
        ));

        // P13 FIX: Wire domain view to DST for config-driven instructions
        self.dialogue_state.write().set_domain_view(view.clone());

        self.domain_view = Some(view);
        self
    }

    /// P8 FIX: Get domain view if available
    pub fn domain_view(&self) -> Option<&Arc<AgentDomainView>> {
        self.domain_view.as_ref()
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

        // Phase 11: Use AgenticRetriever's underlying HybridRetriever for prefetch
        let (agentic_retriever, vector_store) =
            match (&self.agentic_retriever, &self.vector_store) {
                (Some(ar), Some(vs)) => (ar.clone(), vs.clone()),
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

        // Phase 11: Run prefetch using the underlying HybridRetriever from AgenticRetriever
        // This is faster than full agentic retrieval (no query rewriting)
        match agentic_retriever
            .retriever()
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
            }
            Ok(_) => {
                tracing::trace!("RAG prefetch returned no results");
                false
            }
            Err(e) => {
                tracing::warn!("RAG prefetch failed: {}", e);
                false
            }
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

        // Phase 11: Use AgenticRetriever's underlying HybridRetriever for background prefetch
        let (agentic_retriever, vector_store) =
            match (&self.agentic_retriever, &self.vector_store) {
                (Some(ar), Some(vs)) => (ar.clone(), vs.clone()),
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
            // Use underlying HybridRetriever for fast prefetch (no query rewriting)
            match agentic_retriever
                .retriever()
                .prefetch(&partial_transcript, confidence, &vector_store)
                .await
            {
                Ok(results) if !results.is_empty() => {
                    tracing::debug!(count = results.len(), "Background prefetch completed");
                    // Note: Results are not cached in background mode - use prefetch_on_partial for caching
                }
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
        self.conversation.agentic_memory().add_turn(turn);

        // Log memory state after adding turn (P0 FIX: diagnose hallucination)
        let stats = self.conversation.agentic_memory().get_stats();
        tracing::debug!(
            role = "user",
            fifo_tokens = stats.fifo_tokens,
            core_tokens = stats.core_tokens,
            total_turns = self.conversation.agentic_memory().get_recent_turns().len(),
            "Added user turn to agentic memory"
        );

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
                    let _ = self.conversation.agentic_memory().core_memory_append(k, value);
                }
            }
        }

        // Phase 5: Update Dialogue State Tracker with detected intent
        {
            let mut dst = self.dialogue_state.write();
            dst.update(&intent);

            // Update conversation goal based on detected intent
            let turn = dst.history().len();
            dst.update_goal_from_intent(&intent.intent, turn);

            tracing::debug!(
                primary_intent = ?dst.state().primary_intent(),
                filled_slots = ?dst.state().filled_slots(),
                goal_id = %dst.goal_id(),
                pending = ?dst.slots_needing_confirmation(),
                "Dialogue state updated"
            );
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

        // Phase 10: Update lead scoring engine with detected signals
        {
            let mut lead_scoring = self.lead_scoring.write();

            // Update urgency signals from user input (detects "urgent", "jaldi", "abhi", etc.)
            lead_scoring.update_urgency(user_input);

            // Convert intent slots to HashMap<String, String> for lead scoring
            let slot_values: std::collections::HashMap<String, String> = intent
                .slots
                .iter()
                .filter_map(|(k, v)| v.value.as_ref().map(|val| (k.clone(), val.clone())))
                .collect();

            // Update signals based on detected intent
            lead_scoring.update_from_intent(&intent.intent, &slot_values);

            // Update trust level based on engagement (positive signal if providing info)
            if !slot_values.is_empty() {
                lead_scoring.update_trust(true);
            }

            // Check for high-value loan amount that requires escalation
            if let Some(amount_str) = slot_values.get("loan_amount").or(slot_values.get("amount")) {
                if let Ok(amount) = amount_str.replace(",", "").parse::<f64>() {
                    if let Some(_trigger) = lead_scoring.check_high_value_loan(amount) {
                        tracing::info!(
                            amount = amount,
                            "High-value loan detected, escalation may be triggered"
                        );
                        // The trigger will be included in calculate_score() results
                    }
                }
            }

            tracing::debug!(
                engagement_turns = lead_scoring.signals().engagement_turns,
                has_urgency = lead_scoring.signals().has_urgency_signal,
                provided_contact = lead_scoring.signals().provided_contact_info,
                "Lead scoring signals updated"
            );
        }

        // Forward conversation events
        let _ = self
            .event_tx
            .send(AgentEvent::Conversation(ConversationEvent::IntentDetected(
                intent.clone(),
            )));

        // Check for tool calls based on intent
        let mut tool_result = if self.config.tools_enabled {
            self.maybe_call_tool(&intent).await?
        } else {
            None
        };

        // Phase 12: Auto-capture lead when we have contact info (runs independently of main tool)
        // This ensures we capture leads even when serving other goals like balance transfer
        if self.config.tools_enabled {
            let should_capture = {
                let dst = self.dialogue_state.read();
                dst.should_auto_capture_lead()
            };

            if should_capture {
                tracing::info!("Auto-capturing lead with collected contact information");
                // Fire-and-forget lead capture - don't wait for result or add to response
                let lead_result = self.call_tool_by_name("capture_lead", &intent).await;
                if let Ok(Some(_)) = lead_result {
                    tracing::info!("Lead captured successfully");
                } else {
                    tracing::warn!("Auto lead capture failed or returned empty");
                }
            }
        }

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
        self.conversation.agentic_memory().add_turn(assistant_turn);

        // Log memory state after adding assistant turn (P0 FIX: diagnose hallucination)
        let stats = self.conversation.agentic_memory().get_stats();
        tracing::debug!(
            role = "assistant",
            fifo_tokens = stats.fifo_tokens,
            core_tokens = stats.core_tokens,
            total_turns = self.conversation.agentic_memory().get_recent_turns().len(),
            "Added assistant turn to agentic memory"
        );

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
        if self.conversation.agentic_memory().needs_compaction() {
            let stats = self.conversation.agentic_memory().get_stats();
            tracing::debug!(
                core_tokens = stats.core_tokens,
                fifo_tokens = stats.fifo_tokens,
                archival_count = stats.archival_count,
                "Agentic memory high watermark exceeded"
            );
        }

        // Phase 10: Calculate lead score and emit events
        let lead_score = {
            let mut lead_scoring = self.lead_scoring.write();
            lead_scoring.calculate_score()
        };

        // Emit lead score update event
        let _ = self.event_tx.send(AgentEvent::LeadScoreUpdated {
            score: lead_score.total,
            qualification: format!("{:?}", lead_score.qualification),
            classification: format!("{:?}", lead_score.classification),
            conversion_probability: lead_score.conversion_probability,
        });

        tracing::info!(
            score = lead_score.total,
            qualification = ?lead_score.qualification,
            classification = ?lead_score.classification,
            conversion_prob = lead_score.conversion_probability,
            recommendation = ?lead_score.recommendation,
            "Lead score calculated"
        );

        // Check for escalation triggers and emit events
        for trigger in &lead_score.escalation_triggers {
            let trigger_str = match trigger {
                EscalationTrigger::ExcessiveObjections { count, threshold } => {
                    format!("ExcessiveObjections: {} objections (threshold: {})", count, threshold)
                }
                EscalationTrigger::ConversationStalled { turns, threshold } => {
                    format!("ConversationStalled: {} turns (threshold: {})", turns, threshold)
                }
                EscalationTrigger::HighValueLoan { amount, threshold } => {
                    format!("HighValueLoan: ₹{:.0} (threshold: ₹{:.0})", amount, threshold)
                }
                EscalationTrigger::CustomerFrustration => "CustomerFrustration".to_string(),
                EscalationTrigger::CustomerRequested => "CustomerRequested".to_string(),
                EscalationTrigger::ComplexQuery => "ComplexQuery".to_string(),
                EscalationTrigger::ComplianceSensitive => "ComplianceSensitive".to_string(),
            };

            let recommendation_str = match &lead_score.recommendation {
                LeadRecommendation::ContinueConversation => "ContinueConversation".to_string(),
                LeadRecommendation::PushForAppointment => "PushForAppointment".to_string(),
                LeadRecommendation::OfferCallback => "OfferCallback".to_string(),
                LeadRecommendation::EscalateNow { reason } => format!("EscalateNow: {}", reason),
                LeadRecommendation::SendFollowUp => "SendFollowUp".to_string(),
                LeadRecommendation::LowPriority => "LowPriority".to_string(),
            };

            tracing::warn!(
                trigger = %trigger_str,
                recommendation = %recommendation_str,
                "Escalation trigger detected"
            );

            let _ = self.event_tx.send(AgentEvent::EscalationTriggered {
                trigger: trigger_str,
                recommendation: recommendation_str,
            });
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

        // Add memory context with query-based archival retrieval
        // Phase 10: Use get_context_for_query to include relevant archival memories
        let stage = self.conversation.stage();
        let context_budget = stage.context_budget_tokens();
        let context = self.conversation.get_context_for_query(english_input, context_budget);

        // Log memory context for debugging (P0 FIX: diagnose hallucination issues)
        let memory_stats = self.conversation.agentic_memory().get_stats();
        let recent_turns = self.conversation.agentic_memory().get_recent_turns();
        tracing::debug!(
            context_len = context.len(),
            context_budget = context_budget,
            core_tokens = memory_stats.core_tokens,
            fifo_tokens = memory_stats.fifo_tokens,
            archival_count = memory_stats.archival_count,
            recent_turns_count = recent_turns.len(),
            stage = ?stage,
            "Memory context for LLM"
        );

        // Log the actual context being sent (truncated for readability)
        if tracing::enabled!(tracing::Level::TRACE) {
            let context_preview = if context.len() > 500 {
                format!("{}...[truncated {} chars]", &context[..500], context.len() - 500)
            } else {
                context.clone()
            };
            tracing::trace!(context = %context_preview, "Full memory context");
        }

        if !context.is_empty() {
            builder = builder.with_context(&context);
        }

        // Phase 5 + Phase 12: Add DST state context with goal tracking for guided response generation
        {
            let dst = self.dialogue_state.read();
            let dst_context = dst.state_context();

            // Add collected customer information prominently
            // This ensures the LLM can recall customer details like name, phone, loan amount
            if !dst_context.is_empty() && dst_context != "No information collected yet." {
                let dst_section = format!(
                    "## IMPORTANT: Customer Details (Use these for recall)\n{}\n\n## Slots Needing Confirmation\n{}",
                    dst_context,
                    if dst.slots_needing_confirmation().is_empty() {
                        "None".to_string()
                    } else {
                        dst.slots_needing_confirmation().join(", ")
                    }
                );
                builder = builder.with_context(&dst_section);
            }

            // Also add core memory facts for additional recall capability
            let human_block = self.conversation.agentic_memory().core.human_snapshot();
            if !human_block.facts.is_empty() {
                let facts_str = human_block.facts
                    .iter()
                    .map(|(k, entry)| format!("- {}: {}", k, entry.value))
                    .collect::<Vec<_>>()
                    .join("\n");
                builder = builder.with_context(&format!("## Customer Facts from Memory\n{}", facts_str));
            }

            // Phase 12: Add goal ID to context
            let goal_id = dst.goal_id();
            builder = builder.with_context(&format!("Current Goal: {}", goal_id));

            // Log goal tracking for debugging
            tracing::debug!(
                goal = %goal_id,
                "Goal context added to prompt"
            );
        }

        // Phase 11: Add RAG context using Agentic RAG with multi-step retrieval
        if self.config.rag_enabled {
            let stage = self.conversation.stage();
            let rag_fraction = stage.rag_context_fraction();

            if rag_fraction > 0.0 {
                if let (Some(agentic_retriever), Some(vector_store)) =
                    (&self.agentic_retriever, &self.vector_store)
                {
                    // Check for prefetched results first
                    let results = if let Some(prefetched) = self.get_prefetch_results(english_input)
                    {
                        self.clear_prefetch_cache();
                        prefetched
                    } else {
                        // Build query context for agentic retrieval
                        let human_block = self.conversation.agentic_memory().core.human_snapshot();
                        let query_context = QueryContext {
                            // Use conversation context as summary for query rewriting
                            summary: self.conversation.get_context(),
                            stage: Some(stage.display_name().to_string()),
                            entities: human_block
                                .facts
                                .iter()
                                .map(|(k, entry)| (k.clone(), entry.value.clone()))
                                .collect(),
                        };

                        // Use AgenticRetriever for multi-step retrieval with query rewriting
                        match agentic_retriever
                            .search(english_input, vector_store, Some(&query_context))
                            .await
                        {
                            Ok(agentic_result) => {
                                if agentic_result.query_rewritten {
                                    tracing::debug!(
                                        original = %english_input,
                                        rewritten = %agentic_result.final_query,
                                        iterations = agentic_result.iterations,
                                        sufficiency = agentic_result.sufficiency_score,
                                        "Agentic RAG rewrote query"
                                    );
                                }
                                agentic_result.results
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Agentic RAG search failed");
                                vec![]
                            }
                        }
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

    /// Call a tool by name using DST state for arguments (Phase 12 - proactive tool triggering)
    async fn call_tool_by_name(
        &self,
        tool_name: &str,
        intent: &crate::intent::DetectedIntent,
    ) -> Result<Option<String>, AgentError> {
        let _ = self.event_tx.send(AgentEvent::ToolCall {
            name: tool_name.to_string(),
        });

        // Build arguments from DST state (more complete than just current intent slots)
        let mut args = serde_json::Map::new();

        // First, add slots from the current intent
        for (key, slot) in &intent.slots {
            if let Some(ref value) = slot.value {
                args.insert(key.clone(), serde_json::json!(value));
            }
        }

        // Then, enrich with DST state values that may have been collected over multiple turns
        {
            let dst = self.dialogue_state.read();
            let state = dst.state();

            // Map DST slot names to tool argument names
            if let Some(val) = state.customer_name() {
                args.entry("customer_name".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.phone_number() {
                args.entry("phone_number".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.location() {
                args.entry("city".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.gold_weight_grams() {
                args.entry("gold_weight".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.gold_purity() {
                args.entry("gold_purity".to_string()).or_insert(serde_json::json!(val.to_string()));
            }
            if let Some(val) = state.loan_amount() {
                args.entry("loan_amount".to_string()).or_insert(serde_json::json!(val));
                args.entry("current_loan_amount".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.current_lender() {
                args.entry("current_lender".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.current_interest_rate() {
                args.entry("current_interest_rate".to_string()).or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.loan_tenure() {
                args.entry("remaining_tenure_months".to_string()).or_insert(serde_json::json!(val));
            }
        }

        // Apply defaults based on tool type
        let defaults = &self.config.tool_defaults;

        if tool_name == "check_eligibility" && !args.contains_key("gold_purity") {
            args.insert(
                "gold_purity".to_string(),
                serde_json::json!(&defaults.default_gold_purity),
            );
        }

        if tool_name == "calculate_savings" {
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

        if tool_name == "find_branches" && !args.contains_key("city") {
            args.insert(
                "city".to_string(),
                serde_json::json!(&defaults.default_city),
            );
        }

        if tool_name == "capture_lead" {
            // Default interest level to High for proactive capture
            if !args.contains_key("interest_level") {
                args.insert("interest_level".to_string(), serde_json::json!("High"));
            }
        }

        tracing::debug!(
            tool = tool_name,
            args = ?args,
            "Calling tool proactively with DST state"
        );

        let result = self
            .tools
            .execute(tool_name, serde_json::Value::Object(args))
            .await;

        let success = result.is_ok();
        let _ = self.event_tx.send(AgentEvent::ToolResult {
            name: tool_name.to_string(),
            success,
        });

        match result {
            Ok(output) => {
                let text = output
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        voice_agent_tools::mcp::ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(Some(text))
            },
            Err(e) => {
                tracing::warn!("Proactive tool error: {}", e);
                Ok(None)
            },
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

        // Add context from memory with query-based archival retrieval
        // Phase 10: Use get_context_for_query to include relevant archival memories
        let stage = self.conversation.stage();
        let context_budget = stage.context_budget_tokens();
        let context = self.conversation.get_context_for_query(user_input, context_budget);
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
                // Phase 11: Use AgenticRetriever for multi-step retrieval
                if let (Some(agentic_retriever), Some(vector_store)) =
                    (&self.agentic_retriever, &self.vector_store)
                {
                    // First, try to use prefetched results
                    let results = if let Some(prefetched) = self.get_prefetch_results(user_input) {
                        tracing::debug!("Using {} prefetched RAG results", prefetched.len());
                        // Clear cache after use
                        self.clear_prefetch_cache();
                        prefetched
                    } else {
                        // Build query context for agentic retrieval
                        let human_block = self.conversation.agentic_memory().core.human_snapshot();
                        let query_context = QueryContext {
                            // Use conversation context as summary for query rewriting
                            summary: self.conversation.get_context(),
                            stage: Some(stage.display_name().to_string()),
                            entities: human_block
                                .facts
                                .iter()
                                .map(|(k, entry)| (k.clone(), entry.value.clone()))
                                .collect(),
                        };

                        // Use AgenticRetriever for multi-step retrieval
                        match agentic_retriever
                            .search(user_input, vector_store, Some(&query_context))
                            .await
                        {
                            Ok(agentic_result) => {
                                if agentic_result.query_rewritten {
                                    tracing::debug!(
                                        original = %user_input,
                                        rewritten = %agentic_result.final_query,
                                        iterations = agentic_result.iterations,
                                        "Agentic RAG rewrote query (streaming)"
                                    );
                                }
                                agentic_result.results
                            }
                            Err(e) => {
                                tracing::warn!("RAG search failed, continuing without: {}", e);
                                Vec::new()
                            }
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
                // P13 FIX: Get rates from config if available
                let (our_rate, nbfc_rate) = if let Some(view) = &self.domain_view {
                    (view.our_rate_for_amount(500_000.0), view.get_competitor_rate("muthoot").unwrap_or(18.0))
                } else {
                    (10.5, 18.0)
                };
                if is_english {
                    format!("At Kotak, we offer just {:.1}% interest rate, which is much lower than the {:.0}-{:.0}% NBFCs charge. Plus, you get the security of an RBI regulated bank. Would you be interested?", our_rate, nbfc_rate, nbfc_rate + 2.0)
                } else {
                    format!("Dekhiye, Kotak mein aapko sirf {:.1}% rate milega, jo NBFC ke {:.0}-{:.0}% se bahut kam hai. Aur hamare yahan RBI regulated bank ki security bhi hai. Aap interested hain?", our_rate, nbfc_rate, nbfc_rate + 2.0)
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

    /// Phase 10: Get current lead score
    ///
    /// Calculates and returns the current lead score based on collected signals.
    /// This includes qualification level (Cold/Warm/Hot/Qualified), MQL/SQL classification,
    /// conversion probability, and any active escalation triggers.
    pub fn get_lead_score(&self) -> LeadScore {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.calculate_score()
    }

    /// Phase 10: Get lead signals (read-only)
    ///
    /// Returns a snapshot of the current lead signals for inspection.
    /// Useful for debugging and monitoring lead qualification progress.
    pub fn get_lead_signals(&self) -> crate::lead_scoring::LeadSignals {
        self.lead_scoring.read().signals().clone()
    }

    /// Phase 10: Check if escalation is needed
    ///
    /// Returns true if any escalation triggers are active.
    /// Use get_lead_score() to get the specific triggers.
    pub fn needs_escalation(&self) -> bool {
        let score = self.get_lead_score();
        !score.escalation_triggers.is_empty()
    }

    /// Phase 10: Get lead recommendation
    ///
    /// Returns the recommended next action based on lead score.
    pub fn get_lead_recommendation(&self) -> LeadRecommendation {
        self.get_lead_score().recommendation
    }

    /// Phase 10: Mark conversation as stalled (no meaningful progress)
    ///
    /// Call this when the agent detects the conversation is not progressing.
    /// After max_stalled_turns (default: 5), an escalation trigger is raised.
    pub fn mark_conversation_stalled(&self) {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.mark_stalled();
    }

    /// Phase 10: Reset stall counter (progress made)
    ///
    /// Call this when meaningful progress is detected in the conversation.
    pub fn reset_stall_counter(&self) {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.reset_stall();
    }

    /// Phase 10: Reset lead scoring engine
    ///
    /// Resets all signals and score history. Use when starting a new conversation.
    pub fn reset_lead_scoring(&self) {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.reset();
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

    // =========================================================================
    // Small Model Detection Tests
    // =========================================================================

    #[test]
    fn test_small_model_detection_qwen() {
        // Qwen small models
        assert!(is_small_model("qwen2.5:1.5b"));
        assert!(is_small_model("qwen2.5:1.5b-instruct"));
        assert!(is_small_model("qwen2.5:1.5b-instruct-q4_K_M"));
        assert!(is_small_model("qwen2.5:0.5b"));
        assert!(is_small_model("qwen2.5:3b"));

        // Qwen large models
        assert!(!is_small_model("qwen2.5:7b"));
        assert!(!is_small_model("qwen2.5:14b"));
        assert!(!is_small_model("qwen2.5:72b"));
    }

    #[test]
    fn test_small_model_detection_llama() {
        // Llama small models
        assert!(is_small_model("llama3.2:1b"));
        assert!(is_small_model("llama3.2:3b"));

        // Llama large models
        assert!(!is_small_model("llama3.2:7b"));
        assert!(!is_small_model("llama3:8b"));
        assert!(!is_small_model("llama3:70b"));
    }

    #[test]
    fn test_small_model_detection_phi() {
        // Phi small models
        assert!(is_small_model("phi-2"));
        assert!(is_small_model("phi-3-mini"));

        // Other small model patterns
        assert!(is_small_model("some-model-mini"));
        assert!(is_small_model("tiny-llama"));
    }

    #[test]
    fn test_small_model_detection_large() {
        // Large models should not be detected as small
        assert!(!is_small_model("claude-3-opus"));
        assert!(!is_small_model("gpt-4"));
        assert!(!is_small_model("mistral-7b"));
        assert!(!is_small_model("mixtral-8x7b"));
    }

    #[test]
    fn test_agent_config_default_is_small() {
        // Default config uses qwen2.5:1.5b which is small
        let config = AgentConfig::default();
        assert!(config.is_small_model());
        assert_eq!(config.context_window_tokens, 2500);
        assert!(config.small_model.use_extractive_compression);
        assert!(config.small_model.disable_llm_query_rewriting);
    }

    #[test]
    fn test_agent_config_with_large_model() {
        let config = AgentConfig::with_model("llama3:70b");
        assert!(!config.is_small_model());
        assert_eq!(config.context_window_tokens, 4096);
    }

    #[test]
    fn test_agent_config_optimize_for_small() {
        let config = AgentConfig {
            context_window_tokens: 4096,
            small_model: SmallModelConfig::disabled(),
            ..Default::default()
        };

        let optimized = config.optimize_for_small_model();
        assert!(optimized.is_small_model());
        assert_eq!(optimized.context_window_tokens, 2500);
        // Should also configure agentic RAG for small model
        assert!(!optimized.agentic_rag.llm_query_rewriting);
        assert!(optimized.agentic_rag.use_rule_based_expansion);
    }

    #[test]
    fn test_agent_config_agentic_rag_for_small_model() {
        // Small model should have agentic RAG configured for single-shot
        let config = AgentConfig::with_model("qwen2.5:1.5b");
        assert!(config.is_small_model());
        assert!(!config.agentic_rag.llm_query_rewriting);
        assert!(!config.agentic_rag.llm_sufficiency_check);
        assert_eq!(config.agentic_rag.max_iterations, 0);
        assert!(config.agentic_rag.use_rule_based_expansion);
    }

    #[test]
    fn test_agent_config_agentic_rag_for_large_model() {
        // Large model should have full agentic RAG
        let config = AgentConfig::with_model("llama3:70b");
        assert!(!config.is_small_model());
        assert!(config.agentic_rag.llm_query_rewriting);
        assert!(config.agentic_rag.llm_sufficiency_check);
        assert_eq!(config.agentic_rag.max_iterations, 3);
    }

    #[test]
    fn test_small_model_config_values() {
        let config = SmallModelConfig::enabled();
        assert!(config.enabled);
        assert_eq!(config.context_window_tokens, 2500);
        assert_eq!(config.high_watermark_tokens, 2000);
        assert_eq!(config.low_watermark_tokens, 1500);
        assert!(config.use_extractive_compression);
        assert!(config.disable_llm_query_rewriting);
    }
}
