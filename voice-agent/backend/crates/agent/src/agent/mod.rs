//! Domain Voice Agent
//!
//! Main agent implementation combining all components.
//!
//! ## Phase 2 (Domain-Agnosticism): DomainAgent
//!
//! The agent is now named `DomainAgent` to reflect its domain-agnostic design.
//! It uses traits internally (ConversationContext, DialogueStateTracking,
//! PersuasionStrategy) for flexibility and testability.
//!
//! ## Phase 3 (Code Organization): Module Structure
//!
//! The agent implementation is split into focused submodules:
//! - `processing`: Core process() and process_stream() methods
//! - `rag`: RAG and prefetch methods
//! - `tools`: Tool calling logic
//! - `response`: Response generation

// Submodules for focused functionality
mod processing;
mod rag;
mod response;
mod tools;

use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::broadcast;

use voice_agent_llm::{LlmFactory, SpeculativeExecutor};
// P1 FIX: Use LanguageModel trait from core for proper abstraction
use voice_agent_core::LanguageModel;
// P8 FIX: Import AgentDomainView for config-driven domain abstraction
use voice_agent_config::domain::AgentDomainView;
use voice_agent_tools::ToolRegistry;
// P1 FIX: Import RAG components for retrieval-augmented generation
use voice_agent_rag::{AgenticRetriever, SearchResult, VectorStore};
// P4 FIX: Import personalization engine for dynamic response adaptation
use voice_agent_core::personalization::{PersonalizationContext, PersonalizationEngine};
// P5 FIX: Import translator for Translate-Think-Translate pattern
use voice_agent_core::{Language, Translator};
use voice_agent_text_processing::translation::{
    CandleIndicTrans2Config, CandleIndicTrans2Translator,
};

use crate::conversation::{Conversation, ConversationContext, EndReason};
use crate::dst::DialogueStateTracker;
use crate::lead_scoring::{LeadRecommendation, LeadScore, LeadScoringEngine};
use crate::persuasion::{PersuasionEngine, PersuasionStrategy};
use crate::stage::ConversationStage;
use crate::AgentError;

// Re-export config types for backwards compatibility
pub use crate::agent_config::{
    is_small_model, AgentConfig, AgentEvent, PersonaTraits, SmallModelConfig,
    SpeculativeDecodingConfig, ToolDefaults,
};

/// Prefetch cache entry
#[derive(Debug, Clone)]
pub(crate) struct PrefetchEntry {
    /// Query that triggered prefetch
    query: String,
    /// Prefetched results
    results: Vec<SearchResult>,
    /// When prefetch was triggered
    timestamp: std::time::Instant,
}

/// Domain Voice Agent
///
/// A domain-agnostic voice agent that uses configuration and traits
/// to handle any business domain. The agent internally uses:
/// - `ConversationContext` trait for conversation management
/// - `DialogueStateTracking` trait for slot-based state
/// - `PersuasionStrategy` trait for objection handling
///
/// For backwards compatibility, `GoldLoanAgent` is available as a type alias.
pub struct DomainAgent {
    pub(crate) config: AgentConfig,
    /// Phase 2: Uses ConversationContext trait for domain-agnostic conversation management
    pub(crate) conversation: Arc<dyn ConversationContext>,
    pub(crate) tools: Arc<ToolRegistry>,
    /// P1 FIX: Now uses LanguageModel trait instead of LlmBackend for proper abstraction
    pub(crate) llm: Option<Arc<dyn LanguageModel>>,
    /// Phase 11: Agentic RAG retriever for multi-step retrieval with query rewriting
    /// Replaces simple HybridRetriever with iterative retrieval flow
    pub(crate) agentic_retriever: Option<Arc<AgenticRetriever>>,
    /// P1 FIX: Vector store for RAG search (optional, can be injected)
    pub(crate) vector_store: Option<Arc<VectorStore>>,
    pub(crate) event_tx: broadcast::Sender<AgentEvent>,
    /// P2 FIX: Prefetch cache for VAD → RAG prefetch optimization
    pub(crate) prefetch_cache: RwLock<Option<PrefetchEntry>>,
    /// P4 FIX: Personalization engine for dynamic response adaptation
    pub(crate) personalization: PersonalizationEngine,
    /// P4 FIX: Personalization context (updated each turn)
    pub(crate) personalization_ctx: RwLock<PersonalizationContext>,
    /// P5 FIX: Translator for Translate-Think-Translate pattern
    /// Translates user input to English before LLM, then translates response back
    pub(crate) translator: Option<Arc<dyn Translator>>,
    /// P5 FIX: User's language for translation
    pub(crate) user_language: Language,
    /// Phase 2: Uses PersuasionStrategy trait for domain-agnostic objection handling
    pub(crate) persuasion: Arc<dyn PersuasionStrategy>,
    /// P1-2 FIX: Speculative executor for low-latency generation
    /// Uses SLM for fast drafts, LLM for verification/improvement
    pub(crate) speculative: Option<Arc<SpeculativeExecutor>>,
    // NOTE: Agentic memory is now owned by Conversation to avoid desync issues.
    // Use self.conversation.agentic_memory() to access it.
    /// Phase 5: Dialogue State Tracker for slot-based state management
    pub(crate) dialogue_state: RwLock<DialogueStateTracker>,
    /// Phase 10: Lead Scoring Engine for sales conversion optimization
    /// Tracks signals, calculates MQL/SQL, triggers auto-escalation
    pub(crate) lead_scoring: RwLock<LeadScoringEngine>,
    /// P8 FIX: Domain view for config-driven values (optional for backward compat)
    pub(crate) domain_view: Option<Arc<AgentDomainView>>,
}

impl DomainAgent {
    /// Create a new agent
    pub fn new(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        let session_id = session_id.into();

        let conversation = Arc::new(Conversation::new(&session_id, config.conversation.clone()));

        // P15 FIX: Create domain config first, used for tools and persona
        let domain_config = Arc::new(voice_agent_config::MasterDomainConfig::default());
        let agent_view =
            Arc::new(voice_agent_config::AgentDomainView::new(domain_config.clone()));
        let tools_view = Arc::new(voice_agent_config::ToolsDomainView::new(domain_config));

        // Configure the conversation's agentic memory with persona settings
        // NOTE: We use conversation.agentic_memory() to avoid having two separate memory instances
        conversation
            .agentic_memory()
            .core
            .set_persona_name(&config.persona.name);
        // P15 FIX: Use domain config for bank name and role instead of hardcoded values
        conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            agent_view.bank_name(),
            agent_view.agent_role(),
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(
            tools_view,
        ));

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
            }
            Err(e) => {
                tracing::warn!(
                    provider = ?config.llm_provider.provider,
                    error = %e,
                    "Failed to create LLM backend, falling back to None"
                );
                None
            }
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
        let user_language =
            Language::from_str_loose(&config.language).unwrap_or(Language::Hindi);

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
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to create translator, responses will be in English"
                    );
                    None
                }
            }
        } else {
            tracing::debug!("English language selected, translator not needed");
            None
        };

        // P0 FIX: Initialize persuasion engine for objection handling
        let persuasion: Arc<dyn PersuasionStrategy> = Arc::new(PersuasionEngine::new());

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
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to create speculative executor, falling back to direct LLM"
                    );
                    None
                }
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
        let agent_view =
            Arc::new(voice_agent_config::AgentDomainView::new(domain_config.clone()));
        let tools_view = Arc::new(voice_agent_config::ToolsDomainView::new(domain_config));

        // Configure the conversation's agentic memory with persona settings
        conversation
            .agentic_memory()
            .core
            .set_persona_name(&config.persona.name);
        conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            agent_view.bank_name(),
            agent_view.agent_role(),
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(
            tools_view,
        ));

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
        let user_language =
            Language::from_str_loose(&config.language).unwrap_or(Language::Hindi);

        let translator: Option<Arc<dyn Translator>> = if user_language != Language::English {
            Self::create_default_translator()
                .map(|t| Arc::new(t) as Arc<dyn Translator>)
                .ok()
        } else {
            None
        };

        // P0 FIX: Initialize persuasion engine for objection handling
        let persuasion: Arc<dyn PersuasionStrategy> = Arc::new(PersuasionEngine::new());

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
        let agent_view =
            Arc::new(voice_agent_config::AgentDomainView::new(domain_config.clone()));
        let tools_view = Arc::new(voice_agent_config::ToolsDomainView::new(domain_config));

        // Configure the conversation's agentic memory with persona settings
        conversation
            .agentic_memory()
            .core
            .set_persona_name(&config.persona.name);
        conversation.agentic_memory().core.add_persona_goal(&format!(
            "Represent {} as a {} with warmth: {:.0}%, formality: {:.0}%, empathy: {:.0}%",
            agent_view.bank_name(),
            agent_view.agent_role(),
            config.persona.warmth * 100.0,
            config.persona.formality * 100.0,
            config.persona.empathy * 100.0
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(
            tools_view,
        ));

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
        let user_language =
            Language::from_str_loose(&config.language).unwrap_or(Language::Hindi);

        let translator: Option<Arc<dyn Translator>> = if user_language != Language::English {
            Self::create_default_translator()
                .map(|t| Arc::new(t) as Arc<dyn Translator>)
                .ok()
        } else {
            None
        };

        // P0 FIX: Initialize persuasion engine for objection handling
        let persuasion: Arc<dyn PersuasionStrategy> = Arc::new(PersuasionEngine::new());

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
    pub fn with_tools(mut self, tools: Arc<ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }

    /// P5 FIX: Create default translator using Candle-based IndicTrans2
    fn create_default_translator() -> voice_agent_core::Result<CandleIndicTrans2Translator> {
        use std::path::PathBuf;

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
    pub fn with_domain_view(mut self, view: Arc<AgentDomainView>) -> Self {
        // P13 FIX: Reinitialize persuasion engine with config-driven responses
        self.persuasion = Arc::new(PersuasionEngine::from_view(&view));

        // P13 FIX: Update persona goal with brand names from config
        let bank_name = view.bank_name();
        let agent_role = view.agent_role();
        self.conversation
            .agentic_memory()
            .core
            .add_persona_goal(&format!(
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

    /// Get current stage
    pub fn stage(&self) -> ConversationStage {
        self.conversation.stage()
    }

    /// Get conversation context reference
    pub fn conversation(&self) -> &Arc<dyn ConversationContext> {
        &self.conversation
    }

    /// P1 FIX: Get agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// P4 FIX: Set customer profile for personalization
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
    pub fn get_lead_score(&self) -> LeadScore {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.calculate_score()
    }

    /// Phase 10: Get lead signals (read-only)
    pub fn get_lead_signals(&self) -> crate::lead_scoring::LeadSignals {
        self.lead_scoring.read().signals().clone()
    }

    /// Phase 10: Check if escalation is needed
    pub fn needs_escalation(&self) -> bool {
        let score = self.get_lead_score();
        !score.escalation_triggers.is_empty()
    }

    /// Phase 10: Get lead recommendation
    pub fn get_lead_recommendation(&self) -> LeadRecommendation {
        self.get_lead_score().recommendation
    }

    /// Phase 10: Mark conversation as stalled
    pub fn mark_conversation_stalled(&self) {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.mark_stalled();
    }

    /// Phase 10: Reset stall counter
    pub fn reset_stall_counter(&self) {
        let mut lead_scoring = self.lead_scoring.write();
        lead_scoring.reset_stall();
    }

    /// Phase 10: Reset lead scoring engine
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

// Backwards compatibility alias
pub type GoldLoanAgent = DomainAgent;

// P1-1 FIX: Implement Agent trait for DomainAgent
use crate::traits::{Agent, PersonalizableAgent, PrefetchingAgent};

#[async_trait::async_trait]
impl Agent for DomainAgent {
    async fn process(&self, input: &str) -> Result<String, AgentError> {
        DomainAgent::process(self, input).await
    }

    async fn process_stream(
        &self,
        input: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, AgentError> {
        DomainAgent::process_stream(self, input).await
    }

    fn stage(&self) -> ConversationStage {
        DomainAgent::stage(self)
    }

    fn user_language(&self) -> Language {
        DomainAgent::user_language(self)
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        DomainAgent::subscribe(self)
    }

    fn name(&self) -> &str {
        DomainAgent::name(self)
    }

    fn end(&self, reason: crate::conversation::EndReason) {
        DomainAgent::end(self, reason)
    }
}

#[async_trait::async_trait]
impl PrefetchingAgent for DomainAgent {
    async fn prefetch_on_partial(&self, partial_transcript: &str, confidence: f32) -> bool {
        DomainAgent::prefetch_on_partial(self, partial_transcript, confidence).await
    }

    fn prefetch_background(&self, partial_transcript: String, confidence: f32) {
        DomainAgent::prefetch_background(self, partial_transcript, confidence)
    }

    fn clear_prefetch_cache(&self) {
        DomainAgent::clear_prefetch_cache(self)
    }
}

impl PersonalizableAgent for DomainAgent {
    fn set_customer_profile(&self, profile: &voice_agent_core::CustomerProfile) {
        DomainAgent::set_customer_profile(self, profile)
    }

    fn set_customer_name(&self, name: impl Into<String>) {
        DomainAgent::set_customer_name(self, name)
    }

    fn set_customer_segment(&self, segment: voice_agent_core::CustomerSegment) {
        DomainAgent::set_customer_segment(self, segment)
    }
}

/// P0-2 FIX: Find the position of a sentence boundary in text
fn find_sentence_end(text: &str, terminators: &[char]) -> Option<usize> {
    for (i, c) in text.char_indices() {
        if terminators.contains(&c) {
            let next_pos = i + c.len_utf8();
            if next_pos >= text.len() {
                return Some(i);
            }

            if let Some(next_char) = text[next_pos..].chars().next() {
                if next_char.is_whitespace() || next_char == '\n' {
                    return Some(i);
                }
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
        let agent = DomainAgent::new("test-session", AgentConfig::default());

        assert_eq!(agent.name(), "Priya");
        assert_eq!(agent.stage(), ConversationStage::Greeting);
    }

    #[tokio::test]
    async fn test_agent_process() {
        let agent = DomainAgent::new("test", AgentConfig::default());

        let response = agent.process("Hello").await.unwrap();

        assert!(!response.is_empty());
        assert!(
            response.contains("Namaste")
                || response.contains("Hello")
                || response.contains("understand")
                || response.contains("batayein"),
            "Unexpected response: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_agent_conversation_flow() {
        let agent = DomainAgent::new("test", AgentConfig::default());

        let _ = agent.process("Hello").await.unwrap();

        agent
            .conversation()
            .transition_stage(ConversationStage::Discovery)
            .unwrap();

        let response = agent.process("I have a loan from Muthoot").await.unwrap();

        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_agent_english_responses() {
        let config = AgentConfig {
            language: "en".to_string(),
            ..AgentConfig::default()
        };
        let agent = DomainAgent::without_llm("test-english", config);

        let response = agent.process("Hello").await.unwrap();

        assert!(
            response.contains("Hello")
                || response.contains("assist")
                || response.contains("understand")
                || response.contains("needs"),
            "Expected English response, got: {}",
            response
        );
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
        let config = AgentConfig {
            language: "hi".to_string(),
            ..AgentConfig::default()
        };
        let agent = DomainAgent::without_llm("test-hindi", config);

        let response = agent.process("Hello").await.unwrap();

        assert!(
            response.contains("Namaste") || response.contains("hoon"),
            "Expected Hinglish response, got: {}",
            response
        );
    }

    #[tokio::test]
    async fn test_prefetch_requires_rag_components() {
        let agent = DomainAgent::without_llm("test-prefetch", AgentConfig::default());

        let result = agent
            .prefetch_on_partial("gold loan interest rate", 0.8)
            .await;
        assert!(!result, "Prefetch should return false without vector store");
    }

    #[tokio::test]
    async fn test_prefetch_skips_short_partials() {
        let agent = DomainAgent::without_llm("test-prefetch-short", AgentConfig::default());

        let result = agent.prefetch_on_partial("hello", 0.9).await;
        assert!(!result, "Prefetch should skip single-word partials");
    }

    #[test]
    fn test_prefetch_cache_lifecycle() {
        let agent = DomainAgent::without_llm("test-cache", AgentConfig::default());

        assert!(agent.get_prefetch_results("test query").is_none());

        agent.clear_prefetch_cache();
        assert!(agent.get_prefetch_results("test query").is_none());
    }

    #[test]
    fn test_small_model_detection_qwen() {
        assert!(is_small_model("qwen2.5:1.5b"));
        assert!(is_small_model("qwen2.5:1.5b-instruct"));
        assert!(is_small_model("qwen2.5:1.5b-instruct-q4_K_M"));
        assert!(is_small_model("qwen2.5:0.5b"));
        assert!(is_small_model("qwen2.5:3b"));

        assert!(!is_small_model("qwen2.5:7b"));
        assert!(!is_small_model("qwen2.5:14b"));
        assert!(!is_small_model("qwen2.5:72b"));
    }

    #[test]
    fn test_small_model_detection_llama() {
        assert!(is_small_model("llama3.2:1b"));
        assert!(is_small_model("llama3.2:3b"));

        assert!(!is_small_model("llama3.2:7b"));
        assert!(!is_small_model("llama3:8b"));
        assert!(!is_small_model("llama3:70b"));
    }

    #[test]
    fn test_small_model_detection_phi() {
        assert!(is_small_model("phi-2"));
        assert!(is_small_model("phi-3-mini"));
        assert!(is_small_model("some-model-mini"));
        assert!(is_small_model("tiny-llama"));
    }

    #[test]
    fn test_small_model_detection_large() {
        assert!(!is_small_model("claude-3-opus"));
        assert!(!is_small_model("gpt-4"));
        assert!(!is_small_model("mistral-7b"));
        assert!(!is_small_model("mixtral-8x7b"));
    }

    #[test]
    fn test_agent_config_default_is_small() {
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
        assert!(!optimized.agentic_rag.llm_query_rewriting);
        assert!(optimized.agentic_rag.use_rule_based_expansion);
    }

    #[test]
    fn test_agent_config_agentic_rag_for_small_model() {
        let config = AgentConfig::with_model("qwen2.5:1.5b");
        assert!(config.is_small_model());
        assert!(!config.agentic_rag.llm_query_rewriting);
        assert!(!config.agentic_rag.llm_sufficiency_check);
        assert_eq!(config.agentic_rag.max_iterations, 0);
        assert!(config.agentic_rag.use_rule_based_expansion);
    }

    #[test]
    fn test_agent_config_agentic_rag_for_large_model() {
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
