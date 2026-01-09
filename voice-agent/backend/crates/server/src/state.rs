//! Application State
//!
//! Shared state across all handlers.
//!
//! P12 FIX: Removed legacy DomainConfigManager. All domain configuration now flows
//! through MasterDomainConfig and its views (AgentDomainView, LlmDomainView, ToolsDomainView).

use parking_lot::RwLock;
use std::sync::Arc;

use voice_agent_config::{load_settings, MasterDomainConfig, Settings};
use voice_agent_config::domain::{AgentDomainView, LlmDomainView, ToolsDomainView};
use voice_agent_rag::VectorStore;
use voice_agent_tools::ToolRegistry;
// P2 FIX: Text processing pipeline for grammar, PII, compliance
use voice_agent_text_processing::{TextProcessingConfig, TextProcessingPipeline, TextSimplifier};
// Deterministic phonetic error correction
use voice_agent_text_processing::grammar::PhoneticCorrector;
// Translation
use voice_agent_text_processing::translation::{TranslationConfig, create_translator};
use voice_agent_core::Translator;
// P2 FIX: Audit logging for RBI compliance
use voice_agent_persistence::{AuditLog, AuditLogger};

use crate::session::{InMemorySessionStore, SessionManager, SessionStore};

/// Application state
#[derive(Clone)]
pub struct AppState {
    /// P1 FIX: Configuration wrapped in RwLock for hot-reload support
    pub config: Arc<RwLock<Settings>>,
    /// P12 FIX: Hierarchical domain configuration (source of truth for all domain config)
    pub master_domain_config: Arc<MasterDomainConfig>,
    /// P6 FIX: Agent-specific view of domain config
    pub agent_view: Arc<AgentDomainView>,
    /// P6 FIX: LLM-specific view of domain config
    pub llm_view: Arc<LlmDomainView>,
    /// P6 FIX: Tools-specific view of domain config
    pub tools_view: Arc<ToolsDomainView>,
    /// Session manager
    pub sessions: Arc<SessionManager>,
    /// Tool registry
    pub tools: Arc<ToolRegistry>,
    /// P2-3 FIX: Session store for persistence (ScyllaDB or in-memory)
    pub session_store: Arc<dyn SessionStore>,
    /// P0 FIX: Vector store for RAG retrieval (optional - initialized if Qdrant is available)
    pub vector_store: Option<Arc<VectorStore>>,
    /// P2 FIX: Text processing pipeline for grammar, PII, compliance
    pub text_processing: Arc<TextProcessingPipeline>,
    /// P2 FIX: Text simplifier for TTS output (numbers, abbreviations)
    pub text_simplifier: Arc<TextSimplifier>,
    /// Deterministic phonetic corrector for ASR errors
    pub phonetic_corrector: Arc<PhoneticCorrector>,
    /// Translator for language conversion
    pub translator: Arc<dyn Translator>,
    /// P2 FIX: Audit logger for RBI compliance (wrapped in Arc for Clone)
    pub audit_logger: Option<Arc<AuditLogger>>,
    /// Environment name for config reload
    env: Option<String>,
}

impl AppState {
    /// Create default text processing components, phonetic corrector, and translator
    /// Uses empty phonetic corrector when no domain config provided
    fn create_text_processing() -> (Arc<TextProcessingPipeline>, Arc<TextSimplifier>, Arc<PhoneticCorrector>, Arc<dyn Translator>) {
        let text_config = TextProcessingConfig::default();
        let text_processing = Arc::new(TextProcessingPipeline::new(text_config, None));
        let text_simplifier = Arc::new(TextSimplifier::default_config());
        // Empty phonetic corrector - no domain config available
        let phonetic_corrector = Arc::new(PhoneticCorrector::empty());
        tracing::warn!("Initialized empty phonetic corrector (no domain config). Use create_text_processing_with_domain() instead.");
        // Translator for language conversion
        let translator = create_translator(&TranslationConfig::default());
        (text_processing, text_simplifier, phonetic_corrector, translator)
    }

    /// Create text processing components using domain configuration
    /// P16 FIX: Config-driven phonetic corrector - no more hardcoded domain-specific rules
    fn create_text_processing_with_domain(master_config: &MasterDomainConfig) -> (Arc<TextProcessingPipeline>, Arc<TextSimplifier>, Arc<PhoneticCorrector>, Arc<dyn Translator>) {
        use voice_agent_text_processing::grammar::PhoneticCorrectorConfig;

        let text_config = TextProcessingConfig::default();
        let text_processing = Arc::new(TextProcessingPipeline::new(text_config, None));
        let text_simplifier = Arc::new(TextSimplifier::default_config());

        // Config-driven phonetic corrector from domain.yaml
        let phonetic_config = &master_config.phonetic_corrections;
        let vocabulary = &master_config.vocabulary;

        // Convert contextual rules from config format to tuple format
        let contextual_rules: Vec<(String, String, String)> = phonetic_config
            .contextual_rules
            .iter()
            .map(|r| (r.context.clone(), r.error.clone(), r.correction.clone()))
            .collect();

        // Create PhoneticCorrectorConfig from domain config
        let corrector_config = PhoneticCorrectorConfig {
            max_edit_distance: phonetic_config.config.max_edit_distance,
            min_word_length: phonetic_config.config.min_word_length,
            fix_sentence_start: phonetic_config.config.fix_sentence_start,
        };

        let phonetic_corrector = Arc::new(PhoneticCorrector::from_domain_config(
            &vocabulary.terms,
            phonetic_config.confusion_rules.clone(),
            contextual_rules,
            phonetic_config.phrase_rules.clone(),
            corrector_config,
        ));

        tracing::info!(
            domain_id = %master_config.domain_id,
            vocabulary_terms = vocabulary.terms.len(),
            confusion_rules = phonetic_config.confusion_rules.len(),
            phrase_rules = phonetic_config.phrase_rules.len(),
            "Initialized config-driven phonetic corrector"
        );

        // Translator for language conversion
        let translator = create_translator(&TranslationConfig::default());
        (text_processing, text_simplifier, phonetic_corrector, translator)
    }

    /// P6 FIX: Create views from MasterDomainConfig
    fn create_views(master_config: &Arc<MasterDomainConfig>) -> (Arc<AgentDomainView>, Arc<LlmDomainView>, Arc<ToolsDomainView>) {
        let agent_view = Arc::new(AgentDomainView::new(Arc::clone(master_config)));
        let llm_view = Arc::new(LlmDomainView::new(Arc::clone(master_config)));
        let tools_view = Arc::new(ToolsDomainView::new(Arc::clone(master_config)));
        (agent_view, llm_view, tools_view)
    }

    /// Create new application state with in-memory session store
    pub fn new(config: Settings) -> Self {
        let (text_processing, text_simplifier, phonetic_corrector, translator) = Self::create_text_processing();
        let master_domain_config = Arc::new(MasterDomainConfig::default());
        let (agent_view, llm_view, tools_view) = Self::create_views(&master_domain_config);
        // P15 FIX: Create tools before moving tools_view into struct
        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view.clone()));
        Self {
            config: Arc::new(RwLock::new(config)),
            master_domain_config,
            agent_view,
            llm_view,
            tools_view,
            sessions: Arc::new(SessionManager::new(100)),
            tools,
            session_store: Arc::new(InMemorySessionStore::new()),
            vector_store: None,
            text_processing,
            text_simplifier,
            phonetic_corrector,
            translator,
            audit_logger: None,
            env: None,
        }
    }

    /// P12 FIX: Create new application state with master domain config
    pub fn with_master_domain_config(
        config: Settings,
        master_domain_config: Arc<MasterDomainConfig>,
    ) -> Self {
        // P16 FIX: Use config-driven phonetic corrector
        let (text_processing, text_simplifier, phonetic_corrector, translator) = Self::create_text_processing_with_domain(&master_domain_config);
        let (agent_view, llm_view, tools_view) = Self::create_views(&master_domain_config);
        // P15 FIX: Create tools before moving tools_view into struct
        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view.clone()));
        Self {
            config: Arc::new(RwLock::new(config)),
            master_domain_config,
            agent_view,
            llm_view,
            tools_view,
            sessions: Arc::new(SessionManager::new(100)),
            tools,
            session_store: Arc::new(InMemorySessionStore::new()),
            vector_store: None,
            text_processing,
            text_simplifier,
            phonetic_corrector,
            translator,
            audit_logger: None,
            env: None,
        }
    }

    /// Create new application state with environment name for reload support
    pub fn with_env(config: Settings, env: Option<String>) -> Self {
        let (text_processing, text_simplifier, phonetic_corrector, translator) = Self::create_text_processing();
        let master_domain_config = Arc::new(MasterDomainConfig::default());
        let (agent_view, llm_view, tools_view) = Self::create_views(&master_domain_config);
        // P15 FIX: Create tools before moving tools_view into struct
        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view.clone()));
        Self {
            config: Arc::new(RwLock::new(config)),
            master_domain_config,
            agent_view,
            llm_view,
            tools_view,
            sessions: Arc::new(SessionManager::new(100)),
            tools,
            session_store: Arc::new(InMemorySessionStore::new()),
            vector_store: None,
            text_processing,
            text_simplifier,
            phonetic_corrector,
            translator,
            audit_logger: None,
            env,
        }
    }

    /// P2-3 FIX: Create application state with custom session store (e.g., ScyllaDB)
    pub fn with_session_store(config: Settings, store: Arc<dyn SessionStore>) -> Self {
        let (text_processing, text_simplifier, phonetic_corrector, translator) = Self::create_text_processing();
        let master_domain_config = Arc::new(MasterDomainConfig::default());
        let (agent_view, llm_view, tools_view) = Self::create_views(&master_domain_config);
        // P15 FIX: Create tools before moving tools_view into struct
        let tools = Arc::new(voice_agent_tools::registry::create_registry_with_view(tools_view.clone()));
        Self {
            config: Arc::new(RwLock::new(config)),
            master_domain_config,
            agent_view,
            llm_view,
            tools_view,
            sessions: Arc::new(SessionManager::new(100)),
            tools,
            session_store: store,
            vector_store: None,
            text_processing,
            text_simplifier,
            phonetic_corrector,
            translator,
            audit_logger: None,
            env: None,
        }
    }

    /// P12 FIX: Create application state with full persistence layer
    ///
    /// This method wires the SMS and GoldPrice services from the persistence layer
    /// into the tool registry, enabling proper persistence of SMS messages and
    /// gold price queries to ScyllaDB.
    ///
    /// All business config (rates, LTV, etc.) now comes from ToolsDomainView.
    /// P16 FIX: Accept AssetPriceService (generic) instead of GoldPriceService
    pub fn with_full_persistence(
        config: Settings,
        store: Arc<dyn SessionStore>,
        master_domain_config: Arc<MasterDomainConfig>,
        sms_service: Arc<dyn voice_agent_persistence::SmsService>,
        gold_price_service: Arc<dyn voice_agent_persistence::AssetPriceService>,
    ) -> Self {
        // P16 FIX: Use config-driven phonetic corrector
        let (text_processing, text_simplifier, phonetic_corrector, translator) = Self::create_text_processing_with_domain(&master_domain_config);
        let (agent_view, llm_view, tools_view) = Self::create_views(&master_domain_config);

        // P15 FIX: Create tool registry with REQUIRED tools_view and persistence services
        let integration_config = voice_agent_tools::FullIntegrationConfig::new(tools_view.clone())
            .with_sms_service(sms_service)
            .with_gold_price_service(gold_price_service);
        let tools = voice_agent_tools::create_registry_with_persistence(integration_config);

        Self {
            config: Arc::new(RwLock::new(config)),
            master_domain_config,
            agent_view,
            llm_view,
            tools_view,
            sessions: Arc::new(SessionManager::new(100)),
            tools: Arc::new(tools),
            session_store: store,
            vector_store: None,
            text_processing,
            text_simplifier,
            phonetic_corrector,
            translator,
            audit_logger: None,
            env: None,
        }
    }

    /// P0 FIX: Set vector store for RAG retrieval
    pub fn with_vector_store(mut self, vector_store: Arc<VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// P2 FIX: Set audit logger for RBI compliance logging
    pub fn with_audit_logger(mut self, audit_log: Arc<dyn AuditLog>) -> Self {
        self.audit_logger = Some(Arc::new(AuditLogger::new(audit_log)));
        self
    }

    /// P2 FIX: Log an audit event for RBI compliance
    ///
    /// Returns Ok(()) if logger is not configured (noop).
    pub async fn log_conversation_start(
        &self,
        session_id: &str,
        language: &str,
    ) -> Result<(), crate::ServerError> {
        if let Some(ref logger) = self.audit_logger {
            logger
                .log_conversation_start(session_id, language)
                .await
                .map_err(|e| crate::ServerError::Persistence(e.to_string()))?;
        }
        Ok(())
    }

    /// P2 FIX: Log conversation end
    pub async fn log_conversation_end(
        &self,
        session_id: &str,
        reason: &str,
        duration_secs: u64,
    ) -> Result<(), crate::ServerError> {
        if let Some(ref logger) = self.audit_logger {
            logger
                .log_conversation_end(session_id, reason, duration_secs)
                .await
                .map_err(|e| crate::ServerError::Persistence(e.to_string()))?;
        }
        Ok(())
    }

    /// P1 FIX: Reload configuration from files
    ///
    /// Reloads config from disk and updates the shared state.
    /// Returns the new config on success.
    pub fn reload_config(&self) -> Result<(), String> {
        let new_config = load_settings(self.env.as_deref())
            .map_err(|e| format!("Failed to reload config: {}", e))?;

        // Update the config
        let mut config = self.config.write();
        *config = new_config;

        tracing::info!("Configuration reloaded successfully");
        Ok(())
    }

    /// Get a read guard to the current configuration
    pub fn get_config(&self) -> parking_lot::RwLockReadGuard<'_, Settings> {
        self.config.read()
    }

    /// P12 FIX: Get master domain configuration (source of truth for all domain config)
    pub fn get_master_domain_config(&self) -> &Arc<MasterDomainConfig> {
        &self.master_domain_config
    }

    /// P6 FIX: Get agent-specific view of domain config
    pub fn get_agent_view(&self) -> &Arc<AgentDomainView> {
        &self.agent_view
    }

    /// P6 FIX: Get LLM-specific view of domain config
    pub fn get_llm_view(&self) -> &Arc<LlmDomainView> {
        &self.llm_view
    }

    /// P6 FIX: Get tools-specific view of domain config
    pub fn get_tools_view(&self) -> &Arc<ToolsDomainView> {
        &self.tools_view
    }

    /// P2-3 FIX: Persist session metadata to the configured store
    ///
    /// Call this after creating a session or when session state changes
    /// that should be persisted (e.g., stage transitions, turn completion).
    pub async fn persist_session(
        &self,
        session: &crate::session::Session,
    ) -> Result<(), crate::ServerError> {
        self.session_store.store_metadata(session).await
    }

    /// P2-3 FIX: Check if session persistence is distributed (ScyllaDB/Redis)
    pub fn is_distributed_sessions(&self) -> bool {
        self.session_store.is_distributed()
    }

    /// P2 FIX: Recover active sessions on server restart
    ///
    /// Loads session metadata from persistent storage and logs recoverable sessions.
    /// Note: Full agent state recovery requires conversation history serialization
    /// which is not implemented. This method provides visibility into sessions
    /// that were active before restart.
    ///
    /// Returns the count of sessions found (not fully restored).
    pub async fn recover_sessions(&self) -> Result<usize, crate::ServerError> {
        if !self.is_distributed_sessions() {
            tracing::debug!("Session recovery skipped: not using distributed session store");
            return Ok(0);
        }

        match self.session_store.list_active_sessions(100).await {
            Ok(sessions) => {
                let now = chrono::Utc::now();
                let active_sessions: Vec<_> = sessions
                    .into_iter()
                    .filter(|s| s.expires_at > now)
                    .collect();

                if active_sessions.is_empty() {
                    tracing::info!("No active sessions to recover");
                } else {
                    tracing::info!(
                        count = active_sessions.len(),
                        "Found recoverable sessions from previous run"
                    );

                    // Log details of each recoverable session
                    for session in &active_sessions {
                        tracing::info!(
                            session_id = %session.session_id,
                            stage = %session.conversation_stage,
                            turn_count = session.turn_count,
                            language = %session.language,
                            age_minutes = (now - session.created_at).num_minutes(),
                            "Recoverable session found"
                        );
                    }
                }

                Ok(active_sessions.len())
            },
            Err(e) => {
                tracing::warn!(error = %e, "Failed to query active sessions for recovery");
                Err(crate::ServerError::Persistence(e.to_string()))
            },
        }
    }
}
