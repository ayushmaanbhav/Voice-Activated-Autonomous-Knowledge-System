//! MemGPT-Style Agentic Memory System
//!
//! This module implements a hierarchical memory architecture inspired by:
//! - MemGPT (arXiv:2310.08560): Virtual context management
//! - A-MEM (arXiv:2502.12110): Zettelkasten-style memory linking
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      Main Context                            │
//! │  ┌──────────────┬────────────────────┬───────────────────┐  │
//! │  │    System    │   Core Memory      │    FIFO Queue     │  │
//! │  │ Instructions │  (Human + Persona) │  (Recent Turns)   │  │
//! │  └──────────────┴────────────────────┴───────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                           ↕ Memory Functions
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    External Context                          │
//! │  ┌─────────────────────────┬────────────────────────────┐   │
//! │  │    Archival Storage     │      Recall Storage        │   │
//! │  │   (Vector DB / Long)    │   (Conversation Search)    │   │
//! │  └─────────────────────────┴────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Functions (callable by agent)
//!
//! - `core_memory_append`: Add to human block
//! - `core_memory_replace`: Update human block
//! - `archival_memory_insert`: Store in long-term memory
//! - `archival_memory_search`: Search long-term memory
//! - `conversation_search`: Search conversation history

pub mod archival;
pub mod core;
pub mod recall;

pub use archival::{
    ArchivalMemory, ArchivalMemoryConfig, ArchivalSearchResult, MemoryNote, MemorySource,
    MemoryType,
};
pub use core::{
    CoreMemory, CoreMemoryConfig, CoreMemoryError, EntrySource, HumanBlock, MemoryBlockEntry,
    PersonaBlock,
};
pub use recall::{
    ConversationTurn, RecallMemory, RecallMemoryConfig, RecallSearchResult, TurnRole,
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use voice_agent_core::{GenerateRequest, LanguageModel};

/// Unified memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticMemoryConfig {
    /// Core memory configuration
    pub core: CoreMemoryConfig,
    /// Archival memory configuration
    pub archival: ArchivalMemoryConfig,
    /// Recall memory configuration
    pub recall: RecallMemoryConfig,
    /// Maximum tokens for combined context
    pub max_context_tokens: usize,
    /// High watermark for context compaction
    pub high_watermark_tokens: usize,
    /// Low watermark target after compaction
    pub low_watermark_tokens: usize,
    /// Enable automatic summarization
    pub auto_summarize: bool,
}

impl Default for AgenticMemoryConfig {
    fn default() -> Self {
        Self {
            core: CoreMemoryConfig::default(),
            archival: ArchivalMemoryConfig::default(),
            recall: RecallMemoryConfig::default(),
            max_context_tokens: 4096,
            high_watermark_tokens: 3072,
            low_watermark_tokens: 2048,
            auto_summarize: true,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Core memory tokens
    pub core_tokens: usize,
    /// FIFO (recent turns) tokens
    pub fifo_tokens: usize,
    /// Total recall memory tokens
    pub recall_total_tokens: usize,
    /// Archival memory count
    pub archival_count: usize,
    /// Total estimated context tokens
    pub total_context_tokens: usize,
    /// Whether above high watermark
    pub above_high_watermark: bool,
    /// Whether above max limit
    pub above_max_limit: bool,
}

/// Agentic Memory System
///
/// Unified MemGPT-style memory management combining:
/// - Core Memory: Always in context (human + persona blocks)
/// - Recall Memory: Searchable conversation history with FIFO
/// - Archival Memory: Long-term vector-based storage
pub struct AgenticMemory {
    config: AgenticMemoryConfig,
    /// Core memory (always in context)
    pub core: CoreMemory,
    /// Recall memory (conversation history)
    pub recall: RecallMemory,
    /// Archival memory (long-term storage)
    pub archival: ArchivalMemory,
    /// Session ID for this memory instance
    session_id: String,
    /// Optional LLM for summarization
    llm: RwLock<Option<Arc<dyn LanguageModel>>>,
}

impl AgenticMemory {
    /// Create new agentic memory system
    pub fn new(config: AgenticMemoryConfig, session_id: impl Into<String>) -> Self {
        Self {
            core: CoreMemory::new(config.core.clone()),
            recall: RecallMemory::new(config.recall.clone()),
            archival: ArchivalMemory::new(config.archival.clone()),
            config,
            session_id: session_id.into(),
            llm: RwLock::new(None),
        }
    }

    /// Create with default config
    pub fn with_session(session_id: impl Into<String>) -> Self {
        Self::new(AgenticMemoryConfig::default(), session_id)
    }

    /// Set LLM for summarization
    pub fn set_llm(&self, llm: Arc<dyn LanguageModel>) {
        *self.llm.write() = Some(llm);
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    // =========================================================================
    // MemGPT-Style Memory Functions
    // =========================================================================

    /// Append to core memory (human block)
    ///
    /// MemGPT function: core_memory_append
    pub fn core_memory_append(&self, key: &str, value: &str) -> Result<(), CoreMemoryError> {
        self.core.human_append(key, value)
    }

    /// Replace in core memory (human block)
    ///
    /// MemGPT function: core_memory_replace
    pub fn core_memory_replace(
        &self,
        key: &str,
        old_value: &str,
        new_value: &str,
    ) -> Result<(), CoreMemoryError> {
        self.core.human_replace(key, old_value, new_value)
    }

    /// Insert into archival memory
    ///
    /// MemGPT function: archival_memory_insert
    pub fn archival_memory_insert(&self, content: &str, memory_type: MemoryType) -> Uuid {
        let note = MemoryNote::new(&self.session_id, content, memory_type);
        self.archival.insert(note)
    }

    /// Insert detailed memory note
    pub fn archival_memory_insert_note(&self, note: MemoryNote) -> Uuid {
        self.archival.insert(note)
    }

    /// Search archival memory
    ///
    /// MemGPT function: archival_memory_search
    pub fn archival_memory_search(
        &self,
        query: &str,
        top_k: Option<usize>,
    ) -> Vec<ArchivalSearchResult> {
        self.archival.search(query, top_k)
    }

    /// Search conversation history
    ///
    /// MemGPT function: conversation_search
    pub fn conversation_search(&self, query: &str, top_k: Option<usize>) -> Vec<RecallSearchResult> {
        self.recall.search(query, top_k)
    }

    // =========================================================================
    // Conversation Management
    // =========================================================================

    /// Add a user turn
    pub fn add_user_turn(&self, content: &str) -> u64 {
        let turn = ConversationTurn::new(TurnRole::User, content);
        self.recall.add_turn(turn)
    }

    /// Add an assistant turn
    pub fn add_assistant_turn(&self, content: &str) -> u64 {
        let turn = ConversationTurn::new(TurnRole::Assistant, content);
        self.recall.add_turn(turn)
    }

    /// Add a turn with metadata
    pub fn add_turn(&self, turn: ConversationTurn) -> u64 {
        self.recall.add_turn(turn)
    }

    /// Get recent conversation (FIFO)
    pub fn get_recent_turns(&self) -> Vec<ConversationTurn> {
        self.recall.get_fifo()
    }

    /// Get all turns
    pub fn get_all_turns(&self) -> Vec<ConversationTurn> {
        self.recall.get_all()
    }

    // =========================================================================
    // Context Generation
    // =========================================================================

    /// Get formatted context for LLM
    ///
    /// Returns the complete context including:
    /// 1. Core memory (persona + human blocks)
    /// 2. FIFO recent turns
    pub fn get_context(&self) -> String {
        let mut context = String::new();

        // Core memory (always included)
        context.push_str(&self.core.format_for_context());
        context.push('\n');

        // Recent conversation (FIFO)
        let fifo_context = self.recall.format_fifo_for_context();
        if !fifo_context.is_empty() {
            context.push_str("## Recent Conversation\n");
            context.push_str(&fifo_context);
            context.push('\n');
        }

        context
    }

    /// Get context with RAG results
    pub fn get_context_with_rag(&self, rag_context: &str) -> String {
        let mut context = self.get_context();

        if !rag_context.is_empty() {
            context.push_str("\n## Retrieved Knowledge\n");
            context.push_str(rag_context);
            context.push('\n');
        }

        context
    }

    /// Get context limited to token budget
    pub fn get_context_limited(&self, max_tokens: usize) -> String {
        let full_context = self.get_context();
        let estimated = full_context.len() / 4;

        if estimated <= max_tokens {
            return full_context;
        }

        // Prioritize: persona > customer facts > recent turns
        let mut context = String::new();

        // Always include persona
        let persona = self.core.persona_snapshot();
        context.push_str(&persona.format_for_context());
        context.push('\n');

        // Include customer name if available
        let human = self.core.human_snapshot();
        if let Some(name) = &human.name {
            context.push_str(&format!("Customer: {}\n", name));
        }

        // Add as many FIFO turns as fit
        let remaining_tokens = max_tokens.saturating_sub(context.len() / 4);
        let fifo = self.recall.get_fifo();
        let mut fifo_tokens = 0;

        context.push_str("\n## Conversation\n");
        for turn in fifo.iter().rev() {
            if fifo_tokens + turn.estimated_tokens > remaining_tokens {
                break;
            }
            fifo_tokens += turn.estimated_tokens;
        }

        // Add turns in correct order
        let turns_to_include = fifo
            .iter()
            .rev()
            .take_while(|t| {
                let include = fifo_tokens >= t.estimated_tokens;
                fifo_tokens = fifo_tokens.saturating_sub(t.estimated_tokens);
                include
            })
            .collect::<Vec<_>>();

        for turn in turns_to_include.into_iter().rev() {
            context.push_str(&turn.format_for_context());
            context.push('\n');
        }

        context
    }

    // =========================================================================
    // Memory Management
    // =========================================================================

    /// Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let core_tokens = self.core.estimated_tokens();
        let fifo_tokens = self.recall.fifo_tokens();
        let recall_total_tokens = self.recall.total_tokens();
        let archival_count = self.archival.len();

        let total_context_tokens = core_tokens + fifo_tokens;

        MemoryStats {
            core_tokens,
            fifo_tokens,
            recall_total_tokens,
            archival_count,
            total_context_tokens,
            above_high_watermark: total_context_tokens > self.config.high_watermark_tokens,
            above_max_limit: total_context_tokens > self.config.max_context_tokens,
        }
    }

    /// Check if memory needs compaction
    pub fn needs_compaction(&self) -> bool {
        self.get_stats().above_high_watermark
    }

    /// Perform memory compaction
    ///
    /// This:
    /// 1. Summarizes pending recall turns
    /// 2. Moves summaries to archival storage
    /// 3. Cleans up low-confidence facts
    pub async fn compact(&self) -> Result<(), String> {
        // Get pending turns for summarization
        let pending = self.recall.get_pending_summarization();

        if pending.is_empty() {
            return Ok(());
        }

        // Try to summarize with LLM
        let summary = self.summarize_turns(&pending).await?;

        // Store summary in archival
        let note = MemoryNote::new(&self.session_id, &summary, MemoryType::ConversationSummary)
            .with_context("Conversation summary")
            .with_tags(vec!["summary".to_string()]);

        self.archival.insert(note);

        tracing::debug!(
            turns = pending.len(),
            "Compacted conversation turns into summary"
        );

        Ok(())
    }

    /// Summarize turns using LLM
    async fn summarize_turns(&self, turns: &[ConversationTurn]) -> Result<String, String> {
        let llm = {
            let guard = self.llm.read();
            match guard.as_ref() {
                Some(llm) => llm.clone(),
                None => {
                    // Fallback: simple concatenation
                    return Ok(self.simple_summary(turns));
                }
            }
        };

        // Build conversation text
        let conversation: String = turns
            .iter()
            .map(|t| t.format_for_context())
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"Summarize this gold loan conversation segment concisely (1-2 sentences).
Focus on: customer needs, loan details mentioned, any concerns raised.

Conversation:
{}

Summary:"#,
            conversation
        );

        let request = GenerateRequest::new("You are a helpful summarization assistant.")
            .with_user_message(prompt);

        match llm.generate(request).await {
            Ok(response) => Ok(response.text.trim().to_string()),
            Err(e) => {
                tracing::warn!("LLM summarization failed: {}", e);
                Ok(self.simple_summary(turns))
            }
        }
    }

    /// Simple summarization fallback
    fn simple_summary(&self, turns: &[ConversationTurn]) -> String {
        let user_content: Vec<_> = turns
            .iter()
            .filter(|t| t.role == TurnRole::User)
            .map(|t| {
                if t.content.len() > 50 {
                    format!("{}...", &t.content[..50])
                } else {
                    t.content.clone()
                }
            })
            .collect();

        format!("User discussed: {}", user_content.join("; "))
    }

    /// Clear all memory for this session
    pub fn clear(&self) {
        self.core.clear_human_block();
        self.core.clear_persona_goals();
        self.recall.clear();
        self.archival.clear_session(&self.session_id);
    }

    /// Reset to default state (including persona)
    pub fn reset(&self) {
        self.core.reset();
        self.recall.clear();
        self.archival.clear_session(&self.session_id);
    }
}

impl Default for AgenticMemory {
    fn default() -> Self {
        Self::new(AgenticMemoryConfig::default(), Uuid::new_v4().to_string())
    }
}

// ============================================================================
// Backward Compatibility - Re-export legacy types
// ============================================================================

// Re-export config from voice_agent_config for backward compatibility
pub use voice_agent_config::MemoryConfig;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agentic_memory_creation() {
        let memory = AgenticMemory::with_session("test-session");
        assert_eq!(memory.session_id(), "test-session");
    }

    #[test]
    fn test_core_memory_functions() {
        let memory = AgenticMemory::with_session("test-session");

        // Append
        assert!(memory.core_memory_append("loan_amount", "500000").is_ok());

        // Verify
        let human = memory.core.human_snapshot();
        assert!(human.get_fact("loan_amount").is_some());

        // Replace
        assert!(memory
            .core_memory_replace("loan_amount", "500000", "750000")
            .is_ok());
        let human = memory.core.human_snapshot();
        assert_eq!(human.get_fact("loan_amount").unwrap().value, "750000");
    }

    #[test]
    fn test_conversation_flow() {
        let memory = AgenticMemory::with_session("test-session");

        memory.add_user_turn("I want a gold loan");
        memory.add_assistant_turn("Sure! How much gold do you have?");
        memory.add_user_turn("About 50 grams");

        assert_eq!(memory.recall.len(), 3);

        let recent = memory.get_recent_turns();
        assert!(!recent.is_empty());
    }

    #[test]
    fn test_archival_memory() {
        let memory = AgenticMemory::with_session("test-session");

        let id = memory.archival_memory_insert("Customer prefers Hindi", MemoryType::Preference);
        assert!(!id.is_nil());

        let results = memory.archival_memory_search("Hindi", Some(5));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_conversation_search() {
        let memory = AgenticMemory::with_session("test-session");

        memory.add_user_turn("I have 50 grams of gold");
        memory.add_user_turn("The purity is 22 karat");

        let results = memory.conversation_search("gold", Some(5));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_context_generation() {
        let memory = AgenticMemory::with_session("test-session");

        memory.core.set_customer_name("Rajesh");
        memory.add_user_turn("I need a gold loan");
        memory.add_assistant_turn("I can help with that!");

        let context = memory.get_context();

        assert!(context.contains("Priya")); // Default persona
        assert!(context.contains("Rajesh"));
        assert!(context.contains("gold loan"));
    }

    #[test]
    fn test_memory_stats() {
        let memory = AgenticMemory::with_session("test-session");

        memory.add_user_turn("Hello");
        memory.add_assistant_turn("Hi!");

        let stats = memory.get_stats();
        assert!(stats.fifo_tokens > 0);
        assert!(stats.core_tokens > 0);
    }
}
