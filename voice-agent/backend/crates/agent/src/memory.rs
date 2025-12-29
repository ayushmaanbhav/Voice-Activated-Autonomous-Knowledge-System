//! Conversation Memory
//!
//! Hierarchical memory system:
//! - Working memory: Recent turns
//! - Episodic memory: Summarized past segments
//! - Semantic memory: Key facts and entities

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use voice_agent_core::{Turn, TurnRole, LanguageModel, GenerateRequest};

/// Memory configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum turns in working memory
    pub working_memory_size: usize,
    /// Threshold for summarizing to episodic
    pub summarization_threshold: usize,
    /// Maximum episodic summaries
    pub max_episodic_summaries: usize,
    /// Enable semantic memory
    pub semantic_memory_enabled: bool,
    /// P1 FIX: Maximum total tokens before aggressive truncation
    pub max_context_tokens: usize,
    /// P1 FIX: High watermark - trigger summarization when exceeded
    pub high_watermark_tokens: usize,
    /// P1 FIX: Low watermark - target after truncation
    pub low_watermark_tokens: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_size: 8,
            summarization_threshold: 6,
            max_episodic_summaries: 10,
            semantic_memory_enabled: true,
            // P1 FIX: Token-based limits (assuming ~4 chars per token)
            max_context_tokens: 4096,      // Hard limit
            high_watermark_tokens: 3072,   // 75% - trigger summarization
            low_watermark_tokens: 2048,    // 50% - target after cleanup
        }
    }
}

/// P1 FIX: Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Estimated total tokens in memory
    pub estimated_tokens: usize,
    /// Total characters in memory
    pub total_chars: usize,
    /// Number of working memory entries
    pub working_entries: usize,
    /// Number of episodic summaries
    pub episodic_entries: usize,
    /// Number of semantic facts
    pub semantic_facts: usize,
    /// Whether high watermark is exceeded
    pub above_high_watermark: bool,
    /// Whether max limit is exceeded
    pub above_max_limit: bool,
}

/// Memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Role (user/assistant)
    pub role: String,
    /// Content
    pub content: String,
    /// Timestamp (ms since conversation start)
    pub timestamp_ms: u64,
    /// Stage during this entry
    pub stage: Option<String>,
    /// Detected intents
    pub intents: Vec<String>,
    /// Extracted entities
    pub entities: HashMap<String, String>,
}

impl From<&Turn> for MemoryEntry {
    fn from(turn: &Turn) -> Self {
        Self {
            role: match turn.role {
                TurnRole::User => "user",
                TurnRole::Assistant => "assistant",
                TurnRole::System => "system",
            }.to_string(),
            content: turn.content.clone(),
            timestamp_ms: turn.timestamp.timestamp_millis() as u64,
            stage: None,
            intents: Vec::new(),
            entities: HashMap::new(),
        }
    }
}

/// Episodic summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicSummary {
    /// Summary text
    pub summary: String,
    /// Time range covered (start_ms, end_ms)
    pub time_range: (u64, u64),
    /// Key topics discussed
    pub topics: Vec<String>,
    /// Stage transitions
    pub stage_transitions: Vec<String>,
    /// Number of turns summarized
    pub turns_count: usize,
}

/// Semantic fact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFact {
    /// Fact key (e.g., "customer_name", "loan_amount")
    pub key: String,
    /// Fact value
    pub value: String,
    /// Confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Source turn index
    pub source_turn: usize,
    /// Last updated timestamp
    pub updated_at_ms: u64,
}

/// Conversation memory
pub struct ConversationMemory {
    config: MemoryConfig,
    /// Working memory (recent turns)
    working: RwLock<Vec<MemoryEntry>>,
    /// Episodic memory (summaries)
    /// P2 FIX: Uses VecDeque for O(1) removal from front.
    episodic: RwLock<VecDeque<EpisodicSummary>>,
    /// Semantic memory (facts)
    semantic: RwLock<HashMap<String, SemanticFact>>,
    /// Total turns processed
    total_turns: RwLock<usize>,
    /// P0 FIX: Optional LLM backend for real summarization
    /// P1 FIX: Now uses LanguageModel trait for proper abstraction
    llm: RwLock<Option<Arc<dyn LanguageModel>>>,
    /// Entries pending summarization (collected when no LLM available)
    pending_summarization: RwLock<Vec<MemoryEntry>>,
}

impl ConversationMemory {
    /// Create new conversation memory
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            working: RwLock::new(Vec::new()),
            episodic: RwLock::new(VecDeque::new()),
            semantic: RwLock::new(HashMap::new()),
            total_turns: RwLock::new(0),
            llm: RwLock::new(None),
            pending_summarization: RwLock::new(Vec::new()),
        }
    }

    /// P0 FIX: Set LLM backend for real summarization
    /// P1 FIX: Now accepts LanguageModel trait for proper abstraction
    ///
    /// When an LLM is set, the memory system will use it to generate
    /// meaningful summaries instead of just concatenating text.
    pub fn set_llm(&self, llm: Arc<dyn LanguageModel>) {
        *self.llm.write() = Some(llm);
    }

    /// Add a memory entry
    pub fn add(&self, entry: MemoryEntry) {
        let mut working = self.working.write();
        let mut total = self.total_turns.write();

        working.push(entry);
        *total += 1;

        // Check if we need to summarize
        if working.len() > self.config.working_memory_size {
            // In production, would call LLM to summarize
            // For now, just trim
            let to_summarize: Vec<MemoryEntry> = working.drain(..self.config.summarization_threshold).collect();
            self.create_episodic_summary(to_summarize);
        }
    }

    /// Add a turn
    pub fn add_turn(&self, turn: &Turn) {
        let entry = MemoryEntry::from(turn);
        self.add(entry);
    }

    /// Create episodic summary from turns (sync fallback)
    ///
    /// This is used when no LLM is available or when async context is not available.
    /// For real summarization, use `summarize_pending_async()`.
    fn create_episodic_summary(&self, entries: Vec<MemoryEntry>) {
        if entries.is_empty() {
            return;
        }

        // If we have an LLM, store for async summarization later
        if self.llm.read().is_some() {
            self.pending_summarization.write().extend(entries);
            return;
        }

        // Fallback: simple concatenation-based summary
        self.create_simple_summary(entries);
    }

    /// P0 FIX: Create simple summary without LLM (fallback)
    fn create_simple_summary(&self, entries: Vec<MemoryEntry>) {
        if entries.is_empty() {
            return;
        }

        let start_ms = entries.first().map(|e| e.timestamp_ms).unwrap_or(0);
        let end_ms = entries.last().map(|e| e.timestamp_ms).unwrap_or(0);

        let topics: Vec<String> = entries.iter()
            .flat_map(|e| e.intents.clone())
            .collect();

        // P2 FIX: Truncate at word boundaries instead of mid-word
        let summary = entries.iter()
            .filter(|e| e.role == "user")
            .map(|e| Self::truncate_at_word_boundary(&e.content, 50))
            .collect::<Vec<_>>()
            .join("; ");

        let episodic = EpisodicSummary {
            summary: format!("User discussed: {}", summary),
            time_range: (start_ms, end_ms),
            topics,
            stage_transitions: Vec::new(),
            turns_count: entries.len(),
        };

        let mut episodic_memory = self.episodic.write();
        episodic_memory.push_back(episodic);

        // Trim if too many - P2 FIX: O(1) removal with VecDeque
        if episodic_memory.len() > self.config.max_episodic_summaries {
            episodic_memory.pop_front();
        }
    }

    /// P0 FIX: Summarize pending entries using LLM
    ///
    /// This is the async method that should be called periodically to
    /// generate meaningful summaries from pending conversation entries.
    pub async fn summarize_pending_async(&self) -> Result<(), String> {
        let entries: Vec<MemoryEntry> = {
            let mut pending = self.pending_summarization.write();
            std::mem::take(&mut *pending)
        };

        if entries.is_empty() {
            return Ok(());
        }

        let llm = {
            let llm_guard = self.llm.read();
            match llm_guard.as_ref() {
                Some(llm) => llm.clone(),
                None => {
                    // No LLM available, use simple summary
                    self.create_simple_summary(entries);
                    return Ok(());
                }
            }
        };

        // Build conversation text for summarization
        let conversation_text: String = entries.iter()
            .map(|e| format!("{}: {}", e.role, e.content))
            .collect::<Vec<_>>()
            .join("\n");

        let start_ms = entries.first().map(|e| e.timestamp_ms).unwrap_or(0);
        let end_ms = entries.last().map(|e| e.timestamp_ms).unwrap_or(0);
        let topics: Vec<String> = entries.iter()
            .flat_map(|e| e.intents.clone())
            .collect();

        // Create summarization prompt
        let prompt = format!(
            r#"Summarize this gold loan conversation segment concisely (1-2 sentences).
Focus on: customer needs, loan details mentioned, any concerns raised.

Conversation:
{}

Summary:"#,
            conversation_text
        );

        // P1 FIX: Use GenerateRequest for LanguageModel trait
        let request = GenerateRequest::new("You are a helpful summarization assistant.")
            .with_user_message(prompt);

        // Call LLM for summarization
        match llm.generate(request).await {
            Ok(response) => {
                let summary_text = response.text.trim().to_string();

                let episodic = EpisodicSummary {
                    summary: summary_text,
                    time_range: (start_ms, end_ms),
                    topics,
                    stage_transitions: Vec::new(),
                    turns_count: entries.len(),
                };

                let mut episodic_memory = self.episodic.write();
                episodic_memory.push_back(episodic);

                if episodic_memory.len() > self.config.max_episodic_summaries {
                    episodic_memory.pop_front();
                }

                tracing::debug!("Created LLM-based episodic summary for {} turns", entries.len());
                Ok(())
            }
            Err(e) => {
                tracing::warn!("LLM summarization failed, using fallback: {}", e);
                // Fallback to simple summary
                self.create_simple_summary(entries);
                Err(format!("LLM summarization failed: {}", e))
            }
        }
    }

    /// P0 FIX: Check if there are pending entries to summarize
    pub fn has_pending_summarization(&self) -> bool {
        !self.pending_summarization.read().is_empty()
    }

    /// P0 FIX: Get count of pending summarization entries
    pub fn pending_count(&self) -> usize {
        self.pending_summarization.read().len()
    }

    /// Add semantic fact
    pub fn add_fact(&self, key: &str, value: &str, confidence: f32) {
        if !self.config.semantic_memory_enabled {
            return;
        }

        let total = *self.total_turns.read();

        let fact = SemanticFact {
            key: key.to_string(),
            value: value.to_string(),
            confidence,
            source_turn: total,
            updated_at_ms: 0, // Would use actual timestamp
        };

        self.semantic.write().insert(key.to_string(), fact);
    }

    /// Get fact by key
    pub fn get_fact(&self, key: &str) -> Option<SemanticFact> {
        self.semantic.read().get(key).cloned()
    }

    /// Get all facts
    pub fn all_facts(&self) -> HashMap<String, SemanticFact> {
        self.semantic.read().clone()
    }

    /// Get working memory entries
    pub fn working_memory(&self) -> Vec<MemoryEntry> {
        self.working.read().clone()
    }

    /// Get episodic summaries
    pub fn episodic_summaries(&self) -> Vec<EpisodicSummary> {
        self.episodic.read().iter().cloned().collect()
    }

    /// Get context for LLM (formatted)
    pub fn get_context(&self) -> String {
        let mut context = String::new();

        // Add semantic facts
        let facts = self.semantic.read();
        if !facts.is_empty() {
            context.push_str("## Known Facts\n");
            for (key, fact) in facts.iter() {
                context.push_str(&format!("- {}: {}\n", key, fact.value));
            }
            context.push('\n');
        }

        // Add episodic summaries
        let episodic = self.episodic.read();
        if !episodic.is_empty() {
            context.push_str("## Conversation Summary\n");
            for summary in episodic.iter() {
                context.push_str(&format!("- {}\n", summary.summary));
            }
            context.push('\n');
        }

        context
    }

    /// Get recent conversation for LLM
    pub fn get_recent_messages(&self) -> Vec<(String, String)> {
        self.working.read()
            .iter()
            .map(|e| (e.role.clone(), e.content.clone()))
            .collect()
    }

    /// Clear all memory
    pub fn clear(&self) {
        self.working.write().clear();
        self.episodic.write().clear();
        self.semantic.write().clear();
        *self.total_turns.write() = 0;
    }

    /// Get total turns
    pub fn total_turns(&self) -> usize {
        *self.total_turns.read()
    }

    /// P1 FIX: Get memory statistics
    pub fn get_stats(&self) -> MemoryStats {
        let working = self.working.read();
        let episodic = self.episodic.read();
        let semantic = self.semantic.read();

        let working_chars: usize = working.iter().map(|e| e.content.len()).sum();
        let episodic_chars: usize = episodic.iter().map(|e| e.summary.len()).sum();
        let semantic_chars: usize = semantic.values().map(|f| f.value.len()).sum();

        let total_chars = working_chars + episodic_chars + semantic_chars;
        // Rough estimate: ~4 characters per token
        let estimated_tokens = total_chars / 4;

        MemoryStats {
            estimated_tokens,
            total_chars,
            working_entries: working.len(),
            episodic_entries: episodic.len(),
            semantic_facts: semantic.len(),
            above_high_watermark: estimated_tokens > self.config.high_watermark_tokens,
            above_max_limit: estimated_tokens > self.config.max_context_tokens,
        }
    }

    /// P1 FIX: Check if memory needs cleanup (above high watermark)
    pub fn needs_cleanup(&self) -> bool {
        self.get_stats().above_high_watermark
    }

    /// P1 FIX: Perform aggressive memory cleanup to get below low watermark
    ///
    /// This is called when memory exceeds the high watermark.
    /// It will:
    /// 1. Force summarization of all working memory
    /// 2. Remove oldest episodic summaries
    /// 3. Remove low-confidence semantic facts
    pub fn cleanup_to_watermark(&self) {
        let stats = self.get_stats();
        if !stats.above_high_watermark {
            return;
        }

        tracing::info!(
            tokens = stats.estimated_tokens,
            high_watermark = self.config.high_watermark_tokens,
            "Memory cleanup triggered"
        );

        // 1. Force summarize all working memory except last 2 entries
        {
            let mut working = self.working.write();
            let len = working.len();
            if len > 2 {
                let to_summarize: Vec<MemoryEntry> = working.drain(..len - 2).collect();
                drop(working); // Release lock before creating summary
                self.create_simple_summary(to_summarize);
            }
        }

        // 2. Remove oldest episodic summaries if still over limit
        loop {
            let stats = self.get_stats();
            if stats.estimated_tokens <= self.config.low_watermark_tokens {
                break;
            }

            let mut episodic = self.episodic.write();
            if episodic.len() <= 1 {
                break; // Keep at least one summary for context
            }
            episodic.pop_front();
            tracing::debug!("Removed oldest episodic summary");
        }

        // 3. Remove low-confidence semantic facts if still over limit
        {
            let stats = self.get_stats();
            if stats.estimated_tokens > self.config.low_watermark_tokens {
                let mut semantic = self.semantic.write();
                let low_confidence_keys: Vec<String> = semantic.iter()
                    .filter(|(_, f)| f.confidence < 0.5)
                    .map(|(k, _)| k.clone())
                    .collect();

                for key in low_confidence_keys {
                    semantic.remove(&key);
                    tracing::debug!("Removed low-confidence fact: {}", key);
                }
            }
        }

        let final_stats = self.get_stats();
        tracing::info!(
            tokens_before = stats.estimated_tokens,
            tokens_after = final_stats.estimated_tokens,
            "Memory cleanup completed"
        );
    }

    /// P1 FIX: Get context with size limit
    ///
    /// Returns context string truncated to approximately max_tokens.
    pub fn get_context_limited(&self, max_tokens: usize) -> String {
        let full_context = self.get_context();
        let max_chars = max_tokens * 4; // Rough estimate

        if full_context.len() <= max_chars {
            return full_context;
        }

        // Truncate at word boundary
        Self::truncate_at_word_boundary(&full_context, max_chars)
    }

    /// P2 FIX: Truncate text at word boundary to avoid cutting mid-word
    fn truncate_at_word_boundary(text: &str, max_chars: usize) -> String {
        if text.len() <= max_chars {
            return text.to_string();
        }

        // Find the last space before max_chars
        let truncated = &text[..max_chars];
        if let Some(last_space) = truncated.rfind(char::is_whitespace) {
            format!("{}...", &text[..last_space])
        } else {
            // No space found, just truncate with ellipsis
            format!("{}...", truncated)
        }
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entry() {
        let memory = ConversationMemory::default();

        memory.add(MemoryEntry {
            role: "user".to_string(),
            content: "Hello".to_string(),
            timestamp_ms: 0,
            stage: None,
            intents: vec![],
            entities: HashMap::new(),
        });

        assert_eq!(memory.total_turns(), 1);
        assert_eq!(memory.working_memory().len(), 1);
    }

    #[test]
    fn test_semantic_facts() {
        let memory = ConversationMemory::default();

        memory.add_fact("customer_name", "Rajesh Kumar", 0.95);
        memory.add_fact("loan_amount", "500000", 0.9);

        assert!(memory.get_fact("customer_name").is_some());
        assert_eq!(memory.get_fact("customer_name").unwrap().value, "Rajesh Kumar");
    }

    #[test]
    fn test_context_generation() {
        let memory = ConversationMemory::default();

        memory.add_fact("customer_name", "Rajesh", 0.9);
        memory.add(MemoryEntry {
            role: "user".to_string(),
            content: "I want a gold loan".to_string(),
            timestamp_ms: 100,
            stage: None,
            intents: vec!["loan_inquiry".to_string()],
            entities: HashMap::new(),
        });

        let context = memory.get_context();
        assert!(context.contains("Rajesh"));
    }
}
