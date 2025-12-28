//! Conversation Memory
//!
//! Hierarchical memory system:
//! - Working memory: Recent turns
//! - Episodic memory: Summarized past segments
//! - Semantic memory: Key facts and entities

use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use voice_agent_core::{Turn, TurnRole};

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
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            working_memory_size: 8,
            summarization_threshold: 6,
            max_episodic_summaries: 10,
            semantic_memory_enabled: true,
        }
    }
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
        }
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

    /// Create episodic summary from turns
    fn create_episodic_summary(&self, entries: Vec<MemoryEntry>) {
        if entries.is_empty() {
            return;
        }

        let start_ms = entries.first().map(|e| e.timestamp_ms).unwrap_or(0);
        let end_ms = entries.last().map(|e| e.timestamp_ms).unwrap_or(0);

        // Simple summary (in production, use LLM)
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
