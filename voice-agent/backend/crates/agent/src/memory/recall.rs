//! Recall Memory Module
//!
//! Implements MemGPT-style recall storage for conversation history search.
//! Stores the actual conversation turns and allows semantic search over them.
//!
//! Key features:
//! - Searchable conversation history
//! - FIFO queue with configurable size
//! - Conversation summarization triggers
//!
//! Reference: MemGPT paper (arXiv:2310.08560)

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Recall memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMemoryConfig {
    /// Maximum conversation turns to store
    pub max_turns: usize,
    /// Turns before triggering summarization
    pub summarization_threshold: usize,
    /// Number of recent turns to always include in context
    pub fifo_size: usize,
    /// Default search results count
    pub default_top_k: usize,
}

impl Default for RecallMemoryConfig {
    fn default() -> Self {
        Self {
            max_turns: 100,
            summarization_threshold: 10,
            fifo_size: 6,
            default_top_k: 5,
        }
    }
}

/// A conversation turn in recall memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Unique turn ID
    pub id: u64,
    /// Role: user, assistant, or system
    pub role: TurnRole,
    /// The content of the turn
    pub content: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Detected intents (if any)
    pub intents: Vec<String>,
    /// Extracted entities (key-value pairs)
    pub entities: Vec<(String, String)>,
    /// Conversation stage at this turn
    pub stage: Option<String>,
    /// Token estimate for this turn
    pub estimated_tokens: usize,
    /// Embedding vector (optional, for semantic search)
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl ConversationTurn {
    pub fn new(role: TurnRole, content: impl Into<String>) -> Self {
        let content = content.into();
        let estimated_tokens = estimate_tokens(&content);

        Self {
            id: 0, // Set by RecallMemory
            role,
            content,
            timestamp: Utc::now(),
            intents: Vec::new(),
            entities: Vec::new(),
            stage: None,
            estimated_tokens,
            embedding: None,
        }
    }

    /// Add detected intents
    pub fn with_intents(mut self, intents: Vec<String>) -> Self {
        self.intents = intents;
        self
    }

    /// Add extracted entities
    pub fn with_entities(mut self, entities: Vec<(String, String)>) -> Self {
        self.entities = entities;
        self
    }

    /// Set conversation stage
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    /// Set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Format for LLM context
    pub fn format_for_context(&self) -> String {
        format!("{}: {}", self.role.as_str(), self.content)
    }
}

/// Turn role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnRole {
    User,
    Assistant,
    System,
}

impl TurnRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            TurnRole::User => "user",
            TurnRole::Assistant => "assistant",
            TurnRole::System => "system",
        }
    }
}

impl From<&str> for TurnRole {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "user" => TurnRole::User,
            "assistant" => TurnRole::Assistant,
            "system" => TurnRole::System,
            _ => TurnRole::User,
        }
    }
}

/// Search result from recall memory
#[derive(Debug, Clone)]
pub struct RecallSearchResult {
    /// The turn
    pub turn: ConversationTurn,
    /// Relevance score
    pub score: f32,
    /// Context (turns before and after)
    pub context_before: Vec<ConversationTurn>,
    pub context_after: Vec<ConversationTurn>,
}

/// Recall Memory - Conversation History Storage
///
/// MemGPT-style recall storage with searchable conversation history.
pub struct RecallMemory {
    config: RecallMemoryConfig,
    /// All conversation turns
    turns: RwLock<VecDeque<ConversationTurn>>,
    /// Next turn ID
    next_id: RwLock<u64>,
    /// Turns pending summarization
    pending_summarization: RwLock<Vec<ConversationTurn>>,
}

impl RecallMemory {
    /// Create new recall memory
    pub fn new(config: RecallMemoryConfig) -> Self {
        Self {
            config,
            turns: RwLock::new(VecDeque::new()),
            next_id: RwLock::new(1),
            pending_summarization: RwLock::new(Vec::new()),
        }
    }

    // =========================================================================
    // MemGPT-style Functions
    // =========================================================================

    /// Add a conversation turn
    pub fn add_turn(&self, mut turn: ConversationTurn) -> u64 {
        let mut id = self.next_id.write();
        turn.id = *id;
        *id += 1;
        let turn_id = turn.id;

        let mut turns = self.turns.write();
        turns.push_back(turn);

        // Check if we need to trigger summarization
        if turns.len() > self.config.summarization_threshold {
            self.collect_for_summarization(&mut turns);
        }

        // Enforce max size
        while turns.len() > self.config.max_turns {
            if let Some(old) = turns.pop_front() {
                self.pending_summarization.write().push(old);
            }
        }

        turn_id
    }

    /// Search conversation history
    ///
    /// MemGPT function: conversation_search
    pub fn search(&self, query: &str, top_k: Option<usize>) -> Vec<RecallSearchResult> {
        let top_k = top_k.unwrap_or(self.config.default_top_k);
        let turns = self.turns.read();

        // Simple keyword-based scoring
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = turns
            .iter()
            .enumerate()
            .map(|(idx, turn)| {
                let score = compute_relevance(&query_words, turn);
                (idx, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k with context
        scored
            .into_iter()
            .take(top_k)
            .map(|(idx, score)| {
                let turn = turns[idx].clone();

                // Get context (1 turn before and after)
                let context_before = if idx > 0 {
                    vec![turns[idx - 1].clone()]
                } else {
                    Vec::new()
                };

                let context_after = if idx + 1 < turns.len() {
                    vec![turns[idx + 1].clone()]
                } else {
                    Vec::new()
                };

                RecallSearchResult {
                    turn,
                    score,
                    context_before,
                    context_after,
                }
            })
            .collect()
    }

    /// Search by embedding vector (for production use)
    pub fn search_by_embedding(
        &self,
        _embedding: &[f32],
        _top_k: Option<usize>,
    ) -> Vec<RecallSearchResult> {
        // In production, this would use vector similarity
        Vec::new()
    }

    /// Get recent FIFO turns (always included in context)
    pub fn get_fifo(&self) -> Vec<ConversationTurn> {
        let turns = self.turns.read();
        let start = turns.len().saturating_sub(self.config.fifo_size);
        turns.iter().skip(start).cloned().collect()
    }

    /// Get all turns
    pub fn get_all(&self) -> Vec<ConversationTurn> {
        self.turns.read().iter().cloned().collect()
    }

    /// Get turn by ID
    pub fn get_turn(&self, id: u64) -> Option<ConversationTurn> {
        self.turns.read().iter().find(|t| t.id == id).cloned()
    }

    /// Get turns in range
    pub fn get_range(&self, start_id: u64, end_id: u64) -> Vec<ConversationTurn> {
        self.turns
            .read()
            .iter()
            .filter(|t| t.id >= start_id && t.id <= end_id)
            .cloned()
            .collect()
    }

    /// Get turns pending summarization
    pub fn get_pending_summarization(&self) -> Vec<ConversationTurn> {
        std::mem::take(&mut *self.pending_summarization.write())
    }

    /// Check if there are turns pending summarization
    pub fn has_pending_summarization(&self) -> bool {
        !self.pending_summarization.read().is_empty()
    }

    /// Get total turns count
    pub fn len(&self) -> usize {
        self.turns.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.turns.read().is_empty()
    }

    /// Get total estimated tokens
    pub fn total_tokens(&self) -> usize {
        self.turns.read().iter().map(|t| t.estimated_tokens).sum()
    }

    /// Get FIFO token count
    pub fn fifo_tokens(&self) -> usize {
        let turns = self.turns.read();
        let start = turns.len().saturating_sub(self.config.fifo_size);
        turns.iter().skip(start).map(|t| t.estimated_tokens).sum()
    }

    /// Format FIFO for LLM context
    pub fn format_fifo_for_context(&self) -> String {
        self.get_fifo()
            .iter()
            .map(|t| t.format_for_context())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Clear all turns
    pub fn clear(&self) {
        self.turns.write().clear();
        self.pending_summarization.write().clear();
        *self.next_id.write() = 1;
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /// Collect old turns for summarization
    fn collect_for_summarization(&self, turns: &mut VecDeque<ConversationTurn>) {
        let to_summarize = self.config.summarization_threshold - self.config.fifo_size;

        if turns.len() <= self.config.fifo_size {
            return;
        }

        let mut pending = self.pending_summarization.write();
        for _ in 0..to_summarize.min(turns.len() - self.config.fifo_size) {
            if let Some(turn) = turns.pop_front() {
                pending.push(turn);
            }
        }
    }
}

impl Default for RecallMemory {
    fn default() -> Self {
        Self::new(RecallMemoryConfig::default())
    }
}

/// Estimate tokens for text (simple 4-chars-per-token estimate)
fn estimate_tokens(text: &str) -> usize {
    use unicode_segmentation::UnicodeSegmentation;

    let grapheme_count = text.graphemes(true).count();

    // Check for Devanagari (Hindi) - ~2 graphemes per token
    let devanagari_count = text
        .chars()
        .filter(|c| ('\u{0900}'..='\u{097F}').contains(c))
        .count();

    if devanagari_count > grapheme_count / 3 {
        grapheme_count.max(1) / 2
    } else {
        grapheme_count.max(1) / 4
    }
}

/// Compute relevance score for search
fn compute_relevance(query_words: &[&str], turn: &ConversationTurn) -> f32 {
    let content_lower = turn.content.to_lowercase();

    let mut matches = 0;
    for word in query_words {
        if content_lower.contains(word) {
            matches += 1;
        }
    }

    // Bonus for intent matches
    for intent in &turn.intents {
        let intent_lower = intent.to_lowercase();
        for word in query_words {
            if intent_lower.contains(word) {
                matches += 1;
            }
        }
    }

    // Bonus for entity matches
    for (key, value) in &turn.entities {
        let key_lower = key.to_lowercase();
        let value_lower = value.to_lowercase();
        for word in query_words {
            if key_lower.contains(word) || value_lower.contains(word) {
                matches += 1;
            }
        }
    }

    matches as f32 / (query_words.len() as f32 * 2.0) // Normalize to 0-1 range
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turn_creation() {
        let turn = ConversationTurn::new(TurnRole::User, "I want a gold loan")
            .with_intents(vec!["loan_inquiry".to_string()])
            .with_entities(vec![("product".to_string(), "gold_loan".to_string())]);

        assert_eq!(turn.role, TurnRole::User);
        assert!(turn.intents.contains(&"loan_inquiry".to_string()));
    }

    #[test]
    fn test_add_and_get_turns() {
        let recall = RecallMemory::default();

        let turn1 = ConversationTurn::new(TurnRole::User, "Hello");
        let turn2 = ConversationTurn::new(TurnRole::Assistant, "Hi! How can I help?");

        let id1 = recall.add_turn(turn1);
        let id2 = recall.add_turn(turn2);

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(recall.len(), 2);
    }

    #[test]
    fn test_get_fifo() {
        let config = RecallMemoryConfig {
            fifo_size: 2,
            ..Default::default()
        };
        let recall = RecallMemory::new(config);

        for i in 0..5 {
            let turn = ConversationTurn::new(TurnRole::User, format!("Message {}", i));
            recall.add_turn(turn);
        }

        let fifo = recall.get_fifo();
        assert_eq!(fifo.len(), 2);
        assert!(fifo[0].content.contains("3"));
        assert!(fifo[1].content.contains("4"));
    }

    #[test]
    fn test_search() {
        let recall = RecallMemory::default();

        recall.add_turn(ConversationTurn::new(TurnRole::User, "I want a gold loan"));
        recall.add_turn(ConversationTurn::new(
            TurnRole::Assistant,
            "Sure! How much gold do you have?",
        ));
        recall.add_turn(ConversationTurn::new(TurnRole::User, "About 50 grams of 22 karat"));

        let results = recall.search("gold", Some(10));
        assert!(!results.is_empty());

        // Should find both user messages about gold
        let gold_related: Vec<_> = results
            .iter()
            .filter(|r| r.turn.content.to_lowercase().contains("gold"))
            .collect();
        assert!(!gold_related.is_empty());
    }

    #[test]
    fn test_summarization_trigger() {
        let config = RecallMemoryConfig {
            summarization_threshold: 5,
            fifo_size: 2,
            max_turns: 100,
            ..Default::default()
        };
        let recall = RecallMemory::new(config);

        // Add more turns than threshold
        for i in 0..7 {
            let turn = ConversationTurn::new(TurnRole::User, format!("Message {}", i));
            recall.add_turn(turn);
        }

        // Should have pending summarization
        assert!(recall.has_pending_summarization());

        let pending = recall.get_pending_summarization();
        assert!(!pending.is_empty());
    }

    #[test]
    fn test_format_for_context() {
        let recall = RecallMemory::default();

        recall.add_turn(ConversationTurn::new(TurnRole::User, "Hello"));
        recall.add_turn(ConversationTurn::new(TurnRole::Assistant, "Hi there!"));

        let context = recall.format_fifo_for_context();
        assert!(context.contains("user: Hello"));
        assert!(context.contains("assistant: Hi there!"));
    }

    #[test]
    fn test_token_estimation() {
        // English text
        let english = "Hello, how can I help you today?";
        let english_tokens = estimate_tokens(english);
        assert!(english_tokens > 0);
        assert!(english_tokens < 15); // Should be around 8-10 tokens

        // Hindi text (should have different estimation)
        let hindi = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूं?";
        let hindi_tokens = estimate_tokens(hindi);
        assert!(hindi_tokens > 0);
    }
}
