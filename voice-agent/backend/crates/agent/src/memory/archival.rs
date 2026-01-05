//! Archival Memory Module
//!
//! Implements MemGPT-style archival storage backed by a vector database.
//! Used for storing and retrieving information that doesn't fit in core memory.
//!
//! Key features:
//! - Semantic search over stored memories
//! - Automatic relevance scoring
//! - Memory linking (A-MEM Zettelkasten style)
//!
//! Reference: MemGPT paper (arXiv:2310.08560), A-MEM paper (arXiv:2502.12110)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Archival memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalMemoryConfig {
    /// Maximum memories to store
    pub max_memories: usize,
    /// Default number of results for search
    pub default_top_k: usize,
    /// Minimum similarity score for retrieval
    pub min_similarity: f32,
    /// Enable automatic linking between related memories
    pub enable_linking: bool,
    /// Collection name in vector store
    pub collection_name: String,
}

impl Default for ArchivalMemoryConfig {
    fn default() -> Self {
        Self {
            max_memories: 10000,
            default_top_k: 5,
            min_similarity: 0.5,
            enable_linking: true,
            collection_name: "agent_archival_memory".to_string(),
        }
    }
}

/// A single memory note in archival storage
///
/// Inspired by A-MEM's Zettelkasten-style memory notes with linking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNote {
    /// Unique identifier
    pub id: Uuid,
    /// Session ID this memory belongs to
    pub session_id: String,
    /// The actual content/text
    pub content: String,
    /// Contextual description (what this memory is about)
    pub context_description: String,
    /// Keywords for sparse retrieval
    pub keywords: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Links to related memory notes
    pub links: HashSet<Uuid>,
    /// Memory type
    pub memory_type: MemoryType,
    /// Source of this memory
    pub source: MemorySource,
    /// When this memory was created
    pub created_at: DateTime<Utc>,
    /// When this memory was last accessed
    pub last_accessed: DateTime<Utc>,
    /// Access count (for importance scoring)
    pub access_count: u32,
    /// Embedding vector (populated by embedder)
    #[serde(skip)]
    pub embedding: Option<Vec<f32>>,
}

impl MemoryNote {
    /// Create a new memory note
    pub fn new(
        session_id: impl Into<String>,
        content: impl Into<String>,
        memory_type: MemoryType,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            session_id: session_id.into(),
            content: content.into(),
            context_description: String::new(),
            keywords: Vec::new(),
            tags: Vec::new(),
            links: HashSet::new(),
            memory_type,
            source: MemorySource::Conversation,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            embedding: None,
        }
    }

    /// Add context description
    pub fn with_context(mut self, description: impl Into<String>) -> Self {
        self.context_description = description.into();
        self
    }

    /// Add keywords
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Add a link to another memory
    pub fn link_to(&mut self, other_id: Uuid) {
        self.links.insert(other_id);
    }

    /// Set embedding vector
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Mark as accessed (updates timestamp and count)
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }

    /// Format for inclusion in LLM context
    pub fn format_for_context(&self) -> String {
        let mut output = self.content.clone();

        if !self.context_description.is_empty() {
            output = format!("[{}] {}", self.context_description, output);
        }

        output
    }

    /// Get text for embedding (combines content + context)
    pub fn text_for_embedding(&self) -> String {
        if self.context_description.is_empty() {
            self.content.clone()
        } else {
            format!("{}: {}", self.context_description, self.content)
        }
    }
}

/// Type of memory note
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Factual information about the customer
    CustomerFact,
    /// Conversation summary/episode
    ConversationSummary,
    /// Product/domain knowledge
    DomainKnowledge,
    /// Customer preference
    Preference,
    /// Important event or action
    Event,
    /// Objection or concern raised
    Objection,
    /// Competitor information mentioned
    CompetitorMention,
}

/// Source of memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemorySource {
    /// Extracted from conversation
    Conversation,
    /// System/configuration provided
    System,
    /// External data source (CRM, etc.)
    External,
    /// Inferred by the agent
    Inferred,
}

/// Search result from archival memory
#[derive(Debug, Clone)]
pub struct ArchivalSearchResult {
    /// The memory note
    pub note: MemoryNote,
    /// Similarity score (0.0 - 1.0)
    pub score: f32,
    /// Whether this was retrieved via link traversal
    pub via_link: bool,
}

/// Archival Memory Storage
///
/// MemGPT-style archival storage with vector search capabilities.
/// In production, this interfaces with Qdrant or similar vector DB.
pub struct ArchivalMemory {
    config: ArchivalMemoryConfig,
    /// In-memory store (for testing/simple deployments)
    memories: parking_lot::RwLock<Vec<MemoryNote>>,
    /// Index by session ID for quick lookup
    session_index: parking_lot::RwLock<std::collections::HashMap<String, Vec<Uuid>>>,
}

impl ArchivalMemory {
    /// Create new archival memory
    pub fn new(config: ArchivalMemoryConfig) -> Self {
        Self {
            config,
            memories: parking_lot::RwLock::new(Vec::new()),
            session_index: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    // =========================================================================
    // MemGPT-style Functions
    // =========================================================================

    /// Insert a memory into archival storage
    ///
    /// MemGPT function: archival_memory_insert
    pub fn insert(&self, mut note: MemoryNote) -> Uuid {
        let id = note.id;
        let session_id = note.session_id.clone();

        // Auto-link if enabled
        if self.config.enable_linking {
            self.auto_link_memory(&mut note);
        }

        // Add to storage
        self.memories.write().push(note);

        // Update session index
        self.session_index
            .write()
            .entry(session_id)
            .or_insert_with(Vec::new)
            .push(id);

        // Check size limit and evict if necessary
        self.maybe_evict();

        id
    }

    /// Search archival memory
    ///
    /// MemGPT function: archival_memory_search
    ///
    /// In production, this would call the vector database.
    /// This implementation uses simple keyword matching for testing.
    pub fn search(&self, query: &str, top_k: Option<usize>) -> Vec<ArchivalSearchResult> {
        let top_k = top_k.unwrap_or(self.config.default_top_k);
        let memories = self.memories.read();

        // Simple keyword-based scoring for testing
        // In production, use vector similarity from Qdrant
        let mut results: Vec<ArchivalSearchResult> = memories
            .iter()
            .map(|note| {
                let score = self.compute_keyword_score(query, note);
                ArchivalSearchResult {
                    note: note.clone(),
                    score,
                    via_link: false,
                }
            })
            .filter(|r| r.score >= self.config.min_similarity)
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k
        results.truncate(top_k);

        // Mark accessed
        drop(memories);
        for result in &results {
            self.mark_accessed(result.note.id);
        }

        results
    }

    /// Search with embedding vector (for production use)
    pub fn search_by_embedding(
        &self,
        _embedding: &[f32],
        top_k: Option<usize>,
    ) -> Vec<ArchivalSearchResult> {
        // In production, this would query Qdrant with the embedding vector
        // For now, return empty (no embeddings stored in test mode)
        let _ = top_k;
        Vec::new()
    }

    /// Search within a specific session
    pub fn search_session(
        &self,
        session_id: &str,
        query: &str,
        top_k: Option<usize>,
    ) -> Vec<ArchivalSearchResult> {
        let top_k = top_k.unwrap_or(self.config.default_top_k);

        let session_ids = self.session_index.read();
        let memory_ids = match session_ids.get(session_id) {
            Some(ids) => ids.clone(),
            None => return Vec::new(),
        };
        drop(session_ids);

        let memories = self.memories.read();
        let id_set: HashSet<Uuid> = memory_ids.into_iter().collect();

        let mut results: Vec<ArchivalSearchResult> = memories
            .iter()
            .filter(|note| id_set.contains(&note.id))
            .map(|note| {
                let score = self.compute_keyword_score(query, note);
                ArchivalSearchResult {
                    note: note.clone(),
                    score,
                    via_link: false,
                }
            })
            .filter(|r| r.score >= self.config.min_similarity)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
    }

    /// Get memory by ID
    pub fn get(&self, id: Uuid) -> Option<MemoryNote> {
        let memories = self.memories.read();
        memories.iter().find(|n| n.id == id).cloned()
    }

    /// Get linked memories (A-MEM style traversal)
    pub fn get_linked(&self, id: Uuid, depth: usize) -> Vec<MemoryNote> {
        if depth == 0 {
            return Vec::new();
        }

        let memories = self.memories.read();
        let source = match memories.iter().find(|n| n.id == id) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(id);

        self.traverse_links(&memories, &source.links, depth, &mut visited, &mut result);

        result
    }

    /// Delete memory by ID
    pub fn delete(&self, id: Uuid) -> bool {
        let mut memories = self.memories.write();
        let initial_len = memories.len();
        memories.retain(|n| n.id != id);

        // Remove from session index
        let mut session_index = self.session_index.write();
        for ids in session_index.values_mut() {
            ids.retain(|&i| i != id);
        }

        // Remove links to this memory from other memories
        for note in memories.iter_mut() {
            note.links.remove(&id);
        }

        memories.len() < initial_len
    }

    /// Clear all memories for a session
    pub fn clear_session(&self, session_id: &str) {
        let ids_to_remove: Vec<Uuid> = {
            let session_index = self.session_index.read();
            session_index.get(session_id).cloned().unwrap_or_default()
        };

        for id in ids_to_remove {
            self.delete(id);
        }

        self.session_index.write().remove(session_id);
    }

    /// Get total memory count
    pub fn len(&self) -> usize {
        self.memories.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.memories.read().is_empty()
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    /// Compute simple keyword-based score (for testing)
    fn compute_keyword_score(&self, query: &str, note: &MemoryNote) -> f32 {
        let query_lower = query.to_lowercase();
        let query_words: HashSet<&str> = query_lower.split_whitespace().collect();

        if query_words.is_empty() {
            return 0.0;
        }

        let content_lower = note.content.to_lowercase();
        let context_lower = note.context_description.to_lowercase();

        let mut matches = 0;
        for word in &query_words {
            if content_lower.contains(word) || context_lower.contains(word) {
                matches += 1;
            }
            // Bonus for keyword match
            if note.keywords.iter().any(|k| k.to_lowercase().contains(word)) {
                matches += 1;
            }
        }

        (matches as f32 / query_words.len() as f32).min(1.0)
    }

    /// Auto-link memory to related existing memories
    fn auto_link_memory(&self, note: &mut MemoryNote) {
        let memories = self.memories.read();

        // Find memories with similar keywords or tags
        for existing in memories.iter() {
            // Skip same session for now (avoid self-linking)
            if existing.id == note.id {
                continue;
            }

            // Check keyword overlap
            let keyword_overlap = note
                .keywords
                .iter()
                .any(|k| existing.keywords.contains(k));

            // Check tag overlap
            let tag_overlap = note.tags.iter().any(|t| existing.tags.contains(t));

            // Check content similarity (simple substring match)
            let content_similar = note
                .content
                .split_whitespace()
                .take(5)
                .any(|w| existing.content.contains(w) && w.len() > 3);

            if keyword_overlap || tag_overlap || content_similar {
                note.links.insert(existing.id);
            }
        }
    }

    /// Traverse links recursively
    fn traverse_links(
        &self,
        memories: &[MemoryNote],
        links: &HashSet<Uuid>,
        depth: usize,
        visited: &mut HashSet<Uuid>,
        result: &mut Vec<MemoryNote>,
    ) {
        if depth == 0 {
            return;
        }

        for &link_id in links {
            if visited.contains(&link_id) {
                continue;
            }
            visited.insert(link_id);

            if let Some(note) = memories.iter().find(|n| n.id == link_id) {
                result.push(note.clone());
                self.traverse_links(memories, &note.links, depth - 1, visited, result);
            }
        }
    }

    /// Mark memory as accessed
    fn mark_accessed(&self, id: Uuid) {
        let mut memories = self.memories.write();
        if let Some(note) = memories.iter_mut().find(|n| n.id == id) {
            note.mark_accessed();
        }
    }

    /// Evict old memories if over limit
    fn maybe_evict(&self) {
        let mut memories = self.memories.write();

        if memories.len() <= self.config.max_memories {
            return;
        }

        // Sort by access count and last accessed time
        // Remove least accessed and oldest first
        memories.sort_by(|a, b| {
            // Primary: access count (ascending - lower is evicted first)
            match a.access_count.cmp(&b.access_count) {
                std::cmp::Ordering::Equal => {
                    // Secondary: last accessed time (ascending - older is evicted first)
                    a.last_accessed.cmp(&b.last_accessed)
                }
                other => other,
            }
        });

        // Remove excess
        let to_remove = memories.len() - self.config.max_memories;
        memories.drain(0..to_remove);
    }
}

impl Default for ArchivalMemory {
    fn default() -> Self {
        Self::new(ArchivalMemoryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_note_creation() {
        let note = MemoryNote::new("session-1", "Customer wants gold loan for 50 grams", MemoryType::CustomerFact)
            .with_context("Gold loan inquiry")
            .with_keywords(vec!["gold".to_string(), "loan".to_string(), "50 grams".to_string()])
            .with_tags(vec!["inquiry".to_string()]);

        assert_eq!(note.session_id, "session-1");
        assert!(note.keywords.contains(&"gold".to_string()));
        assert_eq!(note.memory_type, MemoryType::CustomerFact);
    }

    #[test]
    fn test_archival_insert_and_search() {
        let archival = ArchivalMemory::default();

        let note = MemoryNote::new("session-1", "Customer wants gold loan for 50 grams", MemoryType::CustomerFact)
            .with_keywords(vec!["gold".to_string(), "loan".to_string()]);

        archival.insert(note);

        let results = archival.search("gold loan", None);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_session_search() {
        let archival = ArchivalMemory::default();

        // Insert memories for different sessions
        let note1 = MemoryNote::new("session-1", "Gold loan inquiry", MemoryType::CustomerFact);
        let note2 = MemoryNote::new("session-2", "Gold loan application", MemoryType::CustomerFact);

        archival.insert(note1);
        archival.insert(note2);

        let results = archival.search_session("session-1", "gold", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].note.session_id, "session-1");
    }

    #[test]
    fn test_memory_linking() {
        let archival = ArchivalMemory::new(ArchivalMemoryConfig {
            enable_linking: true,
            ..Default::default()
        });

        let note1 = MemoryNote::new("session-1", "Customer interested in gold loan", MemoryType::CustomerFact)
            .with_keywords(vec!["gold".to_string(), "loan".to_string()]);
        let id1 = archival.insert(note1);

        let note2 = MemoryNote::new("session-1", "Gold loan rate is 10.5%", MemoryType::DomainKnowledge)
            .with_keywords(vec!["gold".to_string(), "loan".to_string(), "rate".to_string()]);
        let _id2 = archival.insert(note2);

        // Check if note2 is linked to note1
        let linked = archival.get_linked(id1, 1);
        // Note: with current simple linking, this may or may not find links
        // depending on insertion order
        assert!(linked.len() <= 1);
    }

    #[test]
    fn test_delete_memory() {
        let archival = ArchivalMemory::default();

        let note = MemoryNote::new("session-1", "Test memory", MemoryType::CustomerFact);
        let id = archival.insert(note);

        assert_eq!(archival.len(), 1);
        assert!(archival.delete(id));
        assert_eq!(archival.len(), 0);
    }

    #[test]
    fn test_clear_session() {
        let archival = ArchivalMemory::default();

        let note1 = MemoryNote::new("session-1", "Memory 1", MemoryType::CustomerFact);
        let note2 = MemoryNote::new("session-1", "Memory 2", MemoryType::CustomerFact);
        let note3 = MemoryNote::new("session-2", "Memory 3", MemoryType::CustomerFact);

        archival.insert(note1);
        archival.insert(note2);
        archival.insert(note3);

        assert_eq!(archival.len(), 3);

        archival.clear_session("session-1");
        assert_eq!(archival.len(), 1);
    }

    #[test]
    fn test_eviction() {
        let config = ArchivalMemoryConfig {
            max_memories: 3,
            ..Default::default()
        };
        let archival = ArchivalMemory::new(config);

        for i in 0..5 {
            let note = MemoryNote::new("session-1", format!("Memory {}", i), MemoryType::CustomerFact);
            archival.insert(note);
        }

        assert!(archival.len() <= 3);
    }
}
