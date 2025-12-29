//! Retrieval traits for RAG

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use crate::Result;

/// Retriever interface for RAG
///
/// Implementations:
/// - `HybridRetriever` - Dense + Sparse + Reranking
/// - `AgenticRetriever` - Multi-step with query rewriting
///
/// # Example
///
/// ```ignore
/// let retriever: Box<dyn Retriever> = Box::new(HybridRetriever::new(config));
/// let options = RetrieveOptions::default().with_top_k(5);
/// let docs = retriever.retrieve("gold loan eligibility", &options).await?;
/// for doc in docs {
///     println!("{}: {}", doc.score, doc.content);
/// }
/// ```
#[async_trait]
pub trait Retriever: Send + Sync + 'static {
    /// Retrieve relevant documents
    ///
    /// # Arguments
    /// * `query` - Search query
    /// * `options` - Retrieval options (top_k, filters, etc.)
    ///
    /// # Returns
    /// List of documents sorted by relevance (highest first)
    async fn retrieve(
        &self,
        query: &str,
        options: &RetrieveOptions,
    ) -> Result<Vec<Document>>;

    /// Agentic multi-step retrieval
    ///
    /// Iteratively refines query until sufficient documents found.
    /// Uses LLM to rewrite queries and evaluate results.
    ///
    /// # Arguments
    /// * `query` - Initial search query
    /// * `context` - Conversation context for better understanding
    /// * `max_iterations` - Maximum number of retrieval iterations
    ///
    /// # Returns
    /// List of relevant documents after refinement
    async fn retrieve_agentic(
        &self,
        query: &str,
        context: &ConversationContext,
        max_iterations: usize,
    ) -> Result<Vec<Document>>;

    /// Prefetch documents based on partial transcript
    ///
    /// Called on VAD speech detection to reduce latency.
    /// Results are cached and used when full transcript arrives.
    ///
    /// # Arguments
    /// * `partial_transcript` - Partial/in-progress transcript
    fn prefetch(&self, partial_transcript: &str);

    /// Get retriever name for logging
    fn name(&self) -> &str;

    /// Clear any cached results
    fn clear_cache(&self) {}
}

/// Retrieval options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieveOptions {
    /// Number of documents to return
    pub top_k: usize,
    /// Minimum similarity score (0.0 - 1.0)
    pub min_score: f32,
    /// Filter by metadata
    #[serde(default)]
    pub filters: Vec<MetadataFilter>,
    /// Include document metadata in results
    #[serde(default = "default_true")]
    pub include_metadata: bool,
    /// Enable reranking
    #[serde(default = "default_true")]
    pub rerank: bool,
    /// Reranking model to use (if different from default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_model: Option<String>,
}

fn default_true() -> bool {
    true
}

impl Default for RetrieveOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            min_score: 0.0,
            filters: Vec::new(),
            include_metadata: true,
            rerank: true,
            rerank_model: None,
        }
    }
}

impl RetrieveOptions {
    /// Set top_k
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set minimum score
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = min_score.clamp(0.0, 1.0);
        self
    }

    /// Add a metadata filter
    pub fn with_filter(mut self, filter: MetadataFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Disable reranking
    pub fn without_rerank(mut self) -> Self {
        self.rerank = false;
        self
    }
}

/// Metadata filter for retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataFilter {
    /// Field name
    pub field: String,
    /// Filter operation
    pub op: FilterOp,
    /// Value to compare
    pub value: serde_json::Value,
}

impl MetadataFilter {
    /// Create an equals filter
    pub fn eq(field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        Self {
            field: field.into(),
            op: FilterOp::Equals,
            value: value.into(),
        }
    }

    /// Create a contains filter
    pub fn contains(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            op: FilterOp::Contains,
            value: serde_json::Value::String(value.into()),
        }
    }
}

/// Filter operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FilterOp {
    Equals,
    NotEquals,
    Contains,
    GreaterThan,
    LessThan,
}

/// Retrieved document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID
    pub id: String,
    /// Document content
    pub content: String,
    /// Relevance score (0.0 - 1.0)
    pub score: f32,
    /// Document metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    /// Source/origin of the document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

impl Document {
    /// Create a new document
    pub fn new(id: impl Into<String>, content: impl Into<String>, score: f32) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            score,
            metadata: std::collections::HashMap::new(),
            source: None,
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// Conversation context for agentic retrieval
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversationContext {
    /// Recent conversation turns
    pub recent_turns: Vec<ConversationTurn>,
    /// Detected user intent
    pub intent: Option<String>,
    /// Current conversation stage
    pub stage: Option<String>,
    /// Extracted entities
    #[serde(default)]
    pub entities: std::collections::HashMap<String, String>,
}

/// A conversation turn
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
}

impl ConversationContext {
    /// Add a turn to the context
    pub fn add_turn(&mut self, role: impl Into<String>, content: impl Into<String>) {
        self.recent_turns.push(ConversationTurn {
            role: role.into(),
            content: content.into(),
        });
        // Keep only last 5 turns
        if self.recent_turns.len() > 5 {
            self.recent_turns.remove(0);
        }
    }

    /// Set intent
    pub fn with_intent(mut self, intent: impl Into<String>) -> Self {
        self.intent = Some(intent.into());
        self
    }

    /// Add an entity
    pub fn with_entity(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.entities.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieve_options_builder() {
        let options = RetrieveOptions::default()
            .with_top_k(10)
            .with_min_score(0.5)
            .with_filter(MetadataFilter::eq("category", "gold_loan"))
            .without_rerank();

        assert_eq!(options.top_k, 10);
        assert_eq!(options.min_score, 0.5);
        assert_eq!(options.filters.len(), 1);
        assert!(!options.rerank);
    }

    #[test]
    fn test_document_builder() {
        let doc = Document::new("doc-1", "Gold loan eligibility criteria", 0.95)
            .with_metadata("category", "eligibility")
            .with_source("knowledge_base.md");

        assert_eq!(doc.id, "doc-1");
        assert_eq!(doc.score, 0.95);
        assert!(doc.metadata.contains_key("category"));
        assert_eq!(doc.source, Some("knowledge_base.md".to_string()));
    }

    #[test]
    fn test_conversation_context() {
        let mut ctx = ConversationContext::default()
            .with_intent("check_eligibility")
            .with_entity("loan_amount", "50000");

        ctx.add_turn("user", "How much gold loan can I get?");
        ctx.add_turn("assistant", "That depends on your gold's purity and weight.");

        assert_eq!(ctx.recent_turns.len(), 2);
        assert_eq!(ctx.intent, Some("check_eligibility".to_string()));
    }
}
