//! RAG (Retrieval-Augmented Generation) with hybrid search
//!
//! Features:
//! - Dense vector search via Qdrant
//! - Sparse BM25 search via Tantivy
//! - Hybrid fusion with RRF
//! - Early-exit cross-encoder reranking
//! - Candle BERT embeddings (native Rust inference)
//! - FP16/BF16 quantization for reduced memory and faster inference
//! - LRU embedding cache for repeated queries
//! - Agentic RAG multi-step retrieval flow
//! - Query expansion with domain synonyms
//! - Domain-specific term boosting
//! - Cross-lingual query normalization (Hindi/Hinglish/English)
//! - Core Retriever trait implementation

pub mod embeddings;
pub mod vector_store;
pub mod sparse_search;
pub mod reranker;
pub mod retriever;
pub mod candle_embeddings;
pub mod cache;
pub mod agentic;
pub mod query_expansion;
pub mod domain_boost;
pub mod cross_lingual;
pub mod adapter;
// P1 FIX: Context sizing by conversation stage
pub mod context;
// P2 FIX: Knowledge base loading
pub mod knowledge_loader;

pub use embeddings::{Embedder, EmbeddingConfig, SimpleEmbedder};
pub use candle_embeddings::{
    CandleBertEmbedder, CandleEmbeddingConfig, PoolingStrategy,
    UnifiedEmbedder, QuantizationMode,
};
pub use cache::{EmbeddingCache, CachedEmbedder, CacheStats};
pub use vector_store::{VectorStore, VectorStoreConfig, VectorDistance};
pub use sparse_search::{SparseIndex, SparseConfig};
pub use reranker::{EarlyExitReranker, RerankerConfig, ExitStrategy};
pub use retriever::{HybridRetriever, RetrieverConfig, SearchResult};
pub use agentic::{
    AgenticRetriever, AgenticRagConfig, AgenticSearchResult,
    ConversationContext, SufficiencyChecker, QueryRewriter,
    // P1 FIX: LLM-based sufficiency checking
    LlmSufficiencyChecker, LlmSufficiencyConfig, SufficiencyEvaluation,
};
pub use query_expansion::{
    QueryExpander, QueryExpansionConfig, ExpandedQuery,
    WeightedTerm, TermSource, ExpansionStats,
};
pub use domain_boost::{
    DomainBooster, DomainBoostConfig, DomainTerm, TermCategory,
    BoostResult, MatchedTerm, QueryIntent,
};
pub use cross_lingual::{
    CrossLingualNormalizer, DetectedScript, LanguageDetection, NormalizedQuery,
};
pub use adapter::{EnhancedRetriever, EnhancedRetrieverConfig};
pub use context::{
    Stage, ContextBudget, ContextConfig, ContextManager,
    context_budget_for_stage,
};
pub use knowledge_loader::{KnowledgeLoader, KnowledgeDocument, KnowledgeFile};

use thiserror::Error;

/// RAG errors
#[derive(Error, Debug)]
pub enum RagError {
    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Vector store error: {0}")]
    VectorStore(String),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Reranker error: {0}")]
    Reranker(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Connection error: {0}")]
    Connection(String),
}

impl From<RagError> for voice_agent_core::Error {
    fn from(err: RagError) -> Self {
        voice_agent_core::Error::Rag(err.to_string())
    }
}
