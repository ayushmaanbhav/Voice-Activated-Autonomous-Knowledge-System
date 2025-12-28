//! RAG (Retrieval-Augmented Generation) with hybrid search
//!
//! Features:
//! - Dense vector search via Qdrant
//! - Sparse BM25 search via Tantivy
//! - Hybrid fusion with RRF
//! - Early-exit cross-encoder reranking
//! - Candle BERT embeddings (native Rust inference)

pub mod embeddings;
pub mod vector_store;
pub mod sparse_search;
pub mod reranker;
pub mod retriever;
pub mod candle_embeddings;

pub use embeddings::{Embedder, EmbeddingConfig, SimpleEmbedder};
pub use candle_embeddings::{CandleBertEmbedder, CandleEmbeddingConfig, PoolingStrategy, UnifiedEmbedder};
pub use vector_store::{VectorStore, VectorStoreConfig};
pub use sparse_search::{SparseIndex, SparseConfig};
pub use reranker::{EarlyExitReranker, RerankerConfig, ExitStrategy};
pub use retriever::{HybridRetriever, RetrieverConfig, SearchResult};

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
