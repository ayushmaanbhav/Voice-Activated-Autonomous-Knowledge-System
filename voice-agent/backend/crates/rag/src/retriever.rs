//! Hybrid Retriever
//!
//! Combines dense and sparse search with RRF fusion and reranking.

use std::collections::HashMap;
use std::sync::Arc;
// P1 FIX: Use centralized RAG constants
use voice_agent_config::constants::rag;

use crate::embeddings::{EmbeddingConfig, SimpleEmbedder};
use crate::query_expansion::QueryExpander;
use crate::reranker::{EarlyExitReranker, RerankerConfig, SimpleScorer};
use crate::sparse_search::SparseIndex;
use crate::vector_store::{SearchFilter, VectorStore};
use crate::RagError;

/// Retriever configuration
#[derive(Debug, Clone)]
pub struct RetrieverConfig {
    /// Number of candidates from dense search
    pub dense_top_k: usize,
    /// Number of candidates from sparse search
    pub sparse_top_k: usize,
    /// Final number of results after reranking
    pub final_top_k: usize,
    /// Weight for dense scores in fusion (0.0 - 1.0)
    pub dense_weight: f32,
    /// RRF k parameter
    pub rrf_k: f32,
    /// Minimum score threshold
    pub min_score: f32,
    /// Enable reranking
    pub reranking_enabled: bool,
    /// P2 FIX: Minimum confidence threshold for prefetch (0.0 - 1.0)
    pub prefetch_confidence_threshold: f32,
    /// P2 FIX: Number of results to prefetch
    pub prefetch_top_k: usize,
    /// P1 FIX: Enable query expansion for Hindi/Hinglish synonyms
    pub query_expansion_enabled: bool,
}

impl Default for RetrieverConfig {
    fn default() -> Self {
        // P1 FIX: Use centralized constants for consistency with settings.rs
        Self {
            dense_top_k: 20,
            sparse_top_k: 20,
            final_top_k: rag::DEFAULT_TOP_K,
            dense_weight: rag::DENSE_WEIGHT as f32,
            rrf_k: 60.0,
            min_score: rag::MIN_SCORE as f32,
            reranking_enabled: true,
            prefetch_confidence_threshold: rag::PREFETCH_CONFIDENCE_THRESHOLD as f32,
            prefetch_top_k: 3,
            // P1 FIX: Enable query expansion by default for Hindi/Hinglish
            query_expansion_enabled: true,
        }
    }
}

/// P5 FIX: Convert from centralized RagConfig
impl From<&voice_agent_config::RagConfig> for RetrieverConfig {
    fn from(config: &voice_agent_config::RagConfig) -> Self {
        Self {
            dense_top_k: config.dense_top_k,
            sparse_top_k: config.sparse_top_k,
            final_top_k: config.final_top_k,
            dense_weight: config.dense_weight,
            rrf_k: config.rrf_k,
            min_score: config.min_score,
            reranking_enabled: config.reranking_enabled,
            prefetch_confidence_threshold: config.prefetch_confidence_threshold,
            prefetch_top_k: config.prefetch_top_k,
            // P1 FIX: Default to enabled (config crate can add field later)
            query_expansion_enabled: true,
        }
    }
}

/// Final search result
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document ID
    pub id: String,
    /// Document text
    pub text: String,
    /// Final score
    pub score: f32,
    /// Metadata
    pub metadata: HashMap<String, String>,
    /// Source (dense, sparse, or hybrid)
    pub source: SearchSource,
    /// Rerank exit layer (if early exit occurred)
    pub exit_layer: Option<usize>,
}

/// Search source
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchSource {
    Dense,
    Sparse,
    Hybrid,
}

/// Hybrid retriever combining dense and sparse search
pub struct HybridRetriever {
    config: RetrieverConfig,
    embedder: Option<Arc<SimpleEmbedder>>,
    sparse_index: Option<Arc<SparseIndex>>,
    #[allow(dead_code)] // Kept for API compatibility; may be used for lazy reranker init
    reranker_config: RerankerConfig,
    /// P0 FIX: Now properly using EarlyExitReranker when available
    reranker: Option<Arc<EarlyExitReranker>>,
    /// P1 FIX: Query expander for Hindi/Hinglish synonym expansion
    query_expander: Option<QueryExpander>,
}

impl HybridRetriever {
    /// Create a new hybrid retriever
    pub fn new(config: RetrieverConfig, reranker_config: RerankerConfig) -> Self {
        // P1 FIX: Create query expander if expansion is enabled
        let query_expander = if config.query_expansion_enabled {
            Some(QueryExpander::gold_loan())
        } else {
            None
        };

        Self {
            config,
            embedder: Some(Arc::new(SimpleEmbedder::new(EmbeddingConfig::default()))),
            sparse_index: None,
            reranker_config,
            reranker: None, // Will use SimpleScorer fallback if not set
            query_expander,
        }
    }

    /// P1 FIX: Set a custom query expander
    pub fn with_query_expander(mut self, expander: QueryExpander) -> Self {
        self.query_expander = Some(expander);
        self
    }

    /// P1 FIX: Disable query expansion
    pub fn without_query_expansion(mut self) -> Self {
        self.query_expander = None;
        self
    }

    /// P1 FIX: Expand query using configured expander
    fn expand_query(&self, query: &str) -> String {
        if let Some(ref expander) = self.query_expander {
            let expanded = expander.expand_to_string(query);
            tracing::debug!(
                original = query,
                expanded = %expanded,
                "Query expanded for better Hindi/Hinglish recall"
            );
            expanded
        } else {
            query.to_string()
        }
    }

    /// Set sparse index
    pub fn with_sparse_index(mut self, index: Arc<SparseIndex>) -> Self {
        self.sparse_index = Some(index);
        self
    }

    /// Set the EarlyExitReranker (P0 FIX: now actually used!)
    pub fn with_reranker(mut self, reranker: Arc<EarlyExitReranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Search with dense retrieval only
    ///
    /// P1 FIX: Embedding inference now runs in spawn_blocking to avoid blocking async runtime.
    pub async fn search_dense(
        &self,
        query: &str,
        vector_store: &VectorStore,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, RagError> {
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| RagError::Embedding("No embedder configured".to_string()))?;

        // P1 FIX: Run embedding in spawn_blocking to avoid blocking the async runtime
        // ONNX inference is CPU-intensive and should not block the tokio worker threads
        let embedder_clone = Arc::clone(embedder);
        let query_owned = query.to_string();
        let query_embedding =
            tokio::task::spawn_blocking(move || embedder_clone.embed(&query_owned))
                .await
                .map_err(|e| RagError::Embedding(format!("Embedding task failed: {}", e)))?;

        let results = vector_store
            .search(&query_embedding, self.config.dense_top_k, filter)
            .await?;

        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
                source: SearchSource::Dense,
                exit_layer: None,
            })
            .collect())
    }

    /// Search with sparse retrieval only
    pub fn search_sparse(&self, query: &str) -> Result<Vec<SearchResult>, RagError> {
        let sparse = self
            .sparse_index
            .as_ref()
            .ok_or_else(|| RagError::Search("No sparse index configured".to_string()))?;

        let results = sparse.search(query, Some(self.config.sparse_top_k))?;

        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                text: r.text,
                score: r.score,
                metadata: r.metadata,
                source: SearchSource::Sparse,
                exit_layer: None,
            })
            .collect())
    }

    /// Hybrid search with RRF fusion
    ///
    /// P1 FIX: Dense and sparse search now run in parallel using tokio::join!
    /// P1 FIX: Query expansion now applied for Hindi/Hinglish synonym matching
    pub async fn search(
        &self,
        query: &str,
        vector_store: &VectorStore,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SearchResult>, RagError> {
        // P1 FIX: Expand query for better Hindi/Hinglish recall
        let expanded_query = self.expand_query(query);

        // P1 FIX: Run dense and sparse search in parallel with expanded query
        let dense_future = self.search_dense(&expanded_query, vector_store, filter.clone());

        // P1 FIX: Sparse search now runs in spawn_blocking to avoid blocking async runtime
        // Tantivy search is CPU-intensive, so we move it off the async executor
        let sparse_index_clone = self.sparse_index.clone();
        let query_owned = expanded_query.clone();
        let sparse_top_k = self.config.sparse_top_k;

        let sparse_future = async move {
            if let Some(sparse) = sparse_index_clone {
                let results = tokio::task::spawn_blocking(move || {
                    sparse.search(&query_owned, Some(sparse_top_k))
                })
                .await
                .map_err(|e| RagError::Search(format!("Sparse search task failed: {}", e)))??;

                Ok::<Vec<SearchResult>, RagError>(
                    results
                        .into_iter()
                        .map(|r| SearchResult {
                            id: r.id,
                            text: r.text,
                            score: r.score,
                            metadata: r.metadata,
                            source: SearchSource::Sparse,
                            exit_layer: None,
                        })
                        .collect(),
                )
            } else {
                Ok(Vec::new())
            }
        };

        let (dense_result, sparse_result) = tokio::join!(dense_future, sparse_future);
        let dense_results = dense_result?;
        let sparse_results = sparse_result?;

        // Fuse results using RRF
        let fused = self.rrf_fusion(&dense_results, &sparse_results);

        // Apply reranking if enabled
        let final_results = if self.config.reranking_enabled {
            self.rerank(query, fused)?
        } else {
            fused
        };

        // Filter by min score and limit
        let results: Vec<SearchResult> = final_results
            .into_iter()
            .filter(|r| r.score >= self.config.min_score)
            .take(self.config.final_top_k)
            .collect();

        Ok(results)
    }

    /// Reciprocal Rank Fusion
    fn rrf_fusion(&self, dense: &[SearchResult], sparse: &[SearchResult]) -> Vec<SearchResult> {
        let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

        // Add dense results with RRF scores
        for (rank, result) in dense.iter().enumerate() {
            let rrf_score = 1.0 / (self.config.rrf_k + rank as f32 + 1.0);
            let weighted = rrf_score * self.config.dense_weight;

            scores
                .entry(result.id.clone())
                .and_modify(|(s, _)| *s += weighted)
                .or_insert((weighted, result.clone()));
        }

        // Add sparse results with RRF scores
        let sparse_weight = 1.0 - self.config.dense_weight;
        for (rank, result) in sparse.iter().enumerate() {
            let rrf_score = 1.0 / (self.config.rrf_k + rank as f32 + 1.0);
            let weighted = rrf_score * sparse_weight;

            scores
                .entry(result.id.clone())
                .and_modify(|(s, r)| {
                    *s += weighted;
                    r.source = SearchSource::Hybrid;
                })
                .or_insert((weighted, {
                    let mut r = result.clone();
                    r.source = SearchSource::Sparse;
                    r
                }));
        }

        // Sort by fused score
        let mut results: Vec<SearchResult> = scores
            .into_iter()
            .map(|(_, (score, mut result))| {
                result.score = score;
                result
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results
    }

    /// Rerank results using cross-encoder
    ///
    /// P0 FIX: Now properly uses EarlyExitReranker when available,
    /// with SimpleScorer as fallback when model is not loaded.
    fn rerank(
        &self,
        query: &str,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, RagError> {
        // Try to use EarlyExitReranker if available
        if let Some(ref reranker) = self.reranker {
            // Prepare documents for reranker
            let docs: Vec<(String, String)> = results
                .iter()
                .map(|r| (r.id.clone(), r.text.clone()))
                .collect();

            // Run reranking with early exit
            let rerank_results = reranker.rerank(query, &docs)?;

            // Map back to SearchResults with updated scores and exit layers
            let id_to_result: HashMap<String, SearchResult> =
                results.into_iter().map(|r| (r.id.clone(), r)).collect();

            let mut final_results: Vec<SearchResult> = rerank_results
                .into_iter()
                .filter_map(|rr| {
                    id_to_result.get(&rr.id).map(|orig| {
                        let mut r = orig.clone();
                        // Combine original score with rerank score
                        r.score = r.score * 0.3 + rr.score * 0.7;
                        r.exit_layer = rr.exit_layer;
                        r
                    })
                })
                .collect();

            final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            return Ok(final_results);
        }

        // Fallback to SimpleScorer when reranker not available
        tracing::debug!("EarlyExitReranker not available, using SimpleScorer fallback");

        let mut scored: Vec<(SearchResult, f32)> = results
            .into_iter()
            .map(|r| {
                let score = SimpleScorer::score(query, &r.text);
                (r, score)
            })
            .collect();

        // Sort by rerank score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Update scores
        Ok(scored
            .into_iter()
            .map(|(mut r, score)| {
                // Combine original and rerank scores
                r.score = r.score * 0.3 + score * 0.7;
                r
            })
            .collect())
    }

    /// Prefetch results based on partial transcript
    ///
    /// P2 FIX: Now uses configurable prefetch_confidence_threshold and prefetch_top_k
    pub async fn prefetch(
        &self,
        partial_transcript: &str,
        confidence: f32,
        vector_store: &VectorStore,
    ) -> Result<Vec<SearchResult>, RagError> {
        // P2 FIX: Use configurable confidence threshold
        if confidence < self.config.prefetch_confidence_threshold {
            return Ok(Vec::new());
        }

        // Extract likely intent/keywords from partial
        let keywords = Self::extract_keywords(partial_transcript);

        if keywords.is_empty() {
            return Ok(Vec::new());
        }

        // Do a quick search with fewer results
        let embedder = self
            .embedder
            .as_ref()
            .ok_or_else(|| RagError::Embedding("No embedder configured".to_string()))?;

        let query = keywords.join(" ");

        // P1 FIX: Run embedding in spawn_blocking to avoid blocking the async runtime
        let embedder_clone = Arc::clone(embedder);
        let embedding = tokio::task::spawn_blocking(move || embedder_clone.embed(&query))
            .await
            .map_err(|e| RagError::Embedding(format!("Embedding task failed: {}", e)))?;

        // P2 FIX: Use configurable prefetch_top_k
        let results = vector_store
            .search(&embedding, self.config.prefetch_top_k, None)
            .await?;

        Ok(results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                text: r.text,
                score: r.score * confidence, // Weight by transcript confidence
                metadata: r.metadata,
                source: SearchSource::Dense,
                exit_layer: None,
            })
            .collect())
    }

    /// Extract keywords from text
    fn extract_keywords(text: &str) -> Vec<String> {
        // Simple keyword extraction - filter stopwords
        let stopwords: std::collections::HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "i", "you", "we", "they",
            "it", "this", "that", "what", "which", "who", "whom", "whose", "to", "for", "in", "on",
            "at", "by", "with", "from", "and", "or", "but", "if", "then", "else", "main", "mujhe",
            "hai", "hain", "ka", "ki", "ke", "ko",
        ]
        .into_iter()
        .collect();

        text.to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2 && !stopwords.contains(w))
            .map(|w| w.to_string())
            .take(5)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RetrieverConfig::default();
        assert_eq!(config.final_top_k, 5);
        assert!(config.reranking_enabled);
    }

    #[test]
    fn test_rrf_fusion() {
        let retriever = HybridRetriever::new(RetrieverConfig::default(), RerankerConfig::default());

        let dense = vec![
            SearchResult {
                id: "1".to_string(),
                text: "doc1".to_string(),
                score: 0.9,
                metadata: HashMap::new(),
                source: SearchSource::Dense,
                exit_layer: None,
            },
            SearchResult {
                id: "2".to_string(),
                text: "doc2".to_string(),
                score: 0.8,
                metadata: HashMap::new(),
                source: SearchSource::Dense,
                exit_layer: None,
            },
        ];

        let sparse = vec![
            SearchResult {
                id: "2".to_string(),
                text: "doc2".to_string(),
                score: 0.85,
                metadata: HashMap::new(),
                source: SearchSource::Sparse,
                exit_layer: None,
            },
            SearchResult {
                id: "3".to_string(),
                text: "doc3".to_string(),
                score: 0.7,
                metadata: HashMap::new(),
                source: SearchSource::Sparse,
                exit_layer: None,
            },
        ];

        let fused = retriever.rrf_fusion(&dense, &sparse);

        // doc2 should be ranked higher (appears in both)
        assert_eq!(fused.len(), 3);
        let doc2_result = fused.iter().find(|r| r.id == "2").unwrap();
        assert_eq!(doc2_result.source, SearchSource::Hybrid);
    }

    #[test]
    fn test_extract_keywords() {
        let keywords = HybridRetriever::extract_keywords("What is the gold loan interest rate?");
        assert!(!keywords.is_empty());
        assert!(keywords.contains(&"gold".to_string()));
        assert!(keywords.contains(&"loan".to_string()));
    }
}
