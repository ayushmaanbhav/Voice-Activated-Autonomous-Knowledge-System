//! Adapter implementing core Retriever trait
//!
//! Bridges the RAG crate's HybridRetriever/AgenticRetriever with
//! the core Retriever trait interface.

use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::Mutex;

use voice_agent_core::{
    Retriever, RetrieveOptions, Document, ConversationContext as CoreContext,
    Result,
};

use crate::{
    HybridRetriever, AgenticRetriever, VectorStore, SearchResult,
    QueryExpander, DomainBooster,
    agentic::ConversationContext as RagContext,
};

/// Enhanced retriever implementing the core Retriever trait
///
/// Combines:
/// - HybridRetriever (dense + sparse + reranking)
/// - QueryExpander (synonym/transliteration expansion)
/// - DomainBooster (gold loan term boosting)
/// - Optional AgenticRetriever for multi-step refinement
pub struct EnhancedRetriever {
    /// Hybrid retriever for search
    hybrid: Arc<HybridRetriever>,
    /// Optional agentic retriever for multi-step
    agentic: Option<Arc<AgenticRetriever>>,
    /// Vector store for search
    vector_store: Arc<VectorStore>,
    /// Query expander
    expander: QueryExpander,
    /// Domain booster
    booster: DomainBooster,
    /// Prefetch cache
    prefetch_cache: Mutex<Option<PrefetchResult>>,
    /// Configuration
    config: EnhancedRetrieverConfig,
}

/// Configuration for enhanced retriever
#[derive(Debug, Clone)]
pub struct EnhancedRetrieverConfig {
    /// Enable query expansion
    pub query_expansion: bool,
    /// Enable domain boosting
    pub domain_boosting: bool,
    /// Enable agentic multi-step retrieval
    pub agentic_enabled: bool,
    /// Prefetch on partial transcript
    pub prefetch_enabled: bool,
    /// Name for logging
    pub name: String,
}

impl Default for EnhancedRetrieverConfig {
    fn default() -> Self {
        Self {
            query_expansion: true,
            domain_boosting: true,
            agentic_enabled: false,
            prefetch_enabled: true,
            name: "enhanced_retriever".to_string(),
        }
    }
}

/// Prefetch result cache
#[derive(Debug, Clone)]
struct PrefetchResult {
    /// Original partial transcript
    partial: String,
    /// Prefetched results
    results: Vec<Document>,
}

impl EnhancedRetriever {
    /// Create a new enhanced retriever
    pub fn new(
        hybrid: Arc<HybridRetriever>,
        vector_store: Arc<VectorStore>,
        config: EnhancedRetrieverConfig,
    ) -> Self {
        Self {
            hybrid,
            agentic: None,
            vector_store,
            expander: QueryExpander::gold_loan(),
            booster: DomainBooster::gold_loan(),
            prefetch_cache: Mutex::new(None),
            config,
        }
    }

    /// Create with agentic retriever
    pub fn with_agentic(mut self, agentic: Arc<AgenticRetriever>) -> Self {
        self.agentic = Some(agentic);
        self.config.agentic_enabled = true;
        self
    }

    /// Create with custom expander
    pub fn with_expander(mut self, expander: QueryExpander) -> Self {
        self.expander = expander;
        self
    }

    /// Create with custom booster
    pub fn with_booster(mut self, booster: DomainBooster) -> Self {
        self.booster = booster;
        self
    }

    /// Process query with expansion and boosting
    fn process_query(&self, query: &str) -> String {
        if self.config.query_expansion {
            self.expander.expand_to_string(query)
        } else {
            query.to_string()
        }
    }

    /// Apply domain boosting to results
    fn apply_boosting(&self, results: &mut [SearchResult], query: &str) {
        if !self.config.domain_boosting {
            return;
        }

        let boost_result = self.booster.boost(query);

        for result in results.iter_mut() {
            let doc_lower = result.text.to_lowercase();
            let mut doc_boost = 1.0f32;

            for matched in &boost_result.matched_terms {
                if doc_lower.contains(&matched.term.to_lowercase()) {
                    doc_boost *= matched.boost;
                }
            }

            result.score *= doc_boost;
        }

        // Re-sort after boosting
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    }

    /// Convert SearchResult to Document
    fn to_document(result: SearchResult) -> Document {
        let mut doc = Document::new(result.id, result.text, result.score);

        for (key, value) in result.metadata {
            doc = doc.with_metadata(key, value);
        }

        if let Some(layer) = result.exit_layer {
            doc = doc.with_metadata("exit_layer", layer as i64);
        }

        doc = doc.with_metadata("source", format!("{:?}", result.source));
        doc
    }

    /// Convert core ConversationContext to RAG ConversationContext
    fn to_rag_context(ctx: &CoreContext) -> RagContext {
        let summary = ctx
            .recent_turns
            .iter()
            .map(|t| format!("{}: {}", t.role, t.content))
            .collect::<Vec<_>>()
            .join("\n");

        RagContext {
            summary,
            stage: ctx.stage.clone(),
            entities: ctx
                .entities
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    /// Check if prefetch cache matches query
    fn check_prefetch_cache(&self, query: &str) -> Option<Vec<Document>> {
        let cache = self.prefetch_cache.lock();
        if let Some(ref prefetch) = *cache {
            // Check if the query starts with or contains the prefetch partial
            if query.to_lowercase().starts_with(&prefetch.partial.to_lowercase())
                || query.to_lowercase().contains(&prefetch.partial.to_lowercase())
            {
                return Some(prefetch.results.clone());
            }
        }
        None
    }
}

#[async_trait]
impl Retriever for EnhancedRetriever {
    async fn retrieve(
        &self,
        query: &str,
        options: &RetrieveOptions,
    ) -> Result<Vec<Document>> {
        // Check prefetch cache first
        if self.config.prefetch_enabled {
            if let Some(cached) = self.check_prefetch_cache(query) {
                tracing::debug!("Using prefetch cache for query: {}", query);
                return Ok(cached
                    .into_iter()
                    .filter(|d| d.score >= options.min_score)
                    .take(options.top_k)
                    .collect());
            }
        }

        // Process query with expansion
        let processed_query = self.process_query(query);

        // Perform hybrid search
        let mut results = self
            .hybrid
            .search(&processed_query, &self.vector_store, None)
            .await
            .map_err(|e| voice_agent_core::Error::Rag(e.to_string()))?;

        // Apply domain boosting
        self.apply_boosting(&mut results, query);

        // Convert and filter
        let documents: Vec<Document> = results
            .into_iter()
            .filter(|r| r.score >= options.min_score)
            .take(options.top_k)
            .map(Self::to_document)
            .collect();

        Ok(documents)
    }

    async fn retrieve_agentic(
        &self,
        query: &str,
        context: &CoreContext,
        max_iterations: usize,
    ) -> Result<Vec<Document>> {
        // Use agentic retriever if available
        if let Some(ref agentic) = self.agentic {
            if self.config.agentic_enabled {
                let rag_context = Self::to_rag_context(context);

                let result = agentic
                    .search(query, &self.vector_store, Some(&rag_context))
                    .await
                    .map_err(|e| voice_agent_core::Error::Rag(e.to_string()))?;

                tracing::debug!(
                    "Agentic retrieval: {} iterations, rewritten={}",
                    result.iterations,
                    result.query_rewritten
                );

                return Ok(result.results.into_iter().map(Self::to_document).collect());
            }
        }

        // Fall back to standard retrieval
        let options = RetrieveOptions::default().with_top_k(max_iterations * 5);
        self.retrieve(query, &options).await
    }

    fn prefetch(&self, partial_transcript: &str) {
        if !self.config.prefetch_enabled {
            return;
        }

        // Skip very short partials
        if partial_transcript.len() < 5 {
            return;
        }

        let hybrid = Arc::clone(&self.hybrid);
        let vector_store = Arc::clone(&self.vector_store);
        let partial = partial_transcript.to_string();
        let prefetch_cache = self.prefetch_cache.lock();

        // Skip if we already have this partial cached
        if let Some(ref cached) = *prefetch_cache {
            if cached.partial == partial {
                return;
            }
        }
        drop(prefetch_cache);

        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand_to_string(&partial);

        // Spawn async prefetch task
        let cache_ref = Arc::new(Mutex::new(None::<PrefetchResult>));
        let cache_clone = Arc::clone(&cache_ref);

        tokio::spawn(async move {
            match hybrid.prefetch(&expanded, 0.5, &vector_store).await {
                Ok(results) => {
                    let documents: Vec<Document> = results
                        .into_iter()
                        .map(|r| {
                            Document::new(r.id, r.text, r.score)
                                .with_metadata("prefetch", true)
                        })
                        .collect();

                    *cache_clone.lock() = Some(PrefetchResult {
                        partial,
                        results: documents,
                    });

                    tracing::debug!("Prefetch completed with {} results", cache_clone.lock().as_ref().map(|p| p.results.len()).unwrap_or(0));
                }
                Err(e) => {
                    tracing::warn!("Prefetch failed: {}", e);
                }
            }
        });
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn clear_cache(&self) {
        *self.prefetch_cache.lock() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = EnhancedRetrieverConfig::default();
        assert!(config.query_expansion);
        assert!(config.domain_boosting);
        assert!(config.prefetch_enabled);
    }

    #[test]
    fn test_query_expansion_processing() {
        // Test QueryExpander directly without needing full retriever setup
        let expander = QueryExpander::gold_loan();
        let processed = expander.expand_to_string("gold loan rate");

        // Should include expansions
        assert!(processed.len() > "gold loan rate".len());
    }

    #[test]
    fn test_to_document() {
        use crate::retriever::SearchSource;

        let result = SearchResult {
            id: "test-1".to_string(),
            text: "Test document".to_string(),
            score: 0.85,
            metadata: [("category".to_string(), "test".to_string())]
                .into_iter()
                .collect(),
            source: SearchSource::Hybrid,
            exit_layer: Some(3),
        };

        let doc = EnhancedRetriever::to_document(result);
        assert_eq!(doc.id, "test-1");
        assert_eq!(doc.score, 0.85);
        assert!(doc.metadata.contains_key("category"));
    }

    #[test]
    fn test_rag_context_conversion() {
        let mut ctx = CoreContext::default();
        ctx.add_turn("user", "What is the interest rate?");
        ctx.stage = Some("inquiry".to_string());

        let rag_ctx = EnhancedRetriever::to_rag_context(&ctx);
        assert!(rag_ctx.summary.contains("interest rate"));
        assert_eq!(rag_ctx.stage, Some("inquiry".to_string()));
    }
}
