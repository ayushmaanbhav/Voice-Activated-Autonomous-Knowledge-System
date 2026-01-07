//! RAG and Prefetch Methods for DomainAgent
//!
//! This module contains RAG-related functionality including:
//! - Prefetch on partial transcript
//! - Background prefetch
//! - Prefetch cache management

use voice_agent_rag::SearchResult;

use super::{DomainAgent, PrefetchEntry};

impl DomainAgent {
    /// P2 FIX: Prefetch RAG results based on partial transcript from STT
    ///
    /// This method should be called when VAD detects speech and STT provides
    /// partial transcripts. It triggers RAG prefetch in the background so
    /// results are ready when the full utterance completes.
    ///
    /// # Arguments
    /// * `partial_transcript` - Partial text from STT
    /// * `confidence` - STT confidence score (0.0 - 1.0)
    ///
    /// Returns true if prefetch was triggered, false if skipped (no RAG or low confidence)
    pub async fn prefetch_on_partial(&self, partial_transcript: &str, confidence: f32) -> bool {
        // Skip if RAG is disabled or components not available
        if !self.config.rag_enabled {
            return false;
        }

        // Phase 11: Use AgenticRetriever's underlying HybridRetriever for prefetch
        let (agentic_retriever, vector_store) =
            match (&self.agentic_retriever, &self.vector_store) {
                (Some(ar), Some(vs)) => (ar.clone(), vs.clone()),
                _ => return false,
            };

        // P4 FIX: Use timing strategy to determine if we should prefetch
        let stage = self.conversation.stage();
        let strategy = &self.config.rag_timing_strategy;

        // Check if strategy allows prefetch at this point
        if !strategy.should_prefetch(confidence, stage) {
            tracing::trace!(
                strategy = ?strategy,
                confidence = confidence,
                stage = ?stage,
                "Skipping prefetch - timing strategy declined"
            );
            return false;
        }

        // Don't prefetch for very short partials (strategy-aware minimum)
        if partial_transcript.split_whitespace().count() < strategy.min_words() {
            return false;
        }

        // Clone for async task
        let partial = partial_transcript.to_string();
        let cache = self.prefetch_cache.read().clone();

        // Skip if we already prefetched for similar query (strategy-aware TTL)
        let cache_ttl = strategy.cache_ttl_secs();
        if let Some(entry) = &cache {
            if entry.timestamp.elapsed().as_secs() < cache_ttl && partial.contains(&entry.query) {
                tracing::trace!("Skipping prefetch - similar query already cached");
                return false;
            }
        }

        tracing::debug!(
            partial = %partial,
            confidence = confidence,
            strategy = ?strategy,
            stage = ?stage,
            "Triggering RAG prefetch on partial transcript"
        );

        // Phase 11: Run prefetch using the underlying HybridRetriever from AgenticRetriever
        // This is faster than full agentic retrieval (no query rewriting)
        match agentic_retriever
            .retriever()
            .prefetch(&partial, confidence, &vector_store)
            .await
        {
            Ok(results) if !results.is_empty() => {
                tracing::debug!(count = results.len(), "RAG prefetch completed with results");
                // Store in cache
                *self.prefetch_cache.write() = Some(PrefetchEntry {
                    query: partial,
                    results,
                    timestamp: std::time::Instant::now(),
                });
                true
            }
            Ok(_) => {
                tracing::trace!("RAG prefetch returned no results");
                false
            }
            Err(e) => {
                tracing::warn!("RAG prefetch failed: {}", e);
                false
            }
        }
    }

    /// P2 FIX: Spawn prefetch as a background task (non-blocking)
    ///
    /// Use this when you want to trigger prefetch without waiting for results.
    /// The prefetch will run in the background and populate the cache.
    pub fn prefetch_background(&self, partial_transcript: String, confidence: f32) {
        if !self.config.rag_enabled {
            return;
        }

        // Phase 11: Use AgenticRetriever's underlying HybridRetriever for background prefetch
        let (agentic_retriever, vector_store) =
            match (&self.agentic_retriever, &self.vector_store) {
                (Some(ar), Some(vs)) => (ar.clone(), vs.clone()),
                _ => return,
            };

        if partial_transcript.split_whitespace().count() < 2 {
            return;
        }

        // Check cache under read lock, avoiding clone if possible
        {
            let cache = self.prefetch_cache.read();
            if let Some(entry) = &*cache {
                if entry.timestamp.elapsed().as_secs() < 2
                    && partial_transcript.contains(&entry.query)
                {
                    return;
                }
            }
        }

        // Spawn background prefetch task
        // Note: Results are not cached in background mode - use prefetch_on_partial() for caching
        // This is useful for warming up the retriever's internal caches
        tokio::spawn(async move {
            tracing::debug!(
                partial = %partial_transcript,
                confidence = confidence,
                "Background RAG prefetch triggered"
            );
            // Use underlying HybridRetriever for fast prefetch (no query rewriting)
            match agentic_retriever
                .retriever()
                .prefetch(&partial_transcript, confidence, &vector_store)
                .await
            {
                Ok(results) if !results.is_empty() => {
                    tracing::debug!(count = results.len(), "Background prefetch completed");
                    // Note: Results are not cached in background mode - use prefetch_on_partial for caching
                }
                Ok(_) => tracing::trace!("Background prefetch returned no results"),
                Err(e) => tracing::warn!("Background prefetch failed: {}", e),
            }
        });
    }

    /// P2 FIX: Get prefetched results if available and relevant
    ///
    /// Returns cached prefetch results if they match the query and are fresh.
    pub(super) fn get_prefetch_results(&self, query: &str) -> Option<Vec<SearchResult>> {
        let cache = self.prefetch_cache.read();
        if let Some(entry) = &*cache {
            // Check if cache is fresh (within 10 seconds)
            if entry.timestamp.elapsed().as_secs() > 10 {
                return None;
            }
            // Check if query is related to prefetched query
            // Simple check: query contains prefetch query or vice versa
            let query_lower = query.to_lowercase();
            let cached_lower = entry.query.to_lowercase();
            if query_lower.contains(&cached_lower) || cached_lower.contains(&query_lower) {
                tracing::debug!("Using prefetched RAG results");
                return Some(entry.results.clone());
            }
        }
        None
    }

    /// P2 FIX: Clear prefetch cache
    pub fn clear_prefetch_cache(&self) {
        *self.prefetch_cache.write() = None;
    }
}
