//! Agentic RAG Multi-Step Retrieval
//!
//! Implements an iterative retrieval flow that:
//! 1. Performs initial hybrid search
//! 2. Checks if results are sufficient using relevance scoring
//! 3. Rewrites query and re-retrieves if needed
//! 4. Returns reranked results
//!
//! ## P1 FIX: Agentic RAG Implementation
//!
//! This module was added to implement the multi-step retrieval flow
//! that was previously marked as "NOT IMPLEMENTED" in the RAG plan.

use std::sync::Arc;

use crate::{
    HybridRetriever, RagError, RerankerConfig, RetrieverConfig, SearchResult, VectorStore,
};

use voice_agent_llm::{LlmBackend, Message, Role};

/// Configuration for agentic RAG
#[derive(Debug, Clone)]
pub struct AgenticRagConfig {
    /// Minimum sufficiency score to skip rewrite (0.0-1.0)
    pub sufficiency_threshold: f32,

    /// Maximum query rewrite iterations
    pub max_iterations: usize,

    /// Enable/disable agentic flow (fallback to single-shot)
    pub enabled: bool,

    /// Top-k for initial retrieval (before reranking)
    pub initial_top_k: usize,

    /// Final top-k after reranking
    pub final_top_k: usize,
}

impl Default for AgenticRagConfig {
    fn default() -> Self {
        // P6 FIX: Use centralized constants
        use voice_agent_config::constants::rag;

        Self {
            sufficiency_threshold: rag::SUFFICIENCY_THRESHOLD as f32,
            max_iterations: 3,
            enabled: true,
            initial_top_k: 10,
            final_top_k: rag::DEFAULT_TOP_K,
        }
    }
}

/// P2-1 FIX: Renamed from ConversationContext to QueryContext to avoid
/// confusion with voice_agent_core::ConversationContext which has a different structure.
/// This type is specifically for RAG query rewriting and context.
#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    /// Summary of the conversation so far
    pub summary: String,
    /// Current conversation stage
    pub stage: Option<String>,
    /// Extracted entities from conversation
    pub entities: Vec<(String, String)>,
}

/// Type alias for backwards compatibility
#[deprecated(since = "0.2.0", note = "Use QueryContext instead")]
pub type ConversationContext = QueryContext;

/// Result from agentic retrieval
#[derive(Debug, Clone)]
pub struct AgenticSearchResult {
    /// Final search results
    pub results: Vec<SearchResult>,
    /// Number of retrieval iterations performed
    pub iterations: usize,
    /// Whether query was rewritten
    pub query_rewritten: bool,
    /// Final query used (may differ from original if rewritten)
    pub final_query: String,
    /// Sufficiency score of final results
    pub sufficiency_score: f32,
}

/// Agentic retriever with multi-step refinement
pub struct AgenticRetriever {
    config: AgenticRagConfig,
    retriever: HybridRetriever,
    query_rewriter: Option<QueryRewriter>,
    sufficiency_checker: SufficiencyChecker,
}

impl AgenticRetriever {
    /// Create a new agentic retriever
    pub fn new(config: AgenticRagConfig) -> Self {
        let retriever = HybridRetriever::new(
            RetrieverConfig {
                dense_top_k: config.initial_top_k,
                final_top_k: config.final_top_k,
                ..RetrieverConfig::default()
            },
            RerankerConfig::default(),
        );

        Self {
            config,
            retriever,
            query_rewriter: None,
            sufficiency_checker: SufficiencyChecker::new(),
        }
    }

    /// Create with custom retriever
    pub fn with_retriever(config: AgenticRagConfig, retriever: HybridRetriever) -> Self {
        Self {
            config,
            retriever,
            query_rewriter: None,
            sufficiency_checker: SufficiencyChecker::new(),
        }
    }

    /// Set LLM for query rewriting
    pub fn with_llm(mut self, llm: Arc<dyn LlmBackend>) -> Self {
        self.query_rewriter = Some(QueryRewriter::new(llm));
        self
    }

    /// Multi-step retrieval with configurable complexity
    ///
    /// This implements the agentic RAG flow:
    /// 1. Initial hybrid retrieval
    /// 2. Check sufficiency of results
    /// 3. If insufficient and LLM available, rewrite query
    /// 4. Re-retrieve with rewritten query
    /// 5. Repeat up to max_iterations
    /// 6. Return final results
    pub async fn search(
        &self,
        query: &str,
        vector_store: &VectorStore,
        context: Option<&QueryContext>,
    ) -> Result<AgenticSearchResult, RagError> {
        // Fast path: single-shot if agentic disabled
        if !self.config.enabled {
            let results = self.retriever.search(query, vector_store, None).await?;
            return Ok(AgenticSearchResult {
                sufficiency_score: self.sufficiency_checker.score(&results, query),
                results,
                iterations: 1,
                query_rewritten: false,
                final_query: query.to_string(),
            });
        }

        // Step 1: Initial retrieval
        let mut results = self.retriever.search(query, vector_store, None).await?;
        let mut current_query = query.to_string();
        let mut iterations = 1;
        let mut query_rewritten = false;

        // Step 2-4: Iterative refinement
        for iteration in 0..self.config.max_iterations {
            // Check sufficiency
            let score = self.sufficiency_checker.score(&results, &current_query);

            if score >= self.config.sufficiency_threshold {
                tracing::debug!(
                    iteration = iteration + 1,
                    score,
                    "Sufficiency threshold met, stopping iteration"
                );
                return Ok(AgenticSearchResult {
                    results,
                    iterations,
                    query_rewritten,
                    final_query: current_query,
                    sufficiency_score: score,
                });
            }

            // Rewrite query if we have a rewriter
            if let Some(ref rewriter) = self.query_rewriter {
                let default_ctx = QueryContext::default();
                let ctx = context.unwrap_or(&default_ctx);

                match rewriter.rewrite(&current_query, &results, ctx).await {
                    Ok(new_query) => {
                        if new_query != current_query && !new_query.is_empty() {
                            tracing::debug!(
                                iteration = iteration + 1,
                                old_query = %current_query,
                                new_query = %new_query,
                                "Query rewritten"
                            );

                            current_query = new_query;
                            query_rewritten = true;

                            // Re-retrieve with new query
                            results = self
                                .retriever
                                .search(&current_query, vector_store, None)
                                .await?;
                            iterations += 1;
                        } else {
                            tracing::debug!(
                                iteration = iteration + 1,
                                "Query rewriter returned same/empty query, stopping"
                            );
                            break;
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            iteration = iteration + 1,
                            error = %e,
                            "Query rewriting failed, using current results"
                        );
                        break;
                    },
                }
            } else {
                tracing::debug!("No query rewriter available, using single-shot results");
                break; // No rewriter, can't improve
            }
        }

        // Return final results
        let final_score = self.sufficiency_checker.score(&results, &current_query);
        Ok(AgenticSearchResult {
            results,
            iterations,
            query_rewritten,
            final_query: current_query,
            sufficiency_score: final_score,
        })
    }

    /// Get the underlying retriever
    pub fn retriever(&self) -> &HybridRetriever {
        &self.retriever
    }

    /// Check if agentic mode is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Check if query rewriting is available
    pub fn has_query_rewriter(&self) -> bool {
        self.query_rewriter.is_some()
    }
}

/// Checks if retrieved results are sufficient to answer the query
pub struct SufficiencyChecker {
    /// Minimum number of results for sufficiency
    min_results: usize,
    /// Minimum average score for sufficiency
    min_avg_score: f32,
}

impl SufficiencyChecker {
    /// Create a new sufficiency checker
    /// P6 FIX: Use centralized constants for consistency
    pub fn new() -> Self {
        use voice_agent_config::constants::rag;

        Self {
            min_results: 1,
            min_avg_score: rag::SUFFICIENCY_MIN_AVG_SCORE as f32,
        }
    }

    /// Create with custom thresholds
    pub fn with_thresholds(min_results: usize, min_avg_score: f32) -> Self {
        Self {
            min_results,
            min_avg_score,
        }
    }

    /// Score the sufficiency of results for a query
    ///
    /// Returns a score between 0.0 and 1.0:
    /// - 0.0: No results or very low relevance
    /// - 1.0: High confidence results are sufficient
    ///
    /// The score is based on:
    /// 1. Number of results (more is better, up to a point)
    /// 2. Average retrieval score of top results
    /// 3. Score distribution (consistent scores are better)
    pub fn score(&self, results: &[SearchResult], _query: &str) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        // Take top 3 results for scoring
        let top_results: Vec<&SearchResult> = results.iter().take(3).collect();

        if top_results.len() < self.min_results {
            return 0.0;
        }

        // Calculate average score
        let avg_score: f32 =
            top_results.iter().map(|r| r.score).sum::<f32>() / top_results.len() as f32;

        if avg_score < self.min_avg_score {
            return avg_score / self.min_avg_score * 0.5; // Scale up to 0.5
        }

        // Calculate score spread (lower is better - consistent results)
        let max_score = top_results.iter().map(|r| r.score).fold(0.0f32, f32::max);
        let min_score = top_results.iter().map(|r| r.score).fold(f32::MAX, f32::min);
        let spread = max_score - min_score;
        let consistency_bonus = if spread < 0.2 { 0.1 } else { 0.0 };

        // Normalize to 0.0-1.0 range

        (avg_score.min(1.0) + consistency_bonus).min(1.0)
    }
}

impl Default for SufficiencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// P1 FIX: LLM-based Sufficiency Checking
// =============================================================================

/// Configuration for LLM-based sufficiency checking
#[derive(Debug, Clone)]
pub struct LlmSufficiencyConfig {
    /// Minimum coverage score (0.0-1.0) to consider sufficient
    pub min_coverage: f32,
    /// Maximum tokens for LLM evaluation
    pub max_eval_tokens: usize,
    /// Temperature for evaluation (lower = more deterministic)
    pub temperature: f32,
    /// Number of documents to send for evaluation
    pub top_k_for_eval: usize,
    /// Whether to include heuristic score in final decision
    pub use_heuristic_fallback: bool,
}

impl Default for LlmSufficiencyConfig {
    fn default() -> Self {
        Self {
            min_coverage: 0.7,
            max_eval_tokens: 150,
            temperature: 0.1,
            top_k_for_eval: 5,
            use_heuristic_fallback: true,
        }
    }
}

/// P1 FIX: LLM-based sufficiency evaluation result
#[derive(Debug, Clone)]
pub struct SufficiencyEvaluation {
    /// Whether documents are sufficient to answer the query
    pub sufficient: bool,
    /// Coverage score (0.0-1.0) - how well documents cover the query
    pub coverage: f32,
    /// What information is missing (if any)
    pub missing: Option<String>,
    /// Suggested refined query (if coverage is low)
    pub refined_query: Option<String>,
    /// Confidence in the evaluation
    pub confidence: f32,
    /// Method used for evaluation (heuristic or llm)
    pub method: String,
}

impl Default for SufficiencyEvaluation {
    fn default() -> Self {
        Self {
            sufficient: false,
            coverage: 0.0,
            missing: None,
            refined_query: None,
            confidence: 0.5,
            method: "heuristic".to_string(),
        }
    }
}

/// P1 FIX: LLM-enhanced sufficiency checker
///
/// Uses LLM to evaluate whether retrieved documents can answer the query.
/// Falls back to heuristic scoring if LLM is not available or fails.
pub struct LlmSufficiencyChecker {
    llm: Option<Arc<dyn LlmBackend>>,
    config: LlmSufficiencyConfig,
    heuristic_checker: SufficiencyChecker,
}

impl LlmSufficiencyChecker {
    /// Create a new LLM sufficiency checker with heuristic fallback only
    pub fn new() -> Self {
        Self {
            llm: None,
            config: LlmSufficiencyConfig::default(),
            heuristic_checker: SufficiencyChecker::new(),
        }
    }

    /// Create with LLM backend for enhanced evaluation
    pub fn with_llm(llm: Arc<dyn LlmBackend>) -> Self {
        Self {
            llm: Some(llm),
            config: LlmSufficiencyConfig::default(),
            heuristic_checker: SufficiencyChecker::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(llm: Option<Arc<dyn LlmBackend>>, config: LlmSufficiencyConfig) -> Self {
        Self {
            llm,
            config,
            heuristic_checker: SufficiencyChecker::new(),
        }
    }

    /// Evaluate sufficiency of documents for a query
    ///
    /// Uses LLM if available, otherwise falls back to heuristic scoring.
    pub async fn evaluate(
        &self,
        query: &str,
        results: &[SearchResult],
    ) -> Result<SufficiencyEvaluation, RagError> {
        // Quick path: empty results
        if results.is_empty() {
            return Ok(SufficiencyEvaluation {
                sufficient: false,
                coverage: 0.0,
                missing: Some("No documents retrieved".to_string()),
                refined_query: Some(query.to_string()),
                confidence: 1.0,
                method: "empty_check".to_string(),
            });
        }

        // Get heuristic score first
        let heuristic_score = self.heuristic_checker.score(results, query);

        // If no LLM, return heuristic result
        let llm = match &self.llm {
            Some(llm) => llm,
            None => {
                return Ok(SufficiencyEvaluation {
                    sufficient: heuristic_score >= self.config.min_coverage,
                    coverage: heuristic_score,
                    missing: None,
                    refined_query: None,
                    confidence: 0.6, // Lower confidence for heuristic
                    method: "heuristic".to_string(),
                });
            },
        };

        // Build document context for LLM
        let doc_context = results
            .iter()
            .take(self.config.top_k_for_eval)
            .enumerate()
            .map(|(i, r)| format!("[Doc {}] {}", i + 1, Self::truncate_text(&r.content, 300)))
            .collect::<Vec<_>>()
            .join("\n\n");

        let prompt = format!(
            r#"You are evaluating whether retrieved documents can answer a user query about gold loans.

User Query: "{query}"

Retrieved Documents:
{documents}

Evaluate if these documents provide sufficient information to answer the query.

Respond in JSON format:
{{
    "sufficient": true/false,
    "coverage": 0.0-1.0,
    "missing": "description of missing information, or null if sufficient",
    "refined_query": "improved search query if coverage < 0.7, or null if sufficient"
}}

IMPORTANT:
- "sufficient" should be true only if documents directly address the query
- "coverage" represents how much of the query can be answered (0.0 = none, 1.0 = fully)
- For gold loan queries, consider: rates, eligibility, process, documents required, benefits
- Only suggest refined_query if documents are insufficient

JSON response:"#,
            query = query,
            documents = doc_context,
        );

        let messages = vec![Message {
            role: Role::User,
            content: prompt,
            name: None,
            tool_call_id: None,
        }];

        // Call LLM for evaluation
        match llm.generate(&messages).await {
            Ok(response) => {
                // Parse JSON response
                match self.parse_evaluation_response(&response.text) {
                    Ok(mut eval) => {
                        eval.method = "llm".to_string();
                        eval.confidence = 0.85;

                        // Blend with heuristic if configured
                        if self.config.use_heuristic_fallback {
                            eval.coverage = eval.coverage * 0.7 + heuristic_score * 0.3;
                            eval.sufficient = eval.coverage >= self.config.min_coverage;
                        }

                        Ok(eval)
                    },
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to parse LLM evaluation, using heuristic");
                        Ok(SufficiencyEvaluation {
                            sufficient: heuristic_score >= self.config.min_coverage,
                            coverage: heuristic_score,
                            missing: None,
                            refined_query: None,
                            confidence: 0.6,
                            method: "heuristic_fallback".to_string(),
                        })
                    },
                }
            },
            Err(e) => {
                tracing::warn!(error = %e, "LLM evaluation failed, using heuristic");
                Ok(SufficiencyEvaluation {
                    sufficient: heuristic_score >= self.config.min_coverage,
                    coverage: heuristic_score,
                    missing: None,
                    refined_query: None,
                    confidence: 0.6,
                    method: "heuristic_fallback".to_string(),
                })
            },
        }
    }

    /// Quick heuristic check (no LLM call)
    pub fn quick_check(&self, query: &str, results: &[SearchResult]) -> f32 {
        self.heuristic_checker.score(results, query)
    }

    /// Parse the JSON response from LLM
    fn parse_evaluation_response(&self, response: &str) -> Result<SufficiencyEvaluation, RagError> {
        // Try to extract JSON from response (may have extra text)
        let json_str = if let Some(start) = response.find('{') {
            if let Some(end) = response.rfind('}') {
                &response[start..=end]
            } else {
                response
            }
        } else {
            response
        };

        #[derive(serde::Deserialize)]
        struct LlmResponse {
            sufficient: bool,
            coverage: f32,
            missing: Option<String>,
            refined_query: Option<String>,
        }

        let parsed: LlmResponse = serde_json::from_str(json_str)
            .map_err(|e| RagError::Search(format!("Failed to parse LLM response: {}", e)))?;

        Ok(SufficiencyEvaluation {
            sufficient: parsed.sufficient && parsed.coverage >= self.config.min_coverage,
            coverage: parsed.coverage.clamp(0.0, 1.0),
            missing: parsed.missing,
            refined_query: parsed.refined_query,
            confidence: 0.85,
            method: "llm".to_string(),
        })
    }

    /// Truncate text at word boundary
    fn truncate_text(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            return text.to_string();
        }

        // Find last word boundary within limit
        let truncated = &text[..max_len];
        if let Some(last_space) = truncated.rfind(|c: char| c.is_whitespace()) {
            format!("{}...", &text[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

impl Default for LlmSufficiencyChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Rewrites queries for better retrieval using LLM
pub struct QueryRewriter {
    llm: Arc<dyn LlmBackend>,
}

impl QueryRewriter {
    /// Create a new query rewriter with an LLM backend
    pub fn new(llm: Arc<dyn LlmBackend>) -> Self {
        Self { llm }
    }

    /// Rewrite a query to improve retrieval
    ///
    /// Takes the original query, current results, and conversation context
    /// to generate a more specific query that's likely to find relevant information.
    pub async fn rewrite(
        &self,
        query: &str,
        results: &[SearchResult],
        context: &QueryContext,
    ) -> Result<String, RagError> {
        // Build context from results
        let results_text = results
            .iter()
            .take(3)
            .map(|r| format!("- {}", Self::truncate(&r.content, 100)))
            .collect::<Vec<_>>()
            .join("\n");

        // Build context from conversation
        let context_text = if context.summary.is_empty() {
            "No prior conversation context.".to_string()
        } else {
            format!(
                "Stage: {}\nSummary: {}",
                context.stage.as_deref().unwrap_or("unknown"),
                context.summary
            )
        };

        let prompt = format!(
            r#"You are a query rewriting assistant for a gold loan customer service system.

The following query did not retrieve sufficient information:
Query: "{query}"

Top results retrieved (may not be relevant enough):
{results}

Conversation context:
{context}

Rewrite the query to be more specific and likely to find relevant gold loan information.
Focus on:
- Gold loan terms, rates, eligibility
- Kotak Bank specific information
- Customer concerns about switching lenders

Only output the rewritten query (in the same language as the original), nothing else.
If the query is already good, output it unchanged."#,
            query = query,
            results = results_text,
            context = context_text,
        );

        let messages = vec![Message {
            role: Role::User,
            content: prompt,
            name: None,
            tool_call_id: None,
        }];

        // Call LLM for rewriting
        let response = self
            .llm
            .generate(&messages)
            .await
            .map_err(|e| RagError::Search(format!("LLM query rewrite failed: {}", e)))?;

        let rewritten = response.text.trim().to_string();

        // Validate rewritten query
        if rewritten.is_empty() || rewritten.len() > 500 {
            return Ok(query.to_string()); // Return original if invalid
        }

        Ok(rewritten)
    }

    /// Truncate text to a maximum length at word boundary
    fn truncate(text: &str, max_len: usize) -> String {
        if text.len() <= max_len {
            return text.to_string();
        }

        let truncated = &text[..max_len];
        if let Some(last_space) = truncated.rfind(char::is_whitespace) {
            format!("{}...", &text[..last_space])
        } else {
            format!("{}...", truncated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        use voice_agent_config::constants::rag;

        let config = AgenticRagConfig::default();
        assert_eq!(config.max_iterations, 3);
        assert!(config.enabled);
        // P6 FIX: Use centralized constant for expected value
        assert!((config.sufficiency_threshold - rag::SUFFICIENCY_THRESHOLD as f32).abs() < 0.01);
        assert_eq!(config.final_top_k, rag::DEFAULT_TOP_K);
    }

    #[test]
    fn test_sufficiency_checker_empty() {
        let checker = SufficiencyChecker::new();
        assert_eq!(checker.score(&[], "test query"), 0.0);
    }

    #[test]
    fn test_sufficiency_checker_low_scores() {
        let checker = SufficiencyChecker::new();
        let results = vec![
            create_test_result("1", 0.1),
            create_test_result("2", 0.15),
            create_test_result("3", 0.12),
        ];
        let score = checker.score(&results, "test query");
        assert!(score < 0.5); // Low scores should give low sufficiency
    }

    #[test]
    fn test_sufficiency_checker_high_scores() {
        let checker = SufficiencyChecker::new();
        let results = vec![
            create_test_result("1", 0.9),
            create_test_result("2", 0.85),
            create_test_result("3", 0.88),
        ];
        let score = checker.score(&results, "test query");
        assert!(score > 0.8); // High, consistent scores should give high sufficiency
    }

    #[test]
    fn test_agentic_retriever_creation() {
        let config = AgenticRagConfig::default();
        let retriever = AgenticRetriever::new(config);

        assert!(retriever.is_enabled());
        assert!(!retriever.has_query_rewriter());
    }

    #[test]
    fn test_agentic_retriever_disabled() {
        let config = AgenticRagConfig {
            enabled: false,
            ..Default::default()
        };
        let retriever = AgenticRetriever::new(config);

        assert!(!retriever.is_enabled());
    }

    fn create_test_result(id: &str, score: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            content: format!("Test document {}", id),
            score,
            metadata: std::collections::HashMap::new(),
            source: crate::retriever::SearchSource::Dense,
            exit_layer: None,
        }
    }
}
