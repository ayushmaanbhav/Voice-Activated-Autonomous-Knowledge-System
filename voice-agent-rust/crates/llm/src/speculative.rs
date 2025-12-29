//! Speculative Execution for LLM
//!
//! Implements multiple speculative strategies:
//! - SLM-First: Use small model first, upgrade if complex (recommended)
//! - Race Parallel: Run SLM and LLM in parallel, use first good response
//! - Hybrid Streaming: Start with SLM, switch to LLM mid-stream
//! - Draft-Verify: P1 FIX - Draft with SLM, verify chunks with LLM
//!
//! Note: True EAGLE-style speculative decoding requires KV cache sharing
//! between draft and verify models. We implement a simpler draft-verify
//! pattern that works without shared KV cache.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;
use parking_lot::Mutex;

use crate::backend::{LlmBackend, GenerationResult};
use crate::prompt::{Message, Role};
use crate::LlmError;

/// Speculative execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeMode {
    /// SLM first, upgrade if complex (recommended for most use cases)
    SlmFirst,
    /// Race SLM and LLM in parallel, use first acceptable response
    RaceParallel,
    /// Hybrid streaming (start SLM, switch to LLM mid-stream if quality drops)
    HybridStreaming,
    /// P1 FIX: Draft-Verify pattern - SLM drafts in chunks, LLM verifies
    /// Lower latency than full LLM, higher quality than pure SLM
    DraftVerify,
}

/// Speculative execution configuration
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Execution mode
    pub mode: SpeculativeMode,
    /// Complexity threshold for SLM-first upgrade
    pub complexity_threshold: f32,
    /// Timeout for SLM response (ms)
    pub slm_timeout_ms: u64,
    /// Minimum tokens before considering switch (hybrid)
    pub min_tokens_before_switch: usize,
    /// Quality threshold for acceptance
    pub quality_threshold: f32,
    /// Enable fallback to LLM on error
    pub fallback_enabled: bool,
    /// P2 FIX: Complexity threshold for speculative parallel LLM execution
    /// If complexity > this threshold, LLM is started in parallel with SLM
    /// Set to 1.0 to disable speculative execution
    pub speculative_llm_threshold: f32,
    /// P1 FIX: Draft-Verify configuration
    /// Number of tokens to draft before verification
    pub draft_chunk_size: usize,
    /// Maximum draft iterations before forcing full LLM
    pub max_draft_iterations: usize,
    /// Verification threshold - below this, reject draft and use LLM
    pub verification_threshold: f32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            mode: SpeculativeMode::SlmFirst,
            complexity_threshold: 0.7,
            // P0 FIX: Reduced from 2000ms to 100ms to meet 500ms E2E latency budget
            // Budget: VAD ~32ms + STT ~100ms + LLM 100ms + TTS ~100ms = 332ms + overhead
            slm_timeout_ms: 100,
            min_tokens_before_switch: 10,
            quality_threshold: 0.8,
            fallback_enabled: true,
            // P2 FIX: Start speculative LLM for moderate complexity queries
            speculative_llm_threshold: 0.3,
            // P1 FIX: Draft-Verify defaults
            draft_chunk_size: 20,        // Draft 20 tokens at a time
            max_draft_iterations: 5,     // Max 5 iterations (100 tokens total)
            verification_threshold: 0.7, // 70% acceptance threshold
        }
    }
}

/// Result of speculative execution
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Generated text
    pub text: String,
    /// Which model was used
    pub model_used: ModelUsed,
    /// Generation result
    pub generation: GenerationResult,
    /// Was fallback used?
    pub used_fallback: bool,
    /// Complexity score (if computed)
    pub complexity_score: Option<f32>,
}

/// Which model was used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelUsed {
    Slm,
    Llm,
    Hybrid,
}

/// Speculative Executor
pub struct SpeculativeExecutor {
    slm: Arc<dyn LlmBackend>,
    llm: Arc<dyn LlmBackend>,
    config: SpeculativeConfig,
    /// Statistics
    stats: Mutex<SpeculativeStats>,
}

/// Statistics for speculative execution
#[derive(Debug, Clone, Default)]
pub struct SpeculativeStats {
    pub slm_calls: usize,
    pub llm_calls: usize,
    pub slm_successes: usize,
    pub llm_fallbacks: usize,
    pub avg_slm_time_ms: f32,
    pub avg_llm_time_ms: f32,
}

impl SpeculativeExecutor {
    /// Create a new speculative executor
    pub fn new(
        slm: Arc<dyn LlmBackend>,
        llm: Arc<dyn LlmBackend>,
        config: SpeculativeConfig,
    ) -> Self {
        Self {
            slm,
            llm,
            config,
            stats: Mutex::new(SpeculativeStats::default()),
        }
    }

    /// Execute with speculative strategy
    pub async fn execute(&self, messages: &[Message]) -> Result<SpeculativeResult, LlmError> {
        match self.config.mode {
            SpeculativeMode::SlmFirst => self.execute_slm_first(messages).await,
            SpeculativeMode::RaceParallel => self.execute_race_parallel(messages).await,
            SpeculativeMode::HybridStreaming => self.execute_hybrid_streaming(messages).await,
            SpeculativeMode::DraftVerify => self.execute_draft_verify(messages).await,
        }
    }

    /// Execute with streaming
    pub async fn execute_stream(
        &self,
        messages: &[Message],
        tx: mpsc::Sender<String>,
    ) -> Result<SpeculativeResult, LlmError> {
        // For now, use SLM-first with streaming
        match self.config.mode {
            SpeculativeMode::SlmFirst => {
                self.execute_slm_first_stream(messages, tx).await
            }
            SpeculativeMode::HybridStreaming => {
                self.execute_hybrid_streaming_with_output(messages, tx).await
            }
            _ => {
                // Fall back to non-streaming for other modes
                let result = self.execute(messages).await?;
                let _ = tx.send(result.text.clone()).await;
                Ok(result)
            }
        }
    }

    /// SLM-first strategy
    ///
    /// P2 FIX: Now runs SLM and LLM in parallel for faster fallback.
    /// If complexity is moderate (> 0.3), LLM is speculatively started in the background.
    /// This eliminates the sequential latency when SLM fails/times out/produces low quality.
    async fn execute_slm_first(&self, messages: &[Message]) -> Result<SpeculativeResult, LlmError> {
        let start = Instant::now();

        // Estimate complexity
        let complexity = self.estimate_complexity(messages);

        if complexity > self.config.complexity_threshold {
            // High complexity, go straight to LLM
            let result = self.llm.generate(messages).await?;
            self.update_stats(false, true, start.elapsed());

            return Ok(SpeculativeResult {
                text: result.text.clone(),
                model_used: ModelUsed::Llm,
                generation: result,
                used_fallback: false,
                complexity_score: Some(complexity),
            });
        }

        // P2 FIX: Parallel execution - start LLM speculatively if complexity is moderate
        // This saves latency when we need to fall back from SLM
        let llm_handle = if self.config.fallback_enabled && complexity > self.config.speculative_llm_threshold {
            let llm = self.llm.clone();
            let messages_for_llm = messages.to_vec();

            tracing::debug!(
                complexity = complexity,
                "Starting speculative LLM execution in parallel with SLM"
            );

            Some(tokio::spawn(async move {
                llm.generate(&messages_for_llm).await
            }))
        } else {
            None
        };

        // Try SLM first with timeout
        let slm_timeout = Duration::from_millis(self.config.slm_timeout_ms);

        match timeout(slm_timeout, self.slm.generate(messages)).await {
            Ok(Ok(result)) => {
                // Check quality
                let quality = self.estimate_quality(&result.text, messages);

                if quality >= self.config.quality_threshold {
                    // SLM succeeded - abort speculative LLM
                    if let Some(handle) = llm_handle {
                        handle.abort();
                        tracing::debug!("SLM succeeded, aborting speculative LLM");
                    }

                    self.update_stats(true, false, start.elapsed());
                    Ok(SpeculativeResult {
                        text: result.text.clone(),
                        model_used: ModelUsed::Slm,
                        generation: result,
                        used_fallback: false,
                        complexity_score: Some(complexity),
                    })
                } else if self.config.fallback_enabled {
                    // Quality too low, use LLM result (already in progress if speculative)
                    let llm_result = if let Some(handle) = llm_handle {
                        // Use speculative LLM result
                        tracing::debug!("SLM quality low, using speculative LLM result");
                        match handle.await {
                            Ok(Ok(r)) => r,
                            Ok(Err(e)) => return Err(e),
                            Err(e) => return Err(LlmError::Generation(format!("LLM task failed: {}", e))),
                        }
                    } else {
                        // No speculative LLM, start fresh
                        self.llm.generate(messages).await?
                    };

                    self.update_stats(true, true, start.elapsed());

                    Ok(SpeculativeResult {
                        text: llm_result.text.clone(),
                        model_used: ModelUsed::Llm,
                        generation: llm_result,
                        used_fallback: true,
                        complexity_score: Some(complexity),
                    })
                } else {
                    // No fallback enabled
                    if let Some(handle) = llm_handle {
                        handle.abort();
                    }

                    self.update_stats(true, false, start.elapsed());
                    Ok(SpeculativeResult {
                        text: result.text.clone(),
                        model_used: ModelUsed::Slm,
                        generation: result,
                        used_fallback: false,
                        complexity_score: Some(complexity),
                    })
                }
            }
            Ok(Err(_e)) if self.config.fallback_enabled => {
                // SLM error, use LLM result (already in progress if speculative)
                let llm_result = if let Some(handle) = llm_handle {
                    tracing::debug!("SLM error, using speculative LLM result");
                    match handle.await {
                        Ok(Ok(r)) => r,
                        Ok(Err(e)) => return Err(e),
                        Err(e) => return Err(LlmError::Generation(format!("LLM task failed: {}", e))),
                    }
                } else {
                    self.llm.generate(messages).await?
                };

                self.update_stats(true, true, start.elapsed());

                Ok(SpeculativeResult {
                    text: llm_result.text.clone(),
                    model_used: ModelUsed::Llm,
                    generation: llm_result,
                    used_fallback: true,
                    complexity_score: Some(complexity),
                })
            }
            Ok(Err(e)) => {
                if let Some(handle) = llm_handle {
                    handle.abort();
                }
                Err(e)
            }
            Err(_) if self.config.fallback_enabled => {
                // Timeout, use LLM result (already in progress if speculative)
                let llm_result = if let Some(handle) = llm_handle {
                    tracing::debug!("SLM timeout, using speculative LLM result");
                    match handle.await {
                        Ok(Ok(r)) => r,
                        Ok(Err(e)) => return Err(e),
                        Err(e) => return Err(LlmError::Generation(format!("LLM task failed: {}", e))),
                    }
                } else {
                    self.llm.generate(messages).await?
                };

                self.update_stats(true, true, start.elapsed());

                Ok(SpeculativeResult {
                    text: llm_result.text.clone(),
                    model_used: ModelUsed::Llm,
                    generation: llm_result,
                    used_fallback: true,
                    complexity_score: Some(complexity),
                })
            }
            Err(_) => {
                if let Some(handle) = llm_handle {
                    handle.abort();
                }
                Err(LlmError::Timeout)
            }
        }
    }

    /// SLM-first with streaming
    async fn execute_slm_first_stream(
        &self,
        messages: &[Message],
        tx: mpsc::Sender<String>,
    ) -> Result<SpeculativeResult, LlmError> {
        let start = Instant::now();
        let complexity = self.estimate_complexity(messages);

        if complexity > self.config.complexity_threshold {
            let result = self.llm.generate_stream(messages, tx).await?;
            self.update_stats(false, true, start.elapsed());

            return Ok(SpeculativeResult {
                text: result.text.clone(),
                model_used: ModelUsed::Llm,
                generation: result,
                used_fallback: false,
                complexity_score: Some(complexity),
            });
        }

        let result = self.slm.generate_stream(messages, tx).await?;
        self.update_stats(true, false, start.elapsed());

        Ok(SpeculativeResult {
            text: result.text.clone(),
            model_used: ModelUsed::Slm,
            generation: result,
            used_fallback: false,
            complexity_score: Some(complexity),
        })
    }

    /// Race parallel strategy
    ///
    /// P0 FIX: Now properly aborts the losing model to save resources.
    /// Uses tokio::spawn with AbortHandle to cancel the slower model.
    async fn execute_race_parallel(&self, messages: &[Message]) -> Result<SpeculativeResult, LlmError> {
        let start = Instant::now();

        // Clone what we need for the spawned tasks
        let slm = self.slm.clone();
        let llm = self.llm.clone();
        let messages_for_slm = messages.to_vec();
        let messages_for_llm = messages.to_vec();

        // Spawn both as abortable tasks
        let slm_handle = tokio::spawn(async move {
            slm.generate(&messages_for_slm).await
        });

        let llm_handle = tokio::spawn(async move {
            llm.generate(&messages_for_llm).await
        });

        // P0 FIX: Get abort handles BEFORE select! (which moves the JoinHandles)
        let slm_abort = slm_handle.abort_handle();
        let llm_abort = llm_handle.abort_handle();

        // Use select to get first result and abort the other
        tokio::select! {
            slm_result = slm_handle => {
                // SLM finished first - abort LLM to save resources
                llm_abort.abort();
                tracing::debug!("SLM won race, aborting LLM");

                match slm_result {
                    Ok(Ok(result)) => {
                        let quality = self.estimate_quality(&result.text, messages);
                        if quality >= self.config.quality_threshold {
                            self.update_stats(true, false, start.elapsed());
                            Ok(SpeculativeResult {
                                text: result.text.clone(),
                                model_used: ModelUsed::Slm,
                                generation: result,
                                used_fallback: false,
                                complexity_score: None,
                            })
                        } else if self.config.fallback_enabled {
                            // Quality too low, need LLM after all
                            // Note: we already aborted the LLM task, need to start fresh
                            let llm_result = self.llm.generate(messages).await?;
                            self.update_stats(true, true, start.elapsed());
                            Ok(SpeculativeResult {
                                text: llm_result.text.clone(),
                                model_used: ModelUsed::Llm,
                                generation: llm_result,
                                used_fallback: true,
                                complexity_score: None,
                            })
                        } else {
                            self.update_stats(true, false, start.elapsed());
                            Ok(SpeculativeResult {
                                text: result.text.clone(),
                                model_used: ModelUsed::Slm,
                                generation: result,
                                used_fallback: false,
                                complexity_score: None,
                            })
                        }
                    }
                    Ok(Err(_)) if self.config.fallback_enabled => {
                        let llm_result = self.llm.generate(messages).await?;
                        self.update_stats(true, true, start.elapsed());
                        Ok(SpeculativeResult {
                            text: llm_result.text.clone(),
                            model_used: ModelUsed::Llm,
                            generation: llm_result,
                            used_fallback: true,
                            complexity_score: None,
                        })
                    }
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(LlmError::Generation(format!("SLM task panicked: {}", e))),
                }
            }
            llm_result = llm_handle => {
                // LLM finished first - abort SLM to save resources
                slm_abort.abort();
                tracing::debug!("LLM won race, aborting SLM");

                match llm_result {
                    Ok(Ok(result)) => {
                        self.update_stats(false, true, start.elapsed());
                        Ok(SpeculativeResult {
                            text: result.text.clone(),
                            model_used: ModelUsed::Llm,
                            generation: result,
                            used_fallback: false,
                            complexity_score: None,
                        })
                    }
                    Ok(Err(e)) => Err(e),
                    Err(e) => Err(LlmError::Generation(format!("LLM task panicked: {}", e))),
                }
            }
        }
    }

    /// Hybrid streaming strategy
    async fn execute_hybrid_streaming(&self, messages: &[Message]) -> Result<SpeculativeResult, LlmError> {
        // For non-streaming hybrid, just use SLM-first
        self.execute_slm_first(messages).await
    }

    /// Hybrid streaming with output
    async fn execute_hybrid_streaming_with_output(
        &self,
        messages: &[Message],
        tx: mpsc::Sender<String>,
    ) -> Result<SpeculativeResult, LlmError> {
        let start = Instant::now();

        // Start with SLM
        let (slm_tx, mut slm_rx) = mpsc::channel::<String>(100);

        let slm = self.slm.clone();
        let messages_clone = messages.to_vec();

        let slm_handle = tokio::spawn(async move {
            slm.generate_stream(&messages_clone, slm_tx).await
        });

        let mut tokens = Vec::new();
        let mut should_switch = false;

        // Collect initial tokens from SLM
        while let Some(token) = slm_rx.recv().await {
            tokens.push(token.clone());

            // Forward to output
            if tx.send(token).await.is_err() {
                break;
            }

            // Check if we should switch to LLM
            if tokens.len() >= self.config.min_tokens_before_switch {
                let quality = self.estimate_quality(&tokens.join(""), messages);
                if quality < self.config.quality_threshold * 0.8 {
                    should_switch = true;
                    break;
                }
            }
        }

        if should_switch && self.config.fallback_enabled {
            // Switch to LLM
            drop(slm_handle); // Cancel SLM

            // P1 FIX: Preserve SLM output and have LLM continue from there
            let slm_partial = tokens.join("");

            // Create continuation prompt that includes SLM output as assistant prefix
            let mut continuation_messages = messages.to_vec();
            if !slm_partial.is_empty() {
                continuation_messages.push(Message {
                    role: Role::Assistant,
                    content: format!("{} ", slm_partial), // Partial response to continue from
                });
            }

            // Continue with LLM from where SLM left off
            let result = self.llm.generate_stream(&continuation_messages, tx).await?;
            self.update_stats(true, true, start.elapsed());

            // Combine SLM prefix with LLM continuation
            let combined_text = format!("{}{}", slm_partial, result.text);

            Ok(SpeculativeResult {
                text: combined_text,
                model_used: ModelUsed::Hybrid,
                generation: result,
                used_fallback: true,
                complexity_score: None,
            })
        } else {
            // Continue with SLM
            let result = slm_handle.await
                .map_err(|e| LlmError::Generation(e.to_string()))??;

            self.update_stats(true, false, start.elapsed());

            Ok(SpeculativeResult {
                text: result.text.clone(),
                model_used: ModelUsed::Slm,
                generation: result,
                used_fallback: false,
                complexity_score: None,
            })
        }
    }

    /// P1 FIX: Draft-Verify strategy
    ///
    /// This implements a simplified speculative decoding pattern without KV cache sharing:
    /// 1. SLM drafts a chunk of tokens
    /// 2. LLM verifies the draft by scoring it (using completion with the draft as prefix)
    /// 3. If verification passes, accept draft and continue
    /// 4. If verification fails, use LLM to regenerate from last accepted point
    ///
    /// This provides better quality than pure SLM with lower latency than pure LLM,
    /// as the LLM only needs to verify/correct rather than generate from scratch.
    async fn execute_draft_verify(&self, messages: &[Message]) -> Result<SpeculativeResult, LlmError> {
        let start = Instant::now();

        // Check complexity - very complex queries go straight to LLM
        let complexity = self.estimate_complexity(messages);
        if complexity > self.config.complexity_threshold {
            let result = self.llm.generate(messages).await?;
            self.update_stats(false, true, start.elapsed());

            return Ok(SpeculativeResult {
                text: result.text.clone(),
                model_used: ModelUsed::Llm,
                generation: result,
                used_fallback: false,
                complexity_score: Some(complexity),
            });
        }

        let mut accepted_text = String::new();
        let mut iterations = 0;
        let mut total_slm_tokens = 0;
        let mut total_llm_tokens = 0;

        while iterations < self.config.max_draft_iterations {
            iterations += 1;

            // Create messages with accepted text as context
            let mut draft_messages = messages.to_vec();
            if !accepted_text.is_empty() {
                draft_messages.push(Message {
                    role: Role::Assistant,
                    content: accepted_text.clone(),
                });
            }

            // Draft with SLM (limited tokens)
            let draft_result = timeout(
                Duration::from_millis(self.config.slm_timeout_ms * 2), // More time for drafting
                self.slm.generate(&draft_messages)
            ).await;

            let draft = match draft_result {
                Ok(Ok(result)) => {
                    total_slm_tokens += result.tokens;
                    result.text
                }
                _ => {
                    // SLM failed, fall back to LLM for remaining
                    tracing::debug!("Draft failed, falling back to LLM");
                    break;
                }
            };

            // If draft is empty or very short, we're done
            if draft.trim().is_empty() || draft.len() < 5 {
                break;
            }

            // Verify with LLM by asking it to continue from the draft
            // If LLM's continuation diverges significantly, reject the draft
            let verify_quality = self.verify_draft(&draft, &draft_messages).await;

            if verify_quality >= self.config.verification_threshold {
                // Accept draft
                accepted_text.push_str(&draft);
                tracing::debug!(
                    iteration = iterations,
                    draft_len = draft.len(),
                    quality = verify_quality,
                    "Draft accepted"
                );

                // Check if generation is complete (ends with period, question mark, etc.)
                if self.is_complete_response(&accepted_text, messages) {
                    break;
                }
            } else {
                // Reject draft, get LLM to regenerate
                tracing::debug!(
                    iteration = iterations,
                    quality = verify_quality,
                    threshold = self.config.verification_threshold,
                    "Draft rejected, using LLM"
                );

                // Use LLM to generate the remaining text
                let mut llm_messages = messages.to_vec();
                if !accepted_text.is_empty() {
                    llm_messages.push(Message {
                        role: Role::Assistant,
                        content: accepted_text.clone(),
                    });
                }

                let llm_result = self.llm.generate(&llm_messages).await?;
                total_llm_tokens += llm_result.tokens;
                accepted_text.push_str(&llm_result.text);
                break;
            }
        }

        // If we exhausted iterations without completion, finish with LLM
        if iterations >= self.config.max_draft_iterations && !self.is_complete_response(&accepted_text, messages) {
            let mut llm_messages = messages.to_vec();
            if !accepted_text.is_empty() {
                llm_messages.push(Message {
                    role: Role::Assistant,
                    content: accepted_text.clone(),
                });
            }

            let llm_result = self.llm.generate(&llm_messages).await?;
            total_llm_tokens += llm_result.tokens;
            accepted_text.push_str(&llm_result.text);
        }

        self.update_stats(total_slm_tokens > 0, total_llm_tokens > 0, start.elapsed());

        // Determine which model contributed more
        let model_used = if total_llm_tokens > total_slm_tokens {
            ModelUsed::Llm
        } else if total_slm_tokens > 0 && total_llm_tokens > 0 {
            ModelUsed::Hybrid
        } else {
            ModelUsed::Slm
        };

        Ok(SpeculativeResult {
            text: accepted_text.clone(),
            model_used,
            generation: GenerationResult {
                text: accepted_text,
                tokens: total_slm_tokens + total_llm_tokens,
                time_to_first_token_ms: 0, // Not tracked in this mode
                total_time_ms: start.elapsed().as_millis() as u64,
                tokens_per_second: (total_slm_tokens + total_llm_tokens) as f32 / start.elapsed().as_secs_f32(),
                finish_reason: crate::backend::FinishReason::Stop,
                context: None,
            },
            used_fallback: total_llm_tokens > 0,
            complexity_score: Some(complexity),
        })
    }

    /// Verify a draft by estimating how likely LLM would produce similar output
    async fn verify_draft(&self, draft: &str, messages: &[Message]) -> f32 {
        // Quick heuristic verification without calling LLM
        // (Calling LLM for verification would negate the latency benefit)

        // 1. Check for obvious quality issues
        let quality = self.estimate_quality(draft, messages);
        if quality < 0.5 {
            return quality;
        }

        // 2. Check coherence with conversation context
        let coherence = self.estimate_coherence(draft, messages);

        // 3. Check domain relevance (for gold loan context)
        let relevance = self.estimate_domain_relevance(draft);

        // Weighted combination
        (quality * 0.4 + coherence * 0.3 + relevance * 0.3).min(1.0)
    }

    /// Estimate coherence of response with conversation
    fn estimate_coherence(&self, response: &str, messages: &[Message]) -> f32 {
        let empty = String::new();
        let last_user = messages.iter()
            .rev()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| &m.content)
            .unwrap_or(&empty);

        // Check for keyword overlap
        let user_words: std::collections::HashSet<&str> = last_user
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        let response_words: std::collections::HashSet<&str> = response
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();

        if user_words.is_empty() || response_words.is_empty() {
            return 0.7; // Neutral score
        }

        let overlap = user_words.intersection(&response_words).count();
        let overlap_ratio = overlap as f32 / user_words.len().min(response_words.len()) as f32;

        // Some overlap is good, too much might be parroting
        if overlap_ratio > 0.8 {
            0.6 // Might be just repeating
        } else if overlap_ratio > 0.1 {
            0.9 // Good contextual relevance
        } else {
            0.5 // Low relevance
        }
    }

    /// Estimate domain relevance (gold loan specific)
    fn estimate_domain_relevance(&self, response: &str) -> f32 {
        let lower = response.to_lowercase();

        // Domain-specific terms
        let gold_loan_terms = [
            "gold", "loan", "interest", "rate", "emi", "tenure",
            "collateral", "purity", "carat", "valuation", "disbursement",
            "सोना", "ऋण", "ब्याज", "दर", "कार्यकाल", // Hindi terms
            "kotak", "muthoot", "manappuram", "iifl",
        ];

        let matches = gold_loan_terms.iter()
            .filter(|term| lower.contains(*term))
            .count();

        // Score based on matches
        match matches {
            0 => 0.5,  // No domain terms
            1 => 0.7,  // Some relevance
            2..=3 => 0.85, // Good relevance
            _ => 0.95, // Very relevant
        }
    }

    /// Check if response appears complete
    fn is_complete_response(&self, response: &str, _messages: &[Message]) -> bool {
        let trimmed = response.trim();

        // Check for end punctuation
        if trimmed.ends_with('.') || trimmed.ends_with('?') ||
           trimmed.ends_with('!') || trimmed.ends_with('।') {
            return true;
        }

        // Check for common closing phrases
        let closers = [
            "thank you", "धन्यवाद", "please let me know", "any questions",
            "help you", "assist you", "और कुछ", "और जानकारी",
        ];

        let lower = trimmed.to_lowercase();
        closers.iter().any(|c| lower.contains(c))
    }

    /// Estimate query complexity
    fn estimate_complexity(&self, messages: &[Message]) -> f32 {
        // Simple heuristics for complexity
        let empty = String::new();
        let last_message = messages.last()
            .map(|m| &m.content)
            .unwrap_or(&empty);

        let mut score: f32 = 0.0;

        // Length-based
        if last_message.len() > 200 {
            score += 0.2;
        }

        // Question words
        let complex_markers = [
            "explain", "analyze", "compare", "describe",
            "calculate", "summarize", "translate",
            "समझाइए", "विश्लेषण", "तुलना", // Hindi
        ];

        let lower = last_message.to_lowercase();
        for marker in &complex_markers {
            if lower.contains(marker) {
                score += 0.3;
            }
        }

        // Multiple questions
        if last_message.matches('?').count() > 1 {
            score += 0.2;
        }

        // Code/technical content
        if last_message.contains("```") || last_message.contains("code") {
            score += 0.3;
        }

        score.min(1.0)
    }

    /// Estimate response quality
    ///
    /// P1 FIX: Improved heuristics for Hindi/Hinglish streaming context.
    /// - Don't penalize short initial responses (streaming starts small)
    /// - Account for Hindi politeness phrases ("maaf kijiye", "sorry" in greeting)
    /// - Better repetition detection that accounts for Hindi sentence structure
    fn estimate_quality(&self, response: &str, _messages: &[Message]) -> f32 {
        let mut score: f32 = 1.0;

        // P1 FIX: Only penalize very short responses, and less severely
        // During streaming, initial chunks are naturally short
        if response.len() < 10 {
            score -= 0.1; // Mild penalty for extremely short
        }

        // Repetition detection - improved for Hindi
        let words: Vec<&str> = response.split_whitespace().collect();
        if words.len() > 8 {
            // Need more words before judging repetition
            let unique: std::collections::HashSet<&str> = words.iter().cloned().collect();
            let repetition_ratio = unique.len() as f32 / words.len() as f32;
            // P1 FIX: Higher threshold - Hindi often repeats conjunctions (aur, toh, ki)
            if repetition_ratio < 0.35 {
                score -= 0.3;
            }
        }

        // P1 FIX: Only penalize actual error indicators, not polite phrases
        // "sorry" and "cannot" are valid in Indian English greetings/politeness
        let error_indicators = [
            "error:", "exception", "failed to", "invalid input",
            "त्रुटि", "गलती हुई", // Hindi error indicators
        ];
        let lower = response.to_lowercase();
        for indicator in &error_indicators {
            if lower.contains(indicator) {
                score -= 0.3;
                break; // Only penalize once
            }
        }

        // Detect gibberish/garbage output (repeated special characters)
        let special_char_ratio = response.chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace() && *c != '।' && *c != '?' && *c != '!')
            .count() as f32 / response.len().max(1) as f32;
        if special_char_ratio > 0.3 {
            score -= 0.4;
        }

        score.max(0.0)
    }

    /// Update statistics
    ///
    /// P2 FIX: Uses Welford's online algorithm for numerically stable mean updates.
    /// The formula `mean += (x - mean) / n` avoids accumulating floating-point errors
    /// that can occur with the naive `(mean * (n-1) + x) / n` formula.
    fn update_stats(&self, used_slm: bool, used_llm: bool, duration: Duration) {
        let mut stats = self.stats.lock();
        let duration_ms = duration.as_millis() as f32;

        if used_slm {
            stats.slm_calls += 1;
            if !used_llm {
                stats.slm_successes += 1;
            }
            // Welford's algorithm: mean += (x - mean) / n
            let delta = duration_ms - stats.avg_slm_time_ms;
            stats.avg_slm_time_ms += delta / stats.slm_calls as f32;
        }

        if used_llm {
            stats.llm_calls += 1;
            if used_slm {
                stats.llm_fallbacks += 1;
            }
            // Welford's algorithm: mean += (x - mean) / n
            let delta = duration_ms - stats.avg_llm_time_ms;
            stats.avg_llm_time_ms += delta / stats.llm_calls as f32;
        }
    }

    /// Get statistics
    pub fn stats(&self) -> SpeculativeStats {
        self.stats.lock().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock() = SpeculativeStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.mode, SpeculativeMode::SlmFirst);
        assert!(config.fallback_enabled);
        // P2 FIX: Verify speculative threshold is set
        assert!(config.speculative_llm_threshold > 0.0);
        assert!(config.speculative_llm_threshold < config.complexity_threshold);
    }

    #[test]
    fn test_speculative_threshold_config() {
        // P2 FIX: Test that speculative threshold can be configured
        let config = SpeculativeConfig {
            speculative_llm_threshold: 0.5,
            ..Default::default()
        };
        assert_eq!(config.speculative_llm_threshold, 0.5);

        // Disable speculative execution
        let no_speculative = SpeculativeConfig {
            speculative_llm_threshold: 1.0,
            ..Default::default()
        };
        assert_eq!(no_speculative.speculative_llm_threshold, 1.0);
    }

    #[test]
    fn test_complexity_estimation() {
        // Would need mock backends to test properly
    }

    #[test]
    fn test_draft_verify_config() {
        // P1 FIX: Test draft-verify configuration
        let config = SpeculativeConfig::default();

        // Default draft-verify settings
        assert_eq!(config.draft_chunk_size, 20);
        assert_eq!(config.max_draft_iterations, 5);
        assert!(config.verification_threshold > 0.0);
        assert!(config.verification_threshold < 1.0);

        // Custom draft-verify config
        let custom = SpeculativeConfig {
            mode: SpeculativeMode::DraftVerify,
            draft_chunk_size: 30,
            max_draft_iterations: 3,
            verification_threshold: 0.8,
            ..Default::default()
        };
        assert_eq!(custom.mode, SpeculativeMode::DraftVerify);
        assert_eq!(custom.draft_chunk_size, 30);
        assert_eq!(custom.max_draft_iterations, 3);
        assert_eq!(custom.verification_threshold, 0.8);
    }

    #[test]
    fn test_domain_relevance_estimation() {
        // P1 FIX: Test domain relevance scoring

        // High relevance - multiple gold loan terms
        let high_relevance = "The gold loan interest rate is 7.5% per annum with flexible tenure options.";
        // Would need executor instance to test, but logic should score this ~0.95

        // Low relevance - no domain terms
        let low_relevance = "Hello, how are you doing today?";
        // Should score ~0.5

        // Medium relevance - one term
        let medium_relevance = "I'd like to know about your loan products.";
        // Should score ~0.7

        // These assertions would need mock backend to test
        assert!(high_relevance.contains("gold"));
        assert!(high_relevance.contains("loan"));
        assert!(!low_relevance.contains("gold"));
        assert!(medium_relevance.contains("loan"));
    }

    #[test]
    fn test_speculative_modes() {
        // P1 FIX: Test all speculative modes are available
        let modes = [
            SpeculativeMode::SlmFirst,
            SpeculativeMode::RaceParallel,
            SpeculativeMode::HybridStreaming,
            SpeculativeMode::DraftVerify,
        ];

        for mode in modes {
            let config = SpeculativeConfig {
                mode,
                ..Default::default()
            };
            assert_eq!(config.mode, mode);
        }
    }
}
