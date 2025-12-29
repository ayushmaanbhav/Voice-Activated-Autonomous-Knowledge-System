//! P1 FIX: Context Sizing by Conversation Stage
//!
//! Implements stage-aware token budgets for RAG context.
//! Different conversation stages have different information needs:
//! - Greeting: Minimal context needed
//! - Discovery: Moderate context for understanding needs
//! - Presentation: Maximum context for detailed product info
//! - ObjectionHandling: High context for competitive comparisons
//! - Closing: Moderate for final confirmations
//! - Farewell: Minimal wrap-up context

use serde::{Deserialize, Serialize};

/// Conversation stage for context sizing
///
/// Mirrors the ConversationStage from agent crate but kept separate
/// to avoid circular dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Stage {
    #[default]
    Greeting,
    Discovery,
    Qualification,
    Presentation,
    ObjectionHandling,
    Closing,
    Farewell,
}

impl Stage {
    /// Convert from string (for compatibility with other crates)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "greeting" => Self::Greeting,
            "discovery" => Self::Discovery,
            "qualification" => Self::Qualification,
            "presentation" => Self::Presentation,
            "objection_handling" | "objectionhandling" => Self::ObjectionHandling,
            "closing" => Self::Closing,
            "farewell" => Self::Farewell,
            _ => Self::Greeting,
        }
    }

    /// Convert to string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Greeting => "greeting",
            Self::Discovery => "discovery",
            Self::Qualification => "qualification",
            Self::Presentation => "presentation",
            Self::ObjectionHandling => "objection_handling",
            Self::Closing => "closing",
            Self::Farewell => "farewell",
        }
    }
}

/// Token budget for context construction
///
/// Defines how many tokens to allocate for different parts of the prompt.
#[derive(Debug, Clone, Copy)]
pub struct ContextBudget {
    /// Tokens for RAG-retrieved documents
    pub rag_tokens: usize,
    /// Tokens for conversation history
    pub history_tokens: usize,
    /// Tokens for system prompt and instructions
    pub system_tokens: usize,
    /// Tokens reserved for response generation
    pub response_reserve: usize,
}

impl ContextBudget {
    /// Create a new context budget
    pub fn new(rag: usize, history: usize, system: usize, reserve: usize) -> Self {
        Self {
            rag_tokens: rag,
            history_tokens: history,
            system_tokens: system,
            response_reserve: reserve,
        }
    }

    /// Total budget (excluding response reserve)
    pub fn total_context(&self) -> usize {
        self.rag_tokens + self.history_tokens + self.system_tokens
    }

    /// Total including response reserve
    pub fn total(&self) -> usize {
        self.total_context() + self.response_reserve
    }

    /// Scale budget to fit within a maximum context size
    pub fn scale_to_fit(&self, max_context: usize) -> Self {
        let total = self.total();
        if total <= max_context {
            return *self;
        }

        let scale = max_context as f64 / total as f64;
        Self {
            rag_tokens: (self.rag_tokens as f64 * scale) as usize,
            history_tokens: (self.history_tokens as f64 * scale) as usize,
            system_tokens: (self.system_tokens as f64 * scale) as usize,
            response_reserve: (self.response_reserve as f64 * scale) as usize,
        }
    }
}

impl Default for ContextBudget {
    fn default() -> Self {
        // Conservative default
        Self {
            rag_tokens: 800,
            history_tokens: 400,
            system_tokens: 600,
            response_reserve: 500,
        }
    }
}

/// Get context budget for a conversation stage
///
/// Returns stage-specific token allocations based on the information
/// needs at each point in the conversation.
///
/// # Arguments
/// * `stage` - Current conversation stage
///
/// # Returns
/// Context budget with token allocations for RAG, history, and system prompt
pub fn context_budget_for_stage(stage: Stage) -> ContextBudget {
    match stage {
        // Greeting: Minimal context, focus on introduction
        Stage::Greeting => ContextBudget {
            rag_tokens: 200,
            history_tokens: 100,
            system_tokens: 500,
            response_reserve: 300,
        },

        // Discovery: Moderate context for understanding needs
        Stage::Discovery => ContextBudget {
            rag_tokens: 800,
            history_tokens: 400,
            system_tokens: 600,
            response_reserve: 500,
        },

        // Qualification: Higher context for eligibility assessment
        Stage::Qualification => ContextBudget {
            rag_tokens: 1000,
            history_tokens: 500,
            system_tokens: 600,
            response_reserve: 600,
        },

        // Presentation: Maximum RAG context for detailed product info
        Stage::Presentation => ContextBudget {
            rag_tokens: 2000,
            history_tokens: 800,
            system_tokens: 800,
            response_reserve: 800,
        },

        // ObjectionHandling: High context for comparisons and rebuttals
        Stage::ObjectionHandling => ContextBudget {
            rag_tokens: 1500,
            history_tokens: 600,
            system_tokens: 700,
            response_reserve: 700,
        },

        // Closing: Moderate context for final details
        Stage::Closing => ContextBudget {
            rag_tokens: 500,
            history_tokens: 300,
            system_tokens: 500,
            response_reserve: 500,
        },

        // Farewell: Minimal, just wrap-up
        Stage::Farewell => ContextBudget {
            rag_tokens: 100,
            history_tokens: 100,
            system_tokens: 300,
            response_reserve: 200,
        },
    }
}

/// Configuration for context management
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum total context size (model limit)
    pub max_context_tokens: usize,
    /// Enable stage-aware sizing
    pub stage_aware: bool,
    /// Default budget when stage is unknown
    pub default_budget: ContextBudget,
    /// Minimum RAG tokens (floor)
    pub min_rag_tokens: usize,
    /// Maximum RAG tokens (ceiling)
    pub max_rag_tokens: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 4096,
            stage_aware: true,
            default_budget: ContextBudget::default(),
            min_rag_tokens: 100,
            max_rag_tokens: 2500,
        }
    }
}

impl ContextConfig {
    /// Create config for small models (4K context)
    pub fn small_model() -> Self {
        Self {
            max_context_tokens: 4096,
            ..Default::default()
        }
    }

    /// Create config for medium models (8K context)
    pub fn medium_model() -> Self {
        Self {
            max_context_tokens: 8192,
            default_budget: ContextBudget {
                rag_tokens: 1500,
                history_tokens: 800,
                system_tokens: 800,
                response_reserve: 800,
            },
            max_rag_tokens: 4000,
            ..Default::default()
        }
    }

    /// Create config for large models (32K+ context)
    pub fn large_model() -> Self {
        Self {
            max_context_tokens: 32768,
            default_budget: ContextBudget {
                rag_tokens: 4000,
                history_tokens: 2000,
                system_tokens: 1500,
                response_reserve: 2000,
            },
            max_rag_tokens: 8000,
            ..Default::default()
        }
    }
}

/// Context manager for stage-aware token budgeting
pub struct ContextManager {
    config: ContextConfig,
}

impl ContextManager {
    /// Create a new context manager
    pub fn new(config: ContextConfig) -> Self {
        Self { config }
    }

    /// Get budget for a stage, scaled to fit model context
    pub fn get_budget(&self, stage: Stage) -> ContextBudget {
        let base_budget = if self.config.stage_aware {
            context_budget_for_stage(stage)
        } else {
            self.config.default_budget
        };

        // Scale to fit model context
        let mut budget = base_budget.scale_to_fit(self.config.max_context_tokens);

        // Apply floor/ceiling to RAG tokens
        budget.rag_tokens = budget
            .rag_tokens
            .max(self.config.min_rag_tokens)
            .min(self.config.max_rag_tokens);

        budget
    }

    /// Get budget from stage string (convenience method)
    pub fn get_budget_for(&self, stage_str: &str) -> ContextBudget {
        self.get_budget(Stage::from_str(stage_str))
    }

    /// Calculate how many documents can fit in RAG budget
    ///
    /// # Arguments
    /// * `stage` - Conversation stage
    /// * `avg_doc_tokens` - Average tokens per document
    ///
    /// # Returns
    /// Maximum number of documents to retrieve
    pub fn max_documents(&self, stage: Stage, avg_doc_tokens: usize) -> usize {
        let budget = self.get_budget(stage);
        if avg_doc_tokens == 0 {
            return 0;
        }
        budget.rag_tokens / avg_doc_tokens
    }

    /// Truncate documents to fit budget
    ///
    /// # Arguments
    /// * `docs` - Document texts
    /// * `stage` - Conversation stage
    /// * `tokenizer` - Function to count tokens in text
    ///
    /// # Returns
    /// Documents that fit within budget
    pub fn fit_documents<F>(
        &self,
        docs: &[String],
        stage: Stage,
        tokenizer: F,
    ) -> Vec<String>
    where
        F: Fn(&str) -> usize,
    {
        let budget = self.get_budget(stage);
        let mut total_tokens = 0;
        let mut result = Vec::new();

        for doc in docs {
            let doc_tokens = tokenizer(doc);
            if total_tokens + doc_tokens > budget.rag_tokens {
                // Try to fit partial document
                let remaining = budget.rag_tokens.saturating_sub(total_tokens);
                if remaining > 50 {
                    // Worth including partial
                    let truncated = Self::truncate_to_tokens(doc, remaining, &tokenizer);
                    if !truncated.is_empty() {
                        result.push(truncated);
                    }
                }
                break;
            }
            total_tokens += doc_tokens;
            result.push(doc.clone());
        }

        result
    }

    /// Truncate text to approximate token count
    fn truncate_to_tokens<F>(text: &str, max_tokens: usize, tokenizer: &F) -> String
    where
        F: Fn(&str) -> usize,
    {
        // Binary search for right truncation point
        let mut low = 0;
        let mut high = text.len();

        while low < high {
            let mid = (low + high + 1) / 2;
            let truncated = &text[..mid];
            if tokenizer(truncated) <= max_tokens {
                low = mid;
            } else {
                high = mid - 1;
            }
        }

        // Find word boundary
        let truncated = &text[..low];
        if let Some(last_space) = truncated.rfind(char::is_whitespace) {
            truncated[..last_space].to_string()
        } else {
            truncated.to_string()
        }
    }

    /// Get config reference
    pub fn config(&self) -> &ContextConfig {
        &self.config
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new(ContextConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_from_str() {
        assert_eq!(Stage::from_str("greeting"), Stage::Greeting);
        assert_eq!(Stage::from_str("Presentation"), Stage::Presentation);
        assert_eq!(Stage::from_str("objection_handling"), Stage::ObjectionHandling);
        assert_eq!(Stage::from_str("invalid"), Stage::Greeting); // Default
    }

    #[test]
    fn test_budget_for_stage() {
        let greeting = context_budget_for_stage(Stage::Greeting);
        let presentation = context_budget_for_stage(Stage::Presentation);

        // Presentation should have more RAG tokens than Greeting
        assert!(presentation.rag_tokens > greeting.rag_tokens);
        // Greeting should be minimal
        assert!(greeting.rag_tokens <= 300);
    }

    #[test]
    fn test_budget_scaling() {
        let budget = ContextBudget::new(2000, 1000, 800, 800);
        let scaled = budget.scale_to_fit(2000);

        assert!(scaled.total() <= 2000);
        assert!(scaled.rag_tokens < budget.rag_tokens);
    }

    #[test]
    fn test_context_manager() {
        let manager = ContextManager::new(ContextConfig::small_model());

        let greeting_budget = manager.get_budget(Stage::Greeting);
        let presentation_budget = manager.get_budget(Stage::Presentation);

        // Both should fit within model context
        assert!(greeting_budget.total() <= 4096);
        assert!(presentation_budget.total() <= 4096);
    }

    #[test]
    fn test_max_documents() {
        let manager = ContextManager::new(ContextConfig::default());

        // With 100 tokens per doc
        let max_greeting = manager.max_documents(Stage::Greeting, 100);
        let max_presentation = manager.max_documents(Stage::Presentation, 100);

        assert!(max_presentation > max_greeting);
    }

    #[test]
    fn test_fit_documents() {
        let manager = ContextManager::new(ContextConfig::small_model());

        let docs = vec![
            "Short doc".to_string(),
            "Medium length document here".to_string(),
            "This is a longer document that should take more tokens".to_string(),
        ];

        // Simple tokenizer: words
        let tokenizer = |s: &str| s.split_whitespace().count();

        let fitted = manager.fit_documents(&docs, Stage::Greeting, tokenizer);

        // Should fit some documents within greeting budget
        assert!(!fitted.is_empty());
    }
}
