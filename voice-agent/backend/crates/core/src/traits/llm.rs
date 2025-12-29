//! Language Model traits

use async_trait::async_trait;
use std::pin::Pin;
use futures::Stream;
use crate::{Result, GenerateRequest, GenerateResponse, StreamChunk, ToolDefinition};

/// Language Model interface
///
/// Implementations:
/// - `OllamaBackend` - Local Ollama inference
/// - `ClaudeBackend` - Anthropic Claude API (future)
/// - `OpenAIBackend` - OpenAI API (future)
///
/// # Example
///
/// ```ignore
/// let llm: Box<dyn LanguageModel> = Box::new(OllamaBackend::new(config));
/// let request = GenerateRequest::new("You are a helpful assistant")
///     .with_user_message("What is a gold loan?");
/// let response = llm.generate(request).await?;
/// println!("{}", response.text);
/// ```
#[async_trait]
pub trait LanguageModel: Send + Sync + 'static {
    /// Generate completion
    ///
    /// # Arguments
    /// * `request` - Generation request with messages, parameters
    ///
    /// # Returns
    /// Generated response with text and metadata
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

    /// Stream tokens as generated
    ///
    /// Lower latency than `generate()` as tokens are yielded incrementally.
    ///
    /// # Arguments
    /// * `request` - Generation request (will have `stream: true` set)
    ///
    /// # Returns
    /// Stream of text chunks
    fn generate_stream<'a>(
        &'a self,
        request: GenerateRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send + 'a>>;

    /// Generate with tool/function calling
    ///
    /// # Arguments
    /// * `request` - Generation request
    /// * `tools` - Available tool definitions
    ///
    /// # Returns
    /// Response which may include tool calls in addition to text
    async fn generate_with_tools(
        &self,
        request: GenerateRequest,
        tools: &[ToolDefinition],
    ) -> Result<GenerateResponse>;

    /// Check if model is available
    ///
    /// Returns false if model is not loaded or backend is unavailable.
    async fn is_available(&self) -> bool;

    /// Get model name for logging
    fn model_name(&self) -> &str;

    /// Get context window size in tokens
    fn context_size(&self) -> usize {
        4096 // Default, implementations should override
    }

    /// Estimate token count for text
    ///
    /// Uses a simple heuristic by default. Implementations may use
    /// actual tokenizers for more accurate counts.
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimate: 4 chars per token for English, 2 for Indic
        // This is a very rough heuristic
        text.chars().count() / 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockLlm;

    #[async_trait]
    impl LanguageModel for MockLlm {
        async fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse> {
            Ok(GenerateResponse::text("Mock response"))
        }

        fn generate_stream<'a>(
            &'a self,
            _request: GenerateRequest,
        ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send + 'a>> {
            Box::pin(futures::stream::empty())
        }

        async fn generate_with_tools(
            &self,
            request: GenerateRequest,
            _tools: &[ToolDefinition],
        ) -> Result<GenerateResponse> {
            self.generate(request).await
        }

        async fn is_available(&self) -> bool {
            true
        }

        fn model_name(&self) -> &str {
            "mock-llm"
        }
    }

    #[tokio::test]
    async fn test_mock_llm() {
        let llm = MockLlm;
        assert!(llm.is_available().await);
        assert_eq!(llm.model_name(), "mock-llm");

        let request = GenerateRequest::new("Test")
            .with_user_message("Hello");
        let response = llm.generate(request).await.unwrap();
        assert_eq!(response.text, "Mock response");
    }

    #[test]
    fn test_token_estimation() {
        let llm = MockLlm;
        // "Hello world" = 11 chars, ~3-4 tokens
        let estimate = llm.estimate_tokens("Hello world");
        assert!(estimate > 0 && estimate < 10);
    }
}
