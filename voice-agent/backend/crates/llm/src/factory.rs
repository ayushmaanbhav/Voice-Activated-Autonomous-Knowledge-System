//! LLM Factory - Provider Abstraction Layer
//!
//! Creates LLM backends based on configuration with automatic provider detection.
//! Supports hot-reload of configuration for dynamic model switching.
//!
//! ## Supported Providers
//! - **Claude**: Native tool_use support, best for complex tool calling
//! - **Ollama**: Local models with text-based tool injection
//! - **OpenAI**: GPT-4, GPT-3.5, Azure OpenAI
//!
//! ## Example
//! ```ignore
//! let config = LlmProviderConfig::claude("your-api-key")
//!     .with_model("opus");
//! let llm = LlmFactory::create(&config)?;
//! ```

use std::sync::Arc;
use voice_agent_config::constants::endpoints;
use voice_agent_core::{llm_types::ToolDefinition, LanguageModel};

use crate::{
    adapter::LanguageModelAdapter,
    backend::{LlmBackend, LlmConfig, OllamaBackend, OpenAIBackend, OpenAIConfig},
    claude::{ClaudeBackend, ClaudeConfig},
    LlmError,
};

/// LLM provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LlmProvider {
    /// Claude (Anthropic) - native tool_use support
    #[default]
    Claude,
    /// Ollama - local models, text-based tool injection
    Ollama,
    /// OpenAI - GPT-4, GPT-3.5
    OpenAI,
    /// Azure OpenAI - Azure-hosted GPT models
    AzureOpenAI,
}

impl LlmProvider {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "claude" | "anthropic" => Some(LlmProvider::Claude),
            "ollama" | "local" => Some(LlmProvider::Ollama),
            "openai" | "gpt" => Some(LlmProvider::OpenAI),
            "azure" | "azure-openai" => Some(LlmProvider::AzureOpenAI),
            _ => None,
        }
    }
}

/// Unified LLM provider configuration
#[derive(Debug, Clone)]
pub struct LlmProviderConfig {
    /// Provider type
    pub provider: LlmProvider,
    /// API key (for Claude/OpenAI)
    pub api_key: Option<String>,
    /// API endpoint (for Ollama/Azure)
    pub endpoint: Option<String>,
    /// Model name or ID
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature (0.0 - 1.0 for Claude, 0.0 - 2.0 for others)
    pub temperature: f32,
    /// Enable streaming
    pub streaming: bool,
    /// Azure API version (for Azure OpenAI only)
    pub azure_api_version: Option<String>,
    /// Organization ID (for OpenAI only)
    pub organization: Option<String>,
}

impl Default for LlmProviderConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::Claude,
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            endpoint: None,
            model: "opus".to_string(),
            max_tokens: 1024,
            temperature: 0.7,
            streaming: true,
            azure_api_version: None,
            organization: None,
        }
    }
}

impl LlmProviderConfig {
    /// Create Claude config
    pub fn claude(api_key: impl Into<String>) -> Self {
        Self {
            provider: LlmProvider::Claude,
            api_key: Some(api_key.into()),
            model: "opus".to_string(),
            ..Default::default()
        }
    }

    /// Create Ollama config
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            provider: LlmProvider::Ollama,
            api_key: None,
            endpoint: Some(endpoints::OLLAMA_DEFAULT.to_string()),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Create OpenAI config
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: LlmProvider::OpenAI,
            api_key: Some(api_key.into()),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Create Azure OpenAI config
    pub fn azure(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        deployment: impl Into<String>,
        api_version: impl Into<String>,
    ) -> Self {
        Self {
            provider: LlmProvider::AzureOpenAI,
            api_key: Some(api_key.into()),
            endpoint: Some(endpoint.into()),
            model: deployment.into(),
            azure_api_version: Some(api_version.into()),
            ..Default::default()
        }
    }

    /// Set model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Enable/disable streaming
    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    /// Set endpoint (for Ollama/Azure)
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }
}

/// Factory for creating LLM backends
pub struct LlmFactory;

impl LlmFactory {
    /// Create a LanguageModel from config (implements core trait)
    pub fn create(
        config: &LlmProviderConfig,
    ) -> std::result::Result<Arc<dyn LanguageModel>, LlmError> {
        match config.provider {
            LlmProvider::Claude => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    .ok_or_else(|| {
                        LlmError::Configuration("Claude requires ANTHROPIC_API_KEY".to_string())
                    })?;

                let claude_config = ClaudeConfig::new(api_key)
                    .with_model_str(&config.model)
                    .with_max_tokens(config.max_tokens)
                    .with_temperature(config.temperature)
                    .with_streaming(config.streaming);

                let backend = ClaudeBackend::new(claude_config)?;
                Ok(Arc::new(ClaudeLanguageModel::new(backend)))
            },

            LlmProvider::Ollama => {
                let endpoint = config
                    .endpoint
                    .clone()
                    .unwrap_or_else(|| endpoints::OLLAMA_DEFAULT.to_string());

                let ollama_config = LlmConfig {
                    model: config.model.clone(),
                    endpoint,
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                    stream: config.streaming,
                    ..Default::default()
                };

                let backend = OllamaBackend::new(ollama_config)?;
                Ok(Arc::new(LanguageModelAdapter::new(backend)))
            },

            LlmProvider::OpenAI => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .ok_or_else(|| {
                        LlmError::Configuration("OpenAI requires OPENAI_API_KEY".to_string())
                    })?;

                let openai_config = OpenAIConfig::openai(api_key, &config.model);
                let backend = OpenAIBackend::new(openai_config)?;
                Ok(Arc::new(LanguageModelAdapter::new(backend)))
            },

            LlmProvider::AzureOpenAI => {
                let endpoint = config.endpoint.clone().ok_or_else(|| {
                    LlmError::Configuration("Azure OpenAI requires endpoint".to_string())
                })?;
                let api_key = config.api_key.clone().ok_or_else(|| {
                    LlmError::Configuration("Azure OpenAI requires api_key".to_string())
                })?;
                let api_version = config
                    .azure_api_version
                    .clone()
                    .unwrap_or_else(|| "2024-02-01".to_string());

                let azure_config =
                    OpenAIConfig::azure(endpoint, api_key, &config.model, api_version);
                let backend = OpenAIBackend::new(azure_config)?;
                Ok(Arc::new(LanguageModelAdapter::new(backend)))
            },
        }
    }

    /// Create a raw LlmBackend (for low-level access)
    pub fn create_backend(
        config: &LlmProviderConfig,
    ) -> std::result::Result<Arc<dyn LlmBackend>, LlmError> {
        match config.provider {
            LlmProvider::Claude => {
                let api_key = config
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                    .ok_or_else(|| {
                        LlmError::Configuration("Claude requires ANTHROPIC_API_KEY".to_string())
                    })?;

                let claude_config = ClaudeConfig::new(api_key)
                    .with_model_str(&config.model)
                    .with_max_tokens(config.max_tokens)
                    .with_temperature(config.temperature)
                    .with_streaming(config.streaming);

                Ok(Arc::new(ClaudeBackend::new(claude_config)?))
            },

            LlmProvider::Ollama => {
                let endpoint = config
                    .endpoint
                    .clone()
                    .unwrap_or_else(|| endpoints::OLLAMA_DEFAULT.to_string());

                let ollama_config = LlmConfig {
                    model: config.model.clone(),
                    endpoint,
                    max_tokens: config.max_tokens,
                    temperature: config.temperature,
                    stream: config.streaming,
                    ..Default::default()
                };

                Ok(Arc::new(OllamaBackend::new(ollama_config)?))
            },

            LlmProvider::OpenAI | LlmProvider::AzureOpenAI => {
                let api_key = config.api_key.clone().ok_or_else(|| {
                    LlmError::Configuration("OpenAI/Azure requires api_key".to_string())
                })?;

                let openai_config = if config.provider == LlmProvider::AzureOpenAI {
                    let endpoint = config.endpoint.clone().ok_or_else(|| {
                        LlmError::Configuration("Azure OpenAI requires endpoint".to_string())
                    })?;
                    let api_version = config
                        .azure_api_version
                        .clone()
                        .unwrap_or_else(|| "2024-02-01".to_string());
                    OpenAIConfig::azure(endpoint, api_key, &config.model, api_version)
                } else {
                    OpenAIConfig::openai(api_key, &config.model)
                };

                Ok(Arc::new(OpenAIBackend::new(openai_config)?))
            },
        }
    }

    /// Get the default provider from environment
    pub fn default_provider() -> LlmProvider {
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            LlmProvider::Claude
        } else if std::env::var("OPENAI_API_KEY").is_ok() {
            LlmProvider::OpenAI
        } else {
            LlmProvider::Ollama
        }
    }
}

// =============================================================================
// Claude LanguageModel Wrapper
// =============================================================================

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use tokio::sync::mpsc;
use voice_agent_core::{
    llm_types::{FinishReason, TokenUsage},
    Error, GenerateRequest, GenerateResponse, Result, StreamChunk,
};

/// Wrapper that implements core::LanguageModel for ClaudeBackend
/// with native tool_use support
pub struct ClaudeLanguageModel {
    backend: Arc<ClaudeBackend>,
}

impl ClaudeLanguageModel {
    pub fn new(backend: ClaudeBackend) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }
}

#[async_trait]
impl LanguageModel for ClaudeLanguageModel {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse> {
        let messages = convert_to_prompt_messages(&request);

        let response = self
            .backend
            .generate_with_tools(&messages, &[])
            .await
            .map_err(|e| Error::Llm(e.to_string()))?;

        Ok(GenerateResponse {
            text: response.text,
            finish_reason: convert_claude_stop_reason(response.stop_reason),
            usage: Some(TokenUsage::new(
                response.input_tokens as u32,
                response.output_tokens as u32,
            )),
            tool_calls: response.tool_calls,
        })
    }

    fn generate_stream<'a>(
        &'a self,
        request: GenerateRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send + 'a>> {
        let messages = convert_to_prompt_messages(&request);
        let backend = self.backend.clone();

        Box::pin(async_stream::stream! {
            let (tx, mut rx) = mpsc::channel::<String>(100);

            // Spawn the streaming task with cloned backend
            let stream_task = {
                let messages = messages.clone();
                let backend = backend.clone();
                async move {
                    backend.generate_with_tools_stream(&messages, &[], tx).await
                }
            };

            // Start streaming in background
            let handle = tokio::spawn(stream_task);

            // Yield chunks as they arrive
            while let Some(text) = rx.recv().await {
                yield Ok(StreamChunk::text(text));
            }

            // Wait for completion
            match handle.await {
                Ok(Ok(response)) => {
                    yield Ok(StreamChunk {
                        delta: String::new(),
                        is_final: true,
                        finish_reason: Some(convert_claude_stop_reason(response.stop_reason)),
                    });
                }
                Ok(Err(e)) => {
                    yield Err(Error::Llm(e.to_string()));
                }
                Err(e) => {
                    yield Err(Error::Llm(format!("Task join error: {}", e)));
                }
            }
        })
    }

    async fn generate_with_tools(
        &self,
        request: GenerateRequest,
        tools: &[ToolDefinition],
    ) -> Result<GenerateResponse> {
        let messages = convert_to_prompt_messages(&request);

        let response = self
            .backend
            .generate_with_tools(&messages, tools)
            .await
            .map_err(|e| Error::Llm(e.to_string()))?;

        Ok(GenerateResponse {
            text: response.text,
            finish_reason: convert_claude_stop_reason(response.stop_reason),
            usage: Some(TokenUsage::new(
                response.input_tokens as u32,
                response.output_tokens as u32,
            )),
            tool_calls: response.tool_calls,
        })
    }

    async fn is_available(&self) -> bool {
        self.backend.is_available().await
    }

    fn model_name(&self) -> &str {
        self.backend.model_name()
    }

    fn context_size(&self) -> usize {
        200_000 // Claude models support 200k context
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        // Claude tokenization: ~4 chars per token for English
        text.len() / 4
    }
}

fn convert_to_prompt_messages(request: &GenerateRequest) -> Vec<crate::prompt::Message> {
    request
        .messages
        .iter()
        .map(|m| crate::prompt::Message {
            role: match m.role {
                voice_agent_core::llm_types::Role::System => crate::prompt::Role::System,
                voice_agent_core::llm_types::Role::User => crate::prompt::Role::User,
                voice_agent_core::llm_types::Role::Assistant => crate::prompt::Role::Assistant,
                voice_agent_core::llm_types::Role::Tool => crate::prompt::Role::Tool,
            },
            content: m.content.clone(),
        })
        .collect()
}

fn convert_claude_stop_reason(reason: crate::claude::ClaudeStopReason) -> FinishReason {
    match reason {
        crate::claude::ClaudeStopReason::EndTurn => FinishReason::Stop,
        crate::claude::ClaudeStopReason::MaxTokens => FinishReason::Length,
        crate::claude::ClaudeStopReason::StopSequence => FinishReason::Stop,
        crate::claude::ClaudeStopReason::ToolUse => FinishReason::ToolCalls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_str() {
        assert_eq!(LlmProvider::from_str("claude"), Some(LlmProvider::Claude));
        assert_eq!(LlmProvider::from_str("ollama"), Some(LlmProvider::Ollama));
        assert_eq!(LlmProvider::from_str("openai"), Some(LlmProvider::OpenAI));
        assert_eq!(
            LlmProvider::from_str("azure"),
            Some(LlmProvider::AzureOpenAI)
        );
        assert_eq!(LlmProvider::from_str("unknown"), None);
    }

    #[test]
    fn test_claude_config() {
        let config = LlmProviderConfig::claude("test-key")
            .with_model("sonnet")
            .with_max_tokens(2048)
            .with_temperature(0.5);

        assert_eq!(config.provider, LlmProvider::Claude);
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.model, "sonnet");
        assert_eq!(config.max_tokens, 2048);
        assert_eq!(config.temperature, 0.5);
    }

    #[test]
    fn test_ollama_config() {
        let config = LlmProviderConfig::ollama("qwen3:4b-instruct-2507-q4_K_M")
            .with_endpoint("http://custom:11434");

        assert_eq!(config.provider, LlmProvider::Ollama);
        assert_eq!(config.model, "qwen3:4b-instruct-2507-q4_K_M");
        assert_eq!(config.endpoint, Some("http://custom:11434".to_string()));
    }

    #[test]
    fn test_azure_config() {
        let config = LlmProviderConfig::azure(
            "https://my-resource.openai.azure.com",
            "azure-key",
            "gpt-4-deployment",
            "2024-02-01",
        );

        assert_eq!(config.provider, LlmProvider::AzureOpenAI);
        assert_eq!(config.azure_api_version, Some("2024-02-01".to_string()));
    }

    #[test]
    fn test_default_config() {
        let config = LlmProviderConfig::default();
        assert_eq!(config.provider, LlmProvider::Claude);
        assert_eq!(config.model, "opus");
        assert_eq!(config.max_tokens, 1024);
    }
}
