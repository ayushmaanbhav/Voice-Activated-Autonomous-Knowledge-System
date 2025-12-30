//! LLM Backend implementations
//!
//! Supports Ollama with KV cache for multi-turn conversations.
//!
//! ## KV Cache Support
//!
//! The Ollama backend supports KV cache for significant latency reduction in
//! multi-turn conversations. When enabled:
//! - First turn: Full prompt processing (~100-200ms for typical context)
//! - Subsequent turns: Only new tokens processed (~10-50ms saved per turn)
//!
//! Use `OllamaBackend::generate_with_session` for multi-turn conversations.

use async_trait::async_trait;
use parking_lot::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
// P1 FIX: Use centralized constants
use voice_agent_config::constants::endpoints;

use crate::prompt::Message;
use crate::LlmError;

/// LLM configuration
#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Model name/ID
    pub model: String,
    /// API endpoint
    pub endpoint: String,
    /// API key (optional)
    pub api_key: Option<String>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Request timeout
    pub timeout: Duration,
    /// Enable streaming
    pub stream: bool,
    /// P1 FIX: Maximum retry attempts for transient failures
    pub max_retries: u32,
    /// P1 FIX: Initial backoff duration (doubles each retry)
    pub initial_backoff: Duration,
    /// P0 FIX: Keep model loaded in memory between calls.
    /// Values: "5m" (5 minutes), "1h" (1 hour), "-1" (indefinite), "0" (unload immediately)
    /// Default: "5m" - keeps model warm for multi-turn conversations
    pub keep_alive: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "qwen3:4b-instruct-2507-q4_K_M".to_string(),
            endpoint: endpoints::OLLAMA_DEFAULT.to_string(), // P1 FIX: Use centralized constant
            api_key: None,
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            timeout: Duration::from_secs(30),
            stream: true,
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            keep_alive: "5m".to_string(), // P0 FIX: Keep model loaded for 5 minutes
        }
    }
}

/// LLM generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Tokens generated
    pub tokens: usize,
    /// Time to first token (ms)
    pub time_to_first_token_ms: u64,
    /// Total generation time (ms)
    pub total_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f32,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// P0 FIX: Context for KV cache reuse in multi-turn conversations.
    /// Pass this to subsequent calls to avoid re-processing the conversation history.
    pub context: Option<Vec<i64>>,
}

/// Finish reason
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Error,
    Cancelled,
}

/// LLM Backend trait
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Generate a response
    async fn generate(&self, messages: &[Message]) -> Result<GenerationResult, LlmError>;

    /// Generate with streaming
    async fn generate_stream(
        &self,
        messages: &[Message],
        tx: mpsc::Sender<String>,
    ) -> Result<GenerationResult, LlmError>;

    /// Check if model is available
    async fn is_available(&self) -> bool;

    /// Get model name
    fn model_name(&self) -> &str;

    /// Estimate tokens
    ///
    /// P0 FIX: Improved token estimation for multilingual content.
    /// - English text: ~4 chars per token
    /// - Hindi/Devanagari: ~2-3 chars per token (but 3 bytes each in UTF-8)
    /// - Uses unicode grapheme count for accuracy
    fn estimate_tokens(&self, text: &str) -> usize {
        use unicode_segmentation::UnicodeSegmentation;

        // Count actual grapheme clusters (handles Devanagari properly)
        let grapheme_count = text.graphemes(true).count();

        // Count Devanagari characters (U+0900 to U+097F)
        let devanagari_count = text
            .chars()
            .filter(|c| ('\u{0900}'..='\u{097F}').contains(c))
            .count();

        // Estimate: Devanagari has ~2 graphemes per token, English ~4
        if devanagari_count > grapheme_count / 3 {
            // Primarily Hindi/Devanagari text
            // Each Hindi word is roughly 1.5-2 tokens, grapheme count / 2
            grapheme_count.max(1) / 2
        } else {
            // Primarily English or mixed
            // Average ~4 characters per token
            grapheme_count.max(1) / 4
        }
    }
}

/// Ollama backend with KV cache support
///
/// P0 FIX: Now supports KV cache for multi-turn conversations.
/// The context is stored internally and reused across calls within a session.
///
/// P2 FIX: Added Clone for composability (e.g., sharing backend across agents).
/// reqwest::Client and Arc<Mutex<_>> are both Clone-safe.
#[derive(Clone)]
pub struct OllamaBackend {
    client: Client,
    config: LlmConfig,
    /// P0 FIX: Cached context for KV cache reuse
    /// Stores the context from the last generation for multi-turn conversations
    session_context: Arc<Mutex<Option<Vec<i64>>>>,
}

impl OllamaBackend {
    /// Create a new Ollama backend
    ///
    /// P1 FIX: Now returns Result instead of panicking on client creation failure.
    pub fn new(config: LlmConfig) -> Result<Self, LlmError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LlmError::Configuration(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            client,
            config,
            session_context: Arc::new(Mutex::new(None)),
        })
    }

    /// Build the API URL
    fn api_url(&self, path: &str) -> String {
        format!("{}/api{}", self.config.endpoint, path)
    }

    /// P0 FIX: Generate with session context for KV cache reuse.
    ///
    /// This method maintains conversation context between calls, significantly
    /// reducing latency for multi-turn conversations by reusing the KV cache.
    ///
    /// First call: Full prompt processing
    /// Subsequent calls: Only new tokens are processed (2-5x faster)
    pub async fn generate_with_session(
        &self,
        messages: &[Message],
    ) -> Result<GenerationResult, LlmError> {
        let context = self.session_context.lock().clone();
        let result = self
            .generate_with_context(messages, context.as_deref())
            .await?;

        // Store the new context for next call
        if let Some(ref ctx) = result.context {
            *self.session_context.lock() = Some(ctx.clone());
        }

        Ok(result)
    }

    /// P0 FIX: Generate with explicit context (for advanced use cases).
    ///
    /// Use this when you need to manage context manually, e.g., for branching
    /// conversations or context switching.
    pub async fn generate_with_context(
        &self,
        messages: &[Message],
        context: Option<&[i64]>,
    ) -> Result<GenerationResult, LlmError> {
        let start = std::time::Instant::now();

        let request = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: messages.iter().map(|m| m.into()).collect(),
            stream: false,
            options: Some(OllamaOptions {
                temperature: Some(self.config.temperature),
                top_p: Some(self.config.top_p),
                num_predict: Some(self.config.max_tokens as i32),
            }),
            keep_alive: Some(self.config.keep_alive.clone()),
            context: context.map(|c| c.to_vec()),
            think: Some(false), // Disable extended thinking for faster responses
        };

        // Retry loop with exponential backoff
        let mut last_error = None;
        let mut backoff = self.config.initial_backoff;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tracing::warn!(
                    "LLM request failed, retrying in {:?} (attempt {}/{})",
                    backoff,
                    attempt,
                    self.config.max_retries
                );
                tokio::time::sleep(backoff).await;
                backoff *= 2;
            }

            match self.execute_request(&request).await {
                Ok(result) => {
                    let total_time = start.elapsed();
                    return Ok(GenerationResult {
                        text: result.message.content,
                        tokens: result.eval_count.unwrap_or(0) as usize,
                        time_to_first_token_ms: result.prompt_eval_duration.unwrap_or(0)
                            / 1_000_000,
                        total_time_ms: total_time.as_millis() as u64,
                        tokens_per_second: result.eval_count.unwrap_or(0) as f32
                            / (result.eval_duration.unwrap_or(1) as f32 / 1e9),
                        finish_reason: if result.done {
                            FinishReason::Stop
                        } else {
                            FinishReason::Length
                        },
                        context: result.context, // P0 FIX: Capture context for reuse
                    });
                },
                Err(e) if Self::is_retryable(&e) => {
                    last_error = Some(e);
                },
                Err(e) => {
                    return Err(e);
                },
            }
        }

        Err(last_error.unwrap_or_else(|| LlmError::Network("Max retries exceeded".to_string())))
    }

    /// P0 FIX: Clear the session context.
    ///
    /// Call this when starting a new conversation to ensure fresh context.
    pub fn clear_session(&self) {
        *self.session_context.lock() = None;
    }

    /// P0 FIX: Check if there's an active session context.
    pub fn has_session_context(&self) -> bool {
        self.session_context.lock().is_some()
    }

    /// P1 FIX: Execute a single request (used by retry logic)
    async fn execute_request(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<OllamaChatResponse, LlmError> {
        let response = self
            .client
            .post(self.api_url("/chat"))
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error = response.text().await.unwrap_or_default();
            // 5xx errors are retryable, 4xx are not
            if status.is_server_error() {
                return Err(LlmError::Network(format!(
                    "Server error {}: {}",
                    status, error
                )));
            }
            return Err(LlmError::Api(error));
        }

        response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))
    }

    /// P1 FIX: Check if an error is retryable
    fn is_retryable(error: &LlmError) -> bool {
        matches!(error, LlmError::Network(_) | LlmError::Timeout)
    }
}

#[async_trait]
impl LlmBackend for OllamaBackend {
    /// Generate a response with retry logic for transient failures
    ///
    /// P1 FIX: Implements exponential backoff retry for network errors.
    /// P0 FIX: Now includes keep_alive for model caching.
    ///
    /// Note: For multi-turn conversations with KV cache reuse, use
    /// `generate_with_session()` instead for 2-5x latency improvement.
    async fn generate(&self, messages: &[Message]) -> Result<GenerationResult, LlmError> {
        // Use generate_with_context with no context (stateless call)
        // This still benefits from keep_alive (model stays loaded)
        self.generate_with_context(messages, None).await
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
        tx: mpsc::Sender<String>,
    ) -> Result<GenerationResult, LlmError> {
        let start = std::time::Instant::now();
        let mut first_token_time = None;
        let mut total_tokens = 0;
        let mut full_response = String::new();
        let mut final_context = None;

        // P0 FIX: Get cached context for streaming too
        let cached_context = self.session_context.lock().clone();

        let request = OllamaChatRequest {
            model: self.config.model.clone(),
            messages: messages.iter().map(|m| m.into()).collect(),
            stream: true,
            options: Some(OllamaOptions {
                temperature: Some(self.config.temperature),
                top_p: Some(self.config.top_p),
                num_predict: Some(self.config.max_tokens as i32),
            }),
            keep_alive: Some(self.config.keep_alive.clone()),
            context: cached_context,
            think: Some(false), // Disable extended thinking for faster responses
        };

        let response = self
            .client
            .post(self.api_url("/chat"))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(error));
        }

        // Read streaming response
        let mut stream = response.bytes_stream();
        use futures::StreamExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);

            // Parse each line (NDJSON)
            for line in text.lines() {
                if line.is_empty() {
                    continue;
                }

                if let Ok(chunk_response) = serde_json::from_str::<OllamaStreamChunk>(line) {
                    if first_token_time.is_none() {
                        first_token_time = Some(start.elapsed());
                    }

                    let token = &chunk_response.message.content;
                    full_response.push_str(token);
                    total_tokens += 1;

                    // P0 FIX: Capture context from final chunk
                    if chunk_response.done {
                        final_context = chunk_response.context;
                    }

                    // Send token to channel
                    if tx.send(token.clone()).await.is_err() {
                        // Channel closed, generation cancelled
                        return Ok(GenerationResult {
                            text: full_response,
                            tokens: total_tokens,
                            time_to_first_token_ms: first_token_time
                                .map(|t| t.as_millis() as u64)
                                .unwrap_or(0),
                            total_time_ms: start.elapsed().as_millis() as u64,
                            tokens_per_second: 0.0,
                            finish_reason: FinishReason::Cancelled,
                            context: final_context,
                        });
                    }

                    if chunk_response.done {
                        break;
                    }
                }
            }
        }

        // P0 FIX: Update session context for next call
        if let Some(ref ctx) = final_context {
            *self.session_context.lock() = Some(ctx.clone());
        }

        let total_time = start.elapsed();

        Ok(GenerationResult {
            text: full_response,
            tokens: total_tokens,
            time_to_first_token_ms: first_token_time.map(|t| t.as_millis() as u64).unwrap_or(0),
            total_time_ms: total_time.as_millis() as u64,
            tokens_per_second: total_tokens as f32 / total_time.as_secs_f32(),
            finish_reason: FinishReason::Stop,
            context: final_context,
        })
    }

    async fn is_available(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", self.config.endpoint))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

// Ollama API types
#[derive(Debug, Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    /// P0 FIX: Keep model loaded in memory
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
    /// P0 FIX: Context from previous response for KV cache reuse
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<Vec<i64>>,
    /// Disable extended thinking for models like qwen3/deepseek-r1
    #[serde(skip_serializing_if = "Option::is_none")]
    think: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

impl From<&Message> for OllamaMessage {
    fn from(msg: &Message) -> Self {
        Self {
            role: msg.role.to_string(),
            content: msg.content.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
    done: bool,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    /// P0 FIX: Context for KV cache reuse in subsequent calls
    #[serde(default)]
    context: Option<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: OllamaMessage,
    done: bool,
    /// P0 FIX: Context is returned in the final chunk (when done=true)
    #[serde(default)]
    context: Option<Vec<i64>>,
}

// =============================================================================
// P2 FIX: OpenAI-compatible Backend
// =============================================================================

/// Configuration for OpenAI-compatible backends
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// API endpoint (OpenAI: https://api.openai.com/v1, Azure: custom)
    pub endpoint: String,
    /// API key
    pub api_key: String,
    /// Model name (gpt-4, gpt-3.5-turbo, claude-3-sonnet, etc.)
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature (0-2)
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Request timeout
    pub timeout: Duration,
    /// Enable streaming
    pub stream: bool,
    /// Organization ID (OpenAI specific)
    pub organization: Option<String>,
    /// Azure API version (Azure specific)
    pub api_version: Option<String>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "gpt-3.5-turbo".to_string(),
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            timeout: Duration::from_secs(30),
            stream: true,
            organization: None,
            api_version: None,
        }
    }
}

impl OpenAIConfig {
    /// Create config for OpenAI
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            ..Default::default()
        }
    }

    /// Create config for Azure OpenAI
    pub fn azure(
        endpoint: impl Into<String>,
        api_key: impl Into<String>,
        deployment: impl Into<String>,
        api_version: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: api_key.into(),
            model: deployment.into(),
            api_version: Some(api_version.into()),
            ..Default::default()
        }
    }

    /// Create config for local OpenAI-compatible server (vLLM, Ollama, etc.)
    pub fn local(endpoint: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            api_key: "not-needed".to_string(),
            model: model.into(),
            ..Default::default()
        }
    }
}

/// OpenAI-compatible backend
///
/// Works with:
/// - OpenAI (GPT-4, GPT-3.5)
/// - Azure OpenAI
/// - Claude via Anthropic (using OpenAI-compatible mode)
/// - vLLM
/// - Local servers with OpenAI-compatible APIs
pub struct OpenAIBackend {
    config: OpenAIConfig,
    client: Client,
}

impl OpenAIBackend {
    /// Create new OpenAI backend
    pub fn new(config: OpenAIConfig) -> Result<Self, LlmError> {
        if config.api_key.is_empty() && !config.endpoint.starts_with("http://localhost") {
            return Err(LlmError::Configuration(
                "API key required for remote endpoints".to_string(),
            ));
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| LlmError::Network(e.to_string()))?;

        Ok(Self { config, client })
    }

    /// Get the full API URL for chat completions
    fn chat_url(&self) -> String {
        if let Some(ref api_version) = self.config.api_version {
            // Azure format: {endpoint}/openai/deployments/{model}/chat/completions?api-version={version}
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.config.endpoint.trim_end_matches('/'),
                self.config.model,
                api_version
            )
        } else {
            // Standard OpenAI format
            format!(
                "{}/chat/completions",
                self.config.endpoint.trim_end_matches('/')
            )
        }
    }

    /// Build request headers
    fn build_headers(&self) -> reqwest::header::HeaderMap {
        use reqwest::header::HeaderValue;

        let mut headers = reqwest::header::HeaderMap::new();

        if self.config.api_version.is_some() {
            // Azure uses api-key header
            if let Ok(val) = HeaderValue::from_str(&self.config.api_key) {
                headers.insert("api-key", val);
            }
        } else {
            // OpenAI uses Authorization header
            let auth_value = format!("Bearer {}", self.config.api_key);
            if let Ok(val) = HeaderValue::from_str(&auth_value) {
                headers.insert(reqwest::header::AUTHORIZATION, val);
            }
        }

        if let Some(ref org) = self.config.organization {
            if let Ok(val) = HeaderValue::from_str(org) {
                headers.insert("OpenAI-Organization", val);
            }
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        headers
    }
}

#[async_trait]
impl LlmBackend for OpenAIBackend {
    async fn generate(&self, messages: &[Message]) -> Result<GenerationResult, LlmError> {
        let start = std::time::Instant::now();

        let openai_messages: Vec<OpenAIMessage> = messages
            .iter()
            .map(|m| OpenAIMessage {
                role: match m.role {
                    crate::prompt::Role::System => "system".to_string(),
                    crate::prompt::Role::User => "user".to_string(),
                    crate::prompt::Role::Assistant => "assistant".to_string(),
                    crate::prompt::Role::Tool => "tool".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let request = OpenAIChatRequest {
            model: self.config.model.clone(),
            messages: openai_messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            top_p: Some(self.config.top_p),
            stream: Some(false),
        };

        let response = self
            .client
            .post(self.chat_url())
            .headers(self.build_headers())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        let response: OpenAIChatResponse = response
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;

        let choice = response
            .choices
            .first()
            .ok_or_else(|| LlmError::InvalidResponse("No choices in response".to_string()))?;

        let total_time_ms = start.elapsed().as_millis() as u64;
        let tokens = response.usage.map(|u| u.completion_tokens).unwrap_or(0);

        Ok(GenerationResult {
            text: choice.message.content.clone(),
            tokens,
            time_to_first_token_ms: total_time_ms, // Non-streaming, so same as total
            total_time_ms,
            tokens_per_second: if total_time_ms > 0 {
                tokens as f32 / (total_time_ms as f32 / 1000.0)
            } else {
                0.0
            },
            finish_reason: match choice.finish_reason.as_deref() {
                Some("stop") => FinishReason::Stop,
                Some("length") => FinishReason::Length,
                _ => FinishReason::Stop,
            },
            context: None, // OpenAI doesn't expose KV cache
        })
    }

    async fn generate_stream(
        &self,
        messages: &[Message],
        tx: mpsc::Sender<String>,
    ) -> Result<GenerationResult, LlmError> {
        let start = std::time::Instant::now();
        let mut first_token_time: Option<u64> = None;
        let mut full_text = String::new();
        let mut token_count = 0;

        let openai_messages: Vec<OpenAIMessage> = messages
            .iter()
            .map(|m| OpenAIMessage {
                role: match m.role {
                    crate::prompt::Role::System => "system".to_string(),
                    crate::prompt::Role::User => "user".to_string(),
                    crate::prompt::Role::Assistant => "assistant".to_string(),
                    crate::prompt::Role::Tool => "tool".to_string(),
                },
                content: m.content.clone(),
            })
            .collect();

        let request = OpenAIChatRequest {
            model: self.config.model.clone(),
            messages: openai_messages,
            max_tokens: Some(self.config.max_tokens),
            temperature: Some(self.config.temperature),
            top_p: Some(self.config.top_p),
            stream: Some(true),
        };

        let response = self
            .client
            .post(self.chat_url())
            .headers(self.build_headers())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(LlmError::Api(format!("HTTP {}: {}", status, error_text)));
        }

        // Process SSE stream
        use futures::StreamExt;
        let mut stream = response.bytes_stream();

        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| LlmError::Network(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }

                if let Some(json_str) = line.strip_prefix("data: ") {
                    if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(json_str) {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(ref delta) = choice.delta {
                                if let Some(ref content) = delta.content {
                                    if first_token_time.is_none() {
                                        first_token_time = Some(start.elapsed().as_millis() as u64);
                                    }
                                    full_text.push_str(content);
                                    token_count += 1;
                                    let _ = tx.send(content.clone()).await;
                                }
                            }
                        }
                    }
                }
            }
        }

        let total_time_ms = start.elapsed().as_millis() as u64;

        Ok(GenerationResult {
            text: full_text,
            tokens: token_count,
            time_to_first_token_ms: first_token_time.unwrap_or(total_time_ms),
            total_time_ms,
            tokens_per_second: if total_time_ms > 0 {
                token_count as f32 / (total_time_ms as f32 / 1000.0)
            } else {
                0.0
            },
            finish_reason: FinishReason::Stop,
            context: None,
        })
    }

    async fn is_available(&self) -> bool {
        // Try a simple models list request for non-Azure
        if self.config.api_version.is_none() {
            let url = format!("{}/models", self.config.endpoint.trim_end_matches('/'));
            self.client
                .get(&url)
                .headers(self.build_headers())
                .timeout(Duration::from_secs(5))
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false)
        } else {
            // For Azure, try a lightweight request
            true // Assume available; actual check would need deployment-specific logic
        }
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }
}

// OpenAI API types
#[derive(Debug, Serialize)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    completion_tokens: usize,
    #[allow(dead_code)]
    prompt_tokens: usize,
    #[allow(dead_code)]
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: Option<OpenAIDelta>,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::Role;

    #[test]
    fn test_config_default() {
        let config = LlmConfig::default();
        assert!(config.stream);
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.keep_alive, "5m"); // P0 FIX: Verify keep_alive default
    }

    #[test]
    fn test_message_conversion() {
        let msg = Message {
            role: Role::User,
            content: "Hello".to_string(),
            name: None,
            tool_call_id: None,
        };
        let ollama_msg: OllamaMessage = (&msg).into();
        assert_eq!(ollama_msg.role, "user");
        assert_eq!(ollama_msg.content, "Hello");
    }

    #[test]
    fn test_session_context_management() {
        // P0 FIX: Test session context management
        let backend = OllamaBackend::new(LlmConfig::default()).unwrap();

        // Initially no context
        assert!(!backend.has_session_context());

        // Simulate storing context
        *backend.session_context.lock() = Some(vec![1, 2, 3, 4, 5]);
        assert!(backend.has_session_context());

        // Clear session
        backend.clear_session();
        assert!(!backend.has_session_context());
    }

    #[test]
    fn test_ollama_request_serialization() {
        // P0 FIX: Verify request includes keep_alive and context
        let request = OllamaChatRequest {
            model: "test".to_string(),
            messages: vec![],
            stream: false,
            options: None,
            keep_alive: Some("5m".to_string()),
            context: Some(vec![1, 2, 3]),
            think: Some(false),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("keep_alive"));
        assert!(json.contains("context"));
    }

    // P2 FIX: OpenAI backend tests

    #[test]
    fn test_openai_config_default() {
        let config = OpenAIConfig::default();
        assert_eq!(config.endpoint, "https://api.openai.com/v1");
        assert_eq!(config.model, "gpt-3.5-turbo");
        assert!(config.api_key.is_empty());
    }

    #[test]
    fn test_openai_config_openai() {
        let config = OpenAIConfig::openai("sk-xxx", "gpt-4");
        assert_eq!(config.api_key, "sk-xxx");
        assert_eq!(config.model, "gpt-4");
        assert!(config.api_version.is_none());
    }

    #[test]
    fn test_openai_config_azure() {
        let config = OpenAIConfig::azure(
            "https://my-resource.openai.azure.com",
            "azure-key",
            "gpt-4-deployment",
            "2024-02-01",
        );
        assert!(config.api_version.is_some());
        assert_eq!(config.model, "gpt-4-deployment");
    }

    #[test]
    fn test_openai_config_local() {
        let config = OpenAIConfig::local("http://localhost:8000/v1", "llama-3");
        assert_eq!(config.endpoint, "http://localhost:8000/v1");
        assert_eq!(config.api_key, "not-needed");
    }

    #[test]
    fn test_openai_backend_creation() {
        // Local endpoint should work without API key
        let config = OpenAIConfig::local("http://localhost:8000", "test");
        let backend = OpenAIBackend::new(config);
        assert!(backend.is_ok());

        // Remote endpoint requires API key
        let config = OpenAIConfig::default();
        let backend = OpenAIBackend::new(config);
        assert!(backend.is_err());

        // With API key should work
        let config = OpenAIConfig::openai("sk-xxx", "gpt-4");
        let backend = OpenAIBackend::new(config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_openai_chat_url() {
        // Standard OpenAI URL
        let config = OpenAIConfig::openai("sk-xxx", "gpt-4");
        let backend = OpenAIBackend::new(config).unwrap();
        assert_eq!(
            backend.chat_url(),
            "https://api.openai.com/v1/chat/completions"
        );

        // Azure URL format
        let config = OpenAIConfig::azure(
            "https://myresource.openai.azure.com",
            "key",
            "deployment",
            "2024-02-01",
        );
        let backend = OpenAIBackend::new(config).unwrap();
        assert!(backend.chat_url().contains("openai/deployments/deployment"));
        assert!(backend.chat_url().contains("api-version=2024-02-01"));
    }

    #[test]
    fn test_openai_request_serialization() {
        let request = OpenAIChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(256),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(false),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("gpt-4"));
        assert!(json.contains("Hello"));
        assert!(json.contains("max_tokens"));
    }
}
