//! LLM request/response types
//!
//! Common types for interacting with language models (Ollama, Claude, OpenAI, etc.)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LLM generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Messages for chat completion
    pub messages: Vec<Message>,
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Temperature (0.0 - 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Enable streaming
    #[serde(default)]
    pub stream: bool,
    /// Model override (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: None,
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: false,
            model: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }
}

impl GenerateRequest {
    /// Create a new request with a system message
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
            ..Default::default()
        }
    }

    /// Add a user message
    pub fn with_user_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::user(content));
        self
    }

    /// Add an assistant message
    pub fn with_assistant_message(mut self, content: impl Into<String>) -> Self {
        self.messages.push(Message::assistant(content));
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 2.0));
        self
    }

    /// Enable streaming
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = stream;
        self
    }

    /// Set model
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    /// Name of the participant (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool call ID (for tool responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    /// Create a tool response message
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// LLM generation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    /// Generated text
    pub text: String,
    /// Finish reason
    pub finish_reason: FinishReason,
    /// Token usage
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    /// Tool calls (if any)
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

impl GenerateResponse {
    /// Create a simple text response
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            text: content.into(),
            finish_reason: FinishReason::Stop,
            usage: None,
            tool_calls: Vec::new(),
        }
    }

    /// Check if response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Finish reason
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// Normal completion
    #[default]
    Stop,
    /// Hit max tokens limit
    Length,
    /// Model wants to call tools
    ToolCalls,
    /// Content was filtered
    ContentFilter,
    /// Error occurred
    Error,
}

/// Token usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    /// Create from counts
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Stream chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Text delta
    pub delta: String,
    /// Whether this is the final chunk
    #[serde(default)]
    pub is_final: bool,
    /// Finish reason (only on final chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

impl StreamChunk {
    /// Create a text chunk
    pub fn text(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            is_final: false,
            finish_reason: None,
        }
    }

    /// Create final chunk
    pub fn final_chunk(finish_reason: FinishReason) -> Self {
        Self {
            delta: String::new(),
            is_final: true,
            finish_reason: Some(finish_reason),
        }
    }
}

/// Tool definition for function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// JSON schema for parameters
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call ID
    pub id: String,
    /// Tool name
    pub name: String,
    /// Arguments (JSON)
    pub arguments: HashMap<String, serde_json::Value>,
}

impl ToolCall {
    /// Get argument as string
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.arguments.get(key).and_then(|v| v.as_str())
    }

    /// Get argument as i64
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.arguments.get(key).and_then(|v| v.as_i64())
    }

    /// Get argument as f64
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.arguments.get(key).and_then(|v| v.as_f64())
    }

    /// Get argument as bool
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.arguments.get(key).and_then(|v| v.as_bool())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_builder() {
        let req = GenerateRequest::new("You are a helpful assistant")
            .with_user_message("Hello")
            .with_max_tokens(100)
            .with_temperature(0.5)
            .with_streaming(true);

        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.max_tokens, Some(100));
        assert_eq!(req.temperature, Some(0.5));
        assert!(req.stream);
    }

    #[test]
    fn test_message_creation() {
        let sys = Message::system("System prompt");
        let user = Message::user("User message");
        let asst = Message::assistant("Assistant response");

        assert_eq!(sys.role, Role::System);
        assert_eq!(user.role, Role::User);
        assert_eq!(asst.role, Role::Assistant);
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_stream_chunk() {
        let chunk = StreamChunk::text("Hello");
        assert!(!chunk.is_final);
        assert_eq!(chunk.delta, "Hello");

        let final_chunk = StreamChunk::final_chunk(FinishReason::Stop);
        assert!(final_chunk.is_final);
    }
}
