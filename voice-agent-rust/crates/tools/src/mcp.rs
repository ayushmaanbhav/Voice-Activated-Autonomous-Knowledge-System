//! MCP (Model Context Protocol) Tool Interface
//!
//! Provides a standardized tool interface compatible with MCP specification.

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool error with MCP error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    /// Error code (MCP compatible)
    pub code: ErrorCode,
    /// Human-readable message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolError {
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::InvalidParams,
            message: message.into(),
            data: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::InternalError,
            message: message.into(),
            data: None,
        }
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::MethodNotFound,
            message: message.into(),
            data: None,
        }
    }

    /// P1 FIX: Timeout error for tool execution
    pub fn timeout(tool_name: &str, timeout_secs: u64) -> Self {
        Self {
            code: ErrorCode::InternalError,
            message: format!("Tool '{}' timed out after {}s", tool_name, timeout_secs),
            data: None,
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)
    }
}

impl std::error::Error for ToolError {}

/// MCP Error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "i32", try_from = "i32")]
pub enum ErrorCode {
    ParseError,
    InvalidRequest,
    MethodNotFound,
    InvalidParams,
    InternalError,
    /// Custom error range
    Custom(i32),
}

impl From<ErrorCode> for i32 {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::ParseError => -32700,
            ErrorCode::InvalidRequest => -32600,
            ErrorCode::MethodNotFound => -32601,
            ErrorCode::InvalidParams => -32602,
            ErrorCode::InternalError => -32603,
            ErrorCode::Custom(c) => c,
        }
    }
}

impl TryFrom<i32> for ErrorCode {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(match value {
            -32700 => ErrorCode::ParseError,
            -32600 => ErrorCode::InvalidRequest,
            -32601 => ErrorCode::MethodNotFound,
            -32602 => ErrorCode::InvalidParams,
            -32603 => ErrorCode::InternalError,
            c => ErrorCode::Custom(c),
        })
    }
}

/// Tool input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    /// Tool name
    pub name: String,
    /// Input arguments
    pub arguments: Value,
}

/// Tool output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Output content
    pub content: Vec<ContentBlock>,
    /// Is this an error response?
    #[serde(default)]
    pub is_error: bool,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: false,
        }
    }

    pub fn json(value: impl Serialize) -> Self {
        let text = serde_json::to_string_pretty(&value).unwrap_or_default();
        Self {
            content: vec![ContentBlock::Text { text }],
            is_error: false,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: message.into() }],
            is_error: true,
        }
    }

    /// P2 FIX: Create an audio output
    pub fn audio(
        data: impl Into<String>,
        mime_type: impl Into<String>,
        sample_rate: Option<u32>,
        duration_ms: Option<u64>,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Audio {
                data: data.into(),
                mime_type: mime_type.into(),
                sample_rate,
                duration_ms,
            }],
            is_error: false,
        }
    }
}

/// Content block types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { uri: String, mime_type: Option<String> },
    /// P2 FIX: Audio content block for voice response support.
    /// Supports base64-encoded audio data with sample rate and format info.
    Audio {
        /// Base64-encoded audio data
        data: String,
        /// MIME type (e.g., "audio/wav", "audio/mp3", "audio/opus")
        mime_type: String,
        /// Sample rate in Hz (e.g., 16000, 22050, 44100)
        #[serde(skip_serializing_if = "Option::is_none")]
        sample_rate: Option<u32>,
        /// Duration in milliseconds
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
}

/// Tool schema (JSON Schema format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: InputSchema,
}

/// Input schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    #[serde(default)]
    pub properties: HashMap<String, PropertySchema>,
    #[serde(default)]
    pub required: Vec<String>,
}

impl InputSchema {
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
        }
    }

    pub fn property(mut self, name: &str, schema: PropertySchema, required: bool) -> Self {
        self.properties.insert(name.to_string(), schema);
        if required {
            self.required.push(name.to_string());
        }
        self
    }
}

/// Property schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    #[serde(rename = "type")]
    pub prop_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
}

impl PropertySchema {
    pub fn string(description: impl Into<String>) -> Self {
        Self {
            prop_type: "string".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    pub fn number(description: impl Into<String>) -> Self {
        Self {
            prop_type: "number".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    pub fn integer(description: impl Into<String>) -> Self {
        Self {
            prop_type: "integer".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    pub fn boolean(description: impl Into<String>) -> Self {
        Self {
            prop_type: "boolean".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    pub fn enum_type(description: impl Into<String>, values: Vec<String>) -> Self {
        Self {
            prop_type: "string".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: Some(values),
            minimum: None,
            maximum: None,
        }
    }

    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.minimum = Some(min);
        self.maximum = Some(max);
        self
    }
}

/// Tool trait
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get tool name
    fn name(&self) -> &str;

    /// Get tool description
    fn description(&self) -> &str;

    /// Get input schema
    fn schema(&self) -> ToolSchema;

    /// Execute the tool
    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError>;

    /// Validate input (optional, default uses schema)
    ///
    /// P2 FIX: Enhanced validation that checks types, enum values, and ranges,
    /// not just required fields.
    fn validate(&self, input: &Value) -> Result<(), ToolError> {
        let schema = self.schema();

        if let Value::Object(obj) = input {
            // Check required fields
            for required in &schema.input_schema.required {
                if !obj.contains_key(required) {
                    return Err(ToolError::invalid_params(format!(
                        "Missing required field: {}",
                        required
                    )));
                }
            }

            // P2 FIX: Validate each property's type and constraints
            for (name, value) in obj {
                if let Some(prop_schema) = schema.input_schema.properties.get(name) {
                    validate_property(name, value, prop_schema)?;
                }
                // Unknown properties are allowed (no additionalProperties: false)
            }

            Ok(())
        } else if schema.input_schema.properties.is_empty() {
            // No properties required
            Ok(())
        } else {
            Err(ToolError::invalid_params("Input must be an object"))
        }
    }
}

/// P2 FIX: Validate a property value against its schema
fn validate_property(name: &str, value: &Value, schema: &PropertySchema) -> Result<(), ToolError> {
    // Check type
    let type_valid = match schema.prop_type.as_str() {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true, // Unknown types pass through
    };

    if !type_valid {
        return Err(ToolError::invalid_params(format!(
            "Field '{}' must be of type '{}', got '{}'",
            name,
            schema.prop_type,
            json_type_name(value)
        )));
    }

    // Check enum values
    if let Some(enum_values) = &schema.enum_values {
        if let Some(s) = value.as_str() {
            if !enum_values.contains(&s.to_string()) {
                return Err(ToolError::invalid_params(format!(
                    "Field '{}' must be one of: [{}], got '{}'",
                    name,
                    enum_values.join(", "),
                    s
                )));
            }
        }
    }

    // Check numeric range
    if let Some(num) = value.as_f64() {
        if let Some(min) = schema.minimum {
            if num < min {
                return Err(ToolError::invalid_params(format!(
                    "Field '{}' must be >= {}, got {}",
                    name, min, num
                )));
            }
        }
        if let Some(max) = schema.maximum {
            if num > max {
                return Err(ToolError::invalid_params(format!(
                    "Field '{}' must be <= {}, got {}",
                    name, max, num
                )));
            }
        }
    }

    Ok(())
}

/// Get a human-readable type name for a JSON value
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_schema() {
        let schema = InputSchema::object()
            .property("name", PropertySchema::string("Customer name"), true)
            .property("amount", PropertySchema::number("Loan amount"), true)
            .property("city", PropertySchema::string("City"), false);

        assert_eq!(schema.properties.len(), 3);
        assert_eq!(schema.required.len(), 2);
    }

    #[test]
    fn test_tool_output() {
        let output = ToolOutput::text("Hello world");
        assert!(!output.is_error);
        assert_eq!(output.content.len(), 1);

        let error = ToolOutput::error("Something went wrong");
        assert!(error.is_error);
    }

    #[test]
    fn test_tool_error() {
        let err = ToolError::invalid_params("Bad input");
        assert_eq!(err.code, ErrorCode::InvalidParams);
    }

    #[test]
    fn test_validate_property_type() {
        let string_schema = PropertySchema::string("Test");

        // Valid string
        assert!(validate_property("name", &serde_json::json!("hello"), &string_schema).is_ok());

        // Invalid: number instead of string
        assert!(validate_property("name", &serde_json::json!(123), &string_schema).is_err());
    }

    #[test]
    fn test_validate_enum() {
        let enum_schema = PropertySchema::enum_type("Status", vec!["active".into(), "inactive".into()]);

        // Valid enum value
        assert!(validate_property("status", &serde_json::json!("active"), &enum_schema).is_ok());

        // Invalid enum value
        let result = validate_property("status", &serde_json::json!("unknown"), &enum_schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("must be one of"));
    }

    #[test]
    fn test_validate_range() {
        let range_schema = PropertySchema::number("Amount").with_range(100.0, 1000.0);

        // Valid: within range
        assert!(validate_property("amount", &serde_json::json!(500), &range_schema).is_ok());

        // Invalid: below minimum
        let result = validate_property("amount", &serde_json::json!(50), &range_schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains(">="));

        // Invalid: above maximum
        let result = validate_property("amount", &serde_json::json!(2000), &range_schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("<="));
    }
}
