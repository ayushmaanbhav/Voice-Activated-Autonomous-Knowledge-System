//! Tool Schema Configuration
//!
//! Defines config-driven tool schemas for LLM function calling.
//! Provides conversion to core::ToolSchema for use by Tool implementations.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;

// Import core types for schema conversion
use voice_agent_core::traits::{
    InputSchema as CoreInputSchema, PropertySchema as CorePropertySchema,
    ToolSchema as CoreToolSchema,
};

/// Tools configuration loaded from tools/schemas.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    /// Tool definitions keyed by tool name
    #[serde(default)]
    pub tools: HashMap<String, ToolSchema>,
    /// Usage guidelines for the LLM
    #[serde(default)]
    pub usage_guidelines: HashMap<String, String>,
    /// P16 FIX: Intent to tool mapping
    /// Maps intent names to tool configurations
    #[serde(default)]
    pub intent_to_tool: HashMap<String, IntentToolMapping>,
    /// P16 FIX: Slot name aliases for normalization
    #[serde(default)]
    pub slot_aliases: HashMap<String, String>,
    /// P16 FIX: Tool argument defaults
    #[serde(default)]
    pub tool_defaults: HashMap<String, HashMap<String, serde_json::Value>>,
    /// P16 FIX: Tool argument name mappings
    #[serde(default)]
    pub argument_mappings: HashMap<String, HashMap<String, String>>,
}

/// P16 FIX: Mapping from intent to tool with optional conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentToolMapping {
    /// Tool to call for this intent
    pub tool: String,
    /// Required slots that must be present to trigger the tool
    #[serde(default)]
    pub required_slots: Vec<String>,
    /// Alternative tool if required slots are not present
    #[serde(default)]
    pub fallback_tool: Option<String>,
    /// Aliases for this intent (will be auto-expanded to create additional mappings)
    #[serde(default)]
    pub aliases: Vec<String>,
}

/// P16 FIX: Extended intent-to-tool config loaded from intent_tool_mappings.yaml
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntentToolMappingsConfig {
    /// Intent to tool mappings (with aliases)
    #[serde(default)]
    pub intent_to_tool: HashMap<String, IntentToolMapping>,
    /// Slot name aliases for normalization
    #[serde(default)]
    pub slot_aliases: HashMap<String, String>,
    /// Tool argument defaults
    #[serde(default)]
    pub tool_defaults: HashMap<String, HashMap<String, serde_json::Value>>,
    /// Tool argument name mappings
    #[serde(default)]
    pub argument_mappings: HashMap<String, HashMap<String, String>>,
}

impl IntentToolMappingsConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ToolsConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ToolsConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content).map_err(|e| ToolsConfigError::ParseError(e.to_string()))
    }

    /// Expand aliases into flat intent_to_tool map
    /// Returns a HashMap where both primary intents and aliases point to their mappings
    pub fn expand_aliases(&self) -> HashMap<String, IntentToolMapping> {
        let mut expanded = HashMap::new();

        for (intent, mapping) in &self.intent_to_tool {
            // Add the primary intent
            expanded.insert(intent.clone(), mapping.clone());

            // Add all aliases pointing to the same mapping
            for alias in &mapping.aliases {
                expanded.insert(alias.clone(), mapping.clone());
            }
        }

        expanded
    }

    /// Normalize a slot name using slot_aliases
    pub fn normalize_slot<'a>(&'a self, slot: &'a str) -> &'a str {
        self.slot_aliases.get(slot).map(|s| s.as_str()).unwrap_or(slot)
    }

    /// Get tool defaults for a tool
    pub fn get_tool_defaults(&self, tool: &str) -> Option<&HashMap<String, serde_json::Value>> {
        self.tool_defaults.get(tool)
    }

    /// Get argument mapping for a tool
    pub fn get_argument_mapping(&self, tool: &str) -> Option<&HashMap<String, String>> {
        self.argument_mappings.get(tool)
    }
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            tools: HashMap::new(),
            usage_guidelines: HashMap::new(),
            intent_to_tool: HashMap::new(),
            slot_aliases: HashMap::new(),
            tool_defaults: HashMap::new(),
            argument_mappings: HashMap::new(),
        }
    }
}

impl ToolsConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ToolsConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ToolsConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content).map_err(|e| ToolsConfigError::ParseError(e.to_string()))
    }

    /// Get a tool schema by name
    pub fn get_tool(&self, name: &str) -> Option<&ToolSchema> {
        self.tools.get(name)
    }

    /// Get all tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get usage guideline by key
    pub fn get_guideline(&self, key: &str) -> Option<&str> {
        self.usage_guidelines.get(key).map(|s| s.as_str())
    }

    /// Convert all tools to JSON Schema format for LLM consumption
    pub fn to_json_schemas(&self) -> Vec<JsonValue> {
        self.tools.values().map(|t| t.to_json_schema()).collect()
    }

    /// Get tool names that are enabled
    pub fn enabled_tool_names(&self) -> Vec<&str> {
        self.tools
            .iter()
            .filter(|(_, t)| t.enabled.unwrap_or(true))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// P16 FIX: Convert to Vec<ToolDefinition> for LLM crate
    ///
    /// Returns tool definitions in the format expected by voice_agent_core::ToolDefinition.
    /// This replaces the hardcoded gold_loan_tools() function in the llm crate.
    pub fn to_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .filter(|(_, t)| t.enabled.unwrap_or(true))
            .map(|(_, schema)| schema.to_tool_definition())
            .collect()
    }

    /// Get tools by category (if categories are added to schema)
    pub fn tools_by_category(&self, category: &str) -> Vec<&str> {
        self.tools
            .iter()
            .filter(|(_, t)| t.category.as_deref() == Some(category))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get a tool's core schema by name for use by Tool trait implementations
    ///
    /// Returns None if the tool is not defined in config.
    /// Tools should call this in their schema() method to get config-driven schemas.
    pub fn get_core_schema(&self, name: &str) -> Option<CoreToolSchema> {
        self.tools.get(name).map(|t| t.to_core_schema())
    }

    // ====== P16 FIX: Intent to Tool Resolution ======

    /// Get the tool mapping for an intent
    pub fn get_intent_mapping(&self, intent: &str) -> Option<&IntentToolMapping> {
        self.intent_to_tool.get(intent)
    }

    /// Resolve which tool to call for an intent, given the available slots
    /// Returns Some(tool_name) if a tool should be called, None otherwise
    pub fn resolve_tool_for_intent(&self, intent: &str, available_slots: &[&str]) -> Option<&str> {
        if let Some(mapping) = self.intent_to_tool.get(intent) {
            // Check if all required slots are present
            let has_required = mapping.required_slots.iter()
                .all(|slot| available_slots.contains(&slot.as_str()));

            if has_required || mapping.required_slots.is_empty() {
                Some(&mapping.tool)
            } else if let Some(ref fallback) = mapping.fallback_tool {
                // Use fallback tool if required slots are missing
                Some(fallback)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Check if intent-to-tool mappings are configured
    pub fn has_intent_mappings(&self) -> bool {
        !self.intent_to_tool.is_empty()
    }

    /// Get all configured intent names
    pub fn mapped_intents(&self) -> Vec<&str> {
        self.intent_to_tool.keys().map(|s| s.as_str()).collect()
    }

    /// Get tool defaults for a tool
    pub fn get_tool_defaults(&self, tool: &str) -> Option<&HashMap<String, serde_json::Value>> {
        self.tool_defaults.get(tool)
    }

    /// Get argument mapping for a tool
    pub fn get_argument_mapping(&self, tool: &str) -> Option<&HashMap<String, String>> {
        self.argument_mappings.get(tool)
    }

    /// Normalize a slot name using slot_aliases
    pub fn normalize_slot<'a>(&'a self, slot: &'a str) -> &'a str {
        self.slot_aliases.get(slot).map(|s| s.as_str()).unwrap_or(slot)
    }
}

/// Simplified tool definition for LLM consumption (matches core::ToolDefinition)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: JsonValue,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: JsonValue) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// Schema definition for a single tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name (identifier)
    pub name: String,
    /// Human-readable description
    pub description: String,
    /// Tool parameters
    #[serde(default)]
    pub parameters: Vec<ToolParameter>,
    /// Whether the tool is enabled (default: true)
    #[serde(default)]
    pub enabled: Option<bool>,
    /// Tool category for grouping (e.g., "calculation", "communication", "crm")
    #[serde(default)]
    pub category: Option<String>,
}

impl ToolSchema {
    /// Convert config ToolSchema to core::ToolSchema for Tool trait implementations
    ///
    /// This allows tools to read their schema from config instead of hardcoding.
    /// All content (names, descriptions, parameters, enums) comes from YAML config.
    pub fn to_core_schema(&self) -> CoreToolSchema {
        let mut input_schema = CoreInputSchema::object();

        for param in &self.parameters {
            let prop_schema = param.to_core_property_schema();
            input_schema = input_schema.property(&param.name, prop_schema, param.required);
        }

        CoreToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema,
        }
    }

    /// P16 FIX: Convert to ToolDefinition for LLM crate consumption
    pub fn to_tool_definition(&self) -> ToolDefinition {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let mut prop = serde_json::Map::new();
            prop.insert("type".to_string(), JsonValue::String(param.param_type.clone()));
            prop.insert("description".to_string(), JsonValue::String(param.description.clone()));

            if let Some(enum_values) = &param.enum_values {
                let values: Vec<JsonValue> = enum_values.iter()
                    .map(|v| JsonValue::String(v.clone()))
                    .collect();
                prop.insert("enum".to_string(), JsonValue::Array(values));
            }

            if let Some(min) = param.min {
                prop.insert("minimum".to_string(), serde_json::json!(min));
            }
            if let Some(max) = param.max {
                prop.insert("maximum".to_string(), serde_json::json!(max));
            }

            properties.insert(param.name.clone(), JsonValue::Object(prop));

            if param.required {
                required.push(JsonValue::String(param.name.clone()));
            }
        }

        let parameters = serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
        });

        ToolDefinition::new(&self.name, &self.description, parameters)
    }

    /// Convert to JSON Schema format compatible with LLM tool_use
    pub fn to_json_schema(&self) -> JsonValue {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in &self.parameters {
            let mut prop = serde_json::Map::new();
            prop.insert(
                "type".to_string(),
                JsonValue::String(param.param_type.clone()),
            );
            prop.insert(
                "description".to_string(),
                JsonValue::String(param.description.clone()),
            );

            // Add enum constraint if present
            if let Some(enum_values) = &param.enum_values {
                let values: Vec<JsonValue> = enum_values
                    .iter()
                    .map(|v| JsonValue::String(v.clone()))
                    .collect();
                prop.insert("enum".to_string(), JsonValue::Array(values));
            }

            // Add min/max constraints for numbers
            if let Some(min) = param.min {
                prop.insert("minimum".to_string(), serde_json::json!(min));
            }
            if let Some(max) = param.max {
                prop.insert("maximum".to_string(), serde_json::json!(max));
            }

            properties.insert(param.name.clone(), JsonValue::Object(prop));

            if param.required {
                required.push(JsonValue::String(param.name.clone()));
            }
        }

        serde_json::json!({
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        })
    }
}

/// Parameter definition for a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type (string, number, integer, boolean, array, object)
    #[serde(rename = "type")]
    pub param_type: String,
    /// Human-readable description
    pub description: String,
    /// Whether the parameter is required
    #[serde(default)]
    pub required: bool,
    /// Enum values for string parameters
    #[serde(rename = "enum", default)]
    pub enum_values: Option<Vec<String>>,
    /// Minimum value for number parameters
    #[serde(default)]
    pub min: Option<f64>,
    /// Maximum value for number parameters
    #[serde(default)]
    pub max: Option<f64>,
    /// Default value
    #[serde(default)]
    pub default: Option<String>,
}

impl ToolParameter {
    /// Convert to core::PropertySchema for use by Tool trait implementations
    ///
    /// Maps config parameter definition to core schema format including:
    /// - Type mapping (string, number, integer, boolean)
    /// - Enum constraints
    /// - Numeric range constraints (min/max)
    /// - Default values
    pub fn to_core_property_schema(&self) -> CorePropertySchema {
        // Create base schema based on type
        let mut schema = match self.param_type.as_str() {
            "string" => {
                if let Some(ref enum_values) = self.enum_values {
                    CorePropertySchema::enum_type(&self.description, enum_values.clone())
                } else {
                    CorePropertySchema::string(&self.description)
                }
            }
            "number" => CorePropertySchema::number(&self.description),
            "integer" => CorePropertySchema::integer(&self.description),
            "boolean" => CorePropertySchema::boolean(&self.description),
            // Default to string for unknown types
            _ => CorePropertySchema::string(&self.description),
        };

        // Add numeric range constraints
        if let (Some(min), Some(max)) = (self.min, self.max) {
            schema = schema.with_range(min, max);
        } else if let Some(min) = self.min {
            schema.minimum = Some(min);
        } else if let Some(max) = self.max {
            schema.maximum = Some(max);
        }

        // Add default value
        if let Some(ref default) = self.default {
            // Try to parse as appropriate type
            let default_value = match self.param_type.as_str() {
                "number" => default.parse::<f64>().ok().map(|v| serde_json::json!(v)),
                "integer" => default.parse::<i64>().ok().map(|v| serde_json::json!(v)),
                "boolean" => default.parse::<bool>().ok().map(|v| serde_json::json!(v)),
                _ => Some(serde_json::json!(default)),
            };
            if let Some(val) = default_value {
                schema = schema.with_default(val);
            }
        }

        schema
    }
}

/// Errors when loading tools configuration
#[derive(Debug)]
pub enum ToolsConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for ToolsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Tools config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse tools config: {}", err),
        }
    }
}

impl std::error::Error for ToolsConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_schema_deserialization() {
        let yaml = r#"
tools:
  check_eligibility:
    name: check_eligibility
    description: "Check loan eligibility"
    parameters:
      - name: gold_weight
        type: number
        description: "Weight in grams"
        required: true
        min: 1.0
        max: 10000.0
      - name: gold_purity
        type: string
        description: "Purity level"
        required: false
        enum: ["24K", "22K", "18K"]
"#;
        let config: ToolsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.tools.len(), 1);

        let tool = config.get_tool("check_eligibility").unwrap();
        assert_eq!(tool.name, "check_eligibility");
        assert_eq!(tool.parameters.len(), 2);
        assert!(tool.parameters[0].required);
        assert!(!tool.parameters[1].required);
    }

    #[test]
    fn test_to_json_schema() {
        let tool = ToolSchema {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            enabled: None,
            category: Some("test".to_string()),
            parameters: vec![
                ToolParameter {
                    name: "required_param".to_string(),
                    param_type: "string".to_string(),
                    description: "A required param".to_string(),
                    required: true,
                    enum_values: Some(vec!["a".to_string(), "b".to_string()]),
                    min: None,
                    max: None,
                    default: None,
                },
                ToolParameter {
                    name: "optional_param".to_string(),
                    param_type: "number".to_string(),
                    description: "An optional param".to_string(),
                    required: false,
                    enum_values: None,
                    min: Some(0.0),
                    max: Some(100.0),
                    default: None,
                },
            ],
        };

        let schema = tool.to_json_schema();
        assert_eq!(schema["name"], "test_tool");
        assert!(schema["parameters"]["properties"]["required_param"]["enum"].is_array());
        assert_eq!(schema["parameters"]["properties"]["optional_param"]["minimum"], 0.0);
        assert_eq!(schema["parameters"]["required"][0], "required_param");
    }

    #[test]
    fn test_usage_guidelines() {
        let yaml = r#"
tools: {}
usage_guidelines:
  general: "Use tools wisely"
  eligibility: "Check when customer asks"
"#;
        let config: ToolsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.get_guideline("general"), Some("Use tools wisely"));
        assert_eq!(
            config.get_guideline("eligibility"),
            Some("Check when customer asks")
        );
        assert_eq!(config.get_guideline("unknown"), None);
    }
}
