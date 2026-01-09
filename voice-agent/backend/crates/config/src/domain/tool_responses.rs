//! Tool Response Templates Configuration
//!
//! P16 FIX: Config-driven response templates for tools.
//! Replaces hardcoded response messages in tool implementations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Tool response templates configuration loaded from tools/responses.yaml
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolResponsesConfig {
    /// Response templates by tool name
    #[serde(default)]
    pub templates: HashMap<String, ToolTemplates>,
    /// Variable definitions (for documentation and validation)
    #[serde(default)]
    pub variables: HashMap<String, VariableDefinition>,
    /// Rate tier descriptions
    #[serde(default)]
    pub rate_descriptions: HashMap<String, String>,
    /// Trend direction labels
    #[serde(default)]
    pub trend_labels: HashMap<String, HashMap<String, String>>,
}

/// Templates for a specific tool
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolTemplates {
    /// Template variants keyed by scenario name
    #[serde(flatten)]
    pub variants: HashMap<String, TemplateVariant>,
}

/// A template variant with language support
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TemplateVariant {
    /// Simple string template (single language)
    Simple(String),
    /// Multilingual template
    Multilingual(HashMap<String, String>),
}

impl TemplateVariant {
    /// Get template for a language, falling back to English
    pub fn get(&self, language: &str) -> &str {
        match self {
            TemplateVariant::Simple(s) => s,
            TemplateVariant::Multilingual(map) => {
                map.get(language)
                    .or_else(|| map.get("en"))
                    .map(|s| s.as_str())
                    .unwrap_or("")
            }
        }
    }
}

/// Variable definition for documentation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VariableDefinition {
    #[serde(rename = "type", default)]
    pub var_type: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

/// Error type for template config loading
#[derive(Debug)]
pub enum ToolResponsesConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for ToolResponsesConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Tool responses config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse tool responses config: {}", err),
        }
    }
}

impl std::error::Error for ToolResponsesConfigError {}

impl ToolResponsesConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ToolResponsesConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ToolResponsesConfigError::FileNotFound(
                path.as_ref().display().to_string(),
                e.to_string(),
            )
        })?;

        serde_yaml::from_str(&content)
            .map_err(|e| ToolResponsesConfigError::ParseError(e.to_string()))
    }

    /// Get a template for a tool and scenario
    pub fn get_template(&self, tool: &str, scenario: &str, language: &str) -> Option<&str> {
        self.templates
            .get(tool)
            .and_then(|t| t.variants.get(scenario))
            .map(|v| v.get(language))
    }

    /// Render a template with variable substitution
    /// Variables use {variable_name} syntax
    pub fn render_template(
        &self,
        tool: &str,
        scenario: &str,
        language: &str,
        vars: &HashMap<String, String>,
    ) -> Option<String> {
        let template = self.get_template(tool, scenario, language)?;
        Some(Self::substitute_variables(template, vars))
    }

    /// Substitute variables in a template string
    pub fn substitute_variables(template: &str, vars: &HashMap<String, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in vars {
            result = result.replace(&format!("{{{}}}", key), value);
        }
        result
    }

    /// Get rate description for a tier
    pub fn get_rate_description(&self, tier: &str) -> &str {
        self.rate_descriptions
            .get(tier)
            .or_else(|| self.rate_descriptions.get("default"))
            .map(|s| s.as_str())
            .unwrap_or("competitive")
    }

    /// Get trend label for direction and language
    pub fn get_trend_label<'a>(&'a self, direction: &'a str, language: &str) -> &'a str {
        self.trend_labels
            .get(direction)
            .and_then(|m| m.get(language).or_else(|| m.get("en")))
            .map(|s| s.as_str())
            .unwrap_or(direction)
    }

    /// Check if templates are configured for a tool
    pub fn has_tool(&self, tool: &str) -> bool {
        self.templates.contains_key(tool)
    }

    /// Get all template scenario names for a tool
    pub fn scenarios_for_tool(&self, tool: &str) -> Vec<&str> {
        self.templates
            .get(tool)
            .map(|t| t.variants.keys().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_substitution() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), "John".to_string());
        vars.insert("amount".to_string(), "50000".to_string());

        let result = ToolResponsesConfig::substitute_variables(
            "Hello {name}, you are eligible for ₹{amount}",
            &vars,
        );
        assert_eq!(result, "Hello John, you are eligible for ₹50000");
    }

    #[test]
    fn test_multilingual_template() {
        let yaml = r#"
templates:
  check_eligibility:
    eligible:
      en: "You are eligible for {amount}"
      hi: "आप {amount} के लिए पात्र हैं"
"#;
        let config: ToolResponsesConfig = serde_yaml::from_str(yaml).unwrap();

        assert_eq!(
            config.get_template("check_eligibility", "eligible", "en"),
            Some("You are eligible for {amount}")
        );
        assert_eq!(
            config.get_template("check_eligibility", "eligible", "hi"),
            Some("आप {amount} के लिए पात्र हैं")
        );
        // Fallback to English
        assert_eq!(
            config.get_template("check_eligibility", "eligible", "mr"),
            Some("You are eligible for {amount}")
        );
    }
}
