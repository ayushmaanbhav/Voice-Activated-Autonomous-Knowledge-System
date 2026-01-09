//! Customer Segment Configuration
//!
//! Defines customer segment detection patterns loaded from YAML.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Segments configuration loaded from segments.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentsConfig {
    /// Segment definitions keyed by ID
    #[serde(default)]
    pub segments: HashMap<String, SegmentDefinition>,
    /// Priority order for multiple matches
    #[serde(default)]
    pub priority_order: Vec<String>,
    /// Default segment when no match
    #[serde(default)]
    pub default_segment: String,
}

impl Default for SegmentsConfig {
    fn default() -> Self {
        Self {
            segments: HashMap::new(),
            priority_order: Vec::new(),
            default_segment: String::new(),
        }
    }
}

impl SegmentsConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, SegmentsConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            SegmentsConfigError::FileNotFound(path.as_ref().display().to_string(), e.to_string())
        })?;

        serde_yaml::from_str(&content)
            .map_err(|e| SegmentsConfigError::ParseError(e.to_string()))
    }

    /// Get segment by ID
    pub fn get_segment(&self, id: &str) -> Option<&SegmentDefinition> {
        self.segments.get(id)
    }

    /// Get segments sorted by priority
    pub fn by_priority(&self) -> Vec<(&str, &SegmentDefinition)> {
        let mut result: Vec<_> = self
            .segments
            .iter()
            .map(|(id, def)| (id.as_str(), def))
            .collect();

        result.sort_by_key(|(id, def)| {
            // First by explicit priority order, then by definition priority
            self.priority_order
                .iter()
                .position(|p| p == *id)
                .map(|p| (0, p as i32))
                .unwrap_or((1, def.priority))
        });

        result
    }

    /// Detect segments from customer signals
    pub fn detect_segments(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> Vec<&str> {
        let text_lower = text.to_lowercase();
        let mut matches: Vec<(&str, i32)> = Vec::new();

        for (id, def) in &self.segments {
            if self.matches_segment(&text_lower, language, numeric_values, text_values, def) {
                matches.push((id.as_str(), def.priority));
            }
        }

        // Sort by priority (lower is higher priority)
        matches.sort_by_key(|(_, priority)| *priority);
        matches.into_iter().map(|(id, _)| id).collect()
    }

    /// Check if signals match a segment
    fn matches_segment(
        &self,
        text_lower: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
        def: &SegmentDefinition,
    ) -> bool {
        // Check numeric thresholds
        if let Some(ref thresholds) = def.detection.numeric_thresholds {
            for (key, threshold) in thresholds {
                if let Some(value) = numeric_values.get(key) {
                    if let Some(min) = threshold.min {
                        if *value >= min {
                            return true;
                        }
                    }
                }
            }
        }

        // Check text patterns
        if let Some(ref patterns) = def.detection.text_patterns {
            // Check language-specific patterns
            if let Some(lang_patterns) = patterns.get(language) {
                for pattern in lang_patterns {
                    if text_lower.contains(&pattern.to_lowercase()) {
                        return true;
                    }
                }
            }
            // Fallback to English if not found
            if language != "en" {
                if let Some(en_patterns) = patterns.get("en") {
                    for pattern in en_patterns {
                        if text_lower.contains(&pattern.to_lowercase()) {
                            return true;
                        }
                    }
                }
            }
        }

        // Check text values
        if let Some(ref values) = def.detection.text_values {
            for (key, expected_values) in values {
                if let Some(actual_value) = text_values.get(key) {
                    let actual_lower = actual_value.to_lowercase();
                    for expected in expected_values {
                        if actual_lower.contains(&expected.to_lowercase()) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Get value props for a segment in a language
    pub fn get_value_props(&self, segment_id: &str, language: &str) -> Vec<&str> {
        self.segments
            .get(segment_id)
            .and_then(|def| {
                def.value_props
                    .get(language)
                    .or_else(|| def.value_props.get("en"))
            })
            .map(|props| props.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Get features for a segment
    pub fn get_features(&self, segment_id: &str) -> Vec<&str> {
        self.segments
            .get(segment_id)
            .map(|def| def.features.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// P16 FIX: Get high_value segment thresholds
    /// Returns (collateral_min, loan_amount_min) for use with CustomerProfile::infer_segment_with_thresholds
    pub fn get_high_value_thresholds(&self) -> (f64, f64) {
        let defaults = (100.0, 500_000.0);  // Fallback defaults

        if let Some(high_value) = self.segments.get("high_value") {
            if let Some(ref thresholds) = high_value.detection.numeric_thresholds {
                let collateral = thresholds.get("gold_weight_grams")
                    .or_else(|| thresholds.get("asset_quantity"))
                    .and_then(|t| t.min)
                    .unwrap_or(defaults.0);

                let amount = thresholds.get("loan_amount")
                    .or_else(|| thresholds.get("requested_amount"))
                    .and_then(|t| t.min)
                    .unwrap_or(defaults.1);

                return (collateral, amount);
            }
        }

        defaults
    }

    /// Get a numeric threshold value for a segment
    pub fn get_numeric_threshold(&self, segment_id: &str, threshold_key: &str) -> Option<f64> {
        self.segments.get(segment_id)
            .and_then(|def| def.detection.numeric_thresholds.as_ref())
            .and_then(|thresholds| thresholds.get(threshold_key))
            .and_then(|t| t.min)
    }
}

/// Single segment definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentDefinition {
    pub display_name: String,
    #[serde(default = "default_priority")]
    pub priority: i32,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub detection: SegmentDetection,
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub value_props: HashMap<String, Vec<String>>,
}

fn default_priority() -> i32 {
    5
}

/// Detection rules for a segment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SegmentDetection {
    /// Numeric thresholds for detection
    #[serde(default)]
    pub numeric_thresholds: Option<HashMap<String, NumericThreshold>>,
    /// Text patterns for detection (by language)
    #[serde(default)]
    pub text_patterns: Option<HashMap<String, Vec<String>>>,
    /// Text value matches
    #[serde(default)]
    pub text_values: Option<HashMap<String, Vec<String>>>,
}

/// Numeric threshold for segment detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericThreshold {
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

/// Errors when loading segments configuration
#[derive(Debug)]
pub enum SegmentsConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for SegmentsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Segments config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse segments config: {}", err),
        }
    }
}

impl std::error::Error for SegmentsConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segments_deserialization() {
        let yaml = r#"
segments:
  high_value:
    display_name: "High Value"
    priority: 1
    detection:
      numeric_thresholds:
        loan_amount:
          min: 500000
      text_patterns:
        en:
          - "lakh"
    features:
      - "priority_processing"
    value_props:
      en:
        - "Dedicated relationship manager"
priority_order:
  - "high_value"
default_segment: "first_time"
"#;
        let config: SegmentsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.segments.len(), 1);
        assert!(config.segments.contains_key("high_value"));
        assert_eq!(config.default_segment, "first_time");
    }

    #[test]
    fn test_detect_segments() {
        let yaml = r#"
segments:
  high_value:
    display_name: "High Value"
    priority: 1
    detection:
      text_patterns:
        en:
          - "lakh"
  urgent:
    display_name: "Urgent"
    priority: 2
    detection:
      text_patterns:
        en:
          - "urgent"
          - "emergency"
"#;
        let config: SegmentsConfig = serde_yaml::from_str(yaml).unwrap();

        let numeric: HashMap<String, f64> = HashMap::new();
        let text: HashMap<String, String> = HashMap::new();

        // Test lakh detection
        let segments = config.detect_segments("I need 5 lakh urgently", "en", &numeric, &text);
        assert!(segments.contains(&"high_value"));
        assert!(segments.contains(&"urgent"));

        // Test urgent only
        let segments = config.detect_segments("This is an emergency", "en", &numeric, &text);
        assert!(!segments.contains(&"high_value"));
        assert!(segments.contains(&"urgent"));
    }
}
