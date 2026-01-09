//! Segment Detector trait for customer segmentation
//!
//! This module provides a domain-agnostic interface for detecting customer segments
//! based on text patterns, numeric thresholds, and slot values. All segment definitions
//! are loaded from configuration.
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_core::traits::SegmentDetector;
//!
//! // Detector is created from domain config
//! let detector = config_bridge.segment_detector();
//!
//! // Detect segments from user context
//! let segments = detector.detect("I urgently need money for hospital", "en", &values, &slots);
//! let primary = detector.primary_segment(&segments);
//! ```

use std::collections::HashMap;

/// Segment detection match
#[derive(Debug, Clone)]
pub struct SegmentMatch {
    /// Segment ID
    pub segment_id: String,
    /// Match confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Matched patterns or thresholds
    pub match_reasons: Vec<String>,
}

/// Value proposition for a segment
#[derive(Debug, Clone)]
pub struct ValueProposition {
    /// Short headline
    pub headline: String,
    /// Detailed description
    pub description: String,
    /// Language code
    pub language: String,
}

/// Feature emphasis for a segment
#[derive(Debug, Clone)]
pub struct FeatureEmphasis {
    /// Feature ID (e.g., "low_rates", "quick_process")
    pub feature_id: String,
    /// Display name
    pub display_name: String,
    /// Localized display name
    pub localized_name: String,
    /// Importance weight (higher = more important)
    pub weight: u8,
}

/// Segment definition trait
///
/// Defines a customer segment with detection rules, value propositions,
/// and feature emphasis. All loaded from config.
pub trait SegmentDefinition: Send + Sync {
    /// Segment ID (e.g., "high_value", "urgent_need")
    fn id(&self) -> &str;

    /// Human-readable display name
    fn display_name(&self) -> &str;

    /// Segment description
    fn description(&self) -> &str;

    /// Priority level (lower = higher priority)
    fn priority(&self) -> u8;

    /// Check if segment matches based on signals
    ///
    /// Returns true if this segment matches the given context.
    fn matches(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> bool;

    /// Get match confidence (0.0 - 1.0)
    fn match_confidence(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> f32;

    /// Get text patterns for detection
    fn text_patterns(&self, language: &str) -> Vec<&str>;

    /// Get numeric thresholds for detection
    fn numeric_thresholds(&self) -> &HashMap<String, f64>;

    /// Get slot value patterns for detection
    fn slot_patterns(&self) -> &HashMap<String, Vec<String>>;

    /// Get features to emphasize for this segment
    fn features(&self) -> Vec<&str>;

    /// Get value propositions for language
    fn value_propositions(&self, language: &str) -> Vec<&ValueProposition>;

    /// Get feature emphasis details
    fn feature_emphasis(&self) -> Vec<&FeatureEmphasis>;
}

/// Segment detector trait
///
/// Detects customer segments from conversation context.
pub trait SegmentDetector: Send + Sync {
    /// Detect all matching segments
    ///
    /// Returns segment IDs sorted by priority (highest priority first).
    fn detect(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> Vec<SegmentMatch>;

    /// Get primary segment (highest priority match)
    fn primary_segment(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> &str;

    /// Get segment definition by ID
    fn get_segment(&self, id: &str) -> Option<&dyn SegmentDefinition>;

    /// Get all segment definitions
    fn all_segments(&self) -> Vec<&dyn SegmentDefinition>;

    /// Get default segment ID
    fn default_segment(&self) -> &str;

    /// Get all segment IDs
    fn segment_ids(&self) -> Vec<&str>;

    /// Get features to emphasize for detected segments
    fn features_for_segments(&self, segment_ids: &[&str]) -> Vec<String>;

    /// Get value propositions for segment in language
    fn value_propositions(&self, segment_id: &str, language: &str) -> Vec<String>;
}

/// Config-driven segment definition
#[derive(Debug, Clone)]
pub struct ConfigSegmentDefinition {
    id: String,
    display_name: String,
    description: String,
    priority: u8,
    text_patterns: HashMap<String, Vec<String>>,
    numeric_thresholds: HashMap<String, f64>,
    slot_patterns: HashMap<String, Vec<String>>,
    features: Vec<String>,
    value_propositions: HashMap<String, Vec<ValueProposition>>,
    feature_emphasis: Vec<FeatureEmphasis>,
}

impl ConfigSegmentDefinition {
    /// Create a new segment definition
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        description: impl Into<String>,
        priority: u8,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            description: description.into(),
            priority,
            text_patterns: HashMap::new(),
            numeric_thresholds: HashMap::new(),
            slot_patterns: HashMap::new(),
            features: Vec::new(),
            value_propositions: HashMap::new(),
            feature_emphasis: Vec::new(),
        }
    }

    /// Add text patterns for a language
    pub fn with_text_patterns(mut self, language: &str, patterns: Vec<String>) -> Self {
        self.text_patterns.insert(language.to_string(), patterns);
        self
    }

    /// Add numeric thresholds
    pub fn with_numeric_thresholds(mut self, thresholds: HashMap<String, f64>) -> Self {
        self.numeric_thresholds = thresholds;
        self
    }

    /// Add slot patterns
    pub fn with_slot_patterns(mut self, patterns: HashMap<String, Vec<String>>) -> Self {
        self.slot_patterns = patterns;
        self
    }

    /// Add features to emphasize
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    // P0 FIX: All preset segment methods (high_value, trust_seeker, price_sensitive,
    // urgent_need, balance_transfer, first_time, business_owner) have been REMOVED.
    //
    // These presets contained hardcoded domain-specific values (competitor names,
    // gold-specific thresholds, etc.) that violated domain-agnostic principles.
    //
    // All segments MUST be loaded from domain config via:
    //   config/domains/{domain}/segments.yaml
    //
    // To onboard a new domain, define segments in the config file with appropriate:
    // - Numeric thresholds (e.g., collateral quantity, loan amount)
    // - Text patterns (in all supported languages)
    // - Slot patterns (for competitor detection, etc.)
    // - Features to emphasize
    //
    // Example usage:
    //   let segments_config = master_domain_config.segments.clone();
    //   let detector = segments_config.to_segment_detector();
}

impl SegmentDefinition for ConfigSegmentDefinition {
    fn id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn priority(&self) -> u8 {
        self.priority
    }

    fn matches(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> bool {
        let lower_text = text.to_lowercase();

        // Check numeric thresholds (any match triggers)
        for (key, threshold) in &self.numeric_thresholds {
            if let Some(value) = numeric_values.get(key) {
                if *value >= *threshold {
                    return true;
                }
            }
        }

        // Check text patterns
        if let Some(patterns) = self.text_patterns.get(language) {
            for pattern in patterns {
                if lower_text.contains(&pattern.to_lowercase()) {
                    return true;
                }
            }
        }

        // Check slot patterns
        for (slot_name, patterns) in &self.slot_patterns {
            if let Some(slot_value) = text_values.get(slot_name) {
                let lower_value = slot_value.to_lowercase();
                for pattern in patterns {
                    if lower_value.contains(&pattern.to_lowercase()) {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn match_confidence(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> f32 {
        let mut score = 0.0f32;
        let mut max_score = 0.0f32;
        let lower_text = text.to_lowercase();

        // Numeric threshold matches
        for (key, threshold) in &self.numeric_thresholds {
            max_score += 1.0;
            if let Some(value) = numeric_values.get(key) {
                if *value >= *threshold {
                    score += 1.0;
                }
            }
        }

        // Text pattern matches
        if let Some(patterns) = self.text_patterns.get(language) {
            for pattern in patterns {
                max_score += 0.5;
                if lower_text.contains(&pattern.to_lowercase()) {
                    score += 0.5;
                }
            }
        }

        // Slot pattern matches
        for (slot_name, patterns) in &self.slot_patterns {
            if let Some(slot_value) = text_values.get(slot_name) {
                let lower_value = slot_value.to_lowercase();
                for pattern in patterns {
                    max_score += 1.0;
                    if lower_value.contains(&pattern.to_lowercase()) {
                        score += 1.0;
                    }
                }
            }
        }

        if max_score > 0.0 {
            (score / max_score).min(1.0)
        } else {
            0.0
        }
    }

    fn text_patterns(&self, language: &str) -> Vec<&str> {
        self.text_patterns
            .get(language)
            .map(|p| p.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    fn numeric_thresholds(&self) -> &HashMap<String, f64> {
        &self.numeric_thresholds
    }

    fn slot_patterns(&self) -> &HashMap<String, Vec<String>> {
        &self.slot_patterns
    }

    fn features(&self) -> Vec<&str> {
        self.features.iter().map(|s| s.as_str()).collect()
    }

    fn value_propositions(&self, language: &str) -> Vec<&ValueProposition> {
        self.value_propositions
            .get(language)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    fn feature_emphasis(&self) -> Vec<&FeatureEmphasis> {
        self.feature_emphasis.iter().collect()
    }
}

/// Config-driven segment detector
pub struct ConfigSegmentDetector {
    segments: Vec<ConfigSegmentDefinition>,
    default_segment: String,
}

impl ConfigSegmentDetector {
    /// Create a new segment detector
    pub fn new(segments: Vec<ConfigSegmentDefinition>, default_segment: impl Into<String>) -> Self {
        let mut sorted_segments = segments;
        // Sort by priority (lower = higher priority)
        sorted_segments.sort_by_key(|s| s.priority);

        Self {
            segments: sorted_segments,
            default_segment: default_segment.into(),
        }
    }

}

impl SegmentDetector for ConfigSegmentDetector {
    fn detect(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> Vec<SegmentMatch> {
        let mut matches = Vec::new();

        for segment in &self.segments {
            if segment.matches(text, language, numeric_values, text_values) {
                matches.push(SegmentMatch {
                    segment_id: segment.id().to_string(),
                    confidence: segment.match_confidence(text, language, numeric_values, text_values),
                    match_reasons: Vec::new(), // Could populate with specific matches
                });
            }
        }

        // Already sorted by priority (from constructor)
        matches
    }

    fn primary_segment(
        &self,
        text: &str,
        language: &str,
        numeric_values: &HashMap<String, f64>,
        text_values: &HashMap<String, String>,
    ) -> &str {
        for segment in &self.segments {
            if segment.matches(text, language, numeric_values, text_values) {
                return segment.id();
            }
        }
        &self.default_segment
    }

    fn get_segment(&self, id: &str) -> Option<&dyn SegmentDefinition> {
        self.segments
            .iter()
            .find(|s| s.id() == id)
            .map(|s| s as &dyn SegmentDefinition)
    }

    fn all_segments(&self) -> Vec<&dyn SegmentDefinition> {
        self.segments
            .iter()
            .map(|s| s as &dyn SegmentDefinition)
            .collect()
    }

    fn default_segment(&self) -> &str {
        &self.default_segment
    }

    fn segment_ids(&self) -> Vec<&str> {
        self.segments.iter().map(|s| s.id()).collect()
    }

    fn features_for_segments(&self, segment_ids: &[&str]) -> Vec<String> {
        let mut features = Vec::new();
        for id in segment_ids {
            if let Some(segment) = self.get_segment(id) {
                for feature in segment.features() {
                    if !features.contains(&feature.to_string()) {
                        features.push(feature.to_string());
                    }
                }
            }
        }
        features
    }

    fn value_propositions(&self, segment_id: &str, language: &str) -> Vec<String> {
        self.get_segment(segment_id)
            .map(|s| {
                s.value_propositions(language)
                    .iter()
                    .map(|v| v.headline.clone())
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P0 FIX: Create test segment detector with config-driven segments
    /// These are test-only segments that don't contain domain-specific values
    fn test_detector() -> ConfigSegmentDetector {
        ConfigSegmentDetector::new(
            vec![
                // Urgent need segment (priority 1)
                ConfigSegmentDefinition::new("urgent_need", "Urgent Need", "Customer with immediate need", 1)
                    .with_text_patterns("en", vec![
                        "urgent".to_string(),
                        "emergency".to_string(),
                        "immediately".to_string(),
                    ])
                    .with_features(vec!["same_day_disbursement".to_string()]),
                // High value segment (priority 1)
                ConfigSegmentDefinition::new("high_value", "High Value", "High-value customer", 1)
                    .with_numeric_thresholds([
                        ("loan_amount".to_string(), 500_000.0),
                    ].into_iter().collect())
                    .with_text_patterns("en", vec!["lakh".to_string()]),
                // Trust seeker segment (priority 2)
                ConfigSegmentDefinition::new("trust_seeker", "Trust Seeker", "Safety-focused customer", 2)
                    .with_slot_patterns([
                        ("current_lender".to_string(), vec!["competitor".to_string()]),
                    ].into_iter().collect())
                    .with_text_patterns("en", vec!["safe".to_string(), "security".to_string()]),
                // Price sensitive segment (priority 3)
                ConfigSegmentDefinition::new("price_sensitive", "Price Sensitive", "Rate-focused customer", 3)
                    .with_text_patterns("en", vec!["rate".to_string(), "interest".to_string()]),
                // First time segment (priority 4, default)
                ConfigSegmentDefinition::new("first_time", "First Time", "New customer", 4)
                    .with_text_patterns("en", vec!["first time".to_string(), "new to".to_string()]),
            ],
            "first_time",
        )
    }

    #[test]
    fn test_urgent_need_detection() {
        let detector = test_detector();
        let values = HashMap::new();
        let slots = HashMap::new();

        let primary = detector.primary_segment("I urgently need money", "en", &values, &slots);
        assert_eq!(primary, "urgent_need");
    }

    #[test]
    fn test_high_value_by_amount() {
        let detector = test_detector();
        let mut values = HashMap::new();
        values.insert("loan_amount".to_string(), 600_000.0);
        let slots = HashMap::new();

        let primary = detector.primary_segment("I need a loan", "en", &values, &slots);
        assert_eq!(primary, "high_value");
    }

    #[test]
    fn test_trust_seeker_by_lender() {
        let detector = test_detector();
        let values = HashMap::new();
        let mut slots = HashMap::new();
        slots.insert("current_lender".to_string(), "competitor".to_string());

        // Use text that doesn't trigger other segments
        let primary = detector.primary_segment("I need a loan", "en", &values, &slots);
        assert_eq!(primary, "trust_seeker");
    }

    #[test]
    fn test_default_segment() {
        let detector = test_detector();
        let values = HashMap::new();
        let slots = HashMap::new();

        let primary = detector.primary_segment("hello", "en", &values, &slots);
        assert_eq!(primary, "first_time");
    }

    #[test]
    fn test_segment_features() {
        let detector = test_detector();
        let features = detector.features_for_segments(&["urgent_need"]);
        assert!(features.contains(&"same_day_disbursement".to_string()));
    }

    #[test]
    fn test_multiple_segments_detected() {
        let detector = test_detector();
        let mut values = HashMap::new();
        values.insert("loan_amount".to_string(), 600_000.0);
        let slots = HashMap::new();

        // "urgent" + high value amount should detect both
        let matches = detector.detect("I urgently need 6 lakh", "en", &values, &slots);
        assert!(matches.len() >= 2);
    }
}
