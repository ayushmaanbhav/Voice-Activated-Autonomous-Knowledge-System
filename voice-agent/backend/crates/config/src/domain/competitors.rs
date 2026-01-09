//! Competitor Configuration
//!
//! Defines competitor data loaded from YAML for comparison tools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Competitors configuration loaded from competitors.yaml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorsConfig {
    /// Competitor definitions keyed by ID
    #[serde(default)]
    pub competitors: HashMap<String, CompetitorEntry>,
    /// Comparison talking points
    #[serde(default)]
    pub comparison_points: Vec<ComparisonPoint>,
    /// Our (Kotak) features for comparison
    #[serde(default)]
    pub our_features: Vec<String>,
    /// Default assumptions
    #[serde(default)]
    pub defaults: CompetitorDefaults,
}

impl Default for CompetitorsConfig {
    fn default() -> Self {
        Self {
            competitors: HashMap::new(),
            comparison_points: Vec::new(),
            our_features: Vec::new(),
            defaults: CompetitorDefaults::default(),
        }
    }
}

impl CompetitorsConfig {
    /// Load from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CompetitorsConfigError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            CompetitorsConfigError::FileNotFound(
                path.as_ref().display().to_string(),
                e.to_string(),
            )
        })?;

        serde_yaml::from_str(&content)
            .map_err(|e| CompetitorsConfigError::ParseError(e.to_string()))
    }

    /// Get competitor by ID
    pub fn get_competitor(&self, id: &str) -> Option<&CompetitorEntry> {
        self.competitors.get(id)
    }

    /// Find competitor by name or alias
    pub fn find_by_name(&self, name: &str) -> Option<(&str, &CompetitorEntry)> {
        let name_lower = name.to_lowercase();

        // First try direct match
        for (id, entry) in &self.competitors {
            if id == &name_lower {
                return Some((id.as_str(), entry));
            }
        }

        // Search aliases
        for (id, entry) in &self.competitors {
            if entry
                .aliases
                .iter()
                .any(|a| a.to_lowercase() == name_lower)
            {
                return Some((id.as_str(), entry));
            }
        }

        None
    }

    /// Get all NBFCs
    pub fn nbfcs(&self) -> Vec<(&str, &CompetitorEntry)> {
        self.competitors
            .iter()
            .filter(|(_, e)| e.competitor_type == "nbfc")
            .map(|(id, e)| (id.as_str(), e))
            .collect()
    }

    /// Get all banks
    pub fn banks(&self) -> Vec<(&str, &CompetitorEntry)> {
        self.competitors
            .iter()
            .filter(|(_, e)| e.competitor_type == "bank")
            .map(|(id, e)| (id.as_str(), e))
            .collect()
    }

    /// Get default rate for competitor type
    pub fn default_rate_for_type(&self, competitor_type: &str) -> f64 {
        match competitor_type {
            "nbfc" => self.defaults.nbfc_rate,
            "bank" => self.defaults.bank_rate,
            "informal" => self.defaults.local_lender_rate,
            _ => self.defaults.nbfc_rate,
        }
    }

    /// Get highlighted comparison points
    pub fn highlighted_points(&self) -> Vec<&ComparisonPoint> {
        self.comparison_points.iter().filter(|p| p.highlight).collect()
    }

    /// Get our features for comparison
    pub fn our_features(&self) -> &[String] {
        &self.our_features
    }

    /// Get all competitor IDs
    pub fn competitor_ids(&self) -> Vec<&str> {
        self.competitors.keys().map(|k| k.as_str()).collect()
    }

    /// P16 FIX: Generate competitor patterns for IntentDetector::add_competitor_patterns()
    ///
    /// Returns tuples of (id, display_name, regex_pattern) for each competitor.
    /// The regex patterns are generated from the competitor ID and aliases.
    pub fn to_intent_patterns(&self) -> Vec<(String, String, String)> {
        let mut patterns = Vec::new();

        for (id, entry) in &self.competitors {
            // Build pattern from ID and aliases
            let mut all_names = vec![id.clone()];
            all_names.extend(entry.aliases.iter().cloned());

            // Create regex pattern that matches any of the names (case-insensitive)
            let alternatives = all_names
                .iter()
                .map(|name| regex::escape(name))
                .collect::<Vec<_>>()
                .join("|");

            let pattern = format!(r"(?i)\b({})\b", alternatives);

            patterns.push((id.clone(), entry.display_name.clone(), pattern));
        }

        patterns
    }

    /// Get all competitor names and aliases as a flat list (for text processing)
    pub fn all_names_and_aliases(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for (id, entry) in &self.competitors {
            names.push(id.as_str());
            names.push(entry.display_name.as_str());
            names.extend(entry.aliases.iter().map(|s| s.as_str()));
        }
        names
    }
}

/// Single competitor entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorEntry {
    pub display_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub typical_rate: f64,
    #[serde(default)]
    pub rate_range: Option<RateRange>,
    #[serde(default = "default_ltv")]
    pub ltv_percent: f64,
    #[serde(default)]
    pub competitor_type: String,
    #[serde(default)]
    pub strengths: Vec<String>,
    #[serde(default)]
    pub weaknesses: Vec<String>,
    #[serde(default)]
    pub processing_time: String,
}

fn default_ltv() -> f64 {
    75.0
}

/// Rate range for a competitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateRange {
    pub min: f64,
    pub max: f64,
}

/// Comparison point for marketing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonPoint {
    pub category: String,
    pub our_advantage: String,
    #[serde(default)]
    pub highlight: bool,
}

/// Default assumptions for competitors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorDefaults {
    #[serde(default = "default_nbfc_rate")]
    pub nbfc_rate: f64,
    #[serde(default = "default_local_rate")]
    pub local_lender_rate: f64,
    #[serde(default = "default_bank_rate")]
    pub bank_rate: f64,
}

fn default_nbfc_rate() -> f64 {
    18.0
}

fn default_local_rate() -> f64 {
    24.0
}

fn default_bank_rate() -> f64 {
    11.0
}

impl Default for CompetitorDefaults {
    fn default() -> Self {
        Self {
            nbfc_rate: default_nbfc_rate(),
            local_lender_rate: default_local_rate(),
            bank_rate: default_bank_rate(),
        }
    }
}

/// Errors when loading competitors configuration
#[derive(Debug)]
pub enum CompetitorsConfigError {
    FileNotFound(String, String),
    ParseError(String),
}

impl std::fmt::Display for CompetitorsConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path, err) => {
                write!(f, "Competitors config not found at {}: {}", path, err)
            }
            Self::ParseError(err) => write!(f, "Failed to parse competitors config: {}", err),
        }
    }
}

impl std::error::Error for CompetitorsConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_competitors_deserialization() {
        let yaml = r#"
competitors:
  muthoot:
    display_name: "Muthoot Finance"
    aliases:
      - "muthut"
    typical_rate: 12.0
    ltv_percent: 75.0
    competitor_type: "nbfc"
    strengths:
      - "Large network"
comparison_points:
  - category: "Interest Rate"
    our_advantage: "Lower rates"
    highlight: true
"#;
        let config: CompetitorsConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.competitors.len(), 1);
        assert!(config.competitors.contains_key("muthoot"));
        assert_eq!(config.comparison_points.len(), 1);
    }

    #[test]
    fn test_find_by_name() {
        let mut competitors = HashMap::new();
        competitors.insert(
            "muthoot".to_string(),
            CompetitorEntry {
                display_name: "Muthoot Finance".to_string(),
                aliases: vec!["muthut".to_string(), "muthoot finance".to_string()],
                typical_rate: 12.0,
                rate_range: None,
                ltv_percent: 75.0,
                competitor_type: "nbfc".to_string(),
                strengths: vec![],
                weaknesses: vec![],
                processing_time: "Same day".to_string(),
            },
        );

        let config = CompetitorsConfig {
            competitors,
            comparison_points: vec![],
            our_features: vec![],
            defaults: CompetitorDefaults::default(),
        };

        // Direct match
        assert!(config.find_by_name("muthoot").is_some());
        // Alias match
        assert!(config.find_by_name("muthut").is_some());
        assert!(config.find_by_name("Muthoot Finance").is_some());
        // No match
        assert!(config.find_by_name("unknown").is_none());
    }
}
