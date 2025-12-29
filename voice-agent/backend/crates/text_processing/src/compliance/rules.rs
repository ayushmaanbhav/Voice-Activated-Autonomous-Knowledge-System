//! Compliance rules definition and loading

use serde::{Deserialize, Serialize};

/// Compliance rules structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRules {
    /// Version of rules
    pub version: String,
    /// Forbidden phrases (critical violations)
    #[serde(default)]
    pub forbidden_phrases: Vec<String>,
    /// Claims that require disclaimers
    #[serde(default)]
    pub claims_requiring_disclaimer: Vec<ClaimRule>,
    /// Rate validation rules
    #[serde(default)]
    pub rate_rules: RateRules,
    /// Required disclosures
    #[serde(default)]
    pub required_disclosures: Vec<RequiredDisclosure>,
    /// Competitor mention rules
    #[serde(default)]
    pub competitor_rules: CompetitorRules,
}

/// Rule for claims that need disclaimers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRule {
    /// Pattern to match (regex)
    pub pattern: String,
    /// Required disclaimer text
    pub disclaimer: String,
    /// Description of the rule
    #[serde(default)]
    pub description: String,
}

/// Interest rate validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateRules {
    /// Minimum allowed rate (%)
    pub min_rate: f32,
    /// Maximum allowed rate (%)
    pub max_rate: f32,
    /// Rate precision required
    #[serde(default = "default_precision")]
    pub precision: u32,
}

fn default_precision() -> u32 {
    2
}

impl Default for RateRules {
    fn default() -> Self {
        Self {
            min_rate: 7.0,  // Minimum gold loan rate
            max_rate: 24.0, // Maximum gold loan rate
            precision: 2,
        }
    }
}

/// Required disclosure for specific contexts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredDisclosure {
    /// When to add this disclosure
    pub trigger_pattern: String,
    /// The disclosure text
    pub disclosure: String,
    /// Position: start, end
    #[serde(default = "default_position")]
    pub position: String,
}

fn default_position() -> String {
    "end".to_string()
}

/// Rules for competitor mentions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompetitorRules {
    /// List of competitors
    pub competitors: Vec<String>,
    /// Whether disparagement is allowed
    #[serde(default)]
    pub allow_disparagement: bool,
    /// Whether comparison is allowed
    #[serde(default = "default_true")]
    pub allow_comparison: bool,
}

fn default_true() -> bool {
    true
}

impl Default for ComplianceRules {
    fn default() -> Self {
        default_rules()
    }
}

/// Load rules from TOML file
pub fn load_rules(path: &str) -> Result<ComplianceRules, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read rules file: {}", e))?;
    toml::from_str(&content)
        .map_err(|e| format!("Failed to parse rules file: {}", e))
}

/// Get default compliance rules for banking
pub fn default_rules() -> ComplianceRules {
    ComplianceRules {
        version: "1.0.0".to_string(),
        forbidden_phrases: vec![
            // Absolute guarantees (not allowed)
            "guaranteed approval".to_string(),
            "guaranteed lowest rate".to_string(),
            "100% approval".to_string(),
            "no rejection".to_string(),
            "sure approval".to_string(),
            // Misleading claims
            "free loan".to_string(),
            "zero interest".to_string(),
            "no fees".to_string(),
            // Competitor disparagement
            "fraud".to_string(),
            "scam".to_string(),
            "cheating".to_string(),
        ],
        claims_requiring_disclaimer: vec![
            ClaimRule {
                pattern: r"(?i)lowest.{0,10}rate".to_string(),
                disclaimer: "Interest rates are subject to change and depend on various factors including loan amount and tenure.".to_string(),
                description: "Claims about lowest rates".to_string(),
            },
            ClaimRule {
                pattern: r"(?i)instant.{0,10}(approval|disbursement)".to_string(),
                disclaimer: "Subject to document verification and eligibility criteria.".to_string(),
                description: "Claims about instant processing".to_string(),
            },
            ClaimRule {
                pattern: r"(?i)best.{0,10}(offer|deal|rate)".to_string(),
                disclaimer: "Terms and conditions apply.".to_string(),
                description: "Claims about best offers".to_string(),
            },
            ClaimRule {
                pattern: r"\d+(\.\d+)?\s*%".to_string(),
                disclaimer: "Interest rate is subject to change. Please refer to the latest rate card.".to_string(),
                description: "Any rate mention".to_string(),
            },
        ],
        rate_rules: RateRules::default(),
        required_disclosures: vec![
            RequiredDisclosure {
                trigger_pattern: r"(?i)(loan|interest|emi)".to_string(),
                disclosure: "".to_string(), // Only add on explicit loan discussion
                position: "end".to_string(),
            },
        ],
        competitor_rules: CompetitorRules {
            competitors: vec![
                "Muthoot".to_string(),
                "Manappuram".to_string(),
                "IIFL".to_string(),
                "HDFC".to_string(),
                "SBI".to_string(),
                "ICICI".to_string(),
            ],
            allow_disparagement: false,
            allow_comparison: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_rules() {
        let rules = default_rules();
        assert_eq!(rules.version, "1.0.0");
        assert!(!rules.forbidden_phrases.is_empty());
        assert!(!rules.claims_requiring_disclaimer.is_empty());
    }

    #[test]
    fn test_rate_rules() {
        let rate_rules = RateRules::default();
        assert!(rate_rules.min_rate > 0.0);
        assert!(rate_rules.max_rate > rate_rules.min_rate);
    }

    #[test]
    fn test_serialize_rules() {
        let rules = default_rules();
        let toml_str = toml::to_string_pretty(&rules).unwrap();
        assert!(toml_str.contains("version"));
        assert!(toml_str.contains("forbidden_phrases"));
    }
}
