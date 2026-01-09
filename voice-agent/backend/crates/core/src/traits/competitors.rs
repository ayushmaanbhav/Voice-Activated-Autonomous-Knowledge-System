//! Competitor Analyzer trait for competitor comparison and savings calculation
//!
//! This module provides a domain-agnostic interface for competitor analysis,
//! including rate lookups, savings calculations, and comparison talking points.
//! All competitor data is loaded from configuration.
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_core::traits::CompetitorAnalyzer;
//!
//! // Analyzer is created from domain config
//! let analyzer = config_bridge.competitor_analyzer();
//!
//! // Get competitor rate
//! let rate = analyzer.get_rate("muthoot");
//!
//! // Calculate savings
//! let savings = analyzer.calculate_savings(500000.0, 18.0, 12, Some("muthoot"));
//! ```

use std::collections::HashMap;

/// Competitor type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompetitorType {
    /// Scheduled commercial bank (e.g., HDFC, SBI)
    Bank,
    /// Non-Banking Financial Company (e.g., Muthoot, Manappuram)
    Nbfc,
    /// Informal lender (e.g., local jeweler)
    Informal,
}

impl CompetitorType {
    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Bank => "Bank",
            Self::Nbfc => "NBFC",
            Self::Informal => "Informal Lender",
        }
    }

    /// Get default rate for this competitor type
    pub fn default_rate(&self) -> f64 {
        match self {
            Self::Bank => 11.0,
            Self::Nbfc => 18.0,
            Self::Informal => 24.0,
        }
    }
}

/// Competitor information
#[derive(Debug, Clone)]
pub struct CompetitorInfo {
    /// Competitor ID (e.g., "muthoot", "hdfc")
    pub id: String,
    /// Display name (e.g., "Muthoot Finance")
    pub display_name: String,
    /// Typical/advertised interest rate
    pub typical_rate: f64,
    /// Rate range (min, max)
    pub rate_range: Option<(f64, f64)>,
    /// LTV percentage
    pub ltv_percent: f64,
    /// Competitor type
    pub competitor_type: CompetitorType,
    /// Strengths (from customer perspective)
    pub strengths: Vec<String>,
    /// Weaknesses (talking points against)
    pub weaknesses: Vec<String>,
    /// Processing time description
    pub processing_time: String,
    /// Aliases for detection
    pub aliases: Vec<String>,
}

impl CompetitorInfo {
    /// Create a new competitor
    pub fn new(
        id: impl Into<String>,
        display_name: impl Into<String>,
        typical_rate: f64,
        competitor_type: CompetitorType,
    ) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            typical_rate,
            rate_range: None,
            ltv_percent: 75.0,
            competitor_type,
            strengths: Vec::new(),
            weaknesses: Vec::new(),
            processing_time: "Same day".to_string(),
            aliases: Vec::new(),
        }
    }

    /// Set rate range
    pub fn with_rate_range(mut self, min: f64, max: f64) -> Self {
        self.rate_range = Some((min, max));
        self
    }

    /// Set LTV
    pub fn with_ltv(mut self, ltv: f64) -> Self {
        self.ltv_percent = ltv;
        self
    }

    /// Add strengths
    pub fn with_strengths(mut self, strengths: Vec<String>) -> Self {
        self.strengths = strengths;
        self
    }

    /// Add weaknesses
    pub fn with_weaknesses(mut self, weaknesses: Vec<String>) -> Self {
        self.weaknesses = weaknesses;
        self
    }

    /// Add aliases
    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases = aliases;
        self
    }

    // NOTE: Domain-specific factory methods (muthoot, manappuram, iifl, hdfc, sbi, icici,
    // federal, local_jeweler) have been removed. Use config-driven competitors from
    // config/domains/{domain}/competitors.yaml via DomainBridge instead.
}

/// Savings analysis result
#[derive(Debug, Clone)]
pub struct SavingsAnalysis {
    /// Competitor name (if known)
    pub competitor_name: Option<String>,
    /// Competitor rate used
    pub competitor_rate: f64,
    /// Our rate
    pub our_rate: f64,
    /// Rate difference (positive = we're cheaper)
    pub rate_difference: f64,
    /// Monthly EMI savings
    pub monthly_emi_savings: f64,
    /// Monthly interest savings
    pub monthly_interest_savings: f64,
    /// Total savings over tenure
    pub total_savings: f64,
    /// Tenure in months
    pub tenure_months: i64,
    /// Percentage saved
    pub percentage_saved: f64,
}

/// Comparison talking point
#[derive(Debug, Clone)]
pub struct ComparisonPoint {
    /// Category (e.g., "Interest Rate", "Security")
    pub category: String,
    /// Our advantage description
    pub our_advantage: String,
    /// Should this be highlighted?
    pub highlight: bool,
}

/// Competitor analyzer trait
///
/// Analyzes competitors and calculates savings.
pub trait CompetitorAnalyzer: Send + Sync {
    /// Get competitor by ID or alias
    fn get_competitor(&self, name: &str) -> Option<&CompetitorInfo>;

    /// Get competitor rate (with alias resolution)
    fn get_rate(&self, name: &str) -> f64;

    /// Get all competitors
    fn all_competitors(&self) -> Vec<&CompetitorInfo>;

    /// Get competitor IDs
    fn competitor_ids(&self) -> Vec<&str>;

    /// Calculate savings when switching to us
    fn calculate_savings(
        &self,
        loan_amount: f64,
        current_rate: f64,
        tenure_months: i64,
        competitor_name: Option<&str>,
    ) -> SavingsAnalysis;

    /// Get comparison talking points
    fn comparison_points(&self) -> Vec<&ComparisonPoint>;

    /// Get highlighted talking points only
    fn highlighted_points(&self) -> Vec<&ComparisonPoint>;

    /// Get default rate for competitor type
    fn default_rate(&self, competitor_type: CompetitorType) -> f64;

    /// Detect competitor from text
    fn detect_competitor(&self, text: &str) -> Option<&str>;

    /// Get our typical rate for comparison
    fn our_typical_rate(&self) -> f64;

    /// Get our rate for amount (tiered)
    fn our_rate_for_amount(&self, amount: f64) -> f64;

    /// Build comparison message
    fn build_comparison_message(
        &self,
        savings: &SavingsAnalysis,
        language: &str,
    ) -> String;
}

/// Config-driven competitor analyzer
pub struct ConfigCompetitorAnalyzer {
    competitors: HashMap<String, CompetitorInfo>,
    alias_map: HashMap<String, String>,
    comparison_points: Vec<ComparisonPoint>,
    our_base_rate: f64,
    rate_tiers: Vec<(f64, f64)>, // (max_amount, rate)
}

impl ConfigCompetitorAnalyzer {
    /// Create a new analyzer
    pub fn new(
        competitors: Vec<CompetitorInfo>,
        comparison_points: Vec<ComparisonPoint>,
        our_base_rate: f64,
        rate_tiers: Vec<(f64, f64)>,
    ) -> Self {
        let mut alias_map = HashMap::new();
        let mut comp_map = HashMap::new();

        for comp in competitors {
            let id = comp.id.clone();
            // Map aliases to ID
            for alias in &comp.aliases {
                alias_map.insert(alias.to_lowercase(), id.clone());
            }
            alias_map.insert(id.to_lowercase(), id.clone());
            alias_map.insert(comp.display_name.to_lowercase(), id.clone());
            comp_map.insert(id, comp);
        }

        Self {
            competitors: comp_map,
            alias_map,
            comparison_points,
            our_base_rate,
            rate_tiers,
        }
    }

    /// Calculate EMI (uses shared implementation from crate::financial)
    fn calculate_emi(&self, principal: f64, annual_rate: f64, months: i64) -> f64 {
        crate::financial::calculate_emi(principal, annual_rate, months)
    }

    /// Calculate total interest (uses shared implementation from crate::financial)
    fn calculate_total_interest(&self, principal: f64, annual_rate: f64, months: i64) -> f64 {
        crate::financial::calculate_total_interest(principal, annual_rate, months)
    }
}

impl CompetitorAnalyzer for ConfigCompetitorAnalyzer {
    fn get_competitor(&self, name: &str) -> Option<&CompetitorInfo> {
        let lower = name.to_lowercase();
        self.alias_map
            .get(&lower)
            .and_then(|id| self.competitors.get(id))
    }

    fn get_rate(&self, name: &str) -> f64 {
        self.get_competitor(name)
            .map(|c| c.typical_rate)
            .unwrap_or(CompetitorType::Nbfc.default_rate())
    }

    fn all_competitors(&self) -> Vec<&CompetitorInfo> {
        self.competitors.values().collect()
    }

    fn competitor_ids(&self) -> Vec<&str> {
        self.competitors.keys().map(|s| s.as_str()).collect()
    }

    fn calculate_savings(
        &self,
        loan_amount: f64,
        current_rate: f64,
        tenure_months: i64,
        competitor_name: Option<&str>,
    ) -> SavingsAnalysis {
        let our_rate = self.our_rate_for_amount(loan_amount);
        let rate_diff = current_rate - our_rate;

        // Monthly interest savings
        let current_monthly_interest = loan_amount * (current_rate / 100.0 / 12.0);
        let our_monthly_interest = loan_amount * (our_rate / 100.0 / 12.0);
        let monthly_interest_savings = current_monthly_interest - our_monthly_interest;

        // EMI savings
        let current_emi = self.calculate_emi(loan_amount, current_rate, tenure_months);
        let our_emi = self.calculate_emi(loan_amount, our_rate, tenure_months);
        let monthly_emi_savings = current_emi - our_emi;

        // Total savings
        let current_total_interest =
            self.calculate_total_interest(loan_amount, current_rate, tenure_months);
        let our_total_interest =
            self.calculate_total_interest(loan_amount, our_rate, tenure_months);
        let total_savings = current_total_interest - our_total_interest;

        // Percentage saved
        let percentage_saved = if current_total_interest > 0.0 {
            (total_savings / current_total_interest) * 100.0
        } else {
            0.0
        };

        SavingsAnalysis {
            competitor_name: competitor_name.map(String::from),
            competitor_rate: current_rate,
            our_rate,
            rate_difference: rate_diff,
            monthly_emi_savings,
            monthly_interest_savings,
            total_savings,
            tenure_months,
            percentage_saved,
        }
    }

    fn comparison_points(&self) -> Vec<&ComparisonPoint> {
        self.comparison_points.iter().collect()
    }

    fn highlighted_points(&self) -> Vec<&ComparisonPoint> {
        self.comparison_points
            .iter()
            .filter(|p| p.highlight)
            .collect()
    }

    fn default_rate(&self, competitor_type: CompetitorType) -> f64 {
        competitor_type.default_rate()
    }

    fn detect_competitor(&self, text: &str) -> Option<&str> {
        let lower = text.to_lowercase();
        for (alias, id) in &self.alias_map {
            if lower.contains(alias) {
                return Some(id.as_str());
            }
        }
        None
    }

    fn our_typical_rate(&self) -> f64 {
        self.our_base_rate
    }

    fn our_rate_for_amount(&self, amount: f64) -> f64 {
        for (max_amount, rate) in &self.rate_tiers {
            if amount <= *max_amount {
                return *rate;
            }
        }
        self.our_base_rate
    }

    fn build_comparison_message(
        &self,
        savings: &SavingsAnalysis,
        language: &str,
    ) -> String {
        match language {
            "hi" => format!(
                "आप हमारे साथ ₹{:.0} प्रति महीना बचा सकते हैं! कुल बचत ₹{:.0} होगी।",
                savings.monthly_interest_savings,
                savings.total_savings
            ),
            _ => format!(
                "You could save ₹{:.0} per month with us! That's ₹{:.0} total savings over {} months.",
                savings.monthly_interest_savings,
                savings.total_savings,
                savings.tenure_months
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test competitor analyzer with generic test data
    ///
    /// Uses fictional competitor names to keep tests domain-agnostic.
    /// Real competitor data comes from config/domains/{domain}/competitors.yaml
    fn test_analyzer() -> ConfigCompetitorAnalyzer {
        let competitors = vec![
            // Generic NBFC competitor with aliases
            CompetitorInfo::new("nbfc_a", "NBFC Provider A", 12.0, CompetitorType::Nbfc)
                .with_rate_range(12.0, 24.0)
                .with_aliases(vec!["nbfca".to_string(), "Provider A".to_string()])
                .with_strengths(vec!["Branch network".to_string()])
                .with_weaknesses(vec!["Higher rates".to_string()]),
            CompetitorInfo::new("nbfc_b", "NBFC Provider B", 12.0, CompetitorType::Nbfc),
            CompetitorInfo::new("nbfc_c", "NBFC Provider C", 11.0, CompetitorType::Nbfc),
            // Generic bank competitors
            CompetitorInfo::new("bank_a", "Bank A", 10.5, CompetitorType::Bank),
            CompetitorInfo::new("bank_b", "Bank B", 9.85, CompetitorType::Bank),
            // Generic informal lender
            CompetitorInfo::new("informal_lender", "Local Lender", 24.0, CompetitorType::Informal),
        ];

        let comparison_points = vec![
            ComparisonPoint {
                category: "Interest Rate".to_string(),
                our_advantage: "Lower rates".to_string(),
                highlight: true,
            },
            ComparisonPoint {
                category: "Security".to_string(),
                our_advantage: "Regulated institution".to_string(),
                highlight: true,
            },
            ComparisonPoint {
                category: "Prepayment".to_string(),
                our_advantage: "Zero charges".to_string(),
                highlight: true,
            },
        ];

        let rate_tiers = vec![
            (100_000.0, 11.5),
            (500_000.0, 10.5),
            (f64::MAX, 9.5),
        ];

        ConfigCompetitorAnalyzer::new(competitors, comparison_points, 10.5, rate_tiers)
    }

    #[test]
    fn test_competitor_rates() {
        let analyzer = test_analyzer();

        // Test rate lookup by competitor ID
        assert_eq!(analyzer.get_rate("nbfc_a"), 12.0);
        assert_eq!(analyzer.get_rate("nbfc_b"), 12.0);
        assert_eq!(analyzer.get_rate("nbfc_c"), 11.0);
        assert_eq!(analyzer.get_rate("bank_a"), 10.5);
        assert_eq!(analyzer.get_rate("bank_b"), 9.85);
        assert_eq!(analyzer.get_rate("informal_lender"), 24.0);
    }

    #[test]
    fn test_alias_resolution() {
        let analyzer = test_analyzer();

        // Test aliases resolve to correct rates
        assert_eq!(analyzer.get_rate("nbfca"), 12.0); // Alias without underscore
        assert_eq!(analyzer.get_rate("Provider A"), 12.0); // Display name alias
        assert_eq!(analyzer.get_rate("NBFC_A"), 12.0); // Uppercase ID
    }

    #[test]
    fn test_our_rate_tiers() {
        let analyzer = test_analyzer();

        // Test tier boundaries
        assert_eq!(analyzer.our_rate_for_amount(50_000.0), 11.5); // Tier 1
        assert_eq!(analyzer.our_rate_for_amount(100_000.0), 11.5);
        assert_eq!(analyzer.our_rate_for_amount(200_000.0), 10.5); // Tier 2
        assert_eq!(analyzer.our_rate_for_amount(500_000.0), 10.5);
        assert_eq!(analyzer.our_rate_for_amount(600_000.0), 9.5); // Tier 3
    }

    #[test]
    fn test_savings_calculation() {
        let analyzer = test_analyzer();

        // Test savings: 500k loan at 18% vs our 10.5%
        let savings = analyzer.calculate_savings(500_000.0, 18.0, 12, Some("nbfc_a"));

        assert_eq!(savings.our_rate, 10.5);
        assert_eq!(savings.competitor_rate, 18.0);
        assert_eq!(savings.rate_difference, 7.5);
        assert!(savings.monthly_interest_savings > 0.0);
        assert!(savings.total_savings > 0.0);
    }

    #[test]
    fn test_detect_competitor() {
        let analyzer = test_analyzer();

        assert_eq!(
            analyzer.detect_competitor("I have a loan with nbfc_a"),
            Some("nbfc_a")
        );
        assert_eq!(
            analyzer.detect_competitor("Currently using Provider A"),
            Some("nbfc_a")
        );
        assert_eq!(
            analyzer.detect_competitor("Hello there"),
            None
        );
    }

    #[test]
    fn test_comparison_points() {
        let analyzer = test_analyzer();

        let points = analyzer.comparison_points();
        assert!(points.len() >= 3);

        let highlighted = analyzer.highlighted_points();
        assert!(highlighted.len() >= 2);
    }

    #[test]
    fn test_default_rates() {
        let analyzer = test_analyzer();

        // Default rates by competitor type
        assert_eq!(analyzer.default_rate(CompetitorType::Nbfc), 18.0);
        assert_eq!(analyzer.default_rate(CompetitorType::Bank), 11.0);
        assert_eq!(analyzer.default_rate(CompetitorType::Informal), 24.0);
    }

    #[test]
    fn test_competitor_info() {
        let analyzer = test_analyzer();

        let nbfc = analyzer.get_competitor("nbfc_a").unwrap();
        assert_eq!(nbfc.display_name, "NBFC Provider A");
        assert_eq!(nbfc.competitor_type, CompetitorType::Nbfc);
        assert!(nbfc.rate_range.is_some());
        assert!(!nbfc.strengths.is_empty());
        assert!(!nbfc.weaknesses.is_empty());
    }
}
