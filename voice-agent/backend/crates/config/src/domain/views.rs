//! Crate-Specific Domain Views
//!
//! Each crate accesses domain configuration through a "view" that provides
//! only the information that crate needs, in terminology appropriate for that crate.

use std::sync::Arc;

use super::MasterDomainConfig;

/// View for the agent crate
/// Provides access to conversation stages, DST slots, scoring, objections
pub struct AgentDomainView {
    config: Arc<MasterDomainConfig>,
}

impl AgentDomainView {
    pub fn new(config: Arc<MasterDomainConfig>) -> Self {
        Self { config }
    }

    /// Get high-value thresholds for lead scoring
    pub fn high_value_amount_threshold(&self) -> f64 {
        self.config.high_value.amount_threshold
    }

    pub fn high_value_weight_threshold(&self) -> f64 {
        self.config.high_value.weight_threshold_grams
    }

    /// Check if signals indicate high-value customer
    pub fn is_high_value(&self, amount: Option<f64>, weight: Option<f64>) -> bool {
        self.config.is_high_value(amount, weight)
    }

    /// Get high-value features to highlight
    pub fn high_value_features(&self) -> &[String] {
        &self.config.high_value.features
    }

    /// Get competitor by name for comparison
    pub fn get_competitor_rate(&self, name: &str) -> Option<f64> {
        self.config.get_competitor(name).map(|c| c.typical_rate)
    }

    /// Get our rate for comparison
    pub fn our_rate_for_amount(&self, amount: f64) -> f64 {
        self.config.get_rate_for_amount(amount)
    }
}

/// View for the llm crate
/// Provides access to prompts, tool schemas, brand info
pub struct LlmDomainView {
    config: Arc<MasterDomainConfig>,
}

impl LlmDomainView {
    pub fn new(config: Arc<MasterDomainConfig>) -> Self {
        Self { config }
    }

    /// Get domain display name for prompts
    pub fn domain_name(&self) -> &str {
        &self.config.display_name
    }

    /// Get brand information for prompts
    pub fn bank_name(&self) -> &str {
        &self.config.brand.bank_name
    }

    pub fn agent_name(&self) -> &str {
        &self.config.brand.agent_name
    }

    pub fn helpline(&self) -> &str {
        &self.config.brand.helpline
    }

    /// Get key facts for system prompt
    pub fn key_facts(&self) -> Vec<String> {
        let mut facts = Vec::new();

        // Best interest rate
        if let Some(best_tier) = self.config.constants.interest_rates.tiers.last() {
            facts.push(format!("Interest rates: Starting from {}% p.a.", best_tier.rate));
        }

        // LTV
        facts.push(format!("LTV: Up to {}% of gold value", self.config.constants.ltv_percent));

        // Loan range
        let min = self.config.constants.loan_limits.min;
        let max = self.config.constants.loan_limits.max;
        facts.push(format!("Loan range: ₹{} to ₹{}", format_amount(min), format_amount(max)));

        facts
    }

    /// Get product variants for tool responses
    pub fn product_names(&self) -> Vec<&str> {
        self.config.products.values().map(|p| p.name.as_str()).collect()
    }
}

/// View for the tools crate
/// Provides access to tool configs, branch data, SMS templates, constants
pub struct ToolsDomainView {
    config: Arc<MasterDomainConfig>,
}

impl ToolsDomainView {
    pub fn new(config: Arc<MasterDomainConfig>) -> Self {
        Self { config }
    }

    /// Get interest rate for eligibility calculations
    pub fn get_rate_for_amount(&self, amount: f64) -> f64 {
        self.config.get_rate_for_amount(amount)
    }

    /// Get LTV percentage
    pub fn ltv_percent(&self) -> f64 {
        self.config.constants.ltv_percent
    }

    /// Get purity factor for gold type
    pub fn purity_factor(&self, purity: &str) -> f64 {
        self.config.constants.purity_factors
            .get(purity)
            .copied()
            .unwrap_or(1.0)
    }

    /// Get gold price per gram
    pub fn gold_price_per_gram(&self) -> f64 {
        self.config.constants.gold_price_per_gram
    }

    /// Get loan limits
    pub fn min_loan_amount(&self) -> f64 {
        self.config.constants.loan_limits.min
    }

    pub fn max_loan_amount(&self) -> f64 {
        self.config.constants.loan_limits.max
    }

    /// Get processing fee percentage
    pub fn processing_fee_percent(&self) -> f64 {
        self.config.constants.processing_fee_percent
    }

    /// Get competitor info for savings calculations
    pub fn get_competitor(&self, name: &str) -> Option<CompetitorInfo> {
        self.config.get_competitor(name).map(|c| CompetitorInfo {
            name: c.display_name.clone(),
            rate: c.typical_rate,
            ltv: c.ltv_percent,
        })
    }

    /// Get brand info for SMS/responses
    pub fn bank_name(&self) -> &str {
        &self.config.brand.bank_name
    }

    pub fn helpline(&self) -> &str {
        &self.config.brand.helpline
    }
}

/// Simplified competitor info for tools
#[derive(Debug, Clone)]
pub struct CompetitorInfo {
    pub name: String,
    pub rate: f64,
    pub ltv: f64,
}

/// Format amount in Indian style (lakhs/crores)
fn format_amount(amount: f64) -> String {
    if amount >= 10_000_000.0 {
        format!("{:.1} Cr", amount / 10_000_000.0)
    } else if amount >= 100_000.0 {
        format!("{:.1} L", amount / 100_000.0)
    } else {
        format!("{:.0}", amount)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_amount() {
        assert_eq!(format_amount(10000.0), "10000");
        assert_eq!(format_amount(100000.0), "1.0 L");
        assert_eq!(format_amount(2500000.0), "25.0 L");
        assert_eq!(format_amount(10000000.0), "1.0 Cr");
        assert_eq!(format_amount(25000000.0), "2.5 Cr");
    }
}
