//! Domain configuration loader
//!
//! Unified interface for loading and accessing all domain-specific configuration.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use crate::{
    branch::BranchConfig, competitor::CompetitorConfig, product::ProductConfig,
    prompts::PromptTemplates, ConfigError, GoldLoanConfig,
};

/// Complete domain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainConfig {
    /// Domain name
    #[serde(default = "default_domain")]
    pub domain: String,
    /// Domain version
    #[serde(default = "default_version")]
    pub version: String,
    /// Gold loan business configuration
    #[serde(default)]
    pub gold_loan: GoldLoanConfig,
    /// Branch configuration
    #[serde(default)]
    pub branches: BranchConfig,
    /// Product configuration
    #[serde(default)]
    pub product: ProductConfig,
    /// Competitor configuration
    #[serde(default)]
    pub competitors: CompetitorConfig,
    /// Prompt templates
    #[serde(default)]
    pub prompts: PromptTemplates,
}

fn default_domain() -> String {
    "gold_loan".to_string()
}

fn default_version() -> String {
    "1.0.0".to_string()
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            domain: default_domain(),
            version: default_version(),
            gold_loan: GoldLoanConfig::default(),
            branches: BranchConfig::default(),
            product: ProductConfig::default(),
            competitors: CompetitorConfig::default(),
            prompts: PromptTemplates::default(),
        }
    }
}

impl DomainConfig {
    /// Create new domain config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from YAML file
    pub fn from_yaml_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()));
        }

        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::ParseError(e.to_string()))?;

        serde_yaml::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Load from JSON file
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(ConfigError::FileNotFound(path.display().to_string()));
        }

        let content =
            std::fs::read_to_string(path).map_err(|e| ConfigError::ParseError(e.to_string()))?;

        serde_json::from_str(&content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Save to YAML file
    pub fn to_yaml_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let content =
            serde_yaml::to_string(self).map_err(|e| ConfigError::ParseError(e.to_string()))?;

        std::fs::write(path, content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Save to JSON file
    pub fn to_json_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::ParseError(e.to_string()))?;

        std::fs::write(path, content).map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// Validate configuration
    ///
    /// P1 FIX: Comprehensive validation including:
    /// - Interest rate ranges (0-30%)
    /// - LTV ranges (0-90% per RBI guidelines)
    /// - Tiered rate ordering (higher tiers should have lower rates)
    /// - Competitor rates should be higher than Kotak rates
    /// - Processing fee ranges (0-5%)
    /// - Business logic consistency checks
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // === Gold Loan Config Validation ===
        let gl = &self.gold_loan;

        // Interest rate validation (0-30% range)
        if gl.kotak_interest_rate <= 0.0 || gl.kotak_interest_rate > 30.0 {
            errors.push(format!(
                "Interest rate {} out of valid range (0-30%)",
                gl.kotak_interest_rate
            ));
        }

        // LTV validation (0-90% per RBI guidelines for gold loans)
        if gl.ltv_percent <= 0.0 || gl.ltv_percent > 90.0 {
            errors.push(format!("LTV {} out of valid range (0-90%)", gl.ltv_percent));
        }

        // Tiered rate ordering validation (higher tiers should have lower rates)
        if gl.tiered_rates.tier1_rate <= gl.tiered_rates.tier2_rate {
            errors.push(format!(
                "Tier 1 rate ({}) should be > Tier 2 rate ({}) - higher loans get better rates",
                gl.tiered_rates.tier1_rate, gl.tiered_rates.tier2_rate
            ));
        }
        if gl.tiered_rates.tier2_rate <= gl.tiered_rates.tier3_rate {
            errors.push(format!(
                "Tier 2 rate ({}) should be > Tier 3 rate ({}) - higher loans get better rates",
                gl.tiered_rates.tier2_rate, gl.tiered_rates.tier3_rate
            ));
        }

        // Tier threshold ordering validation
        if gl.tiered_rates.tier1_threshold >= gl.tiered_rates.tier2_threshold {
            errors.push(format!(
                "Tier 1 threshold ({}) should be < Tier 2 threshold ({})",
                gl.tiered_rates.tier1_threshold, gl.tiered_rates.tier2_threshold
            ));
        }

        // Competitor rates should be higher than Kotak rates (otherwise no competitive advantage)
        let kotak_best_rate = gl.tiered_rates.tier3_rate; // Best rate for comparison
        if gl.competitor_rates.muthoot < kotak_best_rate {
            errors.push(format!(
                "Muthoot rate ({}) < Kotak best rate ({}) - no competitive advantage",
                gl.competitor_rates.muthoot, kotak_best_rate
            ));
        }
        if gl.competitor_rates.manappuram < kotak_best_rate {
            errors.push(format!(
                "Manappuram rate ({}) < Kotak best rate ({}) - no competitive advantage",
                gl.competitor_rates.manappuram, kotak_best_rate
            ));
        }
        if gl.competitor_rates.iifl < kotak_best_rate {
            errors.push(format!(
                "IIFL rate ({}) < Kotak best rate ({}) - no competitive advantage",
                gl.competitor_rates.iifl, kotak_best_rate
            ));
        }

        // Processing fee validation (0-5% range)
        if gl.processing_fee_percent < 0.0 || gl.processing_fee_percent > 5.0 {
            errors.push(format!(
                "Processing fee {} out of valid range (0-5%)",
                gl.processing_fee_percent
            ));
        }

        // Loan amount range validation
        if gl.min_loan_amount <= 0.0 {
            errors.push("Minimum loan amount must be positive".to_string());
        }
        if gl.max_loan_amount <= gl.min_loan_amount {
            errors.push(format!(
                "Maximum loan ({}) must be greater than minimum ({})",
                gl.max_loan_amount, gl.min_loan_amount
            ));
        }

        // Gold price validation
        if gl.gold_price_per_gram <= 0.0 {
            errors.push("Gold price must be positive".to_string());
        }

        // Purity factor validation (must be 0-1)
        if gl.purity_factors.k24 < 0.0 || gl.purity_factors.k24 > 1.0 {
            errors.push("24K purity factor must be between 0 and 1".to_string());
        }
        if gl.purity_factors.k22 < 0.0 || gl.purity_factors.k22 > 1.0 {
            errors.push("22K purity factor must be between 0 and 1".to_string());
        }
        if gl.purity_factors.k18 < 0.0 || gl.purity_factors.k18 > 1.0 {
            errors.push("18K purity factor must be between 0 and 1".to_string());
        }
        // Purity ordering (24K > 22K > 18K > 14K)
        if gl.purity_factors.k24 < gl.purity_factors.k22 {
            errors.push("24K purity factor should be >= 22K factor".to_string());
        }
        if gl.purity_factors.k22 < gl.purity_factors.k18 {
            errors.push("22K purity factor should be >= 18K factor".to_string());
        }

        // === Product Config Validation ===
        if self.product.variants.is_empty() {
            errors.push("At least one product variant required".to_string());
        }

        // === Prompts Validation ===
        if self.prompts.system_prompt.agent_name.is_empty() {
            errors.push("Agent name required in prompts".to_string());
        }

        // === Branches Validation ===
        // Check for duplicate branch IDs
        let branch_ids: Vec<_> = self.branches.branches.iter().map(|b| &b.id).collect();
        let mut seen = std::collections::HashSet::new();
        for id in &branch_ids {
            if !seen.insert(*id) {
                errors.push(format!("Duplicate branch ID: {}", id));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate and return warnings (non-fatal issues)
    pub fn validate_with_warnings(&self) -> (Result<(), Vec<String>>, Vec<String>) {
        let mut warnings = Vec::new();

        // Check for missing branch coverage
        if self.branches.branches.len() < 10 {
            warnings.push(format!(
                "Only {} branches configured - consider adding more for better coverage",
                self.branches.branches.len()
            ));
        }

        // Check for missing competitors
        if self.competitors.competitors.is_empty() {
            warnings.push("No competitors configured - comparison features disabled".to_string());
        }

        // Check gold price is recent (basic sanity check)
        if self.gold_loan.gold_price_per_gram < 5000.0 {
            warnings.push(format!(
                "Gold price {} seems low - verify it's current",
                self.gold_loan.gold_price_per_gram
            ));
        }
        if self.gold_loan.gold_price_per_gram > 10000.0 {
            warnings.push(format!(
                "Gold price {} seems high - verify it's correct",
                self.gold_loan.gold_price_per_gram
            ));
        }

        (self.validate(), warnings)
    }

    /// Merge with another config (other takes precedence for non-default values)
    pub fn merge(&mut self, other: &DomainConfig) {
        // Merge simple fields
        if other.domain != default_domain() {
            self.domain = other.domain.clone();
        }
        if other.version != default_version() {
            self.version = other.version.clone();
        }

        // Gold loan config - merge rates if different from default
        let default_gold = GoldLoanConfig::default();
        if other.gold_loan.kotak_interest_rate != default_gold.kotak_interest_rate {
            self.gold_loan.kotak_interest_rate = other.gold_loan.kotak_interest_rate;
        }
        if other.gold_loan.gold_price_per_gram != default_gold.gold_price_per_gram {
            self.gold_loan.gold_price_per_gram = other.gold_loan.gold_price_per_gram;
        }

        // Add branches from other
        for branch in &other.branches.branches {
            if !self.branches.branches.iter().any(|b| b.id == branch.id) {
                self.branches.branches.push(branch.clone());
            }
        }

        // Add product variants from other
        for variant in &other.product.variants {
            if !self.product.variants.iter().any(|v| v.id == variant.id) {
                self.product.variants.push(variant.clone());
            }
        }

        // Add competitors from other
        for (id, competitor) in &other.competitors.competitors {
            self.competitors
                .competitors
                .entry(id.clone())
                .or_insert(competitor.clone());
        }
    }
}

/// Domain configuration manager with hot-reload support
pub struct DomainConfigManager {
    /// Current configuration
    config: Arc<RwLock<DomainConfig>>,
    /// Config file path (if loaded from file)
    config_path: Option<String>,
}

impl DomainConfigManager {
    /// Create new manager with default config
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(DomainConfig::default())),
            config_path: None,
        }
    }

    /// Create manager with config
    pub fn with_config(config: DomainConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            config_path: None,
        }
    }

    /// Load from file
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let config = if path_str.ends_with(".yaml") || path_str.ends_with(".yml") {
            DomainConfig::from_yaml_file(&path)?
        } else {
            DomainConfig::from_json_file(&path)?
        };

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            config_path: Some(path_str),
        })
    }

    /// Reload configuration from file
    pub fn reload(&self) -> Result<(), ConfigError> {
        let path = self
            .config_path
            .as_ref()
            .ok_or_else(|| ConfigError::FileNotFound("No config path set".to_string()))?;

        let new_config = if path.ends_with(".yaml") || path.ends_with(".yml") {
            DomainConfig::from_yaml_file(path)?
        } else {
            DomainConfig::from_json_file(path)?
        };

        *self.config.write() = new_config;
        Ok(())
    }

    /// Get current configuration
    pub fn get(&self) -> DomainConfig {
        self.config.read().clone()
    }

    /// Get configuration reference
    pub fn config(&self) -> Arc<RwLock<DomainConfig>> {
        Arc::clone(&self.config)
    }

    /// Update configuration
    pub fn update(&self, config: DomainConfig) {
        *self.config.write() = config;
    }

    /// Get gold loan config
    pub fn gold_loan(&self) -> GoldLoanConfig {
        self.config.read().gold_loan.clone()
    }

    /// Get branch config
    pub fn branches(&self) -> BranchConfig {
        self.config.read().branches.clone()
    }

    /// Get product config
    pub fn product(&self) -> ProductConfig {
        self.config.read().product.clone()
    }

    /// Get competitor config
    pub fn competitors(&self) -> CompetitorConfig {
        self.config.read().competitors.clone()
    }

    /// Get prompts
    pub fn prompts(&self) -> PromptTemplates {
        self.config.read().prompts.clone()
    }

    /// Get current gold price
    pub fn gold_price(&self) -> f64 {
        self.config.read().gold_loan.gold_price_per_gram
    }

    /// Update gold price (real-time update)
    pub fn update_gold_price(&self, price: f64) {
        self.config.write().gold_loan.gold_price_per_gram = price;
    }

    /// Get interest rate for loan amount
    pub fn get_interest_rate(&self, loan_amount: f64) -> f64 {
        self.config.read().gold_loan.get_tiered_rate(loan_amount)
    }

    /// Calculate savings vs competitor
    pub fn calculate_competitor_savings(
        &self,
        competitor: &str,
        loan_amount: f64,
    ) -> Option<crate::competitor::MonthlySavings> {
        let config = self.config.read();
        let kotak_rate = config.gold_loan.get_tiered_rate(loan_amount);
        config
            .competitors
            .calculate_savings(competitor, loan_amount, kotak_rate)
    }

    /// Find nearby branches
    pub fn find_branches_by_city(&self, city: &str) -> Vec<crate::branch::Branch> {
        self.config
            .read()
            .branches
            .find_by_city(city)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Check doorstep service availability
    pub fn doorstep_available(&self, city: &str) -> bool {
        self.config.read().branches.doorstep_available(city)
    }

    /// Get system prompt for stage
    pub fn get_system_prompt(&self, stage: Option<&str>, customer_name: Option<&str>) -> String {
        self.config
            .read()
            .prompts
            .build_system_prompt(stage, customer_name)
    }

    /// Get greeting for current time
    pub fn get_greeting(&self, hour: u32, customer_name: Option<&str>) -> String {
        let config = self.config.read();
        let agent_name = &config.prompts.system_prompt.agent_name;
        config.prompts.get_greeting(hour, agent_name, customer_name)
    }
}

impl Default for DomainConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global domain configuration instance
static DOMAIN_CONFIG: once_cell::sync::Lazy<DomainConfigManager> =
    once_cell::sync::Lazy::new(DomainConfigManager::new);

/// Get global domain configuration
pub fn domain_config() -> &'static DomainConfigManager {
    &DOMAIN_CONFIG
}

/// Initialize global domain configuration from file
pub fn init_domain_config(path: impl AsRef<Path>) -> Result<(), ConfigError> {
    let manager = DomainConfigManager::from_file(path)?;
    DOMAIN_CONFIG.update(manager.get());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DomainConfig::default();
        assert_eq!(config.domain, "gold_loan");
        assert!(!config.product.variants.is_empty());
    }

    #[test]
    fn test_validation() {
        let config = DomainConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = DomainConfig::default();
        bad_config.gold_loan.kotak_interest_rate = -1.0;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_validation_interest_rate_range() {
        let mut config = DomainConfig::default();

        // Too high interest rate
        config.gold_loan.kotak_interest_rate = 35.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("Interest rate"));
    }

    #[test]
    fn test_validation_ltv_range() {
        let mut config = DomainConfig::default();

        // LTV > 90% violates RBI guidelines
        config.gold_loan.ltv_percent = 95.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("LTV")));
    }

    #[test]
    fn test_validation_tiered_rates() {
        let mut config = DomainConfig::default();

        // Invalid: tier1 rate <= tier2 rate (should be descending)
        config.gold_loan.tiered_rates.tier1_rate = 9.0;
        config.gold_loan.tiered_rates.tier2_rate = 10.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .iter()
            .any(|e| e.contains("Tier 1 rate")));
    }

    #[test]
    fn test_validation_competitor_rates() {
        let mut config = DomainConfig::default();

        // Invalid: competitor rate lower than Kotak (no competitive advantage)
        config.gold_loan.competitor_rates.muthoot = 8.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|e| e.contains("Muthoot")));
    }

    #[test]
    fn test_validation_processing_fee() {
        let mut config = DomainConfig::default();

        // Invalid: processing fee > 5%
        config.gold_loan.processing_fee_percent = 6.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .iter()
            .any(|e| e.contains("Processing fee")));
    }

    #[test]
    fn test_validate_with_warnings() {
        let config = DomainConfig::default();
        let (result, warnings) = config.validate_with_warnings();

        // Default config should be valid
        assert!(result.is_ok());
        // But may have warnings about missing branches/competitors
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_manager() {
        let manager = DomainConfigManager::new();

        assert!(manager.gold_price() > 0.0);
        assert!(!manager.product().variants.is_empty());
    }

    #[test]
    fn test_update_gold_price() {
        let manager = DomainConfigManager::new();
        let original = manager.gold_price();

        manager.update_gold_price(8000.0);
        assert_eq!(manager.gold_price(), 8000.0);

        manager.update_gold_price(original);
    }

    #[test]
    fn test_get_interest_rate() {
        let manager = DomainConfigManager::new();

        // Small loan gets tier 1 rate
        let rate1 = manager.get_interest_rate(50_000.0);
        // Large loan gets tier 3 rate
        let rate3 = manager.get_interest_rate(1_000_000.0);

        assert!(rate3 < rate1);
    }

    #[test]
    fn test_competitor_savings() {
        let manager = DomainConfigManager::new();
        let savings = manager.calculate_competitor_savings("muthoot", 100_000.0);

        assert!(savings.is_some());
        let savings = savings.unwrap();
        assert!(savings.monthly_savings > 0.0);
    }

    #[test]
    fn test_doorstep_availability() {
        let manager = DomainConfigManager::new();

        assert!(manager.doorstep_available("Mumbai"));
        assert!(!manager.doorstep_available("SmallVillage"));
    }

    #[test]
    fn test_system_prompt() {
        let manager = DomainConfigManager::new();
        let prompt = manager.get_system_prompt(Some("discovery"), Some("Raj"));

        assert!(prompt.contains("discovery"));
        assert!(prompt.contains("Raj"));
    }

    #[test]
    fn test_greeting() {
        let manager = DomainConfigManager::new();

        let morning = manager.get_greeting(9, Some("Raj"));
        assert!(morning.contains("morning"));

        let evening = manager.get_greeting(19, None);
        assert!(evening.contains("evening"));
    }

    #[test]
    fn test_merge() {
        let mut base = DomainConfig::default();
        let mut overlay = DomainConfig::default();
        overlay.gold_loan.gold_price_per_gram = 8500.0;
        overlay.version = "2.0.0".to_string();

        base.merge(&overlay);

        assert_eq!(base.gold_loan.gold_price_per_gram, 8500.0);
        assert_eq!(base.version, "2.0.0");
    }
}
