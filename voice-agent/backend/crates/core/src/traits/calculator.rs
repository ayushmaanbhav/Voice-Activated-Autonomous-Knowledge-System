//! Domain Calculator trait for business calculations
//!
//! This module provides a domain-agnostic interface for all business calculations
//! including EMI, asset value, max loan, and rate tier selection. All formulas
//! and constants are loaded from configuration.
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_core::traits::DomainCalculator;
//!
//! // Calculator is created from domain config
//! let calc = config_bridge.calculator();
//!
//! // Calculate EMI for a loan
//! let emi = calc.calculate_emi(100000.0, 12.0, 12);
//!
//! // Get rate for loan amount (tiered)
//! let rate = calc.get_rate_for_amount(500000.0);
//! ```

use std::collections::HashMap;

/// Rate tier definition loaded from config
#[derive(Debug, Clone)]
pub struct RateTier {
    /// Maximum amount for this tier (None = unlimited)
    pub max_amount: Option<f64>,
    /// Interest rate for this tier (percentage)
    pub rate: f64,
    /// Human-readable tier label (e.g., "Standard", "Premium", "Elite")
    pub label: String,
}

/// Quality/purity factor definition
#[derive(Debug, Clone)]
pub struct QualityFactor {
    /// Quality grade ID (e.g., "K24", "K22", "Grade_A")
    pub id: String,
    /// Display name (e.g., "24 Karat", "Grade A")
    pub display_name: String,
    /// Multiplication factor (e.g., 1.0, 0.916, 0.75)
    pub factor: f64,
}

/// Savings analysis result
#[derive(Debug, Clone)]
pub struct SavingsResult {
    /// Monthly interest savings
    pub monthly_interest_savings: f64,
    /// Monthly EMI savings
    pub monthly_emi_savings: f64,
    /// Total interest savings over tenure
    pub total_interest_savings: f64,
    /// Total EMI savings over tenure
    pub total_emi_savings: f64,
    /// Our rate used for calculation
    pub our_rate: f64,
    /// Comparison rate used
    pub comparison_rate: f64,
    /// Tenure in months
    pub tenure_months: i64,
}

/// Domain calculator error
#[derive(Debug, Clone)]
pub enum CalculatorError {
    /// Invalid input parameter
    InvalidInput { param: String, message: String },
    /// Configuration missing
    ConfigMissing { key: String },
    /// Calculation overflow/underflow
    CalculationError { message: String },
}

impl std::fmt::Display for CalculatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { param, message } => {
                write!(f, "Invalid input '{}': {}", param, message)
            }
            Self::ConfigMissing { key } => write!(f, "Missing config key: {}", key),
            Self::CalculationError { message } => write!(f, "Calculation error: {}", message),
        }
    }
}

impl std::error::Error for CalculatorError {}

/// Domain-agnostic calculator trait
///
/// Provides all business calculations with formulas and constants
/// loaded from domain configuration. This allows different domains
/// (gold loan, car loan, home loan) to use the same interface with
/// different parameters.
pub trait DomainCalculator: Send + Sync {
    /// Calculate EMI (Equated Monthly Installment)
    ///
    /// Uses the standard amortization formula:
    /// EMI = P × r × (1 + r)^n / [(1 + r)^n - 1]
    ///
    /// Where:
    /// - P = principal loan amount
    /// - r = monthly interest rate (annual_rate / 12 / 100)
    /// - n = tenure in months
    ///
    /// # Arguments
    /// * `principal` - Loan amount
    /// * `annual_rate` - Annual interest rate as percentage (e.g., 12.0 for 12%)
    /// * `tenure_months` - Loan tenure in months
    ///
    /// # Returns
    /// Monthly EMI amount
    fn calculate_emi(&self, principal: f64, annual_rate: f64, tenure_months: i64) -> f64;

    /// Calculate total interest paid over the loan tenure
    ///
    /// Total Interest = (EMI × tenure_months) - principal
    fn calculate_total_interest(
        &self,
        principal: f64,
        annual_rate: f64,
        tenure_months: i64,
    ) -> f64;

    /// Calculate asset value from quantity, unit price, and quality factor
    ///
    /// Asset Value = quantity × unit_price × quality_factor
    ///
    /// For gold loan: weight_grams × price_per_gram × purity_factor
    /// For car loan: base_value × depreciation_factor
    fn calculate_asset_value(
        &self,
        quantity: f64,
        unit_price: f64,
        quality_factor: f64,
    ) -> f64;

    /// Calculate maximum loan amount from asset value
    ///
    /// Max Loan = asset_value × (ltv_percent / 100)
    fn calculate_max_loan(&self, asset_value: f64) -> f64;

    /// Get interest rate for a given loan amount (tiered pricing)
    ///
    /// Looks up the appropriate rate tier based on loan amount
    fn get_rate_for_amount(&self, amount: f64) -> f64;

    /// Get the rate tier label for a given amount
    fn get_rate_tier_label(&self, amount: f64) -> String;

    /// Get LTV (Loan-to-Value) percentage
    fn ltv_percent(&self) -> f64;

    /// Get asset unit price (e.g., gold price per gram)
    fn asset_unit_price(&self) -> f64;

    /// Get quality factor for a given grade
    ///
    /// Returns the multiplication factor for the quality grade.
    /// For gold: K24=1.0, K22=0.916, K18=0.75, K14=0.585
    fn get_quality_factor(&self, grade: &str) -> Option<f64>;

    /// Get all available quality grades
    fn quality_grades(&self) -> Vec<&QualityFactor>;

    /// Get all rate tiers
    fn rate_tiers(&self) -> Vec<&RateTier>;

    /// Get base interest rate
    fn base_rate(&self) -> f64;

    /// Get minimum loan amount
    fn min_loan_amount(&self) -> f64;

    /// Get maximum loan amount
    fn max_loan_amount(&self) -> f64;

    /// Get processing fee percentage
    fn processing_fee_percent(&self) -> f64;

    /// Get foreclosure fee percentage
    fn foreclosure_fee_percent(&self) -> f64;

    /// Calculate savings when switching from another rate
    ///
    /// Compares current rate with our rate and calculates monthly/total savings
    fn calculate_savings(
        &self,
        loan_amount: f64,
        current_rate: f64,
        tenure_months: i64,
    ) -> SavingsResult;

    /// Calculate monthly interest amount
    fn calculate_monthly_interest(&self, principal: f64, annual_rate: f64) -> f64 {
        principal * (annual_rate / 100.0 / 12.0)
    }
}

/// Config-driven calculator implementation
///
/// Loads all formulas and constants from domain configuration.
/// This struct is created by the DomainBridge from MasterDomainConfig.
#[derive(Debug, Clone)]
pub struct ConfigDrivenCalculator {
    /// Rate tiers loaded from config
    rate_tiers: Vec<RateTier>,
    /// Quality factors loaded from config
    quality_factors: HashMap<String, QualityFactor>,
    /// LTV percentage
    ltv_percent: f64,
    /// Asset unit price (e.g., gold price per gram)
    asset_unit_price: f64,
    /// Base interest rate
    base_rate: f64,
    /// Minimum loan amount
    min_loan_amount: f64,
    /// Maximum loan amount
    max_loan_amount: f64,
    /// Processing fee percentage
    processing_fee_percent: f64,
    /// Foreclosure fee percentage
    foreclosure_fee_percent: f64,
}

impl ConfigDrivenCalculator {
    /// Create a new calculator with the given configuration
    pub fn new(
        rate_tiers: Vec<RateTier>,
        quality_factors: Vec<QualityFactor>,
        ltv_percent: f64,
        asset_unit_price: f64,
        base_rate: f64,
        min_loan_amount: f64,
        max_loan_amount: f64,
        processing_fee_percent: f64,
        foreclosure_fee_percent: f64,
    ) -> Self {
        let quality_map = quality_factors
            .into_iter()
            .map(|q| (q.id.clone(), q))
            .collect();

        Self {
            rate_tiers,
            quality_factors: quality_map,
            ltv_percent,
            asset_unit_price,
            base_rate,
            min_loan_amount,
            max_loan_amount,
            processing_fee_percent,
            foreclosure_fee_percent,
        }
    }

}

impl DomainCalculator for ConfigDrivenCalculator {
    fn calculate_emi(&self, principal: f64, annual_rate: f64, tenure_months: i64) -> f64 {
        // Use shared financial calculation from crate::financial
        crate::financial::calculate_emi(principal, annual_rate, tenure_months)
    }

    fn calculate_total_interest(
        &self,
        principal: f64,
        annual_rate: f64,
        tenure_months: i64,
    ) -> f64 {
        // Use shared financial calculation from crate::financial
        crate::financial::calculate_total_interest(principal, annual_rate, tenure_months)
    }

    fn calculate_asset_value(
        &self,
        quantity: f64,
        unit_price: f64,
        quality_factor: f64,
    ) -> f64 {
        // EXACT FORMULA: Asset Value = quantity × unit_price × quality_factor
        quantity * unit_price * quality_factor
    }

    fn calculate_max_loan(&self, asset_value: f64) -> f64 {
        // EXACT FORMULA: Max Loan = asset_value × (LTV / 100)
        asset_value * (self.ltv_percent / 100.0)
    }

    fn get_rate_for_amount(&self, amount: f64) -> f64 {
        // EXACT LOGIC preserved from tools/src/gold_loan/tools.rs:140-159
        for tier in &self.rate_tiers {
            if let Some(max) = tier.max_amount {
                if amount <= max {
                    return tier.rate;
                }
            } else {
                // Unlimited tier (last tier)
                return tier.rate;
            }
        }
        // Fallback to base rate if no tiers defined
        self.base_rate
    }

    fn get_rate_tier_label(&self, amount: f64) -> String {
        for tier in &self.rate_tiers {
            if let Some(max) = tier.max_amount {
                if amount <= max {
                    return tier.label.clone();
                }
            } else {
                return tier.label.clone();
            }
        }
        "Standard".to_string()
    }

    fn ltv_percent(&self) -> f64 {
        self.ltv_percent
    }

    fn asset_unit_price(&self) -> f64 {
        self.asset_unit_price
    }

    fn get_quality_factor(&self, grade: &str) -> Option<f64> {
        self.quality_factors.get(grade).map(|q| q.factor)
    }

    fn quality_grades(&self) -> Vec<&QualityFactor> {
        self.quality_factors.values().collect()
    }

    fn rate_tiers(&self) -> Vec<&RateTier> {
        self.rate_tiers.iter().collect()
    }

    fn base_rate(&self) -> f64 {
        self.base_rate
    }

    fn min_loan_amount(&self) -> f64 {
        self.min_loan_amount
    }

    fn max_loan_amount(&self) -> f64 {
        self.max_loan_amount
    }

    fn processing_fee_percent(&self) -> f64 {
        self.processing_fee_percent
    }

    fn foreclosure_fee_percent(&self) -> f64 {
        self.foreclosure_fee_percent
    }

    fn calculate_savings(
        &self,
        loan_amount: f64,
        current_rate: f64,
        tenure_months: i64,
    ) -> SavingsResult {
        let our_rate = self.get_rate_for_amount(loan_amount);

        // Monthly interest savings
        let current_monthly_interest = self.calculate_monthly_interest(loan_amount, current_rate);
        let our_monthly_interest = self.calculate_monthly_interest(loan_amount, our_rate);
        let monthly_interest_savings = current_monthly_interest - our_monthly_interest;

        // EMI savings
        let current_emi = self.calculate_emi(loan_amount, current_rate, tenure_months);
        let our_emi = self.calculate_emi(loan_amount, our_rate, tenure_months);
        let monthly_emi_savings = current_emi - our_emi;

        // Total interest savings
        let current_total_interest =
            self.calculate_total_interest(loan_amount, current_rate, tenure_months);
        let our_total_interest =
            self.calculate_total_interest(loan_amount, our_rate, tenure_months);
        let total_interest_savings = current_total_interest - our_total_interest;

        SavingsResult {
            monthly_interest_savings,
            monthly_emi_savings,
            total_interest_savings,
            total_emi_savings: monthly_emi_savings * tenure_months as f64,
            our_rate,
            comparison_rate: current_rate,
            tenure_months,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create test calculator with typical loan domain values
    fn test_calculator() -> ConfigDrivenCalculator {
        ConfigDrivenCalculator::new(
            vec![
                RateTier {
                    max_amount: Some(100_000.0),
                    rate: 11.5,
                    label: "Standard".to_string(),
                },
                RateTier {
                    max_amount: Some(500_000.0),
                    rate: 10.5,
                    label: "Premium".to_string(),
                },
                RateTier {
                    max_amount: None,
                    rate: 9.5,
                    label: "Elite".to_string(),
                },
            ],
            vec![
                QualityFactor { id: "K24".to_string(), display_name: "24 Karat".to_string(), factor: 1.0 },
                QualityFactor { id: "K22".to_string(), display_name: "22 Karat".to_string(), factor: 0.916 },
                QualityFactor { id: "K18".to_string(), display_name: "18 Karat".to_string(), factor: 0.75 },
                QualityFactor { id: "K14".to_string(), display_name: "14 Karat".to_string(), factor: 0.585 },
            ],
            75.0,  // ltv
            7500.0, // asset price
            10.5,   // base rate
            10_000.0, // min loan
            25_000_000.0, // max loan
            1.0, // processing fee
            0.0, // foreclosure fee
        )
    }

    fn gold_loan_calculator() -> ConfigDrivenCalculator {
        test_calculator()
    }

    #[test]
    fn test_emi_calculation_exact() {
        let calc = gold_loan_calculator();
        // Test case from original code: P=100000, rate=12%, months=12 → EMI≈8884.88
        let emi = calc.calculate_emi(100_000.0, 12.0, 12);
        assert!((emi - 8884.88).abs() < 0.01, "EMI should be ~8884.88, got {}", emi);
    }

    #[test]
    fn test_emi_zero_rate() {
        let calc = gold_loan_calculator();
        // Edge case: zero rate should return simple division
        let emi = calc.calculate_emi(12000.0, 0.0, 12);
        assert_eq!(emi, 1000.0);
    }

    #[test]
    fn test_asset_value_calculation() {
        let calc = gold_loan_calculator();
        // 100g * 7500 * 0.916 (22K) = 687000
        let value = calc.calculate_asset_value(100.0, 7500.0, 0.916);
        assert_eq!(value, 687_000.0);
    }

    #[test]
    fn test_max_loan_calculation() {
        let calc = gold_loan_calculator();
        // 687000 * 75% = 515250
        let max_loan = calc.calculate_max_loan(687_000.0);
        assert_eq!(max_loan, 515_250.0);
    }

    #[test]
    fn test_rate_tier_selection_exact() {
        let calc = gold_loan_calculator();
        // Tier 1: <= 1 lakh → 11.5%
        assert_eq!(calc.get_rate_for_amount(50_000.0), 11.5);
        assert_eq!(calc.get_rate_for_amount(100_000.0), 11.5);
        // Tier 2: 1-5 lakh → 10.5%
        assert_eq!(calc.get_rate_for_amount(100_001.0), 10.5);
        assert_eq!(calc.get_rate_for_amount(500_000.0), 10.5);
        // Tier 3: > 5 lakh → 9.5%
        assert_eq!(calc.get_rate_for_amount(500_001.0), 9.5);
        assert_eq!(calc.get_rate_for_amount(1_000_000.0), 9.5);
    }

    #[test]
    fn test_quality_factors() {
        let calc = gold_loan_calculator();
        assert_eq!(calc.get_quality_factor("K24"), Some(1.0));
        assert_eq!(calc.get_quality_factor("K22"), Some(0.916));
        assert_eq!(calc.get_quality_factor("K18"), Some(0.75));
        assert_eq!(calc.get_quality_factor("K14"), Some(0.585));
        assert_eq!(calc.get_quality_factor("unknown"), None);
    }

    #[test]
    fn test_total_interest() {
        let calc = gold_loan_calculator();
        let total = calc.calculate_total_interest(100_000.0, 12.0, 12);
        // EMI ≈ 8884.88, total paid = 8884.88 * 12 = 106618.56
        // Interest = 106618.56 - 100000 = 6618.56
        assert!((total - 6618.58).abs() < 0.1);
    }

    #[test]
    fn test_savings_calculation() {
        let calc = gold_loan_calculator();
        // Compare 18% (NBFC) vs our rate for 5 lakh loan
        let savings = calc.calculate_savings(500_000.0, 18.0, 12);

        // Our rate for 5 lakh = 10.5%
        assert_eq!(savings.our_rate, 10.5);
        assert_eq!(savings.comparison_rate, 18.0);

        // Monthly interest savings should be positive
        assert!(savings.monthly_interest_savings > 0.0);
        assert!(savings.total_interest_savings > 0.0);
    }
}
