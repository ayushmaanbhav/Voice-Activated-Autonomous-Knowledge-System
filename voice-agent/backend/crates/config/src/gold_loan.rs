//! Gold Loan Business Configuration
//!
//! Contains configurable business parameters for gold loan calculations.
//!
//! # P3-4 FIX: Interest Rate Configuration
//!
//! There are two types of interest rates in this configuration:
//!
//! 1. **`kotak_interest_rate`** (default: 10.5%): The "headline" rate used for
//!    marketing comparisons and general savings calculations. This represents
//!    the mid-tier rate that most customers see advertised.
//!
//! 2. **`tiered_rates`**: The actual rates applied based on loan amount:
//!    - Tier 1 (up to ₹1L): 11.5% (small loans, higher rate)
//!    - Tier 2 (₹1L-5L): 10.5% (matches headline rate)
//!    - Tier 3 (above ₹5L): 9.5% (premium customers, best rate)
//!
//! Use `get_tiered_rate(loan_amount)` for actual pricing.
//! Use `kotak_interest_rate` for marketing/comparison purposes.

use serde::{Deserialize, Serialize};

/// Gold loan business configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldLoanConfig {
    /// Current gold price per gram in INR
    #[serde(default = "default_gold_price")]
    pub gold_price_per_gram: f64,

    /// Kotak's gold loan interest rate (%)
    ///
    /// P3-4 FIX: This is the "headline" rate for marketing comparisons.
    /// For actual loan pricing, use `get_tiered_rate(loan_amount)` which
    /// applies tiered rates based on loan amount. The tier2_rate (medium loans)
    /// matches this value for consistency.
    #[serde(default = "default_kotak_rate")]
    pub kotak_interest_rate: f64,

    /// Loan-to-Value ratio (%)
    #[serde(default = "default_ltv")]
    pub ltv_percent: f64,

    /// Minimum loan amount in INR
    #[serde(default = "default_min_loan")]
    pub min_loan_amount: f64,

    /// Maximum loan amount in INR
    #[serde(default = "default_max_loan")]
    pub max_loan_amount: f64,

    /// Processing fee percentage
    #[serde(default = "default_processing_fee")]
    pub processing_fee_percent: f64,

    /// Gold purity factors for LTV calculation
    #[serde(default = "default_purity_factors")]
    pub purity_factors: PurityFactors,

    /// Competitor comparison rates
    #[serde(default)]
    pub competitor_rates: CompetitorRates,

    /// P2 FIX: Tiered interest rates based on loan amount
    #[serde(default)]
    pub tiered_rates: TieredRates,
}

/// P2 FIX: Tiered interest rates structure
///
/// Interest rates vary based on loan amount - higher amounts get better rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredRates {
    /// Rate for loans up to tier1_threshold
    #[serde(default = "default_tier1_rate")]
    pub tier1_rate: f64,
    /// Threshold for tier 1 (loans up to this amount get tier1_rate)
    #[serde(default = "default_tier1_threshold")]
    pub tier1_threshold: f64,
    /// Rate for loans between tier1 and tier2 thresholds
    #[serde(default = "default_tier2_rate")]
    pub tier2_rate: f64,
    /// Threshold for tier 2
    #[serde(default = "default_tier2_threshold")]
    pub tier2_threshold: f64,
    /// Rate for loans above tier2 threshold (premium customers)
    #[serde(default = "default_tier3_rate")]
    pub tier3_rate: f64,
}

/// Gold purity factors for different karats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurityFactors {
    #[serde(default = "default_24k_factor")]
    pub k24: f64,
    #[serde(default = "default_22k_factor")]
    pub k22: f64,
    #[serde(default = "default_18k_factor")]
    pub k18: f64,
    #[serde(default = "default_14k_factor")]
    pub k14: f64,
}

/// Competitor interest rates for comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorRates {
    #[serde(default = "default_muthoot_rate")]
    pub muthoot: f64,
    #[serde(default = "default_manappuram_rate")]
    pub manappuram: f64,
    #[serde(default = "default_iifl_rate")]
    pub iifl: f64,
    #[serde(default = "default_other_nbfc_rate")]
    pub other_nbfc: f64,
}

// P1 FIX: Use centralized constants
use crate::constants::{gold_prices, interest_rates, loan_tiers, ltv};

// Default values - now using centralized constants
fn default_gold_price() -> f64 {
    gold_prices::DEFAULT_24K_PER_GRAM
}

fn default_kotak_rate() -> f64 {
    interest_rates::DEFAULT_HEADLINE
}

fn default_ltv() -> f64 {
    ltv::MAX_LTV_PERCENT
}

fn default_min_loan() -> f64 {
    10000.0 // INR
}

fn default_max_loan() -> f64 {
    25000000.0 // INR (2.5 Cr)
}

fn default_processing_fee() -> f64 {
    1.0 // %
}

fn default_purity_factors() -> PurityFactors {
    PurityFactors::default()
}

fn default_24k_factor() -> f64 {
    1.0
}

fn default_22k_factor() -> f64 {
    gold_prices::PURITY_22K
}

fn default_18k_factor() -> f64 {
    gold_prices::PURITY_18K
}

fn default_14k_factor() -> f64 {
    0.585
}

fn default_muthoot_rate() -> f64 {
    18.0
}

fn default_manappuram_rate() -> f64 {
    19.0
}

fn default_iifl_rate() -> f64 {
    17.5
}

fn default_other_nbfc_rate() -> f64 {
    20.0
}

// P2 FIX: Tiered rate defaults - now using centralized constants
fn default_tier1_rate() -> f64 {
    interest_rates::TIER_1_STANDARD
}

fn default_tier1_threshold() -> f64 {
    loan_tiers::TIER_1_MAX
}

fn default_tier2_rate() -> f64 {
    interest_rates::TIER_2_HEADLINE
}

fn default_tier2_threshold() -> f64 {
    loan_tiers::TIER_2_MAX
}

fn default_tier3_rate() -> f64 {
    interest_rates::TIER_3_PREMIUM
}

impl Default for GoldLoanConfig {
    fn default() -> Self {
        Self {
            gold_price_per_gram: default_gold_price(),
            kotak_interest_rate: default_kotak_rate(),
            ltv_percent: default_ltv(),
            min_loan_amount: default_min_loan(),
            max_loan_amount: default_max_loan(),
            processing_fee_percent: default_processing_fee(),
            purity_factors: PurityFactors::default(),
            competitor_rates: CompetitorRates::default(),
            tiered_rates: TieredRates::default(),
        }
    }
}

/// P2 FIX: Default tiered rates
impl Default for TieredRates {
    fn default() -> Self {
        Self {
            tier1_rate: default_tier1_rate(),
            tier1_threshold: default_tier1_threshold(),
            tier2_rate: default_tier2_rate(),
            tier2_threshold: default_tier2_threshold(),
            tier3_rate: default_tier3_rate(),
        }
    }
}

impl Default for PurityFactors {
    fn default() -> Self {
        Self {
            k24: default_24k_factor(),
            k22: default_22k_factor(),
            k18: default_18k_factor(),
            k14: default_14k_factor(),
        }
    }
}

impl Default for CompetitorRates {
    fn default() -> Self {
        Self {
            muthoot: default_muthoot_rate(),
            manappuram: default_manappuram_rate(),
            iifl: default_iifl_rate(),
            other_nbfc: default_other_nbfc_rate(),
        }
    }
}

impl GoldLoanConfig {
    /// Get purity factor for a given karat string
    pub fn get_purity_factor(&self, purity: &str) -> f64 {
        match purity {
            "24K" => self.purity_factors.k24,
            "22K" => self.purity_factors.k22,
            "18K" => self.purity_factors.k18,
            "14K" => self.purity_factors.k14,
            _ => self.purity_factors.k18, // Default to 18K
        }
    }

    /// Get competitor rate by lender name
    pub fn get_competitor_rate(&self, lender: &str) -> f64 {
        match lender.to_lowercase().as_str() {
            "muthoot" | "muthoot finance" => self.competitor_rates.muthoot,
            "manappuram" | "manappuram finance" => self.competitor_rates.manappuram,
            "iifl" | "iifl finance" => self.competitor_rates.iifl,
            _ => self.competitor_rates.other_nbfc,
        }
    }

    /// Calculate gold value
    pub fn calculate_gold_value(&self, weight_grams: f64, purity: &str) -> f64 {
        let purity_factor = self.get_purity_factor(purity);
        weight_grams * self.gold_price_per_gram * purity_factor
    }

    /// Calculate maximum loan amount
    pub fn calculate_max_loan(&self, gold_value: f64) -> f64 {
        let max_from_ltv = gold_value * (self.ltv_percent / 100.0);
        max_from_ltv.min(self.max_loan_amount)
    }

    /// Calculate monthly savings when switching
    pub fn calculate_monthly_savings(&self, loan_amount: f64, current_rate: f64) -> f64 {
        let current_monthly = loan_amount * (current_rate / 100.0 / 12.0);
        let kotak_monthly = loan_amount * (self.kotak_interest_rate / 100.0 / 12.0);
        current_monthly - kotak_monthly
    }

    /// P2 FIX: Get the interest rate based on loan amount using tiered rates.
    ///
    /// Higher loan amounts qualify for better (lower) rates.
    pub fn get_tiered_rate(&self, loan_amount: f64) -> f64 {
        if loan_amount <= self.tiered_rates.tier1_threshold {
            self.tiered_rates.tier1_rate
        } else if loan_amount <= self.tiered_rates.tier2_threshold {
            self.tiered_rates.tier2_rate
        } else {
            self.tiered_rates.tier3_rate
        }
    }

    /// P2 FIX: Calculate monthly savings using tiered rates
    pub fn calculate_monthly_savings_tiered(&self, loan_amount: f64, current_rate: f64) -> f64 {
        let tiered_rate = self.get_tiered_rate(loan_amount);
        let current_monthly = loan_amount * (current_rate / 100.0 / 12.0);
        let kotak_monthly = loan_amount * (tiered_rate / 100.0 / 12.0);
        current_monthly - kotak_monthly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GoldLoanConfig::default();
        assert_eq!(config.kotak_interest_rate, 10.5);
        assert_eq!(config.ltv_percent, 75.0);
    }

    #[test]
    fn test_purity_factor() {
        let config = GoldLoanConfig::default();
        assert_eq!(config.get_purity_factor("22K"), 0.916);
        assert_eq!(config.get_purity_factor("24K"), 1.0);
    }

    #[test]
    fn test_gold_value_calculation() {
        let config = GoldLoanConfig::default();
        let value = config.calculate_gold_value(100.0, "22K");
        // 100g * 7500 * 0.916 = 687000
        assert!((value - 687000.0).abs() < 1.0);
    }

    #[test]
    fn test_savings_calculation() {
        let config = GoldLoanConfig::default();
        let savings = config.calculate_monthly_savings(100000.0, 18.0);
        // (100000 * 18/100/12) - (100000 * 10.5/100/12) = 1500 - 875 = 625
        assert!((savings - 625.0).abs() < 1.0);
    }

    #[test]
    fn test_tiered_rates() {
        let config = GoldLoanConfig::default();

        // Tier 1: Up to 1 lakh gets 11.5%
        assert_eq!(config.get_tiered_rate(50000.0), 11.5);
        assert_eq!(config.get_tiered_rate(100000.0), 11.5);

        // Tier 2: 1-5 lakh gets 10.5%
        assert_eq!(config.get_tiered_rate(200000.0), 10.5);
        assert_eq!(config.get_tiered_rate(500000.0), 10.5);

        // Tier 3: Above 5 lakh gets 9.5%
        assert_eq!(config.get_tiered_rate(600000.0), 9.5);
        assert_eq!(config.get_tiered_rate(1000000.0), 9.5);
    }
}
