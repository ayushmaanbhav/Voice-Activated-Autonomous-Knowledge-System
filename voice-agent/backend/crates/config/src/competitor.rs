//! Competitor configuration
//!
//! Detailed competitor information for comparison and positioning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Competitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorConfig {
    /// Competitor details
    #[serde(default)]
    pub competitors: HashMap<String, Competitor>,
    /// Comparison points
    #[serde(default)]
    pub comparison_points: Vec<ComparisonPoint>,
    /// Switching benefits
    #[serde(default)]
    pub switching_benefits: SwitchingBenefits,
}

impl Default for CompetitorConfig {
    fn default() -> Self {
        let mut competitors = HashMap::new();

        competitors.insert("muthoot".to_string(), Competitor::muthoot());
        competitors.insert("manappuram".to_string(), Competitor::manappuram());
        competitors.insert("iifl".to_string(), Competitor::iifl());
        competitors.insert("hdfc".to_string(), Competitor::hdfc());
        competitors.insert("sbi".to_string(), Competitor::sbi());
        competitors.insert("icici".to_string(), Competitor::icici());

        Self {
            competitors,
            comparison_points: ComparisonPoint::default_points(),
            switching_benefits: SwitchingBenefits::default(),
        }
    }
}

/// Competitor details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Competitor {
    /// Competitor ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Full name
    pub full_name: String,
    /// Competitor type
    pub competitor_type: CompetitorType,
    /// Interest rate range
    pub interest_rate_min: f64,
    pub interest_rate_max: f64,
    /// Processing fee
    pub processing_fee_percent: f64,
    /// LTV offered
    pub ltv_percent: f64,
    /// Known weaknesses
    #[serde(default)]
    pub weaknesses: Vec<String>,
    /// Known strengths
    #[serde(default)]
    pub strengths: Vec<String>,
    /// Common customer complaints
    #[serde(default)]
    pub common_complaints: Vec<String>,
    /// Estimated market share (%)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub market_share: Option<f64>,
    /// Branch count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_count: Option<usize>,
    /// Key regions
    #[serde(default)]
    pub key_regions: Vec<String>,
}

/// Competitor type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompetitorType {
    /// Non-Banking Financial Company
    Nbfc,
    /// Scheduled Commercial Bank
    Bank,
    /// Small Finance Bank
    SmallFinanceBank,
    /// Regional Rural Bank
    Rrb,
}

impl Competitor {
    /// Muthoot Finance
    pub fn muthoot() -> Self {
        Self {
            id: "muthoot".to_string(),
            name: "Muthoot".to_string(),
            full_name: "Muthoot Finance Limited".to_string(),
            competitor_type: CompetitorType::Nbfc,
            interest_rate_min: 12.0,
            interest_rate_max: 24.0,
            processing_fee_percent: 1.5,
            ltv_percent: 75.0,
            weaknesses: vec![
                "Higher interest rates than banks".to_string(),
                "Not RBI bank-regulated (NBFC)".to_string(),
                "Inconsistent customer service".to_string(),
                "Limited digital services".to_string(),
            ],
            strengths: vec![
                "Largest gold loan company".to_string(),
                "Wide branch network".to_string(),
                "Quick processing".to_string(),
            ],
            common_complaints: vec![
                "High interest rates".to_string(),
                "Aggressive recovery practices".to_string(),
                "Hidden charges".to_string(),
                "Poor customer service".to_string(),
            ],
            market_share: Some(25.0),
            branch_count: Some(5500),
            key_regions: vec!["Kerala".to_string(), "Tamil Nadu".to_string(), "Karnataka".to_string()],
        }
    }

    /// Manappuram Finance
    pub fn manappuram() -> Self {
        Self {
            id: "manappuram".to_string(),
            name: "Manappuram".to_string(),
            full_name: "Manappuram Finance Limited".to_string(),
            competitor_type: CompetitorType::Nbfc,
            interest_rate_min: 14.0,
            interest_rate_max: 26.0,
            processing_fee_percent: 1.5,
            ltv_percent: 75.0,
            weaknesses: vec![
                "Higher rates than banks".to_string(),
                "NBFC status (less regulated)".to_string(),
                "Limited insurance coverage".to_string(),
            ],
            strengths: vec![
                "Second largest gold loan company".to_string(),
                "Quick disbursal".to_string(),
                "Strong South India presence".to_string(),
            ],
            common_complaints: vec![
                "Very high interest rates".to_string(),
                "Unclear terms and conditions".to_string(),
                "Auction threats".to_string(),
            ],
            market_share: Some(15.0),
            branch_count: Some(4500),
            key_regions: vec!["Kerala".to_string(), "Tamil Nadu".to_string()],
        }
    }

    /// IIFL Finance
    pub fn iifl() -> Self {
        Self {
            id: "iifl".to_string(),
            name: "IIFL".to_string(),
            full_name: "IIFL Finance Limited".to_string(),
            competitor_type: CompetitorType::Nbfc,
            interest_rate_min: 11.0,
            interest_rate_max: 24.0,
            processing_fee_percent: 1.5,
            ltv_percent: 75.0,
            weaknesses: vec![
                "NBFC status".to_string(),
                "Recent regulatory issues".to_string(),
                "Limited tier-2 city presence".to_string(),
            ],
            strengths: vec![
                "Digital platform".to_string(),
                "Quick processing".to_string(),
                "Metro city focus".to_string(),
            ],
            common_complaints: vec![
                "Regulatory concerns".to_string(),
                "Customer service issues".to_string(),
                "Complex fee structure".to_string(),
            ],
            market_share: Some(8.0),
            branch_count: Some(3000),
            key_regions: vec!["Maharashtra".to_string(), "Gujarat".to_string()],
        }
    }

    /// HDFC Bank
    pub fn hdfc() -> Self {
        Self {
            id: "hdfc".to_string(),
            name: "HDFC".to_string(),
            full_name: "HDFC Bank Limited".to_string(),
            competitor_type: CompetitorType::Bank,
            interest_rate_min: 9.5,
            interest_rate_max: 12.0,
            processing_fee_percent: 1.0,
            ltv_percent: 75.0,
            weaknesses: vec![
                "Longer processing time".to_string(),
                "More documentation required".to_string(),
                "Limited dedicated gold loan branches".to_string(),
            ],
            strengths: vec![
                "Bank safety".to_string(),
                "Low interest rates".to_string(),
                "Digital services".to_string(),
            ],
            common_complaints: vec![
                "Slower processing".to_string(),
                "Complex documentation".to_string(),
            ],
            market_share: Some(5.0),
            branch_count: Some(7000),
            key_regions: vec!["Pan India".to_string()],
        }
    }

    /// SBI
    pub fn sbi() -> Self {
        Self {
            id: "sbi".to_string(),
            name: "SBI".to_string(),
            full_name: "State Bank of India".to_string(),
            competitor_type: CompetitorType::Bank,
            interest_rate_min: 8.5,
            interest_rate_max: 11.0,
            processing_fee_percent: 0.5,
            ltv_percent: 75.0,
            weaknesses: vec![
                "Very slow processing".to_string(),
                "Bureaucratic process".to_string(),
                "Limited gold loan focus".to_string(),
            ],
            strengths: vec![
                "Lowest rates".to_string(),
                "Government bank trust".to_string(),
                "Largest bank network".to_string(),
            ],
            common_complaints: vec![
                "Extremely slow".to_string(),
                "Multiple visits required".to_string(),
                "Complex procedures".to_string(),
            ],
            market_share: Some(3.0),
            branch_count: Some(22000),
            key_regions: vec!["Pan India".to_string()],
        }
    }

    /// ICICI Bank
    pub fn icici() -> Self {
        Self {
            id: "icici".to_string(),
            name: "ICICI".to_string(),
            full_name: "ICICI Bank Limited".to_string(),
            competitor_type: CompetitorType::Bank,
            interest_rate_min: 10.0,
            interest_rate_max: 12.5,
            processing_fee_percent: 1.0,
            ltv_percent: 75.0,
            weaknesses: vec![
                "Longer processing".to_string(),
                "More documentation".to_string(),
            ],
            strengths: vec![
                "Bank safety".to_string(),
                "Good digital platform".to_string(),
                "Competitive rates".to_string(),
            ],
            common_complaints: vec![
                "Slower than NBFCs".to_string(),
                "Rigid processes".to_string(),
            ],
            market_share: Some(4.0),
            branch_count: Some(5500),
            key_regions: vec!["Pan India".to_string()],
        }
    }
}

/// Comparison point for competitive positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonPoint {
    /// Comparison category
    pub category: String,
    /// Kotak advantage
    pub kotak_value: String,
    /// Typical NBFC value
    pub nbfc_value: String,
    /// Why Kotak is better
    pub why_better: String,
}

impl ComparisonPoint {
    /// Get default comparison points
    pub fn default_points() -> Vec<Self> {
        vec![
            ComparisonPoint {
                category: "Interest Rate".to_string(),
                kotak_value: "9.5% - 11.5%".to_string(),
                nbfc_value: "12% - 24%".to_string(),
                why_better: "Save thousands in interest annually".to_string(),
            },
            ComparisonPoint {
                category: "Regulator".to_string(),
                kotak_value: "RBI-regulated Bank".to_string(),
                nbfc_value: "NBFC (less regulated)".to_string(),
                why_better: "Higher safety and compliance standards".to_string(),
            },
            ComparisonPoint {
                category: "Foreclosure Charges".to_string(),
                kotak_value: "Zero".to_string(),
                nbfc_value: "1% - 3%".to_string(),
                why_better: "Prepay anytime without penalty".to_string(),
            },
            ComparisonPoint {
                category: "Gold Security".to_string(),
                kotak_value: "Bank-grade vaults, insured".to_string(),
                nbfc_value: "Variable security standards".to_string(),
                why_better: "Your gold is safer in a bank vault".to_string(),
            },
            ComparisonPoint {
                category: "Processing Time".to_string(),
                kotak_value: "30 minutes".to_string(),
                nbfc_value: "30-60 minutes".to_string(),
                why_better: "Equally fast with better terms".to_string(),
            },
            ComparisonPoint {
                category: "Digital Tracking".to_string(),
                kotak_value: "Full digital access".to_string(),
                nbfc_value: "Limited or none".to_string(),
                why_better: "Monitor your gold status anytime".to_string(),
            },
        ]
    }
}

/// Switching benefits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingBenefits {
    /// Balance transfer benefits
    #[serde(default)]
    pub balance_transfer: BalanceTransferBenefits,
    /// Common objection handlers
    #[serde(default)]
    pub objection_handlers: Vec<ObjectionHandler>,
}

impl Default for SwitchingBenefits {
    fn default() -> Self {
        Self {
            balance_transfer: BalanceTransferBenefits::default(),
            objection_handlers: vec![
                ObjectionHandler {
                    objection: "I'm happy with my current lender".to_string(),
                    response: "That's great! But have you compared the total interest you're paying? Many customers save 30-40% just by switching to our lower rates.".to_string(),
                    follow_up: "Would you like me to show you a quick comparison?".to_string(),
                },
                ObjectionHandler {
                    objection: "Switching is too much hassle".to_string(),
                    response: "We've made it very simple - we handle all the paperwork. Most customers complete the switch in just one visit.".to_string(),
                    follow_up: "Would you like me to explain the 3-step process?".to_string(),
                },
                ObjectionHandler {
                    objection: "My current lender has my gold".to_string(),
                    response: "We can arrange a direct balance transfer where we pay off your existing loan and transfer your gold to our insured vault - all in one day.".to_string(),
                    follow_up: "This way you don't have to handle the gold yourself.".to_string(),
                },
            ],
        }
    }
}

/// Balance transfer benefits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceTransferBenefits {
    /// Processing fee for balance transfer
    #[serde(default = "default_bt_fee")]
    pub processing_fee_percent: f64,
    /// Top-up available
    #[serde(default = "default_true")]
    pub top_up_available: bool,
    /// Same day transfer possible
    #[serde(default = "default_true")]
    pub same_day_transfer: bool,
    /// Benefits list
    #[serde(default)]
    pub benefits: Vec<String>,
}

fn default_bt_fee() -> f64 {
    0.5
}

fn default_true() -> bool {
    true
}

impl Default for BalanceTransferBenefits {
    fn default() -> Self {
        Self {
            processing_fee_percent: default_bt_fee(),
            top_up_available: true,
            same_day_transfer: true,
            benefits: vec![
                "Lower interest rate from day one".to_string(),
                "Reduced processing fee (0.5%)".to_string(),
                "Same day transfer possible".to_string(),
                "Top-up available on existing gold".to_string(),
                "Zero foreclosure charges".to_string(),
                "Bank-grade security for your gold".to_string(),
            ],
        }
    }
}

/// Objection handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectionHandler {
    /// The objection
    pub objection: String,
    /// Response to objection
    pub response: String,
    /// Follow-up question
    pub follow_up: String,
}

impl CompetitorConfig {
    /// Get competitor by ID or name
    pub fn get_competitor(&self, id_or_name: &str) -> Option<&Competitor> {
        let lower = id_or_name.to_lowercase();
        self.competitors.get(&lower).or_else(|| {
            self.competitors
                .values()
                .find(|c| c.name.to_lowercase() == lower || c.full_name.to_lowercase().contains(&lower))
        })
    }

    /// Calculate potential savings when switching from competitor
    pub fn calculate_savings(&self, competitor_id: &str, loan_amount: f64, kotak_rate: f64) -> Option<MonthlySavings> {
        let competitor = self.get_competitor(competitor_id)?;

        // Use average of competitor's rate range
        let competitor_rate = (competitor.interest_rate_min + competitor.interest_rate_max) / 2.0;

        let competitor_monthly = loan_amount * (competitor_rate / 100.0 / 12.0);
        let kotak_monthly = loan_amount * (kotak_rate / 100.0 / 12.0);
        let monthly_savings = competitor_monthly - kotak_monthly;
        let annual_savings = monthly_savings * 12.0;

        Some(MonthlySavings {
            competitor_name: competitor.name.clone(),
            competitor_rate,
            kotak_rate,
            monthly_savings,
            annual_savings,
        })
    }

    /// Get all NBFCs
    pub fn get_nbfcs(&self) -> Vec<&Competitor> {
        self.competitors
            .values()
            .filter(|c| c.competitor_type == CompetitorType::Nbfc)
            .collect()
    }

    /// Get all banks
    pub fn get_banks(&self) -> Vec<&Competitor> {
        self.competitors
            .values()
            .filter(|c| c.competitor_type == CompetitorType::Bank)
            .collect()
    }

    /// Get competitor weaknesses for positioning
    pub fn get_weaknesses(&self, competitor_id: &str) -> Vec<String> {
        self.get_competitor(competitor_id)
            .map(|c| c.weaknesses.clone())
            .unwrap_or_default()
    }
}

/// Monthly savings calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonthlySavings {
    pub competitor_name: String,
    pub competitor_rate: f64,
    pub kotak_rate: f64,
    pub monthly_savings: f64,
    pub annual_savings: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CompetitorConfig::default();
        assert!(!config.competitors.is_empty());
        assert!(config.get_competitor("muthoot").is_some());
        assert!(config.get_competitor("Muthoot Finance").is_some());
    }

    #[test]
    fn test_get_nbfcs() {
        let config = CompetitorConfig::default();
        let nbfcs = config.get_nbfcs();
        assert!(nbfcs.iter().any(|c| c.id == "muthoot"));
        assert!(nbfcs.iter().any(|c| c.id == "manappuram"));
        assert!(!nbfcs.iter().any(|c| c.id == "sbi"));
    }

    #[test]
    fn test_get_banks() {
        let config = CompetitorConfig::default();
        let banks = config.get_banks();
        assert!(banks.iter().any(|c| c.id == "sbi"));
        assert!(banks.iter().any(|c| c.id == "hdfc"));
        assert!(!banks.iter().any(|c| c.id == "muthoot"));
    }

    #[test]
    fn test_calculate_savings() {
        let config = CompetitorConfig::default();

        let savings = config.calculate_savings("muthoot", 100_000.0, 10.0).unwrap();

        // Muthoot avg rate ~18%, Kotak 10%
        // Monthly: 100000 * (18/100/12) - 100000 * (10/100/12) = 1500 - 833 = ~667
        assert!(savings.monthly_savings > 500.0);
        assert!(savings.annual_savings > 6000.0);
    }

    #[test]
    fn test_comparison_points() {
        let config = CompetitorConfig::default();
        assert!(!config.comparison_points.is_empty());

        let rate_comparison = config
            .comparison_points
            .iter()
            .find(|p| p.category == "Interest Rate");
        assert!(rate_comparison.is_some());
    }

    #[test]
    fn test_weaknesses() {
        let config = CompetitorConfig::default();
        let weaknesses = config.get_weaknesses("muthoot");
        assert!(!weaknesses.is_empty());
        assert!(weaknesses.iter().any(|w| w.contains("NBFC")));
    }

    #[test]
    fn test_switching_benefits() {
        let config = CompetitorConfig::default();
        assert!(config.switching_benefits.balance_transfer.top_up_available);
        assert!(!config.switching_benefits.objection_handlers.is_empty());
    }
}
