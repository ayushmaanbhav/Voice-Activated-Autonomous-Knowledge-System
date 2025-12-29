//! Product configuration
//!
//! Defines gold loan product features, eligibility, and benefits.

use serde::{Deserialize, Serialize};

/// Gold loan product configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductConfig {
    /// Product variants
    #[serde(default)]
    pub variants: Vec<ProductVariant>,
    /// Eligibility criteria
    #[serde(default)]
    pub eligibility: EligibilityConfig,
    /// Documentation requirements
    #[serde(default)]
    pub documentation: DocumentationConfig,
    /// Features and benefits
    #[serde(default)]
    pub features: ProductFeatures,
    /// Tenure options
    #[serde(default)]
    pub tenure: TenureConfig,
    /// Fees and charges
    #[serde(default)]
    pub fees: FeesConfig,
}

impl Default for ProductConfig {
    fn default() -> Self {
        Self {
            variants: vec![
                ProductVariant::standard(),
                ProductVariant::shakti_gold(),
                ProductVariant::bullet_repayment(),
                ProductVariant::overdraft(),
            ],
            eligibility: EligibilityConfig::default(),
            documentation: DocumentationConfig::default(),
            features: ProductFeatures::default(),
            tenure: TenureConfig::default(),
            fees: FeesConfig::default(),
        }
    }
}

/// Product variant (different loan types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductVariant {
    /// Variant ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Target customer segment
    #[serde(default)]
    pub target_segment: Vec<String>,
    /// Interest rate range
    pub interest_rate_min: f64,
    pub interest_rate_max: f64,
    /// Special benefits
    #[serde(default)]
    pub benefits: Vec<String>,
    /// Is this variant active
    #[serde(default = "default_true")]
    pub active: bool,
}

fn default_true() -> bool {
    true
}

impl ProductVariant {
    /// Standard gold loan
    pub fn standard() -> Self {
        Self {
            id: "standard".to_string(),
            name: "Kotak Gold Loan".to_string(),
            description: "Standard gold loan with competitive rates and flexible repayment".to_string(),
            target_segment: vec!["all".to_string()],
            interest_rate_min: 9.5,
            interest_rate_max: 11.5,
            benefits: vec![
                "30-minute approval".to_string(),
                "Zero foreclosure charges".to_string(),
                "Flexible tenure".to_string(),
                "Bank-grade security".to_string(),
            ],
            active: true,
        }
    }

    /// Shakti Gold for women
    pub fn shakti_gold() -> Self {
        Self {
            id: "shakti_gold".to_string(),
            name: "Kotak Shakti Gold".to_string(),
            description: "Special gold loan program for women with preferential rates".to_string(),
            target_segment: vec!["women".to_string()],
            interest_rate_min: 9.25,
            interest_rate_max: 11.0,
            benefits: vec![
                "0.25% lower interest rate".to_string(),
                "Women-priority branches".to_string(),
                "Dedicated relationship manager".to_string(),
                "Flexible EMI options".to_string(),
            ],
            active: true,
        }
    }

    /// Bullet repayment scheme
    pub fn bullet_repayment() -> Self {
        Self {
            id: "bullet".to_string(),
            name: "Kotak Gold Bullet".to_string(),
            description: "Pay only interest monthly, repay principal at end of tenure".to_string(),
            target_segment: vec!["business".to_string(), "seasonal_income".to_string()],
            interest_rate_min: 10.0,
            interest_rate_max: 12.0,
            benefits: vec![
                "Lower monthly outflow".to_string(),
                "Principal repaid at tenure end".to_string(),
                "Ideal for business needs".to_string(),
                "Flexible interest payment".to_string(),
            ],
            active: true,
        }
    }

    /// Overdraft facility
    pub fn overdraft() -> Self {
        Self {
            id: "overdraft".to_string(),
            name: "Kotak Gold Overdraft".to_string(),
            description: "Revolving credit against gold - pay interest only on used amount".to_string(),
            target_segment: vec!["high_value".to_string(), "business".to_string()],
            interest_rate_min: 10.5,
            interest_rate_max: 12.5,
            benefits: vec![
                "Pay interest only on used amount".to_string(),
                "Revolving credit facility".to_string(),
                "Withdraw and repay anytime".to_string(),
                "No prepayment charges".to_string(),
            ],
            active: true,
        }
    }
}

/// Eligibility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityConfig {
    /// Minimum age
    #[serde(default = "default_min_age")]
    pub min_age: u32,
    /// Maximum age
    #[serde(default = "default_max_age")]
    pub max_age: u32,
    /// Citizenship requirements
    #[serde(default)]
    pub citizenship: Vec<String>,
    /// Accepted gold purity
    #[serde(default)]
    pub gold_purity: GoldPurityRequirements,
    /// ID proof types accepted
    #[serde(default)]
    pub id_proof_types: Vec<String>,
    /// Address proof types accepted
    #[serde(default)]
    pub address_proof_types: Vec<String>,
    /// Existing customer benefits
    #[serde(default)]
    pub existing_customer_benefits: ExistingCustomerBenefits,
}

fn default_min_age() -> u32 {
    21
}

fn default_max_age() -> u32 {
    65
}

impl Default for EligibilityConfig {
    fn default() -> Self {
        Self {
            min_age: default_min_age(),
            max_age: default_max_age(),
            citizenship: vec!["Indian".to_string(), "NRI".to_string()],
            gold_purity: GoldPurityRequirements::default(),
            id_proof_types: vec![
                "Aadhaar Card".to_string(),
                "PAN Card".to_string(),
                "Passport".to_string(),
                "Voter ID".to_string(),
                "Driving License".to_string(),
            ],
            address_proof_types: vec![
                "Aadhaar Card".to_string(),
                "Utility Bill".to_string(),
                "Passport".to_string(),
                "Bank Statement".to_string(),
                "Rent Agreement".to_string(),
            ],
            existing_customer_benefits: ExistingCustomerBenefits::default(),
        }
    }
}

/// Gold purity requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldPurityRequirements {
    /// Minimum purity accepted
    #[serde(default = "default_min_purity")]
    pub min_purity_karat: u32,
    /// Accepted purity levels
    #[serde(default)]
    pub accepted_purity: Vec<String>,
    /// Items not accepted
    #[serde(default)]
    pub not_accepted: Vec<String>,
}

fn default_min_purity() -> u32 {
    18
}

impl Default for GoldPurityRequirements {
    fn default() -> Self {
        Self {
            min_purity_karat: default_min_purity(),
            accepted_purity: vec![
                "24K".to_string(),
                "22K".to_string(),
                "20K".to_string(),
                "18K".to_string(),
            ],
            not_accepted: vec![
                "Gold-plated items".to_string(),
                "Gold coins (non-hallmarked)".to_string(),
                "Items below 18K".to_string(),
                "Damaged/broken items beyond repair".to_string(),
            ],
        }
    }
}

/// Existing customer benefits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExistingCustomerBenefits {
    /// Reduced documentation
    #[serde(default = "default_true")]
    pub reduced_documentation: bool,
    /// Faster processing
    #[serde(default = "default_true")]
    pub faster_processing: bool,
    /// Rate discount for existing customers
    #[serde(default = "default_existing_discount")]
    pub rate_discount_percent: f64,
    /// Pre-approved limit available
    #[serde(default)]
    pub pre_approved_available: bool,
}

fn default_existing_discount() -> f64 {
    0.25
}

impl Default for ExistingCustomerBenefits {
    fn default() -> Self {
        Self {
            reduced_documentation: true,
            faster_processing: true,
            rate_discount_percent: default_existing_discount(),
            pre_approved_available: true,
        }
    }
}

/// Documentation requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentationConfig {
    /// Required documents
    #[serde(default)]
    pub required: Vec<DocumentRequirement>,
    /// Optional documents
    #[serde(default)]
    pub optional: Vec<DocumentRequirement>,
    /// E-KYC enabled
    #[serde(default = "default_true")]
    pub ekyc_enabled: bool,
    /// Video KYC enabled
    #[serde(default = "default_true")]
    pub video_kyc_enabled: bool,
}

impl Default for DocumentationConfig {
    fn default() -> Self {
        Self {
            required: vec![
                DocumentRequirement {
                    name: "Identity Proof".to_string(),
                    description: "Aadhaar Card, PAN Card, Passport, or Voter ID".to_string(),
                    alternatives: vec![
                        "Aadhaar".to_string(),
                        "PAN".to_string(),
                        "Passport".to_string(),
                        "Voter ID".to_string(),
                    ],
                },
                DocumentRequirement {
                    name: "Address Proof".to_string(),
                    description: "Recent utility bill, bank statement, or Aadhaar".to_string(),
                    alternatives: vec![
                        "Utility Bill".to_string(),
                        "Bank Statement".to_string(),
                        "Aadhaar".to_string(),
                    ],
                },
            ],
            optional: vec![
                DocumentRequirement {
                    name: "Passport Photo".to_string(),
                    description: "Recent passport-size photograph".to_string(),
                    alternatives: vec![],
                },
            ],
            ekyc_enabled: true,
            video_kyc_enabled: true,
        }
    }
}

/// Document requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRequirement {
    /// Document name
    pub name: String,
    /// Description
    pub description: String,
    /// Alternative documents accepted
    #[serde(default)]
    pub alternatives: Vec<String>,
}

/// Product features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductFeatures {
    /// Key selling points
    #[serde(default)]
    pub key_selling_points: Vec<SellingPoint>,
    /// Unique differentiators
    #[serde(default)]
    pub differentiators: Vec<String>,
    /// Digital features
    #[serde(default)]
    pub digital_features: DigitalFeatures,
    /// Safety features
    #[serde(default)]
    pub safety_features: Vec<String>,
}

impl Default for ProductFeatures {
    fn default() -> Self {
        Self {
            key_selling_points: vec![
                SellingPoint {
                    headline: "30-Minute Approval".to_string(),
                    description: "Get your loan approved in just 30 minutes at the branch".to_string(),
                    icon: "clock".to_string(),
                },
                SellingPoint {
                    headline: "Lowest Interest Rates".to_string(),
                    description: "Starting at 9.5% - among the lowest in the market".to_string(),
                    icon: "percent".to_string(),
                },
                SellingPoint {
                    headline: "Bank-Grade Security".to_string(),
                    description: "RBI-regulated bank with insured vaults".to_string(),
                    icon: "shield".to_string(),
                },
                SellingPoint {
                    headline: "Zero Foreclosure Charges".to_string(),
                    description: "Prepay anytime without any penalty".to_string(),
                    icon: "check".to_string(),
                },
            ],
            differentiators: vec![
                "RBI-regulated scheduled bank (not NBFC)".to_string(),
                "1600+ branches across India".to_string(),
                "Digital gold tracking".to_string(),
                "Relationship manager for high-value loans".to_string(),
                "Doorstep service in metro cities".to_string(),
            ],
            digital_features: DigitalFeatures::default(),
            safety_features: vec![
                "Bank-grade security vaults".to_string(),
                "Full insurance coverage".to_string(),
                "CCTV surveillance 24/7".to_string(),
                "Digital tracking of gold status".to_string(),
                "Tamper-proof sealed storage".to_string(),
            ],
        }
    }
}

/// Key selling point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SellingPoint {
    /// Headline
    pub headline: String,
    /// Description
    pub description: String,
    /// Icon name
    pub icon: String,
}

/// Digital features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalFeatures {
    /// Mobile app available
    #[serde(default = "default_true")]
    pub mobile_app: bool,
    /// Online tracking
    #[serde(default = "default_true")]
    pub online_tracking: bool,
    /// Digital repayment
    #[serde(default = "default_true")]
    pub digital_repayment: bool,
    /// E-statements
    #[serde(default = "default_true")]
    pub e_statements: bool,
    /// WhatsApp updates
    #[serde(default = "default_true")]
    pub whatsapp_updates: bool,
}

impl Default for DigitalFeatures {
    fn default() -> Self {
        Self {
            mobile_app: true,
            online_tracking: true,
            digital_repayment: true,
            e_statements: true,
            whatsapp_updates: true,
        }
    }
}

/// Tenure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenureConfig {
    /// Minimum tenure in months
    #[serde(default = "default_min_tenure")]
    pub min_months: u32,
    /// Maximum tenure in months
    #[serde(default = "default_max_tenure")]
    pub max_months: u32,
    /// Available tenure options
    #[serde(default)]
    pub options: Vec<u32>,
    /// Renewal allowed
    #[serde(default = "default_true")]
    pub renewal_allowed: bool,
}

fn default_min_tenure() -> u32 {
    3
}

fn default_max_tenure() -> u32 {
    36
}

impl Default for TenureConfig {
    fn default() -> Self {
        Self {
            min_months: default_min_tenure(),
            max_months: default_max_tenure(),
            options: vec![3, 6, 9, 12, 18, 24, 36],
            renewal_allowed: true,
        }
    }
}

/// Fees configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeesConfig {
    /// Processing fee
    #[serde(default)]
    pub processing_fee: FeeStructure,
    /// Valuation charges
    #[serde(default)]
    pub valuation_charges: FeeStructure,
    /// Foreclosure charges
    #[serde(default)]
    pub foreclosure_charges: FeeStructure,
    /// Late payment charges
    #[serde(default)]
    pub late_payment: FeeStructure,
    /// Documentation charges
    #[serde(default)]
    pub documentation_charges: FeeStructure,
    /// Insurance charges
    #[serde(default)]
    pub insurance: FeeStructure,
}

impl Default for FeesConfig {
    fn default() -> Self {
        Self {
            processing_fee: FeeStructure {
                fee_type: FeeType::Percentage,
                value: 1.0,
                min_amount: Some(500.0),
                max_amount: Some(25000.0),
                description: "1% of loan amount".to_string(),
            },
            valuation_charges: FeeStructure {
                fee_type: FeeType::Free,
                value: 0.0,
                min_amount: None,
                max_amount: None,
                description: "Free gold valuation".to_string(),
            },
            foreclosure_charges: FeeStructure {
                fee_type: FeeType::Free,
                value: 0.0,
                min_amount: None,
                max_amount: None,
                description: "Zero foreclosure charges".to_string(),
            },
            late_payment: FeeStructure {
                fee_type: FeeType::Percentage,
                value: 2.0,
                min_amount: None,
                max_amount: None,
                description: "2% per month on overdue amount".to_string(),
            },
            documentation_charges: FeeStructure {
                fee_type: FeeType::Free,
                value: 0.0,
                min_amount: None,
                max_amount: None,
                description: "No documentation charges".to_string(),
            },
            insurance: FeeStructure {
                fee_type: FeeType::Included,
                value: 0.0,
                min_amount: None,
                max_amount: None,
                description: "Gold insurance included".to_string(),
            },
        }
    }
}

/// Fee structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeStructure {
    /// Fee type
    pub fee_type: FeeType,
    /// Value (percentage or fixed amount)
    pub value: f64,
    /// Minimum amount (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_amount: Option<f64>,
    /// Maximum amount (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_amount: Option<f64>,
    /// Description for display
    pub description: String,
}

/// Fee type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FeeType {
    /// Fixed amount
    Fixed,
    /// Percentage of amount
    #[default]
    Percentage,
    /// Free / waived
    Free,
    /// Included in other charges
    Included,
}

impl Default for FeeStructure {
    fn default() -> Self {
        Self {
            fee_type: FeeType::Free,
            value: 0.0,
            min_amount: None,
            max_amount: None,
            description: "No charges".to_string(),
        }
    }
}

impl ProductConfig {
    /// Get variant by ID
    pub fn get_variant(&self, id: &str) -> Option<&ProductVariant> {
        self.variants.iter().find(|v| v.id == id && v.active)
    }

    /// Get variants for segment
    pub fn get_variants_for_segment(&self, segment: &str) -> Vec<&ProductVariant> {
        let segment_lower = segment.to_lowercase();
        self.variants
            .iter()
            .filter(|v| {
                v.active
                    && (v.target_segment.iter().any(|s| s.to_lowercase() == segment_lower)
                        || v.target_segment.contains(&"all".to_string()))
            })
            .collect()
    }

    /// Check eligibility by age
    pub fn check_age_eligibility(&self, age: u32) -> bool {
        age >= self.eligibility.min_age && age <= self.eligibility.max_age
    }

    /// Get processing fee for amount
    pub fn calculate_processing_fee(&self, loan_amount: f64) -> f64 {
        match self.fees.processing_fee.fee_type {
            FeeType::Fixed => self.fees.processing_fee.value,
            FeeType::Percentage => {
                let fee = loan_amount * (self.fees.processing_fee.value / 100.0);
                let min = self.fees.processing_fee.min_amount.unwrap_or(0.0);
                let max = self.fees.processing_fee.max_amount.unwrap_or(f64::MAX);
                fee.max(min).min(max)
            }
            FeeType::Free | FeeType::Included => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProductConfig::default();
        assert!(!config.variants.is_empty());
        assert!(config.get_variant("standard").is_some());
        assert!(config.get_variant("shakti_gold").is_some());
    }

    #[test]
    fn test_age_eligibility() {
        let config = ProductConfig::default();
        assert!(config.check_age_eligibility(30));
        assert!(!config.check_age_eligibility(18));
        assert!(!config.check_age_eligibility(70));
    }

    #[test]
    fn test_processing_fee() {
        let config = ProductConfig::default();

        // 1% of 100,000 = 1,000
        let fee = config.calculate_processing_fee(100_000.0);
        assert!((fee - 1000.0).abs() < 1.0);

        // Min 500 applies for small loans
        let fee = config.calculate_processing_fee(10_000.0);
        assert_eq!(fee, 500.0);

        // Max 25,000 applies for large loans
        let fee = config.calculate_processing_fee(10_000_000.0);
        assert_eq!(fee, 25_000.0);
    }

    #[test]
    fn test_segment_variants() {
        let config = ProductConfig::default();

        let women_variants = config.get_variants_for_segment("women");
        assert!(women_variants.iter().any(|v| v.id == "shakti_gold"));
        assert!(women_variants.iter().any(|v| v.id == "standard"));
    }

    #[test]
    fn test_documentation() {
        let config = ProductConfig::default();
        assert!(!config.documentation.required.is_empty());
        assert!(config.documentation.ekyc_enabled);
    }

    #[test]
    fn test_features() {
        let config = ProductConfig::default();
        assert!(!config.features.key_selling_points.is_empty());
        assert!(!config.features.differentiators.is_empty());
        assert!(config.features.digital_features.mobile_app);
    }
}
