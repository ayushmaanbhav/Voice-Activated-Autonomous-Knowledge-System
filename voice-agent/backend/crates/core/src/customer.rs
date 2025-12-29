//! Customer profile and segmentation types

use serde::{Deserialize, Serialize};

/// Customer segment for personalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CustomerSegment {
    /// High-value customers (>100g gold, sophisticated)
    HighValue,
    /// Safety-focused, may have been burned by NBFC issues
    TrustSeeker,
    /// New to gold loans
    FirstTime,
    /// Rate-focused comparison shoppers
    PriceSensitive,
    /// Women customers (Shakti Gold segment)
    Women,
    /// Young urban professionals
    Professional,
}

impl CustomerSegment {
    /// Get segment display name
    pub fn display_name(&self) -> &'static str {
        match self {
            CustomerSegment::HighValue => "High Value",
            CustomerSegment::TrustSeeker => "Trust Seeker",
            CustomerSegment::FirstTime => "First Time",
            CustomerSegment::PriceSensitive => "Price Sensitive",
            CustomerSegment::Women => "Women (Shakti)",
            CustomerSegment::Professional => "Young Professional",
        }
    }

    /// Get key messaging points for this segment
    pub fn key_messages(&self) -> Vec<&'static str> {
        match self {
            CustomerSegment::HighValue => vec![
                "Dedicated relationship manager",
                "Priority processing",
                "Higher loan limits",
                "Exclusive rates",
            ],
            CustomerSegment::TrustSeeker => vec![
                "RBI-regulated scheduled bank",
                "Bank-grade security vaults",
                "Full insurance coverage",
                "Digital tracking of gold",
            ],
            CustomerSegment::FirstTime => vec![
                "Simple process",
                "No hidden charges",
                "Friendly support",
                "Clear documentation",
            ],
            CustomerSegment::PriceSensitive => vec![
                "Lowest interest rates",
                "Zero foreclosure charges",
                "Transparent pricing",
                "Savings calculator",
            ],
            CustomerSegment::Women => vec![
                "Special Shakti Gold program",
                "Preferential rates",
                "Women-only branches available",
                "Flexible repayment",
            ],
            CustomerSegment::Professional => vec![
                "Quick digital process",
                "Mobile app tracking",
                "Instant approval",
                "Flexible tenure options",
            ],
        }
    }

    /// Get suggested persona warmth level (0.0 - 1.0)
    pub fn suggested_warmth(&self) -> f32 {
        match self {
            CustomerSegment::HighValue => 0.9,
            CustomerSegment::TrustSeeker => 0.95,
            CustomerSegment::FirstTime => 0.9,
            CustomerSegment::PriceSensitive => 0.7,
            CustomerSegment::Women => 0.95,
            CustomerSegment::Professional => 0.75,
        }
    }
}

impl std::fmt::Display for CustomerSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Customer profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerProfile {
    /// Customer ID (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Customer name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Phone number
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phone: Option<String>,

    /// Customer segment
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<CustomerSegment>,

    /// Current gold loan lender (competitor)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_lender: Option<String>,

    /// Gold weight in grams
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gold_weight: Option<f64>,

    /// Gold purity (e.g., "22K", "24K")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gold_purity: Option<String>,

    /// Current/desired loan amount
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loan_amount: Option<f64>,

    /// Preferred language (ISO 639-1)
    #[serde(default = "default_language")]
    pub preferred_language: String,

    /// Existing relationship with Kotak
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relationship_with_kotak: Option<KotakRelationship>,

    /// City
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,

    /// Pincode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pincode: Option<String>,
}

fn default_language() -> String {
    "en".to_string()
}

impl CustomerProfile {
    /// Create a new empty customer profile
    pub fn new() -> Self {
        Self {
            id: None,
            name: None,
            phone: None,
            segment: None,
            current_lender: None,
            gold_weight: None,
            gold_purity: None,
            loan_amount: None,
            preferred_language: "en".to_string(),
            relationship_with_kotak: None,
            city: None,
            pincode: None,
        }
    }

    /// Create profile with phone number
    pub fn with_phone(phone: impl Into<String>) -> Self {
        let mut profile = Self::new();
        profile.phone = Some(phone.into());
        profile
    }

    /// Set customer name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set segment
    pub fn segment(mut self, segment: CustomerSegment) -> Self {
        self.segment = Some(segment);
        self
    }

    /// Set current lender
    pub fn current_lender(mut self, lender: impl Into<String>) -> Self {
        self.current_lender = Some(lender.into());
        self
    }

    /// Set gold details
    pub fn gold(mut self, weight: f64, purity: impl Into<String>) -> Self {
        self.gold_weight = Some(weight);
        self.gold_purity = Some(purity.into());
        self
    }

    /// Set preferred language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.preferred_language = lang.into();
        self
    }

    /// Check if customer is a switcher (has current lender)
    pub fn is_switcher(&self) -> bool {
        self.current_lender.is_some()
    }

    /// Check if we have gold details
    pub fn has_gold_details(&self) -> bool {
        self.gold_weight.is_some() && self.gold_purity.is_some()
    }

    /// P2 FIX: Default gold price per gram (INR). Should be fetched from API/config.
    pub const DEFAULT_GOLD_PRICE_PER_GRAM: f64 = 7500.0;

    /// Estimate gold value
    ///
    /// P2 FIX: Now accepts configurable gold_price_per_gram parameter.
    /// Use `CustomerProfile::DEFAULT_GOLD_PRICE_PER_GRAM` or pass value from config.
    pub fn estimated_gold_value(&self, gold_price_per_gram: Option<f64>) -> Option<f64> {
        let weight = self.gold_weight?;
        let purity = self.gold_purity.as_ref()?;

        let base_price = gold_price_per_gram.unwrap_or(Self::DEFAULT_GOLD_PRICE_PER_GRAM);

        let purity_factor = match purity.to_uppercase().as_str() {
            "24K" => 1.0,
            "22K" => 0.916,
            "18K" => 0.75,
            "14K" => 0.585,
            _ => 0.75, // Default to 18K
        };

        Some(weight * base_price * purity_factor)
    }

    /// Get display name (name or "Customer")
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("Customer")
    }

    /// Infer segment from profile data
    pub fn infer_segment(&self) -> Option<CustomerSegment> {
        // Already has segment
        if self.segment.is_some() {
            return self.segment;
        }

        // High value: >100g gold
        if let Some(weight) = self.gold_weight {
            if weight >= 100.0 {
                return Some(CustomerSegment::HighValue);
            }
        }

        // Trust seeker: switching from NBFC due to issues
        if let Some(ref lender) = self.current_lender {
            let lender_lower = lender.to_lowercase();
            if lender_lower.contains("iifl")
                || lender_lower.contains("muthoot")
                || lender_lower.contains("manappuram")
            {
                return Some(CustomerSegment::TrustSeeker);
            }
        }

        // First time: no current lender and no gold details
        if self.current_lender.is_none() && !self.has_gold_details() {
            return Some(CustomerSegment::FirstTime);
        }

        None
    }
}

impl Default for CustomerProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Existing relationship with Kotak
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KotakRelationship {
    /// Is existing Kotak customer
    pub is_customer: bool,
    /// Products held
    #[serde(default)]
    pub products: Vec<String>,
    /// Account vintage in months
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vintage_months: Option<u32>,
    /// Is priority/premium customer
    #[serde(default)]
    pub is_priority: bool,
}

impl KotakRelationship {
    pub fn new_customer() -> Self {
        Self {
            is_customer: false,
            products: Vec::new(),
            vintage_months: None,
            is_priority: false,
        }
    }

    pub fn existing(products: Vec<String>) -> Self {
        Self {
            is_customer: true,
            products,
            vintage_months: None,
            is_priority: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_customer_profile() {
        let profile = CustomerProfile::with_phone("9876543210")
            .name("Raj Kumar")
            .current_lender("Muthoot Finance")
            .gold(50.0, "22K")
            .language("hi");

        assert!(profile.is_switcher());
        assert!(profile.has_gold_details());
        assert!(profile.estimated_gold_value(None).is_some());
    }

    #[test]
    fn test_segment_inference() {
        // High value
        let profile = CustomerProfile::new().gold(150.0, "22K");
        assert_eq!(profile.infer_segment(), Some(CustomerSegment::HighValue));

        // Trust seeker (IIFL customer)
        let profile = CustomerProfile::new().current_lender("IIFL Finance");
        assert_eq!(profile.infer_segment(), Some(CustomerSegment::TrustSeeker));

        // First time
        let profile = CustomerProfile::new();
        assert_eq!(profile.infer_segment(), Some(CustomerSegment::FirstTime));
    }

    #[test]
    fn test_segment_messages() {
        let segment = CustomerSegment::TrustSeeker;
        let messages = segment.key_messages();
        assert!(!messages.is_empty());
        assert!(messages.iter().any(|m| m.contains("RBI")));
    }
}
