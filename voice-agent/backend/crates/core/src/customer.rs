//! Customer profile and segmentation types

use serde::{Deserialize, Serialize};

/// Customer segment for personalization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CustomerSegment {
    /// High-value customers (large asset value, sophisticated)
    HighValue,
    /// Safety-focused, may have had issues with other providers
    TrustSeeker,
    /// New to this service type
    FirstTime,
    /// Rate-focused comparison shoppers
    PriceSensitive,
    /// Women customers (special segment programs)
    Women,
    /// Young urban professionals
    Professional,
}

impl CustomerSegment {
    /// Get segment display name (generic - override with config for domain-specific names)
    pub fn display_name(&self) -> &'static str {
        match self {
            CustomerSegment::HighValue => "High Value",
            CustomerSegment::TrustSeeker => "Trust Seeker",
            CustomerSegment::FirstTime => "First Time",
            CustomerSegment::PriceSensitive => "Price Sensitive",
            CustomerSegment::Women => "Women",
            CustomerSegment::Professional => "Young Professional",
        }
    }

    /// Get generic key messaging points for this segment.
    ///
    /// NOTE: For domain-specific messaging, load from config via
    /// `MasterDomainConfig::segment_messages()` instead of using these defaults.
    pub fn key_messages(&self) -> Vec<&'static str> {
        match self {
            CustomerSegment::HighValue => vec![
                "Dedicated relationship manager",
                "Priority processing",
                "Higher limits",
                "Exclusive rates",
            ],
            CustomerSegment::TrustSeeker => vec![
                "Regulated financial institution",
                "Secure storage facilities",
                "Full insurance coverage",
                "Digital tracking",
            ],
            CustomerSegment::FirstTime => vec![
                "Simple process",
                "No hidden charges",
                "Friendly support",
                "Clear documentation",
            ],
            CustomerSegment::PriceSensitive => vec![
                "Competitive rates",
                "Zero foreclosure charges",
                "Transparent pricing",
                "Savings calculator",
            ],
            CustomerSegment::Women => vec![
                "Special programs available",
                "Preferential rates",
                "Dedicated service centers",
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

    /// Current provider (competitor)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_lender: Option<String>,

    /// Collateral weight in units (e.g., grams for gold)
    #[serde(skip_serializing_if = "Option::is_none", alias = "gold_weight")]
    pub collateral_weight: Option<f64>,

    /// Collateral variant/grade (e.g., "22K", "24K" for gold)
    #[serde(skip_serializing_if = "Option::is_none", alias = "gold_purity")]
    pub collateral_variant: Option<String>,

    /// Current/desired loan amount
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loan_amount: Option<f64>,

    /// Preferred language (ISO 639-1)
    #[serde(default = "default_language")]
    pub preferred_language: String,

    /// Existing relationship with company
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relationship_with_company: Option<CompanyRelationship>,

    /// City
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,

    /// Pincode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pincode: Option<String>,
}

// P0 FIX: Legacy gold aliases REMOVED
// The following methods have been removed as they violate domain-agnostic principles:
//   - gold_weight() -> use collateral_weight directly
//   - gold_purity() -> use collateral_variant directly
//
// Code using these aliases should migrate to the generic field names:
//   profile.gold_weight() -> profile.collateral_weight
//   profile.gold_purity() -> profile.collateral_variant.as_deref()



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
            collateral_weight: None,
            collateral_variant: None,
            loan_amount: None,
            preferred_language: "en".to_string(),
            relationship_with_company: None,
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

    /// Set current lender/provider
    pub fn current_lender(mut self, lender: impl Into<String>) -> Self {
        self.current_lender = Some(lender.into());
        self
    }

    /// Set collateral details (weight/quantity and variant/grade)
    pub fn collateral(mut self, weight: f64, variant: impl Into<String>) -> Self {
        self.collateral_weight = Some(weight);
        self.collateral_variant = Some(variant.into());
        self
    }

    // P0 FIX: gold() alias REMOVED
    // Use collateral(weight, variant) instead of gold(weight, purity)

    /// Set preferred language
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.preferred_language = lang.into();
        self
    }

    /// Check if customer is a switcher (has current lender)
    pub fn is_switcher(&self) -> bool {
        self.current_lender.is_some()
    }

    /// Check if we have collateral details
    pub fn has_collateral_details(&self) -> bool {
        self.collateral_weight.is_some() && self.collateral_variant.is_some()
    }

    // P0 FIX: has_gold_details() alias REMOVED
    // Use has_collateral_details() instead

    // P0 FIX: Deprecated constants and methods REMOVED
    //
    // The following have been removed as they contained hardcoded domain-specific values:
    //   - DEFAULT_ASSET_PRICE_PER_UNIT (hardcoded 7500.0)
    //   - DEFAULT_GOLD_PRICE_PER_GRAM (alias)
    //   - default_variant_factor() (hardcoded gold purity factors)
    //
    // Migration guide:
    //   - Load asset price from domain config: master_config.constants.asset_price_per_unit
    //   - Load variant factors from domain config: master_config.constants.variant_factors
    //   - Use estimated_collateral_value_with_config() with config-provided values

    /// Estimate collateral value using config-provided variant factors.
    ///
    /// This is the preferred method - pass variant factors from domain config.
    ///
    /// # Arguments
    /// * `asset_price_per_unit` - Current price per unit from config/API
    /// * `variant_factors` - Map of variant_id -> factor from config
    pub fn estimated_collateral_value_with_config(
        &self,
        asset_price_per_unit: f64,
        variant_factors: &std::collections::HashMap<String, f64>,
    ) -> Option<f64> {
        let weight = self.collateral_weight?;
        let variant = self.collateral_variant.as_ref()?;

        let variant_upper = variant.to_uppercase();
        let variant_factor = variant_factors
            .get(&variant_upper)
            .or_else(|| variant_factors.get(variant))
            .copied()
            .unwrap_or(0.75); // Default to conservative factor

        Some(weight * asset_price_per_unit * variant_factor)
    }

    // P0 FIX: Deprecated estimation methods REMOVED
    //
    // The following have been removed as they used hardcoded fallback values:
    //   - estimated_collateral_value() - used hardcoded price and factors
    //   - estimated_gold_value() - alias for above
    //
    // Use estimated_collateral_value_with_config() with values from domain config:
    //   let price = master_config.constants.asset_price_per_unit;
    //   let factors = master_config.constants.variant_factors;
    //   profile.estimated_collateral_value_with_config(price, &factors)

    /// Get display name (name or "Customer")
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or("Customer")
    }

    /// P16 FIX: Infer segment from profile data with config-driven thresholds
    ///
    /// # Arguments
    /// * `collateral_threshold` - Minimum collateral weight for high value (e.g., 100.0)
    /// * `amount_threshold` - Minimum loan amount for high value (e.g., 500_000.0)
    ///
    /// Use this method with values from segments.yaml:
    /// ```ignore
    /// let thresholds = segments_config.get_high_value_thresholds();
    /// profile.infer_segment_with_thresholds(
    ///     thresholds.collateral_min,
    ///     thresholds.loan_amount_min
    /// )
    /// ```
    pub fn infer_segment_with_thresholds(
        &self,
        collateral_threshold: f64,
        amount_threshold: f64,
    ) -> Option<CustomerSegment> {
        // Already has segment
        if self.segment.is_some() {
            return self.segment;
        }

        // High value: collateral OR loan amount above configured thresholds
        if let Some(weight) = self.collateral_weight {
            if weight >= collateral_threshold {
                return Some(CustomerSegment::HighValue);
            }
        }
        if let Some(amount) = self.loan_amount {
            if amount >= amount_threshold {
                return Some(CustomerSegment::HighValue);
            }
        }

        // Trust seeker: has current lender
        if self.current_lender.is_some() {
            return Some(CustomerSegment::TrustSeeker);
        }

        // First time: no current lender and no collateral details
        if self.current_lender.is_none() && !self.has_collateral_details() {
            return Some(CustomerSegment::FirstTime);
        }

        None
    }

    /// Infer segment from profile data (legacy, uses hardcoded fallback thresholds)
    ///
    /// DEPRECATED: Use `infer_segment_with_thresholds()` for config-driven detection.
    /// These hardcoded values are fallbacks from segments.yaml high_value.detection.numeric_thresholds.
    pub fn infer_segment(&self) -> Option<CustomerSegment> {
        // Legacy fallback thresholds (should match segments.yaml defaults)
        const DEFAULT_COLLATERAL_THRESHOLD: f64 = 100.0;  // grams
        const DEFAULT_AMOUNT_THRESHOLD: f64 = 500_000.0;  // INR (5 lakhs)

        self.infer_segment_with_thresholds(DEFAULT_COLLATERAL_THRESHOLD, DEFAULT_AMOUNT_THRESHOLD)
    }
}

/// P3-2 FIX: Segment detector that analyzes conversation signals
///
/// Uses multiple signals to auto-detect customer segment:
/// - Loan amount mentioned → High Value
/// - Urgency signals → marks as urgent (influences persona)
/// - Price/rate focus → Price Sensitive
/// - Safety concerns → Trust Seeker
/// - Comparison mentions → Price Sensitive
///
/// NOTE: For full config-driven detection, use with patterns loaded from domain config.
#[derive(Debug, Clone)]
pub struct SegmentDetector {
    /// Collateral quantity threshold for high-value detection (domain-specific unit)
    pub high_value_collateral_threshold: f64,
    /// Amount threshold for high-value detection (currency-specific)
    pub high_value_amount_threshold: f64,
    /// Optional custom collateral weight patterns from config
    pub collateral_weight_patterns: Vec<String>,
}

impl Default for SegmentDetector {
    fn default() -> Self {
        Self {
            high_value_collateral_threshold: 100.0,
            high_value_amount_threshold: 500_000.0,
            collateral_weight_patterns: Vec::new(),
        }
    }
}

impl SegmentDetector {
    /// Create a new segment detector with default thresholds
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom thresholds from domain config
    ///
    /// # Arguments
    /// * `collateral_threshold` - Collateral quantity threshold (e.g., 100 for 100g gold)
    /// * `amount_threshold` - Loan amount threshold (e.g., 500000 for 5 lakh INR)
    pub fn with_thresholds(collateral_threshold: f64, amount_threshold: f64) -> Self {
        Self {
            high_value_collateral_threshold: collateral_threshold,
            high_value_amount_threshold: amount_threshold,
            collateral_weight_patterns: Vec::new(),
        }
    }

    /// Create with thresholds and custom collateral patterns from config
    pub fn with_config(
        collateral_threshold: f64,
        amount_threshold: f64,
        collateral_patterns: Vec<String>,
    ) -> Self {
        Self {
            high_value_collateral_threshold: collateral_threshold,
            high_value_amount_threshold: amount_threshold,
            collateral_weight_patterns: collateral_patterns,
        }
    }

    // P0 FIX: with_gold_thresholds() alias REMOVED
    // Use with_thresholds(collateral_threshold, amount_threshold) instead

    /// Detect segment from text content (conversation transcript)
    pub fn detect_from_text(&self, text: &str) -> Option<CustomerSegment> {
        let lower = text.to_lowercase();

        // High Value: Large amounts mentioned
        if self.detect_high_value_amount(&lower) {
            return Some(CustomerSegment::HighValue);
        }

        // Price Sensitive: Rate/price focused
        if self.detect_price_sensitivity(&lower) {
            return Some(CustomerSegment::PriceSensitive);
        }

        // Trust Seeker: Safety/security concerns
        if self.detect_trust_seeking(&lower) {
            return Some(CustomerSegment::TrustSeeker);
        }

        // First Time: New to gold loans
        if self.detect_first_time(&lower) {
            return Some(CustomerSegment::FirstTime);
        }

        None
    }

    /// Detect high-value customer from text
    fn detect_high_value_amount(&self, text: &str) -> bool {
        // Generic currency patterns (lakh/crore are common in South Asian currencies)
        let high_value_patterns = [
            "lakh",
            "lakhs",
            "lac",
            "lacs",
            "crore",
            "crores",
            "500000",
            "1000000",
            "पाँच लाख",
            "दस लाख",
            "करोड़",
        ];

        // Check for large amounts
        for pattern in &high_value_patterns {
            if text.contains(pattern) {
                // Extract number before lakh/crore
                if pattern.contains("lakh") || pattern.contains("lac") {
                    // Check if >= 5 lakhs
                    if self.extract_lakh_amount(text) >= 5.0 {
                        return true;
                    }
                } else if pattern.contains("crore") || pattern.contains("करोड़") {
                    return true; // Any crore amount is high value
                }
            }
        }

        // Check for collateral weight patterns from config
        for pattern in &self.collateral_weight_patterns {
            if text.contains(&pattern.to_lowercase()) {
                return true;
            }
        }

        // Default collateral patterns (can be overridden by config)
        // These are generic weight-based patterns
        if self.collateral_weight_patterns.is_empty() {
            let default_weight_patterns = ["100 gram", "150 gram", "200 gram"];
            for pattern in &default_weight_patterns {
                if text.contains(*pattern) {
                    return true;
                }
            }
        }

        false
    }

    /// Extract lakh amount from text
    fn extract_lakh_amount(&self, text: &str) -> f64 {
        // Simple extraction: look for "N lakh" pattern
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if (word.contains("lakh") || word.contains("lac")) && i > 0 {
                if let Ok(n) = words[i - 1].parse::<f64>() {
                    return n;
                }
                // Hindi number words
                match words[i - 1] {
                    "एक" | "ek" => return 1.0,
                    "दो" | "do" => return 2.0,
                    "तीन" | "teen" => return 3.0,
                    "चार" | "char" => return 4.0,
                    "पाँच" | "paanch" | "panch" => return 5.0,
                    "दस" | "das" => return 10.0,
                    _ => {},
                }
            }
        }
        0.0
    }

    /// Detect price-sensitive customer
    fn detect_price_sensitivity(&self, text: &str) -> bool {
        let price_patterns = [
            // English
            "interest rate",
            "lowest rate",
            "best rate",
            "cheaper",
            "how much interest",
            "what rate",
            "rate kitna",
            "compare rate",
            "other bank",
            "better rate",
            "processing fee",
            "hidden charge",
            "total cost",
            // Hindi
            "kitna byaj",
            "byaj dar",
            "sasta",
            "mehnga",
            "sabse kam",
            "ब्याज दर",
            "सस्ता",
            "महंगा",
        ];

        price_patterns.iter().any(|p| text.contains(p))
    }

    /// Detect trust-seeking customer (config-driven)
    ///
    /// # Arguments
    /// * `text` - The text to analyze
    /// * `competitor_names` - Optional list of competitor names to check
    pub fn detect_trust_seeking_with_config(&self, text: &str, competitor_names: &[String]) -> bool {
        // Check safety patterns
        if self.detect_trust_seeking(text) {
            return true;
        }
        // Check competitor mentions
        competitor_names.iter().any(|c| text.contains(&c.to_lowercase()))
    }

    /// Detect trust-seeking customer (basic patterns only, no competitor matching)
    ///
    /// NOTE: For domain-specific patterns (e.g., "gold safe"), load from config via
    /// `detect_trust_seeking_with_config()` and pass domain-specific patterns.
    fn detect_trust_seeking(&self, text: &str) -> bool {
        let trust_patterns = [
            // Generic safety concerns (domain-agnostic)
            "is it safe",
            "safe hai",
            "surakshit",
            "insurance",
            "vault",
            "locker",
            "rbi",
            "regulated",
            "government",
            "secure",
            "security",
            // Past issues (domain-agnostic)
            "problem",
            "issue",
            "fraud",
            "cheat",
            "lost",
            "missing",
            "stolen",
            // Hindi (generic trust terms)
            "सुरक्षित",
            "भरोसा",
            "विश्वास",
        ];

        trust_patterns.iter().any(|p| text.contains(p))
    }

    /// Detect first-time customer
    fn detect_first_time(&self, text: &str) -> bool {
        let first_time_patterns = [
            "first time",
            "pehli baar",
            "pahli bar",
            "never taken",
            "new to",
            "how does it work",
            "kaise hota hai",
            "process kya hai",
            "what is",      // Generic: "what is [product]"
            "kya hai",      // Generic: "[product] kya hai"
            "पहली बार",
            "कैसे होता है",
        ];

        first_time_patterns.iter().any(|p| text.contains(p))
    }

    /// Detect urgency in text (doesn't map to segment but influences persona)
    pub fn detect_urgency(&self, text: &str) -> bool {
        let urgency_patterns = [
            // English
            "urgent",
            "emergency",
            "asap",
            "today",
            "right now",
            "immediately",
            "quick",
            "fast",
            "need money",
            "hospital",
            "medical",
            // Hindi
            "jaldi",
            "abhi",
            "turant",
            "fauran",
            "जल्दी",
            "अभी",
            "तुरंत",
            "फौरन",
            "paise chahiye",
            "पैसे चाहिए",
        ];

        urgency_patterns.iter().any(|p| text.contains(p))
    }

    /// Combined detection from profile and conversation text
    pub fn detect_combined(
        &self,
        profile: &CustomerProfile,
        conversation_text: Option<&str>,
    ) -> Option<CustomerSegment> {
        // First try profile-based detection
        if let Some(segment) = profile.infer_segment() {
            return Some(segment);
        }

        // Then try text-based detection
        if let Some(text) = conversation_text {
            if let Some(segment) = self.detect_from_text(text) {
                return Some(segment);
            }
        }

        None
    }
}

impl Default for CustomerProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// Existing relationship with company (domain-agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanyRelationship {
    /// Is existing customer
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

impl CompanyRelationship {
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
    use std::collections::HashMap;

    #[test]
    fn test_customer_profile() {
        // P0 FIX: Use generic collateral() method instead of gold()
        let profile = CustomerProfile::with_phone("9876543210")
            .name("Raj Kumar")
            .current_lender("Provider A")
            .collateral(50.0, "STANDARD_GRADE")
            .language("hi");

        assert!(profile.is_switcher());
        assert!(profile.has_collateral_details());

        // P0 FIX: Use config-driven value estimation
        let mut variant_factors = HashMap::new();
        variant_factors.insert("STANDARD_GRADE".to_string(), 0.916);
        let value = profile.estimated_collateral_value_with_config(7500.0, &variant_factors);
        assert!(value.is_some());
    }

    #[test]
    fn test_segment_inference() {
        // High value - use collateral() instead of gold()
        let profile = CustomerProfile::new().collateral(150.0, "HIGH_GRADE");
        assert_eq!(profile.infer_segment(), Some(CustomerSegment::HighValue));

        // Trust seeker (customer with current provider)
        let profile = CustomerProfile::new().current_lender("Provider A");
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

    // P3-2 FIX: Tests for SegmentDetector
    #[test]
    fn test_segment_detector_high_value() {
        let detector = SegmentDetector::new();

        // Amount mention in lakhs
        assert_eq!(
            detector.detect_from_text("I need 10 lakh rupees loan"),
            Some(CustomerSegment::HighValue)
        );

        // Crore mention
        assert_eq!(
            detector.detect_from_text("mujhe 1 crore chahiye"),
            Some(CustomerSegment::HighValue)
        );

        // Large gold weight
        assert_eq!(
            detector.detect_from_text("I have 150 gram gold"),
            Some(CustomerSegment::HighValue)
        );
    }

    #[test]
    fn test_segment_detector_price_sensitive() {
        let detector = SegmentDetector::new();

        assert_eq!(
            detector.detect_from_text("What is your interest rate?"),
            Some(CustomerSegment::PriceSensitive)
        );

        assert_eq!(
            detector.detect_from_text("other bank is offering better rate"),
            Some(CustomerSegment::PriceSensitive)
        );

        // Hindi
        assert_eq!(
            detector.detect_from_text("byaj dar kitna hai"),
            Some(CustomerSegment::PriceSensitive)
        );
    }

    #[test]
    fn test_segment_detector_trust_seeker() {
        let detector = SegmentDetector::new();

        assert_eq!(
            detector.detect_from_text("Is my gold safe with you?"),
            Some(CustomerSegment::TrustSeeker)
        );

        assert_eq!(
            detector.detect_from_text("I had issues with Muthoot before"),
            Some(CustomerSegment::TrustSeeker)
        );

        // RBI mention
        assert_eq!(
            detector.detect_from_text("Are you RBI regulated?"),
            Some(CustomerSegment::TrustSeeker)
        );
    }

    #[test]
    fn test_segment_detector_first_time() {
        let detector = SegmentDetector::new();

        assert_eq!(
            detector.detect_from_text("This is my first time taking a gold loan"),
            Some(CustomerSegment::FirstTime)
        );

        assert_eq!(
            detector.detect_from_text("Gold loan kaise hota hai?"),
            Some(CustomerSegment::FirstTime)
        );
    }

    #[test]
    fn test_segment_detector_urgency() {
        let detector = SegmentDetector::new();

        assert!(detector.detect_urgency("I need money urgently for hospital"));
        assert!(detector.detect_urgency("paise jaldi chahiye"));
        assert!(detector.detect_urgency("emergency hai"));
        assert!(!detector.detect_urgency("I'm just exploring options"));
    }

    #[test]
    fn test_segment_detector_combined() {
        let detector = SegmentDetector::new();

        // P0 FIX: Use collateral() instead of gold()
        // Profile takes precedence
        let profile = CustomerProfile::new().collateral(150.0, "HIGH_GRADE");
        assert_eq!(
            detector.detect_combined(&profile, Some("what is your rate")),
            Some(CustomerSegment::HighValue)
        );

        // Falls back to text when profile doesn't determine segment
        // (profile with collateral details but not enough to be high value)
        let profile = CustomerProfile::new().collateral(20.0, "STANDARD_GRADE");
        assert_eq!(
            detector.detect_combined(&profile, Some("what is your interest rate")),
            Some(CustomerSegment::PriceSensitive)
        );
    }

    #[test]
    fn test_high_value_from_loan_amount() {
        // Test that loan_amount field also triggers high value
        let profile = CustomerProfile {
            loan_amount: Some(600_000.0),
            ..CustomerProfile::new()
        };
        assert_eq!(profile.infer_segment(), Some(CustomerSegment::HighValue));
    }
}
