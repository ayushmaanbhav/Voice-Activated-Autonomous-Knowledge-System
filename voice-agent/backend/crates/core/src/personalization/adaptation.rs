//! Response adaptation based on customer segment
//!
//! Adapts responses to match customer expectations:
//! - Feature emphasis based on segment priorities
//! - Objection handling strategies
//! - Value proposition customization
//! - Competitive positioning

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::CustomerSegment;

/// Feature to emphasize in responses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Feature {
    /// Low interest rates
    LowRates,
    /// Quick processing/approval
    QuickProcess,
    /// Security and safety
    Security,
    /// Transparency
    Transparency,
    /// Flexibility
    Flexibility,
    /// Digital/online services
    Digital,
    /// Personal relationship manager
    RelationshipManager,
    /// Higher loan limits
    HigherLimits,
    /// No hidden charges
    NoHiddenCharges,
    /// RBI regulated bank
    RbiRegulated,
    /// Zero foreclosure charges
    ZeroForeclosure,
    /// Doorstep service
    DoorstepService,
    /// Women-specific benefits
    WomenBenefits,
}

impl Feature {
    /// Get display name
    pub fn display_name(&self) -> &'static str {
        match self {
            Feature::LowRates => "Competitive Interest Rates",
            Feature::QuickProcess => "Quick 30-Minute Approval",
            Feature::Security => "Bank-Grade Security",
            Feature::Transparency => "Transparent Pricing",
            Feature::Flexibility => "Flexible Repayment",
            Feature::Digital => "Digital Services",
            Feature::RelationshipManager => "Dedicated Relationship Manager",
            Feature::HigherLimits => "Higher Loan Limits",
            Feature::NoHiddenCharges => "No Hidden Charges",
            Feature::RbiRegulated => "RBI Regulated Bank",
            Feature::ZeroForeclosure => "Zero Foreclosure Charges",
            Feature::DoorstepService => "Doorstep Service",
            Feature::WomenBenefits => "Shakti Gold Benefits",
        }
    }

    /// Get Hindi equivalent
    pub fn hindi_name(&self) -> &'static str {
        match self {
            Feature::LowRates => "Kam Byaj Dar",
            Feature::QuickProcess => "Tez Processing",
            Feature::Security => "Surakshit",
            Feature::Transparency => "Poori Jankari",
            Feature::Flexibility => "Flexible Bhugtan",
            Feature::Digital => "Digital Suvidha",
            Feature::RelationshipManager => "Personal Manager",
            Feature::HigherLimits => "Zyada Loan",
            Feature::NoHiddenCharges => "Koi Chhupe Charges Nahi",
            Feature::RbiRegulated => "RBI Registered Bank",
            Feature::ZeroForeclosure => "Free Foreclosure",
            Feature::DoorstepService => "Ghar Par Seva",
            Feature::WomenBenefits => "Shakti Gold",
        }
    }
}

/// Common customer objection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Objection {
    /// Worried about gold safety
    GoldSafety,
    /// Current lender offers good rates
    BetterRatesElsewhere,
    /// Too much paperwork expected
    TooMuchPaperwork,
    /// Doesn't want to switch
    DontWantToSwitch,
    /// Needs to think about it
    NeedsTime,
    /// Trust issues with banks
    TrustIssues,
    /// Hidden charges expected
    ExpectsHiddenCharges,
    /// Process takes too long
    TooSlow,
    /// Branch not nearby
    NoNearbyBranch,
    /// Already has other loans
    ExistingLoans,
}

impl Objection {
    /// Detect objection from text
    pub fn detect(text: &str) -> Option<Self> {
        let lower = text.to_lowercase();

        if lower.contains("safe") && lower.contains("gold")
            || lower.contains("sona")
            || lower.contains("security")
            || lower.contains("suraksha")
        {
            return Some(Objection::GoldSafety);
        }

        if lower.contains("better rate")
            || lower.contains("kam rate")
            || lower.contains("muthoot")
            || lower.contains("manappuram")
        {
            return Some(Objection::BetterRatesElsewhere);
        }

        if lower.contains("paperwork")
            || lower.contains("documents")
            || lower.contains("kagaz")
            || lower.contains("dastavez")
        {
            return Some(Objection::TooMuchPaperwork);
        }

        if lower.contains("switch")
            || lower.contains("change")
            || lower.contains("badalna")
        {
            return Some(Objection::DontWantToSwitch);
        }

        if lower.contains("think")
            || lower.contains("sochna")
            || lower.contains("later")
            || lower.contains("baad mein")
        {
            return Some(Objection::NeedsTime);
        }

        if lower.contains("trust")
            || lower.contains("bharosa")
            || lower.contains("fraud")
        {
            return Some(Objection::TrustIssues);
        }

        if lower.contains("hidden")
            || lower.contains("chhupe")
            || lower.contains("extra charge")
        {
            return Some(Objection::ExpectsHiddenCharges);
        }

        if lower.contains("slow")
            || lower.contains("time lag")
            || lower.contains("kitna din")
        {
            return Some(Objection::TooSlow);
        }

        if lower.contains("branch")
            || lower.contains("door")
            || lower.contains("far")
            || lower.contains("paas mein")
        {
            return Some(Objection::NoNearbyBranch);
        }

        if lower.contains("other loan")
            || lower.contains("already")
            || lower.contains("pehle se")
        {
            return Some(Objection::ExistingLoans);
        }

        None
    }
}

/// Objection handling response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectionResponse {
    /// Acknowledgment phrase
    pub acknowledgment: String,
    /// Main response
    pub response: String,
    /// Follow-up question (optional)
    pub follow_up: Option<String>,
    /// Feature to highlight
    pub highlight_feature: Feature,
}

/// Segment adapter for response customization
pub struct SegmentAdapter {
    /// Priority features per segment
    segment_features: HashMap<CustomerSegment, Vec<Feature>>,
    /// Objection responses per segment
    objection_responses: HashMap<(CustomerSegment, Objection), ObjectionResponse>,
    /// Value propositions per segment
    value_propositions: HashMap<CustomerSegment, Vec<String>>,
}

impl SegmentAdapter {
    /// Create a new segment adapter with default configurations
    pub fn new() -> Self {
        let mut adapter = Self {
            segment_features: HashMap::new(),
            objection_responses: HashMap::new(),
            value_propositions: HashMap::new(),
        };
        adapter.load_defaults();
        adapter
    }

    /// Load default configurations
    fn load_defaults(&mut self) {
        // High Value segment
        self.segment_features.insert(
            CustomerSegment::HighValue,
            vec![
                Feature::RelationshipManager,
                Feature::HigherLimits,
                Feature::QuickProcess,
                Feature::Flexibility,
            ],
        );

        // Trust Seeker segment
        self.segment_features.insert(
            CustomerSegment::TrustSeeker,
            vec![
                Feature::RbiRegulated,
                Feature::Security,
                Feature::Transparency,
                Feature::NoHiddenCharges,
            ],
        );

        // First Time segment
        self.segment_features.insert(
            CustomerSegment::FirstTime,
            vec![
                Feature::NoHiddenCharges,
                Feature::QuickProcess,
                Feature::Transparency,
                Feature::Flexibility,
            ],
        );

        // Price Sensitive segment
        self.segment_features.insert(
            CustomerSegment::PriceSensitive,
            vec![
                Feature::LowRates,
                Feature::ZeroForeclosure,
                Feature::NoHiddenCharges,
                Feature::Transparency,
            ],
        );

        // Women segment
        self.segment_features.insert(
            CustomerSegment::Women,
            vec![
                Feature::WomenBenefits,
                Feature::Security,
                Feature::Flexibility,
                Feature::DoorstepService,
            ],
        );

        // Professional segment
        self.segment_features.insert(
            CustomerSegment::Professional,
            vec![
                Feature::QuickProcess,
                Feature::Digital,
                Feature::Flexibility,
                Feature::NoHiddenCharges,
            ],
        );

        // Value propositions
        self.value_propositions.insert(
            CustomerSegment::HighValue,
            vec![
                "Exclusive rates and priority processing for valued customers".to_string(),
                "Dedicated relationship manager for personalized service".to_string(),
                "Higher loan limits to meet your requirements".to_string(),
            ],
        );

        self.value_propositions.insert(
            CustomerSegment::TrustSeeker,
            vec![
                "Kotak is an RBI-regulated scheduled bank with highest safety standards".to_string(),
                "Your gold is stored in bank-grade security vaults with full insurance".to_string(),
                "Digital tracking lets you monitor your gold status anytime".to_string(),
            ],
        );

        self.value_propositions.insert(
            CustomerSegment::FirstTime,
            vec![
                "Simple process with just 2 documents - ID and address proof".to_string(),
                "No hidden charges - what we quote is what you pay".to_string(),
                "Loan approved in just 30 minutes at the branch".to_string(),
            ],
        );

        self.value_propositions.insert(
            CustomerSegment::PriceSensitive,
            vec![
                "Starting at 9.5% - among the lowest rates in the market".to_string(),
                "Zero foreclosure charges - prepay anytime without penalty".to_string(),
                "Use our calculator to see exactly how much you save".to_string(),
            ],
        );

        self.value_propositions.insert(
            CustomerSegment::Women,
            vec![
                "Shakti Gold program with 0.25% lower interest for women".to_string(),
                "Women-priority branches with female staff".to_string(),
                "Flexible EMI options to suit your schedule".to_string(),
            ],
        );

        self.value_propositions.insert(
            CustomerSegment::Professional,
            vec![
                "Complete the process in 30 minutes during lunch break".to_string(),
                "Track everything on our mobile app".to_string(),
                "Instant approval with minimal documentation".to_string(),
            ],
        );

        // Load objection responses
        self.load_objection_responses();
    }

    /// Load objection handling responses
    fn load_objection_responses(&mut self) {
        // Trust Seeker - Gold Safety
        self.objection_responses.insert(
            (CustomerSegment::TrustSeeker, Objection::GoldSafety),
            ObjectionResponse {
                acknowledgment: "I completely understand your concern about gold safety - it's your valuable asset.".to_string(),
                response: "At Kotak, your gold is stored in RBI-regulated bank vaults with 24/7 security and full insurance coverage. Unlike NBFCs, we're a scheduled bank with the highest safety standards.".to_string(),
                follow_up: Some("Would you like to know about our digital tracking system where you can check your gold status anytime?".to_string()),
                highlight_feature: Feature::Security,
            },
        );

        // Price Sensitive - Better Rates Elsewhere
        self.objection_responses.insert(
            (CustomerSegment::PriceSensitive, Objection::BetterRatesElsewhere),
            ObjectionResponse {
                acknowledgment: "Getting the best rate is definitely important.".to_string(),
                response: "Let me share the complete picture - while others may advertise lower rates, they often have processing fees, foreclosure charges, and valuation cuts. Our all-in cost at 9.5% with zero foreclosure is often lower in total.".to_string(),
                follow_up: Some("Can I show you a quick calculation comparing your current loan with Kotak?".to_string()),
                highlight_feature: Feature::ZeroForeclosure,
            },
        );

        // First Time - Too Much Paperwork
        self.objection_responses.insert(
            (CustomerSegment::FirstTime, Objection::TooMuchPaperwork),
            ObjectionResponse {
                acknowledgment: "I understand - paperwork can feel overwhelming.".to_string(),
                response: "Good news - we only need 2 simple documents: your ID proof like Aadhaar and one address proof. That's it! Our team handles everything else.".to_string(),
                follow_up: Some("Do you have your Aadhaar card with you today?".to_string()),
                highlight_feature: Feature::QuickProcess,
            },
        );

        // Trust Seeker - Hidden Charges
        self.objection_responses.insert(
            (CustomerSegment::TrustSeeker, Objection::ExpectsHiddenCharges),
            ObjectionResponse {
                acknowledgment: "You're right to ask - many customers have faced unexpected charges elsewhere.".to_string(),
                response: "At Kotak, we believe in complete transparency. I'll give you a printed breakdown of all charges before you sign anything. Our processing fee is flat 1%, and there are absolutely no hidden costs.".to_string(),
                follow_up: Some("Would you like me to prepare a detailed cost sheet for your gold weight?".to_string()),
                highlight_feature: Feature::Transparency,
            },
        );

        // Generic objection responses for all segments
        for segment in [
            CustomerSegment::HighValue,
            CustomerSegment::TrustSeeker,
            CustomerSegment::FirstTime,
            CustomerSegment::PriceSensitive,
            CustomerSegment::Women,
            CustomerSegment::Professional,
        ] {
            // Needs Time
            self.objection_responses.entry((segment, Objection::NeedsTime)).or_insert(
                ObjectionResponse {
                    acknowledgment: "Of course, taking time to think is wise.".to_string(),
                    response: "This is an important decision. Let me share a quick summary of the benefits, and you can call us anytime when you're ready. There's no pressure.".to_string(),
                    follow_up: Some("Can I send you our brochure on WhatsApp for your reference?".to_string()),
                    highlight_feature: Feature::Transparency,
                },
            );

            // No Nearby Branch
            self.objection_responses.entry((segment, Objection::NoNearbyBranch)).or_insert(
                ObjectionResponse {
                    acknowledgment: "Convenience is important - you shouldn't have to travel far.".to_string(),
                    response: "We have over 1,600 branches across India. Let me check the nearest one to your location. Many customers also use our doorstep service.".to_string(),
                    follow_up: Some("What's your area or pincode? I'll find the closest branch.".to_string()),
                    highlight_feature: Feature::DoorstepService,
                },
            );
        }
    }

    /// Get priority features for a segment
    pub fn get_features(&self, segment: CustomerSegment) -> Vec<Feature> {
        self.segment_features
            .get(&segment)
            .cloned()
            .unwrap_or_default()
    }

    /// Get top N features for a segment
    pub fn get_top_features(&self, segment: CustomerSegment, n: usize) -> Vec<Feature> {
        self.get_features(segment).into_iter().take(n).collect()
    }

    /// Get value propositions for a segment
    pub fn get_value_propositions(&self, segment: CustomerSegment) -> Vec<String> {
        self.value_propositions
            .get(&segment)
            .cloned()
            .unwrap_or_default()
    }

    /// Get objection response
    pub fn get_objection_response(
        &self,
        segment: CustomerSegment,
        objection: Objection,
    ) -> Option<&ObjectionResponse> {
        self.objection_responses.get(&(segment, objection))
    }

    /// Handle objection and return formatted response
    pub fn handle_objection(
        &self,
        segment: CustomerSegment,
        objection: Objection,
        customer_name: Option<&str>,
    ) -> Option<String> {
        let response = self.get_objection_response(segment, objection)?;

        let mut result = response.acknowledgment.clone();
        result.push(' ');
        result.push_str(&response.response);

        if let Some(follow_up) = &response.follow_up {
            result.push(' ');
            if let Some(name) = customer_name {
                result.push_str(&format!("{}, {}", name, follow_up));
            } else {
                result.push_str(follow_up);
            }
        }

        Some(result)
    }

    /// Add custom feature priority for a segment
    pub fn add_feature(&mut self, segment: CustomerSegment, feature: Feature) {
        self.segment_features
            .entry(segment)
            .or_default()
            .push(feature);
    }

    /// Add custom value proposition
    pub fn add_value_proposition(&mut self, segment: CustomerSegment, proposition: String) {
        self.value_propositions
            .entry(segment)
            .or_default()
            .push(proposition);
    }

    /// Add custom objection response
    pub fn add_objection_response(
        &mut self,
        segment: CustomerSegment,
        objection: Objection,
        response: ObjectionResponse,
    ) {
        self.objection_responses.insert((segment, objection), response);
    }
}

impl Default for SegmentAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_display() {
        assert_eq!(Feature::LowRates.display_name(), "Competitive Interest Rates");
        assert_eq!(Feature::Security.hindi_name(), "Surakshit");
    }

    #[test]
    fn test_objection_detection() {
        assert_eq!(
            Objection::detect("Is my gold safe with you?"),
            Some(Objection::GoldSafety)
        );
        assert_eq!(
            Objection::detect("Muthoot gives better rates"),
            Some(Objection::BetterRatesElsewhere)
        );
        assert_eq!(
            Objection::detect("Let me think about it"),
            Some(Objection::NeedsTime)
        );
        assert_eq!(Objection::detect("What's your rate?"), None);
    }

    #[test]
    fn test_segment_features() {
        let adapter = SegmentAdapter::new();

        let features = adapter.get_features(CustomerSegment::TrustSeeker);
        assert!(features.contains(&Feature::RbiRegulated));
        assert!(features.contains(&Feature::Security));

        let features = adapter.get_features(CustomerSegment::PriceSensitive);
        assert!(features.contains(&Feature::LowRates));
        assert!(features.contains(&Feature::ZeroForeclosure));
    }

    #[test]
    fn test_top_features() {
        let adapter = SegmentAdapter::new();
        let top = adapter.get_top_features(CustomerSegment::HighValue, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0], Feature::RelationshipManager);
    }

    #[test]
    fn test_value_propositions() {
        let adapter = SegmentAdapter::new();
        let props = adapter.get_value_propositions(CustomerSegment::Women);
        assert!(!props.is_empty());
        assert!(props.iter().any(|p| p.contains("Shakti")));
    }

    #[test]
    fn test_objection_response() {
        let adapter = SegmentAdapter::new();
        let response = adapter
            .get_objection_response(CustomerSegment::TrustSeeker, Objection::GoldSafety)
            .unwrap();

        assert!(response.acknowledgment.contains("understand"));
        assert!(response.response.contains("RBI"));
        assert_eq!(response.highlight_feature, Feature::Security);
    }

    #[test]
    fn test_handle_objection() {
        let adapter = SegmentAdapter::new();
        let response = adapter.handle_objection(
            CustomerSegment::FirstTime,
            Objection::TooMuchPaperwork,
            Some("Raj"),
        );

        assert!(response.is_some());
        let text = response.unwrap();
        assert!(text.contains("2"));
        assert!(text.contains("Raj"));
    }

    #[test]
    fn test_custom_additions() {
        let mut adapter = SegmentAdapter::new();

        adapter.add_feature(CustomerSegment::Professional, Feature::DoorstepService);
        let features = adapter.get_features(CustomerSegment::Professional);
        assert!(features.contains(&Feature::DoorstepService));

        adapter.add_value_proposition(
            CustomerSegment::HighValue,
            "Airport lounge access".to_string(),
        );
        let props = adapter.get_value_propositions(CustomerSegment::HighValue);
        assert!(props.iter().any(|p| p.contains("Airport")));
    }
}
