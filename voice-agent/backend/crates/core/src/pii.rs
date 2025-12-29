//! PII (Personally Identifiable Information) detection types
//!
//! Includes India-specific PII types like Aadhaar, PAN, etc.

use serde::{Deserialize, Serialize};

/// PII types specific to India
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PIIType {
    // Standard PII
    /// Person's name
    PersonName,
    /// Phone number (mobile or landline)
    PhoneNumber,
    /// Email address
    Email,
    /// Physical address
    Address,
    /// Date of birth
    DateOfBirth,

    // India-specific identifiers
    /// Aadhaar number (12 digits)
    Aadhaar,
    /// PAN (Permanent Account Number) - 5 letters, 4 digits, 1 letter
    PAN,
    /// Voter ID (EPIC)
    VoterId,
    /// Driving License number
    DrivingLicense,
    /// Passport number
    Passport,

    // Financial
    /// Bank account number
    BankAccount,
    /// IFSC code
    IFSC,
    /// Credit/Debit card number
    CardNumber,
    /// UPI ID
    UpiId,
    /// Loan amount mentioned
    LoanAmount,
    /// Interest rate mentioned
    InterestRate,

    // Business
    /// Competitor bank/NBFC name
    CompetitorName,
    /// GSTIN (GST number)
    GSTIN,
}

impl PIIType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::PersonName => "Person Name",
            Self::PhoneNumber => "Phone Number",
            Self::Email => "Email",
            Self::Address => "Address",
            Self::DateOfBirth => "Date of Birth",
            Self::Aadhaar => "Aadhaar Number",
            Self::PAN => "PAN",
            Self::VoterId => "Voter ID",
            Self::DrivingLicense => "Driving License",
            Self::Passport => "Passport",
            Self::BankAccount => "Bank Account",
            Self::IFSC => "IFSC Code",
            Self::CardNumber => "Card Number",
            Self::UpiId => "UPI ID",
            Self::LoanAmount => "Loan Amount",
            Self::InterestRate => "Interest Rate",
            Self::CompetitorName => "Competitor Name",
            Self::GSTIN => "GSTIN",
        }
    }

    /// Check if this PII type should always be redacted
    pub fn always_redact(&self) -> bool {
        matches!(
            self,
            Self::Aadhaar
                | Self::PAN
                | Self::BankAccount
                | Self::CardNumber
                | Self::Passport
                | Self::DrivingLicense
        )
    }

    /// Get severity level for this PII type
    pub fn severity(&self) -> PIISeverity {
        match self {
            Self::Aadhaar | Self::PAN | Self::BankAccount | Self::CardNumber | Self::Passport => {
                PIISeverity::Critical
            }
            Self::PhoneNumber | Self::Email | Self::Address | Self::DrivingLicense | Self::UpiId => {
                PIISeverity::High
            }
            Self::PersonName | Self::DateOfBirth | Self::VoterId | Self::IFSC | Self::GSTIN => {
                PIISeverity::Medium
            }
            Self::LoanAmount | Self::InterestRate | Self::CompetitorName => PIISeverity::Low,
        }
    }
}

/// PII severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PIISeverity {
    /// Must be protected - identity theft risk
    Critical,
    /// Should be protected - privacy risk
    High,
    /// Consider protecting - moderate risk
    Medium,
    /// Optional protection - low risk
    Low,
}

/// Detected PII entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIIEntity {
    /// Type of PII
    pub pii_type: PIIType,
    /// The actual text
    pub text: String,
    /// Start position in original text (byte offset)
    pub start: usize,
    /// End position in original text (byte offset)
    pub end: usize,
    /// Detection confidence (0.0 - 1.0)
    pub confidence: f32,
    /// Detection method used
    #[serde(default)]
    pub method: DetectionMethod,
}

impl PIIEntity {
    /// Create a new PII entity
    pub fn new(pii_type: PIIType, text: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            pii_type,
            text: text.into(),
            start,
            end,
            confidence: 1.0,
            method: DetectionMethod::Regex,
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set detection method
    pub fn with_method(mut self, method: DetectionMethod) -> Self {
        self.method = method;
        self
    }
}

/// How the PII was detected
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DetectionMethod {
    /// Pattern matching with regex
    #[default]
    Regex,
    /// Named Entity Recognition model
    NER,
    /// Dictionary/keyword lookup
    Dictionary,
    /// Hybrid (multiple methods)
    Hybrid,
}

/// Redaction strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedactionStrategy {
    /// Replace with [REDACTED]
    Mask,
    /// Replace with type: [PHONE], [AADHAAR], etc.
    TypeMask,
    /// Replace with asterisks: 98****1234
    PartialMask {
        /// Number of characters to keep visible at start
        visible_start: usize,
        /// Number of characters to keep visible at end
        visible_end: usize,
    },
    /// Remove entirely
    Remove,
    /// Replace with fake but valid-looking data
    Synthesize,
    /// Hash the value (for logging/analytics)
    Hash,
}

impl Default for RedactionStrategy {
    fn default() -> Self {
        Self::PartialMask {
            visible_start: 2,
            visible_end: 2,
        }
    }
}

impl RedactionStrategy {
    /// Apply redaction to text
    pub fn apply(&self, text: &str, pii_type: PIIType) -> String {
        match self {
            Self::Mask => "[REDACTED]".to_string(),
            Self::TypeMask => format!("[{}]", pii_type.name().to_uppercase().replace(' ', "_")),
            Self::PartialMask {
                visible_start,
                visible_end,
            } => {
                let chars: Vec<char> = text.chars().collect();
                let len = chars.len();
                if len <= visible_start + visible_end {
                    return "*".repeat(len);
                }
                let start: String = chars[..*visible_start].iter().collect();
                let end: String = chars[len - visible_end..].iter().collect();
                let middle = "*".repeat(len - visible_start - visible_end);
                format!("{}{}{}", start, middle, end)
            }
            Self::Remove => String::new(),
            Self::Synthesize => format!("[SYNTHETIC_{}]", pii_type.name().to_uppercase()),
            Self::Hash => {
                // Simple hash for demo - use proper crypto in production
                format!("[HASH:{}]", &text.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_type_severity() {
        assert_eq!(PIIType::Aadhaar.severity(), PIISeverity::Critical);
        assert_eq!(PIIType::PhoneNumber.severity(), PIISeverity::High);
        assert_eq!(PIIType::PersonName.severity(), PIISeverity::Medium);
        assert_eq!(PIIType::LoanAmount.severity(), PIISeverity::Low);
    }

    #[test]
    fn test_partial_mask() {
        let strategy = RedactionStrategy::PartialMask {
            visible_start: 2,
            visible_end: 2,
        };
        assert_eq!(
            strategy.apply("1234567890", PIIType::PhoneNumber),
            "12******90"
        );
    }

    #[test]
    fn test_type_mask() {
        let strategy = RedactionStrategy::TypeMask;
        assert_eq!(
            strategy.apply("1234567890", PIIType::PhoneNumber),
            "[PHONE_NUMBER]"
        );
    }

    #[test]
    fn test_pii_entity_builder() {
        let entity = PIIEntity::new(PIIType::Aadhaar, "123456789012", 0, 12)
            .with_confidence(0.95)
            .with_method(DetectionMethod::Regex);

        assert_eq!(entity.pii_type, PIIType::Aadhaar);
        assert_eq!(entity.confidence, 0.95);
        assert_eq!(entity.method, DetectionMethod::Regex);
    }
}
