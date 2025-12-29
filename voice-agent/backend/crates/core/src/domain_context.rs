//! Domain context for text processing
//!
//! Provides domain-specific vocabulary and context for grammar correction,
//! translation, and other text processing tasks.

use serde::{Deserialize, Serialize};

/// Domain-specific context for grammar correction and processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DomainContext {
    /// Domain name (e.g., "gold_loan")
    pub domain: String,
    /// Domain-specific vocabulary that should be preserved
    pub vocabulary: Vec<String>,
    /// Common phrases in this domain
    pub phrases: Vec<String>,
    /// Entity types to preserve (names, numbers, etc.)
    pub preserve_entities: Vec<String>,
    /// Abbreviations and their expansions
    #[serde(default)]
    pub abbreviations: Vec<Abbreviation>,
    /// Competitor names to handle specially
    #[serde(default)]
    pub competitors: Vec<String>,
}

/// Abbreviation with expansion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Abbreviation {
    pub short: String,
    pub full: String,
}

impl DomainContext {
    /// Create a new empty domain context
    pub fn new(domain: impl Into<String>) -> Self {
        Self {
            domain: domain.into(),
            ..Default::default()
        }
    }

    /// Create context for gold loan domain
    pub fn gold_loan() -> Self {
        Self {
            domain: "gold_loan".to_string(),
            vocabulary: vec![
                // Product terms
                "gold loan".to_string(),
                "balance transfer".to_string(),
                "top-up".to_string(),
                "foreclosure".to_string(),
                "prepayment".to_string(),
                "disbursement".to_string(),
                "LTV".to_string(),
                "loan-to-value".to_string(),
                // Rates and fees
                "interest rate".to_string(),
                "processing fee".to_string(),
                "per gram".to_string(),
                "per annum".to_string(),
                "ROI".to_string(),
                // Gold terms
                "gold ornaments".to_string(),
                "gold jewellery".to_string(),
                "purity".to_string(),
                "carat".to_string(),
                "hallmark".to_string(),
                // Bank names (preserve exact spelling)
                "Kotak".to_string(),
                "Kotak Mahindra".to_string(),
                "Kotak Mahindra Bank".to_string(),
            ],
            phrases: vec![
                // Hindi phrases (preserve)
                "Kotak Bank se baat kar rahe hain".to_string(),
                "gold loan balance transfer".to_string(),
                "kam interest rate".to_string(),
                "jaldi disbursement".to_string(),
                "aapka gold safe rahega".to_string(),
                // Common patterns
                "aapka naam kya hai".to_string(),
                "kitna loan chahiye".to_string(),
                "konsi branch".to_string(),
            ],
            preserve_entities: vec![
                "PersonName".to_string(),
                "PhoneNumber".to_string(),
                "LoanAmount".to_string(),
                "InterestRate".to_string(),
                "BankName".to_string(),
            ],
            abbreviations: vec![
                Abbreviation {
                    short: "LTV".to_string(),
                    full: "Loan to Value".to_string(),
                },
                Abbreviation {
                    short: "ROI".to_string(),
                    full: "Rate of Interest".to_string(),
                },
                Abbreviation {
                    short: "EMI".to_string(),
                    full: "Equated Monthly Installment".to_string(),
                },
                Abbreviation {
                    short: "KYC".to_string(),
                    full: "Know Your Customer".to_string(),
                },
                Abbreviation {
                    short: "PAN".to_string(),
                    full: "Permanent Account Number".to_string(),
                },
            ],
            competitors: vec![
                "Muthoot".to_string(),
                "Muthoot Finance".to_string(),
                "Manappuram".to_string(),
                "IIFL".to_string(),
                "HDFC".to_string(),
                "SBI".to_string(),
                "ICICI".to_string(),
                "Axis".to_string(),
                "Federal Bank".to_string(),
            ],
        }
    }

    /// Add vocabulary term
    pub fn add_vocabulary(mut self, term: impl Into<String>) -> Self {
        self.vocabulary.push(term.into());
        self
    }

    /// Add phrase
    pub fn add_phrase(mut self, phrase: impl Into<String>) -> Self {
        self.phrases.push(phrase.into());
        self
    }

    /// Add competitor
    pub fn add_competitor(mut self, competitor: impl Into<String>) -> Self {
        self.competitors.push(competitor.into());
        self
    }

    /// Check if a word is domain vocabulary
    pub fn is_vocabulary(&self, word: &str) -> bool {
        let word_lower = word.to_lowercase();
        self.vocabulary.iter().any(|v| v.to_lowercase() == word_lower)
    }

    /// Check if text contains a competitor name
    pub fn contains_competitor(&self, text: &str) -> Option<&str> {
        let text_lower = text.to_lowercase();
        self.competitors
            .iter()
            .find(|c| text_lower.contains(&c.to_lowercase()))
            .map(|s| s.as_str())
    }

    /// Get expansion for abbreviation
    pub fn expand_abbreviation(&self, abbrev: &str) -> Option<&str> {
        self.abbreviations
            .iter()
            .find(|a| a.short.eq_ignore_ascii_case(abbrev))
            .map(|a| a.full.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gold_loan_context() {
        let ctx = DomainContext::gold_loan();
        assert_eq!(ctx.domain, "gold_loan");
        assert!(ctx.is_vocabulary("Kotak"));
        assert!(ctx.is_vocabulary("LTV"));
    }

    #[test]
    fn test_competitor_detection() {
        let ctx = DomainContext::gold_loan();
        // Returns the first matching competitor (Muthoot matches before Muthoot Finance)
        let competitor = ctx.contains_competitor("I have loan from Muthoot Finance");
        assert!(competitor.is_some());
        assert!(competitor.unwrap().contains("Muthoot"));
        assert!(ctx.contains_competitor("I have no existing loan").is_none());
    }

    #[test]
    fn test_abbreviation_expansion() {
        let ctx = DomainContext::gold_loan();
        assert_eq!(ctx.expand_abbreviation("LTV"), Some("Loan to Value"));
        assert_eq!(ctx.expand_abbreviation("ltv"), Some("Loan to Value"));
        assert!(ctx.expand_abbreviation("XYZ").is_none());
    }

    #[test]
    fn test_builder_pattern() {
        let ctx = DomainContext::new("insurance")
            .add_vocabulary("premium")
            .add_vocabulary("coverage")
            .add_competitor("LIC");

        assert_eq!(ctx.domain, "insurance");
        assert!(ctx.is_vocabulary("premium"));
        assert!(ctx.contains_competitor("LIC policy").is_some());
    }
}
