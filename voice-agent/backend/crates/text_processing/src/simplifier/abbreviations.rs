//! Abbreviation Expander for TTS
//!
//! Expands common abbreviations for clear pronunciation in TTS.
//! Handles domain-specific (banking) and general abbreviations.

use std::collections::HashMap;
use once_cell::sync::Lazy;
use regex::Regex;

/// Common abbreviations with their TTS expansions
static ABBREVIATIONS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();

    // Banking / Financial
    map.insert("EMI", "E M I");
    map.insert("emi", "E M I");
    map.insert("ROI", "R O I");
    map.insert("roi", "R O I");
    map.insert("KYC", "K Y C");
    map.insert("kyc", "K Y C");
    map.insert("PAN", "P A N");
    map.insert("pan", "P A N");  // Note: context-sensitive, "pan" could be utensil
    map.insert("IFSC", "I F S C");
    map.insert("ifsc", "I F S C");
    map.insert("NEFT", "N E F T");
    map.insert("neft", "N E F T");
    map.insert("RTGS", "R T G S");
    map.insert("rtgs", "R T G S");
    map.insert("IMPS", "I M P S");
    map.insert("imps", "I M P S");
    map.insert("UPI", "U P I");
    map.insert("upi", "U P I");
    map.insert("ATM", "A T M");
    map.insert("atm", "A T M");
    map.insert("PIN", "P I N");
    map.insert("pin", "P I N");
    map.insert("OTP", "O T P");
    map.insert("otp", "O T P");
    map.insert("GST", "G S T");
    map.insert("gst", "G S T");
    map.insert("TDS", "T D S");
    map.insert("tds", "T D S");
    map.insert("NRI", "N R I");
    map.insert("nri", "N R I");
    map.insert("NRE", "N R E");
    map.insert("nre", "N R E");
    map.insert("NRO", "N R O");
    map.insert("nro", "N R O");
    map.insert("FD", "F D");
    map.insert("fd", "F D");
    map.insert("RD", "R D");
    map.insert("rd", "R D");
    map.insert("SIP", "S I P");
    map.insert("sip", "S I P");
    map.insert("MF", "M F");
    map.insert("mf", "M F");
    map.insert("CIBIL", "CIBIL");  // Pronounceable as word
    map.insert("cibil", "CIBIL");
    map.insert("RBI", "R B I");
    map.insert("rbi", "R B I");
    map.insert("SEBI", "SEBI");  // Pronounceable
    map.insert("sebi", "SEBI");
    map.insert("NSE", "N S E");
    map.insert("nse", "N S E");
    map.insert("BSE", "B S E");
    map.insert("bse", "B S E");
    map.insert("LTV", "L T V");
    map.insert("ltv", "L T V");
    map.insert("APR", "A P R");
    map.insert("apr", "A P R");

    // Gold Loan Specific
    map.insert("BIS", "B I S");
    map.insert("bis", "B I S");
    map.insert("HUID", "H U I D");
    map.insert("huid", "H U I D");
    map.insert("kt", "karat");
    map.insert("KT", "karat");

    // General
    map.insert("ID", "I D");
    map.insert("id", "I D");
    map.insert("PDF", "P D F");
    map.insert("pdf", "P D F");
    map.insert("SMS", "S M S");
    map.insert("sms", "S M S");
    map.insert("URL", "U R L");
    map.insert("url", "U R L");
    map.insert("FAQ", "F A Q");
    map.insert("faq", "F A Q");
    map.insert("CEO", "C E O");
    map.insert("ceo", "C E O");
    map.insert("CFO", "C F O");
    map.insert("cfo", "C F O");
    map.insert("MD", "M D");
    map.insert("md", "M D");
    map.insert("HR", "H R");
    map.insert("hr", "H R");
    map.insert("PR", "P R");
    map.insert("pr", "P R");
    map.insert("IT", "I T");
    map.insert("AI", "A I");
    map.insert("ML", "M L");
    map.insert("API", "A P I");
    map.insert("api", "A P I");

    // Time / Dates
    map.insert("AM", "A M");
    map.insert("am", "A M");
    map.insert("PM", "P M");
    map.insert("pm", "P M");
    map.insert("IST", "I S T");
    map.insert("ist", "I S T");

    // Units
    map.insert("kg", "kilogram");
    map.insert("KG", "kilogram");
    map.insert("gm", "gram");
    map.insert("GM", "gram");
    map.insert("mg", "milligram");
    map.insert("MG", "milligram");
    map.insert("km", "kilometer");
    map.insert("KM", "kilometer");
    map.insert("cm", "centimeter");
    map.insert("CM", "centimeter");
    map.insert("mm", "millimeter");
    map.insert("MM", "millimeter");
    map.insert("L", "liter");
    map.insert("ml", "milliliter");
    map.insert("ML", "milliliter");

    // Indian Specific
    map.insert("Rs", "rupees");
    map.insert("Rs.", "rupees");
    map.insert("INR", "rupees");
    map.insert("inr", "rupees");
    map.insert("Cr", "crore");
    map.insert("cr", "crore");
    map.insert("L", "lakh");  // Context: after number
    map.insert("lac", "lakh");
    map.insert("lacs", "lakhs");

    map
});

/// Pattern for detecting abbreviation candidates
static ABBREV_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // Match uppercase sequences (2+ letters) or known abbreviations
    Regex::new(r"\b([A-Z]{2,}|[a-z]{2,4})\b").unwrap()
});

/// Abbreviation expander for TTS
pub struct AbbreviationExpander {
    /// Additional custom abbreviations
    custom: HashMap<String, String>,
}

impl AbbreviationExpander {
    /// Create new expander
    pub fn new() -> Self {
        Self {
            custom: HashMap::new(),
        }
    }

    /// Add custom abbreviation
    pub fn add_abbreviation(&mut self, abbrev: &str, expansion: &str) {
        self.custom.insert(abbrev.to_string(), expansion.to_string());
    }

    /// Expand abbreviations in text
    pub fn expand(&self, text: &str) -> String {
        let mut result = text.to_string();

        // First check custom abbreviations
        for (abbrev, expansion) in &self.custom {
            let pattern = format!(r"\b{}\b", regex::escape(abbrev));
            if let Ok(re) = Regex::new(&pattern) {
                result = re.replace_all(&result, expansion.as_str()).to_string();
            }
        }

        // Then check standard abbreviations
        for (abbrev, expansion) in ABBREVIATIONS.iter() {
            let pattern = format!(r"\b{}\b", regex::escape(abbrev));
            if let Ok(re) = Regex::new(&pattern) {
                result = re.replace_all(&result, *expansion).to_string();
            }
        }

        // Expand any remaining uppercase sequences (likely acronyms)
        result = self.expand_unknown_acronyms(&result);

        result
    }

    /// Expand unknown uppercase acronyms by spelling out letters
    fn expand_unknown_acronyms(&self, text: &str) -> String {
        let pattern = Regex::new(r"\b([A-Z]{2,6})\b").unwrap();

        pattern.replace_all(text, |caps: &regex::Captures| {
            let acronym = caps.get(1).unwrap().as_str();

            // Skip if already in our abbreviation list
            if ABBREVIATIONS.contains_key(acronym) {
                return acronym.to_string();
            }

            // Check if it's pronounceable (has vowels in right places)
            if self.is_pronounceable(acronym) {
                return acronym.to_string();
            }

            // Spell it out
            acronym.chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }).to_string()
    }

    /// Check if an acronym is pronounceable as a word
    fn is_pronounceable(&self, word: &str) -> bool {
        if word.len() < 2 || word.len() > 6 {
            return false;
        }

        let word_lower = word.to_lowercase();
        let has_vowel = word_lower.chars().any(|c| "aeiou".contains(c));

        // Must have at least one vowel
        if !has_vowel {
            return false;
        }

        // Check for consonant clusters that are hard to pronounce
        let bad_clusters = ["bcd", "cfg", "dgf", "fgh", "ghj", "hjk", "jkl", "klm"];
        for cluster in bad_clusters {
            if word_lower.contains(cluster) {
                return false;
            }
        }

        // Common pronounceable patterns
        let pronounceable_endings = ["an", "en", "in", "on", "un", "ar", "er", "ir", "or", "ur"];
        for ending in pronounceable_endings {
            if word_lower.ends_with(ending) {
                return true;
            }
        }

        // If has vowel and reasonable length, assume pronounceable
        word.len() >= 3 && word.len() <= 5 && has_vowel
    }
}

impl Default for AbbreviationExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banking_abbreviations() {
        let expander = AbbreviationExpander::new();
        assert_eq!(expander.expand("Your EMI is due"), "Your E M I is due");
        assert_eq!(expander.expand("Complete KYC first"), "Complete K Y C first");
        assert_eq!(expander.expand("Enter OTP"), "Enter O T P");
    }

    #[test]
    fn test_units() {
        let expander = AbbreviationExpander::new();
        assert!(expander.expand("100 gm gold").contains("gram"));
        assert!(expander.expand("50 kg weight").contains("kilogram"));
    }

    #[test]
    fn test_custom_abbreviation() {
        let mut expander = AbbreviationExpander::new();
        expander.add_abbreviation("KMBL", "Kotak Mahindra Bank Limited");
        assert!(expander.expand("Welcome to KMBL").contains("Kotak Mahindra Bank Limited"));
    }

    #[test]
    fn test_pronounceable_detection() {
        let expander = AbbreviationExpander::new();
        assert!(expander.is_pronounceable("NASA"));
        assert!(expander.is_pronounceable("SEBI"));
        assert!(!expander.is_pronounceable("NEFT")); // No vowel pattern
        assert!(!expander.is_pronounceable("XYZ")); // No vowel
    }
}
