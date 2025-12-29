//! Cross-lingual query normalization
//!
//! Handles Hindi/Hinglish/English mixed queries:
//! - Script detection (Devanagari vs Latin)
//! - Code-switching normalization
//! - Transliteration mapping
//! - Query language detection

use std::collections::HashMap;
use parking_lot::RwLock;

/// Detected script in text
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedScript {
    /// Latin/ASCII script (English)
    Latin,
    /// Devanagari script (Hindi)
    Devanagari,
    /// Mixed scripts (code-switching)
    Mixed,
    /// Unknown/other
    Unknown,
}

/// Query language detection result
#[derive(Debug, Clone)]
pub struct LanguageDetection {
    /// Primary script
    pub primary_script: DetectedScript,
    /// Percentage of Devanagari characters
    pub devanagari_ratio: f32,
    /// Percentage of Latin characters
    pub latin_ratio: f32,
    /// Detected as code-switched (mixed language)
    pub is_code_switched: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Normalized query result
#[derive(Debug, Clone)]
pub struct NormalizedQuery {
    /// Original query
    pub original: String,
    /// Normalized query (standardized form)
    pub normalized: String,
    /// Language detection result
    pub language: LanguageDetection,
    /// Transliterated variants
    pub transliterations: Vec<String>,
    /// Whether normalization was applied
    pub was_normalized: bool,
}

/// Cross-lingual query normalizer
pub struct CrossLingualNormalizer {
    /// Transliteration mappings (Roman Hindi -> Devanagari)
    roman_to_devanagari: RwLock<HashMap<String, String>>,
    /// Transliteration mappings (Devanagari -> Roman)
    devanagari_to_roman: RwLock<HashMap<String, String>>,
    /// Common spelling variations
    spelling_variants: RwLock<HashMap<String, String>>,
}

impl CrossLingualNormalizer {
    /// Create a new normalizer
    pub fn new() -> Self {
        let normalizer = Self {
            roman_to_devanagari: RwLock::new(HashMap::new()),
            devanagari_to_roman: RwLock::new(HashMap::new()),
            spelling_variants: RwLock::new(HashMap::new()),
        };
        normalizer.load_default_mappings();
        normalizer
    }

    /// Load default transliteration mappings
    fn load_default_mappings(&self) {
        // Roman Hindi to Devanagari
        let roman_to_dev = vec![
            ("sona", "सोना"),
            ("gold", "गोल्ड"),
            ("loan", "लोन"),
            ("karza", "कर्ज़ा"),
            ("byaj", "ब्याज"),
            ("interest", "इंटरेस्ट"),
            ("rate", "रेट"),
            ("dar", "दर"),
            ("eligibility", "एलिजिबिलिटी"),
            ("patrta", "पात्रता"),
            ("apply", "अप्लाई"),
            ("aavedan", "आवेदन"),
            ("document", "डॉक्यूमेंट"),
            ("dastavez", "दस्तावेज़"),
            ("bank", "बैंक"),
            ("kotak", "कोटक"),
            ("muthoot", "मुथूट"),
            ("emi", "ईएमआई"),
            ("kist", "किस्त"),
            ("amount", "अमाउंट"),
            ("rashi", "राशि"),
            ("customer", "कस्टमर"),
            ("grahak", "ग्राहक"),
            ("kya", "क्या"),
            ("hai", "है"),
            ("hain", "हैं"),
            ("kitna", "कितना"),
            ("kaise", "कैसे"),
            ("kahan", "कहाँ"),
            ("milega", "मिलेगा"),
            ("chahiye", "चाहिए"),
        ];

        let mut r2d = self.roman_to_devanagari.write();
        let mut d2r = self.devanagari_to_roman.write();

        for (roman, dev) in roman_to_dev {
            r2d.insert(roman.to_string(), dev.to_string());
            d2r.insert(dev.to_string(), roman.to_string());
        }

        // Common spelling variations
        let variants = vec![
            ("intrest", "interest"),
            ("interst", "interest"),
            ("intrst", "interest"),
            ("eligiblity", "eligibility"),
            ("eligibilty", "eligibility"),
            ("documnt", "document"),
            ("docment", "document"),
            ("acount", "account"),
            ("accont", "account"),
            ("custmer", "customer"),
            ("customr", "customer"),
            ("muthut", "muthoot"),
            ("manapuram", "manappuram"),
            ("mannappuram", "manappuram"),
            ("kotek", "kotak"),
            ("kotk", "kotak"),
            ("procssing", "processing"),
            ("procesing", "processing"),
            ("disbursl", "disbursal"),
            ("disbursment", "disbursement"),
            ("prepyment", "prepayment"),
            ("forclsure", "foreclosure"),
        ];

        let mut var_map = self.spelling_variants.write();
        for (variant, standard) in variants {
            var_map.insert(variant.to_string(), standard.to_string());
        }
    }

    /// Detect script and language
    pub fn detect_language(&self, text: &str) -> LanguageDetection {
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len().max(1) as f32;

        let devanagari_count = chars
            .iter()
            .filter(|c| Self::is_devanagari(**c))
            .count() as f32;
        let latin_count = chars
            .iter()
            .filter(|c| c.is_ascii_alphabetic())
            .count() as f32;

        let devanagari_ratio = devanagari_count / total;
        let latin_ratio = latin_count / total;

        let primary_script = if devanagari_ratio > 0.7 {
            DetectedScript::Devanagari
        } else if latin_ratio > 0.7 {
            DetectedScript::Latin
        } else if devanagari_ratio > 0.1 && latin_ratio > 0.1 {
            DetectedScript::Mixed
        } else if devanagari_ratio > latin_ratio {
            DetectedScript::Devanagari
        } else {
            DetectedScript::Latin
        };

        let is_code_switched = devanagari_ratio > 0.1 && latin_ratio > 0.1;

        let confidence = if is_code_switched {
            0.6 + (devanagari_ratio.max(latin_ratio) - 0.5).abs() * 0.4
        } else {
            devanagari_ratio.max(latin_ratio)
        };

        LanguageDetection {
            primary_script,
            devanagari_ratio,
            latin_ratio,
            is_code_switched,
            confidence,
        }
    }

    /// Check if character is Devanagari
    fn is_devanagari(c: char) -> bool {
        let code = c as u32;
        // Devanagari range: U+0900 to U+097F
        (0x0900..=0x097F).contains(&code)
    }

    /// Normalize a query
    pub fn normalize(&self, query: &str) -> NormalizedQuery {
        let language = self.detect_language(query);
        let mut normalized = query.to_string();
        let mut transliterations = Vec::new();
        let mut was_normalized = false;

        // Fix common spelling errors
        let variants = self.spelling_variants.read();
        for (variant, standard) in variants.iter() {
            if normalized.to_lowercase().contains(variant) {
                normalized = normalized
                    .to_lowercase()
                    .replace(variant, standard);
                was_normalized = true;
            }
        }

        // Generate transliterations
        match language.primary_script {
            DetectedScript::Latin => {
                // Roman Hindi -> Add Devanagari transliteration
                let r2d = self.roman_to_devanagari.read();
                let mut dev_query = query.to_string();

                for (roman, dev) in r2d.iter() {
                    if query.to_lowercase().contains(roman) {
                        dev_query = dev_query
                            .to_lowercase()
                            .replace(roman, dev);
                    }
                }

                if dev_query != query.to_lowercase() {
                    transliterations.push(dev_query);
                }
            }
            DetectedScript::Devanagari => {
                // Devanagari -> Add Roman transliteration
                let d2r = self.devanagari_to_roman.read();
                let mut roman_query = query.to_string();

                for (dev, roman) in d2r.iter() {
                    if query.contains(dev) {
                        roman_query = roman_query.replace(dev, roman);
                    }
                }

                if roman_query != query {
                    transliterations.push(roman_query);
                }
            }
            DetectedScript::Mixed => {
                // For mixed, generate both directions
                let r2d = self.roman_to_devanagari.read();
                let d2r = self.devanagari_to_roman.read();

                // Try full Devanagari
                let mut dev_query = query.to_string();
                for (roman, dev) in r2d.iter() {
                    if query.to_lowercase().contains(roman) {
                        dev_query = dev_query.to_lowercase().replace(roman, dev);
                    }
                }
                if dev_query != query.to_lowercase() {
                    transliterations.push(dev_query);
                }

                // Try full Roman
                let mut roman_query = query.to_string();
                for (dev, roman) in d2r.iter() {
                    if query.contains(dev) {
                        roman_query = roman_query.replace(dev, roman);
                    }
                }
                if roman_query != query {
                    transliterations.push(roman_query);
                }
            }
            DetectedScript::Unknown => {}
        }

        NormalizedQuery {
            original: query.to_string(),
            normalized,
            language,
            transliterations,
            was_normalized,
        }
    }

    /// Get all query variants (original + transliterations)
    pub fn get_query_variants(&self, query: &str) -> Vec<String> {
        let normalized = self.normalize(query);
        let mut variants = vec![normalized.normalized];

        for trans in normalized.transliterations {
            if !variants.contains(&trans) {
                variants.push(trans);
            }
        }

        variants
    }

    /// Normalize for search (returns best query for retrieval)
    pub fn normalize_for_search(&self, query: &str) -> String {
        let normalized = self.normalize(query);

        // For code-switched queries, prefer Roman for search
        // (most knowledge bases are in Roman Hindi or English)
        if normalized.language.is_code_switched {
            if normalized.language.latin_ratio > normalized.language.devanagari_ratio {
                return normalized.normalized;
            }
            // Return Roman transliteration if available
            for trans in &normalized.transliterations {
                if self.detect_language(trans).primary_script == DetectedScript::Latin {
                    return trans.clone();
                }
            }
        }

        normalized.normalized
    }

    /// Add custom transliteration mapping
    pub fn add_transliteration(&self, roman: &str, devanagari: &str) {
        self.roman_to_devanagari
            .write()
            .insert(roman.to_string(), devanagari.to_string());
        self.devanagari_to_roman
            .write()
            .insert(devanagari.to_string(), roman.to_string());
    }

    /// Add spelling variant
    pub fn add_spelling_variant(&self, variant: &str, standard: &str) {
        self.spelling_variants
            .write()
            .insert(variant.to_string(), standard.to_string());
    }
}

impl Default for CrossLingualNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        let normalizer = CrossLingualNormalizer::new();
        let result = normalizer.detect_language("What is the gold loan interest rate?");

        assert_eq!(result.primary_script, DetectedScript::Latin);
        assert!(result.latin_ratio > 0.8);
        assert!(!result.is_code_switched);
    }

    #[test]
    fn test_detect_hindi() {
        let normalizer = CrossLingualNormalizer::new();
        let result = normalizer.detect_language("सोने का लोन कैसे मिलेगा?");

        assert_eq!(result.primary_script, DetectedScript::Devanagari);
        // Ratio is ~0.74 due to spaces/punctuation in total count
        assert!(result.devanagari_ratio > 0.7);
    }

    #[test]
    fn test_detect_code_switched() {
        let normalizer = CrossLingualNormalizer::new();
        // Use example with more Devanagari for proper mixed detection
        let result = normalizer.detect_language("गोल्ड loan का interest रेट kya है?");

        assert_eq!(result.primary_script, DetectedScript::Mixed);
        assert!(result.is_code_switched);
    }

    #[test]
    fn test_spelling_correction() {
        let normalizer = CrossLingualNormalizer::new();
        let result = normalizer.normalize("what is intrest rate");

        assert!(result.was_normalized);
        assert!(result.normalized.contains("interest"));
    }

    #[test]
    fn test_transliteration_roman_to_dev() {
        let normalizer = CrossLingualNormalizer::new();
        let result = normalizer.normalize("sona loan byaj");

        assert!(!result.transliterations.is_empty());
        // Should have Devanagari transliteration
        let has_devanagari = result.transliterations.iter()
            .any(|t| normalizer.detect_language(t).primary_script == DetectedScript::Devanagari);
        assert!(has_devanagari);
    }

    #[test]
    fn test_get_query_variants() {
        let normalizer = CrossLingualNormalizer::new();
        let variants = normalizer.get_query_variants("gold loan interest");

        assert!(variants.len() >= 1);
        assert!(variants[0].contains("gold") || variants[0].contains("interest"));
    }

    #[test]
    fn test_normalize_for_search() {
        let normalizer = CrossLingualNormalizer::new();

        // English should stay English
        let result = normalizer.normalize_for_search("gold loan rate");
        assert!(!result.is_empty());

        // Code-switched should prefer Roman
        let result = normalizer.normalize_for_search("gold loan का rate kya hai");
        assert!(normalizer.detect_language(&result).primary_script != DetectedScript::Devanagari);
    }

    #[test]
    fn test_custom_transliteration() {
        let normalizer = CrossLingualNormalizer::new();
        normalizer.add_transliteration("custom", "कस्टम");

        let result = normalizer.normalize("custom term");
        assert!(result.transliterations.iter().any(|t| t.contains("कस्टम")));
    }

    #[test]
    fn test_is_devanagari() {
        assert!(CrossLingualNormalizer::is_devanagari('अ'));
        assert!(CrossLingualNormalizer::is_devanagari('क'));
        assert!(!CrossLingualNormalizer::is_devanagari('a'));
        assert!(!CrossLingualNormalizer::is_devanagari('1'));
    }
}
