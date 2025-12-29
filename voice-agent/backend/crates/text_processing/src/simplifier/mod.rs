//! Text Simplifier for TTS
//!
//! Converts text into TTS-friendly format:
//! - Numbers to words (Indian numbering: lakh, crore)
//! - Currency formatting (₹50000 → "fifty thousand rupees")
//! - Abbreviation expansion (EMI → "E M I")
//! - Complex sentence breaking for natural speech
//!
//! # Example
//!
//! ```ignore
//! let simplifier = TextSimplifier::new(TextSimplifierConfig::default());
//! let result = simplifier.simplify("Your EMI is ₹15000 for 12 months");
//! // "Your E M I is fifteen thousand rupees for twelve months"
//! ```

mod numbers;
mod abbreviations;

pub use numbers::{NumberToWords, IndianNumberSystem};
pub use abbreviations::AbbreviationExpander;

use voice_agent_core::Language;
use serde::{Deserialize, Serialize};

/// Configuration for text simplifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSimplifierConfig {
    /// Convert numbers to words
    #[serde(default = "default_true")]
    pub convert_numbers: bool,
    /// Expand abbreviations for TTS
    #[serde(default = "default_true")]
    pub expand_abbreviations: bool,
    /// Break complex sentences
    #[serde(default = "default_true")]
    pub break_sentences: bool,
    /// Maximum sentence length before breaking
    #[serde(default = "default_max_sentence_len")]
    pub max_sentence_length: usize,
    /// Add pauses after numbers for clarity
    #[serde(default)]
    pub pause_after_numbers: bool,
    /// Language for number words
    #[serde(default)]
    pub language: Language,
}

fn default_true() -> bool {
    true
}

fn default_max_sentence_len() -> usize {
    150
}

impl Default for TextSimplifierConfig {
    fn default() -> Self {
        Self {
            convert_numbers: true,
            expand_abbreviations: true,
            break_sentences: true,
            max_sentence_length: 150,
            pause_after_numbers: false,
            language: Language::English,
        }
    }
}

/// Text simplifier for TTS output
pub struct TextSimplifier {
    config: TextSimplifierConfig,
    number_converter: NumberToWords,
    abbreviation_expander: AbbreviationExpander,
}

impl TextSimplifier {
    /// Create a new text simplifier
    pub fn new(config: TextSimplifierConfig) -> Self {
        Self {
            number_converter: NumberToWords::new(config.language),
            abbreviation_expander: AbbreviationExpander::new(),
            config,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(TextSimplifierConfig::default())
    }

    /// Simplify text for TTS
    pub fn simplify(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Step 1: Expand abbreviations first (before number processing)
        if self.config.expand_abbreviations {
            result = self.abbreviation_expander.expand(&result);
        }

        // Step 2: Convert numbers to words
        if self.config.convert_numbers {
            result = self.number_converter.convert(&result);
        }

        // Step 3: Break long sentences
        if self.config.break_sentences {
            result = self.break_long_sentences(&result);
        }

        // Step 4: Clean up whitespace
        result = self.normalize_whitespace(&result);

        result
    }

    /// Break long sentences at natural boundaries
    fn break_long_sentences(&self, text: &str) -> String {
        let max_len = self.config.max_sentence_length;
        let mut result = String::new();
        let mut current_len = 0;

        for word in text.split_whitespace() {
            let word_len = word.chars().count();

            // Check if adding this word exceeds max length
            if current_len > 0 && current_len + word_len + 1 > max_len {
                // Find a natural break point
                if self.is_break_point(word) || self.ends_with_break_hint(&result) {
                    result.push_str(". ");
                    current_len = 0;
                }
            }

            if current_len > 0 {
                result.push(' ');
                current_len += 1;
            }

            result.push_str(word);
            current_len += word_len;
        }

        result
    }

    /// Check if word is a natural break point
    fn is_break_point(&self, word: &str) -> bool {
        let break_words = [
            "and", "but", "or", "so", "then", "also", "however", "therefore",
            "aur", "lekin", "ya", "phir", "toh", // Hindi
            "मतलब", "और", "लेकिन", "या", "फिर", // Hindi Devanagari
        ];
        let lower = word.to_lowercase();
        break_words.iter().any(|w| lower == *w)
    }

    /// Check if result ends with break hint (comma, semicolon)
    fn ends_with_break_hint(&self, text: &str) -> bool {
        text.trim_end().ends_with(',')
            || text.trim_end().ends_with(';')
            || text.trim_end().ends_with(':')
    }

    /// Normalize whitespace
    fn normalize_whitespace(&self, text: &str) -> String {
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Set language for number conversion
    pub fn set_language(&mut self, language: Language) {
        self.config.language = language;
        self.number_converter = NumberToWords::new(language);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_basic() {
        let simplifier = TextSimplifier::default_config();
        let result = simplifier.simplify("Your EMI is ₹15000");
        assert!(result.contains("E M I"));
        assert!(result.contains("fifteen thousand"));
        assert!(result.contains("rupees"));
    }

    #[test]
    fn test_simplify_percentage() {
        let simplifier = TextSimplifier::default_config();
        let result = simplifier.simplify("Interest rate is 8.5%");
        assert!(result.contains("eight point five percent"));
    }

    #[test]
    fn test_simplify_phone() {
        let simplifier = TextSimplifier::default_config();
        let result = simplifier.simplify("Call 9876543210");
        // Should expand digits individually for phone
        assert!(result.contains("nine eight seven six"));
    }
}
