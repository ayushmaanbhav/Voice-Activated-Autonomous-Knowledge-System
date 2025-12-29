//! Language definitions for 22 Indian languages
//!
//! Supports all 22 scheduled Indian languages plus English, as per
//! the Eighth Schedule of the Indian Constitution.

use serde::{Deserialize, Serialize};

/// Supported languages (22 scheduled Indian languages + English)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    #[default]
    English,
    Hindi,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Bengali,
    Marathi,
    Gujarati,
    Punjabi,
    Odia,
    Assamese,
    Urdu,
    Kashmiri,
    Sindhi,
    Konkani,
    Dogri,
    Bodo,
    Maithili,
    Santali,
    Nepali,
    Manipuri,
    Sanskrit,
}

impl Language {
    /// Get ISO 639-1/639-3 code
    pub fn code(&self) -> &'static str {
        match self {
            Self::English => "en",
            Self::Hindi => "hi",
            Self::Tamil => "ta",
            Self::Telugu => "te",
            Self::Kannada => "kn",
            Self::Malayalam => "ml",
            Self::Bengali => "bn",
            Self::Marathi => "mr",
            Self::Gujarati => "gu",
            Self::Punjabi => "pa",
            Self::Odia => "or",
            Self::Assamese => "as",
            Self::Urdu => "ur",
            Self::Kashmiri => "ks",
            Self::Sindhi => "sd",
            Self::Konkani => "kok",
            Self::Dogri => "doi",
            Self::Bodo => "brx",
            Self::Maithili => "mai",
            Self::Santali => "sat",
            Self::Nepali => "ne",
            Self::Manipuri => "mni",
            Self::Sanskrit => "sa",
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::English => "English",
            Self::Hindi => "Hindi",
            Self::Tamil => "Tamil",
            Self::Telugu => "Telugu",
            Self::Kannada => "Kannada",
            Self::Malayalam => "Malayalam",
            Self::Bengali => "Bengali",
            Self::Marathi => "Marathi",
            Self::Gujarati => "Gujarati",
            Self::Punjabi => "Punjabi",
            Self::Odia => "Odia",
            Self::Assamese => "Assamese",
            Self::Urdu => "Urdu",
            Self::Kashmiri => "Kashmiri",
            Self::Sindhi => "Sindhi",
            Self::Konkani => "Konkani",
            Self::Dogri => "Dogri",
            Self::Bodo => "Bodo",
            Self::Maithili => "Maithili",
            Self::Santali => "Santali",
            Self::Nepali => "Nepali",
            Self::Manipuri => "Manipuri",
            Self::Sanskrit => "Sanskrit",
        }
    }

    /// Get script used by this language
    pub fn script(&self) -> Script {
        match self {
            Self::Hindi | Self::Marathi | Self::Sanskrit | Self::Konkani
            | Self::Dogri | Self::Bodo | Self::Maithili | Self::Nepali => Script::Devanagari,
            Self::Tamil => Script::Tamil,
            Self::Telugu => Script::Telugu,
            Self::Kannada => Script::Kannada,
            Self::Malayalam => Script::Malayalam,
            Self::Bengali | Self::Assamese => Script::Bengali,
            Self::Gujarati => Script::Gujarati,
            Self::Punjabi => Script::Gurmukhi,
            Self::Odia => Script::Odia,
            Self::Urdu | Self::Kashmiri | Self::Sindhi => Script::Arabic,
            Self::Santali => Script::OlChiki,
            Self::Manipuri => Script::MeeteiMayek,
            Self::English => Script::Latin,
        }
    }

    /// Check if this language uses right-to-left script
    pub fn is_rtl(&self) -> bool {
        matches!(self.script(), Script::Arabic)
    }

    /// Get sentence terminators for this language's script
    pub fn sentence_terminators(&self) -> &'static [char] {
        match self.script() {
            Script::Devanagari => &['.', '?', '!', '।', '॥'],
            Script::Bengali => &['.', '?', '!', '।'],
            Script::Tamil => &['.', '?', '!', '।'],
            Script::Telugu => &['.', '?', '!', '।'],
            Script::Kannada => &['.', '?', '!', '।', '॥'],
            Script::Malayalam => &['.', '?', '!', '।'],
            Script::Gujarati => &['.', '?', '!', '।'],
            Script::Gurmukhi => &['.', '?', '!', '।', '॥'],
            Script::Odia => &['.', '?', '!', '।'],
            Script::Arabic => &['.', '?', '!', '؟', '۔'],
            Script::OlChiki => &['.', '?', '!', '᱾', '᱿'],
            Script::MeeteiMayek => &['.', '?', '!', '꯫'],
            Script::Latin => &['.', '?', '!'],
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str_loose(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();
        match s.as_str() {
            "en" | "eng" | "english" => Some(Self::English),
            "hi" | "hin" | "hindi" => Some(Self::Hindi),
            "ta" | "tam" | "tamil" => Some(Self::Tamil),
            "te" | "tel" | "telugu" => Some(Self::Telugu),
            "kn" | "kan" | "kannada" => Some(Self::Kannada),
            "ml" | "mal" | "malayalam" => Some(Self::Malayalam),
            "bn" | "ben" | "bengali" | "bangla" => Some(Self::Bengali),
            "mr" | "mar" | "marathi" => Some(Self::Marathi),
            "gu" | "guj" | "gujarati" => Some(Self::Gujarati),
            "pa" | "pan" | "punjabi" | "panjabi" => Some(Self::Punjabi),
            "or" | "ori" | "odia" | "oriya" => Some(Self::Odia),
            "as" | "asm" | "assamese" => Some(Self::Assamese),
            "ur" | "urd" | "urdu" => Some(Self::Urdu),
            "ks" | "kas" | "kashmiri" => Some(Self::Kashmiri),
            "sd" | "snd" | "sindhi" => Some(Self::Sindhi),
            "kok" | "konkani" => Some(Self::Konkani),
            "doi" | "dogri" => Some(Self::Dogri),
            "brx" | "bodo" => Some(Self::Bodo),
            "mai" | "maithili" => Some(Self::Maithili),
            "sat" | "santali" | "santhali" => Some(Self::Santali),
            "ne" | "nep" | "nepali" => Some(Self::Nepali),
            "mni" | "manipuri" | "meitei" => Some(Self::Manipuri),
            "sa" | "san" | "sanskrit" => Some(Self::Sanskrit),
            _ => None,
        }
    }

    /// Get all supported languages
    pub fn all() -> &'static [Language] {
        &[
            Self::English,
            Self::Hindi,
            Self::Tamil,
            Self::Telugu,
            Self::Kannada,
            Self::Malayalam,
            Self::Bengali,
            Self::Marathi,
            Self::Gujarati,
            Self::Punjabi,
            Self::Odia,
            Self::Assamese,
            Self::Urdu,
            Self::Kashmiri,
            Self::Sindhi,
            Self::Konkani,
            Self::Dogri,
            Self::Bodo,
            Self::Maithili,
            Self::Santali,
            Self::Nepali,
            Self::Manipuri,
            Self::Sanskrit,
        ]
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Script systems used by Indian languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Script {
    Latin,
    Devanagari,
    Bengali,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Gujarati,
    Gurmukhi,
    Odia,
    Arabic,
    OlChiki,
    MeeteiMayek,
}

impl Script {
    /// Get Unicode range for this script (first block only)
    pub fn unicode_range(&self) -> (u32, u32) {
        match self {
            Self::Latin => (0x0000, 0x007F),
            Self::Devanagari => (0x0900, 0x097F),
            Self::Bengali => (0x0980, 0x09FF),
            Self::Tamil => (0x0B80, 0x0BFF),
            Self::Telugu => (0x0C00, 0x0C7F),
            Self::Kannada => (0x0C80, 0x0CFF),
            Self::Malayalam => (0x0D00, 0x0D7F),
            Self::Gujarati => (0x0A80, 0x0AFF),
            Self::Gurmukhi => (0x0A00, 0x0A7F),
            Self::Odia => (0x0B00, 0x0B7F),
            Self::Arabic => (0x0600, 0x06FF),
            Self::OlChiki => (0x1C50, 0x1C7F),
            Self::MeeteiMayek => (0xABC0, 0xABFF),
        }
    }

    /// Check if a character belongs to this script
    pub fn contains_char(&self, c: char) -> bool {
        let code = c as u32;
        let (start, end) = self.unicode_range();
        code >= start && code <= end
    }

    /// Detect script from text (returns most frequent script)
    pub fn detect(text: &str) -> Option<Self> {
        let mut counts = std::collections::HashMap::new();

        for c in text.chars() {
            for script in &[
                Self::Devanagari,
                Self::Bengali,
                Self::Tamil,
                Self::Telugu,
                Self::Kannada,
                Self::Malayalam,
                Self::Gujarati,
                Self::Gurmukhi,
                Self::Odia,
                Self::Arabic,
                Self::OlChiki,
                Self::MeeteiMayek,
                Self::Latin,
            ] {
                if script.contains_char(c) {
                    *counts.entry(*script).or_insert(0) += 1;
                    break;
                }
            }
        }

        counts.into_iter().max_by_key(|(_, v)| *v).map(|(k, _)| k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_code() {
        assert_eq!(Language::Hindi.code(), "hi");
        assert_eq!(Language::Tamil.code(), "ta");
        assert_eq!(Language::English.code(), "en");
    }

    #[test]
    fn test_language_script() {
        assert_eq!(Language::Hindi.script(), Script::Devanagari);
        assert_eq!(Language::Tamil.script(), Script::Tamil);
        assert_eq!(Language::Urdu.script(), Script::Arabic);
        assert_eq!(Language::Bengali.script(), Script::Bengali);
    }

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str_loose("hi"), Some(Language::Hindi));
        assert_eq!(Language::from_str_loose("Hindi"), Some(Language::Hindi));
        assert_eq!(Language::from_str_loose("TAMIL"), Some(Language::Tamil));
        assert_eq!(Language::from_str_loose("bangla"), Some(Language::Bengali));
        assert_eq!(Language::from_str_loose("unknown"), None);
    }

    #[test]
    fn test_script_detect() {
        assert_eq!(Script::detect("Hello world"), Some(Script::Latin));
        assert_eq!(Script::detect("नमस्ते"), Some(Script::Devanagari));
        assert_eq!(Script::detect("வணக்கம்"), Some(Script::Tamil));
    }

    #[test]
    fn test_sentence_terminators() {
        let hindi_terms = Language::Hindi.sentence_terminators();
        assert!(hindi_terms.contains(&'।'));
        assert!(hindi_terms.contains(&'.'));
    }

    #[test]
    fn test_all_languages() {
        let all = Language::all();
        assert_eq!(all.len(), 23); // 22 Indian + English
    }
}
