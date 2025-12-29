//! Script and language detection

use voice_agent_core::{Language, Script};
use std::collections::HashMap;

/// Script-based language detector
#[derive(Debug, Clone)]
pub struct ScriptDetector {
    script_to_language: HashMap<Script, Language>,
}

impl ScriptDetector {
    /// Create a new script detector
    pub fn new() -> Self {
        let mut map = HashMap::new();
        map.insert(Script::Devanagari, Language::Hindi);
        map.insert(Script::Tamil, Language::Tamil);
        map.insert(Script::Telugu, Language::Telugu);
        map.insert(Script::Kannada, Language::Kannada);
        map.insert(Script::Malayalam, Language::Malayalam);
        map.insert(Script::Bengali, Language::Bengali);
        map.insert(Script::Gujarati, Language::Gujarati);
        map.insert(Script::Gurmukhi, Language::Punjabi);
        map.insert(Script::Odia, Language::Odia);
        map.insert(Script::Arabic, Language::Urdu);
        map.insert(Script::Latin, Language::English);
        map.insert(Script::OlChiki, Language::Santali);
        map.insert(Script::MeeteiMayek, Language::Manipuri);

        Self { script_to_language: map }
    }

    /// Detect language from text based on script
    pub fn detect(&self, text: &str) -> Language {
        let script = self.detect_script(text);
        self.script_to_language
            .get(&script)
            .copied()
            .unwrap_or(Language::English)
    }

    /// Detect dominant script in text
    pub fn detect_script(&self, text: &str) -> Script {
        let mut counts: HashMap<Script, usize> = HashMap::new();

        for c in text.chars() {
            if c.is_whitespace() || c.is_ascii_punctuation() {
                continue;
            }

            let script = Self::char_to_script(c);
            *counts.entry(script).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(script, _)| script)
            .unwrap_or(Script::Latin)
    }

    /// Detect if text is code-switched (multiple scripts)
    pub fn is_code_switched(&self, text: &str) -> bool {
        let mut scripts_found = std::collections::HashSet::new();

        for c in text.chars() {
            if c.is_whitespace() || c.is_ascii_punctuation() || c.is_ascii_digit() {
                continue;
            }

            let script = Self::char_to_script(c);
            scripts_found.insert(script);

            if scripts_found.len() > 1 {
                return true;
            }
        }

        false
    }

    /// Map character to script based on Unicode range
    fn char_to_script(c: char) -> Script {
        let code = c as u32;
        match code {
            // ASCII/Latin
            0x0000..=0x007F => Script::Latin,
            // Extended Latin
            0x0080..=0x024F => Script::Latin,
            // Arabic
            0x0600..=0x06FF | 0x0750..=0x077F | 0x08A0..=0x08FF => Script::Arabic,
            // Devanagari
            0x0900..=0x097F | 0xA8E0..=0xA8FF => Script::Devanagari,
            // Bengali/Assamese
            0x0980..=0x09FF => Script::Bengali,
            // Gurmukhi
            0x0A00..=0x0A7F => Script::Gurmukhi,
            // Gujarati
            0x0A80..=0x0AFF => Script::Gujarati,
            // Odia
            0x0B00..=0x0B7F => Script::Odia,
            // Tamil
            0x0B80..=0x0BFF => Script::Tamil,
            // Telugu
            0x0C00..=0x0C7F => Script::Telugu,
            // Kannada
            0x0C80..=0x0CFF => Script::Kannada,
            // Malayalam
            0x0D00..=0x0D7F => Script::Malayalam,
            // Ol Chiki (Santali)
            0x1C50..=0x1C7F => Script::OlChiki,
            // Meetei Mayek (Manipuri)
            0xABC0..=0xABFF | 0xAAE0..=0xAAFF => Script::MeeteiMayek,
            // Default to Latin
            _ => Script::Latin,
        }
    }

    /// Get confidence score for language detection
    pub fn detect_with_confidence(&self, text: &str) -> (Language, f32) {
        let mut counts: HashMap<Script, usize> = HashMap::new();
        let mut total = 0usize;

        for c in text.chars() {
            if c.is_whitespace() || c.is_ascii_punctuation() {
                continue;
            }

            let script = Self::char_to_script(c);
            *counts.entry(script).or_insert(0) += 1;
            total += 1;
        }

        if total == 0 {
            return (Language::English, 0.0);
        }

        let (dominant_script, count) = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((Script::Latin, 0));

        let confidence = count as f32 / total as f32;
        let language = self.script_to_language
            .get(&dominant_script)
            .copied()
            .unwrap_or(Language::English);

        (language, confidence)
    }
}

impl Default for ScriptDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_hindi() {
        let detector = ScriptDetector::new();
        assert_eq!(detector.detect("नमस्ते"), Language::Hindi);
        assert_eq!(detector.detect_script("नमस्ते"), Script::Devanagari);
    }

    #[test]
    fn test_detect_tamil() {
        let detector = ScriptDetector::new();
        assert_eq!(detector.detect("வணக்கம்"), Language::Tamil);
        assert_eq!(detector.detect_script("வணக்கம்"), Script::Tamil);
    }

    #[test]
    fn test_detect_english() {
        let detector = ScriptDetector::new();
        assert_eq!(detector.detect("Hello world"), Language::English);
        assert_eq!(detector.detect_script("Hello world"), Script::Latin);
    }

    #[test]
    fn test_code_switching() {
        let detector = ScriptDetector::new();
        assert!(detector.is_code_switched("Hello नमस्ते"));
        assert!(!detector.is_code_switched("Hello world"));
        assert!(!detector.is_code_switched("नमस्ते दुनिया"));
    }

    #[test]
    fn test_detect_with_confidence() {
        let detector = ScriptDetector::new();

        let (lang, conf) = detector.detect_with_confidence("नमस्ते");
        assert_eq!(lang, Language::Hindi);
        assert!(conf > 0.9);

        let (_lang, conf) = detector.detect_with_confidence("Hello नमस्ते");
        assert!(conf < 0.9); // Mixed script = lower confidence
    }

    #[test]
    fn test_detect_bengali() {
        let detector = ScriptDetector::new();
        assert_eq!(detector.detect("নমস্কার"), Language::Bengali);
    }

    #[test]
    fn test_detect_telugu() {
        let detector = ScriptDetector::new();
        assert_eq!(detector.detect("నమస్కారం"), Language::Telugu);
    }
}
