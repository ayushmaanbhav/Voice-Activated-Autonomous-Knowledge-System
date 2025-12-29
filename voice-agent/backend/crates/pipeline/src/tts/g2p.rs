//! Grapheme to Phoneme (G2P) Conversion for Hindi/Hinglish
//!
//! Converts text to phoneme sequences for TTS models.
//! Supports:
//! - Devanagari script (Hindi)
//! - Romanized Hindi (transliteration)
//! - English words (IPA-based)
//! - Code-mixed Hinglish text

use std::collections::HashMap;

use crate::PipelineError;

/// Phoneme for TTS
#[derive(Debug, Clone)]
pub struct Phoneme {
    /// IPA symbol or phoneme code
    pub symbol: String,
    /// Duration modifier (1.0 = normal)
    pub duration: f32,
    /// Stress level (0 = none, 1 = primary, 2 = secondary)
    pub stress: u8,
}

impl Phoneme {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            duration: 1.0,
            stress: 0,
        }
    }

    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    pub fn with_stress(mut self, stress: u8) -> Self {
        self.stress = stress;
        self
    }
}

/// G2P configuration
#[derive(Debug, Clone)]
pub struct G2pConfig {
    /// Primary language
    pub language: Language,
    /// Enable transliteration fallback
    pub transliteration_fallback: bool,
    /// Add word boundaries
    pub add_word_boundaries: bool,
    /// Add silence markers
    pub add_silence: bool,
}

impl Default for G2pConfig {
    fn default() -> Self {
        Self {
            language: Language::Hindi,
            transliteration_fallback: true,
            add_word_boundaries: true,
            add_silence: true,
        }
    }
}

/// Language for G2P
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Hindi,
    English,
    Hinglish,
}

/// Hindi G2P converter
pub struct HindiG2p {
    config: G2pConfig,
    /// Devanagari consonants to phoneme mapping
    consonants: HashMap<char, &'static str>,
    /// Devanagari vowels to phoneme mapping
    vowels: HashMap<char, &'static str>,
    /// Devanagari vowel signs (matras) to phoneme mapping
    matras: HashMap<char, &'static str>,
    /// Roman to Devanagari transliteration
    roman_to_devanagari: HashMap<&'static str, &'static str>,
    /// Common English words phonemes (for code-mixed text)
    english_phonemes: HashMap<&'static str, &'static str>,
}

impl HindiG2p {
    /// Create a new Hindi G2P converter
    pub fn new(config: G2pConfig) -> Self {
        let mut g2p = Self {
            config,
            consonants: HashMap::new(),
            vowels: HashMap::new(),
            matras: HashMap::new(),
            roman_to_devanagari: HashMap::new(),
            english_phonemes: HashMap::new(),
        };
        g2p.init_mappings();
        g2p
    }

    /// Initialize phoneme mappings
    fn init_mappings(&mut self) {
        // Devanagari consonants (व्यंजन)
        self.consonants.insert('क', "k");
        self.consonants.insert('ख', "kʰ");
        self.consonants.insert('ग', "ɡ");
        self.consonants.insert('घ', "ɡʱ");
        self.consonants.insert('ङ', "ŋ");
        self.consonants.insert('च', "tʃ");
        self.consonants.insert('छ', "tʃʰ");
        self.consonants.insert('ज', "dʒ");
        self.consonants.insert('झ', "dʒʱ");
        self.consonants.insert('ञ', "ɲ");
        self.consonants.insert('ट', "ʈ");
        self.consonants.insert('ठ', "ʈʰ");
        self.consonants.insert('ड', "ɖ");
        self.consonants.insert('ढ', "ɖʱ");
        self.consonants.insert('ण', "ɳ");
        self.consonants.insert('त', "t̪");
        self.consonants.insert('थ', "t̪ʰ");
        self.consonants.insert('द', "d̪");
        self.consonants.insert('ध', "d̪ʱ");
        self.consonants.insert('न', "n");
        self.consonants.insert('प', "p");
        self.consonants.insert('फ', "pʰ");
        self.consonants.insert('ब', "b");
        self.consonants.insert('भ', "bʱ");
        self.consonants.insert('म', "m");
        self.consonants.insert('य', "j");
        self.consonants.insert('र', "r");
        self.consonants.insert('ल', "l");
        self.consonants.insert('व', "ʋ");
        self.consonants.insert('श', "ʃ");
        self.consonants.insert('ष', "ʂ");
        self.consonants.insert('स', "s");
        self.consonants.insert('ह', "ɦ");
        // Nukta variants - these are single Unicode codepoints
        // क़ = \u{0958}, ख़ = \u{0959}, ग़ = \u{095A}, ज़ = \u{095B},
        // ड़ = \u{095C}, ढ़ = \u{095D}, फ़ = \u{095E}
        self.consonants.insert('\u{0958}', "q");   // क़
        self.consonants.insert('\u{0959}', "x");   // ख़
        self.consonants.insert('\u{095A}', "ɣ");   // ग़
        self.consonants.insert('\u{095B}', "z");   // ज़
        self.consonants.insert('\u{095C}', "ɽ");   // ड़
        self.consonants.insert('\u{095D}', "ɽʱ");  // ढ़
        self.consonants.insert('\u{095E}', "f");   // फ़

        // Devanagari vowels (स्वर)
        self.vowels.insert('अ', "ə");
        self.vowels.insert('आ', "aː");
        self.vowels.insert('इ', "ɪ");
        self.vowels.insert('ई', "iː");
        self.vowels.insert('उ', "ʊ");
        self.vowels.insert('ऊ', "uː");
        self.vowels.insert('ऋ', "rɪ");
        self.vowels.insert('ए', "eː");
        self.vowels.insert('ऐ', "æː");
        self.vowels.insert('ओ', "oː");
        self.vowels.insert('औ', "ɔː");

        // Vowel signs (matras)
        self.matras.insert('\u{093E}', "aː");  // ा
        self.matras.insert('\u{093F}', "ɪ");   // ि
        self.matras.insert('\u{0940}', "iː");  // ी
        self.matras.insert('\u{0941}', "ʊ");   // ु
        self.matras.insert('\u{0942}', "uː");  // ू
        self.matras.insert('\u{0943}', "rɪ");  // ृ
        self.matras.insert('\u{0947}', "eː");  // े
        self.matras.insert('\u{0948}', "æː");  // ै
        self.matras.insert('\u{094B}', "oː");  // ो
        self.matras.insert('\u{094C}', "ɔː");  // ौ

        // Common Roman to Hindi mappings
        self.roman_to_devanagari.insert("ka", "क");
        self.roman_to_devanagari.insert("kha", "ख");
        self.roman_to_devanagari.insert("ga", "ग");
        self.roman_to_devanagari.insert("gha", "घ");
        self.roman_to_devanagari.insert("cha", "च");
        self.roman_to_devanagari.insert("ja", "ज");
        self.roman_to_devanagari.insert("ta", "त");
        self.roman_to_devanagari.insert("tha", "थ");
        self.roman_to_devanagari.insert("da", "द");
        self.roman_to_devanagari.insert("dha", "ध");
        self.roman_to_devanagari.insert("na", "न");
        self.roman_to_devanagari.insert("pa", "प");
        self.roman_to_devanagari.insert("pha", "फ");
        self.roman_to_devanagari.insert("ba", "ब");
        self.roman_to_devanagari.insert("bha", "भ");
        self.roman_to_devanagari.insert("ma", "म");
        self.roman_to_devanagari.insert("ya", "य");
        self.roman_to_devanagari.insert("ra", "र");
        self.roman_to_devanagari.insert("la", "ल");
        self.roman_to_devanagari.insert("va", "व");
        self.roman_to_devanagari.insert("wa", "व");
        self.roman_to_devanagari.insert("sha", "श");
        self.roman_to_devanagari.insert("sa", "स");
        self.roman_to_devanagari.insert("ha", "ह");

        // Common English words for code-mixed text (gold loan domain)
        self.english_phonemes.insert("gold", "ɡoʊld");
        self.english_phonemes.insert("loan", "loʊn");
        self.english_phonemes.insert("interest", "ˈɪntərəst");
        self.english_phonemes.insert("rate", "reɪt");
        self.english_phonemes.insert("bank", "bæŋk");
        self.english_phonemes.insert("branch", "bræntʃ");
        self.english_phonemes.insert("apply", "əˈplaɪ");
        self.english_phonemes.insert("amount", "əˈmaʊnt");
        self.english_phonemes.insert("rupees", "ruːˈpiːz");
        self.english_phonemes.insert("percent", "pərˈsɛnt");
        self.english_phonemes.insert("emi", "iːɛmˈaɪ");
        self.english_phonemes.insert("processing", "ˈprɒsɛsɪŋ");
        self.english_phonemes.insert("fee", "fiː");
        self.english_phonemes.insert("kotak", "koːˈtək");
        self.english_phonemes.insert("mahindra", "məˈhɪndrə");
        self.english_phonemes.insert("muthoot", "muːˈtuːt");
        self.english_phonemes.insert("manappuram", "mənˈæpʊrəm");
    }

    /// Convert text to phonemes
    pub fn convert(&self, text: &str) -> Result<Vec<Phoneme>, PipelineError> {
        let mut phonemes = Vec::new();

        if self.config.add_silence {
            phonemes.push(Phoneme::new("sil"));
        }

        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            if self.config.add_word_boundaries && i > 0 {
                phonemes.push(Phoneme::new(" "));
            }

            let word_phonemes = self.word_to_phonemes(word)?;
            phonemes.extend(word_phonemes);
        }

        if self.config.add_silence {
            phonemes.push(Phoneme::new("sil"));
        }

        Ok(phonemes)
    }

    /// Convert a single word to phonemes
    fn word_to_phonemes(&self, word: &str) -> Result<Vec<Phoneme>, PipelineError> {
        let word_lower = word.to_lowercase();

        // Check if it's a known English word
        if let Some(ipa) = self.english_phonemes.get(word_lower.as_str()) {
            return Ok(self.ipa_to_phonemes(ipa));
        }

        // Check if it's Devanagari
        if self.is_devanagari(word) {
            return self.devanagari_to_phonemes(word);
        }

        // Try Roman Hindi transliteration
        if self.config.transliteration_fallback {
            return self.roman_hindi_to_phonemes(&word_lower);
        }

        // Fallback: spell it out
        Ok(word.chars().map(|c| Phoneme::new(&c.to_string())).collect())
    }

    /// Check if text is Devanagari
    fn is_devanagari(&self, text: &str) -> bool {
        text.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c))
    }

    /// Convert Devanagari text to phonemes
    fn devanagari_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>, PipelineError> {
        let mut phonemes = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            // Check for vowel (standalone)
            if let Some(phoneme) = self.vowels.get(&c) {
                phonemes.push(Phoneme::new(phoneme));
                i += 1;
                continue;
            }

            // Check for consonant
            if let Some(consonant_phoneme) = self.consonants.get(&c) {
                phonemes.push(Phoneme::new(consonant_phoneme));

                // Look ahead for virama or matra
                if i + 1 < chars.len() {
                    let next = chars[i + 1];

                    // Virama (halant) - no inherent vowel
                    if next == '\u{094D}' {
                        i += 2;
                        continue;
                    }

                    // Matra (vowel sign)
                    if let Some(matra_phoneme) = self.matras.get(&next) {
                        phonemes.push(Phoneme::new(matra_phoneme));
                        i += 2;
                        continue;
                    }
                }

                // Default inherent vowel 'schwa' (ə)
                phonemes.push(Phoneme::new("ə"));
                i += 1;
                continue;
            }

            // Anusvara (nasalization)
            if c == '\u{0902}' {
                phonemes.push(Phoneme::new("̃")); // Nasalization mark
                i += 1;
                continue;
            }

            // Visarga (aspiration)
            if c == '\u{0903}' {
                phonemes.push(Phoneme::new("h"));
                i += 1;
                continue;
            }

            // Chandrabindu
            if c == '\u{0901}' {
                // Nasalized - add to previous
                phonemes.push(Phoneme::new("̃"));
                i += 1;
                continue;
            }

            // Numbers
            if ('\u{0966}'..='\u{096F}').contains(&c) {
                let digit = (c as u32 - '\u{0966}' as u32) as u8;
                phonemes.push(Phoneme::new(&digit.to_string()));
                i += 1;
                continue;
            }

            // Punctuation and spaces
            if c.is_whitespace() || c.is_ascii_punctuation() {
                phonemes.push(Phoneme::new(" "));
                i += 1;
                continue;
            }

            // Unknown character - skip
            i += 1;
        }

        Ok(phonemes)
    }

    /// Convert Roman Hindi to phonemes
    fn roman_hindi_to_phonemes(&self, text: &str) -> Result<Vec<Phoneme>, PipelineError> {
        let mut phonemes = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Try to match longest possible sequence
            let mut matched = false;

            // Try 3-char, 2-char, then 1-char sequences
            for len in (1..=3).rev() {
                if i + len <= chars.len() {
                    let seq: String = chars[i..i + len].iter().collect();

                    // Simple phoneme mapping for common Roman Hindi sounds
                    if let Some(phoneme) = self.roman_sound_to_phoneme(&seq) {
                        phonemes.push(Phoneme::new(phoneme));
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }

            if !matched {
                // Single character fallback
                let c = chars[i];
                if c.is_alphabetic() {
                    if let Some(phoneme) = self.single_char_phoneme(c) {
                        phonemes.push(Phoneme::new(phoneme));
                    }
                } else if c.is_whitespace() {
                    phonemes.push(Phoneme::new(" "));
                }
                i += 1;
            }
        }

        Ok(phonemes)
    }

    /// Map Roman sound sequences to phonemes
    fn roman_sound_to_phoneme(&self, seq: &str) -> Option<&'static str> {
        match seq {
            // Aspirated consonants (must come before basic)
            "kha" | "kh" => Some("kʰ"),
            "gha" | "gh" => Some("ɡʱ"),
            "cha" | "ch" => Some("tʃ"),
            "chh" => Some("tʃʰ"),
            "jha" | "jh" => Some("dʒʱ"),
            "tha" | "th" => Some("t̪ʰ"),
            "dha" | "dh" => Some("d̪ʱ"),
            "pha" | "ph" => Some("pʰ"),
            "bha" | "bh" => Some("bʱ"),
            "sha" | "sh" => Some("ʃ"),
            // Long vowels
            "aa" => Some("aː"),
            "ee" | "ii" => Some("iː"),
            "oo" | "uu" => Some("uː"),
            "ai" => Some("æː"),
            "au" | "ou" => Some("ɔː"),
            // Basic consonants (after aspirated)
            "ka" => Some("kə"),
            "ga" => Some("ɡə"),
            "ja" => Some("dʒə"),
            "ta" => Some("t̪ə"),
            "da" => Some("d̪ə"),
            "na" => Some("nə"),
            "pa" => Some("pə"),
            "ba" => Some("bə"),
            "ma" => Some("mə"),
            "ya" => Some("jə"),
            "ra" => Some("rə"),
            "la" => Some("lə"),
            "va" | "wa" => Some("ʋə"),
            "sa" => Some("sə"),
            "ha" => Some("ɦə"),
            _ => None,
        }
    }

    /// Map single character to phoneme
    fn single_char_phoneme(&self, c: char) -> Option<&'static str> {
        match c {
            'a' => Some("ə"),
            'e' => Some("eː"),
            'i' => Some("ɪ"),
            'o' => Some("oː"),
            'u' => Some("ʊ"),
            'k' => Some("k"),
            'g' => Some("ɡ"),
            'c' => Some("tʃ"),
            'j' => Some("dʒ"),
            't' => Some("t̪"),
            'd' => Some("d̪"),
            'n' => Some("n"),
            'p' => Some("p"),
            'b' => Some("b"),
            'f' => Some("f"),
            'm' => Some("m"),
            'y' => Some("j"),
            'r' => Some("r"),
            'l' => Some("l"),
            'v' | 'w' => Some("ʋ"),
            's' => Some("s"),
            'h' => Some("ɦ"),
            'z' => Some("z"),
            _ => None,
        }
    }

    /// Convert IPA string to phoneme sequence
    fn ipa_to_phonemes(&self, ipa: &str) -> Vec<Phoneme> {
        // Simple split on common IPA boundaries
        // In practice, this would need proper IPA parsing
        ipa.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| Phoneme::new(&c.to_string()))
            .collect()
    }

    /// Get phoneme sequence as string (for TTS input)
    pub fn phonemes_to_string(&self, phonemes: &[Phoneme]) -> String {
        phonemes.iter()
            .map(|p| p.symbol.as_str())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Create default Hindi G2P
pub fn create_hindi_g2p() -> HindiG2p {
    HindiG2p::new(G2pConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_devanagari_to_phonemes() {
        let g2p = create_hindi_g2p();

        // Simple Hindi word: नमस्ते (namaste)
        let phonemes = g2p.convert("नमस्ते").unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_english_word() {
        let g2p = create_hindi_g2p();

        let phonemes = g2p.convert("gold loan").unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_mixed_text() {
        let g2p = create_hindi_g2p();

        // Mixed Hindi-English: "मुझे gold loan चाहिए"
        let phonemes = g2p.convert("मुझे gold loan चाहिए").unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_roman_hindi() {
        let g2p = create_hindi_g2p();

        let phonemes = g2p.convert("kya aap mujhe gold loan de sakte hain").unwrap();
        assert!(!phonemes.is_empty());
    }

    #[test]
    fn test_phoneme_to_string() {
        let g2p = create_hindi_g2p();

        let phonemes = g2p.convert("hello").unwrap();
        let s = g2p.phonemes_to_string(&phonemes);
        assert!(!s.is_empty());
    }
}
