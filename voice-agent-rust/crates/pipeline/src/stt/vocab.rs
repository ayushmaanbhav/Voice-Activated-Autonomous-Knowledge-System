//! Vocabulary Loader for STT Models
//!
//! Supports loading vocabularies for:
//! - Whisper (JSON vocab file)
//! - IndicConformer (SentencePiece .vocab or text file)
//! - Wav2Vec2 (JSON or text vocab)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::PipelineError;
use super::SttEngine;

/// Vocabulary for STT decoding
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token ID to string mapping
    tokens: Vec<String>,
    /// String to token ID mapping (for constrained decoding)
    token_to_id: HashMap<String, u32>,
    /// Special tokens
    pub blank_id: u32,
    pub unk_id: u32,
    pub pad_id: Option<u32>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

impl Vocabulary {
    /// Create vocabulary from token list
    pub fn new(tokens: Vec<String>) -> Self {
        let token_to_id: HashMap<String, u32> = tokens.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        // Find special tokens
        let blank_id = token_to_id.get("<blank>")
            .or_else(|| token_to_id.get("<pad>"))
            .or_else(|| token_to_id.get("[PAD]"))
            .copied()
            .unwrap_or(0);

        let unk_id = token_to_id.get("<unk>")
            .or_else(|| token_to_id.get("[UNK]"))
            .copied()
            .unwrap_or(1);

        let pad_id = token_to_id.get("<pad>")
            .or_else(|| token_to_id.get("[PAD]"))
            .copied();

        let bos_id = token_to_id.get("<s>")
            .or_else(|| token_to_id.get("<bos>"))
            .or_else(|| token_to_id.get("[BOS]"))
            .copied();

        let eos_id = token_to_id.get("</s>")
            .or_else(|| token_to_id.get("<eos>"))
            .or_else(|| token_to_id.get("[EOS]"))
            .copied();

        Self {
            tokens,
            token_to_id,
            blank_id,
            unk_id,
            pad_id,
            bos_id,
            eos_id,
        }
    }

    /// Get token by ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Get ID by token
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    /// Get all tokens
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Convert to Vec<String> for compatibility
    pub fn into_tokens(self) -> Vec<String> {
        self.tokens
    }
}

/// Load vocabulary for specific engine
pub fn load_vocabulary(engine: &SttEngine, model_dir: Option<&Path>) -> Result<Vocabulary, PipelineError> {
    match engine {
        SttEngine::Whisper => load_whisper_vocab(model_dir),
        SttEngine::IndicConformer => load_indicconformer_vocab(model_dir),
        SttEngine::Wav2Vec2 => load_wav2vec2_vocab(model_dir),
    }
}

/// Load Whisper vocabulary
fn load_whisper_vocab(model_dir: Option<&Path>) -> Result<Vocabulary, PipelineError> {
    if let Some(dir) = model_dir {
        // Try to load from vocab.json
        let vocab_path = dir.join("vocab.json");
        if vocab_path.exists() {
            return load_json_vocab(&vocab_path);
        }

        // Try tokenizer.json
        let tokenizer_path = dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            return load_tokenizer_json(&tokenizer_path);
        }
    }

    // Return default Whisper vocabulary (placeholder tokens)
    // Whisper has 51865 tokens
    let tokens: Vec<String> = (0..51865).map(|i| {
        match i {
            0 => "<|endoftext|>".to_string(),
            1 => "<|startoftranscript|>".to_string(),
            2 => "<|en|>".to_string(),  // English
            3 => "<|hi|>".to_string(),  // Hindi
            50257 => "<|translate|>".to_string(),
            50258 => "<|transcribe|>".to_string(),
            50259 => "<|startoflm|>".to_string(),
            50260 => "<|startofprev|>".to_string(),
            50261 => "<|nospeech|>".to_string(),
            50262 => "<|notimestamps|>".to_string(),
            _ => format!("<tok_{}>", i),
        }
    }).collect();

    Ok(Vocabulary::new(tokens))
}

/// Load IndicConformer vocabulary
fn load_indicconformer_vocab(model_dir: Option<&Path>) -> Result<Vocabulary, PipelineError> {
    if let Some(dir) = model_dir {
        // Try SentencePiece vocab file
        let vocab_path = dir.join("vocab.txt");
        if vocab_path.exists() {
            return load_text_vocab(&vocab_path);
        }

        // Try .vocab file (SentencePiece format)
        let spm_vocab_path = dir.join("tokenizer.vocab");
        if spm_vocab_path.exists() {
            return load_sentencepiece_vocab(&spm_vocab_path);
        }
    }

    // Return default IndicConformer vocabulary
    // Hindi/Indian language characters + subword tokens
    let mut tokens = Vec::with_capacity(8000);

    // Special tokens
    tokens.push("<blank>".to_string());
    tokens.push("<unk>".to_string());
    tokens.push("<s>".to_string());
    tokens.push("</s>".to_string());

    // Devanagari characters (Hindi)
    // Vowels
    for c in '\u{0904}'..='\u{0914}' {
        tokens.push(c.to_string());
    }
    // Consonants
    for c in '\u{0915}'..='\u{0939}' {
        tokens.push(c.to_string());
    }
    // Dependent vowel signs
    for c in '\u{093E}'..='\u{094C}' {
        tokens.push(c.to_string());
    }
    // Various signs
    tokens.push('\u{094D}'.to_string()); // Virama (halant)
    tokens.push('\u{0902}'.to_string()); // Anusvara
    tokens.push('\u{0903}'.to_string()); // Visarga
    tokens.push('\u{0901}'.to_string()); // Chandrabindu

    // Devanagari digits
    for c in '\u{0966}'..='\u{096F}' {
        tokens.push(c.to_string());
    }

    // English letters (for code-mixing)
    for c in 'a'..='z' {
        tokens.push(c.to_string());
    }
    for c in 'A'..='Z' {
        tokens.push(c.to_string());
    }

    // Digits
    for c in '0'..='9' {
        tokens.push(c.to_string());
    }

    // Common punctuation
    tokens.push(" ".to_string());
    tokens.push(".".to_string());
    tokens.push(",".to_string());
    tokens.push("?".to_string());
    tokens.push("!".to_string());
    tokens.push("-".to_string());

    // Common Hindi subwords (placeholder)
    let common_subwords = [
        "▁", "▁क", "▁म", "▁ह", "▁स", "▁न", "▁प", "▁ब", "▁त", "▁र",
        "▁गो", "▁लो", "▁का", "▁को", "▁की", "▁के", "▁है", "▁हैं",
        "▁में", "▁से", "▁पर", "▁और", "▁या", "▁लोन", "▁गोल्ड",
        "▁ब्याज", "▁दर", "▁राशि", "▁रुपये", "▁ब्रांच", "▁कोटक",
    ];
    for subword in common_subwords {
        tokens.push(subword.to_string());
    }

    // Fill remaining slots with placeholder tokens
    while tokens.len() < 8000 {
        tokens.push(format!("<tok_{}>", tokens.len()));
    }

    Ok(Vocabulary::new(tokens))
}

/// Load Wav2Vec2 vocabulary
fn load_wav2vec2_vocab(model_dir: Option<&Path>) -> Result<Vocabulary, PipelineError> {
    if let Some(dir) = model_dir {
        let vocab_path = dir.join("vocab.json");
        if vocab_path.exists() {
            return load_json_vocab(&vocab_path);
        }
    }

    // Character-based vocabulary for Wav2Vec2
    let mut tokens = Vec::with_capacity(100);

    tokens.push("<pad>".to_string());    // 0: padding/blank
    tokens.push("<s>".to_string());      // 1: start
    tokens.push("</s>".to_string());     // 2: end
    tokens.push("<unk>".to_string());    // 3: unknown
    tokens.push("|".to_string());        // 4: word boundary

    // Lowercase letters
    for c in 'a'..='z' {
        tokens.push(c.to_string());
    }

    // Space and apostrophe
    tokens.push(" ".to_string());
    tokens.push("'".to_string());

    Ok(Vocabulary::new(tokens))
}

/// Load vocabulary from JSON file
fn load_json_vocab(path: &Path) -> Result<Vocabulary, PipelineError> {
    let file = File::open(path)
        .map_err(|e| PipelineError::Io(format!("Failed to open vocab file: {}", e)))?;

    let vocab_map: HashMap<String, u32> = serde_json::from_reader(file)
        .map_err(|e| PipelineError::Io(format!("Failed to parse vocab JSON: {}", e)))?;

    // Sort by ID to get ordered token list
    let mut pairs: Vec<_> = vocab_map.into_iter().collect();
    pairs.sort_by_key(|(_, id)| *id);

    let tokens: Vec<String> = pairs.into_iter().map(|(token, _)| token).collect();

    Ok(Vocabulary::new(tokens))
}

/// Load vocabulary from tokenizer.json (Hugging Face format)
fn load_tokenizer_json(path: &Path) -> Result<Vocabulary, PipelineError> {
    let file = File::open(path)
        .map_err(|e| PipelineError::Io(format!("Failed to open tokenizer file: {}", e)))?;

    let tokenizer: serde_json::Value = serde_json::from_reader(file)
        .map_err(|e| PipelineError::Io(format!("Failed to parse tokenizer JSON: {}", e)))?;

    // Extract vocab from model.vocab
    let vocab = tokenizer.get("model")
        .and_then(|m| m.get("vocab"))
        .ok_or_else(|| PipelineError::Io("Invalid tokenizer.json format".to_string()))?;

    let vocab_map: HashMap<String, u32> = serde_json::from_value(vocab.clone())
        .map_err(|e| PipelineError::Io(format!("Failed to parse vocab: {}", e)))?;

    let mut pairs: Vec<_> = vocab_map.into_iter().collect();
    pairs.sort_by_key(|(_, id)| *id);

    let tokens: Vec<String> = pairs.into_iter().map(|(token, _)| token).collect();

    Ok(Vocabulary::new(tokens))
}

/// Load vocabulary from text file (one token per line)
fn load_text_vocab(path: &Path) -> Result<Vocabulary, PipelineError> {
    let file = File::open(path)
        .map_err(|e| PipelineError::Io(format!("Failed to open vocab file: {}", e)))?;

    let reader = BufReader::new(file);
    let tokens: Vec<String> = reader.lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.is_empty())
        .collect();

    Ok(Vocabulary::new(tokens))
}

/// Load SentencePiece .vocab file
fn load_sentencepiece_vocab(path: &Path) -> Result<Vocabulary, PipelineError> {
    let file = File::open(path)
        .map_err(|e| PipelineError::Io(format!("Failed to open SentencePiece vocab: {}", e)))?;

    let reader = BufReader::new(file);
    let tokens: Vec<String> = reader.lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| {
            // Format: token\tscore
            line.split('\t').next().map(|s| s.to_string())
        })
        .collect();

    Ok(Vocabulary::new(tokens))
}

/// Load domain-specific vocabulary to boost
pub fn load_domain_vocab(path: &Path) -> Result<Vec<String>, PipelineError> {
    let file = File::open(path)
        .map_err(|e| PipelineError::Io(format!("Failed to open domain vocab: {}", e)))?;

    let reader = BufReader::new(file);
    let terms: Vec<String> = reader.lines()
        .filter_map(|line| line.ok())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();

    Ok(terms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let tokens = vec!["<blank>".to_string(), "<unk>".to_string(), "hello".to_string()];
        let vocab = Vocabulary::new(tokens);

        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.blank_id, 0);
        assert_eq!(vocab.unk_id, 1);
        assert_eq!(vocab.get_token(2), Some("hello"));
        assert_eq!(vocab.get_id("hello"), Some(2));
    }

    #[test]
    fn test_default_whisper_vocab() {
        let vocab = load_whisper_vocab(None).unwrap();
        assert_eq!(vocab.size(), 51865);
    }

    #[test]
    fn test_default_indicconformer_vocab() {
        let vocab = load_indicconformer_vocab(None).unwrap();
        assert_eq!(vocab.size(), 8000);
        // Check Devanagari characters are present
        assert!(vocab.get_id("क").is_some());
    }

    #[test]
    fn test_default_wav2vec2_vocab() {
        let vocab = load_wav2vec2_vocab(None).unwrap();
        assert!(vocab.size() > 30); // At least alphabet + special tokens
        assert!(vocab.get_id("a").is_some());
    }
}
