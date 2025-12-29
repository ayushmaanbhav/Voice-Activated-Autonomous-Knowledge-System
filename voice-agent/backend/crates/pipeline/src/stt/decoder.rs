//! Enhanced Decoder with Hinglish support
//!
//! Features:
//! - Code-switching aware beam search
//! - Indian English phoneme patterns
//! - Named entity boosting
//! - Stability-based partial emission

use std::collections::{HashMap, VecDeque};
use parking_lot::RwLock;

use crate::PipelineError;

/// Decoder configuration
#[derive(Debug, Clone)]
pub struct DecoderConfig {
    /// Beam width for search
    pub beam_width: usize,
    /// Language model weight
    pub lm_weight: f32,
    /// Word insertion penalty
    pub word_insertion_penalty: f32,
    /// Code-switching probability
    pub code_switch_prob: f32,
    /// Stability threshold for partial emission
    pub stability_threshold: f32,
    /// Stability window (frames)
    pub stability_window: usize,
    /// Enable named entity boosting
    pub entity_boosting: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            beam_width: 10,
            lm_weight: 0.3,
            word_insertion_penalty: 0.1,
            code_switch_prob: 0.3,
            stability_threshold: 0.8,
            stability_window: 5,
            entity_boosting: true,
        }
    }
}

/// Beam hypothesis (internal to decoder)
#[derive(Debug, Clone)]
struct Hypothesis {
    /// Token sequence
    tokens: Vec<u32>,
    /// Text so far
    text: String,
    /// Log probability
    log_prob: f32,
    /// Language state (for code-switching)
    language: Language,
    /// Stability count (consecutive frames with same top token)
    #[allow(dead_code)] // Reserved for future stability-based pruning
    stability: usize,
}

/// Language for code-switching
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    English,
    Hindi,
    Mixed,
}

/// Enhanced decoder with Hinglish support
pub struct EnhancedDecoder {
    config: DecoderConfig,
    /// Token vocabulary
    vocab: Vec<String>,
    /// Reverse vocab lookup (for constrained decoding)
    #[allow(dead_code)] // Reserved for future constrained decoding
    vocab_map: HashMap<String, u32>,
    /// Named entities to boost
    entities: RwLock<Vec<String>>,
    /// Current beam
    beam: RwLock<Vec<Hypothesis>>,
    /// Stable prefix (already emitted)
    stable_prefix: RwLock<String>,
    /// Frame history for stability
    /// P2 FIX: VecDeque for O(1) pop_front instead of Vec::remove(0)
    frame_history: RwLock<VecDeque<u32>>,
}

impl EnhancedDecoder {
    /// Create a new decoder with vocabulary
    pub fn new(vocab: Vec<String>, config: DecoderConfig) -> Self {
        let vocab_map: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        Self {
            config,
            vocab,
            vocab_map,
            entities: RwLock::new(Vec::new()),
            beam: RwLock::new(vec![Hypothesis {
                tokens: Vec::new(),
                text: String::new(),
                log_prob: 0.0,
                language: Language::English,
                stability: 0,
            }]),
            stable_prefix: RwLock::new(String::new()),
            frame_history: RwLock::new(VecDeque::new()),
        }
    }

    /// Create a simple decoder for testing
    pub fn simple(config: DecoderConfig) -> Self {
        Self::new(vec!["<blank>".to_string(), "<unk>".to_string()], config)
    }

    /// Add entities to boost
    pub fn add_entities(&self, entities: impl IntoIterator<Item = impl AsRef<str>>) {
        let mut ents = self.entities.write();
        for e in entities {
            ents.push(e.as_ref().to_lowercase());
        }
    }

    /// Process frame logits
    pub fn process_frame(&self, logits: &[f32]) -> Result<Option<String>, PipelineError> {
        let mut beam = self.beam.write();
        let mut frame_history = self.frame_history.write();

        // Get top-k tokens from logits
        let top_k = self.get_top_k(logits, self.config.beam_width * 2);

        // Expand beam
        let mut new_beam = Vec::with_capacity(self.config.beam_width * 2);

        for hyp in beam.iter() {
            for &(token_id, log_prob) in &top_k {
                let mut new_hyp = hyp.clone();
                new_hyp.log_prob += log_prob;

                // Skip blank token (CTC)
                if token_id == 0 {
                    new_beam.push(new_hyp);
                    continue;
                }

                // Skip repeat tokens
                if new_hyp.tokens.last() == Some(&token_id) {
                    new_beam.push(new_hyp);
                    continue;
                }

                // Add token
                new_hyp.tokens.push(token_id);
                if let Some(token_text) = self.vocab.get(token_id as usize) {
                    // Handle word pieces
                    if token_text.starts_with("##") || token_text.starts_with("▁") {
                        new_hyp.text.push_str(&token_text[2..]);
                    } else {
                        if !new_hyp.text.is_empty() && !new_hyp.text.ends_with(' ') {
                            new_hyp.text.push(' ');
                        }
                        new_hyp.text.push_str(token_text);
                    }

                    // Update language state for code-switching
                    new_hyp.language = self.detect_language(token_text);
                }

                // Apply entity boosting
                if self.config.entity_boosting {
                    new_hyp.log_prob += self.entity_boost(&new_hyp.text);
                }

                // Apply code-switching penalty/bonus
                new_hyp.log_prob += self.code_switch_score(&new_hyp, hyp.language);

                new_beam.push(new_hyp);
            }
        }

        // Prune beam
        new_beam.sort_by(|a, b| b.log_prob.partial_cmp(&a.log_prob).unwrap());
        new_beam.truncate(self.config.beam_width);

        // Update stability
        if let Some(best) = new_beam.first() {
            if let Some(&last_token) = best.tokens.last() {
                frame_history.push_back(last_token);
                if frame_history.len() > self.config.stability_window {
                    frame_history.pop_front(); // P2 FIX: O(1) instead of O(n)
                }
            }
        }

        *beam = new_beam;

        // Check for stable partial
        self.check_stable_partial()
    }

    /// Get top-k tokens from logits
    fn get_top_k(&self, logits: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut indexed: Vec<(u32, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &p)| (i as u32, p))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.truncate(k);

        // Convert to log probabilities
        let max = indexed.first().map(|x| x.1).unwrap_or(0.0);
        let sum: f32 = indexed.iter().map(|(_, p)| (p - max).exp()).sum();
        let log_sum = sum.ln() + max;

        indexed.iter()
            .map(|&(id, p)| (id, p - log_sum))
            .collect()
    }

    /// Detect language of token
    fn detect_language(&self, token: &str) -> Language {
        // Simple heuristic based on script
        let has_devanagari = token.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c));
        let has_latin = token.chars().any(|c| c.is_ascii_alphabetic());

        match (has_devanagari, has_latin) {
            (true, false) => Language::Hindi,
            (false, true) => Language::English,
            (true, true) => Language::Mixed,
            _ => Language::English,
        }
    }

    /// Calculate entity boost
    fn entity_boost(&self, text: &str) -> f32 {
        let entities = self.entities.read();
        let lower = text.to_lowercase();

        for entity in entities.iter() {
            if lower.ends_with(entity) {
                return 0.5; // Boost log prob
            }
            // Partial match
            for word in entity.split_whitespace() {
                if lower.ends_with(word) {
                    return 0.2;
                }
            }
        }

        0.0
    }

    /// Calculate code-switching score
    fn code_switch_score(&self, hyp: &Hypothesis, prev_lang: Language) -> f32 {
        if hyp.language == prev_lang || prev_lang == Language::Mixed {
            0.0
        } else {
            // Penalty for unexpected switch, bonus for expected
            if self.config.code_switch_prob > 0.5 {
                0.1 // Common code-switching, small bonus
            } else {
                -0.2 // Rare code-switching, penalty
            }
        }
    }

    /// Check for stable partial to emit
    fn check_stable_partial(&self) -> Result<Option<String>, PipelineError> {
        let beam = self.beam.read();
        let frame_history = self.frame_history.read();
        let mut stable_prefix = self.stable_prefix.write();

        if frame_history.len() < self.config.stability_window {
            return Ok(None);
        }

        // Check if last N frames agree
        let last = frame_history.back().copied(); // P2 FIX: VecDeque uses back() not last()
        let stable = frame_history.iter()
            .rev()
            .take(self.config.stability_window)
            .all(|&t| Some(t) == last);

        if !stable {
            return Ok(None);
        }

        // Get best hypothesis
        if let Some(best) = beam.first() {
            let new_text = &best.text;
            if new_text.len() > stable_prefix.len() {
                // Find stable boundary (last space before current position)
                let diff_start = stable_prefix.len();
                if let Some(space_pos) = new_text[..diff_start.max(1)].rfind(' ') {
                    let stable_end = space_pos + 1;
                    if stable_end > stable_prefix.len() {
                        let emission = new_text[stable_prefix.len()..stable_end].to_string();
                        *stable_prefix = new_text[..stable_end].to_string();
                        return Ok(Some(emission));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Finalize and get full text
    pub fn finalize(&self) -> String {
        let beam = self.beam.read();
        beam.first()
            .map(|h| h.text.clone())
            .unwrap_or_default()
    }

    /// Get current best hypothesis
    pub fn current_best(&self) -> String {
        let beam = self.beam.read();
        beam.first()
            .map(|h| h.text.clone())
            .unwrap_or_default()
    }

    /// Reset decoder state
    pub fn reset(&self) {
        let mut beam = self.beam.write();
        let mut stable_prefix = self.stable_prefix.write();
        let mut frame_history = self.frame_history.write();

        *beam = vec![Hypothesis {
            tokens: Vec::new(),
            text: String::new(),
            log_prob: 0.0,
            language: Language::English,
            stability: 0,
        }];
        stable_prefix.clear();
        frame_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let vocab = vec!["<blank>".to_string(), "hello".to_string(), "world".to_string()];
        let decoder = EnhancedDecoder::new(vocab, DecoderConfig::default());
        assert!(decoder.current_best().is_empty());
    }

    #[test]
    fn test_entity_boosting() {
        let decoder = EnhancedDecoder::simple(DecoderConfig::default());
        decoder.add_entities(["Kotak Mahindra", "gold loan"]);

        let boost = decoder.entity_boost("I want a gold loan");
        assert!(boost > 0.0);
    }

    #[test]
    fn test_reset() {
        let decoder = EnhancedDecoder::simple(DecoderConfig::default());
        decoder.reset();
        assert!(decoder.current_best().is_empty());
    }
}
