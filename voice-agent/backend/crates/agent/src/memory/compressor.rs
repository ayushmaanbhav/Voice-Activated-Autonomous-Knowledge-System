//! RECOMP-Style Extractive Context Compressor
//!
//! Implements an extractive context compression algorithm inspired by:
//! - RECOMP (arXiv:2310.04408): Sentence-level selection for RAG compression
//! - MemGPT (arXiv:2310.08560): Hierarchical memory with virtual context
//!
//! Key features:
//! - Sentence scoring by entity density, intent relevance, recency
//! - Token-budget-aware selection
//! - DST state integration for small model context efficiency
//!
//! For Qwen2.5:1.5B and similar small models, this compressor provides
//! a reliable alternative to LLM-based summarization.

use super::recall::{ConversationTurn, TurnRole};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use unicode_segmentation::UnicodeSegmentation;

/// Configuration for extractive compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractiveCompressorConfig {
    /// Minimum score threshold for sentence inclusion (0.0-1.0)
    pub min_sentence_score: f32,
    /// Maximum sentences to include in compressed output
    pub max_sentences: usize,
    /// Score boost for sentences containing domain entities
    pub entity_boost: f32,
    /// Score boost for sentences matching detected intents
    pub intent_boost: f32,
    /// Decay factor for older turns (applied per turn back)
    pub recency_decay: f32,
    /// Maximum tokens for compressed output
    pub max_tokens: usize,
    /// Include DST state summary at the beginning
    pub include_dst_summary: bool,
    /// Priority entities for gold loan domain
    pub priority_entities: Vec<String>,
}

impl Default for ExtractiveCompressorConfig {
    fn default() -> Self {
        Self {
            min_sentence_score: 0.25,
            max_sentences: 12,
            entity_boost: 2.0,
            intent_boost: 1.5,
            recency_decay: 0.92,
            max_tokens: 800,
            include_dst_summary: true,
            // P16 FIX: Support both domain-specific and generic slot names
            priority_entities: vec![
                // Generic slot names (domain-agnostic)
                "asset_quantity".to_string(),
                "asset_quality".to_string(),
                "amount".to_string(),
                "customer_name".to_string(),
                "phone_number".to_string(),
                "competitor".to_string(),
                "interest_rate".to_string(),
                "preferred_branch".to_string(),
                // Domain-specific aliases (gold loan)
                "gold_weight".to_string(),
                "loan_amount".to_string(),
                "gold_purity".to_string(),
                "current_lender".to_string(),
            ],
        }
    }
}

/// A scored sentence for compression selection
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    /// The sentence content
    pub content: String,
    /// Source turn index (for ordering)
    pub turn_index: usize,
    /// Sentence index within turn
    pub sentence_index: usize,
    /// Role of the speaker
    pub role: TurnRole,
    /// Computed relevance score
    pub score: f32,
    /// Entities found in this sentence
    pub entities: Vec<String>,
    /// Estimated token count
    pub tokens: usize,
}

/// Compression statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ExtractionStats {
    /// Total sentences processed
    pub total_sentences: usize,
    /// Sentences selected
    pub selected_sentences: usize,
    /// Original token count
    pub original_tokens: usize,
    /// Compressed token count
    pub compressed_tokens: usize,
    /// Compression ratio (original / compressed)
    pub compression_ratio: f32,
    /// Entities preserved
    pub entities_preserved: usize,
    /// Entities in original
    pub entities_total: usize,
}

impl ExtractionStats {
    /// Check if compression achieved target ratio (>= 2.0x)
    pub fn is_effective(&self) -> bool {
        self.compression_ratio >= 2.0
    }

    /// Calculate entity preservation rate
    pub fn entity_preservation_rate(&self) -> f32 {
        if self.entities_total == 0 {
            1.0
        } else {
            self.entities_preserved as f32 / self.entities_total as f32
        }
    }
}

/// RECOMP-style extractive context compressor
///
/// This compressor selects the most relevant sentences from conversation
/// history based on entity density, intent relevance, and recency.
pub struct ExtractiveCompressor {
    config: ExtractiveCompressorConfig,
    /// Domain-specific keywords for scoring
    domain_keywords: HashSet<String>,
    /// Entity patterns for detection
    entity_patterns: HashMap<String, Vec<String>>,
}

impl Default for ExtractiveCompressor {
    fn default() -> Self {
        Self::new(ExtractiveCompressorConfig::default())
    }
}

impl ExtractiveCompressor {
    /// Create a new extractive compressor with configuration
    pub fn new(config: ExtractiveCompressorConfig) -> Self {
        let mut domain_keywords = HashSet::new();
        // Gold loan domain keywords (English + Hindi/Hinglish)
        for kw in &[
            "gold", "loan", "rate", "interest", "emi", "branch", "weight",
            "gram", "grams", "tola", "purity", "karat", "22k", "24k", "18k",
            "sona", "karj", "byaj", "rin", "gehne", "jewelry",
            "lakh", "crore", "rupees", "amount", "kotak", "muthoot", "manappuram",
            "eligibility", "document", "disbursal", "repayment", "tenure",
        ] {
            domain_keywords.insert(kw.to_string());
        }

        // P16 FIX: Entity patterns use generic names, with domain-specific aliases
        let mut entity_patterns = HashMap::new();
        // Generic: asset_quantity (gold_weight is a domain alias)
        entity_patterns.insert(
            "asset_quantity".to_string(),
            vec!["gram".to_string(), "grams".to_string(), "gm".to_string(), "tola".to_string()],
        );
        // Generic: amount (loan_amount is a domain alias)
        entity_patterns.insert(
            "amount".to_string(),
            vec!["lakh".to_string(), "crore".to_string(), "rupees".to_string(), "rs".to_string()],
        );
        // Generic: asset_quality (gold_purity is a domain alias)
        entity_patterns.insert(
            "asset_quality".to_string(),
            vec!["22k".to_string(), "24k".to_string(), "18k".to_string(), "karat".to_string()],
        );
        // Generic: competitor (current_lender is a domain alias)
        entity_patterns.insert(
            "competitor".to_string(),
            vec!["muthoot".to_string(), "manappuram".to_string(), "iifl".to_string(), "hdfc".to_string(), "sbi".to_string()],
        );

        Self {
            config,
            domain_keywords,
            entity_patterns,
        }
    }

    /// Compress conversation turns into a concise context string
    ///
    /// Returns compressed context and extraction statistics
    pub fn compress(
        &self,
        turns: &[ConversationTurn],
        dst_state: Option<&str>,
    ) -> (String, ExtractionStats) {
        if turns.is_empty() {
            return (String::new(), ExtractionStats::default());
        }

        // 1. Extract and score all sentences
        let mut scored_sentences = self.extract_and_score_sentences(turns);

        // 2. Sort by score (descending)
        scored_sentences.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Select sentences within token budget
        let mut selected = Vec::new();
        let mut token_budget = self.config.max_tokens;

        // Reserve tokens for DST state if enabled
        let dst_summary = if self.config.include_dst_summary {
            dst_state.map(|s| {
                let summary = format!("[State] {}", s);
                let tokens = estimate_tokens(&summary);
                token_budget = token_budget.saturating_sub(tokens + 10); // +10 for formatting
                summary
            })
        } else {
            None
        };

        // Select top sentences within budget
        let mut seen_entities: HashSet<String> = HashSet::new();
        for sentence in &scored_sentences {
            if selected.len() >= self.config.max_sentences {
                break;
            }
            if sentence.tokens > token_budget {
                continue;
            }
            if sentence.score < self.config.min_sentence_score {
                break; // Sorted, so no more will pass
            }

            // Track entities
            for entity in &sentence.entities {
                seen_entities.insert(entity.clone());
            }

            token_budget = token_budget.saturating_sub(sentence.tokens);
            selected.push(sentence.clone());
        }

        // 4. Re-sort selected by temporal order (turn_index, sentence_index)
        selected.sort_by(|a, b| {
            a.turn_index
                .cmp(&b.turn_index)
                .then(a.sentence_index.cmp(&b.sentence_index))
        });

        // 5. Build compressed output
        let mut result = String::new();

        if let Some(dst) = dst_summary {
            result.push_str(&dst);
            result.push('\n');
        }

        if !selected.is_empty() {
            result.push_str("[History] ");
            let sentences: Vec<String> = selected
                .iter()
                .map(|s| {
                    let role_prefix = match s.role {
                        TurnRole::User => "U:",
                        TurnRole::Assistant => "A:",
                        TurnRole::System => "S:",
                    };
                    format!("{} {}", role_prefix, s.content.trim())
                })
                .collect();
            result.push_str(&sentences.join(" | "));
        }

        // 6. Calculate statistics
        let original_tokens: usize = turns.iter().map(|t| t.estimated_tokens).sum();
        let compressed_tokens = estimate_tokens(&result);
        let total_entities: usize = turns.iter().map(|t| t.entities.len()).sum();

        let stats = ExtractionStats {
            total_sentences: scored_sentences.len(),
            selected_sentences: selected.len(),
            original_tokens,
            compressed_tokens,
            compression_ratio: if compressed_tokens > 0 {
                original_tokens as f32 / compressed_tokens as f32
            } else {
                0.0
            },
            entities_preserved: seen_entities.len(),
            entities_total: total_entities,
        };

        (result, stats)
    }

    /// Extract sentences from turns and score them
    fn extract_and_score_sentences(&self, turns: &[ConversationTurn]) -> Vec<ScoredSentence> {
        let mut sentences = Vec::new();
        let total_turns = turns.len();

        for (turn_idx, turn) in turns.iter().enumerate() {
            // Calculate recency factor (more recent = higher score)
            let turns_back = total_turns.saturating_sub(turn_idx + 1);
            let recency_factor = self.config.recency_decay.powi(turns_back as i32);

            // Split turn content into sentences
            let turn_sentences = self.split_into_sentences(&turn.content);

            for (sent_idx, sentence) in turn_sentences.into_iter().enumerate() {
                if sentence.trim().is_empty() {
                    continue;
                }

                // Score the sentence
                let (score, entities) = self.score_sentence(
                    &sentence,
                    &turn.entities,
                    &turn.intents,
                    recency_factor,
                );

                sentences.push(ScoredSentence {
                    content: sentence.clone(),
                    turn_index: turn_idx,
                    sentence_index: sent_idx,
                    role: turn.role.clone(),
                    score,
                    entities,
                    tokens: estimate_tokens(&sentence),
                });
            }
        }

        sentences
    }

    /// Score a sentence based on multiple factors
    fn score_sentence(
        &self,
        sentence: &str,
        turn_entities: &[(String, String)],
        turn_intents: &[String],
        recency_factor: f32,
    ) -> (f32, Vec<String>) {
        let sentence_lower = sentence.to_lowercase();
        let mut score = 0.3; // Base score
        let mut found_entities = Vec::new();

        // 1. Entity density scoring
        for (entity_key, entity_value) in turn_entities {
            // Check if entity value appears in sentence
            if sentence_lower.contains(&entity_value.to_lowercase()) {
                score += self.config.entity_boost;
                found_entities.push(entity_key.clone());
            }
        }

        // 2. Check for entity patterns in sentence (even without explicit entities)
        for (entity_type, patterns) in &self.entity_patterns {
            for pattern in patterns {
                if sentence_lower.contains(pattern) && !found_entities.contains(entity_type) {
                    score += self.config.entity_boost * 0.5; // Lower boost for pattern match
                    found_entities.push(entity_type.clone());
                    break;
                }
            }
        }

        // 3. Priority entity bonus
        for entity in &found_entities {
            if self.config.priority_entities.contains(entity) {
                score += 0.5;
            }
        }

        // 4. Intent relevance scoring
        for intent in turn_intents {
            let intent_lower = intent.to_lowercase();
            // Check if intent keywords appear in sentence
            if sentence_lower.contains(&intent_lower) ||
               self.intent_matches_sentence(&intent_lower, &sentence_lower) {
                score += self.config.intent_boost;
            }
        }

        // 5. Domain keyword density
        let words: Vec<&str> = sentence_lower.split_whitespace().collect();
        let domain_word_count = words.iter()
            .filter(|w| self.domain_keywords.contains(**w))
            .count();
        if words.len() > 0 {
            let density = domain_word_count as f32 / words.len() as f32;
            score += density * 0.5;
        }

        // 6. Information density (unique words / total words)
        let unique_words: HashSet<&str> = words.iter().cloned().collect();
        if words.len() > 0 {
            let uniqueness = unique_words.len() as f32 / words.len() as f32;
            score += uniqueness * 0.2;
        }

        // 7. Penalize very short sentences (likely greetings/fillers)
        if words.len() < 4 {
            score *= 0.5;
        }

        // 8. Penalize common filler patterns
        if is_filler_sentence(&sentence_lower) {
            score *= 0.3;
        }

        // Apply recency factor
        score *= recency_factor;

        // Normalize to [0, 1] range
        score = score.clamp(0.0, 1.0);

        (score, found_entities)
    }

    /// Check if intent keywords match the sentence
    fn intent_matches_sentence(&self, intent: &str, sentence: &str) -> bool {
        // Map intents to related keywords
        let intent_keywords: HashMap<&str, Vec<&str>> = [
            ("rate_inquiry", vec!["rate", "interest", "byaj", "percent"]),
            ("loan_inquiry", vec!["loan", "borrow", "karj", "rin"]),
            ("eligibility", vec!["eligible", "qualify", "can i get"]),
            ("branch", vec!["branch", "location", "nearby", "visit"]),
            ("competitor", vec!["muthoot", "manappuram", "better", "compare"]),
            ("amount", vec!["how much", "kitna", "amount", "value"]),
        ].into_iter().collect();

        for (key, keywords) in intent_keywords.iter() {
            if intent.contains(key) {
                return keywords.iter().any(|kw| sentence.contains(kw));
            }
        }

        false
    }

    /// Split text into sentences (handles Hindi and English)
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        // Split on sentence boundaries
        for c in text.chars() {
            current.push(c);
            if c == '.' || c == '?' || c == '!' || c == '।' || c == '\n' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() && trimmed.len() > 2 {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        // Add remaining text
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() && trimmed.len() > 5 {
            sentences.push(trimmed);
        }

        // If no sentences found, treat whole text as one sentence
        if sentences.is_empty() && !text.trim().is_empty() {
            sentences.push(text.trim().to_string());
        }

        sentences
    }

    /// Compress with DST state from DynamicDialogueState
    pub fn compress_with_dst_slots(
        &self,
        turns: &[ConversationTurn],
        slots: &HashMap<String, String>,
    ) -> (String, ExtractionStats) {
        // Format DST slots as summary string
        let dst_summary = if !slots.is_empty() {
            let slot_strings: Vec<String> = slots
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .map(|(k, v)| {
                    // P16 FIX: Support both generic and domain-specific slot names
                    let display_key = match k.as_str() {
                        // Generic names first, then domain-specific aliases
                        "asset_quantity" | "gold_weight" | "weight" => "Asset",
                        "amount" | "loan_amount" => "Amount",
                        "asset_quality" | "gold_purity" | "purity" => "Quality",
                        "customer_name" | "name" => "Name",
                        "phone_number" | "phone" => "Phone",
                        "competitor" | "current_lender" => "Competitor",
                        "interest_rate" | "rate" => "Rate",
                        "preferred_branch" | "branch" => "Branch",
                        "urgency" => "Urgency",
                        "purpose" | "loan_purpose" => "Purpose",
                        _ => k.as_str(),
                    };
                    format!("{}={}", display_key, v)
                })
                .collect();
            Some(slot_strings.join(" | "))
        } else {
            None
        };

        self.compress(turns, dst_summary.as_deref())
    }
}

/// Estimate token count for text (handles Hindi/Devanagari)
fn estimate_tokens(text: &str) -> usize {
    let grapheme_count = text.graphemes(true).count();

    // Check for Devanagari script
    let devanagari_count = text.chars()
        .filter(|c| ('\u{0900}'..='\u{097F}').contains(c))
        .count();

    if devanagari_count > grapheme_count / 3 {
        // Devanagari-heavy text: roughly 1 token per 2 graphemes
        grapheme_count.max(1) / 2
    } else {
        // English-heavy text: roughly 1 token per 4 characters
        grapheme_count.max(1) / 4
    }
}

/// Check if sentence is a common filler/greeting
fn is_filler_sentence(sentence: &str) -> bool {
    let filler_patterns = [
        "hello", "hi", "namaste", "good morning", "good afternoon",
        "thank you", "thanks", "dhanyavaad", "okay", "ok", "hmm",
        "yes", "no", "haan", "nahi", "ji", "achha", "theek hai",
        "i see", "understood", "sure", "alright", "great",
    ];

    let sentence_trimmed = sentence.trim();

    // Check if sentence is just a filler
    for pattern in &filler_patterns {
        if sentence_trimmed == *pattern || sentence_trimmed.starts_with(pattern) && sentence_trimmed.len() < pattern.len() + 5 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_turn(role: TurnRole, content: &str) -> ConversationTurn {
        let mut turn = ConversationTurn::new(role, content);
        turn.estimated_tokens = estimate_tokens(content);
        turn
    }

    fn create_turn_with_entities(
        role: TurnRole,
        content: &str,
        entities: Vec<(&str, &str)>,
    ) -> ConversationTurn {
        let mut turn = create_test_turn(role, content);
        turn.entities = entities
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        turn
    }

    #[test]
    fn test_extractive_compressor_basic() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "I want a gold loan."),
            create_test_turn(TurnRole::Assistant, "Sure, how much gold do you have?"),
            create_test_turn(TurnRole::User, "I have 50 grams of 22K gold."),
        ];

        let (compressed, stats) = compressor.compress(&turns, None);

        assert!(!compressed.is_empty());
        assert!(stats.selected_sentences > 0);
        assert!(stats.compression_ratio > 0.0);
    }

    #[test]
    fn test_extractive_compressor_entity_priority() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "Hello, good morning."),
            create_turn_with_entities(
                TurnRole::User,
                "I have 50 grams of gold and need 5 lakh loan.",
                vec![("gold_weight", "50 grams"), ("loan_amount", "5 lakh")],
            ),
            create_test_turn(TurnRole::Assistant, "Thank you for sharing."),
        ];

        let (compressed, stats) = compressor.compress(&turns, None);

        // Should prioritize the sentence with entities
        assert!(compressed.contains("50") || compressed.contains("gram") || compressed.contains("lakh"));
        assert!(stats.entities_preserved > 0);
    }

    #[test]
    fn test_extractive_compressor_recency_bias() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "I want information about gold loans."),
            create_test_turn(TurnRole::Assistant, "Gold loans are secured loans."),
            create_test_turn(TurnRole::User, "What is the interest rate?"),
            create_test_turn(TurnRole::Assistant, "Our rate starts at 10.5% per annum."),
        ];

        let (compressed, stats) = compressor.compress(&turns, None);

        // More recent turns should be included
        assert!(compressed.contains("rate") || compressed.contains("10.5"));
        assert!(stats.selected_sentences >= 2);
    }

    #[test]
    fn test_extractive_compressor_token_budget() {
        let mut config = ExtractiveCompressorConfig::default();
        config.max_tokens = 50; // Very small budget

        let compressor = ExtractiveCompressor::new(config);

        let turns = vec![
            create_test_turn(TurnRole::User, "I want a gold loan for my business expansion."),
            create_test_turn(TurnRole::Assistant, "Sure, I can help you with that."),
            create_test_turn(TurnRole::User, "I have about 100 grams of 22 karat gold jewelry."),
            create_test_turn(TurnRole::Assistant, "Great, that would give you a good loan amount."),
        ];

        let (compressed, stats) = compressor.compress(&turns, None);

        // Should stay within token budget
        assert!(stats.compressed_tokens <= 100); // Some overhead allowed
    }

    #[test]
    fn test_extractive_compressor_dst_integration() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "I need a loan urgently."),
        ];

        let dst_state = "Name=Rajesh | Gold=50g | Amount=5L";
        let (compressed, _stats) = compressor.compress(&turns, Some(dst_state));

        // DST state should be at the beginning
        assert!(compressed.starts_with("[State]"));
        assert!(compressed.contains("Rajesh"));
        assert!(compressed.contains("50g"));
    }

    #[test]
    fn test_extractive_compressor_hindi_support() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "Mujhe gold loan chahiye."),
            create_test_turn(TurnRole::User, "Mere paas 50 gram sona hai."),
        ];

        let (compressed, stats) = compressor.compress(&turns, None);

        // Should handle Hindi content
        assert!(!compressed.is_empty());
        assert!(stats.total_sentences >= 1);
    }

    #[test]
    fn test_extractive_compressor_filler_filtering() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "Hello"),
            create_test_turn(TurnRole::Assistant, "Hi"),
            create_test_turn(TurnRole::User, "I have 50 grams of gold and need a loan of 5 lakh."),
            create_test_turn(TurnRole::Assistant, "Thank you for the information."),
        ];

        let (compressed, _stats) = compressor.compress(&turns, None);

        // Filler sentences should be deprioritized
        // The informative sentence about gold should be present
        assert!(compressed.contains("50") || compressed.contains("gold") || compressed.contains("lakh"));
    }

    #[test]
    fn test_extractive_compressor_compression_ratio() {
        let compressor = ExtractiveCompressor::default();

        // Create a longer conversation
        let mut turns = Vec::new();
        for i in 0..10 {
            turns.push(create_test_turn(
                TurnRole::User,
                &format!("User message number {} with some content about gold loan inquiry.", i),
            ));
            turns.push(create_test_turn(
                TurnRole::Assistant,
                &format!("Assistant response {} with detailed information about rates and eligibility.", i),
            ));
        }

        let (_compressed, stats) = compressor.compress(&turns, None);

        // Should achieve meaningful compression
        assert!(stats.compression_ratio >= 1.5, "Expected ratio >= 1.5, got {}", stats.compression_ratio);
    }

    #[test]
    fn test_compress_with_dst_slots() {
        let compressor = ExtractiveCompressor::default();

        let turns = vec![
            create_test_turn(TurnRole::User, "I need a loan."),
        ];

        let mut slots = HashMap::new();
        slots.insert("customer_name".to_string(), "Rahul".to_string());
        slots.insert("gold_weight".to_string(), "75 grams".to_string());
        slots.insert("loan_amount".to_string(), "4 lakh".to_string());

        let (compressed, _stats) = compressor.compress_with_dst_slots(&turns, &slots);

        assert!(compressed.contains("[State]"));
        assert!(compressed.contains("Rahul") || compressed.contains("Name"));
        assert!(compressed.contains("Gold") || compressed.contains("75"));
    }

    #[test]
    fn test_extraction_stats() {
        let stats = ExtractionStats {
            total_sentences: 20,
            selected_sentences: 5,
            original_tokens: 500,
            compressed_tokens: 100,
            compression_ratio: 5.0,
            entities_preserved: 4,
            entities_total: 5,
        };

        assert!(stats.is_effective());
        assert!((stats.entity_preservation_rate() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_estimate_tokens_english() {
        let english = "This is a test sentence for token estimation.";
        let tokens = estimate_tokens(english);
        // ~11 words, expect roughly 11-15 tokens
        assert!(tokens >= 8 && tokens <= 20);
    }

    #[test]
    fn test_estimate_tokens_hindi() {
        let hindi = "मुझे गोल्ड लोन चाहिए पचास ग्राम सोने पर।";
        let tokens = estimate_tokens(hindi);
        // Hindi text should be estimated appropriately
        assert!(tokens >= 5);
    }

    #[test]
    fn test_is_filler_sentence() {
        assert!(is_filler_sentence("hello"));
        assert!(is_filler_sentence("thank you"));
        assert!(is_filler_sentence("ok"));
        assert!(!is_filler_sentence("I need a gold loan of 5 lakh"));
        assert!(!is_filler_sentence("What is the interest rate?"));
    }
}
