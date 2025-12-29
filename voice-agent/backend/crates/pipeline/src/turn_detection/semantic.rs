//! Semantic Turn Detector
//!
//! Uses a lightweight transformer to classify utterance completeness.
//! Trained on Indian English + Hindi patterns.

use parking_lot::Mutex;
use std::collections::VecDeque;
use std::path::Path;

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;

use crate::PipelineError;

/// Semantic completeness classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompletenessClass {
    /// Definitely incomplete (mid-sentence)
    Incomplete,
    /// Possibly complete (short pause ok)
    PossiblyComplete,
    /// Definitely complete (can respond)
    Complete,
    /// Question detected (shorter wait)
    Question,
    /// Backchanneling (don't interrupt)
    Backchannel,
}

impl CompletenessClass {
    /// Suggested silence threshold in milliseconds
    pub fn suggested_silence_ms(&self) -> u32 {
        match self {
            CompletenessClass::Incomplete => 800,
            CompletenessClass::PossiblyComplete => 500,
            CompletenessClass::Complete => 300,
            CompletenessClass::Question => 250,
            CompletenessClass::Backchannel => 1000,
        }
    }

    /// Confidence threshold for this class
    pub fn confidence_threshold(&self) -> f32 {
        match self {
            CompletenessClass::Incomplete => 0.7,
            CompletenessClass::PossiblyComplete => 0.5,
            CompletenessClass::Complete => 0.8,
            CompletenessClass::Question => 0.75,
            CompletenessClass::Backchannel => 0.9,
        }
    }
}

/// Configuration for semantic turn detection
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Confidence threshold for classification
    pub confidence_threshold: f32,
    /// Enable Hindi/Hinglish patterns
    pub hindi_patterns: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 64,
            confidence_threshold: 0.7,
            hindi_patterns: true,
        }
    }
}

/// Semantic Turn Detector using lightweight transformer
pub struct SemanticTurnDetector {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(feature = "onnx")]
    tokenizer: Tokenizer,
    #[allow(dead_code)] // Used by model_classify() when ONNX enabled
    config: SemanticConfig,
    /// Cache recent predictions for smoothing
    /// P2 FIX: VecDeque for O(1) pop_front instead of Vec::remove(0)
    prediction_cache: Mutex<VecDeque<(CompletenessClass, f32)>>,
}

impl SemanticTurnDetector {
    /// Create a new semantic turn detector
    #[cfg(feature = "onnx")]
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: SemanticConfig,
    ) -> Result<Self, PipelineError> {
        let session = Session::builder()
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| PipelineError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            config,
            prediction_cache: Mutex::new(VecDeque::with_capacity(10)),
        })
    }

    /// Create a new semantic turn detector (no ONNX - rule-based only)
    #[cfg(not(feature = "onnx"))]
    pub fn new(
        _model_path: impl AsRef<Path>,
        _tokenizer_path: impl AsRef<Path>,
        config: SemanticConfig,
    ) -> Result<Self, PipelineError> {
        Self::simple(config)
    }

    /// Create a simple rule-based detector (no model required, only when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn simple(config: SemanticConfig) -> Result<Self, PipelineError> {
        Ok(Self {
            config,
            prediction_cache: Mutex::new(VecDeque::with_capacity(10)),
        })
    }

    /// Create a simple rule-based detector (ONNX enabled - returns error)
    #[cfg(feature = "onnx")]
    pub fn simple(_config: SemanticConfig) -> Result<Self, PipelineError> {
        Err(PipelineError::Model(
            "SemanticTurnDetector::simple() is not available when ONNX feature is enabled. Use new() instead.".to_string()
        ))
    }

    /// Classify utterance completeness
    pub fn classify(&self, text: &str) -> Result<(CompletenessClass, f32), PipelineError> {
        // Quick heuristic checks first
        if let Some(result) = self.quick_classify(text) {
            return Ok(result);
        }

        // Fall back to model inference (or return possibly complete for rule-based)
        #[cfg(feature = "onnx")]
        {
            self.model_classify(text)
        }
        #[cfg(not(feature = "onnx"))]
        {
            Ok((CompletenessClass::PossiblyComplete, 0.5))
        }
    }

    /// Quick rule-based classification for obvious cases
    fn quick_classify(&self, text: &str) -> Option<(CompletenessClass, f32)> {
        let trimmed = text.trim();

        if trimmed.is_empty() {
            return Some((CompletenessClass::Incomplete, 1.0));
        }

        // Question detection
        if trimmed.ends_with('?') {
            return Some((CompletenessClass::Question, 0.95));
        }

        // Hindi question markers
        let hindi_question_markers = ["kya", "kaise", "kyun", "kab", "kahan", "kitna", "kaun"];
        let lower = trimmed.to_lowercase();
        for marker in &hindi_question_markers {
            if lower.starts_with(marker) || lower.contains(&format!(" {} ", marker)) {
                return Some((CompletenessClass::Question, 0.85));
            }
        }

        // Backchannel patterns
        let backchannels = [
            "hmm", "haan", "achha", "theek hai", "ok", "okay", "yes", "no",
            "ji", "sahi", "bilkul", "samajh gaya", "samajh gayi",
        ];
        for bc in &backchannels {
            if lower == *bc || lower.starts_with(&format!("{} ", bc)) {
                return Some((CompletenessClass::Backchannel, 0.9));
            }
        }

        // Incomplete sentence markers (conjunctions, etc.)
        let incomplete_markers = [
            "aur", "lekin", "par", "toh", "ki", "jo", "jab", "agar",
            "and", "but", "so", "that", "which", "when", "if",
        ];
        for marker in &incomplete_markers {
            if lower.ends_with(&format!(" {}", marker)) {
                return Some((CompletenessClass::Incomplete, 0.85));
            }
        }

        // Complete sentence markers
        if trimmed.ends_with('.') || trimmed.ends_with('!') {
            return Some((CompletenessClass::Complete, 0.8));
        }

        None
    }

    /// Model-based classification
    #[cfg(feature = "onnx")]
    fn model_classify(&self, text: &str) -> Result<(CompletenessClass, f32), PipelineError> {
        // Tokenize
        let encoding = self.tokenizer
            .encode(text, true)
            .map_err(|e| PipelineError::TurnDetection(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids()
            .iter()
            .take(self.config.max_seq_len)
            .map(|&id| id as i64)
            .collect();

        let attention_mask: Vec<i64> = vec![1i64; ids.len()];

        // Pad to max_seq_len
        let mut padded_ids = vec![0i64; self.config.max_seq_len];
        let mut padded_mask = vec![0i64; self.config.max_seq_len];

        padded_ids[..ids.len()].copy_from_slice(&ids);
        padded_mask[..attention_mask.len()].copy_from_slice(&attention_mask);

        // Create tensors
        let input_ids = Array2::from_shape_vec((1, self.config.max_seq_len), padded_ids)
            .map_err(|e| PipelineError::TurnDetection(e.to_string()))?;
        let attention = Array2::from_shape_vec((1, self.config.max_seq_len), padded_mask)
            .map_err(|e| PipelineError::TurnDetection(e.to_string()))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids.view(),
            "attention_mask" => attention.view(),
        ].map_err(|e| PipelineError::Model(e.to_string()))?)
        .map_err(|e| PipelineError::Model(e.to_string()))?;

        // Extract logits [batch, num_classes]
        let logits = outputs
            .get("logits")
            .ok_or_else(|| PipelineError::Model("Missing logits output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| PipelineError::Model(e.to_string()))?;

        // Softmax and argmax
        let logits_view = logits.view();
        let probs: Vec<f32> = logits_view.iter().map(|&x| x.exp()).collect();
        let sum: f32 = probs.iter().sum();
        let probs: Vec<f32> = probs.iter().map(|&x| x / sum).collect();

        let (max_idx, &max_prob) = probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let class = match max_idx {
            0 => CompletenessClass::Incomplete,
            1 => CompletenessClass::PossiblyComplete,
            2 => CompletenessClass::Complete,
            3 => CompletenessClass::Question,
            4 => CompletenessClass::Backchannel,
            _ => CompletenessClass::PossiblyComplete,
        };

        // Update cache for smoothing
        let mut cache = self.prediction_cache.lock();
        cache.push_back((class, max_prob));
        if cache.len() > 5 {
            cache.pop_front(); // P2 FIX: O(1) instead of O(n)
        }

        Ok((class, max_prob))
    }

    /// Get smoothed prediction from cache
    pub fn smoothed_prediction(&self) -> Option<(CompletenessClass, f32)> {
        let cache = self.prediction_cache.lock();
        if cache.is_empty() {
            return None;
        }

        // Count votes for each class
        let mut votes = [0u32; 5];
        let mut total_conf = [0.0f32; 5];

        for (class, conf) in cache.iter() {
            let idx = match class {
                CompletenessClass::Incomplete => 0,
                CompletenessClass::PossiblyComplete => 1,
                CompletenessClass::Complete => 2,
                CompletenessClass::Question => 3,
                CompletenessClass::Backchannel => 4,
            };
            votes[idx] += 1;
            total_conf[idx] += conf;
        }

        let (max_idx, &max_votes) = votes.iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .unwrap();

        let avg_conf = if max_votes > 0 {
            total_conf[max_idx] / max_votes as f32
        } else {
            0.0
        };

        let class = match max_idx {
            0 => CompletenessClass::Incomplete,
            1 => CompletenessClass::PossiblyComplete,
            2 => CompletenessClass::Complete,
            3 => CompletenessClass::Question,
            4 => CompletenessClass::Backchannel,
            _ => CompletenessClass::PossiblyComplete,
        };

        Some((class, avg_conf))
    }

    /// Reset prediction cache
    pub fn reset(&self) {
        self.prediction_cache.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_classify_question() {
        let detector = SemanticTurnDetector::simple(SemanticConfig::default()).unwrap();

        let (class, conf) = detector.classify("What is the interest rate?").unwrap();
        assert_eq!(class, CompletenessClass::Question);
        assert!(conf > 0.9);
    }

    #[test]
    fn test_quick_classify_hindi_question() {
        let detector = SemanticTurnDetector::simple(SemanticConfig::default()).unwrap();

        let (class, _) = detector.classify("kya interest rate kam ho sakta hai").unwrap();
        assert_eq!(class, CompletenessClass::Question);
    }

    #[test]
    fn test_quick_classify_backchannel() {
        let detector = SemanticTurnDetector::simple(SemanticConfig::default()).unwrap();

        let (class, _) = detector.classify("achha").unwrap();
        assert_eq!(class, CompletenessClass::Backchannel);
    }

    #[test]
    fn test_quick_classify_incomplete() {
        let detector = SemanticTurnDetector::simple(SemanticConfig::default()).unwrap();

        let (class, _) = detector.classify("I was thinking that").unwrap();
        assert_eq!(class, CompletenessClass::Incomplete);
    }
}
