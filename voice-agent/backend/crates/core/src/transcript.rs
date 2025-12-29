//! Transcript types for STT output

use serde::{Deserialize, Serialize};

/// Transcript result from STT
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptResult {
    /// Transcribed text
    pub text: String,

    /// Is this a final result?
    pub is_final: bool,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,

    /// Start time offset (ms from stream start)
    pub start_time_ms: u64,

    /// End time offset (ms from stream start)
    pub end_time_ms: u64,

    /// Detected language (ISO 639-1 code)
    pub language: Option<String>,

    /// Word-level timestamps
    pub words: Vec<WordTimestamp>,
}

impl TranscriptResult {
    /// Create a new transcript result
    pub fn new(text: String, is_final: bool, confidence: f32) -> Self {
        Self {
            text,
            is_final,
            confidence,
            start_time_ms: 0,
            end_time_ms: 0,
            language: None,
            words: Vec::new(),
        }
    }

    /// Create a partial (non-final) transcript
    pub fn partial(text: String, confidence: f32) -> Self {
        Self::new(text, false, confidence)
    }

    /// Create a final transcript
    pub fn final_result(text: String, confidence: f32) -> Self {
        Self::new(text, true, confidence)
    }

    /// Set time range
    pub fn with_time_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.start_time_ms = start_ms;
        self.end_time_ms = end_ms;
        self
    }

    /// Set language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set word timestamps
    pub fn with_words(mut self, words: Vec<WordTimestamp>) -> Self {
        self.words = words;
        self
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_time_ms.saturating_sub(self.start_time_ms)
    }

    /// Check if transcript is empty
    pub fn is_empty(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Get word count
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

impl Default for TranscriptResult {
    fn default() -> Self {
        Self {
            text: String::new(),
            is_final: false,
            confidence: 0.0,
            start_time_ms: 0,
            end_time_ms: 0,
            language: None,
            words: Vec::new(),
        }
    }
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// The word text
    pub word: String,

    /// Start time in milliseconds
    pub start_ms: u64,

    /// End time in milliseconds
    pub end_ms: u64,

    /// Confidence for this word
    pub confidence: f32,
}

impl WordTimestamp {
    pub fn new(word: impl Into<String>, start_ms: u64, end_ms: u64, confidence: f32) -> Self {
        Self {
            word: word.into(),
            start_ms,
            end_ms,
            confidence,
        }
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Streaming transcript accumulator
#[derive(Debug)]
pub struct TranscriptAccumulator {
    /// Confirmed (stable) text
    stable_text: String,
    /// Current unstable/partial text
    unstable_text: String,
    /// All words with timestamps
    words: Vec<WordTimestamp>,
    /// Current stream position in ms
    current_time_ms: u64,
    /// Minimum stability count for text confirmation
    stability_threshold: usize,
    /// Count of times current unstable text has been seen
    stability_count: usize,
    /// Last unstable text for stability comparison
    last_unstable: String,
}

impl TranscriptAccumulator {
    pub fn new() -> Self {
        Self {
            stable_text: String::new(),
            unstable_text: String::new(),
            words: Vec::new(),
            current_time_ms: 0,
            stability_threshold: 3,
            stability_count: 0,
            last_unstable: String::new(),
        }
    }

    /// Set stability threshold (number of identical partials before confirming)
    pub fn with_stability_threshold(mut self, threshold: usize) -> Self {
        self.stability_threshold = threshold;
        self
    }

    /// Process a transcript result
    pub fn process(&mut self, result: &TranscriptResult) -> Option<TranscriptResult> {
        self.current_time_ms = result.end_time_ms;

        if result.is_final {
            // Final result - confirm all text
            self.stable_text.push_str(&result.text);
            self.stable_text.push(' ');
            self.unstable_text.clear();
            self.stability_count = 0;
            self.last_unstable.clear();
            self.words.extend(result.words.clone());

            return Some(TranscriptResult {
                text: result.text.clone(),
                is_final: true,
                confidence: result.confidence,
                start_time_ms: result.start_time_ms,
                end_time_ms: result.end_time_ms,
                language: result.language.clone(),
                words: result.words.clone(),
            });
        }

        // Partial result - check for stability
        if result.text == self.last_unstable {
            self.stability_count += 1;
        } else {
            self.stability_count = 1;
            self.last_unstable = result.text.clone();
        }

        self.unstable_text = result.text.clone();

        // If stable enough, emit as confirmed
        if self.stability_count >= self.stability_threshold {
            let confirmed = self.unstable_text.clone();
            self.stable_text.push_str(&confirmed);
            self.stable_text.push(' ');
            self.unstable_text.clear();
            self.stability_count = 0;
            self.last_unstable.clear();

            return Some(TranscriptResult {
                text: confirmed,
                is_final: true,
                confidence: result.confidence,
                start_time_ms: result.start_time_ms,
                end_time_ms: result.end_time_ms,
                language: result.language.clone(),
                words: vec![],
            });
        }

        None
    }

    /// Get full transcript so far (stable + unstable)
    pub fn full_text(&self) -> String {
        let mut text = self.stable_text.trim().to_string();
        if !self.unstable_text.is_empty() {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&self.unstable_text);
        }
        text
    }

    /// Get only stable (confirmed) text
    pub fn stable_text(&self) -> &str {
        self.stable_text.trim()
    }

    /// Get unstable (partial) text
    pub fn unstable_text(&self) -> &str {
        &self.unstable_text
    }

    /// Reset accumulator
    pub fn reset(&mut self) {
        self.stable_text.clear();
        self.unstable_text.clear();
        self.words.clear();
        self.current_time_ms = 0;
        self.stability_count = 0;
        self.last_unstable.clear();
    }
}

impl Default for TranscriptAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_result() {
        let result = TranscriptResult::final_result("Hello world".to_string(), 0.95)
            .with_time_range(0, 1000)
            .with_language("en");

        assert!(result.is_final);
        assert_eq!(result.text, "Hello world");
        assert_eq!(result.duration_ms(), 1000);
        assert_eq!(result.word_count(), 2);
    }

    #[test]
    fn test_transcript_accumulator() {
        let mut acc = TranscriptAccumulator::new().with_stability_threshold(2);

        // First partial - no emission
        let result = acc.process(&TranscriptResult::partial("Hello".to_string(), 0.8));
        assert!(result.is_none());

        // Same partial again - should emit as confirmed
        let result = acc.process(&TranscriptResult::partial("Hello".to_string(), 0.8));
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Hello");

        // Final result
        let result = acc.process(&TranscriptResult::final_result("world".to_string(), 0.9));
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "world");

        assert_eq!(acc.stable_text(), "Hello world");
    }
}
