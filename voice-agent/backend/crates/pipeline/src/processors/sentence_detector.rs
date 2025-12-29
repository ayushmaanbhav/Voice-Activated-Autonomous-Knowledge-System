//! Sentence detector for streaming LLM output
//!
//! Buffers LLM chunks and emits complete sentences for TTS.
//! Supports Indic script terminators (।, ॥, etc.) in addition to
//! standard punctuation.

use async_trait::async_trait;
use parking_lot::Mutex;
use voice_agent_core::{
    Frame, FrameProcessor, ProcessorContext, Result, Language,
};

/// Sentence detector configuration
#[derive(Debug, Clone)]
pub struct SentenceDetectorConfig {
    /// Minimum characters before emitting (latency optimization)
    pub min_chars_first_sentence: usize,
    /// Maximum characters to buffer before forcing emission
    pub max_buffer_chars: usize,
    /// Whether to emit partial sentences on flush
    pub emit_partial_on_flush: bool,
    /// Detect language from context for script-aware detection
    pub use_context_language: bool,
}

impl Default for SentenceDetectorConfig {
    fn default() -> Self {
        Self {
            min_chars_first_sentence: 15,
            max_buffer_chars: 500,
            emit_partial_on_flush: true,
            use_context_language: true,
        }
    }
}

/// Sentence detector that buffers LLM chunks and emits sentences
pub struct SentenceDetector {
    config: SentenceDetectorConfig,
    /// Buffer for accumulating text
    buffer: Mutex<String>,
    /// Current sentence index
    sentence_index: Mutex<usize>,
    /// First sentence emitted?
    first_emitted: Mutex<bool>,
    /// Detected or configured language
    language: Mutex<Language>,
}

impl SentenceDetector {
    /// Create a new sentence detector with config
    pub fn new(config: SentenceDetectorConfig) -> Self {
        Self {
            config,
            buffer: Mutex::new(String::new()),
            sentence_index: Mutex::new(0),
            first_emitted: Mutex::new(false),
            language: Mutex::new(Language::English),
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(SentenceDetectorConfig::default())
    }

    /// Set the language for sentence detection
    pub fn set_language(&self, language: Language) {
        *self.language.lock() = language;
    }

    /// Get sentence terminators for current language
    fn terminators(&self) -> &'static [char] {
        self.language.lock().sentence_terminators()
    }

    /// Check if character is a sentence terminator
    fn is_terminator(&self, c: char) -> bool {
        self.terminators().contains(&c)
    }

    /// Find sentence boundaries in text
    /// Returns (sentences, remaining_text)
    fn find_sentences(&self, text: &str) -> (Vec<String>, String) {
        let mut sentences = Vec::new();
        let mut current = String::new();
        let terminators = self.terminators();

        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];
            current.push(c);

            // Check if this is a terminator
            if terminators.contains(&c) {
                // Look ahead for closing quotes or brackets
                let mut end = i + 1;
                while end < chars.len() {
                    let next = chars[end];
                    // Standard and curly quotes, brackets
                    if next == '"' || next == '\'' || next == ')' || next == ']'
                        || next == '"' || next == '\u{2019}' || next == '」'
                    {
                        current.push(next);
                        end += 1;
                    } else if next.is_whitespace() {
                        // Include trailing whitespace
                        current.push(next);
                        end += 1;
                        break;
                    } else {
                        break;
                    }
                }
                i = end;

                // Emit sentence
                let sentence = current.trim().to_string();
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                current.clear();
            } else {
                i += 1;
            }
        }

        (sentences, current)
    }

    /// Try to extract sentences from buffer
    fn extract_sentences(&self) -> Vec<String> {
        let mut buffer = self.buffer.lock();
        let (sentences, remaining) = self.find_sentences(&buffer);
        *buffer = remaining;
        sentences
    }

    /// Force flush buffer (for end of stream)
    fn flush_buffer(&self) -> Option<String> {
        let mut buffer = self.buffer.lock();
        if buffer.is_empty() {
            return None;
        }
        let text = buffer.trim().to_string();
        buffer.clear();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }

    /// Create sentence frames from extracted sentences
    fn create_sentence_frames(&self, sentences: Vec<String>) -> Vec<Frame> {
        let language = *self.language.lock();
        let mut index = self.sentence_index.lock();
        let mut first = self.first_emitted.lock();

        sentences
            .into_iter()
            .map(|text| {
                let frame = Frame::Sentence {
                    text,
                    language,
                    index: *index,
                };
                *index += 1;
                *first = true;
                frame
            })
            .collect()
    }

    /// Check if we should emit early (first sentence optimization)
    fn should_emit_early(&self, buffer_len: usize) -> bool {
        let first = *self.first_emitted.lock();
        if first {
            // After first sentence, use max buffer limit
            buffer_len >= self.config.max_buffer_chars
        } else {
            // First sentence: emit early for low latency
            buffer_len >= self.config.min_chars_first_sentence
        }
    }

    /// Reset state
    pub fn reset(&self) {
        self.buffer.lock().clear();
        *self.sentence_index.lock() = 0;
        *self.first_emitted.lock() = false;
    }
}

#[async_trait]
impl FrameProcessor for SentenceDetector {
    async fn process(
        &self,
        frame: Frame,
        context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        // Update language from context if configured
        if self.config.use_context_language {
            if let Some(lang) = context.language {
                self.set_language(lang);
            }
        }

        match frame {
            Frame::LLMChunk { text, is_final } => {
                // Add text to buffer
                {
                    let mut buffer = self.buffer.lock();
                    buffer.push_str(&text);
                }

                // Extract complete sentences
                let mut sentences = self.extract_sentences();

                // If this is the final chunk, flush remaining buffer
                if is_final {
                    if let Some(remaining) = self.flush_buffer() {
                        sentences.push(remaining);
                    }
                }

                // Check for early emission if no sentences found
                if sentences.is_empty() && !is_final {
                    let buffer_len = self.buffer.lock().len();

                    // Check if we should force emit due to buffer size
                    if self.should_emit_early(buffer_len) {
                        // Try to find a good break point (word boundary)
                        let mut buffer = self.buffer.lock();
                        if let Some(pos) = buffer.rfind(char::is_whitespace) {
                            let partial = buffer[..pos].trim().to_string();
                            let remaining = buffer[pos..].to_string();
                            *buffer = remaining;
                            if !partial.is_empty() {
                                sentences.push(partial);
                            }
                        }
                    }
                }

                // Create sentence frames
                Ok(self.create_sentence_frames(sentences))
            }

            Frame::Control(voice_agent_core::ControlFrame::Flush) => {
                // Flush on control frame
                let mut frames = Vec::new();
                if self.config.emit_partial_on_flush {
                    if let Some(remaining) = self.flush_buffer() {
                        frames.extend(self.create_sentence_frames(vec![remaining]));
                    }
                }
                frames.push(frame);
                Ok(frames)
            }

            Frame::Control(voice_agent_core::ControlFrame::Reset) => {
                self.reset();
                Ok(vec![frame])
            }

            Frame::EndOfStream => {
                // Emit any remaining buffer
                let mut frames = Vec::new();
                if let Some(remaining) = self.flush_buffer() {
                    frames.extend(self.create_sentence_frames(vec![remaining]));
                }
                frames.push(frame);
                Ok(frames)
            }

            // Pass through other frames
            _ => Ok(vec![frame]),
        }
    }

    fn name(&self) -> &'static str {
        "sentence_detector"
    }

    fn description(&self) -> &str {
        "Detects sentence boundaries from LLM chunks with Indic script support"
    }

    async fn on_start(&self, context: &mut ProcessorContext) -> Result<()> {
        // Initialize with context language
        if let Some(lang) = context.language {
            self.set_language(lang);
        }
        Ok(())
    }

    async fn on_stop(&self, _context: &mut ProcessorContext) -> Result<()> {
        self.reset();
        Ok(())
    }

    fn can_handle(&self, frame: &Frame) -> bool {
        matches!(
            frame,
            Frame::LLMChunk { .. }
                | Frame::Control(_)
                | Frame::EndOfStream
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_detector() -> SentenceDetector {
        SentenceDetector::new(SentenceDetectorConfig {
            min_chars_first_sentence: 10,
            max_buffer_chars: 100,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn test_simple_sentence() {
        let detector = create_detector();
        let mut ctx = ProcessorContext::default();

        let frames = detector
            .process(
                Frame::LLMChunk {
                    text: "Hello world. How are you?".to_string(),
                    is_final: true,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should produce 2 sentences
        let sentences: Vec<_> = frames
            .iter()
            .filter_map(|f| match f {
                Frame::Sentence { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].contains("Hello world"));
        assert!(sentences[1].contains("How are you"));
    }

    #[tokio::test]
    async fn test_hindi_terminators() {
        let detector = create_detector();
        detector.set_language(Language::Hindi);
        let mut ctx = ProcessorContext::default();

        let frames = detector
            .process(
                Frame::LLMChunk {
                    text: "नमस्ते। आप कैसे हैं?".to_string(),
                    is_final: true,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should detect Hindi danda (।) as sentence terminator
        let sentences: Vec<_> = frames
            .iter()
            .filter_map(|f| match f {
                Frame::Sentence { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(sentences.len(), 2);
    }

    #[tokio::test]
    async fn test_streaming_chunks() {
        let detector = create_detector();
        let mut ctx = ProcessorContext::default();

        // First chunk - not enough for a sentence
        let frames1 = detector
            .process(
                Frame::LLMChunk {
                    text: "Hello ".to_string(),
                    is_final: false,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // May not emit yet (depends on min chars)

        // Second chunk with sentence end
        let frames2 = detector
            .process(
                Frame::LLMChunk {
                    text: "world.".to_string(),
                    is_final: true,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Should have at least one sentence
        let total_sentences: usize = frames1
            .iter()
            .chain(frames2.iter())
            .filter(|f| matches!(f, Frame::Sentence { .. }))
            .count();

        assert!(total_sentences >= 1);
    }

    #[tokio::test]
    async fn test_passthrough() {
        let detector = create_detector();
        let mut ctx = ProcessorContext::default();

        // Non-LLM frames should pass through
        let frames = detector
            .process(Frame::VoiceStart, &mut ctx)
            .await
            .unwrap();

        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::VoiceStart));
    }

    #[tokio::test]
    async fn test_reset() {
        let detector = create_detector();
        let mut ctx = ProcessorContext::default();

        // Add some text
        detector
            .process(
                Frame::LLMChunk {
                    text: "Hello ".to_string(),
                    is_final: false,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        // Reset
        detector.reset();

        // Buffer should be empty
        assert!(detector.buffer.lock().is_empty());
    }

    #[tokio::test]
    async fn test_sentence_index() {
        let detector = create_detector();
        let mut ctx = ProcessorContext::default();

        let frames = detector
            .process(
                Frame::LLMChunk {
                    text: "One. Two. Three.".to_string(),
                    is_final: true,
                },
                &mut ctx,
            )
            .await
            .unwrap();

        let indices: Vec<_> = frames
            .iter()
            .filter_map(|f| match f {
                Frame::Sentence { index, .. } => Some(*index),
                _ => None,
            })
            .collect();

        assert_eq!(indices, vec![0, 1, 2]);
    }
}
