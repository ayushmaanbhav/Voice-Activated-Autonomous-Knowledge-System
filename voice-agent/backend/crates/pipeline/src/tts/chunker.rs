//! Word-level chunking for TTS
//!
//! Splits text into speakable chunks for early emission.


/// Chunk output from the chunker
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// The text to synthesize
    pub text: String,
    /// Word indices in original text
    pub word_indices: Vec<usize>,
    /// Is this the final chunk?
    pub is_final: bool,
    /// Can pause after this chunk (natural boundary)
    pub can_pause: bool,
}

/// Chunking strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkStrategy {
    /// Single word chunks (lowest latency)
    SingleWord,
    /// Phrase-based chunks (better prosody)
    Phrase,
    /// Sentence-based chunks (best prosody)
    Sentence,
    /// Adaptive based on context
    Adaptive,
}

/// Word chunker configuration
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Chunking strategy
    pub strategy: ChunkStrategy,
    /// Minimum words per chunk (for phrase/sentence)
    pub min_words: usize,
    /// Maximum words per chunk
    pub max_words: usize,
    /// Minimum characters for first chunk (latency optimization)
    pub first_chunk_min_chars: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::Adaptive,
            min_words: 1,
            max_words: 10,
            first_chunk_min_chars: 10,
        }
    }
}

/// Word-level chunker for streaming TTS
pub struct WordChunker {
    config: ChunkerConfig,
    /// Buffer of incoming text
    buffer: String,
    /// Words parsed so far
    words: Vec<String>,
    /// Current word index
    current_word: usize,
    /// First chunk emitted?
    first_chunk_emitted: bool,
    /// Finalized?
    finalized: bool,
}

impl WordChunker {
    /// Create a new word chunker
    pub fn new(config: ChunkerConfig) -> Self {
        Self {
            config,
            buffer: String::new(),
            words: Vec::new(),
            current_word: 0,
            first_chunk_emitted: false,
            finalized: false,
        }
    }

    /// Add text to the buffer
    pub fn add_text(&mut self, text: &str) {
        self.buffer.push_str(text);
        self.parse_words();
    }

    /// Parse words from buffer
    ///
    /// P2 FIX: Optimized to collect word boundaries first, then extract words in a single pass.
    /// Previously O(n²) due to repeated String allocations; now O(n).
    fn parse_words(&mut self) {
        // First pass: collect word boundaries (start, end) without allocating
        let mut boundaries: Vec<(usize, usize)> = Vec::new();
        let mut word_start = None;

        for (i, c) in self.buffer.char_indices() {
            if c.is_whitespace() {
                if let Some(start) = word_start {
                    if start < i {
                        boundaries.push((start, i));
                    }
                    word_start = None;
                }
            } else if word_start.is_none() {
                word_start = Some(i);
            }
        }

        // Extract complete words in a single pass
        for (start, end) in boundaries {
            self.words.push(self.buffer[start..end].to_string());
        }

        // Keep incomplete word in buffer using drain to avoid reallocation
        if let Some(start) = word_start {
            // Only keep the incomplete word portion
            let remaining = self.buffer[start..].to_string();
            self.buffer.clear();
            self.buffer.push_str(&remaining);
        } else {
            self.buffer.clear();
        }
    }

    /// Get next chunk if available
    pub fn next_chunk(&mut self) -> Option<TextChunk> {
        if self.current_word >= self.words.len() {
            if self.finalized && !self.buffer.is_empty() {
                // Emit remaining buffer as final chunk
                let chunk = TextChunk {
                    text: std::mem::take(&mut self.buffer),
                    word_indices: vec![self.current_word],
                    is_final: true,
                    can_pause: true,
                };
                return Some(chunk);
            }
            return None;
        }

        let chunk = match self.config.strategy {
            ChunkStrategy::SingleWord => self.next_single_word(),
            ChunkStrategy::Phrase => self.next_phrase(),
            ChunkStrategy::Sentence => self.next_sentence(),
            ChunkStrategy::Adaptive => self.next_adaptive(),
        };

        if chunk.is_some() {
            self.first_chunk_emitted = true;
        }

        chunk
    }

    /// Get next single word chunk
    fn next_single_word(&mut self) -> Option<TextChunk> {
        if self.current_word >= self.words.len() {
            return None;
        }

        let word = self.words[self.current_word].clone();
        let can_pause = self.is_pause_point(&word);
        let is_final = self.finalized && self.current_word == self.words.len() - 1;

        self.current_word += 1;

        Some(TextChunk {
            text: word,
            word_indices: vec![self.current_word - 1],
            is_final,
            can_pause,
        })
    }

    /// Get next phrase chunk
    fn next_phrase(&mut self) -> Option<TextChunk> {
        if self.current_word >= self.words.len() {
            return None;
        }

        let start = self.current_word;
        let mut end = start;
        let mut text = String::new();

        // Collect words until pause point or max
        while end < self.words.len() && (end - start) < self.config.max_words {
            let word = &self.words[end];
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(word);
            end += 1;

            // Stop at phrase boundary if we have minimum words
            if (end - start) >= self.config.min_words && self.is_pause_point(word) {
                break;
            }
        }

        let indices: Vec<usize> = (start..end).collect();
        let is_final = self.finalized && end == self.words.len();
        let can_pause = end > start && self.is_pause_point(&self.words[end - 1]);

        self.current_word = end;

        Some(TextChunk {
            text,
            word_indices: indices,
            is_final,
            can_pause,
        })
    }

    /// Get next sentence chunk
    fn next_sentence(&mut self) -> Option<TextChunk> {
        if self.current_word >= self.words.len() {
            return None;
        }

        let start = self.current_word;
        let mut end = start;
        let mut text = String::new();

        // Collect words until sentence end or max
        while end < self.words.len() && (end - start) < self.config.max_words {
            let word = &self.words[end];
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(word);
            end += 1;

            // Stop at sentence boundary
            if self.is_sentence_end(word) {
                break;
            }
        }

        let indices: Vec<usize> = (start..end).collect();
        let is_final = self.finalized && end == self.words.len();
        let can_pause = end > start && self.is_pause_point(&self.words[end - 1]);

        self.current_word = end;

        Some(TextChunk {
            text,
            word_indices: indices,
            is_final,
            can_pause,
        })
    }

    /// Get next adaptive chunk
    fn next_adaptive(&mut self) -> Option<TextChunk> {
        if self.current_word >= self.words.len() {
            return None;
        }

        // First chunk: optimize for latency
        if !self.first_chunk_emitted {
            // Emit as soon as we have minimum chars
            let start = self.current_word;
            let mut end = start;
            let mut text = String::new();
            let mut char_count = 0;

            while end < self.words.len() && char_count < self.config.first_chunk_min_chars {
                let word = &self.words[end];
                if !text.is_empty() {
                    text.push(' ');
                    char_count += 1;
                }
                text.push_str(word);
                char_count += word.len();
                end += 1;
            }

            if !text.is_empty() {
                let indices: Vec<usize> = (start..end).collect();
                let is_final = self.finalized && end == self.words.len();

                self.current_word = end;

                return Some(TextChunk {
                    text,
                    word_indices: indices,
                    is_final,
                    can_pause: false, // Don't pause on first chunk
                });
            }
        }

        // Subsequent chunks: use phrase-based
        self.next_phrase()
    }

    /// Check if word is a pause point
    fn is_pause_point(&self, word: &str) -> bool {
        word.ends_with(',') ||
        word.ends_with('.') ||
        word.ends_with('!') ||
        word.ends_with('?') ||
        word.ends_with(':') ||
        word.ends_with(';') ||
        word.ends_with('—') ||
        word.ends_with('-')
    }

    /// Check if word ends a sentence
    fn is_sentence_end(&self, word: &str) -> bool {
        word.ends_with('.') ||
        word.ends_with('!') ||
        word.ends_with('?')
    }

    /// Mark input as finalized
    pub fn finalize(&mut self) {
        self.finalized = true;
        // Parse any remaining buffer
        if !self.buffer.is_empty() {
            let remaining = std::mem::take(&mut self.buffer);
            self.words.push(remaining);
        }
    }

    /// Reset chunker state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.words.clear();
        self.current_word = 0;
        self.first_chunk_emitted = false;
        self.finalized = false;
    }

    /// Get pending word count
    pub fn pending_words(&self) -> usize {
        self.words.len() - self.current_word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_word_chunking() {
        let mut chunker = WordChunker::new(ChunkerConfig {
            strategy: ChunkStrategy::SingleWord,
            ..Default::default()
        });

        chunker.add_text("Hello world ");
        chunker.finalize();

        let chunk1 = chunker.next_chunk().unwrap();
        assert_eq!(chunk1.text, "Hello");

        let chunk2 = chunker.next_chunk().unwrap();
        assert_eq!(chunk2.text, "world");
        assert!(chunk2.is_final);
    }

    #[test]
    fn test_phrase_chunking() {
        let mut chunker = WordChunker::new(ChunkerConfig {
            strategy: ChunkStrategy::Phrase,
            min_words: 2,
            max_words: 5,
            ..Default::default()
        });

        chunker.add_text("Hello, how are you today? I am fine. ");
        chunker.finalize();

        let chunk1 = chunker.next_chunk().unwrap();
        assert!(chunk1.text.contains("Hello"));
        assert!(chunk1.can_pause);
    }

    #[test]
    fn test_adaptive_first_chunk() {
        let mut chunker = WordChunker::new(ChunkerConfig {
            strategy: ChunkStrategy::Adaptive,
            first_chunk_min_chars: 10,
            ..Default::default()
        });

        chunker.add_text("Hi there ");
        let chunk = chunker.next_chunk().unwrap();
        // Should emit early for first chunk
        assert!(chunk.text.len() >= 8); // "Hi there"
    }

    #[test]
    fn test_reset() {
        let mut chunker = WordChunker::new(ChunkerConfig::default());
        chunker.add_text("Test text ");
        chunker.reset();
        assert_eq!(chunker.pending_words(), 0);
    }
}
