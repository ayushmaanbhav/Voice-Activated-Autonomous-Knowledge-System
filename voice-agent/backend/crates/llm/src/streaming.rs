//! Streaming Token Generation
//!
//! Provides streaming interfaces for LLM output.

use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;
use tokio_stream::Stream;

use crate::backend::GenerationResult;

/// Token stream type
pub type TokenStream = Pin<Box<dyn Stream<Item = String> + Send>>;

/// Generation event
#[derive(Debug, Clone)]
pub enum GenerationEvent {
    /// New token generated
    Token(String),
    /// Generation started
    Started,
    /// Generation complete
    Complete(GenerationResult),
    /// Error occurred
    Error(String),
}

/// Streaming generator wrapper
pub struct StreamingGenerator {
    rx: mpsc::Receiver<String>,
    tokens: Vec<String>,
    complete: bool,
}

impl StreamingGenerator {
    /// Create a new streaming generator
    pub fn new(rx: mpsc::Receiver<String>) -> Self {
        Self {
            rx,
            tokens: Vec::new(),
            complete: false,
        }
    }

    /// Create a channel pair for streaming
    pub fn channel(buffer: usize) -> (mpsc::Sender<String>, Self) {
        let (tx, rx) = mpsc::channel(buffer);
        (tx, Self::new(rx))
    }

    /// Get next token
    pub async fn next_token(&mut self) -> Option<String> {
        if self.complete {
            return None;
        }

        match self.rx.recv().await {
            Some(token) => {
                self.tokens.push(token.clone());
                Some(token)
            }
            None => {
                self.complete = true;
                None
            }
        }
    }

    /// Get all tokens collected so far
    pub fn collected(&self) -> &[String] {
        &self.tokens
    }

    /// Get full text
    pub fn text(&self) -> String {
        self.tokens.join("")
    }

    /// Is generation complete?
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Convert to stream
    pub fn into_stream(self) -> impl Stream<Item = String> {
        tokio_stream::wrappers::ReceiverStream::new(self.rx)
    }
}

impl Stream for StreamingGenerator {
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.complete {
            return Poll::Ready(None);
        }

        match Pin::new(&mut self.rx).poll_recv(cx) {
            Poll::Ready(Some(token)) => {
                self.tokens.push(token.clone());
                Poll::Ready(Some(token))
            }
            Poll::Ready(None) => {
                self.complete = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Token buffer for word-level emission
pub struct TokenBuffer {
    tokens: Vec<String>,
    partial_word: String,
}

impl TokenBuffer {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            partial_word: String::new(),
        }
    }

    /// Add a token and get complete words if any
    pub fn add(&mut self, token: &str) -> Vec<String> {
        self.tokens.push(token.to_string());
        self.partial_word.push_str(token);

        let mut words = Vec::new();

        // P2 FIX: Unicode-aware word boundary detection
        // Include Hindi/Indic punctuation marks (danda ।, double danda ॥) for multilingual support
        while let Some(space_idx) = self.partial_word.find(|c: char| {
            c.is_whitespace() || matches!(c, '.' | ',' | '!' | '?' | ';' | '।' | '॥')
        }) {
            // Get the character at the found position to determine its UTF-8 byte length
            let ch = self.partial_word[space_idx..].chars().next().unwrap();
            let ch_len = ch.len_utf8();

            let word = self.partial_word[..space_idx + ch_len].to_string();
            if !word.trim().is_empty() {
                words.push(word);
            }
            self.partial_word = self.partial_word[space_idx + ch_len..].to_string();
        }

        words
    }

    /// Flush remaining content
    pub fn flush(&mut self) -> Option<String> {
        if self.partial_word.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.partial_word))
        }
    }

    /// Get all tokens
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.partial_word.clear();
    }
}

impl Default for TokenBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Word callback type alias
type WordCallback = Box<dyn Fn(&str) + Send>;

/// Streaming response builder
pub struct ResponseBuilder {
    tokens: Vec<String>,
    word_buffer: TokenBuffer,
    word_callback: Option<WordCallback>,
}

impl ResponseBuilder {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            word_buffer: TokenBuffer::new(),
            word_callback: None,
        }
    }

    /// Set callback for word-level events
    pub fn on_word<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str) + Send + 'static,
    {
        self.word_callback = Some(Box::new(callback));
        self
    }

    /// Add a token
    pub fn add_token(&mut self, token: &str) {
        self.tokens.push(token.to_string());

        // Get complete words
        let words = self.word_buffer.add(token);

        // Invoke callback for each word
        if let Some(ref callback) = self.word_callback {
            for word in words {
                callback(&word);
            }
        }
    }

    /// Finalize and get complete text
    pub fn finalize(mut self) -> String {
        // Flush remaining content
        if let Some(word) = self.word_buffer.flush() {
            if let Some(ref callback) = self.word_callback {
                callback(&word);
            }
        }

        self.tokens.join("")
    }

    /// Get current text
    pub fn current_text(&self) -> String {
        self.tokens.join("")
    }
}

impl Default for ResponseBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_buffer() {
        let mut buffer = TokenBuffer::new();

        // Add partial tokens
        let words = buffer.add("Hel");
        assert!(words.is_empty());

        let words = buffer.add("lo ");
        assert_eq!(words.len(), 1);
        assert!(words[0].contains("Hello"));

        let words = buffer.add("world!");
        assert_eq!(words.len(), 1);
    }

    #[test]
    fn test_token_buffer_flush() {
        let mut buffer = TokenBuffer::new();
        buffer.add("partial");

        let remaining = buffer.flush();
        assert!(remaining.is_some());
        assert_eq!(remaining.unwrap(), "partial");
    }

    #[tokio::test]
    async fn test_streaming_generator() {
        let (tx, mut gen) = StreamingGenerator::channel(10);

        // Send some tokens
        tx.send("Hello".to_string()).await.unwrap();
        tx.send(" world".to_string()).await.unwrap();
        drop(tx);

        let mut tokens = Vec::new();
        while let Some(token) = gen.next_token().await {
            tokens.push(token);
        }

        assert_eq!(tokens.len(), 2);
        assert_eq!(gen.text(), "Hello world");
    }
}
