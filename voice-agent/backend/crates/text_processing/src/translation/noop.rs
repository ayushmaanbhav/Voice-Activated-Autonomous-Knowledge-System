//! No-op translator (pass-through)

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use voice_agent_core::{Translator, Language, Result};
use super::ScriptDetector;

/// Pass-through translator that does nothing
#[derive(Debug, Clone, Default)]
pub struct NoopTranslator {
    detector: ScriptDetector,
}

impl NoopTranslator {
    /// Create a new noop translator
    pub fn new() -> Self {
        Self {
            detector: ScriptDetector::new(),
        }
    }
}

#[async_trait]
impl Translator for NoopTranslator {
    async fn translate(&self, text: &str, _from: Language, _to: Language) -> Result<String> {
        Ok(text.to_string())
    }

    async fn detect_language(&self, text: &str) -> Result<Language> {
        Ok(self.detector.detect(text))
    }

    fn translate_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        _from: Language,
        _to: Language,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        use futures::StreamExt;
        Box::pin(text_stream.map(Ok))
    }

    fn supports_pair(&self, _from: Language, _to: Language) -> bool {
        false
    }

    fn name(&self) -> &str {
        "noop"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_passthrough() {
        let translator = NoopTranslator::new();

        let input = "नमस्ते";
        let output = translator.translate(input, Language::Hindi, Language::English).await.unwrap();

        assert_eq!(input, output);
        assert!(!translator.supports_pair(Language::Hindi, Language::English));
    }

    #[tokio::test]
    async fn test_language_detection() {
        let translator = NoopTranslator::new();

        let lang = translator.detect_language("नमस्ते").await.unwrap();
        assert_eq!(lang, Language::Hindi);

        let lang = translator.detect_language("Hello").await.unwrap();
        assert_eq!(lang, Language::English);
    }
}
