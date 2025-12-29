//! No-op grammar corrector (pass-through)

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use voice_agent_core::{GrammarCorrector, DomainContext, Result};

/// Pass-through corrector that does nothing
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopCorrector;

#[async_trait]
impl GrammarCorrector for NoopCorrector {
    async fn correct(&self, text: &str, _context: &DomainContext) -> Result<String> {
        Ok(text.to_string())
    }

    fn correct_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        _context: &'a DomainContext,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        use futures::StreamExt;
        Box::pin(text_stream.map(Ok))
    }

    fn is_enabled(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_passthrough() {
        let corrector = NoopCorrector;
        let context = DomainContext::default();

        let input = "mujhe gol lone chahiye";
        let output = corrector.correct(input, &context).await.unwrap();

        assert_eq!(input, output);
        assert!(!corrector.is_enabled());
    }
}
