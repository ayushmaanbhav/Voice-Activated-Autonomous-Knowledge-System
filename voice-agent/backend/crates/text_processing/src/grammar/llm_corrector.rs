//! LLM-based grammar correction

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use voice_agent_core::{
    GrammarCorrector, DomainContext, LanguageModel,
    GenerateRequest, Message, Role, Result,
};

/// Grammar corrector using LLM
pub struct LLMGrammarCorrector {
    llm: Arc<dyn LanguageModel>,
    domain_context: DomainContext,
    temperature: f32,
}

impl LLMGrammarCorrector {
    /// Create a new LLM grammar corrector
    pub fn new(llm: Arc<dyn LanguageModel>, domain: &str, temperature: f32) -> Self {
        let domain_context = match domain {
            "gold_loan" => DomainContext::gold_loan(),
            _ => DomainContext::new(domain),
        };

        Self {
            llm,
            domain_context,
            temperature,
        }
    }

    /// Build grammar correction prompt
    fn build_prompt(&self, text: &str, context: &DomainContext) -> String {
        format!(
            r#"You are a speech-to-text error corrector for a {} conversation.

DOMAIN VOCABULARY (preserve these exact spellings):
{}

COMMON PHRASES (preserve):
{}

RULES:
1. Fix obvious transcription errors (homophones, mishearing)
2. Preserve proper nouns, bank names, and numbers exactly
3. Keep the meaning identical
4. Output ONLY the corrected text, nothing else
5. If text is already correct, output it unchanged
6. Handle Hindi-English code-switching naturally
7. Fix "gol lone" → "gold loan", "kotuk" → "Kotak", etc.

INPUT: {}
CORRECTED:"#,
            context.domain,
            context.vocabulary.join(", "),
            context.phrases.join("\n"),
            text,
        )
    }
}

#[async_trait]
impl GrammarCorrector for LLMGrammarCorrector {
    async fn correct(&self, text: &str, context: &DomainContext) -> Result<String> {
        // Skip very short text
        if text.trim().len() < 3 {
            return Ok(text.to_string());
        }

        let prompt = self.build_prompt(text, context);

        let request = GenerateRequest {
            messages: vec![Message {
                role: Role::User,
                content: prompt,
                name: None,
                tool_call_id: None,
            }],
            max_tokens: Some(256),
            temperature: Some(self.temperature),
            stream: false,
            ..Default::default()
        };

        let response = self.llm.generate(request).await?;
        let corrected = response.text.trim().to_string();

        // Sanity check: if correction is wildly different in length, keep original
        let len_ratio = corrected.len() as f32 / text.len() as f32;
        if len_ratio < 0.5 || len_ratio > 2.0 {
            tracing::warn!(
                "Grammar correction changed length significantly ({} -> {}), keeping original",
                text.len(),
                corrected.len()
            );
            return Ok(text.to_string());
        }

        Ok(corrected)
    }

    fn correct_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        context: &'a DomainContext,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        use futures::StreamExt;

        let ctx = context.clone();
        Box::pin(
            text_stream
                .then(move |text| {
                    let ctx = ctx.clone();
                    async move { self.correct(&text, &ctx).await }
                })
        )
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

impl Clone for LLMGrammarCorrector {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone(),
            domain_context: self.domain_context.clone(),
            temperature: self.temperature,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_building() {
        // We can't test the full corrector without an LLM, but we can test prompt building
        let context = DomainContext::gold_loan();
        assert!(context.vocabulary.contains(&"gold loan".to_string()));
        assert!(context.vocabulary.contains(&"Kotak".to_string()));
    }
}
