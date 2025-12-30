//! Text processing traits

use crate::{
    ComplianceResult, DomainContext, Language, PIIEntity, PIIType, RedactionStrategy, Result,
};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

/// Grammar correction interface
///
/// Implementations:
/// - `LLMGrammarCorrector` - Uses LLM for domain-aware correction
/// - `NoopCorrector` - Pass-through (disabled)
///
/// # Example
///
/// ```ignore
/// let corrector: Box<dyn GrammarCorrector> = Box::new(LLMGrammarCorrector::new(llm));
/// let context = DomainContext::gold_loan();
/// let corrected = corrector.correct("mujhe gol lone chahiye", &context).await?;
/// // "mujhe gold loan chahiye" - preserves domain vocabulary
/// ```
#[async_trait]
pub trait GrammarCorrector: Send + Sync + 'static {
    /// Correct grammar with domain context
    ///
    /// # Arguments
    /// * `text` - Text to correct
    /// * `context` - Domain context with vocabulary to preserve
    ///
    /// # Returns
    /// Corrected text with domain terms intact
    async fn correct(&self, text: &str, context: &DomainContext) -> Result<String>;

    /// Stream corrections sentence-by-sentence
    ///
    /// # Arguments
    /// * `text_stream` - Stream of text chunks
    /// * `context` - Domain context
    ///
    /// # Returns
    /// Stream of corrected text chunks
    fn correct_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        context: &'a DomainContext,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;

    /// Check if corrector is enabled
    fn is_enabled(&self) -> bool;
}

/// Translation interface
///
/// Implementations:
/// - `CandleIndicTrans2Translator` - Native Rust IndicTrans2 (recommended)
/// - `IndicTrans2Translator` - IndicTrans2 via ONNX (legacy)
/// - `NoopTranslator` - Pass-through (disabled)
///
/// Supports the "Translate-Think-Translate" pattern for LLM reasoning.
///
/// # Example
///
/// ```ignore
/// let translator: Box<dyn Translator> = Box::new(IndicTranslator::new(config));
/// let english = translator.translate(
///     "मुझे गोल्ड लोन चाहिए",
///     Language::Hindi,
///     Language::English
/// ).await?;
/// // "I need a gold loan"
/// ```
#[async_trait]
pub trait Translator: Send + Sync + 'static {
    /// Translate text between languages
    ///
    /// # Arguments
    /// * `text` - Text to translate
    /// * `from` - Source language
    /// * `to` - Target language
    ///
    /// # Returns
    /// Translated text
    async fn translate(&self, text: &str, from: Language, to: Language) -> Result<String>;

    /// Detect language of text
    ///
    /// # Arguments
    /// * `text` - Text to analyze
    ///
    /// # Returns
    /// Detected language
    async fn detect_language(&self, text: &str) -> Result<Language>;

    /// Stream translation sentence-by-sentence
    ///
    /// # Arguments
    /// * `text_stream` - Stream of text chunks
    /// * `from` - Source language
    /// * `to` - Target language
    ///
    /// # Returns
    /// Stream of translated chunks
    fn translate_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        from: Language,
        to: Language,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>>;

    /// Check if language pair is supported
    ///
    /// # Arguments
    /// * `from` - Source language
    /// * `to` - Target language
    ///
    /// # Returns
    /// true if translation between these languages is supported
    fn supports_pair(&self, from: Language, to: Language) -> bool;

    /// Get translator name for logging
    fn name(&self) -> &str;
}

/// PII detection and redaction interface
///
/// Implementations:
/// - `HybridPIIDetector` - Regex + NER
/// - `RegexPIIDetector` - Regex only (faster)
///
/// # Example
///
/// ```ignore
/// let detector: Box<dyn PIIRedactor> = Box::new(HybridPIIDetector::new());
/// let entities = detector.detect("My Aadhaar is 1234 5678 9012").await?;
/// // [PIIEntity { type: Aadhaar, text: "1234 5678 9012", ... }]
///
/// let redacted = detector.redact(text, &RedactionStrategy::PartialMask { visible_chars: 4 }).await?;
/// // "My Aadhaar is 12** **** **12"
/// ```
#[async_trait]
pub trait PIIRedactor: Send + Sync + 'static {
    /// Detect PII entities in text
    ///
    /// # Arguments
    /// * `text` - Text to analyze
    ///
    /// # Returns
    /// List of detected PII entities with positions
    async fn detect(&self, text: &str) -> Result<Vec<PIIEntity>>;

    /// Redact PII from text
    ///
    /// # Arguments
    /// * `text` - Text to redact
    /// * `strategy` - How to redact (mask, remove, etc.)
    ///
    /// # Returns
    /// Text with PII redacted
    async fn redact(&self, text: &str, strategy: &RedactionStrategy) -> Result<String>;

    /// Get supported PII types
    fn supported_types(&self) -> &[PIIType];

    /// Check if a specific PII type is supported
    fn supports_type(&self, pii_type: PIIType) -> bool {
        self.supported_types().contains(&pii_type)
    }
}

/// Compliance checking interface
///
/// Implementations:
/// - `RuleBasedComplianceChecker` - Config-driven rules
/// - `LLMComplianceChecker` - LLM-based checking
///
/// # Example
///
/// ```ignore
/// let checker: Box<dyn ComplianceChecker> = Box::new(RuleBasedComplianceChecker::new(rules));
/// let result = checker.check("We guarantee the lowest interest rate!").await?;
/// if !result.is_compliant {
///     for violation in result.violations {
///         println!("Violation: {} - {}", violation.rule_id, violation.description);
///     }
/// }
/// ```
#[async_trait]
pub trait ComplianceChecker: Send + Sync + 'static {
    /// Check text for compliance violations
    ///
    /// # Arguments
    /// * `text` - Text to check
    ///
    /// # Returns
    /// Compliance result with violations and suggestions
    async fn check(&self, text: &str) -> Result<ComplianceResult>;

    /// Make text compliant by fixing violations
    ///
    /// # Arguments
    /// * `text` - Text to fix
    ///
    /// # Returns
    /// Text with violations fixed and required additions added
    async fn make_compliant(&self, text: &str) -> Result<String>;

    /// Get compliance rules version
    fn rules_version(&self) -> &str;

    /// Reload rules from configuration
    async fn reload_rules(&self) -> Result<()> {
        Ok(()) // Default no-op
    }
}

/// P0 FIX: Unified text processor trait for pipeline integration
///
/// This trait allows the voice pipeline to apply text processing
/// (grammar correction, PII redaction, compliance checking) to
/// transcripts before sending them to the LLM.
///
/// Implementations:
/// - `TextProcessingPipeline` in voice-agent-text-processing crate
///
/// # Example
///
/// ```ignore
/// let processor: Arc<dyn TextProcessor> = create_text_processor();
/// let pipeline = VoicePipeline::simple(config)?
///     .with_text_processor(processor);
///
/// // Now transcripts are automatically processed before LLM
/// ```
#[async_trait]
pub trait TextProcessor: Send + Sync + 'static {
    /// Process text through the full pipeline (grammar, PII, compliance)
    ///
    /// # Arguments
    /// * `text` - Raw text from STT
    ///
    /// # Returns
    /// Processed text ready for LLM, or error if processing fails
    async fn process(&self, text: &str) -> Result<TextProcessorResult>;

    /// Process only PII (faster path when grammar/compliance not needed)
    async fn process_pii_only(&self, text: &str) -> Result<String>;

    /// Check if processor is enabled
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Result from text processing
#[derive(Debug, Clone)]
pub struct TextProcessorResult {
    /// Original input text
    pub original: String,
    /// Processed output text (use this for LLM)
    pub processed: String,
    /// Whether PII was detected and redacted
    pub pii_detected: bool,
    /// Whether compliance issues were found and fixed
    pub compliance_fixed: bool,
}

impl TextProcessorResult {
    /// Create a new result with just processed text (no PII/compliance changes)
    pub fn passthrough(text: String) -> Self {
        Self {
            original: text.clone(),
            processed: text,
            pii_detected: false,
            compliance_fixed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTranslator;

    #[async_trait]
    impl Translator for MockTranslator {
        async fn translate(&self, text: &str, _from: Language, _to: Language) -> Result<String> {
            Ok(format!("[Translated: {}]", text))
        }

        async fn detect_language(&self, _text: &str) -> Result<Language> {
            Ok(Language::Hindi)
        }

        fn translate_stream<'a>(
            &'a self,
            _text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
            _from: Language,
            _to: Language,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            Box::pin(futures::stream::empty())
        }

        fn supports_pair(&self, _from: Language, _to: Language) -> bool {
            true
        }

        fn name(&self) -> &str {
            "mock-translator"
        }
    }

    #[tokio::test]
    async fn test_mock_translator() {
        let translator = MockTranslator;
        assert!(translator.supports_pair(Language::Hindi, Language::English));

        let result = translator
            .translate("नमस्ते", Language::Hindi, Language::English)
            .await
            .unwrap();
        assert!(result.contains("Translated"));
    }
}
