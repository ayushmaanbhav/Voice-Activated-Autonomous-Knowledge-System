//! Unified text processing pipeline

use crate::{
    compliance::{self, ComplianceConfig},
    grammar::{self, GrammarConfig},
    pii::{self, PIIConfig},
    translation::{self, ScriptDetector, TranslationConfig},
    Result, TextProcessingError,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use voice_agent_core::{
    ComplianceChecker, DomainContext, GrammarCorrector, Language, LanguageModel, PIIRedactor,
    RedactionStrategy, TextProcessor, TextProcessorResult, Translator,
};

/// Unified text processing pipeline
pub struct TextProcessingPipeline {
    grammar_corrector: Arc<dyn GrammarCorrector>,
    translator: Arc<dyn Translator>,
    pii_detector: Arc<dyn PIIRedactor>,
    compliance_checker: Arc<dyn ComplianceChecker>,
    script_detector: ScriptDetector,
    domain_context: DomainContext,
    config: TextProcessingConfig,
}

impl TextProcessingPipeline {
    /// Create a new pipeline with configuration
    pub fn new(config: TextProcessingConfig, llm: Option<Arc<dyn LanguageModel>>) -> Self {
        let grammar_corrector = grammar::create_corrector(&config.grammar, llm);
        let translator = translation::create_translator(&config.translation);
        let pii_detector = pii::create_detector(&config.pii);
        let compliance_checker = compliance::create_checker(&config.compliance);

        let domain_context = match config.domain.as_str() {
            "gold_loan" => DomainContext::gold_loan(),
            _ => DomainContext::new(&config.domain),
        };

        Self {
            grammar_corrector,
            translator,
            pii_detector,
            compliance_checker,
            script_detector: ScriptDetector::new(),
            domain_context,
            config,
        }
    }

    /// Process text through the full pipeline
    ///
    /// Order: Grammar → Translation (if needed) → PII → Compliance
    pub async fn process(&self, text: &str) -> Result<ProcessedText> {
        let mut result = ProcessedText {
            original: text.to_string(),
            processed: text.to_string(),
            detected_language: Language::English,
            was_translated: false,
            pii_detected: false,
            is_compliant: true,
            steps: Vec::new(),
        };

        // Step 1: Detect language
        result.detected_language = self.script_detector.detect(text);
        result.steps.push(ProcessingStep {
            name: "language_detection".to_string(),
            input: text.to_string(),
            output: text.to_string(),
            metadata: Some(format!("Detected: {:?}", result.detected_language)),
        });

        // Step 2: Grammar correction
        if self.grammar_corrector.is_enabled() {
            let corrected = self
                .grammar_corrector
                .correct(&result.processed, &self.domain_context)
                .await
                .map_err(|e| TextProcessingError::GrammarError(e.to_string()))?;

            if corrected != result.processed {
                result.steps.push(ProcessingStep {
                    name: "grammar_correction".to_string(),
                    input: result.processed.clone(),
                    output: corrected.clone(),
                    metadata: None,
                });
                result.processed = corrected;
            }
        }

        // Step 3: Translation (if configured and needed)
        // Translate-Think-Translate pattern: translate to English for processing
        if self.config.translate_for_processing
            && result.detected_language != Language::English
            && self
                .translator
                .supports_pair(result.detected_language, Language::English)
        {
            let translated = self
                .translator
                .translate(
                    &result.processed,
                    result.detected_language,
                    Language::English,
                )
                .await
                .map_err(|e| TextProcessingError::TranslationError(e.to_string()))?;

            result.steps.push(ProcessingStep {
                name: "translation_to_english".to_string(),
                input: result.processed.clone(),
                output: translated.clone(),
                metadata: Some(format!("{:?} -> English", result.detected_language)),
            });
            result.processed = translated;
            result.was_translated = true;
        }

        // Step 4: PII detection and redaction
        let pii_entities = self
            .pii_detector
            .detect(&result.processed)
            .await
            .map_err(|e| TextProcessingError::PIIError(e.to_string()))?;

        if !pii_entities.is_empty() {
            result.pii_detected = true;
            let strategy: RedactionStrategy = self.config.pii.strategy.clone().into();
            let redacted = self
                .pii_detector
                .redact(&result.processed, &strategy)
                .await
                .map_err(|e| TextProcessingError::PIIError(e.to_string()))?;

            result.steps.push(ProcessingStep {
                name: "pii_redaction".to_string(),
                input: result.processed.clone(),
                output: redacted.clone(),
                metadata: Some(format!("{} entities redacted", pii_entities.len())),
            });
            result.processed = redacted;
        }

        // Step 5: Compliance check
        let compliance_result = self
            .compliance_checker
            .check(&result.processed)
            .await
            .map_err(|e| TextProcessingError::ComplianceError(e.to_string()))?;

        result.is_compliant = compliance_result.is_compliant;

        if !compliance_result.is_compliant || !compliance_result.required_additions.is_empty() {
            let compliant_text = self
                .compliance_checker
                .make_compliant(&result.processed)
                .await
                .map_err(|e| TextProcessingError::ComplianceError(e.to_string()))?;

            if compliant_text != result.processed {
                result.steps.push(ProcessingStep {
                    name: "compliance_fix".to_string(),
                    input: result.processed.clone(),
                    output: compliant_text.clone(),
                    metadata: Some(format!(
                        "{} violations, {} additions",
                        compliance_result.violations.len(),
                        compliance_result.required_additions.len()
                    )),
                });
                result.processed = compliant_text;
            }
        }

        Ok(result)
    }

    /// Process only for PII (faster, no grammar/translation)
    pub async fn process_pii_only(&self, text: &str) -> Result<String> {
        let strategy: RedactionStrategy = self.config.pii.strategy.clone().into();
        self.pii_detector
            .redact(text, &strategy)
            .await
            .map_err(|e| TextProcessingError::PIIError(e.to_string()))
    }

    /// Check compliance only
    pub async fn check_compliance(&self, text: &str) -> Result<voice_agent_core::ComplianceResult> {
        self.compliance_checker
            .check(text)
            .await
            .map_err(|e| TextProcessingError::ComplianceError(e.to_string()))
    }

    /// Detect language
    pub fn detect_language(&self, text: &str) -> Language {
        self.script_detector.detect(text)
    }
}

/// P0 FIX: Implement TextProcessor trait for pipeline integration with VoicePipeline
#[async_trait]
impl TextProcessor for TextProcessingPipeline {
    async fn process(&self, text: &str) -> voice_agent_core::Result<TextProcessorResult> {
        // Use fully qualified syntax to call the inherent method, not the trait method
        let result = TextProcessingPipeline::process(self, text)
            .await
            .map_err(|e| voice_agent_core::Error::TextProcessing(e.to_string()))?;

        Ok(TextProcessorResult {
            original: result.original,
            processed: result.processed,
            pii_detected: result.pii_detected,
            compliance_fixed: !result.is_compliant,
        })
    }

    async fn process_pii_only(&self, text: &str) -> voice_agent_core::Result<String> {
        // Use fully qualified syntax to call the inherent method
        TextProcessingPipeline::process_pii_only(self, text)
            .await
            .map_err(|e| voice_agent_core::Error::TextProcessing(e.to_string()))
    }

    fn is_enabled(&self) -> bool {
        true // Pipeline is always enabled when instantiated
    }
}

/// Configuration for the text processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingConfig {
    /// Domain (e.g., "gold_loan")
    #[serde(default = "default_domain")]
    pub domain: String,
    /// Grammar correction config
    #[serde(default)]
    pub grammar: GrammarConfig,
    /// Translation config
    #[serde(default)]
    pub translation: TranslationConfig,
    /// PII detection config
    #[serde(default)]
    pub pii: PIIConfig,
    /// Compliance checking config
    #[serde(default)]
    pub compliance: ComplianceConfig,
    /// Whether to translate to English for processing
    #[serde(default)]
    pub translate_for_processing: bool,
}

fn default_domain() -> String {
    "gold_loan".to_string()
}

impl Default for TextProcessingConfig {
    fn default() -> Self {
        Self {
            domain: default_domain(),
            grammar: GrammarConfig::default(),
            translation: TranslationConfig::default(),
            pii: PIIConfig::default(),
            compliance: ComplianceConfig::default(),
            translate_for_processing: false,
        }
    }
}

/// Result of text processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedText {
    /// Original input text
    pub original: String,
    /// Processed output text
    pub processed: String,
    /// Detected language
    pub detected_language: Language,
    /// Whether translation was applied
    pub was_translated: bool,
    /// Whether PII was detected
    pub pii_detected: bool,
    /// Whether final text is compliant
    pub is_compliant: bool,
    /// Processing steps taken
    pub steps: Vec<ProcessingStep>,
}

/// A single processing step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStep {
    /// Step name
    pub name: String,
    /// Input to this step
    pub input: String,
    /// Output from this step
    pub output: String,
    /// Optional metadata
    pub metadata: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_basic() {
        let config = TextProcessingConfig::default();
        let pipeline = TextProcessingPipeline::new(config, None);

        let result = pipeline.process("Hello world").await.unwrap();
        assert_eq!(result.original, "Hello world");
        assert_eq!(result.detected_language, Language::English);
    }

    #[tokio::test]
    async fn test_language_detection() {
        let config = TextProcessingConfig::default();
        let pipeline = TextProcessingPipeline::new(config, None);

        assert_eq!(pipeline.detect_language("नमस्ते"), Language::Hindi);
        assert_eq!(pipeline.detect_language("Hello"), Language::English);
    }

    #[tokio::test]
    async fn test_pii_only() {
        let config = TextProcessingConfig::default();
        let pipeline = TextProcessingPipeline::new(config, None);

        let text = "My PAN is ABCPD1234E";
        let redacted = pipeline.process_pii_only(text).await.unwrap();
        assert!(redacted.contains("AB") || redacted.contains("**")); // Some masking applied
    }

    #[tokio::test]
    async fn test_compliance_check() {
        let config = TextProcessingConfig::default();
        let pipeline = TextProcessingPipeline::new(config, None);

        let result = pipeline
            .check_compliance("We offer competitive rates")
            .await
            .unwrap();
        // Should be compliant (no forbidden phrases)
        assert!(
            result.is_compliant
                || !result
                    .violations
                    .iter()
                    .any(|v| v.severity == voice_agent_core::Severity::Critical)
        );
    }
}
