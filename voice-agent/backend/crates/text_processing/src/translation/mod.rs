//! Translation module with script detection
//!
//! Supports the Translate-Think-Translate pattern for LLM reasoning.
//!
//! P3 FIX: Added IndicTrans2 ONNX-based translation for native Indic language support.

mod detect;
mod noop;
mod grpc;
mod indictrans2;

pub use detect::ScriptDetector;
pub use noop::NoopTranslator;
pub use grpc::{GrpcTranslator, GrpcTranslatorConfig, FallbackTranslator};
pub use indictrans2::{IndicTrans2Translator, IndicTrans2Config};

use voice_agent_core::{Translator, Language};
use std::sync::Arc;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Translation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Which provider to use
    pub provider: TranslationProvider,
    /// gRPC endpoint for fallback
    #[serde(default = "default_grpc_endpoint")]
    pub grpc_endpoint: String,
    /// Whether to fall back to gRPC if ONNX fails
    #[serde(default = "default_true")]
    pub fallback_to_grpc: bool,
    /// P3 FIX: IndicTrans2 model path (for ONNX provider)
    #[serde(default)]
    pub indictrans2_model_path: Option<PathBuf>,
}

fn default_grpc_endpoint() -> String {
    "http://localhost:50051".to_string()
}

fn default_true() -> bool {
    true
}

/// Translation providers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TranslationProvider {
    /// P3 FIX: ONNX-based IndicTrans2 translation (native, fastest)
    #[serde(alias = "onnx")]
    IndicTrans2,
    /// gRPC-based translation (Python sidecar)
    Grpc,
    /// Disabled (pass-through)
    #[default]
    Disabled,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            provider: TranslationProvider::Disabled,
            grpc_endpoint: default_grpc_endpoint(),
            fallback_to_grpc: true,
            indictrans2_model_path: None,
        }
    }
}

/// Create translator based on config
pub fn create_translator(config: &TranslationConfig) -> Arc<dyn Translator> {
    match config.provider {
        TranslationProvider::IndicTrans2 => {
            // P3 FIX: Create IndicTrans2 ONNX translator
            let indictrans2_config = if let Some(ref model_path) = config.indictrans2_model_path {
                IndicTrans2Config {
                    encoder_path: model_path.join("encoder.onnx"),
                    decoder_path: model_path.join("decoder.onnx"),
                    tokenizer_path: model_path.join("tokenizer"),
                    ..Default::default()
                }
            } else {
                IndicTrans2Config::default()
            };

            match IndicTrans2Translator::new(indictrans2_config) {
                Ok(translator) => {
                    let primary = Arc::new(translator);

                    // Wrap with fallback if enabled
                    if config.fallback_to_grpc {
                        tracing::info!("Using IndicTrans2 with gRPC fallback");
                        let grpc_config = GrpcTranslatorConfig {
                            endpoint: config.grpc_endpoint.clone(),
                            ..Default::default()
                        };
                        let fallback = Arc::new(GrpcTranslator::new(grpc_config));
                        Arc::new(FallbackTranslator::new(primary, fallback))
                    } else {
                        tracing::info!("Using IndicTrans2 (no fallback)");
                        primary
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to load IndicTrans2, falling back to gRPC"
                    );
                    // Fall back to gRPC if ONNX fails to load
                    let grpc_config = GrpcTranslatorConfig {
                        endpoint: config.grpc_endpoint.clone(),
                        ..Default::default()
                    };
                    Arc::new(GrpcTranslator::new(grpc_config))
                }
            }
        }
        TranslationProvider::Grpc => {
            let grpc_config = GrpcTranslatorConfig {
                endpoint: config.grpc_endpoint.clone(),
                ..Default::default()
            };
            let grpc_translator = Arc::new(GrpcTranslator::new(grpc_config));

            if config.fallback_to_grpc {
                tracing::info!(
                    endpoint = %config.grpc_endpoint,
                    "Using gRPC translator with fallback enabled"
                );
            } else {
                tracing::info!(
                    endpoint = %config.grpc_endpoint,
                    "Using gRPC translator (fallback disabled)"
                );
            }
            grpc_translator
        }
        TranslationProvider::Disabled => Arc::new(NoopTranslator::new()),
    }
}

/// Create a fallback translator that tries ONNX first, then gRPC
pub fn create_fallback_translator(
    primary: Arc<dyn Translator>,
    config: &TranslationConfig,
) -> Arc<dyn Translator> {
    if config.fallback_to_grpc && matches!(config.provider, TranslationProvider::Grpc) {
        let grpc_config = GrpcTranslatorConfig {
            endpoint: config.grpc_endpoint.clone(),
            ..Default::default()
        };
        let fallback = Arc::new(GrpcTranslator::new(grpc_config));
        Arc::new(FallbackTranslator::new(primary, fallback))
    } else {
        primary
    }
}

/// Supported translation pairs
///
/// P0 FIX: Added all 22 scheduled Indian languages to/from English.
/// IndicTrans2 supports all these language pairs.
pub fn supported_pairs() -> Vec<(Language, Language)> {
    vec![
        // === Indic to English (22 languages) ===
        // Major languages (existing)
        (Language::Hindi, Language::English),
        (Language::Tamil, Language::English),
        (Language::Telugu, Language::English),
        (Language::Bengali, Language::English),
        (Language::Marathi, Language::English),
        (Language::Gujarati, Language::English),
        (Language::Kannada, Language::English),
        (Language::Malayalam, Language::English),
        (Language::Punjabi, Language::English),
        (Language::Odia, Language::English),
        // P0 FIX: Additional 12 scheduled languages
        (Language::Assamese, Language::English),
        (Language::Urdu, Language::English),
        (Language::Kashmiri, Language::English),
        (Language::Sindhi, Language::English),
        (Language::Konkani, Language::English),
        (Language::Dogri, Language::English),
        (Language::Bodo, Language::English),
        (Language::Maithili, Language::English),
        (Language::Santali, Language::English),
        (Language::Nepali, Language::English),
        (Language::Manipuri, Language::English),
        (Language::Sanskrit, Language::English),

        // === English to Indic (22 languages) ===
        // Major languages (existing)
        (Language::English, Language::Hindi),
        (Language::English, Language::Tamil),
        (Language::English, Language::Telugu),
        (Language::English, Language::Bengali),
        (Language::English, Language::Marathi),
        (Language::English, Language::Gujarati),
        (Language::English, Language::Kannada),
        (Language::English, Language::Malayalam),
        (Language::English, Language::Punjabi),
        (Language::English, Language::Odia),
        // P0 FIX: Additional 12 scheduled languages
        (Language::English, Language::Assamese),
        (Language::English, Language::Urdu),
        (Language::English, Language::Kashmiri),
        (Language::English, Language::Sindhi),
        (Language::English, Language::Konkani),
        (Language::English, Language::Dogri),
        (Language::English, Language::Bodo),
        (Language::English, Language::Maithili),
        (Language::English, Language::Santali),
        (Language::English, Language::Nepali),
        (Language::English, Language::Manipuri),
        (Language::English, Language::Sanskrit),
    ]
}

/// Check if a language pair is supported
pub fn is_pair_supported(from: Language, to: Language) -> bool {
    supported_pairs().contains(&(from, to))
}

/// Get all supported source languages (can translate FROM)
pub fn supported_source_languages() -> Vec<Language> {
    vec![
        Language::English,
        Language::Hindi,
        Language::Tamil,
        Language::Telugu,
        Language::Bengali,
        Language::Marathi,
        Language::Gujarati,
        Language::Kannada,
        Language::Malayalam,
        Language::Punjabi,
        Language::Odia,
        Language::Assamese,
        Language::Urdu,
        Language::Kashmiri,
        Language::Sindhi,
        Language::Konkani,
        Language::Dogri,
        Language::Bodo,
        Language::Maithili,
        Language::Santali,
        Language::Nepali,
        Language::Manipuri,
        Language::Sanskrit,
    ]
}

/// Get all supported target languages (can translate TO)
pub fn supported_target_languages() -> Vec<Language> {
    // Same as source - bidirectional support
    supported_source_languages()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TranslationConfig::default();
        assert!(matches!(config.provider, TranslationProvider::Disabled));
    }

    #[test]
    fn test_supported_pairs() {
        let pairs = supported_pairs();
        // Original languages
        assert!(pairs.contains(&(Language::Hindi, Language::English)));
        assert!(pairs.contains(&(Language::English, Language::Hindi)));
        // P0 FIX: Verify new language pairs
        assert!(pairs.contains(&(Language::Assamese, Language::English)));
        assert!(pairs.contains(&(Language::English, Language::Urdu)));
        assert!(pairs.contains(&(Language::Sanskrit, Language::English)));
        assert!(pairs.contains(&(Language::English, Language::Nepali)));
        // Total should be 44 (22 languages × 2 directions)
        assert_eq!(pairs.len(), 44);
    }

    #[test]
    fn test_is_pair_supported() {
        assert!(is_pair_supported(Language::Hindi, Language::English));
        assert!(is_pair_supported(Language::English, Language::Sanskrit));
        // Non-English pairs are not supported (only via English as pivot)
        assert!(!is_pair_supported(Language::Hindi, Language::Tamil));
    }

    #[test]
    fn test_supported_languages() {
        let sources = supported_source_languages();
        let targets = supported_target_languages();
        // 22 Indian languages + English = 23 total
        assert_eq!(sources.len(), 23);
        assert_eq!(targets.len(), 23);
        assert!(sources.contains(&Language::English));
        assert!(sources.contains(&Language::Sanskrit));
    }
}
