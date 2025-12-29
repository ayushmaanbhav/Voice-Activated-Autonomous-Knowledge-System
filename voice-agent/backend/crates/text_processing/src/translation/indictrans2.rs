//! IndicTrans2 ONNX Translator
//!
//! P3 FIX: Native ONNX-based translation for Indic languages.
//!
//! IndicTrans2 is AI4Bharat's multilingual translation model supporting
//! all 22 scheduled Indian languages + English.
//!
//! Model architecture:
//! - Encoder-decoder transformer
//! - BPE tokenization with shared vocabulary
//! - Language-specific tokens for source/target language indication
//!
//! References:
//! - https://github.com/AI4Bharat/IndicTrans2
//! - https://huggingface.co/ai4bharat/indictrans2-en-indic-1B

use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use parking_lot::RwLock;

use voice_agent_core::{Translator, Language, Result, Error};

use super::ScriptDetector;
use super::supported_pairs;

/// IndicTrans2 configuration
#[derive(Debug, Clone)]
pub struct IndicTrans2Config {
    /// Path to encoder ONNX model
    pub encoder_path: PathBuf,
    /// Path to decoder ONNX model
    pub decoder_path: PathBuf,
    /// Path to tokenizer files
    pub tokenizer_path: PathBuf,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Enable caching
    pub cache_enabled: bool,
    /// Maximum cache entries
    pub cache_size: usize,
    /// Number of inference threads
    pub num_threads: usize,
}

impl Default for IndicTrans2Config {
    fn default() -> Self {
        Self {
            encoder_path: PathBuf::from("models/translation/indictrans2/encoder.onnx"),
            decoder_path: PathBuf::from("models/translation/indictrans2/decoder.onnx"),
            tokenizer_path: PathBuf::from("models/translation/indictrans2/tokenizer"),
            max_seq_length: 256,
            cache_enabled: true,
            cache_size: 1000,
            num_threads: 1,
        }
    }
}

/// Language code mapping for IndicTrans2
///
/// IndicTrans2 uses ISO 639-1 codes with script suffixes
fn language_to_indictrans_code(lang: Language) -> &'static str {
    match lang {
        Language::Hindi => "hin_Deva",
        Language::English => "eng_Latn",
        Language::Tamil => "tam_Taml",
        Language::Telugu => "tel_Telu",
        Language::Bengali => "ben_Beng",
        Language::Marathi => "mar_Deva",
        Language::Gujarati => "guj_Gujr",
        Language::Kannada => "kan_Knda",
        Language::Malayalam => "mal_Mlym",
        Language::Punjabi => "pan_Guru",
        Language::Odia => "ory_Orya",
        Language::Assamese => "asm_Beng",
        Language::Konkani => "kok_Deva",
        Language::Maithili => "mai_Deva",
        Language::Nepali => "npi_Deva",
        Language::Sanskrit => "san_Deva",
        Language::Sindhi => "snd_Arab",
        Language::Urdu => "urd_Arab",
        Language::Kashmiri => "kas_Arab",
        Language::Dogri => "doi_Deva",
        Language::Bodo => "brx_Deva",
        Language::Santali => "sat_Olck",
        Language::Manipuri => "mni_Beng",
    }
}

/// Translation cache entry
struct CacheEntry {
    translation: String,
    #[allow(dead_code)]
    timestamp: std::time::Instant,
}

/// Translation cache
struct TranslationCache {
    entries: std::collections::HashMap<String, CacheEntry>,
    max_size: usize,
}

impl TranslationCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            max_size,
        }
    }

    fn make_key(text: &str, from: Language, to: Language) -> String {
        format!("{}:{}:{}", from, to, text)
    }

    fn get(&self, text: &str, from: Language, to: Language) -> Option<&str> {
        let key = Self::make_key(text, from, to);
        self.entries.get(&key).map(|e| e.translation.as_str())
    }

    fn insert(&mut self, text: &str, from: Language, to: Language, translation: String) {
        // Simple eviction: clear half when full
        if self.entries.len() >= self.max_size {
            let keys_to_remove: Vec<_> = self.entries.keys()
                .take(self.max_size / 2)
                .cloned()
                .collect();
            for key in keys_to_remove {
                self.entries.remove(&key);
            }
        }

        let key = Self::make_key(text, from, to);
        self.entries.insert(key, CacheEntry {
            translation,
            timestamp: std::time::Instant::now(),
        });
    }
}

// ============================================================================
// ONNX Implementation (feature-gated)
// ============================================================================

#[cfg(feature = "onnx")]
mod onnx_impl {
    use super::*;
    use ort::{GraphOptimizationLevel, Session};
    use tokenizers::Tokenizer;

    /// IndicTrans2 ONNX-based translator
    pub struct IndicTrans2Translator {
        encoder: Session,
        decoder: Session,
        tokenizer: Tokenizer,
        config: IndicTrans2Config,
        detector: ScriptDetector,
        cache: RwLock<TranslationCache>,
    }

    impl IndicTrans2Translator {
        /// Create a new IndicTrans2 translator
        pub fn new(config: IndicTrans2Config) -> Result<Self> {
            // Load encoder
            let encoder = Session::builder()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .with_intra_threads(config.num_threads)
                .commit_from_file(&config.encoder_path)
                .map_err(|e| Error::Translation(format!("Failed to load encoder: {}", e)))?;

            // Load decoder
            let decoder = Session::builder()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .with_intra_threads(config.num_threads)
                .commit_from_file(&config.decoder_path)
                .map_err(|e| Error::Translation(format!("Failed to load decoder: {}", e)))?;

            // Load tokenizer
            let tokenizer_file = config.tokenizer_path.join("tokenizer.json");
            let tokenizer = Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| Error::Translation(format!("Failed to load tokenizer: {}", e)))?;

            let cache = RwLock::new(TranslationCache::new(config.cache_size));

            Ok(Self {
                encoder,
                decoder,
                tokenizer,
                config,
                detector: ScriptDetector::new(),
                cache,
            })
        }

        /// Translate text using ONNX inference
        async fn translate_onnx(&self, text: &str, from: Language, to: Language) -> Result<String> {
            // Prepare input with language tokens
            let src_code = language_to_indictrans_code(from);
            let tgt_code = language_to_indictrans_code(to);
            let input_text = format!("__{}__ {} __{}__", src_code, text, tgt_code);

            // Tokenize
            let encoding = self.tokenizer.encode(input_text, true)
                .map_err(|e| Error::Translation(format!("Tokenization failed: {}", e)))?;

            let input_ids: Vec<i64> = encoding.get_ids()
                .iter()
                .map(|&id| id as i64)
                .collect();

            // Create input tensor
            let seq_len = input_ids.len();
            let input_array = ndarray::Array2::from_shape_vec(
                (1, seq_len),
                input_ids,
            ).map_err(|e| Error::Translation(format!("Array creation failed: {}", e)))?;

            // Run encoder
            let encoder_outputs = self.encoder.run(ort::inputs![
                "input_ids" => input_array.view(),
            ]).map_err(|e| Error::Translation(format!("Encoder inference failed: {}", e)))?;

            // Get encoder hidden states
            let encoder_hidden = encoder_outputs.get("last_hidden_state")
                .ok_or_else(|| Error::Translation("Missing encoder output".to_string()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| Error::Translation(format!("Failed to extract encoder output: {}", e)))?;

            // Run decoder (simplified - actual implementation would use beam search)
            // For now, use greedy decoding
            let mut output_ids = vec![self.get_bos_token_id()];
            let max_length = self.config.max_seq_length.min(seq_len * 2);

            for _ in 0..max_length {
                let decoder_input = ndarray::Array2::from_shape_vec(
                    (1, output_ids.len()),
                    output_ids.iter().map(|&id| id as i64).collect(),
                ).map_err(|e| Error::Translation(format!("Decoder input creation failed: {}", e)))?;

                let decoder_outputs = self.decoder.run(ort::inputs![
                    "input_ids" => decoder_input.view(),
                    "encoder_hidden_states" => encoder_hidden.view(),
                ]).map_err(|e| Error::Translation(format!("Decoder inference failed: {}", e)))?;

                let logits = decoder_outputs.get("logits")
                    .ok_or_else(|| Error::Translation("Missing decoder logits".to_string()))?
                    .try_extract_tensor::<f32>()
                    .map_err(|e| Error::Translation(format!("Failed to extract logits: {}", e)))?;

                // Greedy decode: take argmax of last position
                let last_logits = logits.slice(ndarray::s![0, -1, ..]);
                let next_token = last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(self.get_eos_token_id());

                if next_token == self.get_eos_token_id() {
                    break;
                }

                output_ids.push(next_token);
            }

            // Decode output tokens
            let output_tokens: Vec<u32> = output_ids.iter().map(|&id| id as u32).collect();
            let translation = self.tokenizer.decode(&output_tokens, true)
                .map_err(|e| Error::Translation(format!("Decoding failed: {}", e)))?;

            // Clean up language tokens from output
            let cleaned = translation
                .replace(&format!("__{}__", tgt_code), "")
                .trim()
                .to_string();

            Ok(cleaned)
        }

        fn get_bos_token_id(&self) -> i64 {
            // IndicTrans2 uses token ID 0 for BOS typically
            // This should be read from tokenizer config
            0
        }

        fn get_eos_token_id(&self) -> i64 {
            // IndicTrans2 uses token ID 2 for EOS typically
            // This should be read from tokenizer config
            2
        }
    }

    #[async_trait]
    impl Translator for IndicTrans2Translator {
        async fn translate(
            &self,
            text: &str,
            from: Language,
            to: Language,
        ) -> Result<String> {
            // Short-circuit if same language
            if from == to {
                return Ok(text.to_string());
            }

            // Check if pair is supported
            if !self.supports_pair(from, to) {
                tracing::warn!(
                    from = ?from,
                    to = ?to,
                    "Translation pair not supported, passing through"
                );
                return Ok(text.to_string());
            }

            // Check cache first
            if self.config.cache_enabled {
                let cache = self.cache.read();
                if let Some(cached) = cache.get(text, from, to) {
                    tracing::trace!("Translation cache hit");
                    return Ok(cached.to_string());
                }
            }

            // Translate
            let translation = self.translate_onnx(text, from, to).await?;

            // Update cache
            if self.config.cache_enabled {
                let mut cache = self.cache.write();
                cache.insert(text, from, to, translation.clone());
            }

            Ok(translation)
        }

        async fn detect_language(&self, text: &str) -> Result<Language> {
            Ok(self.detector.detect(text))
        }

        fn translate_stream<'a>(
            &'a self,
            text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
            from: Language,
            to: Language,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            use futures::StreamExt;

            // For streaming, translate each chunk as it arrives
            Box::pin(text_stream.then(move |text| async move {
                self.translate(&text, from, to).await
            }))
        }

        fn supports_pair(&self, from: Language, to: Language) -> bool {
            supported_pairs().contains(&(from, to))
        }

        fn name(&self) -> &str {
            "indictrans2-onnx"
        }
    }
}

// ============================================================================
// Stub Implementation (when ONNX feature is disabled)
// ============================================================================

#[cfg(not(feature = "onnx"))]
mod stub_impl {
    use super::*;

    /// Stub IndicTrans2 translator (ONNX feature not enabled)
    ///
    /// Returns original text and logs a warning.
    pub struct IndicTrans2Translator {
        detector: ScriptDetector,
        cache: RwLock<TranslationCache>,
        config: IndicTrans2Config,
    }

    impl IndicTrans2Translator {
        /// Create a new stub translator
        pub fn new(config: IndicTrans2Config) -> Result<Self> {
            tracing::warn!(
                "IndicTrans2 ONNX feature not enabled - translation will pass through"
            );

            Ok(Self {
                detector: ScriptDetector::new(),
                cache: RwLock::new(TranslationCache::new(config.cache_size)),
                config,
            })
        }
    }

    #[async_trait]
    impl Translator for IndicTrans2Translator {
        async fn translate(
            &self,
            text: &str,
            from: Language,
            to: Language,
        ) -> Result<String> {
            if from == to {
                return Ok(text.to_string());
            }

            tracing::debug!(
                from = ?from,
                to = ?to,
                "IndicTrans2 ONNX not available - passing through text"
            );

            Ok(text.to_string())
        }

        async fn detect_language(&self, text: &str) -> Result<Language> {
            Ok(self.detector.detect(text))
        }

        fn translate_stream<'a>(
            &'a self,
            text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
            from: Language,
            to: Language,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            use futures::StreamExt;

            Box::pin(text_stream.then(move |text| async move {
                self.translate(&text, from, to).await
            }))
        }

        fn supports_pair(&self, from: Language, to: Language) -> bool {
            supported_pairs().contains(&(from, to))
        }

        fn name(&self) -> &str {
            "indictrans2-stub"
        }
    }
}

// Re-export the appropriate implementation
#[cfg(feature = "onnx")]
pub use onnx_impl::IndicTrans2Translator;

#[cfg(not(feature = "onnx"))]
pub use stub_impl::IndicTrans2Translator;

// Always export config
pub use self::IndicTrans2Config as Config;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_codes() {
        assert_eq!(language_to_indictrans_code(Language::Hindi), "hin_Deva");
        assert_eq!(language_to_indictrans_code(Language::English), "eng_Latn");
        assert_eq!(language_to_indictrans_code(Language::Tamil), "tam_Taml");
    }

    #[test]
    fn test_config_default() {
        let config = IndicTrans2Config::default();
        assert!(config.cache_enabled);
        assert_eq!(config.max_seq_length, 256);
    }

    #[test]
    fn test_cache() {
        let mut cache = TranslationCache::new(10);
        cache.insert("hello", Language::English, Language::Hindi, "नमस्ते".to_string());

        let result = cache.get("hello", Language::English, Language::Hindi);
        assert_eq!(result, Some("नमस्ते"));

        // Different direction should not match
        let result = cache.get("hello", Language::Hindi, Language::English);
        assert!(result.is_none());
    }
}
