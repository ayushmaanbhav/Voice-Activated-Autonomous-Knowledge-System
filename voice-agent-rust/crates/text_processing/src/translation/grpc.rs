//! gRPC/HTTP Translator Fallback
//!
//! Provides fallback translation when ONNX translation fails.
//! Currently uses HTTP/JSON for simplicity; can be upgraded to gRPC.

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use voice_agent_core::{Translator, Language, Result};

use super::ScriptDetector;
use super::supported_pairs;

/// gRPC/HTTP translator configuration
#[derive(Debug, Clone)]
pub struct GrpcTranslatorConfig {
    /// Endpoint URL (http://host:port)
    pub endpoint: String,
    /// Request timeout
    pub timeout: Duration,
    /// Max retries on failure
    pub max_retries: u32,
    /// Enable caching
    pub cache_enabled: bool,
    /// Max cache entries
    pub cache_size: usize,
}

impl Default for GrpcTranslatorConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:50051".to_string(),
            timeout: Duration::from_secs(10),
            max_retries: 2,
            cache_enabled: true,
            cache_size: 1000,
        }
    }
}

/// Simple LRU cache for translations
struct TranslationCache {
    entries: std::collections::HashMap<String, CacheEntry>,
    max_size: usize,
}

struct CacheEntry {
    translation: String,
    #[allow(dead_code)]
    timestamp: std::time::Instant,
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

/// Translation service client using HTTP/JSON
///
/// Calls a Python sidecar service for translation.
/// API format:
/// POST /translate
/// { "text": "...", "from": "hi", "to": "en" }
/// Response: { "translation": "..." }
///
/// NOTE: Actual HTTP client (reqwest) should be added when translation
/// service is deployed. Currently returns original text as placeholder.
pub struct GrpcTranslator {
    config: GrpcTranslatorConfig,
    detector: ScriptDetector,
    cache: RwLock<TranslationCache>,
}

impl GrpcTranslator {
    /// Create a new gRPC/HTTP translator
    pub fn new(config: GrpcTranslatorConfig) -> Self {
        let cache = RwLock::new(TranslationCache::new(config.cache_size));

        Self {
            config,
            detector: ScriptDetector::new(),
            cache,
        }
    }

    /// Call the translation service
    ///
    /// NOTE: Placeholder implementation. Add reqwest dependency and implement
    /// HTTP client when translation service is deployed.
    async fn call_service(&self, text: &str, from: Language, to: Language) -> Result<String> {
        // Log the translation request
        tracing::info!(
            endpoint = %self.config.endpoint,
            from = ?from,
            to = ?to,
            text_len = text.len(),
            "Translation service called (stub - returning original text)"
        );

        // TODO: Implement actual HTTP client call when service is deployed
        // The API format will be:
        // POST {endpoint}/translate
        // Request: { "text": "...", "from": "hi", "to": "en" }
        // Response: { "translation": "..." }

        // For now, return the original text
        Ok(text.to_string())
    }

    /// Translate with caching
    async fn translate_with_cache(
        &self,
        text: &str,
        from: Language,
        to: Language,
    ) -> Result<String> {
        // Check cache first
        if self.config.cache_enabled {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.get(text, from, to) {
                tracing::trace!("Translation cache hit");
                return Ok(cached.to_string());
            }
        }

        // Call service
        let translation = self.call_service(text, from, to).await?;

        // Update cache
        if self.config.cache_enabled {
            let mut cache = self.cache.write().await;
            cache.insert(text, from, to, translation.clone());
        }

        Ok(translation)
    }
}

#[async_trait]
impl Translator for GrpcTranslator {
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

        self.translate_with_cache(text, from, to).await
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

        // For streaming, we translate each chunk as it arrives
        Box::pin(text_stream.then(move |text| async move {
            self.translate(&text, from, to).await
        }))
    }

    fn supports_pair(&self, from: Language, to: Language) -> bool {
        supported_pairs().contains(&(from, to))
    }

    fn name(&self) -> &str {
        "grpc-translator"
    }
}

/// Fallback translator that tries primary first, then falls back to secondary
pub struct FallbackTranslator {
    primary: Arc<dyn Translator>,
    fallback: Arc<dyn Translator>,
}

impl FallbackTranslator {
    /// Create a new fallback translator
    pub fn new(primary: Arc<dyn Translator>, fallback: Arc<dyn Translator>) -> Self {
        Self { primary, fallback }
    }
}

#[async_trait]
impl Translator for FallbackTranslator {
    async fn translate(
        &self,
        text: &str,
        from: Language,
        to: Language,
    ) -> Result<String> {
        // Try primary first
        match self.primary.translate(text, from, to).await {
            Ok(translation) => {
                tracing::trace!("Primary translator succeeded");
                Ok(translation)
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Primary translator failed, trying fallback"
                );
                // Fall back to secondary
                self.fallback.translate(text, from, to).await
            }
        }
    }

    async fn detect_language(&self, text: &str) -> Result<Language> {
        // Language detection should work on both - prefer primary
        match self.primary.detect_language(text).await {
            Ok(lang) => Ok(lang),
            Err(_) => self.fallback.detect_language(text).await,
        }
    }

    fn translate_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        from: Language,
        to: Language,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        use futures::StreamExt;

        // Stream using primary with per-chunk fallback
        let primary = Arc::clone(&self.primary);
        let fallback = Arc::clone(&self.fallback);

        Box::pin(text_stream.then(move |text| {
            let primary = Arc::clone(&primary);
            let fallback = Arc::clone(&fallback);
            async move {
                match primary.translate(&text, from, to).await {
                    Ok(t) => Ok(t),
                    Err(_) => fallback.translate(&text, from, to).await,
                }
            }
        }))
    }

    fn supports_pair(&self, from: Language, to: Language) -> bool {
        self.primary.supports_pair(from, to) || self.fallback.supports_pair(from, to)
    }

    fn name(&self) -> &str {
        "fallback-translator"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = GrpcTranslatorConfig::default();
        assert_eq!(config.endpoint, "http://localhost:50051");
        assert!(config.cache_enabled);
    }

    #[tokio::test]
    async fn test_grpc_translator_creation() {
        let translator = GrpcTranslator::new(GrpcTranslatorConfig::default());
        assert!(translator.supports_pair(Language::Hindi, Language::English));
        assert!(!translator.supports_pair(Language::English, Language::English));
    }

    #[tokio::test]
    async fn test_same_language_passthrough() {
        let translator = GrpcTranslator::new(GrpcTranslatorConfig::default());
        let result = translator.translate("Hello", Language::English, Language::English).await.unwrap();
        assert_eq!(result, "Hello");
    }

    #[tokio::test]
    async fn test_language_detection() {
        let translator = GrpcTranslator::new(GrpcTranslatorConfig::default());

        let lang = translator.detect_language("नमस्ते").await.unwrap();
        assert_eq!(lang, Language::Hindi);

        let lang = translator.detect_language("Hello").await.unwrap();
        assert_eq!(lang, Language::English);
    }

    #[tokio::test]
    async fn test_fallback_translator() {
        use super::super::NoopTranslator;

        let primary = Arc::new(NoopTranslator::new());
        let fallback = Arc::new(NoopTranslator::new());

        let translator = FallbackTranslator::new(primary, fallback);

        let result = translator.translate("Hello", Language::Hindi, Language::English).await.unwrap();
        assert_eq!(result, "Hello"); // Noop just returns the input
    }
}
