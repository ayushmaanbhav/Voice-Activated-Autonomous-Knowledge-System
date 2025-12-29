//! Main settings module

use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{AgentConfig, ConfigError, GoldLoanConfig, PipelineConfig};

/// P1 FIX: Runtime environment enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeEnvironment {
    /// Development mode - relaxed validation, warnings only
    #[default]
    Development,
    /// Staging mode - stricter validation
    Staging,
    /// Production mode - all validations enforced
    Production,
}

impl RuntimeEnvironment {
    /// Check if this is a production environment
    pub fn is_production(&self) -> bool {
        matches!(self, Self::Production)
    }

    /// Check if strict validation should be applied
    pub fn is_strict(&self) -> bool {
        matches!(self, Self::Production | Self::Staging)
    }
}

/// Main application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Settings {
    /// P1 FIX: Runtime environment (development, staging, production)
    #[serde(default)]
    pub environment: RuntimeEnvironment,

    /// Server configuration
    #[serde(default)]
    pub server: ServerConfig,

    /// Pipeline configuration
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// Agent configuration
    #[serde(default)]
    pub agent: AgentConfig,

    /// Gold loan business configuration
    #[serde(default)]
    pub gold_loan: GoldLoanConfig,

    /// Model paths
    #[serde(default)]
    pub models: ModelPaths,

    /// Observability configuration
    #[serde(default)]
    pub observability: ObservabilityConfig,

    /// Feature flags
    #[serde(default)]
    pub features: FeatureFlags,

    /// P4 FIX: Path to domain configuration file (YAML or JSON)
    #[serde(default = "default_domain_config_path")]
    pub domain_config_path: String,

    /// P5 FIX: RAG configuration for retrieval and reranking
    #[serde(default)]
    pub rag: RagConfig,
}

fn default_domain_config_path() -> String {
    "config/domain.yaml".to_string()
}

impl Settings {
    /// Create default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate settings
    pub fn validate(&self) -> Result<(), ConfigError> {
        // P2 FIX: Improved model path validation - check all paths and extensions
        self.validate_model_paths()?;

        // P1 FIX: Comprehensive validation of all config sections
        self.validate_pipeline()?;
        self.validate_rag()?;
        self.validate_server()?;

        Ok(())
    }

    /// P1 FIX: Validate pipeline configuration
    fn validate_pipeline(&self) -> Result<(), ConfigError> {
        // Validate latency budget
        if self.pipeline.latency_budget_ms < 200 {
            return Err(ConfigError::InvalidValue {
                field: "pipeline.latency_budget_ms".to_string(),
                message: "Latency budget too low (minimum 200ms)".to_string(),
            });
        }

        if self.pipeline.latency_budget_ms > 10000 {
            return Err(ConfigError::InvalidValue {
                field: "pipeline.latency_budget_ms".to_string(),
                message: "Latency budget too high (maximum 10000ms)".to_string(),
            });
        }

        Ok(())
    }

    /// P1 FIX: Validate RAG configuration
    fn validate_rag(&self) -> Result<(), ConfigError> {
        let rag = &self.rag;

        // Validate weight is in [0, 1]
        if !(0.0..=1.0).contains(&rag.dense_weight) {
            return Err(ConfigError::InvalidValue {
                field: "rag.dense_weight".to_string(),
                message: format!("Must be between 0.0 and 1.0, got {}", rag.dense_weight),
            });
        }

        // Validate thresholds are in valid ranges
        if !(0.0..=1.0).contains(&rag.min_score) {
            return Err(ConfigError::InvalidValue {
                field: "rag.min_score".to_string(),
                message: format!("Must be between 0.0 and 1.0, got {}", rag.min_score),
            });
        }

        if !(0.0..=1.0).contains(&rag.prefilter_threshold) {
            return Err(ConfigError::InvalidValue {
                field: "rag.prefilter_threshold".to_string(),
                message: format!("Must be between 0.0 and 1.0, got {}", rag.prefilter_threshold),
            });
        }

        if !(0.0..=1.0).contains(&rag.early_termination_threshold) {
            return Err(ConfigError::InvalidValue {
                field: "rag.early_termination_threshold".to_string(),
                message: format!("Must be between 0.0 and 1.0, got {}", rag.early_termination_threshold),
            });
        }

        if !(0.0..=1.0).contains(&rag.prefetch_confidence_threshold) {
            return Err(ConfigError::InvalidValue {
                field: "rag.prefetch_confidence_threshold".to_string(),
                message: format!("Must be between 0.0 and 1.0, got {}", rag.prefetch_confidence_threshold),
            });
        }

        // Validate RRF k parameter (must be positive)
        if rag.rrf_k <= 0.0 {
            return Err(ConfigError::InvalidValue {
                field: "rag.rrf_k".to_string(),
                message: format!("Must be positive, got {}", rag.rrf_k),
            });
        }

        // Validate top-k values are reasonable
        if rag.final_top_k == 0 {
            return Err(ConfigError::InvalidValue {
                field: "rag.final_top_k".to_string(),
                message: "Must be at least 1".to_string(),
            });
        }

        if rag.final_top_k > rag.dense_top_k && rag.final_top_k > rag.sparse_top_k {
            tracing::warn!(
                "rag.final_top_k ({}) is larger than both dense_top_k ({}) and sparse_top_k ({}), \
                 results will be limited by retrieval",
                rag.final_top_k, rag.dense_top_k, rag.sparse_top_k
            );
        }

        // Validate early termination consistency
        if rag.early_termination_min_results > rag.max_full_model_docs {
            return Err(ConfigError::InvalidValue {
                field: "rag.early_termination_min_results".to_string(),
                message: format!(
                    "Cannot be larger than max_full_model_docs ({})",
                    rag.max_full_model_docs
                ),
            });
        }

        Ok(())
    }

    /// P1 FIX: Validate server configuration
    fn validate_server(&self) -> Result<(), ConfigError> {
        let server = &self.server;

        // Validate port
        if server.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.port".to_string(),
                message: "Port cannot be 0".to_string(),
            });
        }

        // Validate max connections
        if server.max_connections == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.max_connections".to_string(),
                message: "Max connections must be at least 1".to_string(),
            });
        }

        // Validate timeout
        if server.timeout_seconds == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.timeout_seconds".to_string(),
                message: "Timeout must be at least 1 second".to_string(),
            });
        }

        // Rate limit validation
        let rate_limit = &server.rate_limit;
        if rate_limit.enabled {
            if rate_limit.messages_per_second == 0 {
                return Err(ConfigError::InvalidValue {
                    field: "server.rate_limit.messages_per_second".to_string(),
                    message: "Must be at least 1 when rate limiting is enabled".to_string(),
                });
            }

            if rate_limit.burst_multiplier < 1.0 {
                return Err(ConfigError::InvalidValue {
                    field: "server.rate_limit.burst_multiplier".to_string(),
                    message: format!("Must be at least 1.0, got {}", rate_limit.burst_multiplier),
                });
            }
        }

        // Auth validation in production
        if self.environment.is_production() && server.auth.enabled {
            if server.auth.api_key.is_none() {
                return Err(ConfigError::InvalidValue {
                    field: "server.auth.api_key".to_string(),
                    message: "API key must be set when auth is enabled in production".to_string(),
                });
            }
        }

        // CORS validation in production
        if self.environment.is_production() && server.cors_enabled && server.cors_origins.is_empty() {
            tracing::warn!(
                "CORS is enabled in production but no origins are configured. \
                 This may block legitimate requests."
            );
        }

        Ok(())
    }

    /// P1 FIX: Validate all model paths with environment-aware strictness
    ///
    /// In production/staging: Missing required models cause errors
    /// In development: Missing models only cause warnings
    fn validate_model_paths(&self) -> Result<(), ConfigError> {
        // Required models - must exist in production
        let required_models = [
            ("models.vad", &self.models.vad, Some(".onnx")),
            ("models.stt", &self.models.stt, Some(".onnx")),
            ("models.tts", &self.models.tts, Some(".onnx")),
        ];

        // Optional models - warnings only
        let optional_models = [
            ("models.turn_detection", &self.models.turn_detection, Some(".onnx")),
            ("models.turn_detection_tokenizer", &self.models.turn_detection_tokenizer, Some(".json")),
            ("models.stt_tokens", &self.models.stt_tokens, Some(".txt")),
            ("models.reranker", &self.models.reranker, Some(".onnx")),
            ("models.embeddings", &self.models.embeddings, Some(".onnx")),
        ];

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Validate required models
        for (field, path, expected_ext) in required_models {
            if path.is_empty() {
                if self.environment.is_strict() {
                    errors.push(format!("{}: path is required in {} mode", field,
                        if self.environment.is_production() { "production" } else { "staging" }));
                } else {
                    tracing::warn!("{}: path not configured (required for production)", field);
                }
                continue;
            }

            // Check file extension
            if let Some(ext) = expected_ext {
                if !path.ends_with(ext) {
                    warnings.push(format!("{}: expected {} extension, got '{}'", field, ext, path));
                }
            }

            // Check path exists
            let path_obj = Path::new(path);
            if !path_obj.exists() {
                if self.environment.is_strict() {
                    errors.push(format!("{}: model file not found: {}", field, path));
                } else {
                    tracing::warn!("Model not found: {} = {}", field, path);
                }
            } else if !path_obj.is_file() {
                errors.push(format!("{}: path exists but is not a file: {}", field, path));
            }
        }

        // Validate optional models (warnings only)
        for (field, path, expected_ext) in optional_models {
            if path.is_empty() {
                continue;
            }

            if let Some(ext) = expected_ext {
                if !path.ends_with(ext) {
                    warnings.push(format!("{}: expected {} extension, got '{}'", field, ext, path));
                }
            }

            let path_obj = Path::new(path);
            if !path_obj.exists() {
                tracing::warn!("Optional model not found: {} = {}", field, path);
            } else if !path_obj.is_file() {
                warnings.push(format!("{}: path exists but is not a file: {}", field, path));
            }
        }

        // Report warnings
        if !warnings.is_empty() {
            tracing::warn!("Model path warnings:\n  - {}", warnings.join("\n  - "));
        }

        // In production/staging, errors are fatal
        if !errors.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "models".to_string(),
                message: format!("Model validation failed:\n  - {}", errors.join("\n  - ")),
            });
        }

        Ok(())
    }
}


/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP server host
    #[serde(default = "default_host")]
    pub host: String,

    /// HTTP server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// WebSocket path
    #[serde(default = "default_ws_path")]
    pub ws_path: String,

    /// Maximum concurrent connections
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Enable CORS
    #[serde(default = "default_true")]
    pub cors_enabled: bool,

    /// CORS allowed origins
    #[serde(default)]
    pub cors_origins: Vec<String>,

    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// P1 FIX: Authentication configuration
    #[serde(default)]
    pub auth: AuthConfig,
}

/// P1 FIX: Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable authentication (set to false for development)
    #[serde(default)]
    pub enabled: bool,

    /// API key for simple authentication (should be set via VOICE_AGENT__SERVER__AUTH__API_KEY env var)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Paths that bypass authentication (e.g., health checks)
    #[serde(default = "default_public_paths")]
    pub public_paths: Vec<String>,
}

fn default_public_paths() -> Vec<String> {
    vec![
        "/health".to_string(),
        "/ready".to_string(),
        "/metrics".to_string(),
    ]
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for development
            api_key: None,
            public_paths: default_public_paths(),
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum messages per second per connection
    #[serde(default = "default_messages_per_second")]
    pub messages_per_second: u32,

    /// Maximum audio bytes per second per connection
    #[serde(default = "default_audio_bytes_per_second")]
    pub audio_bytes_per_second: u32,

    /// Burst allowance (multiple of rate limit)
    #[serde(default = "default_burst_multiplier")]
    pub burst_multiplier: f32,
}

fn default_messages_per_second() -> u32 {
    100 // 100 messages/sec should be plenty for voice
}

fn default_audio_bytes_per_second() -> u32 {
    64000 // 16kHz * 2 bytes * 2 (some headroom) = 64KB/s
}

fn default_burst_multiplier() -> f32 {
    2.0
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            messages_per_second: default_messages_per_second(),
            audio_bytes_per_second: default_audio_bytes_per_second(),
            burst_multiplier: default_burst_multiplier(),
        }
    }
}

/// P5 FIX: RAG configuration for retrieval and reranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Enable RAG retrieval
    #[serde(default = "default_true")]
    pub enabled: bool,

    // Retriever settings
    /// Top-K results from dense (embedding) search
    #[serde(default = "default_dense_top_k")]
    pub dense_top_k: usize,

    /// Top-K results from sparse (BM25/keyword) search
    #[serde(default = "default_sparse_top_k")]
    pub sparse_top_k: usize,

    /// Final top-K results after fusion
    #[serde(default = "default_final_top_k")]
    pub final_top_k: usize,

    /// Weight for dense vs sparse (0.0 = all sparse, 1.0 = all dense)
    #[serde(default = "default_dense_weight")]
    pub dense_weight: f32,

    /// Reciprocal Rank Fusion parameter (higher = more weight to top results)
    #[serde(default = "default_rrf_k")]
    pub rrf_k: f32,

    /// Minimum score threshold (filter out low-scoring results)
    #[serde(default = "default_min_score")]
    pub min_score: f32,

    // Reranker settings
    /// Enable cascaded reranking
    #[serde(default = "default_true")]
    pub reranking_enabled: bool,

    /// Pre-filter threshold (keyword overlap score to pass to full model)
    #[serde(default = "default_prefilter_threshold")]
    pub prefilter_threshold: f32,

    /// Max documents to run through full reranking model
    #[serde(default = "default_max_full_model_docs")]
    pub max_full_model_docs: usize,

    /// Early termination confidence threshold
    #[serde(default = "default_early_termination_threshold")]
    pub early_termination_threshold: f32,

    /// Minimum high-confidence results before early termination
    #[serde(default = "default_early_termination_min_results")]
    pub early_termination_min_results: usize,

    // Prefetch settings
    /// Confidence threshold for VAD-triggered prefetch
    #[serde(default = "default_prefetch_confidence")]
    pub prefetch_confidence_threshold: f32,

    /// Top-K results for prefetch (smaller for speed)
    #[serde(default = "default_prefetch_top_k")]
    pub prefetch_top_k: usize,
}

// RAG default value functions
fn default_dense_top_k() -> usize { 20 }
fn default_sparse_top_k() -> usize { 20 }
fn default_final_top_k() -> usize { 5 }
fn default_dense_weight() -> f32 { 0.7 }  // P5: Increased for semantic queries
fn default_rrf_k() -> f32 { 60.0 }
fn default_min_score() -> f32 { 0.4 }  // P5: Increased for domain-specific queries
fn default_prefilter_threshold() -> f32 { 0.15 }  // P5: Tuned for gold loan domain
fn default_max_full_model_docs() -> usize { 10 }
fn default_early_termination_threshold() -> f32 { 0.92 }  // P5: Slightly lower for faster exits
fn default_early_termination_min_results() -> usize { 3 }
fn default_prefetch_confidence() -> f32 { 0.6 }  // P5: Lower for more aggressive prefetch
fn default_prefetch_top_k() -> usize { 3 }

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dense_top_k: default_dense_top_k(),
            sparse_top_k: default_sparse_top_k(),
            final_top_k: default_final_top_k(),
            dense_weight: default_dense_weight(),
            rrf_k: default_rrf_k(),
            min_score: default_min_score(),
            reranking_enabled: true,
            prefilter_threshold: default_prefilter_threshold(),
            max_full_model_docs: default_max_full_model_docs(),
            early_termination_threshold: default_early_termination_threshold(),
            early_termination_min_results: default_early_termination_min_results(),
            prefetch_confidence_threshold: default_prefetch_confidence(),
            prefetch_top_k: default_prefetch_top_k(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_ws_path() -> String {
    "/ws/conversation".to_string()
}
fn default_max_connections() -> usize {
    1000
}
fn default_timeout() -> u64 {
    30
}
fn default_true() -> bool {
    true
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            ws_path: default_ws_path(),
            max_connections: default_max_connections(),
            timeout_seconds: default_timeout(),
            cors_enabled: default_true(),
            // SECURITY: Empty by default - must be explicitly configured for production
            // Use ["http://localhost:3000"] for local dev, or specific domains for production
            cors_origins: Vec::new(),
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),  // P1 FIX: Auth config
        }
    }
}

/// Model file paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPaths {
    /// VAD model path
    #[serde(default = "default_vad_path")]
    pub vad: String,

    /// Turn detection model path
    #[serde(default = "default_turn_detection_path")]
    pub turn_detection: String,

    /// Turn detection tokenizer path
    #[serde(default = "default_turn_tokenizer_path")]
    pub turn_detection_tokenizer: String,

    /// STT model path
    #[serde(default = "default_stt_path")]
    pub stt: String,

    /// STT tokens path
    #[serde(default = "default_stt_tokens_path")]
    pub stt_tokens: String,

    /// TTS model path
    #[serde(default = "default_tts_path")]
    pub tts: String,

    /// Cross-encoder model path
    #[serde(default = "default_reranker_path")]
    pub reranker: String,

    /// Embedding model path
    #[serde(default = "default_embeddings_path")]
    pub embeddings: String,
}

fn default_vad_path() -> String {
    "models/vad/silero_vad.onnx".to_string()
}
fn default_turn_detection_path() -> String {
    "models/turn_detection/smollm2-135m.onnx".to_string()
}
fn default_turn_tokenizer_path() -> String {
    "models/turn_detection/tokenizer.json".to_string()
}
fn default_stt_path() -> String {
    "models/stt/indicconformer.onnx".to_string()
}
fn default_stt_tokens_path() -> String {
    "models/stt/tokens.txt".to_string()
}
fn default_tts_path() -> String {
    "models/tts/indicf5.onnx".to_string()
}
fn default_reranker_path() -> String {
    "models/reranker/bge-reranker-v2-m3.onnx".to_string()
}
fn default_embeddings_path() -> String {
    "models/embeddings/e5-multilingual.onnx".to_string()
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self {
            vad: default_vad_path(),
            turn_detection: default_turn_detection_path(),
            turn_detection_tokenizer: default_turn_tokenizer_path(),
            stt: default_stt_path(),
            stt_tokens: default_stt_tokens_path(),
            tts: default_tts_path(),
            reranker: default_reranker_path(),
            embeddings: default_embeddings_path(),
        }
    }
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Enable JSON logging
    #[serde(default)]
    pub log_json: bool,

    /// Enable tracing
    #[serde(default = "default_true")]
    pub tracing_enabled: bool,

    /// OTLP endpoint for traces
    #[serde(default)]
    pub otlp_endpoint: Option<String>,

    /// Enable metrics
    #[serde(default = "default_true")]
    pub metrics_enabled: bool,

    /// Metrics port
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

fn default_log_level() -> String {
    "info".to_string()
}
fn default_metrics_port() -> u16 {
    9090
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            log_json: false,
            tracing_enabled: true,
            otlp_endpoint: None,
            metrics_enabled: true,
            metrics_port: default_metrics_port(),
        }
    }
}

/// Feature flags for experimentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable semantic turn detection
    #[serde(default = "default_true")]
    pub semantic_turn_detection: bool,

    /// Enable speculative LLM execution
    #[serde(default = "default_true")]
    pub speculative_llm: bool,

    /// Enable early-exit cross-encoder
    #[serde(default = "default_true")]
    pub early_exit_reranker: bool,

    /// Enable RAG prefetch on partial transcript
    #[serde(default = "default_true")]
    pub rag_prefetch: bool,

    /// Enable word-level TTS
    #[serde(default = "default_true")]
    pub word_level_tts: bool,

    /// Enable barge-in handling
    #[serde(default = "default_true")]
    pub barge_in_enabled: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            semantic_turn_detection: true,
            speculative_llm: true,
            early_exit_reranker: true,
            rag_prefetch: true,
            word_level_tts: true,
            barge_in_enabled: true,
        }
    }
}

/// Load settings from files and environment
///
/// Priority (highest to lowest):
/// 1. Environment variables (VOICE_AGENT_ prefix)
/// 2. config/{env}.yaml (if env specified)
/// 3. config/default.yaml
pub fn load_settings(env: Option<&str>) -> Result<Settings, ConfigError> {
    let mut builder = Config::builder();

    // Load default config
    builder = builder.add_source(
        File::with_name("config/default")
            .required(false)
    );

    // Load environment-specific config
    if let Some(env_name) = env {
        builder = builder.add_source(
            File::with_name(&format!("config/{}", env_name))
                .required(false)
        );
    }

    // Load from environment variables
    builder = builder.add_source(
        Environment::with_prefix("VOICE_AGENT")
            .separator("__")
            .try_parsing(true)
    );

    let config = builder.build()?;
    let settings: Settings = config.try_deserialize()?;

    // Validate
    settings.validate()?;

    Ok(settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.server.port, 8080);
        assert!(settings.features.semantic_turn_detection);
    }

    #[test]
    fn test_settings_validation() {
        let mut settings = Settings::default();
        settings.pipeline.latency_budget_ms = 100; // Too low
        assert!(settings.validate().is_err());

        settings.pipeline.latency_budget_ms = 500;
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_rag_validation_dense_weight() {
        let mut settings = Settings::default();

        // Valid weight
        settings.rag.dense_weight = 0.5;
        assert!(settings.validate_rag().is_ok());

        // Invalid weight (too high)
        settings.rag.dense_weight = 1.5;
        assert!(settings.validate_rag().is_err());

        // Invalid weight (negative)
        settings.rag.dense_weight = -0.1;
        assert!(settings.validate_rag().is_err());
    }

    #[test]
    fn test_rag_validation_thresholds() {
        let mut settings = Settings::default();

        // Invalid min_score
        settings.rag.min_score = 1.5;
        assert!(settings.validate_rag().is_err());
        settings.rag.min_score = 0.4;

        // Invalid early_termination_threshold
        settings.rag.early_termination_threshold = -0.1;
        assert!(settings.validate_rag().is_err());
        settings.rag.early_termination_threshold = 0.92;

        // Invalid rrf_k (not positive)
        settings.rag.rrf_k = 0.0;
        assert!(settings.validate_rag().is_err());
        settings.rag.rrf_k = -1.0;
        assert!(settings.validate_rag().is_err());
    }

    #[test]
    fn test_rag_validation_top_k() {
        let mut settings = Settings::default();

        // final_top_k cannot be 0
        settings.rag.final_top_k = 0;
        assert!(settings.validate_rag().is_err());

        // early_termination_min_results cannot exceed max_full_model_docs
        settings.rag.final_top_k = 5;
        settings.rag.early_termination_min_results = 20;
        settings.rag.max_full_model_docs = 10;
        assert!(settings.validate_rag().is_err());
    }

    #[test]
    fn test_server_validation() {
        let mut settings = Settings::default();

        // Port cannot be 0
        settings.server.port = 0;
        assert!(settings.validate_server().is_err());
        settings.server.port = 8080;

        // max_connections cannot be 0
        settings.server.max_connections = 0;
        assert!(settings.validate_server().is_err());
        settings.server.max_connections = 1000;

        // timeout cannot be 0
        settings.server.timeout_seconds = 0;
        assert!(settings.validate_server().is_err());
        settings.server.timeout_seconds = 30;

        assert!(settings.validate_server().is_ok());
    }

    #[test]
    fn test_rate_limit_validation() {
        let mut settings = Settings::default();
        settings.server.rate_limit.enabled = true;

        // messages_per_second cannot be 0 when enabled
        settings.server.rate_limit.messages_per_second = 0;
        assert!(settings.validate_server().is_err());
        settings.server.rate_limit.messages_per_second = 100;

        // burst_multiplier must be >= 1.0
        settings.server.rate_limit.burst_multiplier = 0.5;
        assert!(settings.validate_server().is_err());
        settings.server.rate_limit.burst_multiplier = 2.0;

        assert!(settings.validate_server().is_ok());
    }

    #[test]
    fn test_production_auth_validation() {
        let mut settings = Settings::default();
        settings.environment = RuntimeEnvironment::Production;
        settings.server.auth.enabled = true;
        settings.server.auth.api_key = None;

        // Production with auth enabled requires API key
        assert!(settings.validate_server().is_err());

        settings.server.auth.api_key = Some("secret-key".to_string());
        assert!(settings.validate_server().is_ok());
    }

    #[test]
    fn test_pipeline_latency_bounds() {
        let mut settings = Settings::default();

        // Too low
        settings.pipeline.latency_budget_ms = 100;
        assert!(settings.validate_pipeline().is_err());

        // Too high
        settings.pipeline.latency_budget_ms = 15000;
        assert!(settings.validate_pipeline().is_err());

        // Valid range
        settings.pipeline.latency_budget_ms = 500;
        assert!(settings.validate_pipeline().is_ok());
    }
}
