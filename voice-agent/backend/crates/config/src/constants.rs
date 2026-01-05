//! Centralized constants for the voice agent
//!
//! This module provides a single source of truth for all business constants
//! and default values used across the codebase. Instead of hardcoding values
//! in multiple files, use these constants to ensure consistency.
//!
//! # P1 FIX: Constants Centralization
//!
//! Previously, values like interest rates and endpoints were duplicated
//! across 6+ files, creating maintenance burden and inconsistency risk.

/// Gold loan interest rates (annual percentage)
///
/// These are Kotak's tiered interest rates based on loan amount.
/// Higher loan amounts get better (lower) rates.
pub mod interest_rates {
    /// Tier 1: Standard rate for small loans (< ₹1L)
    pub const TIER_1_STANDARD: f64 = 11.5;

    /// Tier 2: Headline rate for medium loans (₹1L - ₹5L)
    /// This is the advertised "starting from" rate
    pub const TIER_2_HEADLINE: f64 = 10.5;

    /// Tier 3: Premium rate for high-value loans (> ₹5L)
    pub const TIER_3_PREMIUM: f64 = 9.5;

    /// Default headline rate used in marketing/prompts
    pub const DEFAULT_HEADLINE: f64 = TIER_2_HEADLINE;

    /// Typical NBFC rate for comparison (Muthoot, Manappuram)
    pub const NBFC_TYPICAL_MIN: f64 = 18.0;
    pub const NBFC_TYPICAL_MAX: f64 = 24.0;
}

/// Loan amount tier boundaries (in INR)
pub mod loan_tiers {
    /// Tier 1 upper limit (inclusive)
    pub const TIER_1_MAX: f64 = 100_000.0; // ₹1 lakh

    /// Tier 2 upper limit (inclusive)
    pub const TIER_2_MAX: f64 = 500_000.0; // ₹5 lakh

    // Tier 3 is anything above TIER_2_MAX
}

/// Loan-to-Value ratios
pub mod ltv {
    /// Maximum LTV for gold loans (RBI regulated)
    pub const MAX_LTV_PERCENT: f64 = 75.0;

    /// Conservative LTV for risk calculations
    pub const CONSERVATIVE_LTV_PERCENT: f64 = 70.0;
}

/// Gold prices (default fallback values)
pub mod gold_prices {
    /// Default 24K gold price per gram (INR)
    /// Updated for 2024 prices - should be fetched from live API in production
    pub const DEFAULT_24K_PER_GRAM: f64 = 7500.0;

    /// 24K gold purity factor (pure gold)
    pub const PURITY_24K: f64 = 1.0;

    /// 22K gold purity factor (916/1000)
    pub const PURITY_22K: f64 = 0.916;

    /// 18K gold purity factor
    pub const PURITY_18K: f64 = 0.750;

    /// 14K gold purity factor (P1-5 FIX: was missing)
    pub const PURITY_14K: f64 = 0.585;
}

/// Service endpoints (defaults for local development)
pub mod endpoints {
    /// Ollama LLM endpoint
    pub const OLLAMA_DEFAULT: &str = "http://localhost:11434";

    /// Qdrant vector store endpoint (REST API port - gRPC has h2 issues)
    pub const QDRANT_DEFAULT: &str = "http://127.0.0.1:6333";

    /// OpenAI API endpoint
    pub const OPENAI_DEFAULT: &str = "https://api.openai.com/v1";

    /// Anthropic API endpoint
    pub const ANTHROPIC_DEFAULT: &str = "https://api.anthropic.com";
}

/// Timeouts (in milliseconds unless noted)
pub mod timeouts {
    /// Default tool execution timeout (ms)
    pub const TOOL_DEFAULT_MS: u64 = 30_000;

    /// LLM request timeout (ms)
    pub const LLM_REQUEST_MS: u64 = 60_000;

    /// STT processing timeout (ms)
    pub const STT_TIMEOUT_MS: u64 = 10_000;

    /// TTS synthesis timeout (ms)
    pub const TTS_TIMEOUT_MS: u64 = 15_000;

    /// WebRTC connection timeout (seconds)
    pub const WEBRTC_CONNECT_SECS: u64 = 30;
}

/// RAG (Retrieval-Augmented Generation) defaults
///
/// P6 FIX: Optimized thresholds for 97-document knowledge base
/// Tuned for small LLMs (qwen2.5:1.5b) with Hindi/English bilingual content
pub mod rag {
    /// Weight for dense (semantic) search vs sparse (keyword) search
    /// Higher = more semantic, Lower = more keyword
    /// P6 FIX: Reduced from 0.7 to 0.65 for better Hindi/Hinglish keyword matching
    pub const DENSE_WEIGHT: f64 = 0.65;

    /// Minimum similarity score to include a result
    /// P6 FIX: Reduced from 0.4 to 0.35 for larger corpus coverage
    /// With 97 docs, slightly lower threshold improves recall without hurting precision
    pub const MIN_SCORE: f64 = 0.35;

    /// Confidence threshold for prefetching additional results
    /// P6: Keep at 0.6 - good balance for speculative retrieval
    pub const PREFETCH_CONFIDENCE_THRESHOLD: f64 = 0.6;

    /// Default number of results to retrieve
    /// P6 FIX: Increased from 5 to 6 for better coverage with larger corpus
    pub const DEFAULT_TOP_K: usize = 6;

    /// Default context tokens for small models (4K context window)
    /// P0-2 FIX: Updated from 2048 to match context.rs default
    pub const DEFAULT_CONTEXT_TOKENS: usize = 4096;

    /// Maximum context tokens for large models (32K+ context window)
    pub const MAX_CONTEXT_TOKENS_LARGE: usize = 32768;

    // ==========================================================================
    // P6 FIX: Additional thresholds for confusion matrix optimization
    // ==========================================================================

    /// Minimum average score for sufficiency check (agentic RAG)
    /// Higher values reduce false positives but may require more iterations
    pub const SUFFICIENCY_MIN_AVG_SCORE: f64 = 0.35;

    /// Sufficiency threshold to stop query refinement
    pub const SUFFICIENCY_THRESHOLD: f64 = 0.7;

    /// Prefilter threshold for cascaded reranking
    /// Docs below this keyword overlap score are skipped
    pub const PREFILTER_THRESHOLD: f64 = 0.15;

    /// Early termination threshold for reranker
    /// P6: Lowered from 0.95 to 0.88 for faster exits with larger corpus
    pub const EARLY_TERMINATION_THRESHOLD: f64 = 0.88;

    /// Minimum high-confidence results before early termination
    pub const EARLY_TERMINATION_MIN_RESULTS: usize = 2;
}

/// Audio processing defaults
pub mod audio {
    /// Default sample rate (Hz)
    pub const SAMPLE_RATE: u32 = 16000;

    /// Default frame size (ms)
    pub const FRAME_MS: u32 = 10;

    /// Energy floor for VAD (dB)
    pub const VAD_ENERGY_FLOOR_DB: f32 = -50.0;

    /// Speech probability threshold for VAD
    pub const VAD_THRESHOLD: f32 = 0.5;

    // P1-2 FIX: PCM conversion constants (centralized)
    /// PCM16 normalization divisor (for converting PCM16 to f32)
    /// Use: sample as f32 / PCM16_NORMALIZE
    pub const PCM16_NORMALIZE: f32 = 32768.0;

    /// PCM16 scaling multiplier (for converting f32 to PCM16)
    /// Use: (sample * PCM16_SCALE) as i16
    pub const PCM16_SCALE: f32 = 32767.0;

    // P2-5 FIX: VAD frame count constants
    /// Minimum consecutive speech frames to confirm speech start (250ms at 10ms frames)
    pub const VAD_MIN_SPEECH_FRAMES: usize = 25;

    /// Minimum consecutive silence frames to confirm speech end (300ms at 10ms frames)
    pub const VAD_MIN_SILENCE_FRAMES: usize = 30;
}

/// P1-4 FIX: Turn detection timing constants
pub mod turn_detection {
    /// Base silence threshold before semantic adjustment (ms)
    pub const BASE_SILENCE_MS: u32 = 500;

    /// Minimum silence threshold (ms)
    pub const MIN_SILENCE_MS: u32 = 200;

    /// Maximum silence threshold (ms)
    pub const MAX_SILENCE_MS: u32 = 1000;

    /// Minimum speech duration to consider a valid utterance (ms)
    pub const MIN_SPEECH_MS: u32 = 200;

    /// Default semantic weight for hybrid turn detection
    pub const SEMANTIC_WEIGHT: f32 = 0.6;
}

/// P1-3 FIX: WebRTC configuration constants
pub mod webrtc {
    /// ICE disconnected timeout (seconds) - time before considering peer disconnected
    pub const ICE_DISCONNECTED_TIMEOUT_SECS: u64 = 5;

    /// ICE failed timeout (seconds) - time before declaring connection failed
    pub const ICE_FAILED_TIMEOUT_SECS: u64 = 25;

    /// ICE keep-alive interval (seconds)
    pub const ICE_KEEPALIVE_INTERVAL_SECS: u64 = 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interest_rates_ordering() {
        // Premium should be lowest, standard should be highest
        assert!(interest_rates::TIER_3_PREMIUM < interest_rates::TIER_2_HEADLINE);
        assert!(interest_rates::TIER_2_HEADLINE < interest_rates::TIER_1_STANDARD);
    }

    #[test]
    fn test_tier_boundaries() {
        assert!(loan_tiers::TIER_1_MAX < loan_tiers::TIER_2_MAX);
    }

    #[test]
    fn test_ltv_reasonable() {
        assert!(ltv::MAX_LTV_PERCENT <= 100.0);
        assert!(ltv::MAX_LTV_PERCENT > 0.0);
    }

    #[test]
    fn test_rag_weights_valid() {
        assert!(rag::DENSE_WEIGHT >= 0.0 && rag::DENSE_WEIGHT <= 1.0);
        assert!(rag::MIN_SCORE >= 0.0 && rag::MIN_SCORE <= 1.0);
    }
}
