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

    /// 22K gold purity factor (916/1000)
    pub const PURITY_22K: f64 = 0.916;

    /// 18K gold purity factor
    pub const PURITY_18K: f64 = 0.750;
}

/// Service endpoints (defaults for local development)
pub mod endpoints {
    /// Ollama LLM endpoint
    pub const OLLAMA_DEFAULT: &str = "http://localhost:11434";

    /// Qdrant vector store endpoint
    pub const QDRANT_DEFAULT: &str = "http://localhost:6333";

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
pub mod rag {
    /// Weight for dense (semantic) search vs sparse (keyword) search
    /// Higher = more semantic, Lower = more keyword
    pub const DENSE_WEIGHT: f64 = 0.7;

    /// Minimum similarity score to include a result
    pub const MIN_SCORE: f64 = 0.4;

    /// Confidence threshold for prefetching additional results
    pub const PREFETCH_CONFIDENCE_THRESHOLD: f64 = 0.6;

    /// Default number of results to retrieve
    pub const DEFAULT_TOP_K: usize = 5;

    /// Maximum context tokens for RAG
    pub const MAX_CONTEXT_TOKENS: usize = 2048;
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
