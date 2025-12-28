//! Pipeline configuration

use serde::{Deserialize, Serialize};

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Target end-to-end latency budget in milliseconds
    #[serde(default = "default_latency_budget")]
    pub latency_budget_ms: u64,

    /// VAD configuration
    #[serde(default)]
    pub vad: VadConfig,

    /// Turn detection configuration
    #[serde(default)]
    pub turn_detection: TurnDetectionConfig,

    /// STT configuration
    #[serde(default)]
    pub stt: SttConfig,

    /// TTS configuration
    #[serde(default)]
    pub tts: TtsConfig,

    /// Barge-in configuration
    #[serde(default)]
    pub barge_in: BargeInConfig,

    /// Audio configuration
    #[serde(default)]
    pub audio: AudioConfig,
}

fn default_latency_budget() -> u64 {
    500
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            latency_budget_ms: default_latency_budget(),
            vad: VadConfig::default(),
            turn_detection: TurnDetectionConfig::default(),
            stt: SttConfig::default(),
            tts: TtsConfig::default(),
            barge_in: BargeInConfig::default(),
            audio: AudioConfig::default(),
        }
    }
}

/// Voice Activity Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    /// Speech probability threshold (0.0 - 1.0)
    #[serde(default = "default_vad_threshold")]
    pub threshold: f32,

    /// Frame size in milliseconds
    #[serde(default = "default_frame_ms")]
    pub frame_ms: u32,

    /// Minimum speech duration to confirm speech (ms)
    #[serde(default = "default_min_speech_ms")]
    pub min_speech_ms: u32,

    /// Minimum silence duration to end speech (ms)
    #[serde(default = "default_min_silence_ms")]
    pub min_silence_ms: u32,

    /// Energy floor in dB for quick silence detection
    #[serde(default = "default_energy_floor")]
    pub energy_floor_db: f32,

    /// Use MagicNet-style 10ms frames
    #[serde(default = "default_true")]
    pub use_10ms_frames: bool,
}

fn default_vad_threshold() -> f32 {
    0.5
}
fn default_frame_ms() -> u32 {
    10
}
fn default_min_speech_ms() -> u32 {
    250
}
fn default_min_silence_ms() -> u32 {
    300
}
fn default_energy_floor() -> f32 {
    -50.0
}
fn default_true() -> bool {
    true
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: default_vad_threshold(),
            frame_ms: default_frame_ms(),
            min_speech_ms: default_min_speech_ms(),
            min_silence_ms: default_min_silence_ms(),
            energy_floor_db: default_energy_floor(),
            use_10ms_frames: true,
        }
    }
}

/// Turn detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnDetectionConfig {
    /// Enable semantic turn detection
    #[serde(default = "default_true")]
    pub semantic_enabled: bool,

    /// Minimum silence before semantic check (ms)
    #[serde(default = "default_min_semantic_silence")]
    pub min_silence_for_semantic_ms: u64,

    /// Maximum silence before forced turn end (ms)
    #[serde(default = "default_max_silence")]
    pub max_silence_ms: u64,

    /// Probability threshold for <|im_end|> token
    #[serde(default = "default_end_token_threshold")]
    pub end_token_threshold: f32,

    /// Maximum sequence length for semantic model
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,

    /// History turns to include in context
    #[serde(default = "default_history_turns")]
    pub history_turns: usize,
}

fn default_min_semantic_silence() -> u64 {
    200
}
fn default_max_silence() -> u64 {
    1500
}
fn default_end_token_threshold() -> f32 {
    0.7
}
fn default_max_seq_len() -> usize {
    512
}
fn default_history_turns() -> usize {
    3
}

impl Default for TurnDetectionConfig {
    fn default() -> Self {
        Self {
            semantic_enabled: true,
            min_silence_for_semantic_ms: default_min_semantic_silence(),
            max_silence_ms: default_max_silence(),
            end_token_threshold: default_end_token_threshold(),
            max_seq_len: default_max_seq_len(),
            history_turns: default_history_turns(),
        }
    }
}

/// Speech-to-Text configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttConfig {
    /// Streaming chunk duration (ms)
    #[serde(default = "default_chunk_duration")]
    pub chunk_duration_ms: u32,

    /// Overlap between chunks (ms)
    #[serde(default = "default_overlap")]
    pub overlap_ms: u32,

    /// Emit partial results
    #[serde(default = "default_true")]
    pub emit_partials: bool,

    /// Minimum confidence for final results
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,

    /// Default language
    #[serde(default = "default_language")]
    pub default_language: String,

    /// Enable hallucination prevention
    #[serde(default = "default_true")]
    pub hallucination_prevention: bool,

    /// Beam size for decoding
    #[serde(default = "default_beam_size")]
    pub beam_size: usize,

    /// N-gram block size for repetition prevention
    #[serde(default = "default_ngram_block")]
    pub ngram_block_size: usize,
}

fn default_chunk_duration() -> u32 {
    500
}
fn default_overlap() -> u32 {
    100
}
fn default_min_confidence() -> f32 {
    0.7
}
fn default_language() -> String {
    "hi".to_string()
}
fn default_beam_size() -> usize {
    5
}
fn default_ngram_block() -> usize {
    3
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            chunk_duration_ms: default_chunk_duration(),
            overlap_ms: default_overlap(),
            emit_partials: true,
            min_confidence: default_min_confidence(),
            default_language: default_language(),
            hallucination_prevention: true,
            beam_size: default_beam_size(),
            ngram_block_size: default_ngram_block(),
        }
    }
}

/// Text-to-Speech configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    /// Default voice ID
    #[serde(default = "default_voice")]
    pub voice_id: String,

    /// Speaking rate multiplier (1.0 = normal)
    #[serde(default = "default_rate")]
    pub rate: f32,

    /// Pitch adjustment in semitones
    #[serde(default)]
    pub pitch: f32,

    /// Output sample rate
    #[serde(default = "default_tts_sample_rate")]
    pub sample_rate: u32,

    /// Chunk mode for streaming
    #[serde(default = "default_chunk_mode")]
    pub chunk_mode: TtsChunkMode,

    /// Crossfade duration between chunks (ms)
    #[serde(default = "default_crossfade")]
    pub crossfade_ms: u32,

    /// Maximum queue depth
    #[serde(default = "default_queue_depth")]
    pub max_queue_depth: usize,
}

fn default_voice() -> String {
    "hi-female-1".to_string()
}
fn default_rate() -> f32 {
    1.0
}
fn default_tts_sample_rate() -> u32 {
    22050
}
fn default_chunk_mode() -> TtsChunkMode {
    TtsChunkMode::WordLevel
}
fn default_crossfade() -> u32 {
    20
}
fn default_queue_depth() -> usize {
    5
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            voice_id: default_voice(),
            rate: default_rate(),
            pitch: 0.0,
            sample_rate: default_tts_sample_rate(),
            chunk_mode: default_chunk_mode(),
            crossfade_ms: default_crossfade(),
            max_queue_depth: default_queue_depth(),
        }
    }
}

/// TTS chunking mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsChunkMode {
    /// Fixed duration chunks
    FixedDuration,
    /// Word-level chunks (recommended)
    WordLevel,
    /// Clause-level chunks
    ClauseLevel,
    /// Sentence-level chunks
    SentenceLevel,
}

/// Barge-in configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BargeInConfig {
    /// Enable barge-in detection
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// VAD threshold for interrupt detection (higher than normal)
    #[serde(default = "default_barge_in_threshold")]
    pub vad_threshold: f32,

    /// Minimum speech duration to trigger interrupt (ms)
    #[serde(default = "default_barge_in_speech")]
    pub min_speech_ms: u32,

    /// Energy threshold for interrupt (dB)
    #[serde(default = "default_barge_in_energy")]
    pub energy_threshold_db: f32,

    /// Action on barge-in
    #[serde(default = "default_barge_in_action")]
    pub action: BargeInAction,

    /// Cooldown after barge-in (ms)
    #[serde(default = "default_cooldown")]
    pub cooldown_ms: u32,
}

fn default_barge_in_threshold() -> f32 {
    0.6
}
fn default_barge_in_speech() -> u32 {
    200
}
fn default_barge_in_energy() -> f32 {
    -35.0
}
fn default_barge_in_action() -> BargeInAction {
    BargeInAction::StopAndListen
}
fn default_cooldown() -> u32 {
    500
}

impl Default for BargeInConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            vad_threshold: default_barge_in_threshold(),
            min_speech_ms: default_barge_in_speech(),
            energy_threshold_db: default_barge_in_energy(),
            action: default_barge_in_action(),
            cooldown_ms: default_cooldown(),
        }
    }
}

/// Action on barge-in detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BargeInAction {
    /// Stop output and listen
    StopAndListen,
    /// Stop output and acknowledge
    StopAndAcknowledge,
    /// Reduce volume and continue
    DuckAndContinue,
    /// Ignore (disable barge-in)
    Ignore,
}

/// Audio format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    /// Input sample rate
    #[serde(default = "default_input_sample_rate")]
    pub input_sample_rate: u32,

    /// Output sample rate
    #[serde(default = "default_output_sample_rate")]
    pub output_sample_rate: u32,

    /// Jitter buffer size (ms)
    #[serde(default = "default_jitter_buffer")]
    pub jitter_buffer_ms: u32,

    /// Enable echo cancellation
    ///
    /// P2 FIX: Currently a placeholder flag. Actual AEC implementation requires
    /// a signal processing library (e.g., webrtc-audio-processing, speexdsp).
    /// When implemented, this will:
    /// - Cancel speaker output from microphone input
    /// - Reduce feedback during barge-in
    /// - Improve VAD accuracy during TTS playback
    #[serde(default = "default_true")]
    pub echo_cancellation: bool,

    /// Enable noise suppression
    ///
    /// P2 FIX: Currently a placeholder flag. Actual NS implementation requires
    /// a signal processing library (e.g., rnnoise, webrtc-audio-processing).
    /// When implemented, this will:
    /// - Reduce background noise before STT
    /// - Improve transcription accuracy in noisy environments
    /// - Work with energy-based VAD for better speech detection
    #[serde(default = "default_true")]
    pub noise_suppression: bool,
}

fn default_input_sample_rate() -> u32 {
    16000
}
fn default_output_sample_rate() -> u32 {
    22050
}
fn default_jitter_buffer() -> u32 {
    50
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_sample_rate: default_input_sample_rate(),
            output_sample_rate: default_output_sample_rate(),
            jitter_buffer_ms: default_jitter_buffer(),
            echo_cancellation: true,
            noise_suppression: true,
        }
    }
}
