//! Speech processing traits

use async_trait::async_trait;
use std::pin::Pin;
use futures::Stream;
use crate::{AudioFrame, Language, VoiceConfig, VoiceInfo, Result};
use crate::transcript::TranscriptResult;

/// Alias for transcript frame (matches architecture doc naming)
pub type TranscriptFrame = TranscriptResult;

/// Speech-to-Text interface
///
/// Implementations:
/// - `IndicConformerStt` - AI4Bharat's multilingual STT (22 languages)
/// - `WhisperStt` - OpenAI Whisper (fallback)
///
/// # Example
///
/// ```ignore
/// let stt: Box<dyn SpeechToText> = Box::new(IndicConformerStt::new(config));
/// let transcript = stt.transcribe(&audio_frame).await?;
/// println!("Transcribed: {}", transcript.text);
/// ```
#[async_trait]
pub trait SpeechToText: Send + Sync + 'static {
    /// Transcribe a single audio frame
    ///
    /// # Arguments
    /// * `audio` - Audio frame to transcribe
    ///
    /// # Returns
    /// Transcript with text, confidence, and optional word timestamps
    async fn transcribe(&self, audio: &AudioFrame) -> Result<TranscriptFrame>;

    /// Stream transcription as audio arrives
    ///
    /// Returns partial transcripts followed by final transcript.
    /// Partial transcripts have `is_final = false`.
    ///
    /// # Arguments
    /// * `audio_stream` - Stream of audio frames
    ///
    /// # Returns
    /// Stream of transcript results (partial and final)
    fn transcribe_stream<'a>(
        &'a self,
        audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
    ) -> Pin<Box<dyn Stream<Item = Result<TranscriptFrame>> + Send + 'a>>;

    /// Get supported languages
    fn supported_languages(&self) -> &[Language];

    /// Get model name for logging
    fn model_name(&self) -> &str;

    /// Check if a specific language is supported
    fn supports_language(&self, lang: Language) -> bool {
        self.supported_languages().contains(&lang)
    }
}

/// Text-to-Speech interface
///
/// Implementations:
/// - `IndicF5Tts` - AI4Bharat's multilingual TTS (11 languages)
/// - `PiperTts` - Fast fallback TTS
///
/// # Example
///
/// ```ignore
/// let tts: Box<dyn TextToSpeech> = Box::new(IndicF5Tts::new(config));
/// let config = VoiceConfig::new(Language::Hindi);
/// let audio = tts.synthesize("नमस्ते", &config).await?;
/// ```
#[async_trait]
pub trait TextToSpeech: Send + Sync + 'static {
    /// Synthesize text to audio
    ///
    /// # Arguments
    /// * `text` - Text to synthesize
    /// * `config` - Voice configuration (language, speed, pitch)
    ///
    /// # Returns
    /// Audio frame with synthesized speech
    async fn synthesize(&self, text: &str, config: &VoiceConfig) -> Result<AudioFrame>;

    /// Stream synthesis sentence-by-sentence
    ///
    /// Enables low-latency response by starting audio before text is complete.
    /// Each yielded audio frame corresponds to one sentence.
    ///
    /// # Arguments
    /// * `text_stream` - Stream of text chunks (sentences)
    /// * `config` - Voice configuration
    ///
    /// # Returns
    /// Stream of audio frames
    fn synthesize_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        config: &'a VoiceConfig,
    ) -> Pin<Box<dyn Stream<Item = Result<AudioFrame>> + Send + 'a>>;

    /// Get available voices
    fn available_voices(&self) -> &[VoiceInfo];

    /// Get model name for logging
    fn model_name(&self) -> &str;

    /// Get default voice for a language
    fn default_voice(&self, lang: Language) -> Option<&VoiceInfo> {
        self.available_voices()
            .iter()
            .find(|v| v.language == lang)
    }
}

// =============================================================================
// P1 FIX: VoiceActivityDetector Trait
// =============================================================================

/// Configuration for Voice Activity Detection
///
/// Controls sensitivity and timing thresholds for speech detection.
#[derive(Debug, Clone)]
pub struct VADConfig {
    /// Speech probability threshold (0.0-1.0)
    /// Frames with probability >= threshold are considered speech
    pub threshold: f32,
    /// Minimum consecutive speech frames to confirm speech start (in ms)
    pub min_speech_duration_ms: u32,
    /// Minimum consecutive silence frames to confirm speech end (in ms)
    pub min_silence_duration_ms: u32,
    /// Energy floor in dB - frames below this are quick-rejected as silence
    pub energy_floor_db: f32,
    /// Pre-speech padding in ms - how much audio before speech start to include
    pub pre_speech_padding_ms: u32,
    /// Post-speech padding in ms - how much audio after speech end to include
    pub post_speech_padding_ms: u32,
}

impl Default for VADConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 256,
            min_silence_duration_ms: 320,
            energy_floor_db: -50.0,
            pre_speech_padding_ms: 100,
            post_speech_padding_ms: 100,
        }
    }
}

impl VADConfig {
    /// Create a sensitive config (lower thresholds, catches more speech)
    pub fn sensitive() -> Self {
        Self {
            threshold: 0.3,
            min_speech_duration_ms: 128,
            min_silence_duration_ms: 400,
            energy_floor_db: -55.0,
            pre_speech_padding_ms: 150,
            post_speech_padding_ms: 150,
        }
    }

    /// Create a strict config (higher thresholds, fewer false positives)
    pub fn strict() -> Self {
        Self {
            threshold: 0.7,
            min_speech_duration_ms: 384,
            min_silence_duration_ms: 256,
            energy_floor_db: -45.0,
            pre_speech_padding_ms: 50,
            post_speech_padding_ms: 50,
        }
    }
}

/// Voice Activity Detection events
///
/// Represents the different states/events that can occur during VAD processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VADEvent {
    /// Speech has started (after min_speech_duration_ms of speech)
    SpeechStart,
    /// Speech is continuing (with current probability)
    SpeechContinue {
        /// Speech probability for the current frame
        probability: f32,
    },
    /// Speech has ended (after min_silence_duration_ms of silence)
    SpeechEnd,
    /// Silence detected
    Silence,
}

impl VADEvent {
    /// Check if this event indicates active speech
    pub fn is_speech(&self) -> bool {
        matches!(self, Self::SpeechStart | Self::SpeechContinue { .. })
    }

    /// Get speech probability if available
    pub fn probability(&self) -> Option<f32> {
        match self {
            Self::SpeechContinue { probability } => Some(*probability),
            _ => None,
        }
    }
}

/// VAD state for tracking speech boundaries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VADState {
    /// Waiting for speech
    Idle,
    /// Potential speech detected, waiting for confirmation
    PendingSpeech,
    /// In confirmed speech segment
    InSpeech,
    /// Potential end of speech, waiting for confirmation
    PendingSilence,
}

impl Default for VADState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Voice Activity Detector interface
///
/// Implementations:
/// - `SileroVAD` - High accuracy neural VAD
/// - `WebRTCVAD` - Fast, lightweight VAD
/// - `MagicNetVAD` - Kotak's custom trained VAD
///
/// # Example
///
/// ```ignore
/// let vad: Box<dyn VoiceActivityDetector> = Box::new(SileroVAD::new());
/// let config = VADConfig::default();
///
/// // Process single frame
/// if vad.detect(&audio_frame, 0.5).await {
///     println!("Speech detected!");
/// }
///
/// // Or stream processing
/// let events = vad.process_stream(audio_stream, &config);
/// while let Some(event) = events.next().await {
///     match event {
///         VADEvent::SpeechStart => println!("Started speaking"),
///         VADEvent::SpeechEnd => println!("Stopped speaking"),
///         _ => {}
///     }
/// }
/// ```
#[async_trait]
pub trait VoiceActivityDetector: Send + Sync + 'static {
    /// Detect if a single frame contains speech
    ///
    /// # Arguments
    /// * `audio` - Audio frame to analyze
    /// * `sensitivity` - Detection sensitivity (0.0 = strict, 1.0 = sensitive)
    ///
    /// # Returns
    /// `true` if frame likely contains speech
    async fn detect(&self, audio: &AudioFrame, sensitivity: f32) -> bool;

    /// Get speech probability for a frame
    ///
    /// # Arguments
    /// * `audio` - Audio frame to analyze
    ///
    /// # Returns
    /// Probability of speech (0.0 to 1.0)
    async fn speech_probability(&self, audio: &AudioFrame) -> f32;

    /// Process an audio stream and emit VAD events
    ///
    /// This is the primary streaming interface. It maintains internal state
    /// to track speech boundaries and emits events when:
    /// - Speech starts (after min_speech_duration_ms)
    /// - Speech continues (with probability)
    /// - Speech ends (after min_silence_duration_ms)
    /// - Silence is detected
    ///
    /// # Arguments
    /// * `audio_stream` - Stream of audio frames
    /// * `config` - VAD configuration
    ///
    /// # Returns
    /// Stream of VAD events
    fn process_stream<'a>(
        &'a self,
        audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
        config: &'a VADConfig,
    ) -> Pin<Box<dyn Stream<Item = VADEvent> + Send + 'a>>;

    /// Reset internal state
    ///
    /// Call this when starting a new conversation or after errors.
    fn reset(&self);

    /// Get current VAD state
    fn current_state(&self) -> VADState;

    /// Get model info for logging
    fn model_info(&self) -> &str;

    /// Check if VAD uses neural network (vs energy-based)
    fn is_neural(&self) -> bool {
        true // Most modern VADs are neural
    }

    /// Get recommended frame size in samples
    fn recommended_frame_size(&self) -> usize {
        480 // 30ms at 16kHz
    }
}

/// Audio Processor trait for pre-processing audio
///
/// Used for echo cancellation (AEC), noise suppression, etc.
#[async_trait]
pub trait AudioProcessor: Send + Sync + 'static {
    /// Process audio frame
    ///
    /// # Arguments
    /// * `input` - Input audio frame
    /// * `reference` - Optional reference signal (for AEC)
    ///
    /// # Returns
    /// Processed audio frame
    async fn process(
        &self,
        input: &AudioFrame,
        reference: Option<&AudioFrame>,
    ) -> Result<AudioFrame>;

    /// Get processor name for logging
    fn name(&self) -> &str;

    /// Reset internal state
    fn reset(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing
    struct MockStt {
        languages: Vec<Language>,
    }

    #[async_trait]
    impl SpeechToText for MockStt {
        async fn transcribe(&self, _audio: &AudioFrame) -> Result<TranscriptFrame> {
            Ok(TranscriptResult {
                text: "Test transcription".to_string(),
                confidence: 0.95,
                is_final: true,
                ..Default::default()
            })
        }

        fn transcribe_stream<'a>(
            &'a self,
            _audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
        ) -> Pin<Box<dyn Stream<Item = Result<TranscriptFrame>> + Send + 'a>> {
            Box::pin(futures::stream::empty())
        }

        fn supported_languages(&self) -> &[Language] {
            &self.languages
        }

        fn model_name(&self) -> &str {
            "mock-stt"
        }
    }

    #[test]
    fn test_supports_language() {
        let stt = MockStt {
            languages: vec![Language::Hindi, Language::English],
        };
        assert!(stt.supports_language(Language::Hindi));
        assert!(!stt.supports_language(Language::Tamil));
    }
}
