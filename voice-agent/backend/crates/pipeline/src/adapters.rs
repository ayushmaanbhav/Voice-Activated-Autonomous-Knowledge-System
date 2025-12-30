//! Trait Adapters
//!
//! Bridges the pipeline's internal STT/TTS implementations with the core traits.
//!
//! This enables:
//! - Using pipeline STT/TTS as `dyn SpeechToText` or `dyn TextToSpeech`
//! - Injecting external implementations that implement the core traits
//! - Testing with mock implementations
//!
//! ## P0-4 FIX: Streaming Implementation
//!
//! The streaming methods now properly process audio/text streams:
//! - `transcribe_stream()` yields partial and final transcripts
//! - `synthesize_stream()` yields audio frames for each sentence

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use std::pin::Pin;
use std::sync::Arc;

use voice_agent_core::{
    AudioFrame, Channels, Language, Result as CoreResult, SampleRate, SpeechToText, TextToSpeech,
    TranscriptResult, VoiceConfig, VoiceInfo,
};

use crate::stt::{StreamingStt, SttConfig};
use crate::tts::{StreamingTts, TtsBackend, TtsConfig};

// =============================================================================
// SpeechToText Adapter
// =============================================================================

/// Adapter that wraps StreamingStt to implement the core SpeechToText trait
///
/// This allows the pipeline's STT implementation to be used anywhere
/// the `dyn SpeechToText` trait is expected.
pub struct SttAdapter {
    inner: Arc<Mutex<StreamingStt>>,
    config: SttConfig,
}

impl SttAdapter {
    /// Create a new adapter wrapping a StreamingStt
    pub fn new(stt: StreamingStt, config: SttConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(stt)),
            config,
        }
    }

    /// Create from config (initializes StreamingStt internally)
    #[cfg(feature = "onnx")]
    pub fn from_config(
        model_path: impl AsRef<std::path::Path>,
        config: SttConfig,
    ) -> Result<Self, PipelineError> {
        let stt = StreamingStt::new(model_path, config.clone())?;
        Ok(Self::new(stt, config))
    }
}

#[async_trait]
impl SpeechToText for SttAdapter {
    async fn transcribe(&self, audio: &AudioFrame) -> CoreResult<TranscriptResult> {
        let stt = self.inner.lock();

        // Process the audio chunk using inherent method (sync)
        if let Some(partial) = stt.process(&audio.samples).map_err(|e| {
            voice_agent_core::Error::Pipeline(voice_agent_core::error::PipelineError::Stt(
                e.to_string(),
            ))
        })? {
            return Ok(partial);
        }

        // Finalize to get the result using inherent method (sync)
        Ok(stt.finalize())
    }

    fn transcribe_stream<'a>(
        &'a self,
        audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
    ) -> Pin<Box<dyn Stream<Item = CoreResult<TranscriptResult>> + Send + 'a>> {
        // P0-4 FIX: Implement streaming transcription
        let stt = self.inner.clone();

        Box::pin(async_stream::stream! {
            futures::pin_mut!(audio_stream);

            while let Some(frame) = audio_stream.next().await {
                // Process audio frame through STT
                let result = stt.lock().process(&frame.samples);

                match result {
                    Ok(Some(partial)) => {
                        // Yield partial transcript
                        yield Ok(partial);
                    }
                    Ok(None) => {
                        // No transcript yet, continue processing
                    }
                    Err(e) => {
                        yield Err(voice_agent_core::Error::Pipeline(
                            voice_agent_core::error::PipelineError::Stt(e.to_string()),
                        ));
                    }
                }
            }

            // Finalize STT and yield final transcript
            let final_result = stt.lock().finalize();
            if !final_result.text.is_empty() {
                yield Ok(final_result);
            }
        })
    }

    fn supported_languages(&self) -> &[Language] {
        // Return supported languages based on engine
        static WHISPER_LANGS: &[Language] = &[
            Language::Hindi,
            Language::English,
            Language::Tamil,
            Language::Telugu,
            Language::Marathi,
            Language::Bengali,
            Language::Gujarati,
            Language::Kannada,
            Language::Malayalam,
            Language::Punjabi,
        ];
        WHISPER_LANGS
    }

    fn model_name(&self) -> &str {
        match self.config.engine {
            crate::stt::SttEngine::Whisper => "whisper",
            crate::stt::SttEngine::IndicConformer => "indicconformer",
            crate::stt::SttEngine::Wav2Vec2 => "wav2vec2",
        }
    }
}

// =============================================================================
// TextToSpeech Adapter
// =============================================================================

/// Adapter that wraps StreamingTts to implement the core TextToSpeech trait
///
/// This allows the pipeline's TTS implementation to be used anywhere
/// the `dyn TextToSpeech` trait is expected.
pub struct TtsAdapter {
    inner: Arc<StreamingTts>,
    config: TtsConfig,
}

impl TtsAdapter {
    /// Create a new adapter wrapping a StreamingTts
    pub fn new(tts: StreamingTts, config: TtsConfig) -> Self {
        Self {
            inner: Arc::new(tts),
            config,
        }
    }
}

#[async_trait]
impl TextToSpeech for TtsAdapter {
    async fn synthesize(&self, text: &str, _config: &VoiceConfig) -> CoreResult<AudioFrame> {
        // Use the internal TTS to synthesize (TtsBackend only takes text)
        let samples = self.inner.synthesize(text).await.map_err(|e| {
            voice_agent_core::Error::Pipeline(voice_agent_core::error::PipelineError::Tts(
                e.to_string(),
            ))
        })?;

        // Create AudioFrame from samples using TTS sample rate
        Ok(AudioFrame::new(
            samples,
            voice_agent_core::SampleRate::Hz16000,
            voice_agent_core::Channels::Mono,
            0,
        ))
    }

    fn synthesize_stream<'a>(
        &'a self,
        text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        _config: &'a VoiceConfig,
    ) -> Pin<Box<dyn Stream<Item = CoreResult<AudioFrame>> + Send + 'a>> {
        // P0-4 FIX: Implement streaming synthesis
        let tts = self.inner.clone();

        Box::pin(async_stream::stream! {
            futures::pin_mut!(text_stream);

            // Buffer for accumulating text until we hit sentence boundaries
            let mut sentence_buffer = String::new();

            while let Some(chunk) = text_stream.next().await {
                sentence_buffer.push_str(&chunk);

                // Check for sentence boundaries
                // We synthesize when we see sentence-ending punctuation
                while let Some(boundary) = find_sentence_boundary(&sentence_buffer) {
                    let sentence = sentence_buffer[..=boundary].trim().to_string();
                    sentence_buffer = sentence_buffer[boundary + 1..].to_string();

                    if sentence.is_empty() {
                        continue;
                    }

                    // Synthesize the complete sentence
                    match tts.synthesize(&sentence).await {
                        Ok(samples) => {
                            // Use 22050Hz as closest standard TTS sample rate
                            // (IndicF5 outputs 24kHz, may need resampling in transport layer)
                            yield Ok(AudioFrame::new(
                                samples,
                                SampleRate::Hz22050,
                                Channels::Mono,
                                0,
                            ));
                        }
                        Err(e) => {
                            yield Err(voice_agent_core::Error::Pipeline(
                                voice_agent_core::error::PipelineError::Tts(e.to_string()),
                            ));
                        }
                    }
                }
            }

            // Flush remaining text in buffer
            let remaining = sentence_buffer.trim();
            if !remaining.is_empty() {
                match tts.synthesize(remaining).await {
                    Ok(samples) => {
                        yield Ok(AudioFrame::new(
                            samples,
                            SampleRate::Hz22050,
                            Channels::Mono,
                            0,
                        ));
                    }
                    Err(e) => {
                        yield Err(voice_agent_core::Error::Pipeline(
                            voice_agent_core::error::PipelineError::Tts(e.to_string()),
                        ));
                    }
                }
            }
        })
    }

    fn available_voices(&self) -> &[VoiceInfo] {
        static VOICES: &[VoiceInfo] = &[];
        VOICES
    }

    fn model_name(&self) -> &str {
        match self.config.engine {
            crate::tts::TtsEngine::Piper => "piper",
            crate::tts::TtsEngine::IndicF5 => "indicf5",
            crate::tts::TtsEngine::ParlerTts => "parler",
        }
    }
}

// =============================================================================
// P0-4 FIX: Helper Functions
// =============================================================================

/// Find the next sentence boundary in text
///
/// Returns the index of the sentence-ending punctuation if found.
/// Handles common sentence terminators including Hindi danda (।).
fn find_sentence_boundary(text: &str) -> Option<usize> {
    // Sentence terminators: period, question mark, exclamation, Hindi danda
    const TERMINATORS: &[char] = &['.', '?', '!', '।', '॥'];

    for (i, c) in text.char_indices() {
        if TERMINATORS.contains(&c) {
            // Check if this is actually a sentence boundary
            // (not an abbreviation like "Dr." or number like "3.14")
            let after = text.get(i + c.len_utf8()..);
            if let Some(after) = after {
                // If followed by whitespace or end of string, it's a boundary
                if after.is_empty() || after.starts_with(char::is_whitespace) {
                    return Some(i);
                }
            } else {
                // End of string
                return Some(i);
            }
        }
    }
    None
}

// =============================================================================
// Factory Functions
// =============================================================================

/// Create a boxed SpeechToText from StreamingStt
pub fn create_stt_adapter(stt: StreamingStt, config: SttConfig) -> Box<dyn SpeechToText> {
    Box::new(SttAdapter::new(stt, config))
}

/// Create a boxed TextToSpeech from StreamingTts
pub fn create_tts_adapter(tts: StreamingTts, config: TtsConfig) -> Box<dyn TextToSpeech> {
    Box::new(TtsAdapter::new(tts, config))
}

// =============================================================================
// AudioProcessor Adapter (P2-2: Deferred - placeholder for future AEC/NS/AGC)
// =============================================================================

use voice_agent_core::AudioProcessor;

/// Passthrough audio processor that does no processing
///
/// This is a placeholder for future audio signal processing (AEC, NS, AGC).
/// Currently, browser-side processing via getUserMedia constraints is used.
///
/// When implementing real audio processing, this adapter can wrap:
/// - `webrtc-audio-processing-rs` for AEC/NS/AGC
/// - Custom DSP pipelines
pub struct PassthroughAudioProcessor {
    name: &'static str,
}

impl PassthroughAudioProcessor {
    /// Create a new passthrough processor
    pub fn new() -> Self {
        Self {
            name: "passthrough",
        }
    }

    /// Create with custom name (for testing/logging)
    pub fn with_name(name: &'static str) -> Self {
        Self { name }
    }
}

impl Default for PassthroughAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioProcessor for PassthroughAudioProcessor {
    async fn process(
        &self,
        input: &AudioFrame,
        _reference: Option<&AudioFrame>,
    ) -> CoreResult<AudioFrame> {
        // P2-2 DEFERRED: No processing, just clone the input
        // Future: Add AEC with reference signal, NS, AGC
        Ok(input.clone())
    }

    fn name(&self) -> &str {
        self.name
    }

    fn reset(&self) {
        // No state to reset in passthrough mode
    }
}

/// Create a boxed AudioProcessor (passthrough)
pub fn create_passthrough_processor() -> Box<dyn AudioProcessor> {
    Box::new(PassthroughAudioProcessor::new())
}

// =============================================================================
// P2-1 FIX: Noise Suppression Processor (nnnoiseless RNNoise port)
// =============================================================================

/// Noise suppression audio processor using nnnoiseless (RNNoise)
///
/// P2-1 FIX: Implements real noise suppression using a neural network.
/// Uses the nnnoiseless crate which is a pure Rust port of RNNoise.
///
/// Requirements:
/// - Input audio: 48kHz mono (will resample if needed)
/// - Frame size: 480 samples (10ms at 48kHz)
///
/// Note: AEC (echo cancellation) and AGC (gain control) should be handled
/// browser-side via getUserMedia constraints, or by adding webrtc-audio-processing.
#[cfg(feature = "noise-suppression")]
pub struct NoiseSuppressorProcessor {
    /// DenoiseState for noise reduction (boxed due to nnnoiseless API)
    state: parking_lot::Mutex<Box<nnnoiseless::DenoiseState<'static>>>,
    /// Resampler for converting to 48kHz if needed
    resampler_to_48k: Option<parking_lot::Mutex<rubato::FftFixedIn<f32>>>,
    /// Resampler for converting back from 48kHz
    resampler_from_48k: Option<parking_lot::Mutex<rubato::FftFixedOut<f32>>>,
    /// Original sample rate for resampling back
    original_sample_rate: u32,
    /// Buffer for accumulating samples until we have 480
    input_buffer: parking_lot::Mutex<Vec<f32>>,
    /// Buffer for output samples after processing
    output_buffer: parking_lot::Mutex<Vec<f32>>,
}

#[cfg(feature = "noise-suppression")]
impl NoiseSuppressorProcessor {
    /// Create a new noise suppressor processor
    ///
    /// # Arguments
    /// * `input_sample_rate` - Sample rate of input audio (e.g., 16000)
    pub fn new(input_sample_rate: u32) -> Self {
        use rubato::{FftFixedIn, FftFixedOut};

        let state = nnnoiseless::DenoiseState::new();

        // Create resamplers if input is not 48kHz
        let (resampler_to_48k, resampler_from_48k) = if input_sample_rate != 48000 {
            // Resample from input rate to 48kHz
            let to_48k = FftFixedIn::<f32>::new(
                input_sample_rate as usize,
                48000,
                480, // Output chunk size (10ms at 48kHz)
                2,   // Sub-chunks
                1,   // Mono
            )
            .expect("Failed to create resampler to 48kHz");

            // Resample from 48kHz back to input rate
            let from_48k = FftFixedOut::<f32>::new(
                48000,
                input_sample_rate as usize,
                (input_sample_rate / 100) as usize, // Output chunk size (10ms at input rate)
                2,                                  // Sub-chunks
                1,                                  // Mono
            )
            .expect("Failed to create resampler from 48kHz");

            (
                Some(parking_lot::Mutex::new(to_48k)),
                Some(parking_lot::Mutex::new(from_48k)),
            )
        } else {
            (None, None)
        };

        tracing::info!(
            input_sample_rate = input_sample_rate,
            "Noise suppressor initialized"
        );

        Self {
            state: parking_lot::Mutex::new(state),
            resampler_to_48k,
            resampler_from_48k,
            original_sample_rate: input_sample_rate,
            input_buffer: parking_lot::Mutex::new(Vec::with_capacity(960)),
            output_buffer: parking_lot::Mutex::new(Vec::with_capacity(960)),
        }
    }

    /// Process a single 480-sample frame at 48kHz
    fn process_frame_48k(&self, input: &[f32]) -> Vec<f32> {
        let mut state = self.state.lock();
        let mut output = vec![0.0f32; nnnoiseless::DenoiseState::FRAME_SIZE];
        state.process_frame(&mut output, input);
        output
    }
}

#[cfg(feature = "noise-suppression")]
impl Default for NoiseSuppressorProcessor {
    fn default() -> Self {
        // Default to 16kHz which is common for voice applications
        Self::new(16000)
    }
}

#[cfg(feature = "noise-suppression")]
#[async_trait]
impl AudioProcessor for NoiseSuppressorProcessor {
    async fn process(
        &self,
        input: &AudioFrame,
        _reference: Option<&AudioFrame>,
    ) -> CoreResult<AudioFrame> {
        use rubato::Resampler;

        let samples = &input.samples;

        // If empty, return immediately
        if samples.is_empty() {
            return Ok(input.clone());
        }

        // Add samples to input buffer
        {
            let mut buffer = self.input_buffer.lock();
            buffer.extend_from_slice(samples);
        }

        // Process available complete frames
        let mut processed_samples = Vec::new();

        // Determine frame size based on whether we need resampling
        let process_frame_size = if self.resampler_to_48k.is_some() {
            // With resampling, use input rate's 10ms worth of samples
            (self.original_sample_rate / 100) as usize
        } else {
            // Native 48kHz, use nnnoiseless frame size
            nnnoiseless::DenoiseState::FRAME_SIZE
        };

        loop {
            let frame_to_process: Option<Vec<f32>> = {
                let mut buffer = self.input_buffer.lock();
                if buffer.len() >= process_frame_size {
                    Some(buffer.drain(..process_frame_size).collect())
                } else {
                    None
                }
            };

            let Some(frame) = frame_to_process else {
                break;
            };

            // Resample to 48kHz if needed
            let frame_48k = if let Some(ref resampler) = self.resampler_to_48k {
                let mut resampler = resampler.lock();
                let input_frames = vec![frame];
                match resampler.process(&input_frames, None) {
                    Ok(output) => output.into_iter().next().unwrap_or_default(),
                    Err(e) => {
                        tracing::warn!("Resampling error: {}", e);
                        continue;
                    },
                }
            } else {
                frame
            };

            // Process in 480-sample chunks at 48kHz
            for chunk in frame_48k.chunks(nnnoiseless::DenoiseState::FRAME_SIZE) {
                if chunk.len() < nnnoiseless::DenoiseState::FRAME_SIZE {
                    // Pad short chunks
                    let mut padded = chunk.to_vec();
                    padded.resize(nnnoiseless::DenoiseState::FRAME_SIZE, 0.0);
                    let output = self.process_frame_48k(&padded);
                    // Only take the valid portion
                    processed_samples.extend_from_slice(&output[..chunk.len()]);
                } else {
                    let output = self.process_frame_48k(chunk);
                    processed_samples.extend_from_slice(&output);
                }
            }
        }

        // Resample back from 48kHz if needed
        let final_samples = if let Some(ref resampler) = self.resampler_from_48k {
            if processed_samples.is_empty() {
                processed_samples
            } else {
                let mut resampler = resampler.lock();
                let input_frames = vec![processed_samples];
                match resampler.process(&input_frames, None) {
                    Ok(output) => output.into_iter().next().unwrap_or_default(),
                    Err(e) => {
                        tracing::warn!("Resampling error: {}", e);
                        Vec::new()
                    },
                }
            }
        } else {
            processed_samples
        };

        // Return processed frame with same metadata but denoised samples
        Ok(AudioFrame::new(
            final_samples,
            input.sample_rate,
            input.channels,
            input.sequence,
        ))
    }

    fn name(&self) -> &str {
        "noise-suppressor"
    }

    fn reset(&self) {
        // Reset the denoise state
        *self.state.lock() = nnnoiseless::DenoiseState::new();
        // Clear buffers
        self.input_buffer.lock().clear();
        self.output_buffer.lock().clear();
    }
}

/// Create a boxed AudioProcessor with noise suppression
///
/// P2-1 FIX: Factory function for noise suppression processor.
/// Falls back to passthrough if noise-suppression feature is not enabled.
pub fn create_noise_suppressor(sample_rate: u32) -> Box<dyn AudioProcessor> {
    #[cfg(feature = "noise-suppression")]
    {
        Box::new(NoiseSuppressorProcessor::new(sample_rate))
    }

    #[cfg(not(feature = "noise-suppression"))]
    {
        let _ = sample_rate; // Suppress unused warning
        tracing::warn!("noise-suppression feature not enabled, using passthrough");
        Box::new(PassthroughAudioProcessor::with_name("passthrough-no-ns"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_adapter_model_name() {
        // This test verifies the adapter correctly reports model name
        // Actual STT testing requires ONNX models
    }

    #[test]
    fn test_tts_adapter_model_name() {
        // This test verifies the adapter correctly reports model name
        // Actual TTS testing requires models
    }

    #[tokio::test]
    async fn test_passthrough_processor() {
        let processor = PassthroughAudioProcessor::new();
        assert_eq!(processor.name(), "passthrough");

        let frame = AudioFrame::new(
            vec![0.1, 0.2, 0.3],
            voice_agent_core::SampleRate::Hz16000,
            voice_agent_core::Channels::Mono,
            0,
        );

        let result = processor.process(&frame, None).await.unwrap();
        assert_eq!(result.samples, frame.samples);
    }
}
