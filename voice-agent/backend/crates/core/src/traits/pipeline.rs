//! Pipeline processing traits

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use crate::{Result, AudioFrame, Language};
use crate::transcript::TranscriptResult;

/// Frame types that flow through the pipeline
///
/// Note: Some variants contain non-serializable types (AudioFrame).
/// Use the serializable variants or convert before serializing.
#[derive(Debug, Clone)]
pub enum Frame {
    /// Audio input from microphone/WebRTC
    AudioInput(AudioFrame),

    /// Partial transcript (still being processed)
    TranscriptPartial {
        text: String,
        confidence: f32,
        language: Option<Language>,
    },

    /// Final transcript
    TranscriptFinal(TranscriptResult),

    /// LLM response chunk (streaming)
    LLMChunk {
        text: String,
        is_final: bool,
    },

    /// Complete sentence ready for TTS
    Sentence {
        text: String,
        language: Language,
        index: usize,
    },

    /// Audio output for playback
    AudioOutput(AudioFrame),

    /// User interrupted (barge-in detected)
    BargeIn {
        /// Position in current audio where interruption was detected
        audio_position_ms: u64,
        /// The interruption transcript (if available)
        transcript: Option<String>,
    },

    /// Voice activity detected (speech started)
    VoiceStart,

    /// Voice activity ended (silence detected)
    VoiceEnd {
        /// Duration of speech in milliseconds
        duration_ms: u64,
    },

    /// End of stream marker
    EndOfStream,

    /// Error occurred in pipeline
    Error {
        stage: String,
        message: String,
        recoverable: bool,
    },

    /// Control frame for pipeline management
    Control(ControlFrame),

    /// RAG retrieval results
    RagResults {
        query: String,
        documents: Vec<crate::Document>,
    },

    /// Metrics/telemetry event
    Metrics(Arc<MetricsEvent>),
}

/// Metrics event for telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsEvent {
    /// Event name
    pub name: String,
    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,
    /// Event data
    pub data: HashMap<String, serde_json::Value>,
}

/// Control frames for pipeline management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFrame {
    /// Flush all pending data
    Flush,
    /// Reset pipeline state
    Reset,
    /// Change configuration
    Configure(HashMap<String, serde_json::Value>),
    /// Request metrics
    GetMetrics,
}

impl Frame {
    /// Check if this is an end-of-stream frame
    pub fn is_end_of_stream(&self) -> bool {
        matches!(self, Frame::EndOfStream)
    }

    /// Check if this is an error frame
    pub fn is_error(&self) -> bool {
        matches!(self, Frame::Error { .. })
    }

    /// Check if this is a control frame
    pub fn is_control(&self) -> bool {
        matches!(self, Frame::Control(_))
    }

    /// Get the stage name for this frame type
    pub fn stage_name(&self) -> &'static str {
        match self {
            Frame::AudioInput(_) => "audio_input",
            Frame::TranscriptPartial { .. } => "transcript_partial",
            Frame::TranscriptFinal(_) => "transcript_final",
            Frame::LLMChunk { .. } => "llm_chunk",
            Frame::Sentence { .. } => "sentence",
            Frame::AudioOutput(_) => "audio_output",
            Frame::BargeIn { .. } => "barge_in",
            Frame::VoiceStart => "voice_start",
            Frame::VoiceEnd { .. } => "voice_end",
            Frame::EndOfStream => "end_of_stream",
            Frame::Error { .. } => "error",
            Frame::Control(_) => "control",
            Frame::RagResults { .. } => "rag_results",
            Frame::Metrics(_) => "metrics",
        }
    }
}

/// Context passed to frame processors
#[derive(Debug, Clone, Default)]
pub struct ProcessorContext {
    /// Session ID
    pub session_id: String,
    /// Current conversation turn number
    pub turn_number: usize,
    /// Detected language
    pub language: Option<Language>,
    /// Custom metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Processor-specific state
    state: HashMap<String, serde_json::Value>,
}

impl ProcessorContext {
    /// Create a new context for a session
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            ..Default::default()
        }
    }

    /// Set the language
    pub fn with_language(mut self, language: Language) -> Self {
        self.language = Some(language);
        self
    }

    /// Increment turn number
    pub fn next_turn(&mut self) {
        self.turn_number += 1;
    }

    /// Get state value
    pub fn get_state<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.state.get(key).and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Set state value
    pub fn set_state<T: Serialize>(&mut self, key: impl Into<String>, value: T) {
        if let Ok(v) = serde_json::to_value(value) {
            self.state.insert(key.into(), v);
        }
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Frame processor for pipeline stages
///
/// Each processor receives frames, processes them, and emits output frames.
/// Processors run in separate tokio tasks, connected by channels.
///
/// # Example Implementation
///
/// ```ignore
/// struct SentenceDetector {
///     buffer: String,
/// }
///
/// #[async_trait]
/// impl FrameProcessor for SentenceDetector {
///     async fn process(&self, frame: Frame, ctx: &mut ProcessorContext) -> Result<Vec<Frame>> {
///         match frame {
///             Frame::LLMChunk { text, is_final } => {
///                 self.buffer.push_str(&text);
///                 if is_final || self.buffer.ends_with('.') {
///                     let sentence = std::mem::take(&mut self.buffer);
///                     return Ok(vec![Frame::Sentence { text: sentence, ... }]);
///                 }
///                 Ok(vec![])
///             }
///             _ => Ok(vec![frame]) // Pass through
///         }
///     }
///
///     fn name(&self) -> &'static str {
///         "sentence_detector"
///     }
/// }
/// ```
#[async_trait]
pub trait FrameProcessor: Send + Sync + 'static {
    /// Process a frame and emit zero or more output frames
    ///
    /// # Arguments
    /// * `frame` - Input frame to process
    /// * `context` - Mutable context for storing state
    ///
    /// # Returns
    /// Vector of output frames (may be empty, one, or multiple)
    async fn process(
        &self,
        frame: Frame,
        context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>>;

    /// Get processor name for tracing
    fn name(&self) -> &'static str;

    /// Get processor description
    fn description(&self) -> &str {
        ""
    }

    /// Called when pipeline starts
    async fn on_start(&self, _context: &mut ProcessorContext) -> Result<()> {
        Ok(())
    }

    /// Called when pipeline stops
    async fn on_stop(&self, _context: &mut ProcessorContext) -> Result<()> {
        Ok(())
    }

    /// Check if this processor can handle a frame type
    fn can_handle(&self, frame: &Frame) -> bool {
        let _ = frame;
        true // Default: handle all frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_stage_names() {
        assert_eq!(Frame::VoiceStart.stage_name(), "voice_start");
        assert_eq!(Frame::EndOfStream.stage_name(), "end_of_stream");
    }

    #[test]
    fn test_processor_context() {
        let mut ctx = ProcessorContext::new("session-123")
            .with_language(Language::Hindi);

        assert_eq!(ctx.session_id, "session-123");
        assert_eq!(ctx.language, Some(Language::Hindi));
        assert_eq!(ctx.turn_number, 0);

        ctx.next_turn();
        assert_eq!(ctx.turn_number, 1);

        ctx.set_state("buffer_size", 1024usize);
        assert_eq!(ctx.get_state::<usize>("buffer_size"), Some(1024));
    }

    #[test]
    fn test_frame_predicates() {
        assert!(Frame::EndOfStream.is_end_of_stream());
        assert!(Frame::Error { stage: "test".into(), message: "err".into(), recoverable: false }.is_error());
        assert!(Frame::Control(ControlFrame::Flush).is_control());
    }
}
