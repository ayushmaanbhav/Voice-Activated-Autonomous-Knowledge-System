//! Voice Activity Detection
//!
//! Provides two VAD implementations:
//! - MagicNet: Custom VAD with 10ms frames and mel filterbank features
//! - Silero: Production-ready VAD using raw audio input (recommended)

mod magicnet;
mod silero;

pub use magicnet::{VoiceActivityDetector, VadConfig, VadState, VadResult};
pub use silero::{SileroVad, SileroConfig};

use crate::PipelineError;
use voice_agent_core::AudioFrame;

/// VAD engine trait for pluggable implementations
#[async_trait::async_trait]
pub trait VadEngine: Send + Sync {
    /// Process a single audio frame
    fn process_frame(&mut self, frame: &mut AudioFrame) -> Result<VadResult, PipelineError>;

    /// Reset VAD state
    fn reset(&mut self);

    /// Get current state
    fn state(&self) -> VadState;
}
