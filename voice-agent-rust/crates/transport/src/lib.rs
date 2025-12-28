//! Voice Agent Transport Layer
//!
//! P0 FIX: WebRTC transport for low-latency voice communication.
//!
//! Provides transport abstractions for:
//! - WebRTC (primary, for mobile apps)
//! - WebSocket (fallback, for web browsers)
//!
//! Target latency: <50ms one-way audio transport

pub mod webrtc;
pub mod websocket;
pub mod traits;
pub mod session;

pub use traits::{Transport, TransportEvent, AudioSink, AudioSource};
pub use session::{TransportSession, SessionConfig};
pub use webrtc::{WebRtcTransport, WebRtcConfig, IceServer};

use thiserror::Error;

/// Transport errors
#[derive(Error, Debug)]
pub enum TransportError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("ICE negotiation failed: {0}")]
    IceFailed(String),

    #[error("DTLS handshake failed: {0}")]
    DtlsFailed(String),

    #[error("Media error: {0}")]
    Media(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Session closed")]
    SessionClosed,

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Audio codec configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// Opus (recommended for voice)
    Opus,
    /// G.711 mu-law
    Pcmu,
    /// G.711 A-law
    Pcma,
}

impl Default for AudioCodec {
    fn default() -> Self {
        Self::Opus
    }
}

/// Audio format
#[derive(Debug, Clone)]
pub struct AudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Bits per sample (16 or 32)
    pub bits_per_sample: u8,
    /// Codec
    pub codec: AudioCodec,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            bits_per_sample: 16,
            codec: AudioCodec::Opus,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_default() {
        let format = AudioFormat::default();
        assert_eq!(format.sample_rate, 16000);
        assert_eq!(format.channels, 1);
    }
}
