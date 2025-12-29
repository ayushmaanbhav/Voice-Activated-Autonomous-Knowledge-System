//! Transport Traits
//!
//! Abstract interfaces for transport implementations.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::{AudioFormat, TransportError};

/// Transport event
#[derive(Debug, Clone)]
pub enum TransportEvent {
    /// Connection established
    Connected {
        session_id: String,
        remote_addr: Option<String>,
    },
    /// Audio data received
    AudioReceived {
        /// Raw audio samples (PCM)
        samples: Vec<f32>,
        /// Timestamp in milliseconds
        timestamp_ms: u64,
    },
    /// Audio data ready to send (from pipeline)
    AudioToSend {
        samples: Vec<f32>,
        timestamp_ms: u64,
    },
    /// DTMF tone received
    DtmfReceived {
        digit: char,
    },
    /// Connection quality changed
    QualityChanged {
        /// Packet loss percentage (0-100)
        packet_loss: f32,
        /// Round-trip time in milliseconds
        rtt_ms: u32,
        /// Jitter in milliseconds
        jitter_ms: u32,
    },
    /// P2 FIX: ICE candidate discovered (for trickle ICE signaling)
    IceCandidate {
        /// ICE candidate string
        candidate: String,
        /// SDP mid (media stream identification)
        sdp_mid: Option<String>,
        /// SDP media line index
        sdp_m_line_index: Option<u16>,
    },
    /// Connection closed
    Disconnected {
        reason: String,
    },
    /// Error occurred
    Error {
        message: String,
    },
}

/// Audio sink for sending audio to remote peer
#[async_trait]
pub trait AudioSink: Send + Sync {
    /// Send audio samples to remote peer
    async fn send_audio(&self, samples: &[f32], timestamp_ms: u64) -> Result<(), TransportError>;

    /// Get the audio format
    fn format(&self) -> AudioFormat;

    /// Flush any buffered audio
    async fn flush(&self) -> Result<(), TransportError>;
}

/// Audio source for receiving audio from remote peer
#[async_trait]
pub trait AudioSource: Send + Sync {
    /// Receive audio samples from remote peer
    ///
    /// Returns None if no audio is available (non-blocking)
    async fn recv_audio(&self) -> Result<Option<(Vec<f32>, u64)>, TransportError>;

    /// Get the audio format
    fn format(&self) -> AudioFormat;

    /// Set callback for incoming audio
    fn set_callback(&self, callback: mpsc::Sender<TransportEvent>);
}

/// Transport trait for WebRTC/WebSocket abstraction
#[async_trait]
pub trait Transport: Send + Sync {
    /// Connect to remote peer
    async fn connect(&mut self, offer: &str) -> Result<String, TransportError>;

    /// Accept incoming connection
    async fn accept(&mut self, offer: &str) -> Result<String, TransportError>;

    /// Close the connection
    async fn close(&mut self) -> Result<(), TransportError>;

    /// Check if connected
    fn is_connected(&self) -> bool;

    /// Get audio sink for sending audio
    fn audio_sink(&self) -> Option<Box<dyn AudioSink>>;

    /// Get audio source for receiving audio
    fn audio_source(&self) -> Option<Box<dyn AudioSource>>;

    /// Get session ID
    fn session_id(&self) -> &str;

    /// Get connection statistics
    fn stats(&self) -> ConnectionStats;

    /// Set event callback
    fn set_event_callback(&mut self, callback: mpsc::Sender<TransportEvent>);
}

/// Connection statistics
#[derive(Debug, Clone, Default)]
pub struct ConnectionStats {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets lost
    pub packets_lost: u64,
    /// Round-trip time in milliseconds
    pub rtt_ms: u32,
    /// Jitter in milliseconds
    pub jitter_ms: u32,
    /// Audio level (0.0 - 1.0)
    pub audio_level: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_stats_default() {
        let stats = ConnectionStats::default();
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.rtt_ms, 0);
    }
}
