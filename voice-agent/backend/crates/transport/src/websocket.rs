//! WebSocket Transport (Fallback)
//!
//! WebSocket-based audio transport for browsers that don't support WebRTC.
//! Higher latency than WebRTC but more widely supported.

use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::sync::mpsc;

use crate::{AudioFormat, TransportError};
use crate::traits::{Transport, TransportEvent, AudioSink, AudioSource, ConnectionStats};

/// WebSocket transport state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSocketState {
    New,
    Connecting,
    Connected,
    Disconnected,
    Closed,
}

/// WebSocket transport configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Audio format
    pub audio_format: AudioFormat,
    /// Buffer size in milliseconds
    pub buffer_ms: u32,
    /// Enable compression
    pub compression: bool,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            audio_format: AudioFormat::default(),
            buffer_ms: 100,
            compression: true,
        }
    }
}

/// WebSocket transport implementation
///
/// Note: This is a stub implementation. The actual WebSocket transport
/// is implemented in the server crate. This module provides the trait
/// interface for consistency.
#[allow(dead_code)]
pub struct WebSocketTransport {
    session_id: String,
    config: WebSocketConfig,
    state: Arc<RwLock<WebSocketState>>,
    event_tx: Option<mpsc::Sender<TransportEvent>>,
    stats: Arc<RwLock<ConnectionStats>>,
}

impl WebSocketTransport {
    /// Create a new WebSocket transport
    pub fn new(config: WebSocketConfig) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            config,
            state: Arc::new(RwLock::new(WebSocketState::New)),
            event_tx: None,
            stats: Arc::new(RwLock::new(ConnectionStats::default())),
        }
    }
}

#[async_trait]
impl Transport for WebSocketTransport {
    async fn connect(&mut self, _offer: &str) -> Result<String, TransportError> {
        // WebSocket doesn't use SDP offers
        *self.state.write() = WebSocketState::Connected;

        if let Some(tx) = &self.event_tx {
            let _ = tx.send(TransportEvent::Connected {
                session_id: self.session_id.clone(),
                remote_addr: None,
            }).await;
        }

        Ok("websocket-connected".to_string())
    }

    async fn accept(&mut self, offer: &str) -> Result<String, TransportError> {
        self.connect(offer).await
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        *self.state.write() = WebSocketState::Closed;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        *self.state.read() == WebSocketState::Connected
    }

    fn audio_sink(&self) -> Option<Box<dyn AudioSink>> {
        None // Implemented in server crate
    }

    fn audio_source(&self) -> Option<Box<dyn AudioSource>> {
        None // Implemented in server crate
    }

    fn session_id(&self) -> &str {
        &self.session_id
    }

    fn stats(&self) -> ConnectionStats {
        self.stats.read().clone()
    }

    fn set_event_callback(&mut self, callback: mpsc::Sender<TransportEvent>) {
        self.event_tx = Some(callback);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_config_default() {
        let config = WebSocketConfig::default();
        assert_eq!(config.buffer_ms, 100);
    }

    #[tokio::test]
    async fn test_websocket_connect() {
        let mut transport = WebSocketTransport::new(WebSocketConfig::default());
        let result = transport.connect("").await;
        assert!(result.is_ok());
        assert!(transport.is_connected());
    }
}
