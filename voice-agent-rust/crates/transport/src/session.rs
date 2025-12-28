//! Transport Session Management
//!
//! Manages transport sessions with automatic reconnection and failover.

use std::sync::Arc;
use std::time::Duration;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::{TransportError, AudioFormat};
use crate::traits::{Transport, TransportEvent};
use crate::webrtc::{WebRtcTransport, WebRtcConfig};
use crate::websocket::{WebSocketTransport, WebSocketConfig};

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Prefer WebRTC over WebSocket
    pub prefer_webrtc: bool,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Reconnection attempts
    pub reconnect_attempts: u32,
    /// Reconnection delay
    pub reconnect_delay: Duration,
    /// Audio format
    pub audio_format: AudioFormat,
    /// WebRTC-specific config
    pub webrtc: WebRtcConfig,
    /// WebSocket-specific config
    pub websocket: WebSocketConfig,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            prefer_webrtc: true,
            connect_timeout: Duration::from_secs(10),
            reconnect_attempts: 3,
            reconnect_delay: Duration::from_secs(1),
            audio_format: AudioFormat::default(),
            webrtc: WebRtcConfig::default(),
            websocket: WebSocketConfig::default(),
        }
    }
}

/// Transport session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Initial state
    New,
    /// Connecting
    Connecting,
    /// Connected via WebRTC
    ConnectedWebRtc,
    /// Connected via WebSocket
    ConnectedWebSocket,
    /// Reconnecting
    Reconnecting,
    /// Disconnected
    Disconnected,
    /// Closed
    Closed,
}

/// Transport session manager
///
/// Manages transport lifecycle with automatic failover from WebRTC to WebSocket.
pub struct TransportSession {
    session_id: String,
    config: SessionConfig,
    state: Arc<RwLock<SessionState>>,
    transport: Arc<RwLock<Option<Box<dyn Transport>>>>,
    event_tx: Option<mpsc::Sender<TransportEvent>>,
    reconnect_count: Arc<RwLock<u32>>,
}

impl TransportSession {
    /// Create a new transport session
    pub fn new(config: SessionConfig) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            config,
            state: Arc::new(RwLock::new(SessionState::New)),
            transport: Arc::new(RwLock::new(None)),
            event_tx: None,
            reconnect_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get current state
    pub fn state(&self) -> SessionState {
        *self.state.read()
    }

    /// Set event callback
    pub fn set_event_callback(&mut self, callback: mpsc::Sender<TransportEvent>) {
        self.event_tx = Some(callback);
    }

    /// Connect with SDP offer
    ///
    /// Tries WebRTC first, falls back to WebSocket if WebRTC fails.
    pub async fn connect(&mut self, offer: &str) -> Result<String, TransportError> {
        *self.state.write() = SessionState::Connecting;

        // Try WebRTC first if preferred
        if self.config.prefer_webrtc {
            match self.connect_webrtc(offer).await {
                Ok(answer) => {
                    *self.state.write() = SessionState::ConnectedWebRtc;
                    return Ok(answer);
                }
                Err(e) => {
                    tracing::warn!("WebRTC connection failed, falling back to WebSocket: {}", e);
                }
            }
        }

        // Fall back to WebSocket
        match self.connect_websocket(offer).await {
            Ok(answer) => {
                *self.state.write() = SessionState::ConnectedWebSocket;
                Ok(answer)
            }
            Err(e) => {
                *self.state.write() = SessionState::Disconnected;
                Err(e)
            }
        }
    }

    /// Connect via WebRTC
    async fn connect_webrtc(&mut self, offer: &str) -> Result<String, TransportError> {
        let mut transport = WebRtcTransport::new(self.config.webrtc.clone()).await?;

        if let Some(tx) = &self.event_tx {
            transport.set_event_callback(tx.clone());
        }

        let result = timeout(
            self.config.connect_timeout,
            transport.connect(offer),
        ).await
        .map_err(|_| TransportError::Timeout("WebRTC connection timeout".to_string()))?;

        match result {
            Ok(answer) => {
                *self.transport.write() = Some(Box::new(transport));
                Ok(answer)
            }
            Err(e) => Err(e),
        }
    }

    /// Connect via WebSocket
    async fn connect_websocket(&mut self, offer: &str) -> Result<String, TransportError> {
        let mut transport = WebSocketTransport::new(self.config.websocket.clone());

        if let Some(tx) = &self.event_tx {
            transport.set_event_callback(tx.clone());
        }

        let result = timeout(
            self.config.connect_timeout,
            transport.connect(offer),
        ).await
        .map_err(|_| TransportError::Timeout("WebSocket connection timeout".to_string()))?;

        match result {
            Ok(answer) => {
                *self.transport.write() = Some(Box::new(transport));
                Ok(answer)
            }
            Err(e) => Err(e),
        }
    }

    /// Close the session
    pub async fn close(&mut self) -> Result<(), TransportError> {
        if let Some(mut transport) = self.transport.write().take() {
            transport.close().await?;
        }

        *self.state.write() = SessionState::Closed;
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        matches!(
            *self.state.read(),
            SessionState::ConnectedWebRtc | SessionState::ConnectedWebSocket
        )
    }

    /// Get transport reference
    pub fn transport(&self) -> Option<impl std::ops::Deref<Target = Box<dyn Transport>> + '_> {
        let guard = self.transport.read();
        if guard.is_some() {
            Some(parking_lot::RwLockReadGuard::map(guard, |opt| opt.as_ref().unwrap()))
        } else {
            None
        }
    }

    /// Attempt reconnection
    pub async fn reconnect(&mut self, offer: &str) -> Result<String, TransportError> {
        let mut attempts = 0;
        let max_attempts = self.config.reconnect_attempts;

        *self.state.write() = SessionState::Reconnecting;

        while attempts < max_attempts {
            attempts += 1;
            *self.reconnect_count.write() = attempts;

            tracing::info!("Reconnection attempt {}/{}", attempts, max_attempts);

            match self.connect(offer).await {
                Ok(answer) => {
                    *self.reconnect_count.write() = 0;
                    return Ok(answer);
                }
                Err(e) => {
                    tracing::warn!("Reconnection attempt {} failed: {}", attempts, e);

                    if attempts < max_attempts {
                        tokio::time::sleep(self.config.reconnect_delay).await;
                    }
                }
            }
        }

        *self.state.write() = SessionState::Disconnected;
        Err(TransportError::ConnectionFailed(format!(
            "Failed to reconnect after {} attempts",
            max_attempts
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert!(config.prefer_webrtc);
        assert_eq!(config.reconnect_attempts, 3);
    }

    #[tokio::test]
    async fn test_session_new() {
        let session = TransportSession::new(SessionConfig::default());
        assert_eq!(session.state(), SessionState::New);
        assert!(!session.is_connected());
    }
}
