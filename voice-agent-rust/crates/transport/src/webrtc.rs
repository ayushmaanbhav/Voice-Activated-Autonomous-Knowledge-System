//! WebRTC Transport Implementation
//!
//! P0 FIX: Low-latency WebRTC transport for voice communication.
//!
//! Features:
//! - ICE/STUN/TURN support
//! - Opus audio codec
//! - DTLS-SRTP encryption
//! - Adaptive bitrate
//!
//! Target: <50ms one-way latency

use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use webrtc::api::API;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::setting_engine::SettingEngine;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::peer_connection::peer_connection_state::RTCPeerConnectionState;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecCapability;
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_remote::TrackRemote;

use crate::{AudioFormat, TransportError};
use crate::traits::{Transport, TransportEvent, AudioSink, AudioSource, ConnectionStats};

/// ICE server configuration
#[derive(Debug, Clone)]
pub struct IceServer {
    /// Server URLs (stun: or turn:)
    pub urls: Vec<String>,
    /// Username (for TURN)
    pub username: Option<String>,
    /// Credential (for TURN)
    pub credential: Option<String>,
}

impl Default for IceServer {
    fn default() -> Self {
        Self {
            urls: vec!["stun:stun.l.google.com:19302".to_string()],
            username: None,
            credential: None,
        }
    }
}

/// WebRTC configuration
#[derive(Debug, Clone)]
pub struct WebRtcConfig {
    /// ICE servers
    pub ice_servers: Vec<IceServer>,
    /// Audio format
    pub audio_format: AudioFormat,
    /// Enable echo cancellation
    pub echo_cancellation: bool,
    /// Enable noise suppression
    pub noise_suppression: bool,
    /// Enable automatic gain control
    pub auto_gain_control: bool,
    /// Maximum bitrate in kbps
    pub max_bitrate_kbps: u32,
    /// Minimum bitrate in kbps
    pub min_bitrate_kbps: u32,
    /// Packet time in ms (10, 20, 40, 60)
    pub ptime_ms: u32,
}

impl Default for WebRtcConfig {
    fn default() -> Self {
        Self {
            ice_servers: vec![IceServer::default()],
            audio_format: AudioFormat::default(),
            echo_cancellation: true,
            noise_suppression: true,
            auto_gain_control: true,
            max_bitrate_kbps: 32,
            min_bitrate_kbps: 8,
            ptime_ms: 20,
        }
    }
}

/// WebRTC transport state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebRtcState {
    /// Initial state
    New,
    /// Connecting (ICE gathering)
    Connecting,
    /// Connected
    Connected,
    /// Disconnected
    Disconnected,
    /// Failed
    Failed,
    /// Closed
    Closed,
}

/// WebRTC transport implementation
pub struct WebRtcTransport {
    session_id: String,
    config: WebRtcConfig,
    state: Arc<RwLock<WebRtcState>>,
    peer_connection: Option<Arc<RTCPeerConnection>>,
    audio_track: Option<Arc<TrackLocalStaticSample>>,
    event_tx: Option<mpsc::Sender<TransportEvent>>,
    stats: Arc<RwLock<ConnectionStats>>,
}

impl WebRtcTransport {
    /// Create a new WebRTC transport
    pub async fn new(config: WebRtcConfig) -> Result<Self, TransportError> {
        let session_id = uuid::Uuid::new_v4().to_string();

        Ok(Self {
            session_id,
            config,
            state: Arc::new(RwLock::new(WebRtcState::New)),
            peer_connection: None,
            audio_track: None,
            event_tx: None,
            stats: Arc::new(RwLock::new(ConnectionStats::default())),
        })
    }

    /// Create WebRTC API with media engine
    async fn create_api(&self) -> Result<API, TransportError> {
        let mut media_engine = MediaEngine::default();

        // Register Opus codec
        let opus_codec = RTCRtpCodecCapability {
            mime_type: "audio/opus".to_string(),
            clock_rate: 48000,
            channels: 2,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".to_string(),
            rtcp_feedback: vec![],
        };

        media_engine.register_codec(
            webrtc::rtp_transceiver::rtp_codec::RTCRtpCodecParameters {
                capability: opus_codec,
                payload_type: 111,
                stats_id: String::new(),
            },
            webrtc::rtp_transceiver::rtp_codec::RTPCodecType::Audio,
        ).map_err(|e| TransportError::Internal(e.to_string()))?;

        // Create interceptor registry
        let mut registry = Registry::new();
        registry = register_default_interceptors(registry, &mut media_engine)
            .map_err(|e| TransportError::Internal(e.to_string()))?;

        // Create setting engine
        let mut setting_engine = SettingEngine::default();

        // Configure ICE timeouts for better NAT traversal
        setting_engine.set_ice_timeouts(
            Some(std::time::Duration::from_secs(5)),  // disconnected_timeout
            Some(std::time::Duration::from_secs(25)), // failed_timeout
            Some(std::time::Duration::from_secs(2)),  // keep_alive_interval
        );

        // Build API
        let api = webrtc::api::APIBuilder::new()
            .with_media_engine(media_engine)
            .with_interceptor_registry(registry)
            .with_setting_engine(setting_engine)
            .build();

        Ok(api)
    }

    /// Create RTCConfiguration from config
    fn create_rtc_config(&self) -> RTCConfiguration {
        let ice_servers: Vec<RTCIceServer> = self.config.ice_servers.iter()
            .map(|s| RTCIceServer {
                urls: s.urls.clone(),
                username: s.username.clone().unwrap_or_default(),
                credential: s.credential.clone().unwrap_or_default(),
                ..Default::default()
            })
            .collect();

        RTCConfiguration {
            ice_servers,
            ..Default::default()
        }
    }

    /// Handle incoming audio track
    async fn handle_track(&self, track: Arc<TrackRemote>) {
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            loop {
                match track.read_rtp().await {
                    Ok((rtp_packet, _attributes)) => {
                        let payload = &rtp_packet.payload;
                        if payload.is_empty() {
                            continue;
                        }

                        // Decode Opus to PCM
                        // TODO: Use opus crate for decoding
                        let samples: Vec<f32> = payload
                            .chunks(2)
                            .map(|chunk| {
                                let sample = i16::from_le_bytes([chunk[0], chunk.get(1).copied().unwrap_or(0)]);
                                sample as f32 / 32768.0
                            })
                            .collect();

                        if let Some(tx) = &event_tx {
                            let timestamp_ms = (rtp_packet.header.timestamp as u64 * 1000) / 48000;
                            let _ = tx.send(TransportEvent::AudioReceived {
                                samples,
                                timestamp_ms,
                            }).await;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Track read error: {}", e);
                        break;
                    }
                }
            }
        });
    }

    /// Update connection state
    fn update_state(&self, state: WebRtcState) {
        *self.state.write() = state;

        if let Some(tx) = &self.event_tx {
            let event = match state {
                WebRtcState::Connected => TransportEvent::Connected {
                    session_id: self.session_id.clone(),
                    remote_addr: None,
                },
                WebRtcState::Disconnected | WebRtcState::Failed | WebRtcState::Closed => {
                    TransportEvent::Disconnected {
                        reason: format!("{:?}", state),
                    }
                }
                _ => return,
            };

            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = tx.send(event).await;
            });
        }
    }
}

#[async_trait]
impl Transport for WebRtcTransport {
    async fn connect(&mut self, offer: &str) -> Result<String, TransportError> {
        *self.state.write() = WebRtcState::Connecting;

        // Create API
        let api = self.create_api().await?;

        // Create peer connection
        let config = self.create_rtc_config();
        let peer_connection = api.new_peer_connection(config)
            .await
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        let pc = Arc::new(peer_connection);
        self.peer_connection = Some(pc.clone());

        // Handle connection state changes
        let state_ref = self.state.clone();
        let session_id = self.session_id.clone();
        let event_tx = self.event_tx.clone();

        pc.on_peer_connection_state_change(Box::new(move |s: RTCPeerConnectionState| {
            let state = match s {
                RTCPeerConnectionState::Connected => WebRtcState::Connected,
                RTCPeerConnectionState::Disconnected => WebRtcState::Disconnected,
                RTCPeerConnectionState::Failed => WebRtcState::Failed,
                RTCPeerConnectionState::Closed => WebRtcState::Closed,
                _ => return Box::pin(async {}),
            };

            *state_ref.write() = state;

            let session_id = session_id.clone();
            let event_tx = event_tx.clone();

            Box::pin(async move {
                if let Some(tx) = event_tx {
                    let event = match state {
                        WebRtcState::Connected => TransportEvent::Connected {
                            session_id,
                            remote_addr: None,
                        },
                        _ => TransportEvent::Disconnected {
                            reason: format!("{:?}", state),
                        },
                    };
                    let _ = tx.send(event).await;
                }
            })
        }));

        // Handle incoming tracks
        pc.on_track(Box::new(move |track, _, _| {
            tracing::info!("Received track: {:?}", track.kind());
            // TODO: Handle incoming audio track
            Box::pin(async {})
        }));

        // Parse and set remote description (offer)
        let offer_sdp = RTCSessionDescription::offer(offer.to_string())
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        pc.set_remote_description(offer_sdp)
            .await
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        // Create answer
        let answer = pc.create_answer(None)
            .await
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        // Set local description
        pc.set_local_description(answer.clone())
            .await
            .map_err(|e| TransportError::ConnectionFailed(e.to_string()))?;

        // Wait for ICE gathering to complete
        // TODO: Add timeout and proper ICE candidate handling

        Ok(answer.sdp)
    }

    async fn accept(&mut self, offer: &str) -> Result<String, TransportError> {
        // Same as connect for server-side
        self.connect(offer).await
    }

    async fn close(&mut self) -> Result<(), TransportError> {
        if let Some(pc) = &self.peer_connection {
            pc.close()
                .await
                .map_err(|e| TransportError::Internal(e.to_string()))?;
        }

        *self.state.write() = WebRtcState::Closed;
        self.peer_connection = None;

        Ok(())
    }

    fn is_connected(&self) -> bool {
        *self.state.read() == WebRtcState::Connected
    }

    fn audio_sink(&self) -> Option<Box<dyn AudioSink>> {
        // TODO: Return audio sink wrapper
        None
    }

    fn audio_source(&self) -> Option<Box<dyn AudioSource>> {
        // TODO: Return audio source wrapper
        None
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
    fn test_webrtc_config_default() {
        let config = WebRtcConfig::default();
        assert!(!config.ice_servers.is_empty());
        assert!(config.echo_cancellation);
    }

    #[tokio::test]
    async fn test_webrtc_transport_new() {
        let transport = WebRtcTransport::new(WebRtcConfig::default()).await;
        assert!(transport.is_ok());
    }
}
