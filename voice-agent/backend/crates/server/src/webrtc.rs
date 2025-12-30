//! WebRTC Signaling Handler
//!
//! HTTP endpoints for WebRTC signaling (SDP offer/answer, ICE candidates).
//! Works alongside WebSocket for low-latency audio transport.
//!
//! # Integration Flow
//!
//! 1. Client creates session via POST /api/sessions
//! 2. Client sends SDP offer via POST /api/webrtc/:session_id/offer
//! 3. Server returns SDP answer
//! 4. Both sides exchange ICE candidates via trickle ICE
//! 5. WebRTC connection established, audio flows directly
//!
//! # Benefits over WebSocket-only
//!
//! - Lower latency (<50ms vs ~100ms)
//! - Better NAT traversal (STUN/TURN)
//! - Native browser support
//! - Adaptive bitrate

use std::sync::Arc;
use axum::{
    extract::{State, Path, Json},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock, Mutex};

use voice_agent_transport::{
    WebRtcTransport, WebRtcConfig, IceServer, IceCandidate,
    TransportEvent, Transport,
};
use voice_agent_pipeline::{VoicePipeline, PipelineConfig, PipelineEvent};
use voice_agent_core::{AudioFrame, SampleRate, Channels};

use crate::state::AppState;
use crate::session::Session;

/// WebRTC session state stored alongside the voice session
pub struct WebRtcSession {
    /// The WebRTC transport (using tokio RwLock for async safety)
    pub transport: Arc<RwLock<WebRtcTransport>>,
    /// Event receiver for transport events
    pub event_rx: mpsc::Receiver<TransportEvent>,
    /// ICE candidate receiver (for trickle ICE)
    pub ice_rx: mpsc::Receiver<IceCandidate>,
    /// P1 FIX: Voice pipeline for audio processing
    pub pipeline: Option<Arc<Mutex<VoicePipeline>>>,
    /// P1 FIX: Audio processing task handle
    pub audio_task: Option<tokio::task::JoinHandle<()>>,
    /// P1 FIX: Pipeline event task handle
    pub pipeline_task: Option<tokio::task::JoinHandle<()>>,
}

/// SDP offer from/to client
#[derive(Debug, Serialize, Deserialize)]
pub struct SdpOffer {
    /// SDP type (always "offer")
    #[serde(rename = "type")]
    pub sdp_type: String,
    /// SDP content
    pub sdp: String,
}

/// SDP answer to client
#[derive(Debug, Serialize)]
pub struct SdpAnswer {
    /// SDP type (always "answer")
    #[serde(rename = "type")]
    pub sdp_type: String,
    /// SDP content
    pub sdp: String,
    /// Session ID for subsequent requests
    pub session_id: String,
}

/// ICE candidate from/to client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IceCandidateRequest {
    /// Candidate string
    pub candidate: String,
    /// SDP media line index
    #[serde(rename = "sdpMLineIndex")]
    pub sdp_m_line_index: Option<u16>,
    /// SDP mid
    #[serde(rename = "sdpMid")]
    pub sdp_mid: Option<String>,
    /// Username fragment
    #[serde(rename = "usernameFragment")]
    pub username_fragment: Option<String>,
}

impl From<IceCandidate> for IceCandidateRequest {
    fn from(c: IceCandidate) -> Self {
        Self {
            candidate: c.candidate,
            sdp_m_line_index: c.sdp_m_line_index,
            sdp_mid: c.sdp_mid,
            username_fragment: c.username_fragment,
        }
    }
}

impl From<IceCandidateRequest> for IceCandidate {
    fn from(c: IceCandidateRequest) -> Self {
        Self {
            candidate: c.candidate,
            sdp_m_line_index: c.sdp_m_line_index,
            sdp_mid: c.sdp_mid,
            username_fragment: c.username_fragment,
        }
    }
}

/// WebRTC connection status response
#[derive(Debug, Serialize)]
pub struct WebRtcStatus {
    /// Connection state
    pub state: String,
    /// ICE gathering state
    pub ice_gathering_state: Option<String>,
    /// ICE connection state
    pub ice_connection_state: Option<String>,
    /// Number of local ICE candidates
    pub local_candidate_count: usize,
}

/// Handle WebRTC offer and return answer
///
/// POST /api/webrtc/:session_id/offer
///
/// Accepts an SDP offer from the client, creates a WebRTC transport,
/// and returns the SDP answer.
pub async fn handle_offer(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(offer): Json<SdpOffer>,
) -> Result<Json<SdpAnswer>, (StatusCode, Json<serde_json::Value>)> {
    // Verify session exists
    let session = state.sessions.get(&session_id)
        .ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Session not found" }))
        ))?;

    // Get WebRTC config from settings
    let webrtc_config = {
        let config = state.config.read();
        build_webrtc_config(&config)
    };

    // Create WebRTC transport
    let mut transport = WebRtcTransport::new(webrtc_config)
        .await
        .map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to create transport: {}", e) }))
        ))?;

    // Set up event channel
    let (event_tx, event_rx) = mpsc::channel::<TransportEvent>(100);
    transport.set_event_callback(event_tx);

    // Set up ICE candidate channel for trickle ICE
    let (ice_tx, ice_rx) = mpsc::channel::<IceCandidate>(50);
    transport.set_ice_candidate_callback(ice_tx);

    // Process the offer and get answer
    let answer_sdp = transport.connect(&offer.sdp)
        .await
        .map_err(|e| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": format!("Failed to process offer: {}", e) }))
        ))?;

    // P1 FIX: Create voice pipeline for WebRTC audio processing
    let pipeline = match VoicePipeline::simple(PipelineConfig::default()) {
        Ok(p) => {
            tracing::info!("Created voice pipeline for WebRTC session {}", session_id);
            Some(Arc::new(Mutex::new(p)))
        }
        Err(e) => {
            tracing::warn!(
                session_id = %session_id,
                error = %e,
                "Failed to create voice pipeline for WebRTC, audio processing disabled"
            );
            None
        }
    };

    let transport = Arc::new(RwLock::new(transport));

    // P1 FIX: Spawn audio processing task if pipeline is available
    let (audio_task, pipeline_task) = if let Some(ref pipeline) = pipeline {
        let (audio_handle, pipeline_handle) = spawn_webrtc_audio_processor(
            transport.clone(),
            pipeline.clone(),
            session.clone(),
        ).await;
        (Some(audio_handle), Some(pipeline_handle))
    } else {
        (None, None)
    };

    // Store WebRTC session
    let webrtc_session = WebRtcSession {
        transport,
        event_rx,
        ice_rx,
        pipeline,
        audio_task,
        pipeline_task,
    };

    // Store in session
    session.set_webrtc_transport(webrtc_session);

    tracing::info!(
        session_id = %session_id,
        "WebRTC offer processed, answer generated, audio pipeline wired"
    );

    Ok(Json(SdpAnswer {
        sdp_type: "answer".to_string(),
        sdp: answer_sdp,
        session_id: session_id.clone(),
    }))
}

/// Add ICE candidate from client
///
/// POST /api/webrtc/:session_id/ice
///
/// Receives an ICE candidate from the client and adds it to the
/// peer connection for connectivity establishment.
#[axum::debug_handler]
pub async fn add_ice_candidate(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(candidate): Json<IceCandidateRequest>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    let session = state.sessions.get(&session_id)
        .ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Session not found" }))
        ))?;

    // Get the WebRTC transport from session
    let transport = session.get_webrtc_transport()
        .ok_or_else(|| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "WebRTC not initialized for this session" }))
        ))?;

    let ice_candidate: IceCandidate = candidate.into();

    transport.read().await
        .add_ice_candidate(&ice_candidate)
        .await
        .map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to add ICE candidate: {}", e) }))
        ))?;

    tracing::debug!(
        session_id = %session_id,
        candidate = %ice_candidate.candidate,
        "Added remote ICE candidate"
    );

    Ok(StatusCode::NO_CONTENT)
}

/// Get server ICE candidates
///
/// GET /api/webrtc/:session_id/candidates
///
/// Returns all collected local ICE candidates. Use this for
/// non-trickle ICE or to get candidates that were discovered
/// before the client started polling.
pub async fn get_ice_candidates(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<Vec<IceCandidateRequest>>, (StatusCode, Json<serde_json::Value>)> {
    let session = state.sessions.get(&session_id)
        .ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Session not found" }))
        ))?;

    let transport = session.get_webrtc_transport()
        .ok_or_else(|| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "WebRTC not initialized for this session" }))
        ))?;

    let candidates: Vec<IceCandidateRequest> = transport.read().await
        .local_candidates()
        .into_iter()
        .map(|c| c.into())
        .collect();

    Ok(Json(candidates))
}

/// Get WebRTC connection status
///
/// GET /api/webrtc/:session_id/status
///
/// Returns the current state of the WebRTC connection.
pub async fn get_status(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<WebRtcStatus>, (StatusCode, Json<serde_json::Value>)> {
    let session = state.sessions.get(&session_id)
        .ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Session not found" }))
        ))?;

    let transport = session.get_webrtc_transport()
        .ok_or_else(|| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "WebRTC not initialized for this session" }))
        ))?;

    let guard = transport.read().await;

    // Use ice_connection_state as the primary state indicator
    let state = if guard.is_connected() {
        "connected".to_string()
    } else {
        guard.ice_connection_state().unwrap_or_else(|| "unknown".to_string())
    };

    Ok(Json(WebRtcStatus {
        state,
        ice_gathering_state: guard.ice_gathering_state(),
        ice_connection_state: guard.ice_connection_state(),
        local_candidate_count: guard.local_candidates().len(),
    }))
}

/// Initiate ICE restart
///
/// POST /api/webrtc/:session_id/restart
///
/// Triggers an ICE restart when connectivity is lost or degraded.
/// Returns a new SDP offer that the client should process.
#[axum::debug_handler]
pub async fn ice_restart(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<SdpOffer>, (StatusCode, Json<serde_json::Value>)> {
    let session = state.sessions.get(&session_id)
        .ok_or_else(|| (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Session not found" }))
        ))?;

    let transport = session.get_webrtc_transport()
        .ok_or_else(|| (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "WebRTC not initialized for this session" }))
        ))?;

    let new_offer = transport.read().await
        .ice_restart()
        .await
        .map_err(|e| (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("ICE restart failed: {}", e) }))
        ))?;

    tracing::info!(
        session_id = %session_id,
        "ICE restart initiated"
    );

    Ok(Json(SdpOffer {
        sdp_type: "offer".to_string(),
        sdp: new_offer,
    }))
}

/// P1 FIX: Spawn WebRTC audio processing tasks
///
/// Creates two tasks:
/// 1. Audio receiver task: Receives audio from WebRTC transport, resamples 48kHz→16kHz,
///    and feeds to the voice pipeline
/// 2. Pipeline event task: Handles pipeline events (transcripts) and sends to agent/TTS output
async fn spawn_webrtc_audio_processor(
    transport: Arc<RwLock<WebRtcTransport>>,
    pipeline: Arc<Mutex<VoicePipeline>>,
    session: Arc<Session>,
) -> (tokio::task::JoinHandle<()>, tokio::task::JoinHandle<()>) {
    let session_id = session.id.clone();

    // Get audio source and sink from transport
    let (audio_source, audio_sink) = {
        let guard = transport.read().await;
        (guard.audio_source(), guard.audio_sink())
    };

    // P1 FIX: Audio receiver task - receives WebRTC audio and feeds to pipeline
    let pipeline_for_audio = pipeline.clone();
    let session_for_audio = session.clone();
    let session_id_for_audio = session_id.clone();

    let audio_task = tokio::spawn(async move {
        // Unwrap audio source - if None, task exits immediately
        let audio_source = match audio_source {
            Some(source) => source,
            None => {
                tracing::warn!(
                    session_id = %session_id_for_audio,
                    "No audio source available for WebRTC, audio task exiting"
                );
                return;
            }
        };

        let mut frame_count: u64 = 0;

        // P1 FIX: Simple resampling buffer (48kHz → 16kHz = 3:1 ratio)
        // Every 3 samples at 48kHz becomes 1 sample at 16kHz
        const RESAMPLE_RATIO: usize = 3;

        loop {
            // Try to receive audio from WebRTC
            match audio_source.recv_audio().await {
                Ok(Some((samples_48k, _timestamp_ms))) => {
                    session_for_audio.touch();

                    // P1 FIX: Resample from 48kHz to 16kHz by averaging every 3 samples
                    let samples_16k: Vec<f32> = samples_48k
                        .chunks(RESAMPLE_RATIO)
                        .map(|chunk| {
                            chunk.iter().sum::<f32>() / chunk.len() as f32
                        })
                        .collect();

                    if samples_16k.is_empty() {
                        continue;
                    }

                    // Create audio frame at 16kHz for pipeline
                    let frame = AudioFrame::new(
                        samples_16k,
                        SampleRate::Hz16000,
                        Channels::Mono,
                        frame_count,
                    );
                    frame_count += 1;

                    // Feed to pipeline
                    let pipeline_guard = pipeline_for_audio.lock().await;
                    if let Err(e) = pipeline_guard.process_audio(frame).await {
                        tracing::debug!(
                            session_id = %session_id_for_audio,
                            error = %e,
                            "WebRTC pipeline processing error"
                        );
                    }
                }
                Ok(None) => {
                    // No audio available, yield and try again
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
                Err(e) => {
                    tracing::debug!(
                        session_id = %session_id_for_audio,
                        error = %e,
                        "WebRTC audio receive error, stopping audio task"
                    );
                    break;
                }
            }
        }

        tracing::info!(
            session_id = %session_id_for_audio,
            "WebRTC audio receiver task ended"
        );
    });

    // P1 FIX: Pipeline event task - handles transcripts and sends to agent
    let session_for_pipeline = session.clone();
    let session_id_for_pipeline = session_id.clone();

    let pipeline_task = tokio::spawn(async move {
        let mut pipeline_events = pipeline.lock().await.subscribe();
        // P2 FIX: Track timestamp for TTS audio output
        let mut tts_timestamp_ms: u64 = 0;

        while let Ok(event) = pipeline_events.recv().await {
            match event {
                PipelineEvent::PartialTranscript(transcript) => {
                    tracing::debug!(
                        session_id = %session_id_for_pipeline,
                        text = %transcript.text,
                        "WebRTC partial transcript"
                    );
                    // Could send to WebRTC data channel if available
                }
                PipelineEvent::FinalTranscript(transcript) => {
                    let text = transcript.text.clone();
                    tracing::info!(
                        session_id = %session_id_for_pipeline,
                        text = %text,
                        "WebRTC final transcript, processing with agent"
                    );

                    // Process through agent
                    if !text.trim().is_empty() {
                        match session_for_pipeline.agent.process(&text).await {
                            Ok(response) => {
                                tracing::info!(
                                    session_id = %session_id_for_pipeline,
                                    response_len = response.len(),
                                    "Agent response generated for WebRTC"
                                );
                                // TTS audio will come through PipelineEvent::TtsAudio
                            }
                            Err(e) => {
                                tracing::error!(
                                    session_id = %session_id_for_pipeline,
                                    error = %e,
                                    "Agent processing failed for WebRTC"
                                );
                            }
                        }
                    }
                }
                PipelineEvent::VadStateChanged(vad_state) => {
                    tracing::debug!(
                        session_id = %session_id_for_pipeline,
                        is_speaking = ?vad_state,
                        "WebRTC VAD state changed"
                    );
                }
                PipelineEvent::TurnStateChanged(turn_state) => {
                    tracing::debug!(
                        session_id = %session_id_for_pipeline,
                        turn_type = ?turn_state,
                        "WebRTC turn state changed"
                    );
                }
                PipelineEvent::TtsAudio { samples, text: _, is_final } => {
                    // P2 FIX: Send TTS audio back via WebRTC audio sink
                    if let Some(ref sink) = audio_sink {
                        // Upsample from 16kHz to 48kHz for WebRTC (Opus expects 48kHz)
                        let samples_48k: Vec<f32> = samples
                            .iter()
                            .flat_map(|&s| std::iter::repeat(s).take(3))
                            .collect();

                        if let Err(e) = sink.send_audio(&samples_48k, tts_timestamp_ms).await {
                            tracing::debug!(
                                session_id = %session_id_for_pipeline,
                                error = %e,
                                "Failed to send TTS audio via WebRTC"
                            );
                        } else {
                            tracing::debug!(
                                session_id = %session_id_for_pipeline,
                                samples_len = samples_48k.len(),
                                is_final = is_final,
                                "Sent TTS audio via WebRTC"
                            );
                        }
                        // Advance timestamp by duration (samples at 48kHz)
                        tts_timestamp_ms += (samples_48k.len() as u64 * 1000) / 48000;
                    } else {
                        tracing::debug!(
                            session_id = %session_id_for_pipeline,
                            samples_len = samples.len(),
                            is_final = is_final,
                            "WebRTC TTS audio (no audio sink available)"
                        );
                    }
                }
                PipelineEvent::BargeIn { at_word } => {
                    tracing::debug!(
                        session_id = %session_id_for_pipeline,
                        at_word = at_word,
                        "WebRTC barge-in detected"
                    );
                    // P2 FIX: Flush any pending TTS audio on barge-in
                    if let Some(ref sink) = audio_sink {
                        let _ = sink.flush().await;
                    }
                }
                PipelineEvent::Error(e) => {
                    tracing::error!(
                        session_id = %session_id_for_pipeline,
                        error = %e,
                        "WebRTC pipeline error"
                    );
                }
            }
        }

        tracing::info!(
            session_id = %session_id_for_pipeline,
            "WebRTC pipeline event task ended"
        );
    });

    (audio_task, pipeline_task)
}

/// Build WebRTC config from application settings
fn build_webrtc_config(config: &voice_agent_config::Settings) -> WebRtcConfig {
    // Use configured STUN/TURN servers or defaults
    let ice_servers = if config.server.stun_servers.is_empty() {
        vec![IceServer::default()] // Google's public STUN server
    } else {
        config.server.stun_servers.iter().map(|url| {
            IceServer {
                urls: vec![url.clone()],
                username: None,
                credential: None,
            }
        }).collect()
    };

    // Add TURN servers if configured
    let mut all_servers = ice_servers;
    for turn in &config.server.turn_servers {
        all_servers.push(IceServer {
            urls: vec![turn.url.clone()],
            username: Some(turn.username.clone()),
            credential: Some(turn.credential.clone()),
        });
    }

    WebRtcConfig {
        ice_servers: all_servers,
        echo_cancellation: true,
        noise_suppression: true,
        auto_gain_control: true,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ice_candidate_conversion() {
        let request = IceCandidateRequest {
            candidate: "candidate:1 1 UDP 2122252543 192.168.1.1 12345 typ host".to_string(),
            sdp_m_line_index: Some(0),
            sdp_mid: Some("audio".to_string()),
            username_fragment: None,
        };

        let ice: IceCandidate = request.clone().into();
        assert_eq!(ice.candidate, request.candidate);
        assert_eq!(ice.sdp_m_line_index, request.sdp_m_line_index);

        let back: IceCandidateRequest = ice.into();
        assert_eq!(back.candidate, request.candidate);
    }
}
