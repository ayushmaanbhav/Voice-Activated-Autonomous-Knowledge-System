//! Voice Agent Server
//!
//! Provides WebSocket, WebRTC, and HTTP endpoints for the voice agent.

pub mod auth;
pub mod http;
pub mod mcp_server; // P2 FIX: MCP JSON-RPC endpoint
pub mod metrics;
pub mod rate_limit;
pub mod session;
pub mod state;
pub mod webrtc; // P2 FIX: WebRTC signaling endpoints
pub mod websocket; // P1 FIX: Auth middleware

pub use auth::auth_middleware;
pub use http::create_router;
pub use metrics::{
    init_metrics, record_error, record_llm_latency, record_request, record_stt_latency,
    record_total_latency, record_tts_latency,
};
pub use rate_limit::{RateLimitError, RateLimiter};
pub use session::{
    InMemorySessionStore, RecoverableSession, ScyllaSessionStore, Session, SessionManager,
    SessionMetadata, SessionStore,
};
pub use state::AppState;
pub use webrtc::WebRtcSession;
pub use websocket::WebSocketHandler;

use thiserror::Error;

/// Server errors
#[derive(Error, Debug)]
pub enum ServerError {
    #[error("Session error: {0}")]
    Session(String),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("WebRTC error: {0}")]
    WebRtc(String),

    #[error("Authentication error: {0}")]
    Auth(String),

    #[error("Rate limit exceeded")]
    RateLimit,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Internal error: {0}")]
    Internal(String),

    /// P2 FIX: Persistence error for audit logging
    #[error("Persistence error: {0}")]
    Persistence(String),
}

impl From<ServerError> for axum::http::StatusCode {
    fn from(err: ServerError) -> Self {
        match err {
            ServerError::Session(_) => axum::http::StatusCode::NOT_FOUND,
            ServerError::WebSocket(_) => axum::http::StatusCode::BAD_REQUEST,
            ServerError::WebRtc(_) => axum::http::StatusCode::BAD_REQUEST,
            ServerError::Auth(_) => axum::http::StatusCode::UNAUTHORIZED,
            ServerError::RateLimit => axum::http::StatusCode::TOO_MANY_REQUESTS,
            ServerError::InvalidRequest(_) => axum::http::StatusCode::BAD_REQUEST,
            ServerError::Internal(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            ServerError::Persistence(_) => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}
