//! Voice Agent Server
//!
//! Provides WebSocket, WebRTC, and HTTP endpoints for the voice agent.

pub mod websocket;
pub mod webrtc;  // P2 FIX: WebRTC signaling endpoints
pub mod http;
pub mod session;
pub mod state;
pub mod rate_limit;
pub mod metrics;
pub mod auth;  // P1 FIX: Auth middleware

pub use websocket::WebSocketHandler;
pub use webrtc::WebRtcSession;
pub use http::create_router;
pub use auth::auth_middleware;
pub use session::{Session, SessionManager, SessionStore, SessionMetadata, InMemorySessionStore, ScyllaSessionStore, RecoverableSession};
pub use state::AppState;
pub use rate_limit::{RateLimiter, RateLimitError};
pub use metrics::{init_metrics, record_request, record_stt_latency, record_llm_latency, record_tts_latency, record_total_latency, record_error};

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
