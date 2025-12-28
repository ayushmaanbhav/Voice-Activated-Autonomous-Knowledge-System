//! HTTP Endpoints
//!
//! REST API for the voice agent.

use axum::{
    routing::{get, post, delete},
    Router,
    extract::{State, Path, Json},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tower_http::compression::CompressionLayer;

use crate::state::AppState;
use crate::websocket::{WebSocketHandler, create_session};
use crate::metrics::metrics_handler;
use voice_agent_tools::ToolExecutor;

/// Create the application router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Session endpoints
        .route("/api/sessions", post(create_session))
        .route("/api/sessions/:id", get(get_session))
        .route("/api/sessions/:id", delete(delete_session))
        .route("/api/sessions", get(list_sessions))

        // Chat endpoint (non-streaming)
        .route("/api/chat/:session_id", post(chat))

        // Tool endpoints
        .route("/api/tools", get(list_tools))
        .route("/api/tools/:name", post(call_tool))

        // Health check
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))

        // P0 FIX: Prometheus metrics endpoint
        .route("/metrics", get(metrics_handler))

        // WebSocket
        .route("/ws/:session_id", get(ws_handler))

        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        )
        .with_state(state)
}

/// Get session info
async fn get_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let session = state.sessions.get(&id)
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(serde_json::json!({
        "session_id": session.id,
        "active": session.is_active(),
        "stage": session.agent.stage().display_name(),
        "turn_count": session.agent.conversation().turn_count(),
    })))
}

/// Delete session
async fn delete_session(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> StatusCode {
    state.sessions.remove(&id);
    StatusCode::NO_CONTENT
}

/// List sessions
async fn list_sessions(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let sessions = state.sessions.list();
    Json(serde_json::json!({
        "sessions": sessions,
        "count": sessions.len(),
    }))
}

/// Chat request
#[derive(Debug, Deserialize)]
struct ChatRequest {
    message: String,
}

/// Chat response
#[derive(Debug, Serialize)]
struct ChatResponse {
    response: String,
    stage: String,
    turn_count: usize,
}

/// Chat endpoint
async fn chat(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(request): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let session = state.sessions.get(&session_id)
        .ok_or(StatusCode::NOT_FOUND)?;

    session.touch();

    match session.agent.process(&request.message).await {
        Ok(response) => {
            Ok(Json(ChatResponse {
                response,
                stage: session.agent.stage().display_name().to_string(),
                turn_count: session.agent.conversation().turn_count(),
            }))
        }
        Err(e) => {
            tracing::error!("Chat error: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// List tools
async fn list_tools(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let tools: Vec<serde_json::Value> = state.tools.list_tools()
        .into_iter()
        .map(|t| serde_json::json!({
            "name": t.name,
            "description": t.description,
        }))
        .collect();

    Json(serde_json::json!({
        "tools": tools,
    }))
}

/// Tool call request
#[derive(Debug, Deserialize)]
struct ToolCallRequest {
    arguments: serde_json::Value,
}

/// Call tool
async fn call_tool(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ToolCallRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    use voice_agent_tools::ToolExecutor;

    match state.tools.execute(&name, request.arguments).await {
        Ok(output) => {
            let content: Vec<serde_json::Value> = output.content
                .into_iter()
                .map(|c| match c {
                    voice_agent_tools::mcp::ContentBlock::Text { text } => {
                        serde_json::json!({ "type": "text", "text": text })
                    }
                    voice_agent_tools::mcp::ContentBlock::Image { data, mime_type } => {
                        serde_json::json!({ "type": "image", "data": data, "mime_type": mime_type })
                    }
                    voice_agent_tools::mcp::ContentBlock::Resource { uri, mime_type } => {
                        serde_json::json!({ "type": "resource", "uri": uri, "mime_type": mime_type })
                    }
                    // P2 FIX: Handle Audio content block for voice responses
                    voice_agent_tools::mcp::ContentBlock::Audio { data, mime_type, sample_rate, duration_ms } => {
                        serde_json::json!({
                            "type": "audio",
                            "data": data,
                            "mime_type": mime_type,
                            "sample_rate": sample_rate,
                            "duration_ms": duration_ms
                        })
                    }
                })
                .collect();

            Ok(Json(serde_json::json!({
                "content": content,
                "is_error": output.is_error,
            })))
        }
        Err(e) => {
            tracing::error!("Tool error: {:?}", e);
            Ok(Json(serde_json::json!({
                "content": [{ "type": "text", "text": e.message }],
                "is_error": true,
            })))
        }
    }
}

/// Health check
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Readiness check
async fn readiness_check(
    State(state): State<AppState>,
) -> impl IntoResponse {
    let session_count = state.sessions.count();

    Json(serde_json::json!({
        "status": "ready",
        "sessions": session_count,
    }))
}

/// WebSocket handler wrapper
async fn ws_handler(
    ws: axum::extract::ws::WebSocketUpgrade,
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<impl IntoResponse, StatusCode> {
    WebSocketHandler::handle(ws, State(state), Path(session_id)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use voice_agent_config::Settings;

    #[test]
    fn test_router_creation() {
        let state = AppState::new(Settings::default());
        let _ = create_router(state);
    }
}
