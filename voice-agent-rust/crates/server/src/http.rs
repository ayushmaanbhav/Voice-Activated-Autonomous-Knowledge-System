//! HTTP Endpoints
//!
//! REST API for the voice agent.

use axum::{
    routing::{get, post, delete},
    Router,
    extract::{State, Path, Json},
    http::{StatusCode, HeaderValue, Method},
    response::IntoResponse,
    Extension,
};
use serde::{Deserialize, Serialize};
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tower_http::compression::CompressionLayer;

use crate::state::AppState;
use crate::websocket::{WebSocketHandler, create_session};
use crate::metrics::metrics_handler;
use crate::auth::auth_middleware;
use voice_agent_tools::ToolExecutor;

/// Create the application router
pub fn create_router(state: AppState) -> Router {
    // P0 FIX: Build CORS layer from configured origins instead of wildcard Any
    // P1 FIX: Now uses RwLock for hot-reload support
    let config = state.config.read();
    let cors_layer = build_cors_layer(&config.server.cors_origins, config.server.cors_enabled);
    drop(config); // Release lock before building router

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

        // P1 FIX: Config reload endpoint (admin only)
        .route("/admin/reload-config", post(reload_config))

        // WebSocket
        .route("/ws/:session_id", get(ws_handler))

        // Middleware (order matters - auth runs after CORS but before handlers)
        // P1 FIX: Apply auth middleware layer via Extension
        .layer(axum::middleware::from_fn(|req: axum::extract::Request, next: axum::middleware::Next| async move {
            auth_middleware(req, next).await
        }))
        .layer(Extension(state.config.clone()))
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(cors_layer)
        .with_state(state)
}

/// P0 FIX: Build CORS layer from configured origins
///
/// - If cors_enabled is false, returns permissive layer (for dev)
/// - If cors_origins is empty, defaults to localhost:3000 for safety
/// - Otherwise, uses the configured origins
fn build_cors_layer(origins: &[String], enabled: bool) -> CorsLayer {
    if !enabled {
        // CORS disabled - allow all (only for development!)
        tracing::warn!("CORS is disabled - allowing all origins (NOT FOR PRODUCTION)");
        return CorsLayer::permissive();
    }

    if origins.is_empty() {
        // No origins configured - default to localhost for safety
        tracing::info!("No CORS origins configured, defaulting to localhost:3000");
        return CorsLayer::new()
            .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers(Any);
    }

    // Parse configured origins
    let parsed_origins: Vec<HeaderValue> = origins
        .iter()
        .filter_map(|origin| {
            origin.parse::<HeaderValue>().ok().or_else(|| {
                tracing::warn!("Invalid CORS origin: {}", origin);
                None
            })
        })
        .collect();

    if parsed_origins.is_empty() {
        tracing::error!("All configured CORS origins are invalid, falling back to localhost");
        return CorsLayer::new()
            .allow_origin("http://localhost:3000".parse::<HeaderValue>().unwrap())
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers(Any);
    }

    tracing::info!("CORS configured with {} origins", parsed_origins.len());
    CorsLayer::new()
        .allow_origin(parsed_origins)
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any)
        .allow_credentials(true)
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

/// P2 FIX: Enhanced health check that verifies actual dependencies
async fn health_check(
    State(state): State<AppState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let config = state.get_config();
    let mut checks = serde_json::Map::new();
    let mut all_healthy = true;

    // Check 1: Tool registry initialized
    let tool_count = state.tools.list_tools().len();
    checks.insert("tools".to_string(), serde_json::json!({
        "status": if tool_count > 0 { "ok" } else { "degraded" },
        "count": tool_count
    }));

    // Check 2: VAD model exists
    let vad_path = std::path::Path::new(&config.models.vad);
    let vad_ok = vad_path.exists();
    checks.insert("vad_model".to_string(), serde_json::json!({
        "status": if vad_ok { "ok" } else { "missing" },
        "path": config.models.vad.clone()
    }));
    if !vad_ok { all_healthy = false; }

    // Check 3: TTS model
    let tts_path = std::path::Path::new(&config.models.tts);
    let tts_ok = tts_path.exists() || tts_path.parent().map(|p| p.exists()).unwrap_or(false);
    checks.insert("tts_model".to_string(), serde_json::json!({
        "status": if tts_ok { "ok" } else { "missing" },
        "path": config.models.tts.clone()
    }));

    // Check 4: STT model
    let stt_path = std::path::Path::new(&config.models.stt);
    let stt_ok = stt_path.exists() || stt_path.parent().map(|p| p.exists()).unwrap_or(false);
    checks.insert("stt_model".to_string(), serde_json::json!({
        "status": if stt_ok { "ok" } else { "missing" },
        "path": config.models.stt.clone()
    }));

    drop(config);

    let status = if all_healthy { "healthy" } else { "degraded" };
    let status_code = if all_healthy { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (status_code, Json(serde_json::json!({
        "status": status,
        "version": env!("CARGO_PKG_VERSION"),
        "checks": checks
    })))
}

/// P2 FIX: Enhanced readiness check with LLM backend connectivity
async fn readiness_check(
    State(state): State<AppState>,
) -> (StatusCode, Json<serde_json::Value>) {
    let session_count = state.sessions.count();

    // Extract config values before any await - parking_lot guards aren't Send
    let llm_endpoint = {
        let config = state.get_config();
        config.agent.llm.endpoint.clone()
    };

    let mut checks = serde_json::Map::new();
    let mut ready = true;

    // Check 1: Sessions system
    checks.insert("sessions".to_string(), serde_json::json!({
        "status": "ok",
        "count": session_count
    }));

    // Check 2: LLM backend (Ollama) connectivity
    let llm_url = format!("{}/api/tags", llm_endpoint);

    let llm_status = match tokio::time::timeout(
        std::time::Duration::from_secs(2),
        reqwest::get(&llm_url)
    ).await {
        Ok(Ok(resp)) if resp.status().is_success() => "ok",
        Ok(Ok(_)) => { ready = false; "error" },
        Ok(Err(_)) => { ready = false; "unreachable" },
        Err(_) => { ready = false; "timeout" },
    };

    checks.insert("llm_backend".to_string(), serde_json::json!({
        "status": llm_status,
        "url": llm_url
    }));

    let status = if ready { "ready" } else { "not_ready" };
    let status_code = if ready { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };

    (status_code, Json(serde_json::json!({
        "status": status,
        "checks": checks
    })))
}

/// P1 FIX: Config reload endpoint
///
/// POST /admin/reload-config
///
/// Reloads configuration from disk. Useful for updating settings without restart.
/// Note: Some settings (like CORS) are only applied at startup.
async fn reload_config(
    State(state): State<AppState>,
) -> impl IntoResponse {
    match state.reload_config() {
        Ok(()) => {
            (StatusCode::OK, Json(serde_json::json!({
                "status": "success",
                "message": "Configuration reloaded successfully"
            })))
        }
        Err(e) => {
            tracing::error!("Config reload failed: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "status": "error",
                "message": e
            })))
        }
    }
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
