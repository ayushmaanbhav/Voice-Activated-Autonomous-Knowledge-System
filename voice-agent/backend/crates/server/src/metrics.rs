//! Observability Metrics
//!
//! P0 FIX: Prometheus metrics endpoint for monitoring.

use axum::{
    extract::State,
    response::IntoResponse,
    http::{header, StatusCode},
};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use std::sync::OnceLock;

/// Global Prometheus handle
static METRICS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

/// Initialize metrics recorder
///
/// Must be called once at startup before recording any metrics.
pub fn init_metrics() -> PrometheusHandle {
    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("Failed to install Prometheus recorder");

    // Register default metrics
    register_default_metrics();

    METRICS_HANDLE.get_or_init(|| handle.clone());
    handle
}

/// Get the global metrics handle
pub fn get_metrics_handle() -> Option<&'static PrometheusHandle> {
    METRICS_HANDLE.get()
}

/// Register default application metrics
fn register_default_metrics() {
    // Session metrics
    gauge!("voice_agent_sessions_active").set(0.0);
    counter!("voice_agent_sessions_created_total").absolute(0);

    // Request metrics
    counter!("voice_agent_requests_total", "endpoint" => "health").absolute(0);
    counter!("voice_agent_requests_total", "endpoint" => "chat").absolute(0);
    counter!("voice_agent_requests_total", "endpoint" => "ws").absolute(0);

    // Pipeline metrics
    histogram!("voice_agent_stt_duration_seconds").record(0.0);
    histogram!("voice_agent_llm_duration_seconds").record(0.0);
    histogram!("voice_agent_tts_duration_seconds").record(0.0);
    histogram!("voice_agent_total_latency_seconds").record(0.0);

    // Error metrics
    counter!("voice_agent_errors_total", "type" => "stt").absolute(0);
    counter!("voice_agent_errors_total", "type" => "llm").absolute(0);
    counter!("voice_agent_errors_total", "type" => "tts").absolute(0);
    counter!("voice_agent_errors_total", "type" => "tool").absolute(0);
}

/// Record session created
pub fn record_session_created() {
    counter!("voice_agent_sessions_created_total").increment(1);
}

/// Record active sessions gauge
pub fn record_active_sessions(count: usize) {
    gauge!("voice_agent_sessions_active").set(count as f64);
}

/// Record request to endpoint
pub fn record_request(endpoint: &'static str) {
    counter!("voice_agent_requests_total", "endpoint" => endpoint).increment(1);
}

/// Record STT latency
pub fn record_stt_latency(duration_secs: f64) {
    histogram!("voice_agent_stt_duration_seconds").record(duration_secs);
}

/// Record LLM latency
pub fn record_llm_latency(duration_secs: f64) {
    histogram!("voice_agent_llm_duration_seconds").record(duration_secs);
}

/// Record TTS latency
pub fn record_tts_latency(duration_secs: f64) {
    histogram!("voice_agent_tts_duration_seconds").record(duration_secs);
}

/// Record total pipeline latency
pub fn record_total_latency(duration_secs: f64) {
    histogram!("voice_agent_total_latency_seconds").record(duration_secs);
}

/// Record error by type
pub fn record_error(error_type: &'static str) {
    counter!("voice_agent_errors_total", "type" => error_type).increment(1);
}

use crate::state::AppState;

/// Metrics endpoint handler
///
/// Returns Prometheus-formatted metrics.
pub async fn metrics_handler(
    State(state): State<AppState>,
) -> impl IntoResponse {
    // Update active sessions gauge
    let session_count = state.sessions.count();
    record_active_sessions(session_count);

    match get_metrics_handle() {
        Some(handle) => {
            let metrics = handle.render();
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
                metrics,
            )
        }
        None => {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(header::CONTENT_TYPE, "text/plain")],
                "Metrics not initialized".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_helpers() {
        // These should not panic
        record_request("test");
        record_stt_latency(0.1);
        record_llm_latency(0.5);
        record_tts_latency(0.2);
        record_total_latency(0.8);
        record_error("test");
    }
}
