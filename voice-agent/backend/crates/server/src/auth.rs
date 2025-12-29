//! P1 FIX: Authentication Middleware
//!
//! Simple API key authentication for the voice agent HTTP API.
//! Supports Bearer token authentication via Authorization header.

use axum::{
    extract::Request,
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use parking_lot::RwLock;
use voice_agent_config::Settings;

/// P1 FIX: Track if we've warned about auth being disabled (warn once only)
static AUTH_DISABLED_WARNED: AtomicBool = AtomicBool::new(false);

/// Authentication result after checking config
enum AuthCheck {
    /// Authentication disabled, pass through
    Disabled,
    /// Path is public, pass through
    PublicPath,
    /// Config error
    ConfigError(&'static str),
    /// Need to check API key with this expected key
    CheckKey(String),
}

/// Check auth config and return what action to take
///
/// This function extracts all needed config values synchronously
/// to avoid holding the RwLock guard across await points.
fn check_auth_config(
    config: &Arc<RwLock<Settings>>,
    path: &str,
) -> AuthCheck {
    let config_guard = config.read();
    let auth_config = &config_guard.server.auth;

    // P1 FIX: Log warning when auth is disabled (only once)
    // This is a security risk in production environments
    if !auth_config.enabled {
        if !AUTH_DISABLED_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "⚠️  API authentication is DISABLED! Set VOICE_AGENT__SERVER__AUTH__ENABLED=true for production."
            );
        }
        return AuthCheck::Disabled;
    }

    // Check if path is public (bypasses auth)
    if auth_config.public_paths.iter().any(|p| path.starts_with(p)) {
        return AuthCheck::PublicPath;
    }

    // Get the API key from config
    match &auth_config.api_key {
        Some(key) if !key.is_empty() => AuthCheck::CheckKey(key.clone()),
        _ => AuthCheck::ConfigError("Auth is enabled but no API key is configured"),
    }
    // config_guard is dropped here
}

/// Authentication middleware that checks for valid API key
///
/// # Authorization
/// - Checks for `Authorization: Bearer <api_key>` header
/// - Skips authentication for public paths (health, metrics)
/// - Returns 401 Unauthorized if auth is enabled but key is missing/invalid
///
/// # Configuration
/// Set via environment: `VOICE_AGENT__SERVER__AUTH__API_KEY=your-secret-key`
/// Enable via: `VOICE_AGENT__SERVER__AUTH__ENABLED=true`
pub async fn auth_middleware(
    request: Request,
    next: Next,
) -> Response {
    // Get config from request extensions
    let config = match request.extensions().get::<Arc<RwLock<Settings>>>() {
        Some(cfg) => cfg.clone(),
        None => {
            tracing::error!("Config extension not found in request");
            return (StatusCode::INTERNAL_SERVER_ERROR, "Server configuration error").into_response();
        }
    };

    // Check auth config synchronously (no await, so guard is dropped)
    let path = request.uri().path().to_string();
    let auth_check = check_auth_config(&config, &path);

    // Now handle the result without holding the lock
    match auth_check {
        AuthCheck::Disabled | AuthCheck::PublicPath => {
            next.run(request).await
        }
        AuthCheck::ConfigError(msg) => {
            tracing::error!("{}", msg);
            (StatusCode::INTERNAL_SERVER_ERROR, "Server authentication not configured").into_response()
        }
        AuthCheck::CheckKey(expected_key) => {
            // Extract Authorization header
            let auth_header = request.headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string());

            match auth_header {
                Some(header) if header.starts_with("Bearer ") => {
                    let provided_key = &header[7..]; // Skip "Bearer "

                    // Constant-time comparison to prevent timing attacks
                    if constant_time_compare(provided_key.as_bytes(), expected_key.as_bytes()) {
                        // Auth successful
                        next.run(request).await
                    } else {
                        tracing::warn!("Invalid API key provided from {:?}", request.headers().get("X-Forwarded-For"));
                        (StatusCode::UNAUTHORIZED, "Invalid API key").into_response()
                    }
                }
                Some(_) => {
                    (StatusCode::BAD_REQUEST, "Invalid Authorization header format. Expected: Bearer <token>").into_response()
                }
                None => {
                    (StatusCode::UNAUTHORIZED, "Missing Authorization header").into_response()
                }
            }
        }
    }
}

/// Constant-time comparison to prevent timing attacks
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare(b"secret", b"secret"));
        assert!(!constant_time_compare(b"secret", b"secre"));
        assert!(!constant_time_compare(b"secret", b"secreT"));
        assert!(!constant_time_compare(b"abc", b"xyz"));
    }
}
