//! Voice Agent Server Entry Point

use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use voice_agent_config::Settings;
use voice_agent_server::{create_router, AppState, init_metrics};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "voice_agent=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Voice Agent Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = Settings::default();
    tracing::info!("Loaded configuration");

    // P0 FIX: Initialize Prometheus metrics
    let _metrics_handle = init_metrics();
    tracing::info!("Initialized Prometheus metrics at /metrics");

    // Create application state
    let state = AppState::new(config.clone());
    tracing::info!("Initialized application state");

    // Create router
    let app = create_router(state);

    // Bind address
    let addr = SocketAddr::from(([0, 0, 0, 0], config.server.port));
    tracing::info!("Listening on {}", addr);

    // Start server with graceful shutdown
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // P1 FIX: Graceful shutdown on SIGTERM/SIGINT
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}

/// Wait for shutdown signal (Ctrl+C or SIGTERM)
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, initiating graceful shutdown...");
        }
        _ = terminate => {
            tracing::info!("Received SIGTERM, initiating graceful shutdown...");
        }
    }
}
