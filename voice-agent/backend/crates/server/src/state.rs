//! Application State
//!
//! Shared state across all handlers.

use std::sync::Arc;
use parking_lot::RwLock;

use voice_agent_config::{Settings, load_settings, DomainConfigManager};
use voice_agent_tools::ToolRegistry;

use crate::session::SessionManager;

/// Application state
#[derive(Clone)]
pub struct AppState {
    /// P1 FIX: Configuration wrapped in RwLock for hot-reload support
    pub config: Arc<RwLock<Settings>>,
    /// P4 FIX: Domain configuration manager for gold loan specific config
    pub domain_config: Arc<DomainConfigManager>,
    /// Session manager
    pub sessions: Arc<SessionManager>,
    /// Tool registry
    pub tools: Arc<ToolRegistry>,
    /// Environment name for config reload
    env: Option<String>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: Settings) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            domain_config: Arc::new(DomainConfigManager::new()),
            sessions: Arc::new(SessionManager::new(100)),
            tools: Arc::new(voice_agent_tools::registry::create_default_registry()),
            env: None,
        }
    }

    /// P4 FIX: Create new application state with domain config
    pub fn with_domain_config(config: Settings, domain_config: DomainConfigManager) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            domain_config: Arc::new(domain_config),
            sessions: Arc::new(SessionManager::new(100)),
            tools: Arc::new(voice_agent_tools::registry::create_default_registry()),
            env: None,
        }
    }

    /// Create new application state with environment name for reload support
    pub fn with_env(config: Settings, env: Option<String>) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            domain_config: Arc::new(DomainConfigManager::new()),
            sessions: Arc::new(SessionManager::new(100)),
            tools: Arc::new(voice_agent_tools::registry::create_default_registry()),
            env,
        }
    }

    /// P1 FIX: Reload configuration from files
    ///
    /// Reloads config from disk and updates the shared state.
    /// Returns the new config on success.
    pub fn reload_config(&self) -> Result<(), String> {
        let new_config = load_settings(self.env.as_deref())
            .map_err(|e| format!("Failed to reload config: {}", e))?;

        // Update the config
        let mut config = self.config.write();
        *config = new_config;

        tracing::info!("Configuration reloaded successfully");
        Ok(())
    }

    /// P4 FIX: Reload domain configuration
    pub fn reload_domain_config(&self) -> Result<(), String> {
        self.domain_config
            .reload()
            .map_err(|e| format!("Failed to reload domain config: {}", e))?;

        tracing::info!("Domain configuration reloaded successfully");
        Ok(())
    }

    /// Get a read guard to the current configuration
    pub fn get_config(&self) -> parking_lot::RwLockReadGuard<'_, Settings> {
        self.config.read()
    }

    /// P4 FIX: Get domain configuration manager
    pub fn get_domain_config(&self) -> &DomainConfigManager {
        &self.domain_config
    }
}
