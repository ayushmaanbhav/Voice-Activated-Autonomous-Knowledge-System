//! Hierarchical Domain Configuration
//!
//! Provides a layered configuration system:
//! 1. Base config (config/base/defaults.yaml)
//! 2. Domain config (config/domains/{domain}/domain.yaml)
//! 3. Runtime overrides (per-session)
//!
//! Each crate accesses config through a specific "view" that translates
//! raw config into crate-specific terminology.

mod master;
mod views;

pub use master::MasterDomainConfig;
pub use views::{AgentDomainView, CompetitorInfo, LlmDomainView, ToolsDomainView};

// Re-export legacy DomainConfig for backward compatibility
pub use crate::domain_config::{
    domain_config, init_domain_config, DomainConfig, DomainConfigManager,
};
