//! Configuration management for the voice agent
//!
//! Supports loading configuration from:
//! - YAML/TOML files
//! - Environment variables (VOICE_AGENT_ prefix)
//! - Runtime overrides
//!
//! # Phase 6: Domain Configuration
//!
//! Comprehensive domain-specific configuration:
//! - Branch/location information
//! - Product features and eligibility
//! - Competitor details and comparison
//! - Prompt templates and scripts
//! - Unified domain config loader

pub mod agent;
pub mod branch;
pub mod competitor;
// P1 FIX: Centralized constants module
pub mod constants;
pub mod domain;
pub mod domain_config;
pub mod gold_loan;
pub mod pipeline;
pub mod product;
pub mod prompts;
pub mod settings;

pub use agent::{AgentConfig, MemoryConfig, PersonaConfig};
pub use gold_loan::{CompetitorRates, GoldLoanConfig, PurityFactors, TieredRates};
pub use pipeline::PipelineConfig;
pub use settings::{
    load_settings, AuthConfig, PersistenceConfig, RagConfig, RateLimitConfig, RuntimeEnvironment,
    ServerConfig, Settings, TurnServerConfig,
};

// Phase 6 exports
pub use branch::{
    Branch, BranchConfig, BranchFeatures, Coordinates, DoorstepServiceConfig, OperatingHours,
};
pub use competitor::{
    BalanceTransferBenefits, ComparisonPoint, Competitor, CompetitorConfig, CompetitorType,
    MonthlySavings, ObjectionHandler, SwitchingBenefits,
};
pub use domain::{domain_config, init_domain_config, DomainConfig, DomainConfigManager};
pub use product::{
    DigitalFeatures, DocumentationConfig, EligibilityConfig, ExistingCustomerBenefits,
    FeeStructure, FeeType, FeesConfig, GoldPurityRequirements, ProductConfig, ProductFeatures,
    ProductVariant, SellingPoint, TenureConfig,
};
pub use prompts::{
    ClosingTemplates, FallbackTemplates, GreetingTemplates, PromptTemplates, ResponseTemplates,
    StagePrompt, SystemPrompt,
};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Configuration file not found: {0}")]
    FileNotFound(String),

    #[error("Failed to parse configuration: {0}")]
    ParseError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    #[error("Environment error: {0}")]
    Environment(String),
}

impl From<config::ConfigError> for ConfigError {
    fn from(err: config::ConfigError) -> Self {
        ConfigError::ParseError(err.to_string())
    }
}
