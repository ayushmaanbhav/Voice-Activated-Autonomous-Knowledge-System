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

pub mod settings;
pub mod pipeline;
pub mod agent;
pub mod gold_loan;
pub mod branch;
pub mod product;
pub mod competitor;
pub mod prompts;
pub mod domain;

pub use settings::{Settings, ServerConfig, RateLimitConfig, AuthConfig, RuntimeEnvironment, load_settings};
pub use pipeline::PipelineConfig;
pub use agent::{AgentConfig, PersonaConfig};
pub use gold_loan::{GoldLoanConfig, PurityFactors, CompetitorRates, TieredRates};

// Phase 6 exports
pub use branch::{
    BranchConfig, Branch, Coordinates, OperatingHours, BranchFeatures,
    DoorstepServiceConfig,
};
pub use product::{
    ProductConfig, ProductVariant, EligibilityConfig, DocumentationConfig,
    ProductFeatures, TenureConfig, FeesConfig, FeeStructure, FeeType,
    GoldPurityRequirements, ExistingCustomerBenefits, SellingPoint, DigitalFeatures,
};
pub use competitor::{
    CompetitorConfig, Competitor, CompetitorType, ComparisonPoint,
    SwitchingBenefits, BalanceTransferBenefits, ObjectionHandler, MonthlySavings,
};
pub use prompts::{
    PromptTemplates, SystemPrompt, StagePrompt, ResponseTemplates,
    GreetingTemplates, ClosingTemplates, FallbackTemplates,
};
pub use domain::{
    DomainConfig, DomainConfigManager, domain_config, init_domain_config,
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
