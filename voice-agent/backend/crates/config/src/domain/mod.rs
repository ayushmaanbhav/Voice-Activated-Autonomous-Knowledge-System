//! Hierarchical Domain Configuration
//!
//! Provides a layered configuration system:
//! 1. Base config (config/base/defaults.yaml)
//! 2. Domain config (config/domains/{domain}/domain.yaml)
//! 3. Runtime overrides (per-session)
//!
//! Each crate accesses config through a specific "view" that translates
//! raw config into crate-specific terminology.

mod branches;
mod bridge;
mod competitors;
mod documents;
mod features;
mod goals;
mod intents;
mod master;
mod objections;
mod prompts;
mod scoring;
mod segments;
mod slots;
mod sms_templates;
mod stages;
mod tool_responses;
mod tools;
mod validator;
mod views;

pub use branches::{BranchDefaults, BranchEntry, BranchesConfig, BranchesConfigError, DoorstepServiceConfig};
pub use documents::{
    CustomerTypeEntry, DocumentEntry, DocumentsConfig, DocumentsConfigError, DocumentToolConfig,
    ImportantNotes, ServiceTypeEntry,
};
pub use competitors::{
    ComparisonPoint, CompetitorDefaults, CompetitorEntry, CompetitorsConfig,
    CompetitorsConfigError, RateRange,
};
pub use features::{FeatureDefinition, FeatureId, FeaturesConfig};
pub use goals::{
    ActionContext, ActionTemplate, ActionTemplatesConfig, GoalEntry, GoalsConfig, GoalsConfigError,
};
pub use intents::{IntentDefinition, IntentsConfig, IntentsConfigError};
pub use master::{
    ContextualRule, DomainBoostConfig, DomainBoostTermEntry, MasterDomainConfig,
    PhoneticCorrectionsConfig, PhoneticCorrectorParams, QueryExpansionConfig,
    QueryExpansionSettings, VocabularyConfig,
};
pub use objections::{
    ObjectionDefinition, ObjectionResponse, ObjectionsConfig, ObjectionsConfigError,
};
pub use prompts::{PromptsConfig, PromptsConfigError};
pub use scoring::{
    CategoryWeights, ConversionMultipliers, EscalationConfig, QualificationThresholds,
    ScoringConfig, ScoringConfigError, TrustScores,
};
pub use segments::{
    NumericThreshold, SegmentDefinition, SegmentDetection, SegmentsConfig, SegmentsConfigError,
};
pub use slots::{
    EnumValue, GoalDefinition, SlotDefinition, SlotType, SlotsConfig, SlotsConfigError,
};
pub use sms_templates::{SmsCategories, SmsConfig, SmsTemplatesConfig, SmsTemplatesConfigError};
pub use stages::{
    StageDefinition, StageRequirements, StagesConfig, StagesConfigError, TransitionTrigger,
};
pub use tool_responses::{ToolResponsesConfig, ToolResponsesConfigError, ToolTemplates, TemplateVariant};
pub use tools::{IntentToolMapping, IntentToolMappingsConfig, ToolDefinition, ToolParameter, ToolSchema, ToolsConfig, ToolsConfigError};
pub use views::{AgentDomainView, CompetitorInfo, LlmDomainView, MonthlySavings, ToolsDomainView};

// P13 FIX: Domain bridge for trait implementations
pub use bridge::DomainBridge;

// P5.2 FIX: Config validator for startup validation
pub use validator::{
    ConfigValidator, ValidationCategory, ValidationError, ValidationResult, ValidationSeverity,
};

// P13 FIX: DomainConfig and DomainConfigManager removed - use MasterDomainConfig + views
