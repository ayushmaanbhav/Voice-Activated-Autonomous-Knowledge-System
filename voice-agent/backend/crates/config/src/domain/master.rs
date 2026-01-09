//! Master Domain Configuration
//!
//! Loads and merges hierarchical YAML configuration:
//! - Base defaults (config/base/defaults.yaml)
//! - Domain-specific (config/domains/{domain}/domain.yaml)

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;

use crate::ConfigError;
use super::branches::BranchesConfig;
use super::competitors::CompetitorsConfig;
use super::documents::DocumentsConfig;
use super::features::FeaturesConfig;
use super::goals::GoalsConfig;
use super::objections::ObjectionsConfig;
use super::prompts::PromptsConfig;
use super::scoring::ScoringConfig;
use super::segments::SegmentsConfig;
use super::slots::SlotsConfig;
use super::sms_templates::SmsTemplatesConfig;
use super::stages::StagesConfig;
use super::tool_responses::ToolResponsesConfig;
use super::tools::ToolsConfig;

/// Brand configuration - domain-agnostic
///
/// P16 FIX: Renamed bank_name to company_name for domain-agnostic design.
/// The company_name field is the organization offering the service.
/// The product_name field is the specific product/service being offered.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BrandConfig {
    /// Company/organization name (e.g., "Kotak Mahindra Bank", "ABC Corp")
    #[serde(alias = "bank_name")]  // Backwards compatibility
    pub company_name: String,
    /// Product/service name (e.g., "Gold Loan", "Insurance")
    #[serde(default)]
    pub product_name: String,
    /// AI agent name
    pub agent_name: String,
    /// Agent role/title for persona (e.g., "Advisor", "Assistant")
    #[serde(default)]
    pub agent_role: String,
    /// Contact helpline number
    pub helpline: String,
    /// Website URL
    #[serde(default)]
    pub website: String,
}

/// Interest rate tier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateTier {
    /// Tier name (e.g., "Standard", "Premium", "Elite")
    #[serde(default)]
    pub name: String,
    /// Maximum amount for this tier (null = unlimited)
    pub max_amount: Option<f64>,
    /// Interest rate percentage
    pub rate: f64,
}

/// Interest rates configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InterestRatesConfig {
    #[serde(default)]
    pub tiers: Vec<RateTier>,
    #[serde(default)]
    pub base_rate: f64,
}

/// Loan limits configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LoanLimitsConfig {
    pub min: f64,
    pub max: f64,
}

/// Domain constants (source of truth for business values)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DomainConstants {
    #[serde(default)]
    pub interest_rates: InterestRatesConfig,
    #[serde(default)]
    pub ltv_percent: f64,
    #[serde(default)]
    pub loan_limits: LoanLimitsConfig,
    #[serde(default)]
    pub processing_fee_percent: f64,
    /// Asset price per unit (e.g., gold price per gram for gold loan domain)
    #[serde(default, alias = "gold_price_per_gram")]
    pub asset_price_per_unit: f64,
    /// Variant factors (e.g., purity factors for gold: K24=1.0, K22=0.916)
    #[serde(default, alias = "purity_factors")]
    pub variant_factors: HashMap<String, f64>,
}

impl DomainConstants {
    /// Legacy accessor for gold_price_per_gram
    pub fn gold_price_per_gram(&self) -> f64 {
        self.asset_price_per_unit
    }

    /// Legacy accessor for purity_factors
    pub fn purity_factors(&self) -> &HashMap<String, f64> {
        &self.variant_factors
    }
}

/// Competitor configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompetitorEntry {
    pub display_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub typical_rate: f64,
    #[serde(default)]
    pub ltv_percent: f64,
    #[serde(default)]
    pub competitor_type: String,
}

/// Product variant configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProductVariant {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub min_amount: Option<f64>,
    #[serde(default)]
    pub max_amount: Option<f64>,
    #[serde(default)]
    pub features: Vec<String>,
}

/// High-value customer configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HighValueConfig {
    #[serde(default)]
    pub amount_threshold: f64,
    #[serde(default)]
    pub weight_threshold_grams: f64,
    #[serde(default)]
    pub features: Vec<String>,
}

/// P15 FIX: Domain vocabulary configuration for text processing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VocabularyConfig {
    /// Domain-specific terms to preserve
    #[serde(default)]
    pub terms: Vec<String>,
    /// Common phrases in domain
    #[serde(default)]
    pub phrases: Vec<String>,
    /// Abbreviations and their expansions
    #[serde(default)]
    pub abbreviations: HashMap<String, String>,
    /// Entity types to preserve
    #[serde(default)]
    pub preserve_entities: Vec<String>,
}

/// Contextual correction rule for phonetic corrector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualRule {
    /// Previous word context
    pub context: String,
    /// Error word to correct
    pub error: String,
    /// Corrected value
    pub correction: String,
}

/// Phonetic corrector configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhoneticCorrectorParams {
    /// Maximum edit distance for fuzzy matching
    #[serde(default = "default_max_edit_distance")]
    pub max_edit_distance: i64,
    /// Minimum word length to attempt correction
    #[serde(default = "default_min_word_length")]
    pub min_word_length: usize,
    /// Whether to fix sentence-start errors (e.g., "Why" -> "I")
    #[serde(default = "default_true")]
    pub fix_sentence_start: bool,
}

fn default_max_edit_distance() -> i64 { 2 }
fn default_min_word_length() -> usize { 3 }
fn default_true() -> bool { true }

impl Default for PhoneticCorrectorParams {
    fn default() -> Self {
        Self {
            max_edit_distance: 2,
            min_word_length: 3,
            fix_sentence_start: true,
        }
    }
}

/// P16 FIX: Phonetic ASR error correction configuration
/// Moved from hardcoded PhoneticCorrector::gold_loan() to config-driven
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PhoneticCorrectionsConfig {
    /// Direct confusion rules: misspelling -> correct
    #[serde(default)]
    pub confusion_rules: HashMap<String, String>,
    /// Contextual rules: (prev_word, error_word) -> correction
    #[serde(default)]
    pub contextual_rules: Vec<ContextualRule>,
    /// Phrase-level corrections: phrase -> replacement
    #[serde(default)]
    pub phrase_rules: HashMap<String, String>,
    /// Configuration parameters
    #[serde(default)]
    pub config: PhoneticCorrectorParams,
}

/// Query expansion configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExpansionSettings {
    /// Maximum number of expansions per term
    #[serde(default = "default_max_expansions")]
    pub max_expansions: usize,
    /// Enable synonym expansion
    #[serde(default = "default_true")]
    pub use_synonyms: bool,
    /// Enable transliteration expansion
    #[serde(default = "default_true")]
    pub use_transliterations: bool,
    /// Enable stopword filtering
    #[serde(default = "default_true")]
    pub enable_stopword_filter: bool,
}

fn default_max_expansions() -> usize {
    3
}

impl Default for QueryExpansionSettings {
    fn default() -> Self {
        Self {
            max_expansions: 3,
            use_synonyms: true,
            use_transliterations: true,
            enable_stopword_filter: true,
        }
    }
}

/// Query expansion configuration for RAG
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryExpansionConfig {
    /// Configuration settings
    #[serde(default)]
    pub config: QueryExpansionSettings,
    /// Stopwords to filter from queries
    #[serde(default)]
    pub stopwords: Vec<String>,
    /// Synonym mappings (term -> list of synonyms)
    #[serde(default)]
    pub synonyms: HashMap<String, Vec<String>>,
    /// Hindi-Roman transliterations
    #[serde(default)]
    pub transliterations: HashMap<String, Vec<String>>,
}

/// Domain boost term entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainBoostTermEntry {
    /// The term to boost
    pub term: String,
    /// Category of the term
    pub category: String,
    /// Boost multiplier
    pub boost: f64,
    /// Related terms
    #[serde(default)]
    pub related: Vec<String>,
}

/// Domain boosting configuration for RAG
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DomainBoostConfig {
    /// Default boost multiplier
    #[serde(default = "default_boost")]
    pub default_boost: f64,
    /// Category-specific boost multipliers
    #[serde(default)]
    pub category_boosts: HashMap<String, f64>,
    /// Domain-specific terms with boosting
    #[serde(default)]
    pub terms: Vec<DomainBoostTermEntry>,
}

fn default_boost() -> f64 {
    1.0
}

/// Master domain configuration - the complete config for a domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterDomainConfig {
    /// Domain identifier (e.g., "gold_loan")
    pub domain_id: String,
    /// Human-readable domain name
    pub display_name: String,
    /// Version
    #[serde(default = "default_version")]
    pub version: String,
    /// Brand configuration
    #[serde(default)]
    pub brand: BrandConfig,
    /// Business constants
    #[serde(default)]
    pub constants: DomainConstants,
    /// Competitors
    #[serde(default)]
    pub competitors: HashMap<String, CompetitorEntry>,
    /// Product variants
    #[serde(default)]
    pub products: HashMap<String, ProductVariant>,
    /// High-value customer config
    #[serde(default)]
    pub high_value: HighValueConfig,
    /// P15 FIX: Domain vocabulary for text processing
    #[serde(default)]
    pub vocabulary: VocabularyConfig,
    /// P16 FIX: Phonetic ASR error correction rules (loaded from domain.yaml)
    #[serde(default)]
    pub phonetic_corrections: PhoneticCorrectionsConfig,
    /// P16 FIX: Domain-specific terms for LLM relevance scoring in speculative execution
    /// Used by SpeculativeExecutor.estimate_domain_relevance()
    #[serde(default)]
    pub relevance_terms: Vec<String>,
    /// P17 FIX: Query expansion configuration for RAG
    #[serde(default)]
    pub query_expansion: QueryExpansionConfig,
    /// P17 FIX: Domain boosting configuration for RAG
    #[serde(default)]
    pub domain_boost: DomainBoostConfig,
    /// Slot definitions for DST (loaded from slots.yaml)
    #[serde(skip)]
    pub slots: SlotsConfig,
    /// Stage definitions for conversation flow (loaded from stages.yaml)
    #[serde(skip)]
    pub stages: StagesConfig,
    /// Lead scoring configuration (loaded from scoring.yaml)
    #[serde(skip)]
    pub scoring: ScoringConfig,
    /// Tool schemas for LLM function calling (loaded from tools/schemas.yaml)
    #[serde(skip)]
    pub tools: ToolsConfig,
    /// Prompt templates (loaded from prompts/system.yaml)
    #[serde(skip)]
    pub prompts: PromptsConfig,
    /// Objection handling configuration (loaded from objections.yaml)
    #[serde(skip)]
    pub objections: ObjectionsConfig,
    /// Branch data (loaded from tools/branches.yaml)
    #[serde(skip)]
    pub branches: BranchesConfig,
    /// SMS templates (loaded from tools/sms_templates.yaml)
    #[serde(skip)]
    pub sms_templates: SmsTemplatesConfig,
    /// Extended competitor data (loaded from competitors.yaml)
    #[serde(skip)]
    pub competitors_config: CompetitorsConfig,
    /// Customer segment definitions (loaded from segments.yaml)
    #[serde(skip)]
    pub segments: SegmentsConfig,
    /// Goals configuration (loaded from goals.yaml)
    #[serde(skip)]
    pub goals: GoalsConfig,
    /// Features configuration (loaded from features.yaml)
    #[serde(skip)]
    pub features: FeaturesConfig,
    /// P16 FIX: Document requirements (loaded from tools/documents.yaml)
    #[serde(skip)]
    pub documents: DocumentsConfig,
    /// P16 FIX: Tool response templates (loaded from tools/responses.yaml)
    #[serde(skip)]
    pub tool_responses: ToolResponsesConfig,
    /// Raw JSON for dynamic access
    #[serde(skip)]
    raw_config: Option<JsonValue>,
}

fn default_version() -> String {
    "1.0.0".to_string()
}

impl Default for MasterDomainConfig {
    /// Creates an unconfigured default - should only be used for testing.
    /// In production, always load from YAML config files.
    fn default() -> Self {
        Self {
            domain_id: "unconfigured".to_string(),
            display_name: "Unconfigured Domain".to_string(),
            version: default_version(),
            brand: BrandConfig::default(),
            constants: DomainConstants::default(),
            competitors: HashMap::new(),
            products: HashMap::new(),
            high_value: HighValueConfig::default(),
            vocabulary: VocabularyConfig::default(),
            phonetic_corrections: PhoneticCorrectionsConfig::default(),
            relevance_terms: Vec::new(),
            query_expansion: QueryExpansionConfig::default(),
            domain_boost: DomainBoostConfig::default(),
            slots: SlotsConfig::default(),
            stages: StagesConfig::default(),
            scoring: ScoringConfig::default(),
            tools: ToolsConfig::default(),
            prompts: PromptsConfig::default(),
            objections: ObjectionsConfig::default(),
            branches: BranchesConfig::default(),
            sms_templates: SmsTemplatesConfig::default(),
            competitors_config: CompetitorsConfig::default(),
            segments: SegmentsConfig::default(),
            goals: GoalsConfig::default(),
            features: FeaturesConfig::default(),
            documents: DocumentsConfig::default(),
            tool_responses: ToolResponsesConfig::default(),
            raw_config: None,
        }
    }
}

impl MasterDomainConfig {
    /// Load configuration from a directory structure
    ///
    /// Expects:
    /// - config_dir/base/defaults.yaml (optional)
    /// - config_dir/domains/{domain_id}/domain.yaml
    pub fn load(domain_id: &str, config_dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let config_dir = config_dir.as_ref();

        // 1. Try to load base defaults
        let base_path = config_dir.join("base/defaults.yaml");
        let base_config: Option<JsonValue> = if base_path.exists() {
            let content = std::fs::read_to_string(&base_path)
                .map_err(|e| ConfigError::ParseError(format!("Failed to read base config: {}", e)))?;
            Some(serde_yaml::from_str(&content)
                .map_err(|e| ConfigError::ParseError(format!("Failed to parse base config: {}", e)))?)
        } else {
            tracing::debug!("No base config found at {:?}", base_path);
            None
        };

        // 2. Load domain-specific config
        let domain_path = config_dir.join(format!("domains/{}/domain.yaml", domain_id));
        if !domain_path.exists() {
            return Err(ConfigError::FileNotFound(domain_path.display().to_string()));
        }

        let domain_content = std::fs::read_to_string(&domain_path)
            .map_err(|e| ConfigError::ParseError(format!("Failed to read domain config: {}", e)))?;

        let domain_config: JsonValue = serde_yaml::from_str(&domain_content)
            .map_err(|e| ConfigError::ParseError(format!("Failed to parse domain config: {}", e)))?;

        // 3. Merge configs (domain overrides base)
        let merged = if let Some(base) = base_config {
            merge_json(base, domain_config)
        } else {
            domain_config
        };

        // 4. Parse into typed config
        let mut config: MasterDomainConfig = serde_json::from_value(merged.clone())
            .map_err(|e| ConfigError::ParseError(format!("Failed to parse merged config: {}", e)))?;

        config.raw_config = Some(merged);

        // 5. Load slots configuration (optional)
        let slots_path = config_dir.join(format!("domains/{}/slots.yaml", domain_id));
        if slots_path.exists() {
            match SlotsConfig::load(&slots_path) {
                Ok(slots) => {
                    tracing::info!(
                        slots_count = slots.slots.len(),
                        goals_count = slots.goals.len(),
                        "Loaded slots configuration"
                    );
                    config.slots = slots;
                }
                Err(e) => {
                    tracing::warn!("Failed to load slots config: {}", e);
                }
            }
        } else {
            tracing::debug!("No slots config found at {:?}", slots_path);
        }

        // 6. Load stages configuration (optional)
        let stages_path = config_dir.join(format!("domains/{}/stages.yaml", domain_id));
        if stages_path.exists() {
            match StagesConfig::load(&stages_path) {
                Ok(stages) => {
                    tracing::info!(
                        stages_count = stages.stages.len(),
                        initial_stage = %stages.initial_stage,
                        "Loaded stages configuration"
                    );
                    config.stages = stages;
                }
                Err(e) => {
                    tracing::warn!("Failed to load stages config: {}", e);
                }
            }
        } else {
            tracing::debug!("No stages config found at {:?}", stages_path);
        }

        // 7. Load scoring configuration (optional)
        let scoring_path = config_dir.join(format!("domains/{}/scoring.yaml", domain_id));
        if scoring_path.exists() {
            match ScoringConfig::load(&scoring_path) {
                Ok(scoring) => {
                    tracing::info!(
                        high_value_threshold = %scoring.escalation.high_value_threshold,
                        "Loaded scoring configuration"
                    );
                    config.scoring = scoring;
                }
                Err(e) => {
                    tracing::warn!("Failed to load scoring config: {}", e);
                }
            }
        } else {
            tracing::debug!("No scoring config found at {:?}", scoring_path);
        }

        // 8. Load tools configuration (optional)
        let tools_path = config_dir.join(format!("domains/{}/tools/schemas.yaml", domain_id));
        if tools_path.exists() {
            match ToolsConfig::load(&tools_path) {
                Ok(tools) => {
                    tracing::info!(
                        tools_count = tools.tools.len(),
                        "Loaded tools configuration"
                    );
                    config.tools = tools;
                }
                Err(e) => {
                    tracing::warn!("Failed to load tools config: {}", e);
                }
            }
        } else {
            tracing::debug!("No tools config found at {:?}", tools_path);
        }

        // 8b. P16 FIX: Load intent-to-tool mappings (optional, merges into tools config)
        let mappings_path = config_dir.join(format!("domains/{}/intent_tool_mappings.yaml", domain_id));
        if mappings_path.exists() {
            match super::tools::IntentToolMappingsConfig::load(&mappings_path) {
                Ok(mappings) => {
                    // Expand aliases and merge into tools.intent_to_tool
                    let expanded = mappings.expand_aliases();
                    tracing::info!(
                        intent_mappings = expanded.len(),
                        slot_aliases = mappings.slot_aliases.len(),
                        tool_defaults = mappings.tool_defaults.len(),
                        argument_mappings = mappings.argument_mappings.len(),
                        "Loaded intent-to-tool mappings"
                    );
                    // Merge expanded mappings into tools config
                    for (intent, mapping) in expanded {
                        config.tools.intent_to_tool.insert(intent, mapping);
                    }
                    // P16 FIX: Merge slot_aliases, tool_defaults, and argument_mappings
                    for (alias, canonical) in mappings.slot_aliases {
                        config.tools.slot_aliases.insert(alias, canonical);
                    }
                    for (tool, defaults) in mappings.tool_defaults {
                        config.tools.tool_defaults.insert(tool, defaults);
                    }
                    for (tool, arg_mapping) in mappings.argument_mappings {
                        config.tools.argument_mappings.insert(tool, arg_mapping);
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to load intent-to-tool mappings: {}", e);
                }
            }
        } else {
            tracing::debug!("No intent-to-tool mappings found at {:?}", mappings_path);
        }

        // 9. Load prompts configuration (optional)
        let prompts_path = config_dir.join(format!("domains/{}/prompts/system.yaml", domain_id));
        if prompts_path.exists() {
            match PromptsConfig::load(&prompts_path) {
                Ok(prompts) => {
                    tracing::info!(
                        has_system_prompt = !prompts.system_prompt.is_empty(),
                        "Loaded prompts configuration"
                    );
                    config.prompts = prompts;
                }
                Err(e) => {
                    tracing::warn!("Failed to load prompts config: {}", e);
                }
            }
        } else {
            tracing::debug!("No prompts config found at {:?}", prompts_path);
        }

        // 10. Load objections configuration (optional)
        let objections_path = config_dir.join(format!("domains/{}/objections.yaml", domain_id));
        if objections_path.exists() {
            match ObjectionsConfig::load(&objections_path) {
                Ok(objections) => {
                    tracing::info!(
                        objection_types = objections.objections.len(),
                        "Loaded objections configuration"
                    );
                    config.objections = objections;
                }
                Err(e) => {
                    tracing::warn!("Failed to load objections config: {}", e);
                }
            }
        } else {
            tracing::debug!("No objections config found at {:?}", objections_path);
        }

        // 11. Load branches configuration (optional)
        let branches_path = config_dir.join(format!("domains/{}/tools/branches.yaml", domain_id));
        if branches_path.exists() {
            match BranchesConfig::load(&branches_path) {
                Ok(branches) => {
                    tracing::info!(
                        branches_count = branches.branches.len(),
                        "Loaded branches configuration"
                    );
                    config.branches = branches;
                }
                Err(e) => {
                    tracing::warn!("Failed to load branches config: {}", e);
                }
            }
        } else {
            tracing::debug!("No branches config found at {:?}", branches_path);
        }

        // 12. Load SMS templates configuration (optional)
        let sms_path = config_dir.join(format!("domains/{}/tools/sms_templates.yaml", domain_id));
        if sms_path.exists() {
            match SmsTemplatesConfig::load(&sms_path) {
                Ok(sms) => {
                    tracing::info!(
                        template_types = sms.templates.len(),
                        "Loaded SMS templates configuration"
                    );
                    config.sms_templates = sms;
                }
                Err(e) => {
                    tracing::warn!("Failed to load SMS templates config: {}", e);
                }
            }
        } else {
            tracing::debug!("No SMS templates config found at {:?}", sms_path);
        }

        // 13. Load competitors configuration (optional)
        let competitors_path = config_dir.join(format!("domains/{}/competitors.yaml", domain_id));
        if competitors_path.exists() {
            match CompetitorsConfig::load(&competitors_path) {
                Ok(competitors) => {
                    tracing::info!(
                        competitors_count = competitors.competitors.len(),
                        "Loaded competitors configuration"
                    );
                    config.competitors_config = competitors;
                }
                Err(e) => {
                    tracing::warn!("Failed to load competitors config: {}", e);
                }
            }
        } else {
            tracing::debug!("No competitors config found at {:?}", competitors_path);
        }

        // 14. Load segments configuration (optional)
        let segments_path = config_dir.join(format!("domains/{}/segments.yaml", domain_id));
        if segments_path.exists() {
            match SegmentsConfig::load(&segments_path) {
                Ok(segments) => {
                    tracing::info!(
                        segments_count = segments.segments.len(),
                        "Loaded segments configuration"
                    );
                    config.segments = segments;
                }
                Err(e) => {
                    tracing::warn!("Failed to load segments config: {}", e);
                }
            }
        } else {
            tracing::debug!("No segments config found at {:?}", segments_path);
        }

        // 15. Load goals configuration (optional)
        let goals_path = config_dir.join(format!("domains/{}/goals.yaml", domain_id));
        if goals_path.exists() {
            match GoalsConfig::load(&goals_path) {
                Ok(goals) => {
                    tracing::info!(
                        goals_count = goals.goals.len(),
                        intent_mappings_count = goals.intent_mappings.len(),
                        "Loaded goals configuration"
                    );
                    config.goals = goals;
                }
                Err(e) => {
                    tracing::warn!("Failed to load goals config: {}", e);
                }
            }
        } else {
            tracing::debug!("No goals config found at {:?}", goals_path);
        }

        // 16. Load features configuration (optional)
        let features_path = config_dir.join(format!("domains/{}/features.yaml", domain_id));
        if features_path.exists() {
            let content = std::fs::read_to_string(&features_path)
                .map_err(|e| ConfigError::ParseError(format!("Failed to read features config: {}", e)))?;
            match serde_yaml::from_str::<FeaturesConfig>(&content) {
                Ok(features) => {
                    tracing::info!(
                        features_count = features.features.len(),
                        segments_with_features = features.segment_features.len(),
                        "Loaded features configuration"
                    );
                    config.features = features;
                }
                Err(e) => {
                    tracing::warn!("Failed to parse features config: {}", e);
                }
            }
        } else {
            tracing::debug!("No features config found at {:?}", features_path);
        }

        // 17. Load documents configuration (optional)
        let documents_path = config_dir.join(format!("domains/{}/tools/documents.yaml", domain_id));
        if documents_path.exists() {
            match DocumentsConfig::load(&documents_path) {
                Ok(documents) => {
                    tracing::info!(
                        service_types = documents.service_types.len(),
                        customer_types = documents.customer_types.len(),
                        mandatory_docs = documents.mandatory_documents.len(),
                        "Loaded documents configuration"
                    );
                    config.documents = documents;
                }
                Err(e) => {
                    tracing::warn!("Failed to load documents config: {}", e);
                }
            }
        } else {
            tracing::debug!("No documents config found at {:?}", documents_path);
        }

        // 18. P16 FIX: Load tool response templates (optional)
        let responses_path = config_dir.join(format!("domains/{}/tools/responses.yaml", domain_id));
        if responses_path.exists() {
            match ToolResponsesConfig::load(&responses_path) {
                Ok(responses) => {
                    tracing::info!(
                        tools_with_templates = responses.templates.len(),
                        "Loaded tool response templates"
                    );
                    config.tool_responses = responses;
                }
                Err(e) => {
                    tracing::warn!("Failed to load tool response templates: {}", e);
                }
            }
        } else {
            tracing::debug!("No tool response templates found at {:?}", responses_path);
        }

        tracing::info!(
            domain_id = %config.domain_id,
            display_name = %config.display_name,
            "Loaded domain configuration"
        );

        Ok(config)
    }

    /// Load from environment variable DOMAIN_ID (required, no default)
    ///
    /// P16 FIX: DOMAIN_ID is now REQUIRED - this is a domain-agnostic system.
    /// Returns an error if DOMAIN_ID is not set.
    pub fn load_from_env(config_dir: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let domain_id = std::env::var("DOMAIN_ID")
            .map_err(|_| ConfigError::MissingField(
                "DOMAIN_ID environment variable is not set. \
                 This is a domain-agnostic system - you MUST specify which domain to use. \
                 Set DOMAIN_ID to the name of the domain config directory \
                 (e.g., DOMAIN_ID=gold_loan for config/domains/gold_loan/).".to_string()
            ))?;

        if domain_id.is_empty() {
            return Err(ConfigError::MissingField(
                "DOMAIN_ID environment variable is empty. Please specify a domain ID.".to_string()
            ));
        }

        Self::load(&domain_id, config_dir)
    }

    /// Get a constant value by dot-notation key path
    /// e.g., "interest_rates.base_rate" or "ltv_percent"
    pub fn get_constant(&self, key_path: &str) -> Option<JsonValue> {
        let raw = self.raw_config.as_ref()?;
        let constants = raw.get("constants")?;

        let parts: Vec<&str> = key_path.split('.').collect();
        let mut current = constants;

        for part in parts {
            current = current.get(part)?;
        }

        Some(current.clone())
    }

    /// Get the best interest rate for a given loan amount
    pub fn get_rate_for_amount(&self, amount: f64) -> f64 {
        for tier in &self.constants.interest_rates.tiers {
            if let Some(max) = tier.max_amount {
                if amount <= max {
                    return tier.rate;
                }
            } else {
                // No max = this is the rate for amounts above all thresholds
                return tier.rate;
            }
        }
        // Fallback to base rate
        self.constants.interest_rates.base_rate
    }

    /// Check if this is a high-value customer
    pub fn is_high_value(&self, amount: Option<f64>, weight_grams: Option<f64>) -> bool {
        if let Some(amt) = amount {
            if amt >= self.high_value.amount_threshold {
                return true;
            }
        }
        if let Some(wt) = weight_grams {
            if wt >= self.high_value.weight_threshold_grams {
                return true;
            }
        }
        false
    }

    /// Get competitor by name or alias
    pub fn get_competitor(&self, name: &str) -> Option<&CompetitorEntry> {
        let name_lower = name.to_lowercase();

        // Direct match
        if let Some(comp) = self.competitors.get(&name_lower) {
            return Some(comp);
        }

        // Search aliases
        for (_, comp) in &self.competitors {
            if comp.aliases.iter().any(|a| a.to_lowercase() == name_lower) {
                return Some(comp);
            }
        }

        None
    }
}

/// Deep merge two JSON values (right overrides left)
fn merge_json(left: JsonValue, right: JsonValue) -> JsonValue {
    match (left, right) {
        (JsonValue::Object(mut left_map), JsonValue::Object(right_map)) => {
            for (key, right_val) in right_map {
                let merged_val = if let Some(left_val) = left_map.remove(&key) {
                    merge_json(left_val, right_val)
                } else {
                    right_val
                };
                left_map.insert(key, merged_val);
            }
            JsonValue::Object(left_map)
        }
        (_, right) => right, // Right value wins for non-objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MasterDomainConfig::default();
        // Default is now "unconfigured" to enforce explicit domain configuration
        assert_eq!(config.domain_id, "unconfigured");
        assert_eq!(config.display_name, "Unconfigured Domain");
    }

    #[test]
    fn test_merge_json() {
        let base = serde_json::json!({
            "a": 1,
            "b": { "c": 2, "d": 3 }
        });
        let overlay = serde_json::json!({
            "a": 10,
            "b": { "c": 20 },
            "e": 5
        });
        let merged = merge_json(base, overlay);

        assert_eq!(merged["a"], 10);
        assert_eq!(merged["b"]["c"], 20);
        assert_eq!(merged["b"]["d"], 3);
        assert_eq!(merged["e"], 5);
    }
}
