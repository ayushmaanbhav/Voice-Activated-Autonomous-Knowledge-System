# Voice Agent Backend - Unified Refactoring & Domain Abstraction Plan

**Date:** 2026-01-06
**Scope:** `voice-agent/backend` (11 crates, 141 Rust files)
**Replaces:** `BACKEND_REFACTORING_PLAN.md` (will merge into this document)

---

## Execution Order

| Order | Phase | Source Plan | Description |
|-------|-------|-------------|-------------|
| 1 | P0-Perf | Backend Refactoring | Fix STT mutex, regex compilation |
| 2 | P0-Config | Backend Refactoring | Service URLs, model paths → env/config |
| 3 | P1-Infra | **This Plan** | Create hierarchical config infrastructure |
| 4 | P1-SRP | Backend Refactoring | Split large files (agent.rs, gold_loan.rs) |
| 5 | P2-P5 | **This Plan** | Domain abstraction (crate-by-crate) |
| 6 | P2-Factory | Backend Refactoring | VAD/STT/TTS factories |
| 7 | P3-Cleanup | Backend Refactoring | Type consolidation, traits |

**Rationale:** P0-Config creates the config loading infrastructure that domain abstraction builds upon. SRP splitting (P1) should happen after infrastructure but before deep domain work.

---

## Core Design Principles

### 1. Hierarchical Config with Trickle-Down

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 0: ENVIRONMENT (.env / secrets)                       │
│ - API keys (ANTHROPIC_API_KEY, QDRANT_API_KEY)             │
│ - Service URLs (STT_URL, TTS_URL, SCYLLA_HOSTS)            │
│ - Feature flags (USE_CONFIG_DRIVEN=true)                   │
│ - Owned by: Deployment / Ops                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: BASE CONFIG (config/base/*.yaml)                   │
│ - Domain-agnostic defaults (timeouts, buffer sizes)        │
│ - Pipeline defaults (VAD thresholds, sample rates)         │
│ - Owned by: Platform team                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: DOMAIN CONFIG (config/domains/{domain}/*.yaml)     │
│ - Business constants (rates, LTV, limits)                  │
│ - Stages, slots, prompts, competitors                      │
│ - Owned by: Product/Domain team                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: RUNTIME OVERRIDES (per-request or session)         │
│ - User language preference                                 │
│ - A/B test variants                                        │
│ - Customer segment overrides                               │
│ - Owned by: Runtime / Session context                      │
└─────────────────────────────────────────────────────────────┘
```

### 2. Crate Ownership & Security Boundaries

| Crate | Owns | Env Vars It Reads | Config It Receives | Never Sees |
|-------|------|-------------------|-------------------|------------|
| `config` | MasterDomainConfig, env loading | All (validates at startup) | Base + Domain YAML | N/A |
| `server` | HTTP/WS setup, request routing | `PORT`, `HOST`, `TLS_*` | ServerConfig | Domain business logic |
| `llm` | LLM clients, prompt building | `ANTHROPIC_API_KEY`, `OLLAMA_URL` | LlmDomainView | DST slots, scoring |
| `rag` | Embeddings, vector search | `QDRANT_URL`, `QDRANT_API_KEY` | RagConfig | Objection responses |
| `pipeline` | STT/TTS/VAD orchestration | `STT_URL`, `TTS_URL` | PipelineConfig | Lead scoring |
| `tools` | Tool implementations | (none - gets from config) | ToolsDomainView | Conversation stages |
| `agent` | Conversation orchestration | (none - gets from views) | AgentDomainView | Service URLs |
| `core` | Generic traits only | (none) | Generic patterns only | Any domain terminology |
| `persistence` | ScyllaDB client | `SCYLLA_HOSTS`, `SCYLLA_KEYSPACE` | PersistenceConfig | Business logic |

### 3. Traits Driving Factory Patterns

```rust
// crates/core/src/traits/mod.rs

/// All pluggable components implement these traits
pub trait SttProvider: Send + Sync {
    async fn transcribe(&self, audio: &[i16]) -> Result<String>;
}

pub trait TtsProvider: Send + Sync {
    async fn synthesize(&self, text: &str, voice: &VoiceConfig) -> Result<Vec<i16>>;
}

pub trait VadProvider: Send + Sync {
    fn process_frame(&mut self, frame: &[i16]) -> VadResult;
}

pub trait EmbeddingProvider: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

pub trait LlmProvider: Send + Sync {
    async fn generate(&self, messages: &[Message], tools: &[Tool]) -> Result<Response>;
}
```

```rust
// crates/config/src/factories.rs

/// Runtime config drives which implementation gets created
pub struct FactoryConfig {
    pub stt: SttBackend,      // Indicconformer | Whisper | External
    pub tts: TtsBackend,      // IndicF5 | Piper | External
    pub vad: VadBackend,      // Silero | WebRTC
    pub llm: LlmBackend,      // Claude | Ollama | OpenAI
    pub embedding: EmbeddingBackend,  // Ollama | OpenAI | Local
}

pub enum SttBackend {
    Indicconformer { model_path: PathBuf },
    Whisper { model_size: String },
    External { url: String, api_key: Option<String> },
}
```

### 4. Env Module Pattern

```rust
// crates/config/src/env.rs

use std::env;

/// Centralized environment variable handling with validation
pub struct EnvConfig {
    // Security: API keys (required for production)
    pub anthropic_api_key: Secret<String>,
    pub qdrant_api_key: Option<Secret<String>>,

    // Service URLs (with defaults for dev)
    pub stt_url: String,
    pub tts_url: String,
    pub scylla_hosts: Vec<String>,

    // Feature flags
    pub use_config_driven: bool,
    pub domain_id: String,
}

impl EnvConfig {
    pub fn load() -> Result<Self, EnvError> {
        Ok(Self {
            anthropic_api_key: Secret::new(
                env::var("ANTHROPIC_API_KEY")
                    .map_err(|_| EnvError::MissingRequired("ANTHROPIC_API_KEY"))?
            ),
            stt_url: env::var("STT_URL").unwrap_or_else(|_| "http://127.0.0.1:8091".into()),
            domain_id: env::var("DOMAIN_ID").unwrap_or_else(|_| "gold_loan".into()),
            use_config_driven: env::var("USE_CONFIG_DRIVEN")
                .map(|v| v == "true")
                .unwrap_or(false),
            // ...
        })
    }
}
```

### 5. Runtime Config (Language, A/B Tests, Segments)

```rust
// crates/core/src/runtime.rs

/// Per-session runtime context (can override domain defaults)
pub struct RuntimeContext {
    pub session_id: String,
    pub user_language: Language,           // en | hi | hinglish
    pub ab_test_variants: HashMap<String, String>,
    pub customer_segment: Option<SegmentId>,
}

impl RuntimeContext {
    /// Get config value with runtime override
    pub fn get_prompt_language(&self) -> Language {
        // Runtime language overrides domain default
        self.user_language
    }

    /// Check if A/B variant is active
    pub fn is_variant_active(&self, test_name: &str, variant: &str) -> bool {
        self.ab_test_variants.get(test_name) == Some(&variant.to_string())
    }
}
```

**Language switching flow:**
```
1. Session starts → detect language from first utterance
2. RuntimeContext.user_language = detected
3. All prompts/responses use runtime language
4. Domain config provides templates for each language
```

---

## Phase 0: Performance & Basic Config (From Backend Plan)

### P0-Perf: Critical Performance Fixes

| Issue | Location | Fix |
|-------|----------|-----|
| Global STT mutex | `ptt.rs:36-37` | Create STT connection pool |
| Regex per-request | `ptt.rs:169-222` | Use `lazy_static!` + `OnceCell` |
| Lock across await | `websocket.rs:262` | Release lock before `.await` |

### P0-Config: Move Hardcoded Values to Config

| Value | Current Location | Target |
|-------|-----------------|--------|
| STT URL `127.0.0.1:8091` | `ptt.rs:30` | `STT_URL` env var |
| TTS URL `127.0.0.1:8092` | `ptt.rs:33` | `TTS_URL` env var |
| ScyllaDB `127.0.0.1:9042` | `persistence/client.rs:19` | `SCYLLA_HOSTS` env var |
| Ollama `localhost:11434` | `ollama_embeddings.rs:38` | `OLLAMA_URL` env var |
| Model paths | `ptt.rs`, `orchestrator.rs` | `config/base/pipeline.yaml` |

---

## Problem Statement

~40% of the core agent code is gold-loan specific, tightly coupling the backend to a single domain. This prevents:
- Multi-domain deployment (personal loans, insurance, credit cards)
- White-labeling for different banks
- A/B testing domain configurations

## Current State Analysis

### Domain-Specific Code Locations (by crate)

| Crate | File | Lines | Domain Coupling Type |
|-------|------|-------|---------------------|
| agent | agent.rs | 3,070 | GoldLoanAgent name, prompt building |
| agent | stage.rs | 712 | Hardcoded ConversationStage enum |
| agent | persuasion.rs | 748 | Objection scripts with rates (9.5%, 18-20%) |
| agent | lead_scoring.rs | 556 | Thresholds (1M high-value), scoring weights |
| agent | dst/slots.rs | 399 | Gold-specific slots (purity, weight) |
| agent | dst/extractor.rs | 600 | Lender patterns (Muthoot, Manappuram, etc.) |
| config | constants.rs | 75 | Interest rates, LTV, gold prices |
| config | competitor.rs | 550 | 6 competitor profiles with rates |
| config | prompts.rs | 879 | Kotak-branded prompts, SMS templates |
| config | product.rs | 560 | 4 product variants (Shakti Gold, Bullet) |
| llm | prompt.rs | 903 | Tool definitions, system prompts |
| tools | gold_loan.rs | 2,230 | 10 tools, hardcoded branches |
| core | customer.rs | 560 | Segment detection patterns |
| core | domain_context.rs | 127 | Hardcoded vocabulary, competitors |
| core | personalization/ | 420 | Features, objection responses |

### Existing Infrastructure (Leverage Points)

1. **DomainConfig** in `config/src/domain.rs` - Already has YAML/JSON loading with hot-reload
2. **ToolRegistry** in `tools/src/registry.rs` - Dynamic tool registration exists
3. **PromptBuilder** in `llm/src/prompt.rs` - Template system available

---

## Target Architecture

### Design Principles

1. **Hierarchical Inheritance**: Master config → Domain config → Crate-specific views
2. **Ownership Boundaries**: Each crate owns its config interpretation within its domain
3. **Trickle-Down Translation**: Raw YAML → Typed master → Crate-specific structs
4. **Awareness Isolation**: Crates only see what they need (no gold loan terms in core)

### Layered Config Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LAYER 1: RAW YAML FILES                      │
│  config/domains/gold_loan/*.yaml (human-editable, version-controlled) │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 2: MASTER DOMAIN CONFIG                   │
│  crates/config/src/domain/ → MasterDomainConfig (typed, validated) │
│  - Owns: Raw parsing, validation, hot-reload                     │
│  - Exposes: Full domain knowledge to downstream crates           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│ LAYER 3a: AGENT     │ │ LAYER 3b: LLM   │ │ LAYER 3c: TOOLS     │
│ AgentDomainView     │ │ LlmDomainView   │ │ ToolsDomainView     │
│ - Stages, Scoring   │ │ - Prompts, Tools│ │ - Tool configs      │
│ - DST slots         │ │ - System prompt │ │ - Branch data       │
│ - Persuasion        │ │                 │ │ - SMS templates     │
└─────────────────────┘ └─────────────────┘ └─────────────────────┘
         │                      │                    │
         ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LAYER 4: CORE (Domain-Agnostic)                  │
│  crates/core/ → Generic traits only (no gold loan awareness)    │
│  - Segment trait (not "HighValue gold customer")                │
│  - Objection trait (not "GoldSecurity objection")              │
└─────────────────────────────────────────────────────────────────┘
```

### Config Directory Structure

```
config/
  base/                       # Domain-agnostic defaults (inherited)
    defaults.yaml             # Fallback values for all domains
    compliance.yaml           # Regulatory rules (RBI, etc.)

  domains/
    gold_loan/
      domain.yaml             # Master: identity, constants, brand
      stages.yaml             # Agent: conversation flow graph
      slots.yaml              # Agent: DST slot definitions
      scoring.yaml            # Agent: lead scoring weights
      objections.yaml         # Agent: objection patterns + responses
      segments.yaml           # Core: customer segmentation rules

      competitors.yaml        # Config: competitor registry
      products.yaml           # Config: product variants

      tools/
        schemas.yaml          # LLM: tool JSON schemas for Claude
        branches.yaml         # Tools: branch data (can be DB later)
        sms_templates.yaml    # Tools: SMS message templates

      prompts/
        system.txt            # LLM: main system prompt template
        stages/               # LLM: per-stage guidance
          greeting.txt
          discovery.txt
          ...
        responses/            # Agent: response templates
          objection_*.txt

    personal_loan/            # Future domain (same structure)
    insurance/                # Future domain
```

### Crate Ownership & Boundaries

| Crate | Owns | Sees | Never Sees |
|-------|------|------|------------|
| `config` | MasterDomainConfig, parsing, validation | All YAML files | N/A (owns everything) |
| `agent` | AgentDomainView: stages, DST, scoring, persuasion | Stages, slots, scoring, objections | Branch data, SMS templates |
| `llm` | LlmDomainView: prompts, tool schemas | System prompt, tool definitions | Lead scoring weights |
| `tools` | ToolsDomainView: tool configs, branches, SMS | Tool schemas, branch data, templates | Conversation stages |
| `core` | Generic traits only | Segment rules (as generic patterns) | Any domain-specific terminology |

### Core Traits (Domain-Agnostic)

```rust
// crates/core/src/domain/traits.rs

/// Generic customer segment (no gold loan awareness)
pub trait CustomerSegment: Send + Sync {
    fn segment_id(&self) -> &str;
    fn matches(&self, signals: &CustomerSignals) -> bool;
    fn priority_features(&self) -> &[FeatureId];
}

/// Generic objection handler (no gold loan awareness)
pub trait ObjectionHandler: Send + Sync {
    fn objection_id(&self) -> &str;
    fn patterns(&self) -> &[Pattern];
    fn response_template(&self) -> &ResponseTemplate;
}

/// Generic conversation stage (no gold loan awareness)
pub trait ConversationStage: Send + Sync {
    fn stage_id(&self) -> &str;
    fn guidance(&self) -> &str;
    fn allowed_transitions(&self) -> &[StageId];
    fn context_budget(&self) -> usize;
}
```

### Crate-Specific View Traits

```rust
// crates/config/src/domain/views.rs

/// What the agent crate needs (translated from master)
pub trait AgentDomainView: Send + Sync {
    fn stages(&self) -> &StageGraph;
    fn slots(&self) -> &SlotSchema;
    fn scoring_config(&self) -> &ScoringConfig;
    fn objection_library(&self) -> &ObjectionLibrary;
    fn segment_rules(&self) -> &SegmentRules;
}

/// What the llm crate needs (translated from master)
pub trait LlmDomainView: Send + Sync {
    fn system_prompt_template(&self) -> &str;
    fn stage_prompts(&self) -> &HashMap<StageId, String>;
    fn tool_schemas(&self) -> &[ToolSchema];
    fn brand(&self) -> &BrandConfig;
}

/// What the tools crate needs (translated from master)
pub trait ToolsDomainView: Send + Sync {
    fn tool_configs(&self) -> &HashMap<ToolId, ToolConfig>;
    fn branches(&self) -> &[Branch];
    fn sms_templates(&self) -> &HashMap<SmsType, String>;
    fn constants(&self) -> &ToolConstants;  // rates, LTV, etc.
}
```

### Master Config with Trickle-Down

```rust
// crates/config/src/domain/master.rs

pub struct MasterDomainConfig {
    // Identity
    pub domain_id: String,
    pub display_name: String,
    pub brand: BrandConfig,

    // Business constants (source of truth)
    pub constants: DomainConstants,

    // All sub-configs (raw, not yet translated)
    pub stages: StagesConfig,
    pub slots: SlotsConfig,
    pub scoring: ScoringConfig,
    pub objections: ObjectionsConfig,
    pub segments: SegmentsConfig,
    pub competitors: CompetitorsConfig,
    pub products: ProductsConfig,
    pub tools: ToolsConfig,
    pub prompts: PromptsConfig,
}

impl MasterDomainConfig {
    /// Load from YAML directory with inheritance from base/
    pub fn load(domain_id: &str, config_dir: &Path) -> Result<Self>;

    /// Create agent-specific view (translates to agent's terms)
    pub fn agent_view(&self) -> impl AgentDomainView;

    /// Create llm-specific view (translates to llm's terms)
    pub fn llm_view(&self) -> impl LlmDomainView;

    /// Create tools-specific view (translates to tools' terms)
    pub fn tools_view(&self) -> impl ToolsDomainView;
}
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (P1)

#### 1.1 Create Layered Domain Traits in Core
**Files to create:**
- `crates/core/src/domain/mod.rs`
- `crates/core/src/domain/traits.rs` - Generic traits (no domain awareness)

**Core stays domain-agnostic:**
```rust
// crates/core/src/domain/traits.rs

/// Generic identifiers (core never knows "gold_loan")
pub type StageId = String;
pub type SegmentId = String;
pub type ObjectionId = String;
pub type FeatureId = String;

/// Generic customer signals (no gold-specific fields)
pub struct CustomerSignals {
    pub numeric_values: HashMap<String, f64>,   // "gold_weight" is just a string key
    pub text_values: HashMap<String, String>,   // "current_lender" is just a string key
    pub flags: HashSet<String>,                 // "has_urgency" is just a flag name
}

/// Generic pattern for matching (regex or keyword)
pub struct Pattern {
    pub pattern_type: PatternType,
    pub value: String,
    pub language: Option<String>,  // "en", "hi", etc.
}
```

#### 1.2 Create Master Config System in Config Crate
**Files to create:**
- `crates/config/src/domain/mod.rs`
- `crates/config/src/domain/master.rs` - MasterDomainConfig
- `crates/config/src/domain/loader.rs` - YAML loading with inheritance
- `crates/config/src/domain/views.rs` - View traits for downstream crates

**Inheritance logic:**
```rust
// crates/config/src/domain/loader.rs

impl MasterDomainConfig {
    pub fn load(domain_id: &str, config_dir: &Path) -> Result<Self> {
        // 1. Load base/defaults.yaml
        let base = load_yaml(config_dir.join("base/defaults.yaml"))?;

        // 2. Load domain-specific config
        let domain_dir = config_dir.join(format!("domains/{}", domain_id));
        let domain = load_yaml(domain_dir.join("domain.yaml"))?;

        // 3. Merge: domain overrides base
        let merged = merge_configs(base, domain);

        // 4. Load sub-configs (stages, slots, etc.)
        let stages = load_yaml(domain_dir.join("stages.yaml"))?;
        // ...

        Ok(Self { merged, stages, ... })
    }
}
```

#### 1.3 Create View Implementations
**Files to create:**
- `crates/config/src/domain/agent_view.rs`
- `crates/config/src/domain/llm_view.rs`
- `crates/config/src/domain/tools_view.rs`

**Each view translates master config to crate-specific structs:**
```rust
// crates/config/src/domain/agent_view.rs

pub struct AgentView<'a> {
    master: &'a MasterDomainConfig,
}

impl AgentDomainView for AgentView<'_> {
    fn stages(&self) -> &StageGraph {
        // Translate YAML stages to agent's StageGraph struct
        &self.master.stages_graph
    }

    fn scoring_config(&self) -> &ScoringConfig {
        // Translate constants + scoring.yaml to ScoringConfig
        ScoringConfig {
            high_value_threshold: self.master.constants.loan_limits.high_value,
            // ...
        }
    }
}
```

#### 1.4 Create Base Config Files
**Files to create:**
```
voice-agent/backend/config/
  base/
    defaults.yaml           # Fallback values
  domains/
    gold_loan/
      domain.yaml           # Identity + constants
```

**domain.yaml:**
```yaml
domain_id: gold_loan
display_name: "Kotak Gold Loan"

brand:
  bank_name: "Kotak Mahindra Bank"
  agent_name: "Priya"
  helpline: "1800-xxx-xxxx"

constants:
  interest_rates:
    tiers:
      - { max_amount: 100000, rate: 11.5 }
      - { max_amount: 500000, rate: 10.5 }
      - { max_amount: null, rate: 9.5 }    # null = unlimited
  ltv_percent: 75.0
  loan_limits:
    min: 10000
    max: 25000000
  processing_fee_percent: 1.0
```

### Phase 2: Agent Crate Abstraction (P2)

#### 2.1 Make DST Slots Config-Driven
**Files to modify:**
- `crates/agent/src/dst/slots.rs` - Replace hardcoded GoldPurity enum
- `crates/agent/src/dst/extractor.rs` - Load patterns from config

**Files to create:**
- `config/domains/gold_loan/slots.yaml`

**Agent receives SlotSchema from AgentDomainView:**
```rust
// crates/agent/src/dst/slots.rs

pub struct SlotSchema {
    pub slots: HashMap<String, SlotDefinition>,
    pub goals: HashMap<String, GoalDefinition>,
}

pub struct SlotDefinition {
    pub slot_type: SlotType,  // String, Number, Enum, Date
    pub validation: Option<String>,
    pub unit_conversions: HashMap<String, f64>,
    pub enum_values: Option<Vec<String>>,
}

// Agent doesn't know these are "gold" slots - just string keys
impl DstExtractor {
    pub fn new(schema: SlotSchema) -> Self {
        // Build regex patterns from schema
    }
}
```

**slots.yaml (owned by domain config):**
```yaml
slots:
  gold_weight_grams:
    type: number
    min: 1
    max: 10000
    extraction_patterns:
      - { regex: '(\d+(?:\.\d+)?)\s*(?:grams?|gm|g)', language: en }
      - { regex: '(\d+(?:\.\d+)?)\s*(?:ग्राम)', language: hi }
    unit_conversions:
      tola: 11.66

  gold_purity:
    type: enum
    values: [K24, K22, K18, K14]
    metadata:
      K24: { purity_factor: 1.0 }
      K22: { purity_factor: 0.916 }

goals:
  balance_transfer:
    required_slots: [current_lender, loan_amount]
    optional_slots: [current_interest_rate, gold_weight_grams]
    completion_action: calculate_savings
```

#### 2.2 Make Stages Config-Driven
**Files to modify:**
- `crates/agent/src/stage.rs` - Remove ConversationStage enum

**Files to create:**
- `config/domains/gold_loan/stages.yaml`

**Agent receives StageGraph from AgentDomainView:**
```rust
// crates/agent/src/stage.rs

// REMOVE this enum:
// pub enum ConversationStage { Greeting, Discovery, ... }

// REPLACE with config-driven:
pub struct StageGraph {
    stages: HashMap<StageId, Stage>,
    initial_stage: StageId,
    transitions: HashMap<StageId, Vec<StageId>>,
}

pub struct Stage {
    pub id: StageId,
    pub guidance: String,
    pub questions: Vec<String>,
    pub context_budget: usize,
    pub rag_fraction: f32,
}
```

#### 2.3 Make Lead Scoring Config-Driven
**Files to modify:**
- `crates/agent/src/lead_scoring.rs` - Remove hardcoded thresholds

**Files to create:**
- `config/domains/gold_loan/scoring.yaml`

```yaml
# scoring.yaml
thresholds:
  high_value_amount: 500000
  high_value_weight: 100  # grams

classification:
  sql_criteria:
    - has_urgency AND provided_contact AND has_requirements
  mql_criteria:
    - engagement_turns >= 3 AND (asked_about_rates OR asked_for_comparison)

weights:
  urgency: 25
  engagement: 25
  information: 25
  intent: 25
```

### Phase 3: LLM Crate Abstraction (P3)

#### 3.1 Make Tool Schemas Config-Driven
**Files to modify:**
- `crates/llm/src/prompt.rs` - Remove hardcoded gold_loan_tools()

**LLM receives tool schemas from LlmDomainView:**
```rust
// crates/llm/src/prompt.rs

impl PromptBuilder {
    pub fn new(domain_view: Arc<dyn LlmDomainView>) -> Self {
        Self { domain_view }
    }

    pub fn build_tools(&self) -> Vec<ToolSchema> {
        // Get from config instead of hardcoded
        self.domain_view.tool_schemas().to_vec()
    }
}
```

**Files to create:**
- `config/domains/gold_loan/tools/schemas.yaml`

```yaml
# tools/schemas.yaml
tools:
  check_eligibility:
    description: "Check if customer is eligible for a {domain_name}"
    parameters:
      gold_weight:
        type: number
        description: "Weight of gold in grams"
        range: [1, 10000]
      gold_purity:
        type: string
        enum: [24K, 22K, 18K, 14K]

  calculate_savings:
    description: "Calculate monthly savings when switching lenders"
    parameters:
      current_lender:
        type: string
        description: "Name of current lender"
      current_rate:
        type: number
        range: [0, 50]
```

#### 3.2 Make System Prompts Template-Driven
**Files to modify:**
- `crates/llm/src/prompt.rs` - Load templates
- `crates/config/src/prompts.rs` - Remove hardcoded prompts

**Files to create:**
- `config/domains/gold_loan/prompts/system.txt`
- `config/domains/gold_loan/prompts/stages/*.txt`

**Template with variable interpolation:**
```
# prompts/system.txt
You are {brand.agent_name}, a friendly {domain.display_name} specialist at {brand.bank_name}.

Key facts about our offering:
- Interest rates: Starting from {constants.interest_rates.tiers[-1].rate}% p.a.
- LTV: Up to {constants.ltv_percent}% of gold value
- Loan range: {constants.loan_limits.min|currency} to {constants.loan_limits.max|currency}

{stage_guidance}
```

#### 3.3 Make Objection Responses Config-Driven
**Files to modify:**
- `crates/agent/src/persuasion.rs` - Remove hardcoded responses

**Files to create:**
- `config/domains/gold_loan/objections.yaml`

**Agent receives ObjectionLibrary from AgentDomainView:**
```yaml
# objections.yaml
objections:
  interest_rate:
    patterns:
      en: ["rate.*high", "expensive", "too much"]
      hi: ["ज्यादा", "महंगा", "kam rate"]
    response:
      acknowledge: "I understand rates are important to you."
      reframe: "Our rate of {constants.interest_rates.tiers[-1].rate}% compares favorably."
      evidence: "On a 3 lakh loan, you'd save approximately ₹3,500/month vs NBFCs."
      cta: "Would you like me to calculate your exact savings?"

  gold_security:
    patterns:
      en: ["gold.*safe", "security", "theft"]
      hi: ["सुरक्षा", "suraksha", "chori"]
    response:
      acknowledge: "Your concern about gold safety is completely valid."
      reframe: "We store gold in RBI-regulated bank vaults with 24/7 security."
```

### Phase 4: Tools Crate & Data Externalization (P4)

#### 4.1 Make Tools Config-Driven
**Files to modify:**
- `crates/tools/src/gold_loan.rs` - Remove hardcoded branches, SMS templates

**Tools receives ToolsDomainView:**
```rust
// crates/tools/src/gold_loan.rs

impl BranchLocatorTool {
    pub fn new(domain_view: Arc<dyn ToolsDomainView>) -> Self {
        Self { branches: domain_view.branches().to_vec() }
    }
}

impl SendSmsTool {
    pub fn new(domain_view: Arc<dyn ToolsDomainView>) -> Self {
        Self { templates: domain_view.sms_templates().clone() }
    }
}
```

**Files to create:**
- `config/domains/gold_loan/tools/branches.yaml`
- `config/domains/gold_loan/tools/sms_templates.yaml`

```yaml
# tools/branches.yaml (can be replaced with DB later)
branches:
  - id: KMBL001
    name: "Kotak Mahindra Bank - Andheri West"
    city: Mumbai
    address: "Ground Floor, Kora Kendra, S.V. Road, Andheri West"
    pincode: "400058"
    phone: "022-66006060"
    timing: "10:00 AM - 5:00 PM (Mon-Sat)"
    facilities: [gold_valuation, same_day_disbursement]
```

```yaml
# tools/sms_templates.yaml
templates:
  appointment_confirmation:
    en: "Dear {name}, your {brand.bank_name} {domain.display_name} appointment is confirmed for {date}. Please bring gold and KYC. Call {brand.helpline}."
    hi: "प्रिय {name}, आपकी {brand.bank_name} गोल्ड लोन अपॉइंटमेंट {date} को कन्फर्म है।"

  promotional:
    en: "Special Offer: Get {domain.display_name} at just {constants.interest_rates.tiers[-1].rate}%* p.a. with instant disbursement! T&C apply."
```

#### 4.2 Externalize Competitor Data
**Files to modify:**
- `crates/config/src/competitor.rs` - Load from YAML

**Files to create:**
- `config/domains/gold_loan/competitors.yaml`

```yaml
# competitors.yaml
competitors:
  muthoot:
    display_name: "Muthoot Finance"
    aliases: [muthoot, muthut, "muthoot finance"]
    typical_rate: 18.0
    ltv_percent: 75.0
    type: nbfc
    strengths: ["Large branch network", "Quick processing"]
    weaknesses: ["Higher rates", "Not RBI-regulated bank"]

  manappuram:
    display_name: "Manappuram Finance"
    aliases: [manappuram, manapuram, "manappuram gold"]
    typical_rate: 19.0
    ltv_percent: 75.0
    type: nbfc

comparison_points:
  - category: "Interest Rate"
    ours: "{constants.interest_rates.tiers[-1].rate}%"
    theirs: "12-24%"
    advantage: true
```

#### 4.3 Externalize Segment Detection
**Files to modify:**
- `crates/core/src/customer.rs` - Remove hardcoded patterns

**Files to create:**
- `config/domains/gold_loan/segments.yaml`

**Core receives generic SegmentRules (no gold awareness):**
```yaml
# segments.yaml
segments:
  high_value:
    detection:
      numeric_thresholds:
        gold_weight_grams: { min: 100 }
        loan_amount: { min: 500000 }
      text_patterns:
        en: ["lakh", "crore", "100 gram"]
        hi: ["लाख", "करोड़", "सौ ग्राम"]
    features: [relationship_manager, priority_processing]
    value_props:
      - "Dedicated relationship manager"
      - "Priority processing"

  trust_seeker:
    detection:
      text_values:
        current_lender: [muthoot, manappuram, iifl]
      text_patterns:
        en: ["safe", "security", "rbi"]
        hi: ["सुरक्षा", "भरोसा"]
    features: [rbi_regulated, bank_security]
```

### Phase 5: Final Integration (P5)

#### 5.1 Rename GoldLoanAgent to DomainAgent
**Files to modify:**
- `crates/agent/src/agent.rs` - Rename struct, inject views
- `crates/agent/src/lib.rs` - Update exports
- `crates/server/src/*.rs` - Update imports

```rust
// crates/agent/src/agent.rs

// BEFORE:
pub struct GoldLoanAgent { ... }

// AFTER:
pub struct DomainAgent {
    agent_view: Arc<dyn AgentDomainView>,
    // ... other fields (no gold-specific knowledge)
}

impl DomainAgent {
    pub fn new(
        agent_view: Arc<dyn AgentDomainView>,
        llm: Arc<dyn LlmClient>,
        // ... other deps
    ) -> Self {
        // Agent is domain-agnostic - all domain knowledge comes from view
    }
}
```

#### 5.2 Wire Up Domain Loading at Startup
**Files to modify:**
- `crates/server/src/main.rs` - Load domain config at startup

```rust
// crates/server/src/main.rs

#[tokio::main]
async fn main() -> Result<()> {
    // Load domain config (can switch domain via env var)
    let domain_id = std::env::var("DOMAIN_ID").unwrap_or("gold_loan".into());
    let config_dir = PathBuf::from("config");

    let master_config = MasterDomainConfig::load(&domain_id, &config_dir)?;

    // Create crate-specific views
    let agent_view = Arc::new(master_config.agent_view());
    let llm_view = Arc::new(master_config.llm_view());
    let tools_view = Arc::new(master_config.tools_view());

    // Inject views into components
    let agent = DomainAgent::new(agent_view, llm_client, ...);
    let prompt_builder = PromptBuilder::new(llm_view);
    let tools = ToolRegistry::new(tools_view);

    // ... start server
}
```

#### 5.3 Add Hot-Reload Support
**Files to modify:**
- `crates/config/src/domain/master.rs` - Add watch capability

```rust
impl MasterDomainConfig {
    /// Watch for config changes and reload
    pub fn watch(&self, callback: impl Fn(&Self) + Send + 'static) {
        // Use notify crate to watch config directory
        // On change: reload and call callback
    }
}
```

---

## Migration Strategy

### Step 1: Create config files alongside hardcoded values
- Don't remove any hardcoded code yet
- Add YAML configs that mirror current behavior
- Add feature flag: `use_config_driven_domain`

### Step 2: Add config loading with fallback
```rust
fn get_interest_rate(&self, tier: Tier) -> f64 {
    if self.use_config_driven {
        self.domain.constants().rates.get(tier)
    } else {
        match tier {
            Tier::Standard => 11.5,  // Legacy fallback
            Tier::Premium => 10.5,
        }
    }
}
```

### Step 3: Validate config matches hardcoded
- Add tests comparing config values to hardcoded
- Run in CI to catch drift

### Step 4: Remove hardcoded values
- Once config is validated, remove legacy code
- Remove feature flag

---

## Critical Files Summary

### By Phase

| Phase | Crate | File | Change |
|-------|-------|------|--------|
| P1 | core | `src/domain/traits.rs` | Create generic domain traits (no gold awareness) |
| P1 | config | `src/domain/master.rs` | MasterDomainConfig with inheritance |
| P1 | config | `src/domain/views.rs` | AgentDomainView, LlmDomainView, ToolsDomainView |
| P1 | config | `src/domain/loader.rs` | YAML loading with base/domain merge |
| P2 | agent | `src/stage.rs` | Replace ConversationStage enum with StageGraph |
| P2 | agent | `src/dst/slots.rs` | Replace GoldPurity enum with SlotSchema |
| P2 | agent | `src/dst/extractor.rs` | Load patterns from SlotSchema |
| P2 | agent | `src/lead_scoring.rs` | Load thresholds from ScoringConfig |
| P3 | llm | `src/prompt.rs` | Load tool schemas from LlmDomainView |
| P3 | agent | `src/persuasion.rs` | Load objections from ObjectionLibrary |
| P4 | tools | `src/gold_loan.rs` | Load branches, SMS from ToolsDomainView |
| P4 | config | `src/competitor.rs` | Load from competitors.yaml |
| P4 | core | `src/customer.rs` | Load segment patterns from config |
| P5 | agent | `src/agent.rs` | Rename GoldLoanAgent → DomainAgent |
| P5 | server | `src/main.rs` | Wire up domain loading at startup |

### Complete Config Directory Structure

```
voice-agent/backend/config/
├── .env.example                    # Template for environment variables
│
├── base/                           # LAYER 1: Domain-agnostic defaults
│   ├── server.yaml                 # HTTP/WS server config
│   │   └── port, host, cors, timeouts
│   ├── pipeline.yaml               # STT/TTS/VAD config
│   │   ├── stt:
│   │   │   └── backend: indicconformer | whisper | external
│   │   │   └── model_path: models/stt/indicconformer
│   │   ├── tts:
│   │   │   └── backend: indicf5 | piper | external
│   │   │   └── model_path: models/tts/IndicF5
│   │   └── vad:
│   │       └── backend: silero | webrtc
│   │       └── model_path: models/vad/silero_vad.onnx
│   ├── llm.yaml                    # LLM provider config
│   │   └── provider: claude | ollama | openai
│   │   └── model: claude-sonnet-4-20250514
│   │   └── max_tokens, temperature
│   ├── rag.yaml                    # Vector search config
│   │   └── provider: qdrant | milvus
│   │   └── collection, embedding_dim
│   ├── persistence.yaml            # Database config
│   │   └── driver: scylla | postgres
│   │   └── keyspace, replication
│   └── compliance.yaml             # Regulatory rules (RBI, etc.)
│
├── domains/                        # LAYER 2: Domain-specific config
│   └── gold_loan/
│       ├── domain.yaml             # Master: identity, brand, constants
│       │   ├── domain_id: gold_loan
│       │   ├── display_name: "Kotak Gold Loan"
│       │   ├── brand:
│       │   │   └── bank_name, agent_name, helpline
│       │   └── constants:
│       │       └── interest_rates, ltv, loan_limits
│       │
│       ├── stages.yaml             # Agent: conversation flow graph
│       ├── slots.yaml              # Agent: DST slot definitions
│       ├── scoring.yaml            # Agent: lead scoring weights
│       ├── objections.yaml         # Agent: objection patterns + responses
│       ├── segments.yaml           # Core: customer segmentation rules
│       ├── competitors.yaml        # Config: competitor registry
│       ├── products.yaml           # Config: product variants
│       │
│       ├── tools/
│       │   ├── schemas.yaml        # LLM: tool JSON schemas for Claude
│       │   ├── branches.yaml       # Tools: branch data
│       │   └── sms_templates.yaml  # Tools: SMS message templates
│       │
│       └── prompts/
│           ├── system.txt          # LLM: main system prompt (with {placeholders})
│           ├── stages/             # LLM: per-stage guidance
│           │   ├── greeting.txt
│           │   ├── discovery.txt
│           │   ├── qualification.txt
│           │   ├── presentation.txt
│           │   ├── objection_handling.txt
│           │   ├── closing.txt
│           │   └── farewell.txt
│           └── responses/          # Agent: response templates by language
│               ├── en/
│               │   └── objection_*.txt
│               └── hi/
│                   └── objection_*.txt
│
└── runtime/                        # LAYER 3: Runtime override schemas
    └── ab_tests.yaml               # A/B test definitions
```

### Environment Variables (.env)

```bash
# === SECURITY: API Keys (required for production) ===
ANTHROPIC_API_KEY=sk-ant-...
QDRANT_API_KEY=...
OPENAI_API_KEY=...  # Optional: if using OpenAI

# === SERVICE URLs (defaults for dev) ===
STT_URL=http://127.0.0.1:8091
TTS_URL=http://127.0.0.1:8092
OLLAMA_URL=http://localhost:11434
SCYLLA_HOSTS=127.0.0.1:9042

# === FEATURE FLAGS ===
USE_CONFIG_DRIVEN=true
DOMAIN_ID=gold_loan

# === SERVER ===
PORT=8080
HOST=0.0.0.0
RUST_LOG=info
```

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Domain code in core agent | ~40% | <5% |
| Hardcoded rates/thresholds | 50+ | 0 |
| New domain setup time | N/A (requires code) | <1 hour (config only) |
| Config hot-reload support | Partial | Full |

---

## Out of Scope

- Database-backed configuration (future enhancement)
- Multi-tenant runtime (single domain per deployment)
- Full i18n framework (use simple template + language key for now)
- Tool implementation refactoring (only schemas)

---

## Appendix: File Relationship

This unified plan **supersedes** `voice-agent/docs/BACKEND_REFACTORING_PLAN.md`:

| Old Plan Section | New Location |
|-----------------|--------------|
| Part 1: File Size & SRP | P1-SRP (execution order #4) |
| Part 2: Hardcoded Values | P0-Config + Env Module Pattern |
| Part 3: Design Patterns | Traits Driving Factory Patterns |
| Part 4: Code Duplication | P3-Cleanup (execution order #7) |
| Part 5: Concurrency | P0-Perf (execution order #1) |
| Part 6: Domain Coupling | Phases P1-P5 (full domain abstraction) |
| Part 7: Action Plan | Execution Order table |

**Action:** After approval, delete old file and save this as:
- `voice-agent/docs/UNIFIED_REFACTORING_PLAN.md`

---

*Generated: 2026-01-06*
