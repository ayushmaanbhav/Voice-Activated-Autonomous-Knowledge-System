# Critical Fixes Required - Priority P0/P1

This document details the specific fixes required before any deployment.

---

## P0 - BLOCKING ISSUES

### 1. Enable Translation ONNX Feature

**File:** `voice-agent-rust/crates/text_processing/Cargo.toml`

**Current:**
```toml
[features]
default = []
onnx = ["dep:ort", "dep:ndarray", "dep:tokenizers"]
```

**Fix:**
```toml
[features]
default = ["onnx"]
onnx = ["dep:ort", "dep:ndarray", "dep:tokenizers"]
```

**Additional:** Add 12 missing language pairs in `translation/mod.rs`:
- Assamese, Urdu, Kashmiri, Sindhi, Konkani, Dogri
- Bodo, Maithili, Santali, Nepali, Manipuri, Sanskrit

---

### 2. Fix STT Confidence Scores

**File:** `voice-agent-rust/crates/pipeline/src/stt/indicconformer.rs`

**Current (multiple locations):**
```rust
confidence: 0.8,  // HARDCODED
confidence: 0.9,  // HARDCODED
```

**Fix:** Extract from model logits
```rust
fn extract_confidence(logits: &[f32], token_id: usize) -> f32 {
    // Apply softmax to get probabilities
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let prob = (logits[token_id] - max_logit).exp() / exp_sum;
    prob
}

// Use in transcription
let confidence = extract_confidence(&logits, predicted_token);
```

**Also fix word timestamps** in same file:
```rust
// Current: naive 50ms per character
let char_ms = 50;

// Fix: Use frame-level timing from model output
fn compute_word_timestamps(
    frames: &[Frame],
    words: &[String],
    frame_duration_ms: u64,
) -> Vec<WordTimestamp> {
    // Implement CTC alignment or frame-level tracking
}
```

---

### 3. Implement AI Disclosure Compliance

**File:** `voice-agent-rust/crates/text_processing/src/compliance/rules.rs`

**Add:**
```rust
pub struct AiDisclosureRule {
    pub enabled: bool,
    pub messages: HashMap<Language, String>,
}

impl Default for AiDisclosureRule {
    fn default() -> Self {
        let mut messages = HashMap::new();
        messages.insert(
            Language::Hindi,
            "यह एक AI सहायक है। आप किसी भी समय मानव एजेंट से बात कर सकते हैं।".to_string()
        );
        messages.insert(
            Language::English,
            "This is an AI assistant. You can speak with a human agent at any time.".to_string()
        );
        // Add for other languages...

        Self { enabled: true, messages }
    }
}
```

**File:** `voice-agent-rust/crates/agent/src/conversation.rs`

**Add disclosure tracking:**
```rust
pub struct Conversation {
    // ... existing fields
    pub ai_disclosure_given: bool,
    pub ai_disclosure_timestamp: Option<DateTime<Utc>>,
}

impl Conversation {
    pub fn mark_ai_disclosed(&mut self) {
        self.ai_disclosure_given = true;
        self.ai_disclosure_timestamp = Some(Utc::now());
        // Log compliance event
    }
}
```

---

### 4. Implement Consent Tracking

**File:** `voice-agent-rust/crates/core/src/conversation.rs`

**Add:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub recording_consent: bool,
    pub pii_processing_consent: bool,
    pub marketing_consent: Option<bool>,
    pub consent_timestamp: DateTime<Utc>,
    pub consent_method: ConsentMethod,
    pub consent_language: Language,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConsentMethod {
    Voice,      // Said "yes" to recording
    Text,       // Typed confirmation
    Implied,    // Continued after disclosure
    Explicit,   // Clicked/tapped consent
}
```

**Integrate into Conversation:**
```rust
pub struct Conversation {
    // ... existing
    pub consent: Option<ConsentRecord>,
}
```

---

### 5. Implement Audit Logging

**File:** `voice-agent-rust/crates/persistence/src/audit.rs` (NEW)

```rust
use sha2::{Sha256, Digest};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub actor: Actor,
    pub resource_type: String,
    pub resource_id: String,
    pub action: String,
    pub outcome: AuditOutcome,
    pub details: serde_json::Value,
    pub previous_hash: String,
    pub hash: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AuditEventType {
    AiDisclosureGiven,
    RecordingConsentObtained,
    PiiAccessed,
    PiiRedacted,
    ComplianceCheckPerformed,
    ComplianceViolationDetected,
    LoanRecommendationMade,
    HumanEscalationRequested,
    ConversationStarted,
    ConversationEnded,
}

impl AuditEntry {
    pub fn new(
        event_type: AuditEventType,
        actor: Actor,
        resource_type: &str,
        resource_id: &str,
        action: &str,
        outcome: AuditOutcome,
        details: serde_json::Value,
        previous_hash: &str,
    ) -> Self {
        let id = Uuid::new_v4();
        let timestamp = Utc::now();

        let hash = Self::compute_hash(
            &id, &timestamp, &event_type, &actor,
            resource_type, resource_id, action,
            &outcome, &details, previous_hash
        );

        Self {
            id,
            timestamp,
            event_type,
            actor,
            resource_type: resource_type.to_string(),
            resource_id: resource_id.to_string(),
            action: action.to_string(),
            outcome,
            details,
            previous_hash: previous_hash.to_string(),
            hash,
        }
    }

    fn compute_hash(/* params */) -> String {
        let mut hasher = Sha256::new();
        // Hash all fields
        format!("{:x}", hasher.finalize())
    }

    pub fn verify(&self, expected_previous: &str) -> bool {
        self.previous_hash == expected_previous &&
        Self::compute_hash(/* self fields */) == self.hash
    }
}

#[async_trait]
pub trait AuditLog: Send + Sync {
    async fn log(&self, entry: AuditEntry) -> Result<(), AuditError>;
    async fn query(&self, query: AuditQuery) -> Result<Vec<AuditEntry>, AuditError>;
    async fn verify_chain(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Result<bool, AuditError>;
}
```

**ScyllaDB Schema:**
```cql
CREATE TABLE audit_log (
    partition_date date,
    timestamp timestamp,
    id uuid,
    event_type text,
    actor_type text,
    actor_id text,
    resource_type text,
    resource_id text,
    action text,
    outcome text,
    details text,
    previous_hash text,
    hash text,
    PRIMARY KEY ((partition_date), timestamp, id)
) WITH default_time_to_live = 220924800;  -- 7 years in seconds
```

---

### 6. Implement Missing MCP Tools

**File:** `voice-agent-rust/crates/tools/src/gold_loan.rs`

#### 6a. get_gold_price Tool

```rust
pub struct GoldPriceTool {
    config: Arc<RwLock<DomainConfig>>,
}

impl Tool for GoldPriceTool {
    fn name(&self) -> &'static str { "get_gold_price" }

    fn description(&self) -> &'static str {
        "Get current gold price per gram for different purities"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::builder()
            .property("purity", PropertySchema::string()
                .description("Gold purity")
                .enum_values(vec!["24K", "22K", "18K", "14K"]))
            .build()
    }

    async fn execute(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let purity = args.get("purity")
            .and_then(|v| v.as_str())
            .unwrap_or("22K");

        let config = self.config.read();
        let base_price = config.gold_loan.gold_price_per_gram;
        let factor = config.gold_loan.get_purity_factor(purity);
        let price = base_price * factor;

        Ok(ToolOutput::text(json!({
            "purity": purity,
            "price_per_gram_inr": price,
            "base_price_24k": base_price,
            "last_updated": config.gold_loan.price_updated_at,
        }).to_string()))
    }
}
```

#### 6b. escalate_to_human Tool

```rust
pub struct EscalateToHumanTool {
    queue_manager: Arc<dyn EscalationQueueManager>,
}

impl Tool for EscalateToHumanTool {
    fn name(&self) -> &'static str { "escalate_to_human" }

    fn description(&self) -> &'static str {
        "Transfer conversation to a human agent"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::builder()
            .required("reason")
            .property("reason", PropertySchema::string()
                .description("Reason for escalation"))
            .property("customer_name", PropertySchema::string())
            .property("phone_number", PropertySchema::string())
            .property("priority", PropertySchema::string()
                .enum_values(vec!["high", "normal", "low"]))
            .build()
    }

    async fn execute(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let reason = args.get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("reason required"))?;

        let priority = args.get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("normal");

        let escalation = self.queue_manager.create_escalation(
            reason,
            args.get("customer_name").and_then(|v| v.as_str()),
            args.get("phone_number").and_then(|v| v.as_str()),
            priority,
        ).await?;

        // Log audit event
        // ...

        Ok(ToolOutput::text(json!({
            "escalation_id": escalation.id,
            "queue_position": escalation.position,
            "estimated_wait_minutes": escalation.estimated_wait,
            "message": "Your call is being transferred to a human agent. Please hold."
        }).to_string()))
    }
}
```

#### 6c. send_sms Tool (wrap existing service)

```rust
pub struct SendSmsTool {
    sms_service: Arc<dyn SmsService>,
}

impl Tool for SendSmsTool {
    fn name(&self) -> &'static str { "send_sms" }

    fn description(&self) -> &'static str {
        "Send SMS to customer"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema::builder()
            .required("phone_number")
            .required("message_type")
            .property("phone_number", PropertySchema::string()
                .pattern(r"^[6-9]\d{9}$"))
            .property("message_type", PropertySchema::string()
                .enum_values(vec![
                    "appointment_confirmation",
                    "follow_up",
                    "welcome",
                    "rate_update"
                ]))
            .property("custom_message", PropertySchema::string())
            .build()
    }

    async fn execute(&self, args: Value) -> Result<ToolOutput, ToolError> {
        let phone = args.get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number required"))?;

        let msg_type = args.get("message_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("message_type required"))?;

        let sms_type = match msg_type {
            "appointment_confirmation" => SmsType::AppointmentConfirmation,
            "follow_up" => SmsType::FollowUp,
            "welcome" => SmsType::Welcome,
            "rate_update" => SmsType::RateUpdate,
            _ => return Err(ToolError::invalid_params("invalid message_type")),
        };

        let result = self.sms_service.send(phone, sms_type).await?;

        Ok(ToolOutput::text(json!({
            "message_id": result.id,
            "status": "queued",
            "phone": phone,
        }).to_string()))
    }
}
```

---

### 7. Add VoiceActivityDetector Trait

**File:** `voice-agent-rust/crates/core/src/traits/speech.rs`

```rust
#[derive(Debug, Clone)]
pub struct VADConfig {
    pub threshold: f32,           // Speech probability threshold (0.5)
    pub min_speech_duration_ms: u32,  // Minimum speech to confirm (256ms)
    pub min_silence_duration_ms: u32, // Minimum silence to end (320ms)
    pub energy_floor_db: f32,     // Quick silence rejection (-50dB)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VADEvent {
    SpeechStart,
    SpeechContinue { probability: f32 },
    SpeechEnd,
    Silence,
}

#[async_trait]
pub trait VoiceActivityDetector: Send + Sync + 'static {
    /// Detect if frame contains speech
    async fn detect(&self, audio: &AudioFrame, sensitivity: f32) -> bool;

    /// Get speech probability for frame
    async fn speech_probability(&self, audio: &AudioFrame) -> f32;

    /// Process audio stream, emit VAD events
    fn process_stream<'a>(
        &'a self,
        audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
        config: &'a VADConfig,
    ) -> Pin<Box<dyn Stream<Item = VADEvent> + Send + 'a>>;

    /// Reset internal state
    fn reset(&self);

    /// Get model info
    fn model_info(&self) -> &str;
}
```

---

## P1 - HIGH PRIORITY FIXES

### 8. RAG Sufficiency Check with LLM

**File:** `voice-agent-rust/crates/rag/src/agentic.rs`

**Current (heuristic only):**
```rust
impl SufficiencyChecker {
    pub fn score(&self, results: &[RetrievalResult], query: &str) -> SufficiencyResult {
        // Only checks count and average score
    }
}
```

**Fix - Add LLM evaluation:**
```rust
pub struct LlmSufficiencyChecker {
    llm: Arc<dyn LanguageModel>,
    config: SufficiencyConfig,
}

impl LlmSufficiencyChecker {
    pub async fn evaluate(
        &self,
        query: &str,
        documents: &[Document],
    ) -> Result<SufficiencyResult, RagError> {
        let context = documents.iter()
            .take(5)
            .map(|d| d.content.as_str())
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prompt = format!(r#"
Given the following documents:
{context}

And the user query: "{query}"

Evaluate if these documents can fully answer the query.
Return JSON: {{"sufficient": true/false, "coverage": 0.0-1.0, "missing": "what's missing if any"}}
"#);

        let response = self.llm.generate(GenerateRequest {
            prompt,
            max_tokens: 100,
            temperature: 0.1,
            ..Default::default()
        }).await?;

        // Parse JSON response
        let eval: SufficiencyEval = serde_json::from_str(&response.text)?;

        Ok(SufficiencyResult {
            sufficient: eval.sufficient && eval.coverage >= self.config.threshold,
            coverage: eval.coverage,
            missing: eval.missing,
            refined_query: None, // Optionally generate refined query
        })
    }
}
```

---

### 9. Context Sizing by Stage

**File:** `voice-agent-rust/crates/rag/src/context.rs` (NEW)

```rust
use voice_agent_core::conversation::ConversationStage;

#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub rag_tokens: usize,
    pub history_tokens: usize,
    pub system_tokens: usize,
}

pub fn context_budget_for_stage(stage: ConversationStage) -> ContextBudget {
    match stage {
        ConversationStage::Greeting => ContextBudget {
            rag_tokens: 200,
            history_tokens: 100,
            system_tokens: 500,
        },
        ConversationStage::Discovery => ContextBudget {
            rag_tokens: 800,
            history_tokens: 400,
            system_tokens: 600,
        },
        ConversationStage::Qualification => ContextBudget {
            rag_tokens: 1000,
            history_tokens: 500,
            system_tokens: 600,
        },
        ConversationStage::Presentation => ContextBudget {
            rag_tokens: 2000,
            history_tokens: 800,
            system_tokens: 800,
        },
        ConversationStage::ObjectionHandling => ContextBudget {
            rag_tokens: 1500,
            history_tokens: 600,
            system_tokens: 700,
        },
        ConversationStage::Closing => ContextBudget {
            rag_tokens: 500,
            history_tokens: 300,
            system_tokens: 500,
        },
        ConversationStage::Farewell => ContextBudget {
            rag_tokens: 100,
            history_tokens: 100,
            system_tokens: 300,
        },
    }
}
```

---

### 10. Fix Config Validation

**File:** `voice-agent-rust/crates/config/src/domain.rs`

```rust
impl DomainConfig {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Gold loan validation
        let gl = &self.gold_loan;
        if gl.kotak_interest_rate <= 0.0 || gl.kotak_interest_rate > 30.0 {
            errors.push(format!(
                "Interest rate {} out of valid range (0-30%)",
                gl.kotak_interest_rate
            ));
        }

        if gl.ltv_percent <= 0.0 || gl.ltv_percent > 90.0 {
            errors.push(format!(
                "LTV {} out of valid range (0-90%)",
                gl.ltv_percent
            ));
        }

        // Validate tier ordering (should decrease)
        if gl.tiered_rates.tier1_rate <= gl.tiered_rates.tier2_rate {
            errors.push("Tier 1 rate should be > Tier 2 rate".to_string());
        }
        if gl.tiered_rates.tier2_rate <= gl.tiered_rates.tier3_rate {
            errors.push("Tier 2 rate should be > Tier 3 rate".to_string());
        }

        // Validate competitor rates are higher than Kotak
        for competitor in &self.competitors.list {
            if competitor.min_interest_rate < gl.kotak_interest_rate {
                errors.push(format!(
                    "Competitor {} min rate ({}) < Kotak rate ({})",
                    competitor.name,
                    competitor.min_interest_rate,
                    gl.kotak_interest_rate
                ));
            }
        }

        // Validate processing fee
        if gl.processing_fee_percent < 0.0 || gl.processing_fee_percent > 5.0 {
            errors.push(format!(
                "Processing fee {} out of valid range (0-5%)",
                gl.processing_fee_percent
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
```

---

## Verification Checklist

After implementing fixes, verify:

- [ ] Translation returns translated text (not pass-through)
- [ ] STT confidence varies between 0.0-1.0 based on actual model output
- [ ] AI disclosure given at conversation start (logged)
- [ ] Consent recorded before any PII processing
- [ ] Audit log entries have valid hash chain
- [ ] get_gold_price returns current rates
- [ ] escalate_to_human creates queue entry
- [ ] Config validation rejects invalid rates
- [ ] RAG respects stage-based token budgets
