//! P0 FIX: Audit Logging with Merkle Chain Verification
//!
//! Provides immutable audit logging for RBI compliance:
//! - AI disclosure events
//! - Consent tracking events
//! - PII access events
//! - Compliance check events
//! - Conversation lifecycle events
//!
//! Each entry is chained to the previous using SHA-256 hashing,
//! creating a tamper-evident merkle chain.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;
use crate::{ScyllaClient, PersistenceError};

/// Audit event types for compliance tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditEventType {
    /// AI disclosure was given to customer
    AiDisclosureGiven,
    /// Recording consent was obtained
    RecordingConsentObtained,
    /// Recording consent was denied
    RecordingConsentDenied,
    /// PII processing consent obtained
    PiiConsentObtained,
    /// PII was accessed
    PiiAccessed,
    /// PII was redacted
    PiiRedacted,
    /// Compliance check was performed
    ComplianceCheckPerformed,
    /// Compliance violation was detected
    ComplianceViolationDetected,
    /// Loan recommendation was made
    LoanRecommendationMade,
    /// Human escalation was requested
    HumanEscalationRequested,
    /// Conversation started
    ConversationStarted,
    /// Conversation ended
    ConversationEnded,
    /// Tool was executed
    ToolExecuted,
    /// Stage transition occurred
    StageTransition,
    /// Data was exported
    DataExported,
}

impl AuditEventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AiDisclosureGiven => "ai_disclosure_given",
            Self::RecordingConsentObtained => "recording_consent_obtained",
            Self::RecordingConsentDenied => "recording_consent_denied",
            Self::PiiConsentObtained => "pii_consent_obtained",
            Self::PiiAccessed => "pii_accessed",
            Self::PiiRedacted => "pii_redacted",
            Self::ComplianceCheckPerformed => "compliance_check_performed",
            Self::ComplianceViolationDetected => "compliance_violation_detected",
            Self::LoanRecommendationMade => "loan_recommendation_made",
            Self::HumanEscalationRequested => "human_escalation_requested",
            Self::ConversationStarted => "conversation_started",
            Self::ConversationEnded => "conversation_ended",
            Self::ToolExecuted => "tool_executed",
            Self::StageTransition => "stage_transition",
            Self::DataExported => "data_exported",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "ai_disclosure_given" => Self::AiDisclosureGiven,
            "recording_consent_obtained" => Self::RecordingConsentObtained,
            "recording_consent_denied" => Self::RecordingConsentDenied,
            "pii_consent_obtained" => Self::PiiConsentObtained,
            "pii_accessed" => Self::PiiAccessed,
            "pii_redacted" => Self::PiiRedacted,
            "compliance_check_performed" => Self::ComplianceCheckPerformed,
            "compliance_violation_detected" => Self::ComplianceViolationDetected,
            "loan_recommendation_made" => Self::LoanRecommendationMade,
            "human_escalation_requested" => Self::HumanEscalationRequested,
            "conversation_started" => Self::ConversationStarted,
            "conversation_ended" => Self::ConversationEnded,
            "tool_executed" => Self::ToolExecuted,
            "stage_transition" => Self::StageTransition,
            "data_exported" => Self::DataExported,
            _ => Self::ComplianceCheckPerformed, // Default
        }
    }
}

/// Audit outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditOutcome {
    Success,
    Failure,
    Pending,
    Skipped,
}

impl AuditOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Pending => "pending",
            Self::Skipped => "skipped",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "success" => Self::Success,
            "failure" => Self::Failure,
            "pending" => Self::Pending,
            "skipped" => Self::Skipped,
            _ => Self::Success,
        }
    }
}

/// Actor who performed the action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actor {
    /// Actor type (system, agent, user, admin)
    pub actor_type: String,
    /// Actor identifier
    pub actor_id: String,
    /// Session ID if applicable
    pub session_id: Option<String>,
}

impl Actor {
    pub fn system() -> Self {
        Self {
            actor_type: "system".to_string(),
            actor_id: "voice-agent".to_string(),
            session_id: None,
        }
    }

    pub fn agent(session_id: &str) -> Self {
        Self {
            actor_type: "agent".to_string(),
            actor_id: "voice-agent".to_string(),
            session_id: Some(session_id.to_string()),
        }
    }

    pub fn user(session_id: &str, phone: Option<&str>) -> Self {
        Self {
            actor_type: "user".to_string(),
            actor_id: phone.unwrap_or("anonymous").to_string(),
            session_id: Some(session_id.to_string()),
        }
    }
}

/// Audit log entry with merkle chain linking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID
    pub id: Uuid,
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,
    /// Type of audit event
    pub event_type: AuditEventType,
    /// Who performed the action
    pub actor: Actor,
    /// Resource type (conversation, lead, appointment, etc.)
    pub resource_type: String,
    /// Resource identifier
    pub resource_id: String,
    /// Action performed
    pub action: String,
    /// Outcome of the action
    pub outcome: AuditOutcome,
    /// Additional details (JSON)
    pub details: serde_json::Value,
    /// Hash of the previous entry (merkle chain)
    pub previous_hash: String,
    /// Hash of this entry
    pub hash: String,
}

impl AuditEntry {
    /// Create a new audit entry with computed hash
    pub fn new(
        event_type: AuditEventType,
        actor: Actor,
        resource_type: impl Into<String>,
        resource_id: impl Into<String>,
        action: impl Into<String>,
        outcome: AuditOutcome,
        details: serde_json::Value,
        previous_hash: impl Into<String>,
    ) -> Self {
        let id = Uuid::new_v4();
        let timestamp = Utc::now();
        let resource_type = resource_type.into();
        let resource_id = resource_id.into();
        let action = action.into();
        let previous_hash = previous_hash.into();

        let hash = Self::compute_hash(
            &id,
            &timestamp,
            &event_type,
            &actor,
            &resource_type,
            &resource_id,
            &action,
            &outcome,
            &details,
            &previous_hash,
        );

        Self {
            id,
            timestamp,
            event_type,
            actor,
            resource_type,
            resource_id,
            action,
            outcome,
            details,
            previous_hash,
            hash,
        }
    }

    /// Compute SHA-256 hash of the entry
    fn compute_hash(
        id: &Uuid,
        timestamp: &DateTime<Utc>,
        event_type: &AuditEventType,
        actor: &Actor,
        resource_type: &str,
        resource_id: &str,
        action: &str,
        outcome: &AuditOutcome,
        details: &serde_json::Value,
        previous_hash: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(id.to_string().as_bytes());
        hasher.update(timestamp.to_rfc3339().as_bytes());
        hasher.update(event_type.as_str().as_bytes());
        hasher.update(actor.actor_type.as_bytes());
        hasher.update(actor.actor_id.as_bytes());
        if let Some(ref session_id) = actor.session_id {
            hasher.update(session_id.as_bytes());
        }
        hasher.update(resource_type.as_bytes());
        hasher.update(resource_id.as_bytes());
        hasher.update(action.as_bytes());
        hasher.update(outcome.as_str().as_bytes());
        hasher.update(details.to_string().as_bytes());
        hasher.update(previous_hash.as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Verify the hash of this entry
    pub fn verify(&self) -> bool {
        let computed = Self::compute_hash(
            &self.id,
            &self.timestamp,
            &self.event_type,
            &self.actor,
            &self.resource_type,
            &self.resource_id,
            &self.action,
            &self.outcome,
            &self.details,
            &self.previous_hash,
        );

        computed == self.hash
    }

    /// Verify chain integrity (this entry's previous_hash matches expected)
    pub fn verify_chain(&self, expected_previous: &str) -> bool {
        self.previous_hash == expected_previous && self.verify()
    }
}

/// Query for audit log entries
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by session ID
    pub session_id: Option<String>,
    /// Filter by event type
    pub event_type: Option<AuditEventType>,
    /// Filter by resource type
    pub resource_type: Option<String>,
    /// Filter by resource ID
    pub resource_id: Option<String>,
    /// Filter by date range start
    pub from: Option<DateTime<Utc>>,
    /// Filter by date range end
    pub to: Option<DateTime<Utc>>,
    /// Maximum results
    pub limit: Option<i32>,
}

/// Audit log service trait
#[async_trait]
pub trait AuditLog: Send + Sync {
    /// Log an audit entry
    async fn log(&self, entry: AuditEntry) -> Result<(), PersistenceError>;

    /// Query audit entries
    async fn query(&self, query: AuditQuery) -> Result<Vec<AuditEntry>, PersistenceError>;

    /// Get the latest entry hash (for chaining)
    async fn get_latest_hash(&self, session_id: &str) -> Result<String, PersistenceError>;

    /// Verify chain integrity for a session
    async fn verify_chain(&self, session_id: &str) -> Result<bool, PersistenceError>;
}

/// ScyllaDB-backed audit log implementation
#[derive(Clone)]
pub struct ScyllaAuditLog {
    client: ScyllaClient,
}

impl ScyllaAuditLog {
    pub fn new(client: ScyllaClient) -> Self {
        Self { client }
    }

    /// Genesis hash for new chains
    pub fn genesis_hash() -> String {
        "0".repeat(64) // SHA-256 produces 64 hex chars
    }
}

#[async_trait]
impl AuditLog for ScyllaAuditLog {
    async fn log(&self, entry: AuditEntry) -> Result<(), PersistenceError> {
        let date = entry.timestamp.format("%Y-%m-%d").to_string();
        let session_id = entry.actor.session_id.as_deref().unwrap_or("system");

        let query = format!(
            "INSERT INTO {}.audit_log (
                partition_date, session_id, timestamp, id, event_type,
                actor_type, actor_id, resource_type, resource_id,
                action, outcome, details, previous_hash, hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                &date,
                session_id,
                entry.timestamp.timestamp_millis(),
                entry.id,
                entry.event_type.as_str(),
                &entry.actor.actor_type,
                &entry.actor.actor_id,
                &entry.resource_type,
                &entry.resource_id,
                &entry.action,
                entry.outcome.as_str(),
                entry.details.to_string(),
                &entry.previous_hash,
                &entry.hash,
            ),
        ).await?;

        tracing::debug!(
            event_type = entry.event_type.as_str(),
            resource_id = %entry.resource_id,
            hash = %entry.hash,
            "Audit entry logged"
        );

        Ok(())
    }

    async fn query(&self, query: AuditQuery) -> Result<Vec<AuditEntry>, PersistenceError> {
        // Build query based on filters
        let limit = query.limit.unwrap_or(100);

        // Simple query - in production would build dynamic WHERE clause
        let cql = format!(
            "SELECT partition_date, session_id, timestamp, id, event_type,
                    actor_type, actor_id, resource_type, resource_id,
                    action, outcome, details, previous_hash, hash
             FROM {}.audit_log
             LIMIT ?",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(cql, (limit,))
            .await?;

        let mut entries = Vec::new();
        if let Some(rows) = result.rows {
            for row in rows {
                let (
                    _date,
                    session_id,
                    timestamp,
                    id,
                    event_type,
                    actor_type,
                    actor_id,
                    resource_type,
                    resource_id,
                    action,
                    outcome,
                    details_str,
                    previous_hash,
                    hash,
                ): (
                    String, String, i64, Uuid, String,
                    String, String, String, String,
                    String, String, String, String, String,
                ) = row.into_typed().map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                entries.push(AuditEntry {
                    id,
                    timestamp: DateTime::from_timestamp_millis(timestamp).unwrap_or_else(Utc::now),
                    event_type: AuditEventType::from_str(&event_type),
                    actor: Actor {
                        actor_type,
                        actor_id,
                        session_id: Some(session_id),
                    },
                    resource_type,
                    resource_id,
                    action,
                    outcome: AuditOutcome::from_str(&outcome),
                    details: serde_json::from_str(&details_str).unwrap_or(serde_json::Value::Null),
                    previous_hash,
                    hash,
                });
            }
        }

        Ok(entries)
    }

    async fn get_latest_hash(&self, session_id: &str) -> Result<String, PersistenceError> {
        let query = format!(
            "SELECT hash FROM {}.audit_log WHERE session_id = ? ORDER BY timestamp DESC LIMIT 1",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (session_id,))
            .await?;

        if let Some(rows) = result.rows {
            if let Some(row) = rows.into_iter().next() {
                let (hash,): (String,) = row.into_typed()
                    .map_err(|e| PersistenceError::InvalidData(e.to_string()))?;
                return Ok(hash);
            }
        }

        // No previous entries - return genesis hash
        Ok(Self::genesis_hash())
    }

    async fn verify_chain(&self, session_id: &str) -> Result<bool, PersistenceError> {
        let query = format!(
            "SELECT timestamp, id, event_type, actor_type, actor_id,
                    resource_type, resource_id, action, outcome,
                    details, previous_hash, hash
             FROM {}.audit_log
             WHERE session_id = ?
             ORDER BY timestamp ASC",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (session_id,))
            .await?;

        let mut expected_previous = Self::genesis_hash();

        if let Some(rows) = result.rows {
            for row in rows {
                let (
                    timestamp,
                    id,
                    event_type,
                    actor_type,
                    actor_id,
                    resource_type,
                    resource_id,
                    action,
                    outcome,
                    details_str,
                    previous_hash,
                    hash,
                ): (
                    i64, Uuid, String, String, String,
                    String, String, String, String,
                    String, String, String,
                ) = row.into_typed().map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                let entry = AuditEntry {
                    id,
                    timestamp: DateTime::from_timestamp_millis(timestamp).unwrap_or_else(Utc::now),
                    event_type: AuditEventType::from_str(&event_type),
                    actor: Actor {
                        actor_type,
                        actor_id,
                        session_id: Some(session_id.to_string()),
                    },
                    resource_type,
                    resource_id,
                    action,
                    outcome: AuditOutcome::from_str(&outcome),
                    details: serde_json::from_str(&details_str).unwrap_or(serde_json::Value::Null),
                    previous_hash,
                    hash: hash.clone(),
                };

                if !entry.verify_chain(&expected_previous) {
                    tracing::error!(
                        entry_id = %entry.id,
                        expected = %expected_previous,
                        actual = %entry.previous_hash,
                        "Audit chain verification failed"
                    );
                    return Ok(false);
                }

                expected_previous = hash;
            }
        }

        Ok(true)
    }
}

/// Helper for common audit logging operations
pub struct AuditLogger {
    log: std::sync::Arc<dyn AuditLog>,
}

impl AuditLogger {
    pub fn new(log: std::sync::Arc<dyn AuditLog>) -> Self {
        Self { log }
    }

    /// Log AI disclosure event
    pub async fn log_ai_disclosure(
        &self,
        session_id: &str,
        language: &str,
        disclosure_text: &str,
    ) -> Result<(), PersistenceError> {
        let previous_hash = self.log.get_latest_hash(session_id).await?;

        let entry = AuditEntry::new(
            AuditEventType::AiDisclosureGiven,
            Actor::agent(session_id),
            "conversation",
            session_id,
            "gave_ai_disclosure",
            AuditOutcome::Success,
            serde_json::json!({
                "language": language,
                "disclosure_text": disclosure_text,
            }),
            previous_hash,
        );

        self.log.log(entry).await
    }

    /// Log consent event
    pub async fn log_consent(
        &self,
        session_id: &str,
        consent_type: &str,
        given: bool,
        method: &str,
    ) -> Result<(), PersistenceError> {
        let previous_hash = self.log.get_latest_hash(session_id).await?;

        let event_type = if consent_type == "recording" {
            if given { AuditEventType::RecordingConsentObtained }
            else { AuditEventType::RecordingConsentDenied }
        } else {
            AuditEventType::PiiConsentObtained
        };

        let entry = AuditEntry::new(
            event_type,
            Actor::user(session_id, None),
            "conversation",
            session_id,
            format!("{}_consent_{}", consent_type, if given { "given" } else { "denied" }),
            AuditOutcome::Success,
            serde_json::json!({
                "consent_type": consent_type,
                "given": given,
                "method": method,
            }),
            previous_hash,
        );

        self.log.log(entry).await
    }

    /// Log conversation start
    pub async fn log_conversation_start(
        &self,
        session_id: &str,
        language: &str,
    ) -> Result<(), PersistenceError> {
        let entry = AuditEntry::new(
            AuditEventType::ConversationStarted,
            Actor::system(),
            "conversation",
            session_id,
            "conversation_started",
            AuditOutcome::Success,
            serde_json::json!({
                "language": language,
                "started_at": Utc::now().to_rfc3339(),
            }),
            ScyllaAuditLog::genesis_hash(),
        );

        self.log.log(entry).await
    }

    /// Log conversation end
    pub async fn log_conversation_end(
        &self,
        session_id: &str,
        reason: &str,
        duration_seconds: u64,
    ) -> Result<(), PersistenceError> {
        let previous_hash = self.log.get_latest_hash(session_id).await?;

        let entry = AuditEntry::new(
            AuditEventType::ConversationEnded,
            Actor::system(),
            "conversation",
            session_id,
            "conversation_ended",
            AuditOutcome::Success,
            serde_json::json!({
                "reason": reason,
                "duration_seconds": duration_seconds,
                "ended_at": Utc::now().to_rfc3339(),
            }),
            previous_hash,
        );

        self.log.log(entry).await
    }

    /// Log tool execution
    pub async fn log_tool_execution(
        &self,
        session_id: &str,
        tool_name: &str,
        success: bool,
        details: serde_json::Value,
    ) -> Result<(), PersistenceError> {
        let previous_hash = self.log.get_latest_hash(session_id).await?;

        let entry = AuditEntry::new(
            AuditEventType::ToolExecuted,
            Actor::agent(session_id),
            "tool",
            tool_name,
            format!("execute_{}", tool_name),
            if success { AuditOutcome::Success } else { AuditOutcome::Failure },
            details,
            previous_hash,
        );

        self.log.log(entry).await
    }

    /// Log human escalation request
    pub async fn log_escalation(
        &self,
        session_id: &str,
        reason: &str,
        escalation_id: &str,
    ) -> Result<(), PersistenceError> {
        let previous_hash = self.log.get_latest_hash(session_id).await?;

        let entry = AuditEntry::new(
            AuditEventType::HumanEscalationRequested,
            Actor::agent(session_id),
            "escalation",
            escalation_id,
            "request_human_escalation",
            AuditOutcome::Success,
            serde_json::json!({
                "reason": reason,
                "escalation_id": escalation_id,
            }),
            previous_hash,
        );

        self.log.log(entry).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new(
            AuditEventType::AiDisclosureGiven,
            Actor::agent("test-session"),
            "conversation",
            "test-session",
            "gave_ai_disclosure",
            AuditOutcome::Success,
            serde_json::json!({"language": "hi"}),
            ScyllaAuditLog::genesis_hash(),
        );

        assert!(!entry.hash.is_empty());
        assert!(entry.verify());
    }

    #[test]
    fn test_audit_chain_verification() {
        let genesis = ScyllaAuditLog::genesis_hash();

        let entry1 = AuditEntry::new(
            AuditEventType::ConversationStarted,
            Actor::system(),
            "conversation",
            "session-1",
            "started",
            AuditOutcome::Success,
            serde_json::json!({}),
            &genesis,
        );

        assert!(entry1.verify_chain(&genesis));

        let entry2 = AuditEntry::new(
            AuditEventType::AiDisclosureGiven,
            Actor::agent("session-1"),
            "conversation",
            "session-1",
            "disclosed",
            AuditOutcome::Success,
            serde_json::json!({}),
            &entry1.hash,
        );

        assert!(entry2.verify_chain(&entry1.hash));
        assert!(!entry2.verify_chain(&genesis)); // Should fail with wrong previous
    }

    #[test]
    fn test_tamper_detection() {
        let mut entry = AuditEntry::new(
            AuditEventType::AiDisclosureGiven,
            Actor::agent("test"),
            "conversation",
            "test",
            "disclosed",
            AuditOutcome::Success,
            serde_json::json!({}),
            ScyllaAuditLog::genesis_hash(),
        );

        assert!(entry.verify());

        // Tamper with the entry
        entry.action = "tampered".to_string();

        // Should now fail verification
        assert!(!entry.verify());
    }

    #[test]
    fn test_event_type_serialization() {
        assert_eq!(AuditEventType::AiDisclosureGiven.as_str(), "ai_disclosure_given");
        assert_eq!(
            AuditEventType::from_str("ai_disclosure_given"),
            AuditEventType::AiDisclosureGiven
        );
    }
}
