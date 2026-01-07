//! Human Escalation Tool
//!
//! Escalate the conversation to a human agent.

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Human escalation tool
pub struct EscalateToHumanTool {
    on_escalate: Option<Arc<dyn Fn(String, String, String) + Send + Sync>>,
}

impl EscalateToHumanTool {
    pub fn new() -> Self {
        Self { on_escalate: None }
    }

    pub fn with_callback<F>(callback: F) -> Self
    where
        F: Fn(String, String, String) + Send + Sync + 'static,
    {
        Self {
            on_escalate: Some(Arc::new(callback)),
        }
    }
}

#[async_trait]
impl Tool for EscalateToHumanTool {
    fn name(&self) -> &str {
        "escalate_to_human"
    }

    fn description(&self) -> &str {
        "Escalate the conversation to a human agent when the customer requests it or when the AI cannot help"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "reason",
                    PropertySchema::enum_type(
                        "Reason for escalation",
                        vec![
                            "customer_request".into(),
                            "complex_query".into(),
                            "complaint".into(),
                            "technical_issue".into(),
                            "sensitive_matter".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "session_id",
                    PropertySchema::string("Current session ID"),
                    true,
                )
                .property(
                    "customer_phone",
                    PropertySchema::string("Customer phone number"),
                    false,
                )
                .property(
                    "summary",
                    PropertySchema::string("Brief summary of conversation so far"),
                    false,
                )
                .property(
                    "priority",
                    PropertySchema::enum_type(
                        "Escalation priority",
                        vec!["normal".into(), "high".into(), "urgent".into()],
                    )
                    .with_default(json!("normal")),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let reason = input
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("reason is required"))?;

        let session_id = input
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("session_id is required"))?;

        let customer_phone = input
            .get("customer_phone")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let summary = input
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("No summary provided");

        let priority = input
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("normal");

        let escalation_id = format!(
            "ESC{}",
            uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
        );

        let estimated_wait = match priority {
            "urgent" => "1-2 minutes",
            "high" => "2-5 minutes",
            _ => "5-10 minutes",
        };

        if let Some(ref callback) = self.on_escalate {
            callback(
                escalation_id.clone(),
                session_id.to_string(),
                reason.to_string(),
            );
        }

        tracing::info!(
            escalation_id = %escalation_id,
            session_id = %session_id,
            reason = %reason,
            priority = %priority,
            "Human escalation requested"
        );

        let result = json!({
            "success": true,
            "escalation_id": escalation_id,
            "session_id": session_id,
            "customer_phone": customer_phone,
            "reason": reason,
            "priority": priority,
            "summary": summary,
            "status": "queued",
            "estimated_wait": estimated_wait,
            "queue_position": 1,
            "created_at": Utc::now().to_rfc3339(),
            "message": format!(
                "Your request has been escalated to a human agent. Escalation ID: {}. Estimated wait time: {}. Please hold.",
                escalation_id, estimated_wait
            ),
            "instructions": "A human agent will join this conversation shortly. Please stay on the line."
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        10
    }
}

impl Default for EscalateToHumanTool {
    fn default() -> Self {
        Self::new()
    }
}
