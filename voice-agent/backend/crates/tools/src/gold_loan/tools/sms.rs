//! Send SMS Tool
//!
//! Send SMS messages to customers for appointment confirmations, follow-ups, etc.

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Send SMS tool
pub struct SendSmsTool {
    sms_service: Option<Arc<dyn voice_agent_persistence::SmsService>>,
}

impl SendSmsTool {
    pub fn new() -> Self {
        Self { sms_service: None }
    }

    pub fn with_sms_service(service: Arc<dyn voice_agent_persistence::SmsService>) -> Self {
        Self {
            sms_service: Some(service),
        }
    }
}

#[async_trait]
impl Tool for SendSmsTool {
    fn name(&self) -> &str {
        "send_sms"
    }

    fn description(&self) -> &str {
        "Send an SMS message to the customer for appointment confirmations, follow-ups, or information sharing"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "phone_number",
                    PropertySchema::string("10-digit mobile number"),
                    true,
                )
                .property(
                    "message_type",
                    PropertySchema::enum_type(
                        "Type of SMS message",
                        vec![
                            "appointment_confirmation".into(),
                            "appointment_reminder".into(),
                            "follow_up".into(),
                            "welcome".into(),
                            "promotional".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "customer_name",
                    PropertySchema::string("Customer name for personalization"),
                    false,
                )
                .property(
                    "custom_message",
                    PropertySchema::string("Custom message text (for follow_up type)"),
                    false,
                )
                .property(
                    "appointment_details",
                    PropertySchema::string("Appointment details (date, time, branch)"),
                    false,
                )
                .property(
                    "session_id",
                    PropertySchema::string("Session ID for tracking"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let phone = input
            .get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        if phone.len() != 10 || !phone.chars().all(|c| c.is_ascii_digit()) {
            return Err(ToolError::invalid_params("phone_number must be 10 digits"));
        }

        let msg_type_str = input
            .get("message_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("message_type is required"))?;

        let customer_name = input
            .get("customer_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Customer");

        let session_id = input.get("session_id").and_then(|v| v.as_str());

        let msg_type = match msg_type_str {
            "appointment_confirmation" => voice_agent_persistence::SmsType::AppointmentConfirmation,
            "appointment_reminder" => voice_agent_persistence::SmsType::AppointmentReminder,
            "follow_up" => voice_agent_persistence::SmsType::FollowUp,
            "welcome" => voice_agent_persistence::SmsType::Welcome,
            "promotional" => voice_agent_persistence::SmsType::Promotional,
            _ => voice_agent_persistence::SmsType::FollowUp,
        };

        // TODO: SMS templates should come from domain config
        let message_text = match msg_type {
            voice_agent_persistence::SmsType::AppointmentConfirmation => {
                let details = input
                    .get("appointment_details")
                    .and_then(|v| v.as_str())
                    .unwrap_or("scheduled date and time");
                format!(
                    "Dear {}, your Gold Loan appointment is confirmed for {}. Please bring your gold and KYC documents. - Bank",
                    customer_name, details
                )
            }
            voice_agent_persistence::SmsType::AppointmentReminder => {
                let details = input
                    .get("appointment_details")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tomorrow");
                format!(
                    "Reminder: Dear {}, your Gold Loan appointment is {}. Please bring your gold and KYC documents. - Bank",
                    customer_name, details
                )
            }
            voice_agent_persistence::SmsType::FollowUp => input
                .get("custom_message")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    format!(
                        "Dear {}, thank you for your interest in Gold Loan. Get up to 75% of gold value at competitive rates. - Bank",
                        customer_name
                    )
                }),
            voice_agent_persistence::SmsType::Welcome => {
                format!(
                    "Welcome, {}! We're excited to help you with your gold loan needs. - Bank",
                    customer_name
                )
            }
            voice_agent_persistence::SmsType::Promotional => {
                format!(
                    "Special Offer for {}: Get gold loan at competitive rates with instant disbursement! T&C apply. - Bank",
                    customer_name
                )
            }
            _ => format!(
                "Dear {}, thank you for contacting us. - Bank",
                customer_name
            ),
        };

        let (message_id, status, simulated) = if let Some(ref service) = self.sms_service {
            match service
                .send_sms(phone, &message_text, msg_type, session_id)
                .await
            {
                Ok(result) => (
                    result.message_id.to_string(),
                    result.status.as_str().to_string(),
                    result.simulated,
                ),
                Err(e) => {
                    tracing::warn!("SMS service failed: {}", e);
                    let id = format!(
                        "SMS{}",
                        uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
                    );
                    (id, "failed".to_string(), false)
                }
            }
        } else {
            let id = format!(
                "SMS{}",
                uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
            );
            (id, "simulated_not_sent".to_string(), true)
        };

        let success = status != "failed";

        let result = json!({
            "success": success,
            "message_id": message_id,
            "phone_number": phone,
            "message_type": msg_type_str,
            "message_text": message_text,
            "status": status,
            "simulated": simulated,
            "sent_at": if success { Some(Utc::now().to_rfc3339()) } else { None },
            "message": if success {
                format!("SMS {} to {}.", if simulated { "simulated" } else { "sent" }, phone)
            } else {
                "Failed to send SMS. Please try again.".to_string()
            }
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        30
    }
}

impl Default for SendSmsTool {
    fn default() -> Self {
        Self::new()
    }
}
