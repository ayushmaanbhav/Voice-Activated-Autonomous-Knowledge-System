//! Lead Capture Tool
//!
//! Capture customer lead information for follow-up.

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::integrations::{CrmIntegration, CrmLead, InterestLevel, LeadSource, LeadStatus};
use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Lead capture tool
pub struct LeadCaptureTool {
    crm: Option<Arc<dyn CrmIntegration>>,
}

impl LeadCaptureTool {
    pub fn new() -> Self {
        Self { crm: None }
    }

    pub fn with_crm(crm: Arc<dyn CrmIntegration>) -> Self {
        Self { crm: Some(crm) }
    }
}

#[async_trait]
impl Tool for LeadCaptureTool {
    fn name(&self) -> &str {
        "capture_lead"
    }

    fn description(&self) -> &str {
        "Capture customer lead information for follow-up"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "customer_name",
                    PropertySchema::string("Customer's full name"),
                    true,
                )
                .property(
                    "phone_number",
                    PropertySchema::string("10-digit mobile number"),
                    true,
                )
                .property("city", PropertySchema::string("Customer's city"), false)
                .property(
                    "preferred_branch",
                    PropertySchema::string("Preferred branch location"),
                    false,
                )
                .property(
                    "estimated_gold_weight",
                    PropertySchema::number("Estimated gold weight in grams"),
                    false,
                )
                .property(
                    "interest_level",
                    PropertySchema::enum_type(
                        "Customer's interest level",
                        vec!["High".into(), "Medium".into(), "Low".into()],
                    ),
                    false,
                )
                .property(
                    "notes",
                    PropertySchema::string("Additional notes from conversation"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input
            .get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input
            .get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        if phone.len() != 10 || !phone.chars().all(|c| c.is_ascii_digit()) {
            return Err(ToolError::invalid_params("phone_number must be 10 digits"));
        }

        let city = input.get("city").and_then(|v| v.as_str()).map(String::from);
        let estimated_gold = input.get("estimated_gold_weight").and_then(|v| v.as_f64());
        let notes = input
            .get("notes")
            .and_then(|v| v.as_str())
            .map(String::from);
        let interest_str = input
            .get("interest_level")
            .and_then(|v| v.as_str())
            .unwrap_or("Medium");

        let interest_level = match interest_str.to_lowercase().as_str() {
            "high" => InterestLevel::High,
            "low" => InterestLevel::Low,
            _ => InterestLevel::Medium,
        };

        if let Some(ref crm) = self.crm {
            let lead = CrmLead {
                id: None,
                name: name.to_string(),
                phone: phone.to_string(),
                email: None,
                city,
                source: LeadSource::VoiceAgent,
                interest_level,
                estimated_gold_grams: estimated_gold,
                current_lender: None,
                notes,
                assigned_to: None,
                status: LeadStatus::New,
            };

            match crm.create_lead(lead).await {
                Ok(lead_id) => {
                    let result = json!({
                        "success": true,
                        "lead_id": lead_id,
                        "customer_name": name,
                        "phone_number": phone,
                        "city": input.get("city").and_then(|v| v.as_str()),
                        "interest_level": interest_str,
                        "estimated_gold_weight": estimated_gold,
                        "created_at": Utc::now().to_rfc3339(),
                        "crm_integrated": true,
                        "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
                    });
                    return Ok(ToolOutput::json(result));
                }
                Err(e) => {
                    tracing::warn!("CRM integration failed, falling back to local: {}", e);
                }
            }
        }

        let lead_id = format!("GL{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

        let result = json!({
            "success": true,
            "lead_id": lead_id,
            "customer_name": name,
            "phone_number": phone,
            "city": input.get("city").and_then(|v| v.as_str()),
            "preferred_branch": input.get("preferred_branch").and_then(|v| v.as_str()),
            "estimated_gold_weight": estimated_gold,
            "interest_level": interest_str,
            "notes": input.get("notes").and_then(|v| v.as_str()),
            "created_at": Utc::now().to_rfc3339(),
            "crm_integrated": false,
            "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        45
    }
}

impl Default for LeadCaptureTool {
    fn default() -> Self {
        Self::new()
    }
}
