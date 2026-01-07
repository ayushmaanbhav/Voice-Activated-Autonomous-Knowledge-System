//! Appointment Scheduler Tool
//!
//! Schedule branch visit appointments for gold valuation.

use async_trait::async_trait;
use chrono::{NaiveDate, Utc};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::integrations::{
    Appointment, AppointmentPurpose, AppointmentStatus, CalendarIntegration,
};
use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Appointment scheduler tool
pub struct AppointmentSchedulerTool {
    calendar: Option<Arc<dyn CalendarIntegration>>,
}

impl AppointmentSchedulerTool {
    pub fn new() -> Self {
        Self { calendar: None }
    }

    pub fn with_calendar(calendar: Arc<dyn CalendarIntegration>) -> Self {
        Self {
            calendar: Some(calendar),
        }
    }
}

#[async_trait]
impl Tool for AppointmentSchedulerTool {
    fn name(&self) -> &str {
        "schedule_appointment"
    }

    fn description(&self) -> &str {
        "Schedule a branch visit appointment for gold valuation"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "customer_name",
                    PropertySchema::string("Customer's name"),
                    true,
                )
                .property(
                    "phone_number",
                    PropertySchema::string("Contact number"),
                    true,
                )
                .property(
                    "branch_id",
                    PropertySchema::string("Branch ID or location"),
                    true,
                )
                .property(
                    "preferred_date",
                    PropertySchema::string("Preferred date (YYYY-MM-DD)"),
                    true,
                )
                .property(
                    "preferred_time",
                    PropertySchema::enum_type(
                        "Preferred time slot",
                        vec![
                            "10:00 AM".into(),
                            "11:00 AM".into(),
                            "12:00 PM".into(),
                            "2:00 PM".into(),
                            "3:00 PM".into(),
                            "4:00 PM".into(),
                            "5:00 PM".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "purpose",
                    PropertySchema::enum_type(
                        "Purpose of visit",
                        vec![
                            "New Gold Loan".into(),
                            "Gold Loan Transfer".into(),
                            "Top-up".into(),
                            "Closure".into(),
                        ],
                    ),
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

        let branch = input
            .get("branch_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("branch_id is required"))?;

        let date_str = input
            .get("preferred_date")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_date is required"))?;

        let parsed_date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d-%m-%Y"))
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d/%m/%Y"))
            .map_err(|_| {
                ToolError::invalid_params(
                    "preferred_date must be in format YYYY-MM-DD, DD-MM-YYYY, or DD/MM/YYYY",
                )
            })?;

        let today = Utc::now().date_naive();
        if parsed_date < today {
            return Err(ToolError::invalid_params(
                "preferred_date cannot be in the past",
            ));
        }

        let date = parsed_date.format("%Y-%m-%d").to_string();

        let time = input
            .get("preferred_time")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_time is required"))?;

        let purpose_str = input
            .get("purpose")
            .and_then(|v| v.as_str())
            .unwrap_or("New Gold Loan");

        let purpose_enum = match purpose_str {
            "Gold Loan Transfer" => AppointmentPurpose::GoldLoanTransfer,
            "Top-up" => AppointmentPurpose::TopUp,
            "Closure" => AppointmentPurpose::Closure,
            "Consultation" => AppointmentPurpose::Consultation,
            _ => AppointmentPurpose::NewGoldLoan,
        };

        if let Some(ref calendar) = self.calendar {
            let appointment = Appointment {
                id: None,
                customer_name: name.to_string(),
                customer_phone: phone.to_string(),
                branch_id: branch.to_string(),
                date: date.clone(),
                time_slot: time.to_string(),
                purpose: purpose_enum,
                notes: None,
                status: AppointmentStatus::Scheduled,
                confirmation_sent: false,
            };

            match calendar.schedule_appointment(appointment).await {
                Ok(appointment_id) => {
                    let confirmation_sent =
                        calendar.send_confirmation(&appointment_id).await.is_ok();

                    let result = json!({
                        "success": true,
                        "appointment_id": appointment_id,
                        "customer_name": name,
                        "phone_number": phone,
                        "branch_id": branch,
                        "date": date,
                        "time": time,
                        "purpose": purpose_str,
                        "confirmation_sent": confirmation_sent,
                        "calendar_integrated": true,
                        "status": "pending_confirmation",
                        "confirmation_method": "agent_will_call_to_confirm",
                        "next_action": "Agent will call customer to confirm appointment",
                        "message": if confirmation_sent {
                            format!(
                                "Appointment scheduled for {} on {} at {}. Confirmation sent to {}.",
                                name, date, time, phone
                            )
                        } else {
                            format!(
                                "Appointment scheduled for {} on {} at {}. Our team will call to confirm.",
                                name, date, time
                            )
                        }
                    });
                    return Ok(ToolOutput::json(result));
                }
                Err(e) => {
                    tracing::warn!("Calendar integration failed, falling back to local: {}", e);
                }
            }
        }

        let appointment_id = format!(
            "APT{}",
            uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
        );

        let result = json!({
            "success": true,
            "appointment_id": appointment_id,
            "customer_name": name,
            "phone_number": phone,
            "branch_id": branch,
            "date": date,
            "time": time,
            "purpose": purpose_str,
            "confirmation_sent": false,
            "calendar_integrated": false,
            "status": "pending_confirmation",
            "confirmation_method": "agent_will_call_to_confirm",
            "next_action": "Agent will call customer to confirm appointment",
            "message": format!(
                "Appointment scheduled for {} on {} at {}. Our team will call to confirm.",
                name, date, time
            )
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        60
    }
}

impl Default for AppointmentSchedulerTool {
    fn default() -> Self {
        Self::new()
    }
}
