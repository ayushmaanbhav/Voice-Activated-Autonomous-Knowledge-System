//! External System Integrations
//!
//! P0 FIX: Traits and stubs for CRM and Calendar integrations.
//! These will be implemented when actual systems are available.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Integration errors
#[derive(Error, Debug)]
pub enum IntegrationError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Authentication failed: {0}")]
    AuthFailed(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// P2 FIX: Convert IntegrationError to ToolError for unified error handling
impl From<IntegrationError> for crate::mcp::ToolError {
    fn from(err: IntegrationError) -> Self {
        match err {
            IntegrationError::NotFound(msg) => crate::mcp::ToolError::not_found(msg),
            IntegrationError::InvalidRequest(msg) => crate::mcp::ToolError::invalid_params(msg),
            IntegrationError::RateLimited => crate::mcp::ToolError::internal("Rate limited - please retry later"),
            _ => crate::mcp::ToolError::internal(err.to_string()),
        }
    }
}

// ============================================================================
// CRM Integration
// ============================================================================

/// Lead data for CRM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrmLead {
    /// Lead ID (assigned by CRM)
    pub id: Option<String>,
    /// Customer name
    pub name: String,
    /// Phone number
    pub phone: String,
    /// Email (optional)
    pub email: Option<String>,
    /// City
    pub city: Option<String>,
    /// Lead source
    pub source: LeadSource,
    /// Interest level
    pub interest_level: InterestLevel,
    /// Estimated gold weight
    pub estimated_gold_grams: Option<f64>,
    /// Current lender (if switching)
    pub current_lender: Option<String>,
    /// Notes from conversation
    pub notes: Option<String>,
    /// Assigned sales rep ID
    pub assigned_to: Option<String>,
    /// Lead status
    pub status: LeadStatus,
}

/// Lead source
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LeadSource {
    #[default]
    VoiceAgent,
    Website,
    Branch,
    Referral,
    Campaign,
}

/// Interest level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InterestLevel {
    High,
    #[default]
    Medium,
    Low,
}

/// Lead status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LeadStatus {
    #[default]
    New,
    Contacted,
    Qualified,
    Proposal,
    Negotiation,
    Won,
    Lost,
}

/// CRM integration trait
///
/// Implement this trait to integrate with your CRM system
/// (e.g., Salesforce, HubSpot, Zoho CRM).
#[async_trait]
pub trait CrmIntegration: Send + Sync {
    /// Create a new lead
    async fn create_lead(&self, lead: CrmLead) -> Result<String, IntegrationError>;

    /// Update an existing lead
    async fn update_lead(&self, id: &str, lead: CrmLead) -> Result<(), IntegrationError>;

    /// Get lead by ID
    async fn get_lead(&self, id: &str) -> Result<CrmLead, IntegrationError>;

    /// Search leads by phone number
    async fn find_by_phone(&self, phone: &str) -> Result<Vec<CrmLead>, IntegrationError>;

    /// Assign lead to sales rep
    async fn assign_lead(&self, lead_id: &str, rep_id: &str) -> Result<(), IntegrationError>;

    /// Add note to lead
    async fn add_note(&self, lead_id: &str, note: &str) -> Result<(), IntegrationError>;

    /// Update lead status
    async fn update_status(&self, lead_id: &str, status: LeadStatus) -> Result<(), IntegrationError>;
}

/// Stub CRM implementation for development/testing
///
/// Returns mock responses without connecting to a real CRM.
pub struct StubCrmIntegration;

impl StubCrmIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StubCrmIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CrmIntegration for StubCrmIntegration {
    async fn create_lead(&self, mut lead: CrmLead) -> Result<String, IntegrationError> {
        let id = format!("LEAD-{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());
        lead.id = Some(id.clone());
        tracing::info!(lead_id = %id, name = %lead.name, "Stub CRM: Created lead");
        Ok(id)
    }

    async fn update_lead(&self, id: &str, lead: CrmLead) -> Result<(), IntegrationError> {
        tracing::info!(lead_id = %id, name = %lead.name, "Stub CRM: Updated lead");
        Ok(())
    }

    async fn get_lead(&self, id: &str) -> Result<CrmLead, IntegrationError> {
        tracing::info!(lead_id = %id, "Stub CRM: Get lead");
        // Return a mock lead
        Ok(CrmLead {
            id: Some(id.to_string()),
            name: "Mock Customer".to_string(),
            phone: "9999999999".to_string(),
            email: None,
            city: Some("Mumbai".to_string()),
            source: LeadSource::VoiceAgent,
            interest_level: InterestLevel::Medium,
            estimated_gold_grams: Some(50.0),
            current_lender: None,
            notes: None,
            assigned_to: None,
            status: LeadStatus::New,
        })
    }

    async fn find_by_phone(&self, phone: &str) -> Result<Vec<CrmLead>, IntegrationError> {
        tracing::info!(phone = %phone, "Stub CRM: Find by phone");
        Ok(vec![])
    }

    async fn assign_lead(&self, lead_id: &str, rep_id: &str) -> Result<(), IntegrationError> {
        tracing::info!(lead_id = %lead_id, rep_id = %rep_id, "Stub CRM: Assigned lead");
        Ok(())
    }

    async fn add_note(&self, lead_id: &str, note: &str) -> Result<(), IntegrationError> {
        tracing::info!(lead_id = %lead_id, note_len = note.len(), "Stub CRM: Added note");
        Ok(())
    }

    async fn update_status(&self, lead_id: &str, status: LeadStatus) -> Result<(), IntegrationError> {
        tracing::info!(lead_id = %lead_id, status = ?status, "Stub CRM: Updated status");
        Ok(())
    }
}

// ============================================================================
// Calendar Integration
// ============================================================================

/// Appointment data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Appointment {
    /// Appointment ID
    pub id: Option<String>,
    /// Customer name
    pub customer_name: String,
    /// Customer phone
    pub customer_phone: String,
    /// Branch ID
    pub branch_id: String,
    /// Date (YYYY-MM-DD)
    pub date: String,
    /// Time slot (e.g., "10:00 AM")
    pub time_slot: String,
    /// Purpose of visit
    pub purpose: AppointmentPurpose,
    /// Additional notes
    pub notes: Option<String>,
    /// Status
    pub status: AppointmentStatus,
    /// Confirmation sent
    pub confirmation_sent: bool,
}

/// Appointment purpose
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AppointmentPurpose {
    #[default]
    NewGoldLoan,
    GoldLoanTransfer,
    TopUp,
    Closure,
    Consultation,
}

/// Appointment status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AppointmentStatus {
    #[default]
    Scheduled,
    Confirmed,
    InProgress,
    Completed,
    Cancelled,
    NoShow,
}

/// Available time slot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSlot {
    pub time: String,
    pub available: bool,
    pub remaining_capacity: u32,
}

/// Calendar integration trait
///
/// Implement this trait to integrate with your calendar/scheduling system.
#[async_trait]
pub trait CalendarIntegration: Send + Sync {
    /// Get available time slots for a branch on a date
    async fn get_available_slots(
        &self,
        branch_id: &str,
        date: &str,
    ) -> Result<Vec<TimeSlot>, IntegrationError>;

    /// Schedule an appointment
    async fn schedule_appointment(
        &self,
        appointment: Appointment,
    ) -> Result<String, IntegrationError>;

    /// Cancel an appointment
    async fn cancel_appointment(&self, id: &str) -> Result<(), IntegrationError>;

    /// Reschedule an appointment
    async fn reschedule_appointment(
        &self,
        id: &str,
        new_date: &str,
        new_time: &str,
    ) -> Result<(), IntegrationError>;

    /// Get appointment by ID
    async fn get_appointment(&self, id: &str) -> Result<Appointment, IntegrationError>;

    /// Send confirmation notification
    async fn send_confirmation(&self, id: &str) -> Result<(), IntegrationError>;
}

/// Stub calendar implementation for development/testing
pub struct StubCalendarIntegration;

impl StubCalendarIntegration {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StubCalendarIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CalendarIntegration for StubCalendarIntegration {
    async fn get_available_slots(
        &self,
        branch_id: &str,
        date: &str,
    ) -> Result<Vec<TimeSlot>, IntegrationError> {
        tracing::info!(branch_id = %branch_id, date = %date, "Stub Calendar: Get slots");

        // Return mock available slots
        Ok(vec![
            TimeSlot { time: "10:00 AM".to_string(), available: true, remaining_capacity: 3 },
            TimeSlot { time: "11:00 AM".to_string(), available: true, remaining_capacity: 2 },
            TimeSlot { time: "12:00 PM".to_string(), available: false, remaining_capacity: 0 },
            TimeSlot { time: "2:00 PM".to_string(), available: true, remaining_capacity: 4 },
            TimeSlot { time: "3:00 PM".to_string(), available: true, remaining_capacity: 3 },
            TimeSlot { time: "4:00 PM".to_string(), available: true, remaining_capacity: 5 },
            TimeSlot { time: "5:00 PM".to_string(), available: true, remaining_capacity: 2 },
        ])
    }

    async fn schedule_appointment(
        &self,
        mut appointment: Appointment,
    ) -> Result<String, IntegrationError> {
        let id = format!("APT-{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());
        appointment.id = Some(id.clone());
        tracing::info!(
            appointment_id = %id,
            customer = %appointment.customer_name,
            branch = %appointment.branch_id,
            date = %appointment.date,
            time = %appointment.time_slot,
            "Stub Calendar: Scheduled appointment"
        );
        Ok(id)
    }

    async fn cancel_appointment(&self, id: &str) -> Result<(), IntegrationError> {
        tracing::info!(appointment_id = %id, "Stub Calendar: Cancelled appointment");
        Ok(())
    }

    async fn reschedule_appointment(
        &self,
        id: &str,
        new_date: &str,
        new_time: &str,
    ) -> Result<(), IntegrationError> {
        tracing::info!(
            appointment_id = %id,
            new_date = %new_date,
            new_time = %new_time,
            "Stub Calendar: Rescheduled appointment"
        );
        Ok(())
    }

    async fn get_appointment(&self, id: &str) -> Result<Appointment, IntegrationError> {
        tracing::info!(appointment_id = %id, "Stub Calendar: Get appointment");
        Ok(Appointment {
            id: Some(id.to_string()),
            customer_name: "Mock Customer".to_string(),
            customer_phone: "9999999999".to_string(),
            branch_id: "KMBL001".to_string(),
            date: "2024-12-30".to_string(),
            time_slot: "10:00 AM".to_string(),
            purpose: AppointmentPurpose::NewGoldLoan,
            notes: None,
            status: AppointmentStatus::Scheduled,
            confirmation_sent: true,
        })
    }

    async fn send_confirmation(&self, id: &str) -> Result<(), IntegrationError> {
        tracing::info!(appointment_id = %id, "Stub Calendar: Sent confirmation");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_crm_create_lead() {
        let crm = StubCrmIntegration::new();
        let lead = CrmLead {
            id: None,
            name: "Test Customer".to_string(),
            phone: "9876543210".to_string(),
            email: None,
            city: Some("Mumbai".to_string()),
            source: LeadSource::VoiceAgent,
            interest_level: InterestLevel::High,
            estimated_gold_grams: Some(100.0),
            current_lender: Some("Muthoot".to_string()),
            notes: None,
            assigned_to: None,
            status: LeadStatus::New,
        };

        let id = crm.create_lead(lead).await.unwrap();
        assert!(id.starts_with("LEAD-"));
    }

    #[tokio::test]
    async fn test_stub_calendar_get_slots() {
        let calendar = StubCalendarIntegration::new();
        let slots = calendar.get_available_slots("KMBL001", "2024-12-30").await.unwrap();
        assert!(!slots.is_empty());
        assert!(slots.iter().any(|s| s.available));
    }

    #[tokio::test]
    async fn test_stub_calendar_schedule() {
        let calendar = StubCalendarIntegration::new();
        let appointment = Appointment {
            id: None,
            customer_name: "Test Customer".to_string(),
            customer_phone: "9876543210".to_string(),
            branch_id: "KMBL001".to_string(),
            date: "2024-12-30".to_string(),
            time_slot: "10:00 AM".to_string(),
            purpose: AppointmentPurpose::NewGoldLoan,
            notes: None,
            status: AppointmentStatus::Scheduled,
            confirmation_sent: false,
        };

        let id = calendar.schedule_appointment(appointment).await.unwrap();
        assert!(id.starts_with("APT-"));
    }
}
