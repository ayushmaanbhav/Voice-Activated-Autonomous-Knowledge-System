//! Appointment persistence using ScyllaDB

use async_trait::async_trait;
use chrono::{DateTime, Utc, NaiveDate};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::{ScyllaClient, PersistenceError};

/// Appointment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppointmentStatus {
    Scheduled,
    Confirmed,
    Cancelled,
    Completed,
    NoShow,
}

impl AppointmentStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scheduled => "scheduled",
            Self::Confirmed => "confirmed",
            Self::Cancelled => "cancelled",
            Self::Completed => "completed",
            Self::NoShow => "no_show",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "scheduled" => Self::Scheduled,
            "confirmed" => Self::Confirmed,
            "cancelled" => Self::Cancelled,
            "completed" => Self::Completed,
            "no_show" => Self::NoShow,
            _ => Self::Scheduled,
        }
    }
}

/// Appointment data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Appointment {
    pub appointment_id: Uuid,
    pub session_id: Option<String>,
    pub customer_phone: String,
    pub customer_name: Option<String>,
    pub branch_id: String,
    pub branch_name: String,
    pub branch_address: String,
    pub appointment_date: NaiveDate,
    pub appointment_time: String,
    pub status: AppointmentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub confirmation_sms_id: Option<Uuid>,
    pub notes: Option<String>,
}

impl Appointment {
    pub fn new(
        customer_phone: &str,
        branch_id: &str,
        branch_name: &str,
        branch_address: &str,
        date: NaiveDate,
        time: &str,
    ) -> Self {
        let now = Utc::now();
        Self {
            appointment_id: Uuid::new_v4(),
            session_id: None,
            customer_phone: customer_phone.to_string(),
            customer_name: None,
            branch_id: branch_id.to_string(),
            branch_name: branch_name.to_string(),
            branch_address: branch_address.to_string(),
            appointment_date: date,
            appointment_time: time.to_string(),
            status: AppointmentStatus::Scheduled,
            created_at: now,
            updated_at: now,
            confirmation_sms_id: None,
            notes: None,
        }
    }
}

/// Appointment store trait
#[async_trait]
pub trait AppointmentStore: Send + Sync {
    async fn create(&self, appointment: &Appointment) -> Result<(), PersistenceError>;
    async fn get(&self, phone: &str, appointment_id: Uuid) -> Result<Option<Appointment>, PersistenceError>;
    async fn update_status(&self, phone: &str, appointment_id: Uuid, status: AppointmentStatus) -> Result<(), PersistenceError>;
    async fn set_confirmation_sms(&self, phone: &str, appointment_id: Uuid, sms_id: Uuid) -> Result<(), PersistenceError>;
    async fn list_for_customer(&self, phone: &str, limit: i32) -> Result<Vec<Appointment>, PersistenceError>;
    async fn list_for_date(&self, date: NaiveDate) -> Result<Vec<Appointment>, PersistenceError>;
}

/// ScyllaDB implementation of appointment store
#[derive(Clone)]
pub struct ScyllaAppointmentStore {
    client: ScyllaClient,
}

impl ScyllaAppointmentStore {
    pub fn new(client: ScyllaClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl AppointmentStore for ScyllaAppointmentStore {
    async fn create(&self, appointment: &Appointment) -> Result<(), PersistenceError> {
        let query = format!(
            "INSERT INTO {}.appointments (
                customer_phone, appointment_id, session_id, customer_name,
                branch_id, branch_name, branch_address,
                appointment_date, appointment_time, status,
                created_at, updated_at, confirmation_sms_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                &appointment.customer_phone,
                appointment.appointment_id,
                &appointment.session_id,
                &appointment.customer_name,
                &appointment.branch_id,
                &appointment.branch_name,
                &appointment.branch_address,
                appointment.appointment_date.to_string(),
                &appointment.appointment_time,
                appointment.status.as_str(),
                appointment.created_at.timestamp_millis(),
                appointment.updated_at.timestamp_millis(),
                appointment.confirmation_sms_id,
                &appointment.notes,
            ),
        ).await?;

        tracing::info!(
            appointment_id = %appointment.appointment_id,
            customer_phone = %appointment.customer_phone,
            branch = %appointment.branch_name,
            date = %appointment.appointment_date,
            "Appointment created in ScyllaDB"
        );

        Ok(())
    }

    async fn get(&self, phone: &str, appointment_id: Uuid) -> Result<Option<Appointment>, PersistenceError> {
        let query = format!(
            "SELECT customer_phone, appointment_id, session_id, customer_name,
                    branch_id, branch_name, branch_address,
                    appointment_date, appointment_time, status,
                    created_at, updated_at, confirmation_sms_id, notes
             FROM {}.appointments WHERE customer_phone = ? AND appointment_id = ?",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (phone, appointment_id))
            .await?;

        if let Some(rows) = result.rows {
            if let Some(row) = rows.into_iter().next() {
                return Ok(Some(self.row_to_appointment(row)?));
            }
        }

        Ok(None)
    }

    async fn update_status(&self, phone: &str, appointment_id: Uuid, status: AppointmentStatus) -> Result<(), PersistenceError> {
        let query = format!(
            "UPDATE {}.appointments SET status = ?, updated_at = ?
             WHERE customer_phone = ? AND appointment_id = ?",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (status.as_str(), Utc::now().timestamp_millis(), phone, appointment_id),
        ).await?;

        tracing::info!(
            appointment_id = %appointment_id,
            status = ?status,
            "Appointment status updated"
        );

        Ok(())
    }

    async fn set_confirmation_sms(&self, phone: &str, appointment_id: Uuid, sms_id: Uuid) -> Result<(), PersistenceError> {
        let query = format!(
            "UPDATE {}.appointments SET confirmation_sms_id = ?, updated_at = ?
             WHERE customer_phone = ? AND appointment_id = ?",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (sms_id, Utc::now().timestamp_millis(), phone, appointment_id),
        ).await?;

        Ok(())
    }

    async fn list_for_customer(&self, phone: &str, limit: i32) -> Result<Vec<Appointment>, PersistenceError> {
        let query = format!(
            "SELECT customer_phone, appointment_id, session_id, customer_name,
                    branch_id, branch_name, branch_address,
                    appointment_date, appointment_time, status,
                    created_at, updated_at, confirmation_sms_id, notes
             FROM {}.appointments WHERE customer_phone = ? LIMIT ?",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (phone, limit))
            .await?;

        let mut appointments = Vec::new();
        if let Some(rows) = result.rows {
            for row in rows {
                appointments.push(self.row_to_appointment(row)?);
            }
        }

        Ok(appointments)
    }

    async fn list_for_date(&self, _date: NaiveDate) -> Result<Vec<Appointment>, PersistenceError> {
        // Note: This would require a secondary index or materialized view in production
        // For now, return empty - would need ALLOW FILTERING or different partition key
        tracing::warn!("list_for_date requires secondary index - returning empty");
        Ok(Vec::new())
    }
}

impl ScyllaAppointmentStore {
    fn row_to_appointment(&self, row: scylla::frame::response::result::Row) -> Result<Appointment, PersistenceError> {
        let (
            customer_phone,
            appointment_id,
            session_id,
            customer_name,
            branch_id,
            branch_name,
            branch_address,
            appointment_date,
            appointment_time,
            status,
            created_at,
            updated_at,
            confirmation_sms_id,
            notes,
        ): (
            String, Uuid, Option<String>, Option<String>,
            String, String, String,
            String, String, String,
            i64, i64, Option<Uuid>, Option<String>,
        ) = row.into_typed().map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

        Ok(Appointment {
            appointment_id,
            session_id,
            customer_phone,
            customer_name,
            branch_id,
            branch_name,
            branch_address,
            appointment_date: NaiveDate::parse_from_str(&appointment_date, "%Y-%m-%d")
                .unwrap_or_else(|_| Utc::now().date_naive()),
            appointment_time,
            status: AppointmentStatus::from_str(&status),
            created_at: DateTime::from_timestamp_millis(created_at).unwrap_or_else(Utc::now),
            updated_at: DateTime::from_timestamp_millis(updated_at).unwrap_or_else(Utc::now),
            confirmation_sms_id,
            notes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_appointment_new() {
        let date = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        let apt = Appointment::new(
            "+919876543210",
            "branch-001",
            "Kotak Andheri",
            "123 Link Road",
            date,
            "10:00 AM"
        );

        assert_eq!(apt.customer_phone, "+919876543210");
        assert_eq!(apt.branch_id, "branch-001");
        assert_eq!(apt.status, AppointmentStatus::Scheduled);
    }

    #[test]
    fn test_status_conversion() {
        assert_eq!(AppointmentStatus::from_str("confirmed"), AppointmentStatus::Confirmed);
        assert_eq!(AppointmentStatus::Confirmed.as_str(), "confirmed");
    }
}
