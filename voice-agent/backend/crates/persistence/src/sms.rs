//! Simulated SMS service with ScyllaDB persistence
//!
//! This module provides SMS simulation - messages are NOT actually sent,
//! but are persisted to ScyllaDB for audit trail and testing.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::{ScyllaClient, PersistenceError};

/// SMS message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmsType {
    AppointmentConfirmation,
    AppointmentReminder,
    FollowUp,
    Welcome,
    Promotional,
    Otp,
}

impl SmsType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AppointmentConfirmation => "appointment_confirmation",
            Self::AppointmentReminder => "appointment_reminder",
            Self::FollowUp => "follow_up",
            Self::Welcome => "welcome",
            Self::Promotional => "promotional",
            Self::Otp => "otp",
        }
    }
}

/// SMS delivery status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmsStatus {
    Queued,
    SimulatedSent,
    Delivered,
    Failed,
}

impl SmsStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::SimulatedSent => "simulated_sent",
            Self::Delivered => "delivered",
            Self::Failed => "failed",
        }
    }
}

/// SMS message record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsMessage {
    pub message_id: Uuid,
    pub phone_number: String,
    pub session_id: Option<String>,
    pub message_text: String,
    pub message_type: SmsType,
    pub status: SmsStatus,
    pub created_at: DateTime<Utc>,
    pub sent_at: Option<DateTime<Utc>>,
    pub metadata: Option<serde_json::Value>,
}

/// Result of sending an SMS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsResult {
    pub message_id: Uuid,
    pub status: SmsStatus,
    pub sent_at: DateTime<Utc>,
    pub simulated: bool,
}

/// SMS service trait
#[async_trait]
pub trait SmsService: Send + Sync {
    async fn send_sms(
        &self,
        phone: &str,
        message: &str,
        msg_type: SmsType,
        session_id: Option<&str>,
    ) -> Result<SmsResult, PersistenceError>;

    async fn get_messages_for_phone(&self, phone: &str, limit: i32) -> Result<Vec<SmsMessage>, PersistenceError>;

    async fn get_message(&self, phone: &str, message_id: Uuid) -> Result<Option<SmsMessage>, PersistenceError>;
}

/// Simulated SMS service that persists to ScyllaDB
#[derive(Clone)]
pub struct SimulatedSmsService {
    client: ScyllaClient,
}

impl SimulatedSmsService {
    pub fn new(client: ScyllaClient) -> Self {
        Self { client }
    }

    /// Generate appointment confirmation message
    pub fn format_appointment_confirmation(
        customer_name: &str,
        date: &str,
        time: &str,
        branch_name: &str,
        branch_address: &str,
    ) -> String {
        format!(
            "Dear {}, your Kotak Gold Loan appointment is confirmed for {} at {}. \
             Branch: {}, {}. Please bring your gold and KYC documents. \
             For queries, call 1800-xxx-xxxx. - Kotak Mahindra Bank",
            customer_name, date, time, branch_name, branch_address
        )
    }

    /// Generate follow-up message
    pub fn format_follow_up(customer_name: &str) -> String {
        format!(
            "Dear {}, thank you for your interest in Kotak Gold Loan. \
             Get up to 75% of gold value at competitive rates. \
             Call 1800-xxx-xxxx or visit your nearest branch. - Kotak Mahindra Bank",
            customer_name
        )
    }
}

#[async_trait]
impl SmsService for SimulatedSmsService {
    async fn send_sms(
        &self,
        phone: &str,
        message: &str,
        msg_type: SmsType,
        session_id: Option<&str>,
    ) -> Result<SmsResult, PersistenceError> {
        let message_id = Uuid::new_v4();
        let now = Utc::now();

        // Persist to ScyllaDB (this is the "sending")
        let query = format!(
            "INSERT INTO {}.sms_messages (
                phone_number, message_id, session_id, message_text,
                message_type, status, created_at, sent_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                phone,
                message_id,
                session_id,
                message,
                msg_type.as_str(),
                SmsStatus::SimulatedSent.as_str(),
                now.timestamp_millis(),
                now.timestamp_millis(),
            ),
        ).await?;

        tracing::info!(
            phone = %phone,
            message_id = %message_id,
            msg_type = ?msg_type,
            "SMS simulated and persisted to ScyllaDB"
        );

        // Log the message content for debugging
        tracing::debug!(
            phone = %phone,
            message = %message,
            "SMS content (simulated)"
        );

        Ok(SmsResult {
            message_id,
            status: SmsStatus::SimulatedSent,
            sent_at: now,
            simulated: true,
        })
    }

    async fn get_messages_for_phone(&self, phone: &str, limit: i32) -> Result<Vec<SmsMessage>, PersistenceError> {
        let query = format!(
            "SELECT phone_number, message_id, session_id, message_text,
                    message_type, status, created_at, sent_at, metadata_json
             FROM {}.sms_messages WHERE phone_number = ? LIMIT ?",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (phone, limit))
            .await?;

        let mut messages = Vec::new();
        if let Some(rows) = result.rows {
            for row in rows {
                let (
                    phone_number,
                    message_id,
                    session_id,
                    message_text,
                    message_type,
                    status,
                    created_at,
                    sent_at,
                    metadata_json,
                ): (
                    String, Uuid, Option<String>, String,
                    String, String, i64, Option<i64>, Option<String>,
                ) = row.into_typed().map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                messages.push(SmsMessage {
                    message_id,
                    phone_number,
                    session_id,
                    message_text,
                    message_type: match message_type.as_str() {
                        "appointment_confirmation" => SmsType::AppointmentConfirmation,
                        "appointment_reminder" => SmsType::AppointmentReminder,
                        "follow_up" => SmsType::FollowUp,
                        "welcome" => SmsType::Welcome,
                        "promotional" => SmsType::Promotional,
                        "otp" => SmsType::Otp,
                        _ => SmsType::FollowUp,
                    },
                    status: match status.as_str() {
                        "queued" => SmsStatus::Queued,
                        "simulated_sent" => SmsStatus::SimulatedSent,
                        "delivered" => SmsStatus::Delivered,
                        "failed" => SmsStatus::Failed,
                        _ => SmsStatus::SimulatedSent,
                    },
                    created_at: DateTime::from_timestamp_millis(created_at).unwrap_or_else(Utc::now),
                    sent_at: sent_at.and_then(DateTime::from_timestamp_millis),
                    metadata: metadata_json.and_then(|s| serde_json::from_str(&s).ok()),
                });
            }
        }

        Ok(messages)
    }

    async fn get_message(&self, phone: &str, message_id: Uuid) -> Result<Option<SmsMessage>, PersistenceError> {
        let messages = self.get_messages_for_phone(phone, 100).await?;
        Ok(messages.into_iter().find(|m| m.message_id == message_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_appointment_confirmation() {
        let msg = SimulatedSmsService::format_appointment_confirmation(
            "Rahul",
            "2024-01-15",
            "10:00 AM",
            "Kotak Bank Andheri",
            "123 Link Road, Mumbai"
        );
        assert!(msg.contains("Rahul"));
        assert!(msg.contains("2024-01-15"));
        assert!(msg.contains("Kotak Bank Andheri"));
    }

    #[test]
    fn test_sms_type_as_str() {
        assert_eq!(SmsType::AppointmentConfirmation.as_str(), "appointment_confirmation");
        assert_eq!(SmsType::FollowUp.as_str(), "follow_up");
    }
}
