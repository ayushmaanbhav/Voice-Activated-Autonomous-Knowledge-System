//! Session persistence using ScyllaDB

use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use crate::{ScyllaClient, PersistenceError};

/// Session data stored in ScyllaDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub customer_phone: Option<String>,
    pub customer_name: Option<String>,
    pub customer_segment: Option<String>,
    pub language: String,
    pub conversation_stage: String,
    pub turn_count: i32,
    pub memory_json: Option<String>,
    pub metadata_json: Option<String>,
}

impl SessionData {
    pub fn new(session_id: &str) -> Self {
        let now = Utc::now();
        Self {
            session_id: session_id.to_string(),
            created_at: now,
            updated_at: now,
            expires_at: now + Duration::hours(24),
            customer_phone: None,
            customer_name: None,
            customer_segment: None,
            language: "en".to_string(),
            conversation_stage: "greeting".to_string(),
            turn_count: 0,
            memory_json: None,
            metadata_json: None,
        }
    }
}

/// Session store trait for abstraction
#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn create(&self, session: &SessionData) -> Result<(), PersistenceError>;
    async fn get(&self, session_id: &str) -> Result<Option<SessionData>, PersistenceError>;
    async fn update(&self, session: &SessionData) -> Result<(), PersistenceError>;
    async fn delete(&self, session_id: &str) -> Result<(), PersistenceError>;
    async fn touch(&self, session_id: &str) -> Result<(), PersistenceError>;
    async fn list_active(&self, limit: i32) -> Result<Vec<SessionData>, PersistenceError>;
}

/// ScyllaDB implementation of session store
#[derive(Clone)]
pub struct ScyllaSessionStore {
    client: ScyllaClient,
}

impl ScyllaSessionStore {
    pub fn new(client: ScyllaClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl SessionStore for ScyllaSessionStore {
    async fn create(&self, session: &SessionData) -> Result<(), PersistenceError> {
        let query = format!(
            "INSERT INTO {}.sessions (
                session_id, created_at, updated_at, expires_at,
                customer_phone, customer_name, customer_segment,
                language, conversation_stage, turn_count,
                memory_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                &session.session_id,
                session.created_at.timestamp_millis(),
                session.updated_at.timestamp_millis(),
                session.expires_at.timestamp_millis(),
                &session.customer_phone,
                &session.customer_name,
                &session.customer_segment,
                &session.language,
                &session.conversation_stage,
                session.turn_count,
                &session.memory_json,
                &session.metadata_json,
            ),
        ).await?;

        tracing::debug!(session_id = %session.session_id, "Session created in ScyllaDB");
        Ok(())
    }

    async fn get(&self, session_id: &str) -> Result<Option<SessionData>, PersistenceError> {
        let query = format!(
            "SELECT session_id, created_at, updated_at, expires_at,
                    customer_phone, customer_name, customer_segment,
                    language, conversation_stage, turn_count,
                    memory_json, metadata_json
             FROM {}.sessions WHERE session_id = ?",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (session_id,))
            .await?;

        if let Some(rows) = result.rows {
            if let Some(row) = rows.into_iter().next() {
                let (
                    session_id,
                    created_at,
                    updated_at,
                    expires_at,
                    customer_phone,
                    customer_name,
                    customer_segment,
                    language,
                    conversation_stage,
                    turn_count,
                    memory_json,
                    metadata_json,
                ): (
                    String,
                    i64,
                    i64,
                    i64,
                    Option<String>,
                    Option<String>,
                    Option<String>,
                    String,
                    String,
                    i32,
                    Option<String>,
                    Option<String>,
                ) = row.into_typed().map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                return Ok(Some(SessionData {
                    session_id,
                    created_at: DateTime::from_timestamp_millis(created_at).unwrap_or_else(Utc::now),
                    updated_at: DateTime::from_timestamp_millis(updated_at).unwrap_or_else(Utc::now),
                    expires_at: DateTime::from_timestamp_millis(expires_at).unwrap_or_else(Utc::now),
                    customer_phone,
                    customer_name,
                    customer_segment,
                    language,
                    conversation_stage,
                    turn_count,
                    memory_json,
                    metadata_json,
                }));
            }
        }

        Ok(None)
    }

    async fn update(&self, session: &SessionData) -> Result<(), PersistenceError> {
        let query = format!(
            "UPDATE {}.sessions SET
                updated_at = ?,
                customer_phone = ?,
                customer_name = ?,
                customer_segment = ?,
                language = ?,
                conversation_stage = ?,
                turn_count = ?,
                memory_json = ?,
                metadata_json = ?
             WHERE session_id = ?",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(
            query,
            (
                Utc::now().timestamp_millis(),
                &session.customer_phone,
                &session.customer_name,
                &session.customer_segment,
                &session.language,
                &session.conversation_stage,
                session.turn_count,
                &session.memory_json,
                &session.metadata_json,
                &session.session_id,
            ),
        ).await?;

        tracing::debug!(session_id = %session.session_id, "Session updated in ScyllaDB");
        Ok(())
    }

    async fn delete(&self, session_id: &str) -> Result<(), PersistenceError> {
        let query = format!(
            "DELETE FROM {}.sessions WHERE session_id = ?",
            self.client.keyspace()
        );

        self.client.session().query_unpaged(query, (session_id,)).await?;
        tracing::debug!(session_id = %session_id, "Session deleted from ScyllaDB");
        Ok(())
    }

    async fn touch(&self, session_id: &str) -> Result<(), PersistenceError> {
        let query = format!(
            "UPDATE {}.sessions SET updated_at = ?, expires_at = ? WHERE session_id = ?",
            self.client.keyspace()
        );

        let now = Utc::now();
        let expires = now + Duration::hours(24);

        self.client.session().query_unpaged(
            query,
            (now.timestamp_millis(), expires.timestamp_millis(), session_id),
        ).await?;

        Ok(())
    }

    async fn list_active(&self, limit: i32) -> Result<Vec<SessionData>, PersistenceError> {
        // Note: This requires ALLOW FILTERING in production you'd use a secondary index
        let query = format!(
            "SELECT session_id, created_at, updated_at, expires_at,
                    customer_phone, customer_name, customer_segment,
                    language, conversation_stage, turn_count,
                    memory_json, metadata_json
             FROM {}.sessions LIMIT ?",
            self.client.keyspace()
        );

        let result = self.client.session()
            .query_unpaged(query, (limit,))
            .await?;

        let mut sessions = Vec::new();
        if let Some(rows) = result.rows {
            for row in rows {
                let (
                    session_id,
                    created_at,
                    updated_at,
                    expires_at,
                    customer_phone,
                    customer_name,
                    customer_segment,
                    language,
                    conversation_stage,
                    turn_count,
                    memory_json,
                    metadata_json,
                ): (
                    String, i64, i64, i64,
                    Option<String>, Option<String>, Option<String>,
                    String, String, i32,
                    Option<String>, Option<String>,
                ) = row.into_typed().map_err(|e| PersistenceError::InvalidData(e.to_string()))?;

                sessions.push(SessionData {
                    session_id,
                    created_at: DateTime::from_timestamp_millis(created_at).unwrap_or_else(Utc::now),
                    updated_at: DateTime::from_timestamp_millis(updated_at).unwrap_or_else(Utc::now),
                    expires_at: DateTime::from_timestamp_millis(expires_at).unwrap_or_else(Utc::now),
                    customer_phone,
                    customer_name,
                    customer_segment,
                    language,
                    conversation_stage,
                    turn_count,
                    memory_json,
                    metadata_json,
                });
            }
        }

        Ok(sessions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_data_new() {
        let session = SessionData::new("test-123");
        assert_eq!(session.session_id, "test-123");
        assert_eq!(session.language, "en");
        assert_eq!(session.conversation_stage, "greeting");
        assert_eq!(session.turn_count, 0);
    }
}
