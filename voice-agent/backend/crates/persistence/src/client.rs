//! ScyllaDB client and connection management

use scylla::{Session, SessionBuilder};
use std::sync::Arc;
use crate::error::PersistenceError;
use crate::schema;

/// ScyllaDB configuration
#[derive(Debug, Clone)]
pub struct ScyllaConfig {
    pub hosts: Vec<String>,
    pub keyspace: String,
    pub replication_factor: u8,
}

impl Default for ScyllaConfig {
    fn default() -> Self {
        Self {
            hosts: vec!["127.0.0.1:9042".to_string()],
            keyspace: "voice_agent".to_string(),
            replication_factor: 1,
        }
    }
}

/// ScyllaDB client wrapper
#[derive(Clone)]
pub struct ScyllaClient {
    session: Arc<Session>,
    config: ScyllaConfig,
}

impl ScyllaClient {
    /// Connect to ScyllaDB cluster
    pub async fn connect(config: ScyllaConfig) -> Result<Self, PersistenceError> {
        tracing::info!(hosts = ?config.hosts, keyspace = %config.keyspace, "Connecting to ScyllaDB");

        let session = SessionBuilder::new()
            .known_nodes(&config.hosts)
            .build()
            .await?;

        let client = Self {
            session: Arc::new(session),
            config,
        };

        Ok(client)
    }

    /// Ensure keyspace and tables exist
    pub async fn ensure_schema(&self) -> Result<(), PersistenceError> {
        schema::create_keyspace(&self.session, &self.config.keyspace, self.config.replication_factor).await?;
        schema::create_tables(&self.session, &self.config.keyspace).await?;
        tracing::info!(keyspace = %self.config.keyspace, "Schema ensured");
        Ok(())
    }

    /// Get the underlying session
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Get keyspace name
    pub fn keyspace(&self) -> &str {
        &self.config.keyspace
    }
}
