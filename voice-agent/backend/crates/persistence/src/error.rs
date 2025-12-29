//! Persistence error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PersistenceError {
    #[error("ScyllaDB connection error: {0}")]
    Connection(String),

    #[error("ScyllaDB query error: {0}")]
    Query(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Schema creation failed: {0}")]
    SchemaError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

impl From<scylla::transport::errors::NewSessionError> for PersistenceError {
    fn from(e: scylla::transport::errors::NewSessionError) -> Self {
        PersistenceError::Connection(e.to_string())
    }
}

impl From<scylla::transport::errors::QueryError> for PersistenceError {
    fn from(e: scylla::transport::errors::QueryError) -> Self {
        PersistenceError::Query(e.to_string())
    }
}
