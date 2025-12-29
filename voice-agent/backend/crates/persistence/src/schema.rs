//! ScyllaDB schema creation

use scylla::Session;
use crate::error::PersistenceError;

/// Create the keyspace if it doesn't exist
pub async fn create_keyspace(session: &Session, keyspace: &str, replication_factor: u8) -> Result<(), PersistenceError> {
    let query = format!(
        "CREATE KEYSPACE IF NOT EXISTS {} WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': {}}}",
        keyspace, replication_factor
    );

    session.query_unpaged(query, &[]).await
        .map_err(|e| PersistenceError::SchemaError(format!("Failed to create keyspace: {}", e)))?;

    Ok(())
}

/// Create all required tables
pub async fn create_tables(session: &Session, keyspace: &str) -> Result<(), PersistenceError> {
    // Sessions table
    let sessions_table = format!(r#"
        CREATE TABLE IF NOT EXISTS {}.sessions (
            session_id TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            expires_at TIMESTAMP,
            customer_phone TEXT,
            customer_name TEXT,
            customer_segment TEXT,
            language TEXT,
            conversation_stage TEXT,
            turn_count INT,
            memory_json TEXT,
            metadata_json TEXT,
            PRIMARY KEY (session_id)
        ) WITH default_time_to_live = 86400
    "#, keyspace);

    session.query_unpaged(sessions_table, &[]).await
        .map_err(|e| PersistenceError::SchemaError(format!("Failed to create sessions table: {}", e)))?;

    // SMS messages table (for simulation audit trail)
    let sms_table = format!(r#"
        CREATE TABLE IF NOT EXISTS {}.sms_messages (
            phone_number TEXT,
            message_id TIMEUUID,
            session_id TEXT,
            message_text TEXT,
            message_type TEXT,
            status TEXT,
            created_at TIMESTAMP,
            sent_at TIMESTAMP,
            metadata_json TEXT,
            PRIMARY KEY ((phone_number), message_id)
        ) WITH CLUSTERING ORDER BY (message_id DESC)
    "#, keyspace);

    session.query_unpaged(sms_table, &[]).await
        .map_err(|e| PersistenceError::SchemaError(format!("Failed to create sms_messages table: {}", e)))?;

    // Gold prices history table
    let gold_prices_table = format!(r#"
        CREATE TABLE IF NOT EXISTS {}.gold_prices (
            date DATE,
            hour INT,
            price_per_gram DOUBLE,
            price_24k DOUBLE,
            price_22k DOUBLE,
            price_18k DOUBLE,
            source TEXT,
            created_at TIMESTAMP,
            PRIMARY KEY ((date), hour)
        ) WITH CLUSTERING ORDER BY (hour DESC)
    "#, keyspace);

    session.query_unpaged(gold_prices_table, &[]).await
        .map_err(|e| PersistenceError::SchemaError(format!("Failed to create gold_prices table: {}", e)))?;

    // Latest gold price (single row)
    let gold_latest_table = format!(r#"
        CREATE TABLE IF NOT EXISTS {}.gold_price_latest (
            singleton INT,
            price_per_gram DOUBLE,
            price_24k DOUBLE,
            price_22k DOUBLE,
            price_18k DOUBLE,
            updated_at TIMESTAMP,
            source TEXT,
            PRIMARY KEY (singleton)
        )
    "#, keyspace);

    session.query_unpaged(gold_latest_table, &[]).await
        .map_err(|e| PersistenceError::SchemaError(format!("Failed to create gold_price_latest table: {}", e)))?;

    // Appointments table
    let appointments_table = format!(r#"
        CREATE TABLE IF NOT EXISTS {}.appointments (
            customer_phone TEXT,
            appointment_id TIMEUUID,
            session_id TEXT,
            customer_name TEXT,
            branch_id TEXT,
            branch_name TEXT,
            branch_address TEXT,
            appointment_date DATE,
            appointment_time TEXT,
            status TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            confirmation_sms_id TIMEUUID,
            notes TEXT,
            PRIMARY KEY ((customer_phone), appointment_id)
        ) WITH CLUSTERING ORDER BY (appointment_id DESC)
    "#, keyspace);

    session.query_unpaged(appointments_table, &[]).await
        .map_err(|e| PersistenceError::SchemaError(format!("Failed to create appointments table: {}", e)))?;

    tracing::info!("All tables created successfully");
    Ok(())
}
