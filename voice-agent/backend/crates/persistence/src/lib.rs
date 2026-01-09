//! ScyllaDB persistence layer for voice-agent-rust
//!
//! Provides persistent storage for:
//! - Sessions (replaces Redis stub)
//! - SMS messages (simulated, persisted for audit)
//! - Gold prices (simulated with realistic fluctuation)
//! - Appointments
//! - Audit logging (P0 FIX: RBI compliance)

pub mod appointments;
pub mod audit;
pub mod client;
pub mod error;
pub mod gold_price;
pub mod schema;
pub mod sessions;
pub mod sms;

pub use appointments::{Appointment, AppointmentStatus, AppointmentStore, ScyllaAppointmentStore};
pub use audit::{
    Actor, AuditEntry, AuditEventType, AuditLog, AuditLogger, AuditOutcome, AuditQuery,
    ScyllaAuditLog,
};
pub use client::{ScyllaClient, ScyllaConfig};
pub use error::PersistenceError;
// Asset price types (domain-agnostic) with legacy aliases
pub use gold_price::{
    AssetPrice, AssetPriceService, AssetVariant, SimulatedAssetPriceService,
    // Legacy aliases for backwards compatibility
    GoldPrice, GoldPriceService, GoldPurity, SimulatedGoldPriceService,
};
pub use sessions::{ScyllaSessionStore, SessionData, SessionStore};
pub use sms::{SimulatedSmsService, SmsMessage, SmsService, SmsStatus, SmsType};

/// Initialize the persistence layer with ScyllaDB
pub async fn init(config: ScyllaConfig) -> Result<PersistenceLayer, PersistenceError> {
    let client = ScyllaClient::connect(config).await?;
    client.ensure_schema().await?;

    Ok(PersistenceLayer {
        sessions: ScyllaSessionStore::new(client.clone()),
        sms: SimulatedSmsService::new(client.clone()),
        asset_price: SimulatedAssetPriceService::new(client.clone(), 7500.0),
        appointments: ScyllaAppointmentStore::new(client.clone()),
        audit: ScyllaAuditLog::new(client),
    })
}

/// Combined persistence layer with all services
pub struct PersistenceLayer {
    pub sessions: ScyllaSessionStore,
    pub sms: SimulatedSmsService,
    /// P16 FIX: Generic asset price service (gold_price alias kept for backwards compatibility)
    pub asset_price: SimulatedAssetPriceService,
    pub appointments: ScyllaAppointmentStore,
    /// P0 FIX: Audit logging for RBI compliance
    pub audit: ScyllaAuditLog,
}

impl PersistenceLayer {
    /// Legacy accessor for backwards compatibility
    #[inline]
    pub fn gold_price(&self) -> &SimulatedAssetPriceService {
        &self.asset_price
    }
}
