//! Session Management
//!
//! Manages voice agent sessions.
//!
//! ## P1 FIX: Session Store Abstraction
//!
//! The session management uses a trait-based abstraction for storage,
//! allowing different backends to be used.
//!
//! - `InMemorySessionStore` - Default, uses HashMap
//! - `ScyllaSessionStore` - Production persistence using ScyllaDB
//!
//! P3-1 FIX: Removed deprecated RedisSessionStore stub.
//! Use ScyllaSessionStore for distributed session persistence.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::watch;

use voice_agent_agent::{AgentConfig, DomainAgent};

use crate::ServerError;

/// P1 FIX: Session metadata for Redis storage
///
/// Contains serializable session information that can be stored in Redis.
/// The full agent state is kept in-memory with session affinity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionMetadata {
    /// Session ID
    pub id: String,
    /// Creation timestamp (Unix epoch milliseconds)
    pub created_at_ms: u64,
    /// Last activity timestamp (Unix epoch milliseconds)
    pub last_activity_ms: u64,
    /// Is session active
    pub active: bool,
    /// Current conversation stage
    pub stage: String,
    /// Number of turns in conversation
    pub turn_count: usize,
    /// Instance ID that owns this session (for affinity)
    pub instance_id: Option<String>,
}

/// P2 FIX: Session data for recovery (matches persistence layer)
#[derive(Debug, Clone)]
pub struct RecoverableSession {
    pub session_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub conversation_stage: String,
    pub turn_count: i32,
    pub language: String,
}

/// P1 FIX: Session store trait for pluggable backends
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Store session metadata
    async fn store_metadata(&self, session: &Session) -> Result<(), ServerError>;

    /// Get session metadata by ID
    async fn get_metadata(&self, id: &str) -> Result<Option<SessionMetadata>, ServerError>;

    /// Delete session metadata
    async fn delete_metadata(&self, id: &str) -> Result<(), ServerError>;

    /// List all session IDs
    async fn list_ids(&self) -> Result<Vec<String>, ServerError>;

    /// Update last activity timestamp
    async fn touch(&self, id: &str) -> Result<(), ServerError>;

    /// Check if this store supports distributed sessions
    fn is_distributed(&self) -> bool;

    /// P2 FIX: List active sessions for recovery on restart
    async fn list_active_sessions(
        &self,
        limit: i32,
    ) -> Result<Vec<RecoverableSession>, ServerError>;
}

/// P1 FIX: In-memory session store (default)
///
/// This is the current implementation - sessions are stored in memory
/// with no persistence across restarts.
#[derive(Default)]
pub struct InMemorySessionStore {
    metadata: RwLock<HashMap<String, SessionMetadata>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn store_metadata(&self, session: &Session) -> Result<(), ServerError> {
        let metadata = SessionMetadata {
            id: session.id.clone(),
            created_at_ms: session.created_at.elapsed().as_millis() as u64,
            last_activity_ms: session.last_activity.read().elapsed().as_millis() as u64,
            active: *session.active.read(),
            stage: session.agent.stage().display_name().to_string(),
            turn_count: session.agent.conversation().turn_count(),
            instance_id: None,
        };
        self.metadata.write().insert(session.id.clone(), metadata);
        Ok(())
    }

    async fn get_metadata(&self, id: &str) -> Result<Option<SessionMetadata>, ServerError> {
        Ok(self.metadata.read().get(id).cloned())
    }

    async fn delete_metadata(&self, id: &str) -> Result<(), ServerError> {
        self.metadata.write().remove(id);
        Ok(())
    }

    async fn list_ids(&self) -> Result<Vec<String>, ServerError> {
        Ok(self.metadata.read().keys().cloned().collect())
    }

    async fn touch(&self, id: &str) -> Result<(), ServerError> {
        if let Some(meta) = self.metadata.write().get_mut(id) {
            meta.last_activity_ms = 0; // Would need system time for real timestamp
        }
        Ok(())
    }

    fn is_distributed(&self) -> bool {
        false
    }

    async fn list_active_sessions(
        &self,
        _limit: i32,
    ) -> Result<Vec<RecoverableSession>, ServerError> {
        // In-memory sessions don't survive restarts, so nothing to recover
        Ok(Vec::new())
    }
}

// P3-1 FIX: Removed deprecated RedisSessionStore stub.
// Use ScyllaSessionStore for distributed session persistence.

/// P1 FIX: ScyllaDB session store for production persistence
///
/// Uses the voice-agent-persistence crate for durable session storage.
/// Sessions are persisted to ScyllaDB and survive server restarts.
pub struct ScyllaSessionStore {
    store: voice_agent_persistence::ScyllaSessionStore,
    instance_id: String,
}

impl ScyllaSessionStore {
    /// Create a new ScyllaDB session store
    pub fn new(store: voice_agent_persistence::ScyllaSessionStore) -> Self {
        Self {
            store,
            instance_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Create with a specific instance ID (for session affinity)
    pub fn with_instance_id(
        store: voice_agent_persistence::ScyllaSessionStore,
        instance_id: String,
    ) -> Self {
        Self { store, instance_id }
    }

    /// Get the instance ID
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }
}

#[async_trait]
impl SessionStore for ScyllaSessionStore {
    async fn store_metadata(&self, session: &Session) -> Result<(), ServerError> {
        use chrono::Utc;
        use voice_agent_persistence::sessions::{
            SessionData, SessionStore as PersistenceSessionStore,
        };

        let now = Utc::now();
        let expires_at = now + chrono::Duration::hours(1);

        // Get memory context from agent if available
        let memory_json = serde_json::to_string(&session.agent.conversation().get_context()).ok();

        let data = SessionData {
            session_id: session.id.clone(),
            created_at: now,
            updated_at: now,
            expires_at,
            customer_phone: None, // Will be set when customer provides phone
            customer_name: None,
            customer_segment: None,
            language: session.agent.config().language.clone(),
            conversation_stage: session.agent.stage().display_name().to_string(),
            turn_count: session.agent.conversation().turn_count() as i32,
            memory_json,
            metadata_json: Some(
                serde_json::json!({
                    "instance_id": self.instance_id
                })
                .to_string(),
            ),
        };

        self.store
            .create(&data)
            .await
            .map_err(|e| ServerError::Session(format!("ScyllaDB error: {}", e)))?;

        tracing::debug!(
            session_id = %session.id,
            stage = %data.conversation_stage,
            "Session persisted to ScyllaDB"
        );

        Ok(())
    }

    async fn get_metadata(&self, id: &str) -> Result<Option<SessionMetadata>, ServerError> {
        use voice_agent_persistence::sessions::SessionStore as PersistenceSessionStore;

        match self.store.get(id).await {
            Ok(Some(data)) => {
                // Extract instance_id from metadata_json if present
                let instance_id = data
                    .metadata_json
                    .as_ref()
                    .and_then(|json| serde_json::from_str::<serde_json::Value>(json).ok())
                    .and_then(|v| {
                        v.get("instance_id")
                            .and_then(|i| i.as_str())
                            .map(String::from)
                    });

                Ok(Some(SessionMetadata {
                    id: data.session_id,
                    created_at_ms: data.created_at.timestamp_millis() as u64,
                    last_activity_ms: data.updated_at.timestamp_millis() as u64,
                    active: data.expires_at > chrono::Utc::now(),
                    stage: data.conversation_stage,
                    turn_count: data.turn_count as usize,
                    instance_id,
                }))
            },
            Ok(None) => Ok(None),
            Err(e) => Err(ServerError::Session(format!("ScyllaDB error: {}", e))),
        }
    }

    async fn delete_metadata(&self, id: &str) -> Result<(), ServerError> {
        use voice_agent_persistence::sessions::SessionStore as PersistenceSessionStore;

        self.store
            .delete(id)
            .await
            .map_err(|e| ServerError::Session(format!("ScyllaDB error: {}", e)))?;
        tracing::debug!(session_id = %id, "Session deleted from ScyllaDB");
        Ok(())
    }

    async fn list_ids(&self) -> Result<Vec<String>, ServerError> {
        use voice_agent_persistence::sessions::SessionStore as PersistenceSessionStore;

        // P2-3 FIX: Actually list sessions from ScyllaDB
        let sessions = self
            .store
            .list_active(100)
            .await
            .map_err(|e| ServerError::Session(format!("ScyllaDB list error: {}", e)))?;

        Ok(sessions.into_iter().map(|s| s.session_id).collect())
    }

    async fn touch(&self, id: &str) -> Result<(), ServerError> {
        use voice_agent_persistence::sessions::SessionStore as PersistenceSessionStore;

        self.store
            .touch(id)
            .await
            .map_err(|e| ServerError::Session(format!("ScyllaDB error: {}", e)))?;
        Ok(())
    }

    fn is_distributed(&self) -> bool {
        true
    }

    async fn list_active_sessions(
        &self,
        limit: i32,
    ) -> Result<Vec<RecoverableSession>, ServerError> {
        use voice_agent_persistence::sessions::SessionStore as PersistenceSessionStore;

        let sessions = self
            .store
            .list_active(limit)
            .await
            .map_err(|e| ServerError::Session(format!("ScyllaDB list error: {}", e)))?;

        Ok(sessions
            .into_iter()
            .map(|s| RecoverableSession {
                session_id: s.session_id,
                created_at: s.created_at,
                expires_at: s.expires_at,
                conversation_stage: s.conversation_stage,
                turn_count: s.turn_count,
                language: s.language,
            })
            .collect())
    }
}

/// Session state
pub struct Session {
    /// Session ID
    pub id: String,
    /// Agent instance
    pub agent: Arc<DomainAgent>,
    /// Creation time
    pub created_at: Instant,
    /// Last activity
    pub last_activity: RwLock<Instant>,
    /// Is active
    pub active: RwLock<bool>,
    #[cfg(feature = "webrtc")]
    webrtc: RwLock<Option<crate::webrtc::WebRtcSession>>,
}

impl Session {
    /// Create a new session
    pub fn new(id: impl Into<String>, config: AgentConfig) -> Self {
        let id = id.into();
        Self {
            agent: Arc::new(DomainAgent::new(&id, config)),
            id,
            created_at: Instant::now(),
            last_activity: RwLock::new(Instant::now()),
            active: RwLock::new(true),
            #[cfg(feature = "webrtc")]
            webrtc: RwLock::new(None),
        }
    }

    /// Create a new session with vector store for RAG
    pub fn with_vector_store(
        id: impl Into<String>,
        config: AgentConfig,
        vector_store: Arc<voice_agent_rag::VectorStore>,
    ) -> Self {
        let id = id.into();
        let agent = DomainAgent::new(&id, config).with_vector_store(vector_store);
        Self {
            agent: Arc::new(agent),
            id,
            created_at: Instant::now(),
            last_activity: RwLock::new(Instant::now()),
            active: RwLock::new(true),
            #[cfg(feature = "webrtc")]
            webrtc: RwLock::new(None),
        }
    }

    /// Create a new session with full integration (RAG + persistence-wired tools)
    pub fn with_full_integration(
        id: impl Into<String>,
        config: AgentConfig,
        vector_store: Option<Arc<voice_agent_rag::VectorStore>>,
        tools: Arc<voice_agent_tools::ToolRegistry>,
    ) -> Self {
        let id = id.into();
        let mut agent = DomainAgent::new(&id, config).with_tools(tools);
        if let Some(vs) = vector_store {
            agent = agent.with_vector_store(vs);
        }
        Self {
            agent: Arc::new(agent),
            id,
            created_at: Instant::now(),
            last_activity: RwLock::new(Instant::now()),
            active: RwLock::new(true),
            #[cfg(feature = "webrtc")]
            webrtc: RwLock::new(None),
        }
    }

    #[cfg(feature = "webrtc")]
    pub fn set_webrtc_transport(&self, session: crate::webrtc::WebRtcSession) {
        *self.webrtc.write() = Some(session);
    }

    #[cfg(feature = "webrtc")]
    pub fn get_webrtc_transport(
        &self,
    ) -> Option<std::sync::Arc<tokio::sync::RwLock<voice_agent_transport::WebRtcTransport>>> {
        self.webrtc.read().as_ref().map(|s| s.transport.clone())
    }

    #[cfg(feature = "webrtc")]
    pub fn has_webrtc(&self) -> bool {
        self.webrtc.read().is_some()
    }

    /// Update last activity
    pub fn touch(&self) {
        *self.last_activity.write() = Instant::now();
    }

    /// Check if session is expired
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_activity.read().elapsed() > timeout
    }

    /// Close session
    pub fn close(&self) {
        *self.active.write() = false;
    }

    /// Is session active
    pub fn is_active(&self) -> bool {
        *self.active.read()
    }
}

/// Session manager
pub struct SessionManager {
    sessions: RwLock<HashMap<String, Arc<Session>>>,
    max_sessions: usize,
    session_timeout: Duration,
    /// P2 FIX: Cleanup interval for passive session cleanup
    cleanup_interval: Duration,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            max_sessions,
            session_timeout: Duration::from_secs(3600), // 1 hour
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }

    /// Create a new session manager with custom timeout and cleanup interval
    pub fn with_config(
        max_sessions: usize,
        session_timeout: Duration,
        cleanup_interval: Duration,
    ) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            max_sessions,
            session_timeout,
            cleanup_interval,
        }
    }

    /// P2 FIX: Start a background task that periodically cleans up expired sessions.
    ///
    /// Returns a shutdown sender that can be used to stop the cleanup task.
    /// The task runs every `cleanup_interval` and removes sessions that have
    /// exceeded `session_timeout` since their last activity.
    pub fn start_cleanup_task(self: &Arc<Self>) -> watch::Sender<bool> {
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
        let manager = Arc::clone(self);
        let interval = manager.cleanup_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            interval_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        let before = manager.count();
                        manager.cleanup_expired();
                        let after = manager.count();
                        if before != after {
                            tracing::info!(
                                "Session cleanup: removed {} expired sessions ({} remaining)",
                                before - after,
                                after
                            );
                        }
                    }
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            tracing::info!("Session cleanup task shutting down");
                            break;
                        }
                    }
                }
            }
        });

        shutdown_tx
    }

    /// Create a new session
    pub fn create(&self, config: AgentConfig) -> Result<Arc<Session>, ServerError> {
        self.create_with_vector_store(config, None)
    }

    /// P0 FIX: Create a new session with optional vector store for RAG
    pub fn create_with_vector_store(
        &self,
        config: AgentConfig,
        vector_store: Option<Arc<voice_agent_rag::VectorStore>>,
    ) -> Result<Arc<Session>, ServerError> {
        self.create_with_full_integration(config, vector_store, None)
    }

    /// P0 FIX: Create a new session with full integration (RAG + persistence-wired tools)
    ///
    /// This method creates a session with:
    /// - Vector store for RAG retrieval (optional)
    /// - Tool registry with persistence services wired (optional, uses default if None)
    ///
    /// Use this for production deployments where tool calls should persist to ScyllaDB.
    pub fn create_with_full_integration(
        &self,
        config: AgentConfig,
        vector_store: Option<Arc<voice_agent_rag::VectorStore>>,
        tools: Option<Arc<voice_agent_tools::ToolRegistry>>,
    ) -> Result<Arc<Session>, ServerError> {
        let mut sessions = self.sessions.write();

        // Check capacity
        if sessions.len() >= self.max_sessions {
            // Try to clean expired sessions
            self.cleanup_expired_internal(&mut sessions);

            if sessions.len() >= self.max_sessions {
                return Err(ServerError::Session("Max sessions reached".to_string()));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let rag_enabled = vector_store.is_some();
        let tools_wired = tools.is_some();

        let session = match (vector_store, tools) {
            (Some(vs), Some(t)) => {
                Arc::new(Session::with_full_integration(&id, config, Some(vs), t))
            },
            (Some(vs), None) => Arc::new(Session::with_vector_store(&id, config, vs)),
            (None, Some(t)) => Arc::new(Session::with_full_integration(&id, config, None, t)),
            (None, None) => Arc::new(Session::new(&id, config)),
        };
        sessions.insert(id.clone(), session.clone());

        tracing::info!(
            session_id = %id,
            rag_enabled = rag_enabled,
            tools_wired = tools_wired,
            "Created session"
        );

        Ok(session)
    }

    /// Get a session by ID
    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        let sessions = self.sessions.read();
        sessions.get(id).cloned()
    }

    /// Remove a session
    pub fn remove(&self, id: &str) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.remove(id) {
            session.close();
            tracing::info!("Removed session: {}", id);
        }
    }

    /// Get active session count
    pub fn count(&self) -> usize {
        self.sessions.read().len()
    }

    /// Cleanup expired sessions
    pub fn cleanup_expired(&self) {
        let mut sessions = self.sessions.write();
        self.cleanup_expired_internal(&mut sessions);
    }

    fn cleanup_expired_internal(&self, sessions: &mut HashMap<String, Arc<Session>>) {
        let timeout = self.session_timeout;
        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| s.is_expired(timeout))
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            if let Some(session) = sessions.remove(&id) {
                session.close();
                tracing::info!("Expired session: {}", id);
            }
        }
    }

    /// List all session IDs
    pub fn list(&self) -> Vec<String> {
        self.sessions.read().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();

        assert!(session.is_active());
        assert!(!session.is_expired(Duration::from_secs(60)));
    }

    #[test]
    fn test_session_get() {
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();
        let id = session.id.clone();

        let retrieved = manager.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);
    }

    #[test]
    fn test_session_remove() {
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();
        let id = session.id.clone();

        manager.remove(&id);
        assert!(manager.get(&id).is_none());
    }

    #[tokio::test]
    async fn test_in_memory_session_store() {
        let store = InMemorySessionStore::new();
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();

        // Store metadata
        store.store_metadata(&session).await.unwrap();

        // Retrieve metadata
        let metadata = store.get_metadata(&session.id).await.unwrap();
        assert!(metadata.is_some());
        let meta = metadata.unwrap();
        assert_eq!(meta.id, session.id);
        assert!(meta.active);

        // List IDs
        let ids = store.list_ids().await.unwrap();
        assert!(ids.contains(&session.id));

        // Delete
        store.delete_metadata(&session.id).await.unwrap();
        let metadata = store.get_metadata(&session.id).await.unwrap();
        assert!(metadata.is_none());

        // Check distributed flag
        assert!(!store.is_distributed());
    }

    // P3-1 FIX: Removed Redis session store tests (deprecated)
}
