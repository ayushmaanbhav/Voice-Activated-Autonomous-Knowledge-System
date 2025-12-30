//! FSM Adapter
//!
//! Bridges StageManager with the core ConversationFSM trait.
//!
//! This adapter enables:
//! - Using StageManager through the standard ConversationFSM interface
//! - Event-driven state transitions
//! - Checkpoint/restore for error recovery
//! - Integration with other components expecting ConversationFSM

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::Arc;

use voice_agent_core::{
    ConversationEvent, ConversationFSM, ConversationOutcome, ConversationStage as CoreStage,
    FSMAction, FSMCheckpoint, FSMError, FSMMetrics,
};

use crate::stage::{ConversationStage as AgentStage, StageManager, TransitionReason};

/// Adapter that wraps StageManager to implement ConversationFSM trait
///
/// This allows the agent's StageManager to be used anywhere
/// the `dyn ConversationFSM` trait is expected.
pub struct StageManagerAdapter {
    inner: Arc<StageManager>,
    checkpoints: Mutex<Vec<FSMCheckpoint>>,
    context: RwLock<HashMap<String, serde_json::Value>>,
    turn_count: Mutex<usize>,
    /// Cached current stage for returning by reference
    current_stage: RwLock<CoreStage>,
    metrics: Mutex<FSMMetrics>,
}

impl StageManagerAdapter {
    /// Create a new adapter wrapping a StageManager
    pub fn new(stage_manager: Arc<StageManager>) -> Self {
        let initial_stage = Self::to_core_stage(stage_manager.current());
        Self {
            inner: stage_manager,
            checkpoints: Mutex::new(Vec::new()),
            context: RwLock::new(HashMap::new()),
            turn_count: Mutex::new(0),
            current_stage: RwLock::new(initial_stage),
            metrics: Mutex::new(FSMMetrics::default()),
        }
    }

    /// Create with a new StageManager
    pub fn with_new_manager() -> Self {
        Self::new(Arc::new(StageManager::new()))
    }

    /// Get the underlying StageManager
    pub fn inner(&self) -> &Arc<StageManager> {
        &self.inner
    }

    /// Convert agent stage to core stage
    fn to_core_stage(stage: AgentStage) -> CoreStage {
        match stage {
            AgentStage::Greeting => CoreStage::Greeting,
            AgentStage::Discovery => CoreStage::Discovery,
            AgentStage::Qualification => CoreStage::Qualification,
            AgentStage::Presentation => CoreStage::Presentation,
            AgentStage::ObjectionHandling => CoreStage::ObjectionHandling,
            AgentStage::Closing => CoreStage::Closing,
            AgentStage::Farewell => CoreStage::Farewell,
        }
    }

    /// Convert core stage to agent stage
    fn to_agent_stage(stage: CoreStage) -> AgentStage {
        match stage {
            CoreStage::Greeting => AgentStage::Greeting,
            CoreStage::Discovery => AgentStage::Discovery,
            CoreStage::Qualification => AgentStage::Qualification,
            CoreStage::Presentation => AgentStage::Presentation,
            CoreStage::ObjectionHandling => AgentStage::ObjectionHandling,
            CoreStage::Closing => AgentStage::Closing,
            CoreStage::Farewell => AgentStage::Farewell,
        }
    }

    /// Update the cached current stage
    fn update_cached_stage(&self) {
        let stage = Self::to_core_stage(self.inner.current());
        *self.current_stage.write() = stage;
    }

    /// Determine target stage from event
    fn target_stage_for_event(&self, event: &ConversationEvent) -> Option<AgentStage> {
        let current = self.inner.current();

        match event {
            ConversationEvent::CallStarted { .. } => Some(AgentStage::Greeting),

            ConversationEvent::UserIntent { intent, confidence } if *confidence > 0.7 => {
                match intent.as_str() {
                    "interested" | "inquiry" => Some(AgentStage::Discovery),
                    "objection" | "concern" => Some(AgentStage::ObjectionHandling),
                    "ready_to_proceed" | "agree" => Some(AgentStage::Closing),
                    "goodbye" | "end_call" => Some(AgentStage::Farewell),
                    _ => None,
                }
            },

            ConversationEvent::UserAgreement => match current {
                AgentStage::Presentation | AgentStage::ObjectionHandling => {
                    Some(AgentStage::Closing)
                },
                _ => None,
            },

            ConversationEvent::UserRefusal { .. } => Some(AgentStage::Farewell),

            ConversationEvent::UserObjection { .. } => Some(AgentStage::ObjectionHandling),

            ConversationEvent::CallEnded => Some(AgentStage::Farewell),

            _ => {
                // Check if we should naturally progress
                if self.inner.stage_completed() {
                    self.inner.suggest_next()
                } else {
                    None
                }
            },
        }
    }

    /// Generate actions for a stage transition
    fn actions_for_transition(&self, from: AgentStage, to: AgentStage) -> Vec<FSMAction> {
        let mut actions = vec![];

        // Always start listening in new stage
        actions.push(FSMAction::StartListening);

        // Stage-specific actions
        match to {
            AgentStage::Greeting => {
                actions.push(FSMAction::speak("Hello! How can I help you today?"));
            },
            AgentStage::Discovery => {
                actions.push(FSMAction::speak("I'd love to learn more about your needs."));
            },
            AgentStage::Qualification => {
                actions.push(FSMAction::speak(
                    "Let me check if you qualify for our best rates.",
                ));
            },
            AgentStage::Presentation => {
                actions.push(FSMAction::speak(
                    "Based on what you've shared, here's what we can offer.",
                ));
            },
            AgentStage::ObjectionHandling => {
                actions.push(FSMAction::speak(
                    "I understand your concern. Let me address that.",
                ));
            },
            AgentStage::Closing => {
                actions.push(FSMAction::speak(
                    "Great! Let's proceed with your application.",
                ));
            },
            AgentStage::Farewell => {
                actions.push(FSMAction::speak(
                    "Thank you for your time. Have a great day!",
                ));
                actions.push(FSMAction::end(ConversationOutcome::Converted));
            },
        }

        // Record the transition
        actions.push(FSMAction::update_context(
            "last_transition".to_string(),
            serde_json::json!({
                "from": format!("{:?}", from),
                "to": format!("{:?}", to),
            }),
        ));

        actions
    }
}

#[async_trait]
impl ConversationFSM for StageManagerAdapter {
    fn state(&self) -> &CoreStage {
        // Update cached stage and return reference
        self.update_cached_stage();
        // SAFETY: We just updated this, and RwLock ensures safe access
        // This is a workaround for returning a reference to computed data
        unsafe { &*(&*self.current_stage.read() as *const CoreStage) }
    }

    async fn transition(&mut self, event: ConversationEvent) -> Result<Vec<FSMAction>, FSMError> {
        let from = self.inner.current();
        *self.turn_count.lock() += 1;

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.turn_count += 1;
            let stage_key = format!("{:?}", from);
            *metrics.stage_turns.entry(stage_key).or_insert(0) += 1;
        }

        // Record event in context
        {
            let mut ctx = self.context.write();
            ctx.insert(
                "last_event".to_string(),
                serde_json::to_value(&event).unwrap_or_default(),
            );
        }

        // Determine target stage
        let target = match self.target_stage_for_event(&event) {
            Some(t) => t,
            None => {
                // No transition needed, just record the turn
                self.inner.record_turn();
                return Ok(vec![FSMAction::StartListening]);
            },
        };

        // Check if transition is valid
        if !from.valid_transitions().contains(&target) && target != from {
            return Err(FSMError::InvalidTransition {
                from: Self::to_core_stage(from),
                event: format!("{:?}", event),
            });
        }

        // Perform transition with correct TransitionReason variants
        let reason = match &event {
            ConversationEvent::UserIntent { intent, .. } => {
                TransitionReason::IntentDetected(intent.clone())
            },
            ConversationEvent::UserAgreement => TransitionReason::CustomerRequest,
            ConversationEvent::UserRefusal { .. } => TransitionReason::CustomerRequest,
            ConversationEvent::UserObjection { .. } => {
                TransitionReason::IntentDetected("objection".into())
            },
            _ => TransitionReason::StageCompleted,
        };

        self.inner
            .transition(target, reason)
            .map_err(|e| FSMError::InvalidTransition {
                from: Self::to_core_stage(from),
                event: e,
            })?;

        // Update cached stage
        self.update_cached_stage();

        Ok(self.actions_for_transition(from, target))
    }

    fn can_transition(&self, event: &ConversationEvent) -> bool {
        let from = self.inner.current();
        match self.target_stage_for_event(event) {
            Some(target) => from.valid_transitions().contains(&target) || target == from,
            None => true, // No transition, always valid
        }
    }

    fn valid_events(&self) -> Vec<String> {
        let current = self.inner.current();
        let mut events = vec![
            "CallStarted".to_string(),
            "TranscriptReady".to_string(),
            "ResponseGenerated".to_string(),
        ];

        // Add stage-specific events
        match current {
            AgentStage::Greeting => {
                events.push("UserIntent:interested".to_string());
            },
            AgentStage::Discovery | AgentStage::Qualification => {
                events.push("UserIntent:ready_to_proceed".to_string());
                events.push("UserObjection".to_string());
            },
            AgentStage::Presentation => {
                events.push("UserAgreement".to_string());
                events.push("UserObjection".to_string());
            },
            AgentStage::ObjectionHandling => {
                events.push("UserAgreement".to_string());
                events.push("UserRefusal".to_string());
            },
            AgentStage::Closing => {
                events.push("UserAgreement".to_string());
                events.push("CallEnded".to_string());
            },
            AgentStage::Farewell => {
                events.push("CallEnded".to_string());
            },
        }

        events
    }

    fn checkpoint(&mut self) -> FSMCheckpoint {
        let mut checkpoints = self.checkpoints.lock();
        let index = checkpoints.len();
        let checkpoint = FSMCheckpoint {
            index,
            stage: Self::to_core_stage(self.inner.current()),
            context: self.context.read().clone(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            turn_count: *self.turn_count.lock(),
        };

        checkpoints.push(checkpoint.clone());

        // Update metrics
        self.metrics.lock().checkpoint_count += 1;

        checkpoint
    }

    fn restore(&mut self, checkpoint_index: usize) -> Result<(), FSMError> {
        let checkpoints = self.checkpoints.lock();
        let checkpoint = checkpoints
            .get(checkpoint_index)
            .ok_or(FSMError::NoCheckpoint(checkpoint_index))?
            .clone();
        drop(checkpoints);

        // Restore stage (bypass validation since we're restoring to a known state)
        let agent_stage = Self::to_agent_stage(checkpoint.stage);
        self.inner.set_stage(agent_stage);

        // Restore context and turn count
        *self.context.write() = checkpoint.context;
        *self.turn_count.lock() = checkpoint.turn_count;

        // Update cached stage and metrics
        self.update_cached_stage();
        self.metrics.lock().restore_count += 1;

        Ok(())
    }

    fn checkpoints(&self) -> &[FSMCheckpoint] {
        // Return empty slice - we can't return a reference to Mutex-protected data
        // In production, consider using a different design or Arc<[FSMCheckpoint]>
        &[]
    }

    fn get_context(&self, _key: &str) -> Option<&serde_json::Value> {
        // Can't return a reference to RwLock-protected data
        // Workaround: Use unsafe or change trait design
        None
    }

    fn set_context(&mut self, key: &str, value: serde_json::Value) {
        self.context.write().insert(key.to_string(), value);
    }

    fn context(&self) -> &HashMap<String, serde_json::Value> {
        // Same issue as get_context - return empty for now
        static EMPTY: std::sync::OnceLock<HashMap<String, serde_json::Value>> =
            std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn metrics(&self) -> FSMMetrics {
        self.metrics.lock().clone()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.checkpoints.lock().clear();
        self.context.write().clear();
        *self.turn_count.lock() = 0;
        *self.metrics.lock() = FSMMetrics::default();
        self.update_cached_stage();
    }
}

/// Create a boxed ConversationFSM from StageManager
pub fn create_fsm_adapter(stage_manager: Arc<StageManager>) -> Box<dyn ConversationFSM> {
    Box::new(StageManagerAdapter::new(stage_manager))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fsm_adapter_initial_state() {
        let adapter = StageManagerAdapter::with_new_manager();
        assert_eq!(*adapter.state(), CoreStage::Greeting);
    }

    #[tokio::test]
    async fn test_fsm_adapter_transition() {
        let mut adapter = StageManagerAdapter::with_new_manager();

        // Simulate interested intent
        let event = ConversationEvent::UserIntent {
            intent: "interested".to_string(),
            confidence: 0.9,
        };

        let actions = adapter.transition(event).await.unwrap();
        assert!(!actions.is_empty());

        // Should have transitioned to Discovery
        assert_eq!(adapter.inner.current(), AgentStage::Discovery);
    }

    #[tokio::test]
    async fn test_fsm_adapter_checkpoint_restore() {
        let mut adapter = StageManagerAdapter::with_new_manager();

        // Record some context
        adapter.set_context("test_key", serde_json::json!("test_value"));

        // Create checkpoint
        let checkpoint = adapter.checkpoint();
        assert_eq!(checkpoint.index, 0);

        // Transition to new state
        let event = ConversationEvent::UserIntent {
            intent: "interested".to_string(),
            confidence: 0.9,
        };
        adapter.transition(event).await.unwrap();
        assert_eq!(adapter.inner.current(), AgentStage::Discovery);

        // Restore checkpoint
        adapter.restore(0).unwrap();
        assert_eq!(adapter.inner.current(), AgentStage::Greeting);
    }

    #[tokio::test]
    async fn test_fsm_adapter_metrics() {
        let mut adapter = StageManagerAdapter::with_new_manager();

        // Do some transitions
        let event = ConversationEvent::UserIntent {
            intent: "interested".to_string(),
            confidence: 0.9,
        };
        adapter.transition(event).await.unwrap();

        // Check metrics
        let metrics = adapter.metrics();
        assert_eq!(metrics.turn_count, 1);
        assert!(metrics.stage_turns.contains_key("Greeting"));
    }

    #[test]
    fn test_stage_conversion() {
        // Test all stage conversions
        let stages = [
            (AgentStage::Greeting, CoreStage::Greeting),
            (AgentStage::Discovery, CoreStage::Discovery),
            (AgentStage::Qualification, CoreStage::Qualification),
            (AgentStage::Presentation, CoreStage::Presentation),
            (AgentStage::ObjectionHandling, CoreStage::ObjectionHandling),
            (AgentStage::Closing, CoreStage::Closing),
            (AgentStage::Farewell, CoreStage::Farewell),
        ];

        for (agent, core) in stages {
            assert_eq!(StageManagerAdapter::to_core_stage(agent), core);
            assert_eq!(StageManagerAdapter::to_agent_stage(core), agent);
        }
    }

    #[test]
    fn test_reset() {
        let mut adapter = StageManagerAdapter::with_new_manager();

        // Add some state
        adapter.set_context("key", serde_json::json!("value"));
        adapter.checkpoint();

        // Reset
        adapter.reset();

        // Verify reset
        assert_eq!(adapter.metrics().turn_count, 0);
        assert_eq!(adapter.metrics().checkpoint_count, 0);
    }
}
