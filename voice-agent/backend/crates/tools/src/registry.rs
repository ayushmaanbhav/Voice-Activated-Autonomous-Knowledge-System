//! Tool Registry
//!
//! Manages tool registration, discovery, and execution.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use serde_json::Value;

use crate::mcp::{Tool, ToolSchema, ToolOutput, ToolError};

/// Default timeout for tool execution (30 seconds)
const DEFAULT_TOOL_TIMEOUT_SECS: u64 = 30;

/// Tool executor trait
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool by name
    async fn execute(&self, name: &str, arguments: Value) -> Result<ToolOutput, ToolError>;

    /// List available tools
    fn list_tools(&self) -> Vec<ToolSchema>;

    /// Get tool schema by name
    fn get_tool(&self, name: &str) -> Option<ToolSchema>;
}

/// Tool registry
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Register a boxed tool
    pub fn register_boxed(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Get tool by name
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Check if tool exists
    pub fn has(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Remove a tool
    pub fn remove(&mut self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.remove(name)
    }

    /// Get number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Get all tool names
    pub fn tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolExecutor for ToolRegistry {
    /// Execute a tool with timeout protection
    ///
    /// P1 FIX: Wraps tool execution in a timeout to prevent indefinite blocking.
    /// P5 FIX: Uses per-tool timeout instead of global default.
    async fn execute(&self, name: &str, arguments: Value) -> Result<ToolOutput, ToolError> {
        let tool = self.tools.get(name)
            .ok_or_else(|| ToolError::not_found(format!("Tool not found: {}", name)))?;

        // Validate input
        tool.validate(&arguments)?;

        // P5 FIX: Use per-tool timeout, falling back to default
        let timeout_secs = tool.timeout_secs();
        let timeout_duration = Duration::from_secs(timeout_secs);

        tracing::trace!(tool = name, timeout_secs = timeout_secs, "Executing tool with timeout");

        match tokio::time::timeout(timeout_duration, tool.execute(arguments)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(ToolError::timeout(name, timeout_secs)),
        }
    }

    fn list_tools(&self) -> Vec<ToolSchema> {
        self.tools.values()
            .map(|t| t.schema())
            .collect()
    }

    fn get_tool(&self, name: &str) -> Option<ToolSchema> {
        self.tools.get(name).map(|t| t.schema())
    }
}

/// Tool call result for conversation tracking
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Tool name
    pub name: String,
    /// Input arguments
    pub arguments: Value,
    /// Output result
    pub output: ToolOutput,
    /// Execution duration (ms)
    pub duration_ms: u64,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Tool call tracker
///
/// P2 FIX: Uses VecDeque for O(1) removal from front.
pub struct ToolCallTracker {
    calls: VecDeque<ToolCall>,
    max_history: usize,
}

impl ToolCallTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            calls: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Record a tool call
    pub fn record(&mut self, call: ToolCall) {
        if self.calls.len() >= self.max_history {
            self.calls.pop_front(); // P2 FIX: O(1) instead of O(n)
        }
        self.calls.push_back(call);
    }

    /// Get recent calls as a slice
    ///
    /// P2 FIX: Returns contiguous slice by ensuring make_contiguous.
    pub fn recent(&mut self, n: usize) -> &[ToolCall] {
        self.calls.make_contiguous();
        let (slice, _) = self.calls.as_slices();
        let start = slice.len().saturating_sub(n);
        &slice[start..]
    }

    /// Get all calls as a slice
    ///
    /// P2 FIX: Returns contiguous slice by ensuring make_contiguous.
    pub fn all(&mut self) -> &[ToolCall] {
        self.calls.make_contiguous();
        let (slice, _) = self.calls.as_slices();
        slice
    }

    /// Get calls by tool name
    pub fn by_name(&self, name: &str) -> Vec<&ToolCall> {
        self.calls.iter()
            .filter(|c| c.name == name)
            .collect()
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.calls.clear();
    }
}

/// Create default registry with gold loan tools
pub fn create_default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Register gold loan tools
    registry.register(crate::gold_loan::EligibilityCheckTool::new());
    registry.register(crate::gold_loan::SavingsCalculatorTool::new());
    registry.register(crate::gold_loan::LeadCaptureTool::new());
    registry.register(crate::gold_loan::AppointmentSchedulerTool::new());
    registry.register(crate::gold_loan::BranchLocatorTool::new());

    // P0 FIX: Register missing tools that were implemented but not registered
    registry.register(crate::gold_loan::GetGoldPriceTool::new());
    registry.register(crate::gold_loan::EscalateToHumanTool::new());
    registry.register(crate::gold_loan::SendSmsTool::new());

    registry
}

/// P4 FIX: Integration configuration for tool registry
pub struct IntegrationConfig {
    /// CRM integration for lead management
    pub crm: Option<Arc<dyn crate::integrations::CrmIntegration>>,
    /// Calendar integration for appointment scheduling
    pub calendar: Option<Arc<dyn crate::integrations::CalendarIntegration>>,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            crm: None,
            calendar: None,
        }
    }
}

impl IntegrationConfig {
    /// Create with stub integrations (for development/testing)
    pub fn with_stubs() -> Self {
        Self {
            crm: Some(Arc::new(crate::integrations::StubCrmIntegration::new())),
            calendar: Some(Arc::new(crate::integrations::StubCalendarIntegration::new())),
        }
    }

    /// Set CRM integration
    pub fn with_crm(mut self, crm: Arc<dyn crate::integrations::CrmIntegration>) -> Self {
        self.crm = Some(crm);
        self
    }

    /// Set calendar integration
    pub fn with_calendar(mut self, calendar: Arc<dyn crate::integrations::CalendarIntegration>) -> Self {
        self.calendar = Some(calendar);
        self
    }
}

/// P4 FIX: Create registry with integration support
///
/// Creates a tool registry with optional CRM and calendar integrations
/// injected into the appropriate tools (LeadCaptureTool, AppointmentSchedulerTool).
pub fn create_registry_with_integrations(config: IntegrationConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Register gold loan tools
    registry.register(crate::gold_loan::EligibilityCheckTool::new());
    registry.register(crate::gold_loan::SavingsCalculatorTool::new());
    registry.register(crate::gold_loan::BranchLocatorTool::new());

    // P4 FIX: Register LeadCaptureTool with CRM integration if available
    if let Some(crm) = config.crm {
        registry.register(crate::gold_loan::LeadCaptureTool::with_crm(crm));
    } else {
        registry.register(crate::gold_loan::LeadCaptureTool::new());
    }

    // P4 FIX: Register AppointmentSchedulerTool with calendar integration if available
    if let Some(calendar) = config.calendar {
        registry.register(crate::gold_loan::AppointmentSchedulerTool::with_calendar(calendar));
    } else {
        registry.register(crate::gold_loan::AppointmentSchedulerTool::new());
    }

    // P0 FIX: Register missing tools that were implemented but not registered
    registry.register(crate::gold_loan::GetGoldPriceTool::new());
    registry.register(crate::gold_loan::EscalateToHumanTool::new());
    registry.register(crate::gold_loan::SendSmsTool::new());

    registry
}

/// P2 FIX: Full configuration for tool registry with persistence
///
/// Includes both business integrations (CRM, Calendar) and persistence
/// services (SMS, Gold Price) for production deployment.
pub struct FullIntegrationConfig {
    /// CRM integration for lead management
    pub crm: Option<Arc<dyn crate::integrations::CrmIntegration>>,
    /// Calendar integration for appointment scheduling
    pub calendar: Option<Arc<dyn crate::integrations::CalendarIntegration>>,
    /// SMS service for sending messages (persisted to ScyllaDB)
    pub sms_service: Option<Arc<dyn voice_agent_persistence::SmsService>>,
    /// Gold price service (persisted to ScyllaDB)
    pub gold_price_service: Option<Arc<dyn voice_agent_persistence::GoldPriceService>>,
}

impl Default for FullIntegrationConfig {
    fn default() -> Self {
        Self {
            crm: None,
            calendar: None,
            sms_service: None,
            gold_price_service: None,
        }
    }
}

impl FullIntegrationConfig {
    /// Create from persistence layer
    pub fn from_persistence(persistence: &voice_agent_persistence::PersistenceLayer) -> Self {
        Self {
            crm: Some(Arc::new(crate::integrations::StubCrmIntegration::new())),
            calendar: Some(Arc::new(crate::integrations::StubCalendarIntegration::new())),
            sms_service: Some(Arc::new(persistence.sms.clone()) as Arc<dyn voice_agent_persistence::SmsService>),
            gold_price_service: Some(Arc::new(persistence.gold_price.clone()) as Arc<dyn voice_agent_persistence::GoldPriceService>),
        }
    }

    /// Set CRM integration
    pub fn with_crm(mut self, crm: Arc<dyn crate::integrations::CrmIntegration>) -> Self {
        self.crm = Some(crm);
        self
    }

    /// Set calendar integration
    pub fn with_calendar(mut self, calendar: Arc<dyn crate::integrations::CalendarIntegration>) -> Self {
        self.calendar = Some(calendar);
        self
    }

    /// Set SMS service
    pub fn with_sms_service(mut self, sms: Arc<dyn voice_agent_persistence::SmsService>) -> Self {
        self.sms_service = Some(sms);
        self
    }

    /// Set gold price service
    pub fn with_gold_price_service(mut self, price: Arc<dyn voice_agent_persistence::GoldPriceService>) -> Self {
        self.gold_price_service = Some(price);
        self
    }
}

/// P2 FIX: Create registry with full persistence support
///
/// Creates a tool registry with:
/// - Business integrations (CRM, Calendar)
/// - Persistence services (SMS → ScyllaDB, Gold Price → ScyllaDB)
/// - All MCP tools properly wired
pub fn create_registry_with_persistence(config: FullIntegrationConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // Register gold loan tools (no persistence needed)
    registry.register(crate::gold_loan::EligibilityCheckTool::new());
    registry.register(crate::gold_loan::SavingsCalculatorTool::new());
    registry.register(crate::gold_loan::BranchLocatorTool::new());

    // LeadCaptureTool with CRM integration
    if let Some(crm) = config.crm {
        registry.register(crate::gold_loan::LeadCaptureTool::with_crm(crm));
    } else {
        registry.register(crate::gold_loan::LeadCaptureTool::new());
    }

    // AppointmentSchedulerTool with calendar integration
    if let Some(calendar) = config.calendar {
        registry.register(crate::gold_loan::AppointmentSchedulerTool::with_calendar(calendar));
    } else {
        registry.register(crate::gold_loan::AppointmentSchedulerTool::new());
    }

    // P2 FIX: GetGoldPriceTool with persistence service
    if let Some(price_service) = config.gold_price_service {
        registry.register(crate::gold_loan::GetGoldPriceTool::with_price_service(price_service));
    } else {
        registry.register(crate::gold_loan::GetGoldPriceTool::new());
    }

    // P2 FIX: EscalateToHumanTool (no persistence needed, logs via audit)
    registry.register(crate::gold_loan::EscalateToHumanTool::new());

    // P2 FIX: SendSmsTool with persistence service
    if let Some(sms_service) = config.sms_service {
        registry.register(crate::gold_loan::SendSmsTool::with_sms_service(sms_service));
    } else {
        registry.register(crate::gold_loan::SendSmsTool::new());
    }

    tracing::info!(
        tools = registry.len(),
        "Created tool registry with persistence support"
    );

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gold_loan::EligibilityCheckTool;

    #[test]
    fn test_registry_basic() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(EligibilityCheckTool::new());
        assert_eq!(registry.len(), 1);
        assert!(registry.has("check_eligibility"));
    }

    #[test]
    fn test_registry_list_tools() {
        let registry = create_default_registry();
        let tools = registry.list_tools();

        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "check_eligibility"));
    }

    #[test]
    fn test_tool_call_tracker() {
        let mut tracker = ToolCallTracker::new(100);

        tracker.record(ToolCall {
            name: "test".to_string(),
            arguments: serde_json::json!({}),
            output: ToolOutput::text("result"),
            duration_ms: 10,
            timestamp: std::time::Instant::now(),
        });

        assert_eq!(tracker.all().len(), 1);
    }

    // P4 FIX: Tests for integration config

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert!(config.crm.is_none());
        assert!(config.calendar.is_none());
    }

    #[test]
    fn test_integration_config_with_stubs() {
        let config = IntegrationConfig::with_stubs();
        assert!(config.crm.is_some());
        assert!(config.calendar.is_some());
    }

    #[test]
    fn test_registry_with_integrations() {
        let config = IntegrationConfig::with_stubs();
        let registry = create_registry_with_integrations(config);

        // P0 FIX: Should have all 8 tools (5 original + 3 P0 tools)
        assert_eq!(registry.len(), 8);
        assert!(registry.has("check_eligibility"));
        assert!(registry.has("calculate_savings"));
        assert!(registry.has("capture_lead"));
        assert!(registry.has("schedule_appointment"));
        assert!(registry.has("find_branches"));
        // P0 FIX: Verify new tools are registered
        assert!(registry.has("get_gold_price"));
        assert!(registry.has("escalate_to_human"));
        assert!(registry.has("send_sms"));
    }

    #[test]
    fn test_registry_without_integrations() {
        let config = IntegrationConfig::default();
        let registry = create_registry_with_integrations(config);

        // P0 FIX: Should still have all 8 tools (just without integrations)
        assert_eq!(registry.len(), 8);
        assert!(registry.has("capture_lead"));
        assert!(registry.has("schedule_appointment"));
        // P0 FIX: Verify new tools are registered
        assert!(registry.has("get_gold_price"));
        assert!(registry.has("escalate_to_human"));
        assert!(registry.has("send_sms"));
    }

    #[test]
    fn test_default_registry_has_all_tools() {
        let registry = create_default_registry();

        // P0 FIX: Default registry should have all 8 tools
        assert_eq!(registry.len(), 8);
        assert!(registry.has("check_eligibility"));
        assert!(registry.has("calculate_savings"));
        assert!(registry.has("capture_lead"));
        assert!(registry.has("schedule_appointment"));
        assert!(registry.has("find_branches"));
        assert!(registry.has("get_gold_price"));
        assert!(registry.has("escalate_to_human"));
        assert!(registry.has("send_sms"));
    }
}
