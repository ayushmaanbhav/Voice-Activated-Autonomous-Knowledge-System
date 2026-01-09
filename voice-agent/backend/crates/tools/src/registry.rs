//! Tool Registry
//!
//! Manages tool registration, discovery, and execution.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use crate::mcp::{Tool, ToolError, ToolOutput, ToolSchema};

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
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::not_found(format!("Tool not found: {}", name)))?;

        // Validate input
        tool.validate(&arguments)?;

        // P5 FIX: Use per-tool timeout, falling back to default
        let timeout_secs = tool.timeout_secs();
        let timeout_duration = Duration::from_secs(timeout_secs);

        tracing::trace!(
            tool = name,
            timeout_secs = timeout_secs,
            "Executing tool with timeout"
        );

        match tokio::time::timeout(timeout_duration, tool.execute(arguments)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(ToolError::timeout(name, timeout_secs)),
        }
    }

    fn list_tools(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
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
        self.calls.iter().filter(|c| c.name == name).collect()
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.calls.clear();
    }
}

// P15 FIX: Removed create_default_registry() - ToolsDomainView is now REQUIRED
// Tools cannot be created without domain configuration.
// Use create_registry_with_view() instead.

// =============================================================================
// P13 FIX: Domain Config Wiring via ToolsDomainView
// =============================================================================

/// P15 FIX: Create registry with REQUIRED ToolsDomainView configuration
///
/// Uses ToolsDomainView for all configurable values (rates, LTV, competitor info).
/// This is the ONLY way to create a tool registry - domain config is mandatory.
pub fn create_registry_with_view(
    view: Arc<voice_agent_config::ToolsDomainView>,
) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // P15: Register gold loan tools - ALL require ToolsDomainView
    registry.register(crate::domain_tools::EligibilityCheckTool::new(view.clone()));
    registry.register(crate::domain_tools::SavingsCalculatorTool::new(view.clone()));
    registry.register(crate::domain_tools::GetGoldPriceTool::new(view.clone()));
    registry.register(crate::domain_tools::CompetitorComparisonTool::new(view.clone()));

    // Tools that don't need domain config (CRM/calendar integrations only)
    registry.register(crate::domain_tools::LeadCaptureTool::new());
    // P16 FIX: Appointment tool uses view for config-driven purposes/times
    registry.register(crate::domain_tools::AppointmentSchedulerTool::with_view(view.clone()));
    registry.register(crate::domain_tools::BranchLocatorTool::new());
    registry.register(crate::domain_tools::EscalateToHumanTool::new());
    // P16 FIX: SMS and Document tools now use view for config-driven content
    registry.register(crate::domain_tools::SendSmsTool::with_view(view.clone()));
    registry.register(crate::domain_tools::DocumentChecklistTool::with_view(view.clone()));

    tracing::info!(
        bank_name = view.company_name(),
        base_rate = view.base_interest_rate(),
        ltv = view.ltv_percent(),
        "Created tool registry with ToolsDomainView (domain config required)"
    );

    registry
}

/// P13 FIX: Configurable tool registry with hot-reload support
///
/// Wraps a ToolRegistry with ToolsDomainView management, allowing tools to be
/// recreated when configuration changes.
pub struct ConfigurableToolRegistry {
    inner: parking_lot::RwLock<ToolRegistry>,
    view: parking_lot::RwLock<Arc<voice_agent_config::ToolsDomainView>>,
}

impl ConfigurableToolRegistry {
    /// Create with ToolsDomainView
    pub fn new(view: Arc<voice_agent_config::ToolsDomainView>) -> Self {
        let registry = create_registry_with_view(view.clone());
        Self {
            inner: parking_lot::RwLock::new(registry),
            view: parking_lot::RwLock::new(view),
        }
    }

    /// Create with default view from MasterDomainConfig
    pub fn with_defaults() -> Self {
        let config = Arc::new(voice_agent_config::MasterDomainConfig::default());
        let view = Arc::new(voice_agent_config::ToolsDomainView::new(config));
        Self::new(view)
    }

    /// Reload configuration and recreate tools
    ///
    /// This is the hot-reload entry point. Call this when config changes.
    pub fn reload(&self, new_view: Arc<voice_agent_config::ToolsDomainView>) {
        let old_view = self.view.read();
        tracing::info!(
            old_rate = %old_view.base_interest_rate(),
            new_rate = %new_view.base_interest_rate(),
            "Hot-reloading tool configuration"
        );
        drop(old_view);

        // Update view
        *self.view.write() = new_view.clone();

        // Recreate registry with new view
        let new_registry = create_registry_with_view(new_view);
        *self.inner.write() = new_registry;

        tracing::info!("Tool registry reloaded with new configuration");
    }

    /// Get current view
    pub fn view(&self) -> Arc<voice_agent_config::ToolsDomainView> {
        self.view.read().clone()
    }

    /// Execute a tool
    pub async fn execute(&self, name: &str, arguments: Value) -> Result<ToolOutput, ToolError> {
        // Get the tool without holding the lock across await
        let tool = {
            let registry = self.inner.read();
            registry.get(name).cloned()
        };

        let tool = tool.ok_or_else(|| ToolError::not_found(format!("Tool not found: {}", name)))?;

        // Validate input
        tool.validate(&arguments)?;

        // Execute with timeout (from the Tool trait default)
        let timeout_secs = tool.timeout_secs();
        let timeout_duration = Duration::from_secs(timeout_secs);

        match tokio::time::timeout(timeout_duration, tool.execute(arguments)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(ToolError::timeout(name, timeout_secs)),
        }
    }

    /// List available tools
    pub fn list_tools(&self) -> Vec<ToolSchema> {
        self.inner.read().list_tools()
    }

    /// Get tool schema
    pub fn get_tool(&self, name: &str) -> Option<ToolSchema> {
        self.inner.read().get_tool(name)
    }

    /// Check if tool exists
    pub fn has(&self, name: &str) -> bool {
        self.inner.read().has(name)
    }

    /// Get tool count
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

#[async_trait]
impl ToolExecutor for ConfigurableToolRegistry {
    async fn execute(&self, name: &str, arguments: Value) -> Result<ToolOutput, ToolError> {
        self.execute(name, arguments).await
    }

    fn list_tools(&self) -> Vec<ToolSchema> {
        self.list_tools()
    }

    fn get_tool(&self, name: &str) -> Option<ToolSchema> {
        self.get_tool(name)
    }
}

/// P15 FIX: Integration configuration for tool registry - view is REQUIRED
pub struct IntegrationConfig {
    /// REQUIRED: ToolsDomainView for domain configuration
    pub view: Arc<voice_agent_config::ToolsDomainView>,
    /// CRM integration for lead management
    pub crm: Option<Arc<dyn crate::integrations::CrmIntegration>>,
    /// Calendar integration for appointment scheduling
    pub calendar: Option<Arc<dyn crate::integrations::CalendarIntegration>>,
}

impl IntegrationConfig {
    /// Create with required view
    pub fn new(view: Arc<voice_agent_config::ToolsDomainView>) -> Self {
        Self {
            view,
            crm: None,
            calendar: None,
        }
    }

    /// Create with stub integrations (for development/testing)
    pub fn with_stubs(view: Arc<voice_agent_config::ToolsDomainView>) -> Self {
        Self {
            view,
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
    pub fn with_calendar(
        mut self,
        calendar: Arc<dyn crate::integrations::CalendarIntegration>,
    ) -> Self {
        self.calendar = Some(calendar);
        self
    }
}

/// P15 FIX: Create registry with integration support - view is REQUIRED
///
/// Creates a tool registry with:
/// - REQUIRED ToolsDomainView for all domain configuration
/// - Optional CRM and calendar integrations
pub fn create_registry_with_integrations(config: IntegrationConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // P15: All tools that need domain config use the REQUIRED view
    registry.register(crate::domain_tools::EligibilityCheckTool::new(config.view.clone()));
    registry.register(crate::domain_tools::SavingsCalculatorTool::new(config.view.clone()));
    registry.register(crate::domain_tools::GetGoldPriceTool::new(config.view.clone()));
    registry.register(crate::domain_tools::CompetitorComparisonTool::new(config.view.clone()));
    registry.register(crate::domain_tools::BranchLocatorTool::new());

    // LeadCaptureTool with optional CRM integration
    if let Some(crm) = config.crm {
        registry.register(crate::domain_tools::LeadCaptureTool::with_crm(crm));
    } else {
        registry.register(crate::domain_tools::LeadCaptureTool::new());
    }

    // P16 FIX: AppointmentSchedulerTool with optional calendar integration and view
    if let Some(calendar) = config.calendar {
        registry.register(crate::domain_tools::AppointmentSchedulerTool::with_calendar_and_view(
            calendar,
            config.view.clone(),
        ));
    } else {
        registry.register(crate::domain_tools::AppointmentSchedulerTool::with_view(config.view.clone()));
    }

    registry.register(crate::domain_tools::EscalateToHumanTool::new());
    // P16 FIX: SMS and Document tools now use view for config-driven content
    registry.register(crate::domain_tools::SendSmsTool::with_view(config.view.clone()));
    registry.register(crate::domain_tools::DocumentChecklistTool::with_view(config.view.clone()));

    tracing::info!(
        bank_name = config.view.company_name(),
        base_rate = config.view.base_interest_rate(),
        "Created tool registry with integrations (domain config required)"
    );

    registry
}

/// P15 FIX: Full configuration for tool registry with persistence - view is REQUIRED
///
/// Includes:
/// - REQUIRED ToolsDomainView for domain configuration
/// - Optional business integrations (CRM, Calendar)
/// - Optional persistence services (SMS, Gold Price)
pub struct FullIntegrationConfig {
    /// REQUIRED: ToolsDomainView for domain configuration
    pub view: Arc<voice_agent_config::ToolsDomainView>,
    /// CRM integration for lead management
    pub crm: Option<Arc<dyn crate::integrations::CrmIntegration>>,
    /// Calendar integration for appointment scheduling
    pub calendar: Option<Arc<dyn crate::integrations::CalendarIntegration>>,
    /// SMS service for sending messages (persisted to ScyllaDB)
    pub sms_service: Option<Arc<dyn voice_agent_persistence::SmsService>>,
    /// P16 FIX: Asset price service (generic, gold_price_service for backwards compatibility)
    pub gold_price_service: Option<Arc<dyn voice_agent_persistence::AssetPriceService>>,
}

impl FullIntegrationConfig {
    /// Create with required view
    pub fn new(view: Arc<voice_agent_config::ToolsDomainView>) -> Self {
        Self {
            view,
            crm: None,
            calendar: None,
            sms_service: None,
            gold_price_service: None,
        }
    }

    /// Create from persistence layer with REQUIRED view
    pub fn from_persistence(
        view: Arc<voice_agent_config::ToolsDomainView>,
        persistence: &voice_agent_persistence::PersistenceLayer,
    ) -> Self {
        Self {
            view,
            crm: Some(Arc::new(crate::integrations::StubCrmIntegration::new())),
            calendar: Some(Arc::new(crate::integrations::StubCalendarIntegration::new())),
            sms_service: Some(
                Arc::new(persistence.sms.clone()) as Arc<dyn voice_agent_persistence::SmsService>
            ),
            // P16 FIX: Use generic asset_price field (AssetPriceService)
            gold_price_service: Some(Arc::new(persistence.asset_price.clone())
                as Arc<dyn voice_agent_persistence::AssetPriceService>),
        }
    }

    /// Set CRM integration
    pub fn with_crm(mut self, crm: Arc<dyn crate::integrations::CrmIntegration>) -> Self {
        self.crm = Some(crm);
        self
    }

    /// Set calendar integration
    pub fn with_calendar(
        mut self,
        calendar: Arc<dyn crate::integrations::CalendarIntegration>,
    ) -> Self {
        self.calendar = Some(calendar);
        self
    }

    /// Set SMS service
    pub fn with_sms_service(mut self, sms: Arc<dyn voice_agent_persistence::SmsService>) -> Self {
        self.sms_service = Some(sms);
        self
    }

    /// P16 FIX: Set asset price service (gold_price_service alias for backwards compatibility)
    pub fn with_gold_price_service(
        mut self,
        price: Arc<dyn voice_agent_persistence::AssetPriceService>,
    ) -> Self {
        self.gold_price_service = Some(price);
        self
    }

    /// P16 FIX: Set asset price service (preferred method name)
    pub fn with_asset_price_service(
        mut self,
        price: Arc<dyn voice_agent_persistence::AssetPriceService>,
    ) -> Self {
        self.gold_price_service = Some(price);
        self
    }
}

/// P15 FIX: Create registry with full persistence support - view is REQUIRED
///
/// Creates a tool registry with:
/// - REQUIRED ToolsDomainView for domain configuration
/// - Optional business integrations (CRM, Calendar)
/// - Optional persistence services (SMS, Gold Price)
pub fn create_registry_with_persistence(config: FullIntegrationConfig) -> ToolRegistry {
    let mut registry = ToolRegistry::new();

    // P15: All tools that need domain config use the REQUIRED view
    registry.register(crate::domain_tools::EligibilityCheckTool::new(config.view.clone()));
    registry.register(crate::domain_tools::SavingsCalculatorTool::new(config.view.clone()));
    registry.register(crate::domain_tools::CompetitorComparisonTool::new(config.view.clone()));
    registry.register(crate::domain_tools::BranchLocatorTool::new());

    // LeadCaptureTool with optional CRM integration
    if let Some(crm) = config.crm {
        registry.register(crate::domain_tools::LeadCaptureTool::with_crm(crm));
    } else {
        registry.register(crate::domain_tools::LeadCaptureTool::new());
    }

    // P16 FIX: AppointmentSchedulerTool with optional calendar integration and view
    if let Some(calendar) = config.calendar {
        registry.register(crate::domain_tools::AppointmentSchedulerTool::with_calendar_and_view(
            calendar,
            config.view.clone(),
        ));
    } else {
        registry.register(crate::domain_tools::AppointmentSchedulerTool::with_view(config.view.clone()));
    }

    // GetGoldPriceTool with REQUIRED view and optional price service
    if let Some(service) = config.gold_price_service {
        registry.register(crate::domain_tools::GetGoldPriceTool::with_price_service(
            service,
            config.view.clone(),
        ));
    } else {
        registry.register(crate::domain_tools::GetGoldPriceTool::new(config.view.clone()));
    }

    // EscalateToHumanTool (no domain config needed)
    registry.register(crate::domain_tools::EscalateToHumanTool::new());

    // P16 FIX: SendSmsTool with view and optional persistence service
    if let Some(sms_service) = config.sms_service {
        registry.register(crate::domain_tools::SendSmsTool::with_service_and_view(
            sms_service,
            config.view.clone(),
        ));
    } else {
        registry.register(crate::domain_tools::SendSmsTool::with_view(config.view.clone()));
    }

    // P16 FIX: Document tool uses view for config-driven content
    registry.register(crate::domain_tools::DocumentChecklistTool::with_view(config.view.clone()));

    tracing::info!(
        tools = registry.len(),
        bank_name = config.view.company_name(),
        base_rate = config.view.base_interest_rate(),
        ltv = config.view.ltv_percent(),
        "Created tool registry with persistence (domain config required)"
    );

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P15 FIX: Helper to create test view
    fn test_view() -> Arc<voice_agent_config::ToolsDomainView> {
        let config = Arc::new(voice_agent_config::MasterDomainConfig::default());
        Arc::new(voice_agent_config::ToolsDomainView::new(config))
    }

    #[test]
    fn test_registry_basic() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        // P15 FIX: Tools now require view
        let view = test_view();
        registry.register(crate::domain_tools::EligibilityCheckTool::new(view));
        assert_eq!(registry.len(), 1);
        assert!(registry.has("check_eligibility"));
    }

    #[test]
    fn test_registry_list_tools() {
        // P15 FIX: Use create_registry_with_view
        let view = test_view();
        let registry = create_registry_with_view(view);
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

    // P15 FIX: Tests for integration config (now require view)

    #[test]
    fn test_integration_config_new() {
        let view = test_view();
        let config = IntegrationConfig::new(view);
        assert!(config.crm.is_none());
        assert!(config.calendar.is_none());
    }

    #[test]
    fn test_integration_config_with_stubs() {
        let view = test_view();
        let config = IntegrationConfig::with_stubs(view);
        assert!(config.crm.is_some());
        assert!(config.calendar.is_some());
    }

    #[test]
    fn test_registry_with_integrations() {
        let view = test_view();
        let config = IntegrationConfig::with_stubs(view);
        let registry = create_registry_with_integrations(config);

        // Should have all 10 tools
        assert_eq!(registry.len(), 10);
        assert!(registry.has("check_eligibility"));
        assert!(registry.has("calculate_savings"));
        assert!(registry.has("capture_lead"));
        assert!(registry.has("schedule_appointment"));
        assert!(registry.has("find_branches"));
        assert!(registry.has("get_gold_price"));
        assert!(registry.has("escalate_to_human"));
        assert!(registry.has("send_sms"));
        assert!(registry.has("get_document_checklist"));
        assert!(registry.has("compare_lenders"));
    }

    #[test]
    fn test_registry_without_integrations() {
        let view = test_view();
        let config = IntegrationConfig::new(view);
        let registry = create_registry_with_integrations(config);

        // Should still have all 10 tools (just without integrations)
        assert_eq!(registry.len(), 10);
        assert!(registry.has("capture_lead"));
        assert!(registry.has("schedule_appointment"));
        assert!(registry.has("get_gold_price"));
        assert!(registry.has("escalate_to_human"));
        assert!(registry.has("send_sms"));
        assert!(registry.has("get_document_checklist"));
        assert!(registry.has("compare_lenders"));
    }

    #[test]
    fn test_registry_with_view_has_all_tools() {
        let view = test_view();
        let registry = create_registry_with_view(view);

        // Registry should have all 10 tools
        assert_eq!(registry.len(), 10);
        assert!(registry.has("check_eligibility"));
        assert!(registry.has("calculate_savings"));
        assert!(registry.has("capture_lead"));
        assert!(registry.has("schedule_appointment"));
        assert!(registry.has("find_branches"));
        assert!(registry.has("get_gold_price"));
        assert!(registry.has("escalate_to_human"));
        assert!(registry.has("send_sms"));
        assert!(registry.has("get_document_checklist"));
        assert!(registry.has("compare_lenders"));
    }
}
