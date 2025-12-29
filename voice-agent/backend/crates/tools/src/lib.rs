//! MCP Tools for Gold Loan Voice Agent
//!
//! Implements MCP (Model Context Protocol) compatible tool interface
//! with domain-specific tools for gold loan operations.

pub mod mcp;
pub mod registry;
pub mod gold_loan;
pub mod integrations;

pub use mcp::{Tool, ToolInput, ToolOutput, ToolSchema, ToolError};
pub use registry::{ToolRegistry, ToolExecutor, IntegrationConfig, create_registry_with_integrations};
pub use gold_loan::{
    EligibilityCheckTool,
    SavingsCalculatorTool,
    LeadCaptureTool,
    AppointmentSchedulerTool,
    BranchLocatorTool,
    BranchData,
    get_branches,
    reload_branches,
    // P0 FIX: New missing MCP tools
    GetGoldPriceTool,
    EscalateToHumanTool,
    SendSmsTool,
};
pub use integrations::{
    CrmIntegration, StubCrmIntegration, CrmLead, LeadSource, LeadStatus, InterestLevel,
    CalendarIntegration, StubCalendarIntegration, Appointment, AppointmentPurpose, AppointmentStatus, TimeSlot,
    IntegrationError,
};

/// P2 FIX: Removed redundant ToolsError enum.
/// Use mcp::ToolError for tool execution errors instead.
/// This unifies error handling across the tools crate.

impl From<ToolError> for voice_agent_core::Error {
    fn from(err: ToolError) -> Self {
        voice_agent_core::Error::Tool(voice_agent_core::error::ToolError::ExecutionFailed(err.to_string()))
    }
}
