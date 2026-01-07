//! MCP Tools for Gold Loan Voice Agent
//!
//! Implements MCP (Model Context Protocol) compatible tool interface
//! with domain-specific tools for gold loan operations.

pub mod gold_loan;
pub mod integrations;
pub mod mcp;
pub mod registry;

pub use gold_loan::{
    // Branch data management
    get_branches, get_mock_branches, load_branches_from_file, reload_branches, BranchData,
    // Utility functions
    calculate_emi, calculate_total_interest,
    // Tool implementations
    AppointmentSchedulerTool, BranchLocatorTool, CompetitorComparisonTool, DocumentChecklistTool,
    EligibilityCheckTool, EscalateToHumanTool, GetGoldPriceTool, LeadCaptureTool,
    SavingsCalculatorTool, SendSmsTool,
};
pub use integrations::{
    Appointment, AppointmentPurpose, AppointmentStatus, CalendarIntegration, CrmIntegration,
    CrmLead, IntegrationError, InterestLevel, LeadSource, LeadStatus, StubCalendarIntegration,
    StubCrmIntegration, TimeSlot,
};
pub use mcp::{
    methods,
    ContentBlock,
    ErrorCode,
    InputSchema,
    JsonRpcError,
    // P3-3 FIX: Full MCP protocol types
    JsonRpcRequest,
    JsonRpcResponse,
    ProgressParams,
    ProgressToken,
    PropertySchema,
    RequestId,
    Resource,
    ResourceCapabilities,
    ResourceContent,
    ResourceProvider,
    ServerCapabilities,
    // Core tool types (from voice_agent_core)
    Tool,
    ToolCallParams,
    ToolCapabilities,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolSchema,
};
pub use registry::{
    // P13 FIX: Domain config wiring via ToolsDomainView
    create_registry_with_view,
    create_registry_with_integrations,
    create_registry_with_persistence,
    ConfigurableToolRegistry,
    FullIntegrationConfig,
    IntegrationConfig,
    ToolExecutor,
    ToolRegistry,
};

// P2 FIX: Removed redundant ToolsError enum.
// Use mcp::ToolError for tool execution errors instead.
// This unifies error handling across the tools crate.
//
// P3 FIX: The From<ToolError> impl is now in voice_agent_core::error
// since both types are defined in the core crate.
