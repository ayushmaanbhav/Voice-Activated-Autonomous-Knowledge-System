//! Gold Loan Tool Implementations
//!
//! MCP-compatible tools for the gold loan voice agent.
//!
//! Each tool is in its own module for better maintainability.

mod appointment;
mod branch_locator;
mod competitor;
mod document_checklist;
mod eligibility;
mod escalate;
mod gold_price;
mod lead_capture;
mod savings;
mod sms;

// Re-export all tools
pub use appointment::AppointmentSchedulerTool;
pub use branch_locator::BranchLocatorTool;
pub use competitor::CompetitorComparisonTool;
pub use document_checklist::DocumentChecklistTool;
pub use eligibility::EligibilityCheckTool;
pub use escalate::EscalateToHumanTool;
pub use gold_price::GetGoldPriceTool;
pub use lead_capture::LeadCaptureTool;
pub use savings::SavingsCalculatorTool;
pub use sms::SendSmsTool;
