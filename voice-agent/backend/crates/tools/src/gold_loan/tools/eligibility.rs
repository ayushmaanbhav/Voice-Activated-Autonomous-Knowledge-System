//! Eligibility Check Tool
//!
//! Check customer eligibility for gold loan based on gold weight and purity.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Check eligibility tool
///
/// P13 FIX: Uses ToolsDomainView instead of GoldLoanConfig
/// P15 FIX: ToolsDomainView is now REQUIRED - no more hardcoded fallbacks
pub struct EligibilityCheckTool {
    view: Arc<ToolsDomainView>,
}

impl EligibilityCheckTool {
    /// Create with required ToolsDomainView - domain config is mandatory
    pub fn new(view: Arc<ToolsDomainView>) -> Self {
        Self { view }
    }

    /// Alias for new() for backwards compatibility during migration
    pub fn with_view(view: Arc<ToolsDomainView>) -> Self {
        Self::new(view)
    }

    fn get_rate(&self, amount: f64) -> f64 {
        self.view.get_rate_for_amount(amount)
    }

    fn get_ltv(&self) -> f64 {
        self.view.ltv_percent()
    }

    fn get_min_loan(&self) -> f64 {
        self.view.min_loan_amount()
    }

    fn get_processing_fee(&self) -> f64 {
        self.view.processing_fee_percent()
    }

    fn calculate_gold_value(&self, weight: f64, purity: &str) -> f64 {
        self.view.calculate_gold_value(weight, purity)
    }

    fn calculate_max_loan(&self, gold_value: f64) -> f64 {
        self.view.calculate_max_loan(gold_value)
    }
}

#[async_trait]
impl Tool for EligibilityCheckTool {
    fn name(&self) -> &str {
        "check_eligibility"
    }

    fn description(&self) -> &str {
        "Check customer eligibility for gold loan based on gold weight and purity"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "gold_weight_grams",
                    PropertySchema::number("Gold weight in grams"),
                    true,
                )
                .property(
                    "gold_purity",
                    PropertySchema::enum_type(
                        "Gold purity (22K, 18K, etc.)",
                        vec!["24K".into(), "22K".into(), "18K".into(), "14K".into()],
                    )
                    .with_default(json!("22K")),
                    false,
                )
                .property(
                    "existing_loan_amount",
                    PropertySchema::number("Existing loan amount if any"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let weight: f64 = input
            .get("gold_weight_grams")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("gold_weight_grams is required"))?;

        let purity = input
            .get("gold_purity")
            .and_then(|v| v.as_str())
            .unwrap_or("22K");

        let existing_loan = input
            .get("existing_loan_amount")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // P13 FIX: Calculate eligibility using ToolsDomainView
        let gold_value = self.calculate_gold_value(weight, purity);
        let max_loan = self.calculate_max_loan(gold_value);
        let available_loan = max_loan - existing_loan;

        // Use tiered interest rates based on loan amount
        let interest_rate = self.get_rate(available_loan.max(0.0));
        let min_loan = self.get_min_loan();

        let result = json!({
            "eligible": available_loan >= min_loan,
            "gold_value_inr": gold_value.round(),
            "max_loan_amount_inr": max_loan.round(),
            "existing_loan_inr": existing_loan,
            "available_loan_inr": available_loan.max(0.0).round(),
            "ltv_percent": self.get_ltv(),
            "interest_rate_percent": interest_rate,
            "processing_fee_percent": self.get_processing_fee(),
            "rate_tier": self.view.get_rate_tier_name(available_loan),
            "message": if available_loan >= min_loan {
                format!(
                    "You are eligible for a gold loan up to ₹{:.0} at {}% interest!",
                    available_loan, interest_rate
                )
            } else if available_loan > 0.0 {
                format!("You can get an additional ₹{:.0} on your gold.", available_loan)
            } else {
                "Based on your existing loan, no additional loan is available at this time.".to_string()
            }
        });

        Ok(ToolOutput::json(result))
    }
}
