//! Savings Calculator Tool
//!
//! Calculate potential savings when switching from NBFC to our gold loan.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};
use super::super::utils::{calculate_emi, calculate_total_interest};

/// Savings calculator tool
///
/// P13 FIX: Uses ToolsDomainView instead of GoldLoanConfig
/// P15 FIX: ToolsDomainView is now REQUIRED - no more hardcoded fallbacks
pub struct SavingsCalculatorTool {
    view: Arc<ToolsDomainView>,
}

impl SavingsCalculatorTool {
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

    fn get_competitor_rate(&self, lender: &str) -> f64 {
        self.view.get_competitor_rate(lender)
    }

    fn get_rate_tier_name(&self, amount: f64) -> &str {
        self.view.get_rate_tier_name(amount)
    }

    fn company_name(&self) -> &str {
        self.view.company_name()
    }
}

#[async_trait]
impl Tool for SavingsCalculatorTool {
    fn name(&self) -> &str {
        "calculate_savings"
    }

    fn description(&self) -> &str {
        "Calculate potential savings when switching from NBFC to our gold loan"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "current_loan_amount",
                    PropertySchema::number("Current loan amount in INR"),
                    true,
                )
                .property(
                    "current_interest_rate",
                    PropertySchema::number("Current interest rate (%)").with_range(10.0, 30.0),
                    true,
                )
                .property(
                    "remaining_tenure_months",
                    PropertySchema::integer("Remaining tenure in months"),
                    true,
                )
                .property(
                    "current_lender",
                    PropertySchema::enum_type(
                        "Current lender",
                        vec![
                            "Muthoot".into(),
                            "Manappuram".into(),
                            "IIFL".into(),
                            "Other NBFC".into(),
                        ],
                    ),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let loan_amount: f64 = input
            .get("current_loan_amount")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("current_loan_amount is required"))?;

        let current_lender = input
            .get("current_lender")
            .and_then(|v| v.as_str())
            .unwrap_or("Other NBFC");

        // P13 FIX: Use ToolsDomainView for competitor rates
        let current_rate: f64 = input
            .get("current_interest_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| self.get_competitor_rate(current_lender));

        let tenure_months: i64 = input
            .get("remaining_tenure_months")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| ToolError::invalid_params("remaining_tenure_months is required"))?;

        // P15 FIX: Use config-driven rates and bank name
        let our_rate = self.get_rate(loan_amount);
        let rate_tier = self.get_rate_tier_name(loan_amount);
        let company_name = self.company_name();

        let current_emi = calculate_emi(loan_amount, current_rate, tenure_months);
        let our_emi = calculate_emi(loan_amount, our_rate, tenure_months);
        let emi_savings = current_emi - our_emi;

        let current_monthly_interest = loan_amount * (current_rate / 100.0 / 12.0);
        let our_monthly_interest = loan_amount * (our_rate / 100.0 / 12.0);
        let monthly_interest_savings = current_monthly_interest - our_monthly_interest;

        let total_emi_savings = emi_savings * tenure_months as f64;
        let total_interest_savings =
            calculate_total_interest(loan_amount, current_rate, tenure_months)
                - calculate_total_interest(loan_amount, our_rate, tenure_months);

        // P16 FIX: Use config-driven response templates
        let message = if self.view.has_response_templates("calculate_savings") {
            let mut vars = self.view.default_template_vars();
            vars.insert("company_name".to_string(), company_name.to_string());
            vars.insert("rate_tier".to_string(), rate_tier.to_string());
            vars.insert("our_rate".to_string(), format!("{:.1}", our_rate));
            vars.insert("emi_savings".to_string(), format!("{:.0}", emi_savings));
            vars.insert("interest_savings".to_string(), format!("{:.0}", monthly_interest_savings));
            vars.insert("total_savings".to_string(), format!("{:.0}", total_emi_savings));
            vars.insert("tenure_months".to_string(), tenure_months.to_string());
            vars.insert("current_lender".to_string(), current_lender.to_string());
            vars.insert("rate_reduction".to_string(), format!("{:.1}", current_rate - our_rate));
            self.view.render_response("calculate_savings", "savings_available", "en", &vars)
                .unwrap_or_else(|| format!(
                    "By switching to {} at our {} rate of {}%, you can save ₹{:.0} per month on EMI (or ₹{:.0} on interest-only) and ₹{:.0} total over the remaining {} months!",
                    company_name, rate_tier, our_rate, emi_savings, monthly_interest_savings, total_emi_savings, tenure_months
                ))
        } else {
            format!(
                "By switching to {} at our {} rate of {}%, you can save ₹{:.0} per month on EMI (or ₹{:.0} on interest-only) and ₹{:.0} total over the remaining {} months!",
                company_name, rate_tier, our_rate, emi_savings, monthly_interest_savings, total_emi_savings, tenure_months
            )
        };

        let result = json!({
            "current_lender": current_lender,
            "current_interest_rate_percent": current_rate,
            "our_interest_rate_percent": our_rate,
            "rate_reduction_percent": current_rate - our_rate,
            "current_emi_inr": current_emi.round(),
            "our_emi_inr": our_emi.round(),
            "monthly_emi_savings_inr": emi_savings.round(),
            "total_emi_savings_inr": total_emi_savings.round(),
            "current_monthly_interest_inr": current_monthly_interest.round(),
            "our_monthly_interest_inr": our_monthly_interest.round(),
            "monthly_interest_savings_inr": monthly_interest_savings.round(),
            "total_interest_savings_inr": total_interest_savings.round(),
            "tenure_months": tenure_months,
            "rate_tier": rate_tier,
            "company_name": company_name,
            "message": message
        });

        Ok(ToolOutput::json(result))
    }
}
