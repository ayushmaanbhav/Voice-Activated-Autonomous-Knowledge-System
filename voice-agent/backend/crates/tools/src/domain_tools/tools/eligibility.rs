//! Eligibility Check Tool
//!
//! Check customer eligibility based on collateral weight and variant.
//! All schema content (names, descriptions, parameters) comes from YAML config.
//! Domain-specific parameter names (e.g., "gold_weight_grams") should be defined in config.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Tool name as defined in config - used to look up schema
const TOOL_NAME: &str = "check_eligibility";

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

    fn calculate_collateral_value(&self, weight: f64, variant: &str) -> f64 {
        // Uses domain-specific calculation from config
        self.view.calculate_gold_value(weight, variant)
    }

    fn calculate_max_loan(&self, collateral_value: f64) -> f64 {
        self.view.calculate_max_loan(collateral_value)
    }
}

#[async_trait]
impl Tool for EligibilityCheckTool {
    fn name(&self) -> &str {
        // Return tool name from config, fallback to constant
        self.view
            .tools_config()
            .get_tool(TOOL_NAME)
            .map(|t| t.name.as_str())
            .unwrap_or(TOOL_NAME)
    }

    fn description(&self) -> &str {
        // Return description from config if available
        // Note: We can't return &str from owned String, so use static fallback
        // The actual description is included in schema()
        "Check eligibility based on collateral weight and variant"
    }

    fn schema(&self) -> ToolSchema {
        // P16 FIX: Read schema from config - all content comes from YAML
        if let Some(core_schema) = self.view.tools_config().get_core_schema(TOOL_NAME) {
            core_schema
        } else {
            // Fallback if config not available (should not happen in production)
            // Uses generic parameter names - domain-specific names should come from config
            tracing::warn!("Tool schema not found in config for {}, using generic fallback", TOOL_NAME);
            ToolSchema {
                name: TOOL_NAME.to_string(),
                description: "Check eligibility based on collateral".to_string(),
                input_schema: InputSchema::object()
                    .property(
                        "collateral_weight",
                        PropertySchema::number("Weight/quantity of collateral"),
                        true,
                    )
                    .property(
                        "collateral_variant",
                        PropertySchema::string("Variant/grade of collateral"),
                        false,
                    ),
            }
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        // Accept both generic and legacy (gold-specific) parameter names
        let weight: f64 = input
            .get("collateral_weight")
            .or_else(|| input.get("gold_weight_grams")) // Legacy alias
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("collateral_weight is required"))?;

        let variant = input
            .get("collateral_variant")
            .or_else(|| input.get("gold_purity")) // Legacy alias
            .and_then(|v| v.as_str())
            .unwrap_or("22K"); // Default variant

        let existing_loan = input
            .get("existing_loan_amount")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Calculate eligibility using domain config
        let collateral_value = self.calculate_collateral_value(weight, variant);
        let max_loan = self.calculate_max_loan(collateral_value);
        let available_loan = max_loan - existing_loan;

        // Use tiered interest rates based on loan amount
        let interest_rate = self.get_rate(available_loan.max(0.0));
        let min_loan = self.get_min_loan();

        // P16 FIX: Use config-driven response templates
        let message = if available_loan >= min_loan {
            // Try config template, fallback to hardcoded
            if self.view.has_response_templates("check_eligibility") {
                let mut vars = self.view.default_template_vars();
                vars.insert("max_amount".to_string(), format!("{:.0}", available_loan));
                vars.insert("interest_rate".to_string(), format!("{:.1}", interest_rate));
                vars.insert("rate_description".to_string(),
                    self.view.get_rate_description(self.view.get_rate_tier_name(available_loan)).to_string());
                vars.insert("collateral_type".to_string(), "gold".to_string());
                self.view.render_response("check_eligibility", "eligible", "en", &vars)
                    .unwrap_or_else(|| format!(
                        "You are eligible for a loan up to ₹{:.0} at {}% interest!",
                        available_loan, interest_rate
                    ))
            } else {
                format!(
                    "You are eligible for a loan up to ₹{:.0} at {}% interest!",
                    available_loan, interest_rate
                )
            }
        } else if available_loan > 0.0 {
            if self.view.has_response_templates("check_eligibility") {
                let mut vars = self.view.default_template_vars();
                vars.insert("available_amount".to_string(), format!("{:.0}", available_loan));
                vars.insert("collateral_type".to_string(), "gold".to_string());
                self.view.render_response("check_eligibility", "additional_available", "en", &vars)
                    .unwrap_or_else(|| format!("You can get an additional ₹{:.0} on your collateral.", available_loan))
            } else {
                format!("You can get an additional ₹{:.0} on your collateral.", available_loan)
            }
        } else {
            if self.view.has_response_templates("check_eligibility") {
                let vars = self.view.default_template_vars();
                self.view.render_response("check_eligibility", "not_eligible", "en", &vars)
                    .unwrap_or_else(|| "Based on your existing loan, no additional loan is available at this time.".to_string())
            } else {
                "Based on your existing loan, no additional loan is available at this time.".to_string()
            }
        };

        let result = json!({
            "eligible": available_loan >= min_loan,
            "collateral_value_inr": collateral_value.round(),
            "gold_value_inr": collateral_value.round(), // Legacy alias
            "max_loan_amount_inr": max_loan.round(),
            "existing_loan_inr": existing_loan,
            "available_loan_inr": available_loan.max(0.0).round(),
            "ltv_percent": self.get_ltv(),
            "interest_rate_percent": interest_rate,
            "processing_fee_percent": self.get_processing_fee(),
            "rate_tier": self.view.get_rate_tier_name(available_loan),
            "message": message
        });

        Ok(ToolOutput::json(result))
    }
}
