//! Competitor Comparison Tool
//!
//! Compare gold loan offerings with other major lenders.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Competitor comparison tool
///
/// P13 FIX: Uses ToolsDomainView instead of GoldLoanConfig
/// P15 FIX: ToolsDomainView is now REQUIRED - no more hardcoded fallbacks
pub struct CompetitorComparisonTool {
    view: Arc<ToolsDomainView>,
}

impl CompetitorComparisonTool {
    /// Create with required ToolsDomainView - domain config is mandatory
    pub fn new(view: Arc<ToolsDomainView>) -> Self {
        Self { view }
    }

    /// Alias for new() for backwards compatibility during migration
    pub fn with_view(view: Arc<ToolsDomainView>) -> Self {
        Self::new(view)
    }

    fn get_our_rate(&self) -> f64 {
        self.view.base_interest_rate()
    }

    fn get_our_ltv(&self) -> f64 {
        self.view.ltv_percent()
    }

    fn bank_name(&self) -> &str {
        self.view.bank_name()
    }

    /// Get competitors from config - no fallback
    fn get_competitors(&self) -> Vec<(String, String, f64, f64, Vec<String>)> {
        self.view
            .all_competitors_data()
            .into_iter()
            .map(|(id, name, rate, ltv, features)| {
                (
                    id.to_string(),
                    name.to_string(),
                    rate,
                    ltv,
                    features.into_iter().map(|s| s.to_string()).collect(),
                )
            })
            .collect()
    }

    /// Get competitor IDs for schema enum - from config
    fn get_competitor_ids(&self) -> Vec<String> {
        self.view.competitor_ids()
    }

    /// Get our features from config - no fallback
    fn get_our_features(&self) -> Vec<String> {
        self.view.our_features().to_vec()
    }
}

#[async_trait]
impl Tool for CompetitorComparisonTool {
    fn name(&self) -> &str {
        "compare_lenders"
    }

    fn description(&self) -> &str {
        "Compare our gold loan offerings with other major lenders including interest rates, LTV, and features"
    }

    fn schema(&self) -> ToolSchema {
        // P15 FIX: Build competitor enum from config dynamically
        let mut competitor_options: Vec<String> = self.get_competitor_ids();
        competitor_options.push("all".to_string());

        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "competitor",
                    PropertySchema::enum_type(
                        "Competitor to compare with",
                        competitor_options,
                    ),
                    false,
                )
                .property(
                    "loan_amount",
                    PropertySchema::number("Loan amount for comparison").with_default(json!(100000)),
                    false,
                )
                .property(
                    "tenure_months",
                    PropertySchema::integer("Tenure in months").with_default(json!(12)),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let competitor = input
            .get("competitor")
            .and_then(|v| v.as_str())
            .unwrap_or("all");

        let loan_amount = input
            .get("loan_amount")
            .and_then(|v| v.as_f64())
            .unwrap_or(100000.0);

        let tenure_months = input
            .get("tenure_months")
            .and_then(|v| v.as_i64())
            .unwrap_or(12);

        // P15 FIX: All values from config, no hardcoded fallbacks
        let competitors = self.get_competitors();
        let our_rate = self.get_our_rate();
        let our_ltv = self.get_our_ltv();
        let bank_name = self.bank_name();

        let selected_competitors: Vec<_> = if competitor == "all" {
            competitors.clone()
        } else {
            competitors
                .iter()
                .filter(|(id, _, _, _, _)| id == competitor)
                .cloned()
                .collect()
        };

        let our_monthly_interest = loan_amount * our_rate / 100.0 / 12.0;
        let our_annual_interest = loan_amount * our_rate / 100.0;

        let mut comparisons: Vec<Value> = vec![];
        let mut our_advantages: Vec<String> = vec![];

        for (id, name, rate, ltv, features) in selected_competitors {
            let competitor_monthly = loan_amount * rate / 100.0 / 12.0;
            let competitor_annual = loan_amount * rate / 100.0;
            let monthly_savings = competitor_monthly - our_monthly_interest;
            let annual_savings = competitor_annual - our_annual_interest;

            let comparison = json!({
                "lender_id": id,
                "lender_name": name,
                "interest_rate": rate,
                "ltv_percent": ltv,
                "features": features,
                "monthly_interest": competitor_monthly,
                "annual_interest": competitor_annual,
                "vs_us": {
                    "rate_difference": rate - our_rate,
                    "monthly_savings": monthly_savings,
                    "annual_savings": annual_savings,
                    "tenure_savings": monthly_savings * tenure_months as f64,
                    "we_are_cheaper": our_rate < rate
                }
            });
            comparisons.push(comparison);

            if our_rate < rate {
                our_advantages.push(format!(
                    "{}% lower rate than {} (saving ₹{:.0}/month)",
                    ((rate - our_rate) * 100.0).round() / 100.0,
                    name,
                    monthly_savings
                ));
            }
        }

        // P15 FIX: Get our features from config
        let our_features = self.get_our_features();

        let result = json!({
            "comparison_for": {
                "loan_amount": loan_amount,
                "tenure_months": tenure_months
            },
            "our_bank": {
                "name": bank_name,
                "interest_rate": our_rate,
                "ltv_percent": our_ltv,
                "monthly_interest": our_monthly_interest,
                "annual_interest": our_annual_interest,
                "features": our_features
            },
            "competitors": comparisons,
            "our_advantages": our_advantages,
            "summary": format!(
                "For a loan of ₹{:.0}, {} offers {}% p.a. with monthly interest of ₹{:.0}. {}",
                loan_amount,
                bank_name,
                our_rate,
                our_monthly_interest,
                if our_advantages.is_empty() {
                    format!("{} rates are competitive with the market.", bank_name)
                } else {
                    format!("You can save compared to: {}", our_advantages.join(", "))
                }
            )
        });

        Ok(ToolOutput::json(result))
    }
}
