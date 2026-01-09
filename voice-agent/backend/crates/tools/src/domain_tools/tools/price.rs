//! Price Tool
//!
//! Get current asset prices per gram for different purities/variants.
//! All schema content (names, descriptions, parameters) comes from YAML config.

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Tool name as defined in config - used to look up schema
/// Note: Must match the key in schemas.yaml for this domain
const TOOL_NAME: &str = "get_price";

/// Get asset price tool (generic name for domain-agnostic code)
///
/// P15 FIX: ToolsDomainView is now REQUIRED - no more hardcoded fallbacks
pub struct GetPriceTool {
    price_service: Option<Arc<dyn voice_agent_persistence::AssetPriceService>>,
    view: Arc<ToolsDomainView>,
}

impl GetPriceTool {
    /// Create with required ToolsDomainView - domain config is mandatory
    pub fn new(view: Arc<ToolsDomainView>) -> Self {
        Self {
            price_service: None,
            view,
        }
    }

    /// Alias for new() for backwards compatibility during migration
    pub fn with_view(view: Arc<ToolsDomainView>) -> Self {
        Self::new(view)
    }

    /// Create with price service and required view
    pub fn with_price_service(
        service: Arc<dyn voice_agent_persistence::AssetPriceService>,
        view: Arc<ToolsDomainView>,
    ) -> Self {
        Self {
            price_service: Some(service),
            view,
        }
    }

    /// Alias for with_price_service - clearer naming
    pub fn with_service_and_view(
        service: Arc<dyn voice_agent_persistence::AssetPriceService>,
        view: Arc<ToolsDomainView>,
    ) -> Self {
        Self::with_price_service(service, view)
    }

    /// Get fallback base price from config
    fn fallback_base_price(&self) -> f64 {
        self.view.asset_price_per_unit()
    }

    /// Get purity/variant factor from config
    fn purity_factor(&self, purity: &str) -> f64 {
        self.view.purity_factor(purity)
    }
}

#[async_trait]
impl Tool for GetPriceTool {
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
        "Get current prices per gram for different purities"
    }

    fn schema(&self) -> ToolSchema {
        // P16 FIX: Read schema from config - all content comes from YAML
        if let Some(core_schema) = self.view.tools_config().get_core_schema(TOOL_NAME) {
            core_schema
        } else {
            // Fallback if config not available (should not happen in production)
            tracing::warn!("Tool schema not found in config for {}, using fallback", TOOL_NAME);
            ToolSchema {
                name: TOOL_NAME.to_string(),
                description: "Get current prices".to_string(),
                input_schema: InputSchema::object()
                    .property(
                        "purity",
                        PropertySchema::enum_type(
                            "Purity to get price for",
                            vec!["24K".into(), "22K".into(), "18K".into()],
                        ),
                        false,
                    )
                    .property(
                        "weight_grams",
                        PropertySchema::number("Weight to calculate total value"),
                        false,
                    ),
            }
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let purity = input.get("purity").and_then(|v| v.as_str());
        let weight = input.get("weight_grams").and_then(|v| v.as_f64());

        // P14 FIX: Use config-driven fallback prices and purity factors
        let (price_24k, price_22k, price_18k, source) =
            if let Some(ref service) = self.price_service {
                match service.get_current_price().await {
                    Ok(price) => (
                        price.price_24k,
                        price.price_22k,
                        price.price_18k,
                        price.source,
                    ),
                    Err(e) => {
                        tracing::warn!("Failed to get gold price from service: {}", e);
                        let base = self.fallback_base_price();
                        (
                            base * self.purity_factor("24K"),
                            base * self.purity_factor("22K"),
                            base * self.purity_factor("18K"),
                            "fallback".to_string(),
                        )
                    }
                }
            } else {
                let base = self.fallback_base_price();
                (
                    base * self.purity_factor("24K"),
                    base * self.purity_factor("22K"),
                    base * self.purity_factor("18K"),
                    "fallback".to_string(),
                )
            };

        let mut result = json!({
            "prices": {
                "24K": {
                    "price_per_gram_inr": price_24k.round(),
                    "description": "Pure gold (99.9%)"
                },
                "22K": {
                    "price_per_gram_inr": price_22k.round(),
                    "description": "Standard jewelry gold (91.6%)"
                },
                "18K": {
                    "price_per_gram_inr": price_18k.round(),
                    "description": "Fashion jewelry gold (75%)"
                }
            },
            "source": source,
            "updated_at": Utc::now().to_rfc3339(),
            "disclaimer": "Prices are indicative. Final value determined at branch during valuation."
        });

        if let Some(w) = weight {
            let values = json!({
                "24K": (w * price_24k).round(),
                "22K": (w * price_22k).round(),
                "18K": (w * price_18k).round()
            });
            result["estimated_values_inr"] = values;
            result["weight_grams"] = json!(w);
        }

        // P16 FIX: Use config-driven response templates
        if let Some(p) = purity {
            let price = match p {
                "24K" => price_24k,
                "22K" => price_22k,
                "18K" => price_18k,
                _ => price_22k,
            };
            result["requested_purity"] = json!(p);
            let message = if self.view.has_response_templates("get_price") {
                let mut vars = self.view.default_template_vars();
                vars.insert("purity".to_string(), p.to_string());
                vars.insert("price".to_string(), format!("{:.0}", price));
                self.view.render_response("get_price", "single_purity", "en", &vars)
                    .unwrap_or_else(|| format!("Current {} gold price is ₹{:.0} per gram.", p, price))
            } else {
                format!("Current {} gold price is ₹{:.0} per gram.", p, price)
            };
            result["message"] = json!(message);
        } else {
            let message = if self.view.has_response_templates("get_price") {
                let mut vars = self.view.default_template_vars();
                vars.insert("price_24k".to_string(), format!("{:.0}", price_24k));
                vars.insert("price_22k".to_string(), format!("{:.0}", price_22k));
                vars.insert("price_18k".to_string(), format!("{:.0}", price_18k));
                self.view.render_response("get_price", "all_prices", "en", &vars)
                    .unwrap_or_else(|| format!(
                        "Current gold prices - 24K: ₹{:.0}/g, 22K: ₹{:.0}/g, 18K: ₹{:.0}/g",
                        price_24k, price_22k, price_18k
                    ))
            } else {
                format!(
                    "Current gold prices - 24K: ₹{:.0}/g, 22K: ₹{:.0}/g, 18K: ₹{:.0}/g",
                    price_24k, price_22k, price_18k
                )
            };
            result["message"] = json!(message);
        }

        Ok(ToolOutput::json(result))
    }
}
