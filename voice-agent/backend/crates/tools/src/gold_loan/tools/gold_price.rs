//! Gold Price Tool
//!
//! Get current gold prices per gram for different purities.

use async_trait::async_trait;
use chrono::Utc;
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

/// Get gold price tool
///
/// P15 FIX: ToolsDomainView is now REQUIRED - no more hardcoded fallbacks
pub struct GetGoldPriceTool {
    price_service: Option<Arc<dyn voice_agent_persistence::GoldPriceService>>,
    view: Arc<ToolsDomainView>,
}

impl GetGoldPriceTool {
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
        service: Arc<dyn voice_agent_persistence::GoldPriceService>,
        view: Arc<ToolsDomainView>,
    ) -> Self {
        Self {
            price_service: Some(service),
            view,
        }
    }

    /// Alias for with_price_service - clearer naming
    pub fn with_service_and_view(
        service: Arc<dyn voice_agent_persistence::GoldPriceService>,
        view: Arc<ToolsDomainView>,
    ) -> Self {
        Self::with_price_service(service, view)
    }

    /// Get fallback base price from config
    fn fallback_base_price(&self) -> f64 {
        self.view.gold_price_per_gram()
    }

    /// Get purity factor from config
    fn purity_factor(&self, purity: &str) -> f64 {
        self.view.purity_factor(purity)
    }
}

#[async_trait]
impl Tool for GetGoldPriceTool {
    fn name(&self) -> &str {
        "get_gold_price"
    }

    fn description(&self) -> &str {
        "Get current gold prices per gram for different purities (24K, 22K, 18K)"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "purity",
                    PropertySchema::enum_type(
                        "Gold purity to get price for (optional, returns all if not specified)",
                        vec!["24K".into(), "22K".into(), "18K".into()],
                    ),
                    false,
                )
                .property(
                    "weight_grams",
                    PropertySchema::number("Optional weight to calculate total value"),
                    false,
                ),
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

        if let Some(p) = purity {
            let price = match p {
                "24K" => price_24k,
                "22K" => price_22k,
                "18K" => price_18k,
                _ => price_22k,
            };
            result["requested_purity"] = json!(p);
            result["message"] = json!(format!(
                "Current {} gold price is ₹{:.0} per gram.",
                p, price
            ));
        } else {
            result["message"] = json!(format!(
                "Current gold prices - 24K: ₹{:.0}/g, 22K: ₹{:.0}/g, 18K: ₹{:.0}/g",
                price_24k, price_22k, price_18k
            ));
        }

        Ok(ToolOutput::json(result))
    }
}
