//! Branch Locator Tool
//!
//! Find nearby bank branches offering gold loan services.

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

use super::super::branches::{get_branches, BranchData};

/// Branch locator tool
pub struct BranchLocatorTool;

impl BranchLocatorTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for BranchLocatorTool {
    fn name(&self) -> &str {
        "find_branches"
    }

    fn description(&self) -> &str {
        "Find nearby bank branches offering gold loan services"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("city", PropertySchema::string("City name"), true)
                .property("area", PropertySchema::string("Area or locality"), false)
                .property("pincode", PropertySchema::string("6-digit PIN code"), false)
                .property(
                    "max_results",
                    PropertySchema::integer("Maximum results to return").with_default(json!(5)),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let city = input
            .get("city")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("city is required"))?;

        let area = input.get("area").and_then(|v| v.as_str());
        let pincode = input.get("pincode").and_then(|v| v.as_str());
        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_i64())
            .unwrap_or(5) as usize;

        let branches = filter_branches_json(city, area, pincode, max_results);

        let result = json!({
            "city": city,
            "area": area,
            "branches_found": branches.len(),
            "branches": branches,
            "message": if branches.is_empty() {
                format!("No branches found in {}. Please try a nearby city.", city)
            } else {
                format!("Found {} branches in {}.", branches.len(), city)
            }
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for BranchLocatorTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter branches and return as JSON values for tool output
fn filter_branches_json(
    city: &str,
    area: Option<&str>,
    pincode: Option<&str>,
    max: usize,
) -> Vec<Value> {
    let city_lower = city.to_lowercase();
    let branches = get_branches();

    let mut filtered: Vec<BranchData> = branches
        .into_iter()
        .filter(|b| {
            b.city.to_lowercase().contains(&city_lower)
                || city_lower.contains(&b.city.to_lowercase())
        })
        .collect();

    if let Some(pin) = pincode {
        let pin_matches: Vec<BranchData> = filtered
            .iter()
            .filter(|b| b.pincode == pin)
            .cloned()
            .collect();
        if !pin_matches.is_empty() {
            filtered = pin_matches;
        }
    }

    if let Some(area_str) = area {
        let area_lower = area_str.to_lowercase();
        let area_matches: Vec<BranchData> = filtered
            .iter()
            .filter(|b| b.area.to_lowercase().contains(&area_lower))
            .cloned()
            .collect();
        if !area_matches.is_empty() {
            filtered = area_matches;
        }
    }

    filtered.truncate(max);
    filtered
        .into_iter()
        .map(|b| {
            json!({
                "branch_id": b.branch_id,
                "name": b.name,
                "city": b.city,
                "area": b.area,
                "address": b.address,
                "pincode": b.pincode,
                "phone": b.phone,
                "gold_loan_available": b.gold_loan_available,
                "timing": b.timing,
                "facilities": b.facilities
            })
        })
        .collect()
}
