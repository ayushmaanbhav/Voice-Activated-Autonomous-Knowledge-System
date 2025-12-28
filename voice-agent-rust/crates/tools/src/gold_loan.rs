//! Gold Loan Domain Tools
//!
//! Specific tools for the gold loan voice agent.

use async_trait::async_trait;
use chrono::{NaiveDate, Utc};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::RwLock;
use voice_agent_config::GoldLoanConfig;

use crate::mcp::{Tool, ToolSchema, ToolOutput, ToolError, InputSchema, PropertySchema};

/// P0 FIX: Branch data structure for JSON loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchData {
    pub branch_id: String,
    pub name: String,
    pub city: String,
    pub area: String,
    pub address: String,
    #[serde(default)]
    pub pincode: String,
    pub phone: String,
    pub gold_loan_available: bool,
    pub timing: String,
    #[serde(default)]
    pub facilities: Vec<String>,
}

/// Branch data file structure
#[derive(Debug, Deserialize)]
struct BranchDataFile {
    branches: Vec<BranchData>,
}

/// P0 FIX: Global branch data loaded from JSON
static BRANCH_DATA: Lazy<RwLock<Vec<BranchData>>> = Lazy::new(|| {
    // Try to load from default paths
    let default_paths = [
        "data/branches.json",
        "../data/branches.json",
        "../../data/branches.json",
        "./branches.json",
    ];

    for path in &default_paths {
        if let Ok(data) = load_branches_from_file(path) {
            tracing::info!("Loaded {} branches from {}", data.len(), path);
            return RwLock::new(data);
        }
    }

    // Fall back to embedded default data
    tracing::warn!("Could not load branches from file, using embedded defaults");
    RwLock::new(get_default_branches())
});

/// Load branches from a JSON file
pub fn load_branches_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<BranchData>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let file: BranchDataFile = serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(file.branches)
}

/// Reload branches from a file (for runtime updates)
pub fn reload_branches<P: AsRef<Path>>(path: P) -> Result<usize, std::io::Error> {
    let branches = load_branches_from_file(path)?;
    let count = branches.len();
    *BRANCH_DATA.write().unwrap() = branches;
    Ok(count)
}

/// Get all loaded branches
pub fn get_branches() -> Vec<BranchData> {
    BRANCH_DATA.read().unwrap().clone()
}

/// Get default embedded branches (fallback)
fn get_default_branches() -> Vec<BranchData> {
    vec![
        BranchData {
            branch_id: "KMBL001".to_string(),
            name: "Kotak Mahindra Bank - Andheri West".to_string(),
            city: "Mumbai".to_string(),
            area: "Andheri West".to_string(),
            address: "Ground Floor, Kora Kendra, S.V. Road, Andheri West, Mumbai - 400058".to_string(),
            pincode: "400058".to_string(),
            phone: "022-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
        BranchData {
            branch_id: "KMBL101".to_string(),
            name: "Kotak Mahindra Bank - Connaught Place".to_string(),
            city: "Delhi".to_string(),
            area: "Connaught Place".to_string(),
            address: "M-Block, Connaught Place, New Delhi - 110001".to_string(),
            pincode: "110001".to_string(),
            phone: "011-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
        BranchData {
            branch_id: "KMBL201".to_string(),
            name: "Kotak Mahindra Bank - MG Road".to_string(),
            city: "Bangalore".to_string(),
            area: "MG Road".to_string(),
            address: "Church Street, MG Road, Bangalore - 560001".to_string(),
            pincode: "560001".to_string(),
            phone: "080-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
        BranchData {
            branch_id: "KMBL301".to_string(),
            name: "Kotak Mahindra Bank - T Nagar".to_string(),
            city: "Chennai".to_string(),
            area: "T Nagar".to_string(),
            address: "Usman Road, T Nagar, Chennai - 600017".to_string(),
            pincode: "600017".to_string(),
            phone: "044-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
    ]
}

/// Check eligibility tool
pub struct EligibilityCheckTool {
    config: GoldLoanConfig,
}

impl EligibilityCheckTool {
    pub fn new() -> Self {
        Self {
            config: GoldLoanConfig::default(),
        }
    }

    pub fn with_config(config: GoldLoanConfig) -> Self {
        Self { config }
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
                .property("gold_weight_grams", PropertySchema::number("Gold weight in grams"), true)
                .property("gold_purity", PropertySchema::enum_type(
                    "Gold purity (22K, 18K, etc.)",
                    vec!["24K".into(), "22K".into(), "18K".into(), "14K".into()]
                ).with_default(json!("22K")), false)
                .property("existing_loan_amount", PropertySchema::number("Existing loan amount if any"), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let weight: f64 = input.get("gold_weight_grams")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("gold_weight_grams is required"))?;

        let purity = input.get("gold_purity")
            .and_then(|v| v.as_str())
            .unwrap_or("22K");

        let existing_loan = input.get("existing_loan_amount")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Calculate eligibility using config values
        let gold_value = self.config.calculate_gold_value(weight, purity);
        let max_loan = self.config.calculate_max_loan(gold_value);
        let available_loan = max_loan - existing_loan;

        let result = json!({
            "eligible": available_loan >= self.config.min_loan_amount,
            "gold_value_inr": gold_value.round(),
            "max_loan_amount_inr": max_loan.round(),
            "existing_loan_inr": existing_loan,
            "available_loan_inr": available_loan.max(0.0).round(),
            "ltv_percent": self.config.ltv_percent,
            "interest_rate_percent": self.config.kotak_interest_rate,
            "processing_fee_percent": self.config.processing_fee_percent,
            "message": if available_loan >= self.config.min_loan_amount {
                format!("You are eligible for a gold loan up to ₹{:.0}!", available_loan)
            } else if available_loan > 0.0 {
                format!("You can get an additional ₹{:.0} on your gold.", available_loan)
            } else {
                "Based on your existing loan, no additional loan is available at this time.".to_string()
            }
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for EligibilityCheckTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Savings calculator tool
pub struct SavingsCalculatorTool {
    config: GoldLoanConfig,
}

impl SavingsCalculatorTool {
    pub fn new() -> Self {
        Self {
            config: GoldLoanConfig::default(),
        }
    }

    pub fn with_config(config: GoldLoanConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for SavingsCalculatorTool {
    fn name(&self) -> &str {
        "calculate_savings"
    }

    fn description(&self) -> &str {
        "Calculate potential savings when switching from NBFC to Kotak gold loan"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("current_loan_amount", PropertySchema::number("Current loan amount in INR"), true)
                .property("current_interest_rate", PropertySchema::number("Current interest rate (%)").with_range(10.0, 30.0), true)
                .property("remaining_tenure_months", PropertySchema::integer("Remaining tenure in months"), true)
                .property("current_lender", PropertySchema::enum_type(
                    "Current lender",
                    vec!["Muthoot".into(), "Manappuram".into(), "IIFL".into(), "Other NBFC".into()]
                ), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let loan_amount: f64 = input.get("current_loan_amount")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("current_loan_amount is required"))?;

        // Get current rate - either from input or infer from lender
        let current_lender = input.get("current_lender")
            .and_then(|v| v.as_str())
            .unwrap_or("Other NBFC");

        let current_rate: f64 = input.get("current_interest_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| self.config.get_competitor_rate(current_lender));

        let tenure_months: i64 = input.get("remaining_tenure_months")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| ToolError::invalid_params("remaining_tenure_months is required"))?;

        // Use config for Kotak rate
        let kotak_rate = self.config.kotak_interest_rate;

        // Calculate monthly savings using config helper
        let monthly_savings = self.config.calculate_monthly_savings(loan_amount, current_rate);

        // Calculate monthly payments
        let current_monthly = loan_amount * (current_rate / 100.0 / 12.0);
        let kotak_monthly = loan_amount * (kotak_rate / 100.0 / 12.0);

        // Total interest over remaining tenure
        let total_savings = monthly_savings * tenure_months as f64;

        let result = json!({
            "current_lender": current_lender,
            "current_interest_rate_percent": current_rate,
            "kotak_interest_rate_percent": kotak_rate,
            "rate_reduction_percent": current_rate - kotak_rate,
            "current_monthly_interest_inr": current_monthly.round(),
            "kotak_monthly_interest_inr": kotak_monthly.round(),
            "monthly_savings_inr": monthly_savings.round(),
            "total_savings_inr": total_savings.round(),
            "tenure_months": tenure_months,
            "message": format!(
                "By switching to Kotak, you can save ₹{:.0} per month and ₹{:.0} over the remaining {} months!",
                monthly_savings, total_savings, tenure_months
            )
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for SavingsCalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Lead capture tool
pub struct LeadCaptureTool;

impl LeadCaptureTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for LeadCaptureTool {
    fn name(&self) -> &str {
        "capture_lead"
    }

    fn description(&self) -> &str {
        "Capture customer lead information for follow-up"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("customer_name", PropertySchema::string("Customer's full name"), true)
                .property("phone_number", PropertySchema::string("10-digit mobile number"), true)
                .property("city", PropertySchema::string("Customer's city"), false)
                .property("preferred_branch", PropertySchema::string("Preferred branch location"), false)
                .property("estimated_gold_weight", PropertySchema::number("Estimated gold weight in grams"), false)
                .property("interest_level", PropertySchema::enum_type(
                    "Customer's interest level",
                    vec!["High".into(), "Medium".into(), "Low".into()]
                ), false)
                .property("notes", PropertySchema::string("Additional notes from conversation"), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input.get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input.get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        // Validate phone number
        if phone.len() != 10 || !phone.chars().all(|c| c.is_ascii_digit()) {
            return Err(ToolError::invalid_params("phone_number must be 10 digits"));
        }

        // Generate lead ID
        let lead_id = format!("GL{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

        let result = json!({
            "success": true,
            "lead_id": lead_id,
            "customer_name": name,
            "phone_number": phone,
            "city": input.get("city").and_then(|v| v.as_str()),
            "preferred_branch": input.get("preferred_branch").and_then(|v| v.as_str()),
            "estimated_gold_weight": input.get("estimated_gold_weight").and_then(|v| v.as_f64()),
            "interest_level": input.get("interest_level").and_then(|v| v.as_str()).unwrap_or("Medium"),
            "notes": input.get("notes").and_then(|v| v.as_str()),
            "created_at": Utc::now().to_rfc3339(),
            "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for LeadCaptureTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Appointment scheduler tool
pub struct AppointmentSchedulerTool;

impl AppointmentSchedulerTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for AppointmentSchedulerTool {
    fn name(&self) -> &str {
        "schedule_appointment"
    }

    fn description(&self) -> &str {
        "Schedule a branch visit appointment for gold valuation"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("customer_name", PropertySchema::string("Customer's name"), true)
                .property("phone_number", PropertySchema::string("Contact number"), true)
                .property("branch_id", PropertySchema::string("Branch ID or location"), true)
                .property("preferred_date", PropertySchema::string("Preferred date (YYYY-MM-DD)"), true)
                .property("preferred_time", PropertySchema::enum_type(
                    "Preferred time slot",
                    vec!["10:00 AM".into(), "11:00 AM".into(), "12:00 PM".into(),
                         "2:00 PM".into(), "3:00 PM".into(), "4:00 PM".into(), "5:00 PM".into()]
                ), true)
                .property("purpose", PropertySchema::enum_type(
                    "Purpose of visit",
                    vec!["New Gold Loan".into(), "Gold Loan Transfer".into(), "Top-up".into(), "Closure".into()]
                ), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input.get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input.get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        let branch = input.get("branch_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("branch_id is required"))?;

        let date_str = input.get("preferred_date")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_date is required"))?;

        // P1 FIX: Validate date format and ensure it's not in the past
        let parsed_date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d-%m-%Y"))
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d/%m/%Y"))
            .map_err(|_| ToolError::invalid_params(
                "preferred_date must be in format YYYY-MM-DD, DD-MM-YYYY, or DD/MM/YYYY"
            ))?;

        let today = Utc::now().date_naive();
        if parsed_date < today {
            return Err(ToolError::invalid_params("preferred_date cannot be in the past"));
        }

        // Use standardized format
        let date = parsed_date.format("%Y-%m-%d").to_string();

        let time = input.get("preferred_time")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_time is required"))?;

        let purpose = input.get("purpose")
            .and_then(|v| v.as_str())
            .unwrap_or("New Gold Loan");

        // Generate appointment ID
        let appointment_id = format!("APT{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

        let result = json!({
            "success": true,
            "appointment_id": appointment_id,
            "customer_name": name,
            "phone_number": phone,
            "branch_id": branch,
            "date": date,
            "time": time,
            "purpose": purpose,
            "confirmation_sent": true,
            "message": format!(
                "Appointment scheduled for {} on {} at {}. Confirmation SMS sent to {}.",
                name, date, time, phone
            )
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for AppointmentSchedulerTool {
    fn default() -> Self {
        Self::new()
    }
}

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
        "Find nearby Kotak Mahindra Bank branches offering gold loan services"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("city", PropertySchema::string("City name"), true)
                .property("area", PropertySchema::string("Area or locality"), false)
                .property("pincode", PropertySchema::string("6-digit PIN code"), false)
                .property("max_results", PropertySchema::integer("Maximum results to return").with_default(json!(5)), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let city = input.get("city")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("city is required"))?;

        let area = input.get("area").and_then(|v| v.as_str());
        let pincode = input.get("pincode").and_then(|v| v.as_str());
        let max_results = input.get("max_results")
            .and_then(|v| v.as_i64())
            .unwrap_or(5) as usize;

        // Mock branch data (in production, this would query a database or API)
        let branches = get_mock_branches(city, area, pincode, max_results);

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

/// P0 FIX: Get branches from JSON data instead of hardcoded mock
fn get_mock_branches(city: &str, area: Option<&str>, pincode: Option<&str>, max: usize) -> Vec<Value> {
    let city_lower = city.to_lowercase();
    let branches = get_branches();

    // Filter by city
    let mut filtered: Vec<BranchData> = branches
        .into_iter()
        .filter(|b| {
            b.city.to_lowercase().contains(&city_lower) ||
            city_lower.contains(&b.city.to_lowercase())
        })
        .collect();

    // Filter by pincode if provided (exact match)
    if let Some(pin) = pincode {
        let pin_matches: Vec<BranchData> = filtered.iter()
            .filter(|b| b.pincode == pin)
            .cloned()
            .collect();
        if !pin_matches.is_empty() {
            filtered = pin_matches;
        }
    }

    // Filter by area if provided
    if let Some(area_str) = area {
        let area_lower = area_str.to_lowercase();
        let area_matches: Vec<BranchData> = filtered.iter()
            .filter(|b| b.area.to_lowercase().contains(&area_lower))
            .cloned()
            .collect();
        if !area_matches.is_empty() {
            filtered = area_matches;
        }
    }

    // Convert to JSON Value and truncate
    filtered.truncate(max);
    filtered.into_iter()
        .map(|b| json!({
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
        }))
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_eligibility_check() {
        let tool = EligibilityCheckTool::new();
        let input = json!({
            "gold_weight_grams": 50.0,
            "gold_purity": "22K"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_savings_calculator() {
        let tool = SavingsCalculatorTool::new();
        let input = json!({
            "current_loan_amount": 100000.0,
            "current_interest_rate": 18.0,
            "remaining_tenure_months": 12,
            "current_lender": "Muthoot"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_lead_capture() {
        let tool = LeadCaptureTool::new();
        let input = json!({
            "customer_name": "Rajesh Kumar",
            "phone_number": "9876543210",
            "city": "Mumbai"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_branch_locator() {
        let tool = BranchLocatorTool::new();
        let input = json!({
            "city": "Mumbai"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }
}
