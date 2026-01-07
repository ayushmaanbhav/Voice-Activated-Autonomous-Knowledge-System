//! Gold Loan Tool Implementations
//!
//! MCP-compatible tools for the gold loan voice agent.

use crate::integrations::{
    Appointment, AppointmentPurpose, AppointmentStatus, CalendarIntegration, CrmIntegration,
    CrmLead, InterestLevel, LeadSource, LeadStatus,
};
use async_trait::async_trait;
use chrono::{NaiveDate, Utc};
use serde_json::{json, Value};
use std::sync::Arc;
use voice_agent_config::ToolsDomainView;

use crate::mcp::{InputSchema, PropertySchema, Tool, ToolError, ToolOutput, ToolSchema};

use super::branches::{get_branches, BranchData};
use super::utils::{calculate_emi, calculate_total_interest};

/// Check eligibility tool
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

// P15 FIX: Removed Default impl - ToolsDomainView is required

/// Savings calculator tool
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

    fn bank_name(&self) -> &str {
        self.view.bank_name()
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
        let bank_name = self.bank_name();

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
            "bank_name": bank_name,
            "message": format!(
                "By switching to {} at our {} rate of {}%, you can save ₹{:.0} per month on EMI (or ₹{:.0} on interest-only) and ₹{:.0} total over the remaining {} months!",
                bank_name, rate_tier, our_rate, emi_savings, monthly_interest_savings, total_emi_savings, tenure_months
            )
        });

        Ok(ToolOutput::json(result))
    }
}

// P15 FIX: Removed Default impl - ToolsDomainView is required

/// Lead capture tool
pub struct LeadCaptureTool {
    crm: Option<Arc<dyn CrmIntegration>>,
}

impl LeadCaptureTool {
    pub fn new() -> Self {
        Self { crm: None }
    }

    pub fn with_crm(crm: Arc<dyn CrmIntegration>) -> Self {
        Self { crm: Some(crm) }
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
                .property(
                    "customer_name",
                    PropertySchema::string("Customer's full name"),
                    true,
                )
                .property(
                    "phone_number",
                    PropertySchema::string("10-digit mobile number"),
                    true,
                )
                .property("city", PropertySchema::string("Customer's city"), false)
                .property(
                    "preferred_branch",
                    PropertySchema::string("Preferred branch location"),
                    false,
                )
                .property(
                    "estimated_gold_weight",
                    PropertySchema::number("Estimated gold weight in grams"),
                    false,
                )
                .property(
                    "interest_level",
                    PropertySchema::enum_type(
                        "Customer's interest level",
                        vec!["High".into(), "Medium".into(), "Low".into()],
                    ),
                    false,
                )
                .property(
                    "notes",
                    PropertySchema::string("Additional notes from conversation"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input
            .get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input
            .get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        if phone.len() != 10 || !phone.chars().all(|c| c.is_ascii_digit()) {
            return Err(ToolError::invalid_params("phone_number must be 10 digits"));
        }

        let city = input.get("city").and_then(|v| v.as_str()).map(String::from);
        let estimated_gold = input.get("estimated_gold_weight").and_then(|v| v.as_f64());
        let notes = input
            .get("notes")
            .and_then(|v| v.as_str())
            .map(String::from);
        let interest_str = input
            .get("interest_level")
            .and_then(|v| v.as_str())
            .unwrap_or("Medium");

        let interest_level = match interest_str.to_lowercase().as_str() {
            "high" => InterestLevel::High,
            "low" => InterestLevel::Low,
            _ => InterestLevel::Medium,
        };

        if let Some(ref crm) = self.crm {
            let lead = CrmLead {
                id: None,
                name: name.to_string(),
                phone: phone.to_string(),
                email: None,
                city,
                source: LeadSource::VoiceAgent,
                interest_level,
                estimated_gold_grams: estimated_gold,
                current_lender: None,
                notes,
                assigned_to: None,
                status: LeadStatus::New,
            };

            match crm.create_lead(lead).await {
                Ok(lead_id) => {
                    let result = json!({
                        "success": true,
                        "lead_id": lead_id,
                        "customer_name": name,
                        "phone_number": phone,
                        "city": input.get("city").and_then(|v| v.as_str()),
                        "interest_level": interest_str,
                        "estimated_gold_weight": estimated_gold,
                        "created_at": Utc::now().to_rfc3339(),
                        "crm_integrated": true,
                        "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
                    });
                    return Ok(ToolOutput::json(result));
                }
                Err(e) => {
                    tracing::warn!("CRM integration failed, falling back to local: {}", e);
                }
            }
        }

        let lead_id = format!("GL{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

        let result = json!({
            "success": true,
            "lead_id": lead_id,
            "customer_name": name,
            "phone_number": phone,
            "city": input.get("city").and_then(|v| v.as_str()),
            "preferred_branch": input.get("preferred_branch").and_then(|v| v.as_str()),
            "estimated_gold_weight": estimated_gold,
            "interest_level": interest_str,
            "notes": input.get("notes").and_then(|v| v.as_str()),
            "created_at": Utc::now().to_rfc3339(),
            "crm_integrated": false,
            "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        45
    }
}

impl Default for LeadCaptureTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Appointment scheduler tool
pub struct AppointmentSchedulerTool {
    calendar: Option<Arc<dyn CalendarIntegration>>,
}

impl AppointmentSchedulerTool {
    pub fn new() -> Self {
        Self { calendar: None }
    }

    pub fn with_calendar(calendar: Arc<dyn CalendarIntegration>) -> Self {
        Self {
            calendar: Some(calendar),
        }
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
                .property(
                    "customer_name",
                    PropertySchema::string("Customer's name"),
                    true,
                )
                .property(
                    "phone_number",
                    PropertySchema::string("Contact number"),
                    true,
                )
                .property(
                    "branch_id",
                    PropertySchema::string("Branch ID or location"),
                    true,
                )
                .property(
                    "preferred_date",
                    PropertySchema::string("Preferred date (YYYY-MM-DD)"),
                    true,
                )
                .property(
                    "preferred_time",
                    PropertySchema::enum_type(
                        "Preferred time slot",
                        vec![
                            "10:00 AM".into(),
                            "11:00 AM".into(),
                            "12:00 PM".into(),
                            "2:00 PM".into(),
                            "3:00 PM".into(),
                            "4:00 PM".into(),
                            "5:00 PM".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "purpose",
                    PropertySchema::enum_type(
                        "Purpose of visit",
                        vec![
                            "New Gold Loan".into(),
                            "Gold Loan Transfer".into(),
                            "Top-up".into(),
                            "Closure".into(),
                        ],
                    ),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input
            .get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input
            .get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        let branch = input
            .get("branch_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("branch_id is required"))?;

        let date_str = input
            .get("preferred_date")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_date is required"))?;

        let parsed_date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d-%m-%Y"))
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d/%m/%Y"))
            .map_err(|_| {
                ToolError::invalid_params(
                    "preferred_date must be in format YYYY-MM-DD, DD-MM-YYYY, or DD/MM/YYYY",
                )
            })?;

        let today = Utc::now().date_naive();
        if parsed_date < today {
            return Err(ToolError::invalid_params(
                "preferred_date cannot be in the past",
            ));
        }

        let date = parsed_date.format("%Y-%m-%d").to_string();

        let time = input
            .get("preferred_time")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_time is required"))?;

        let purpose_str = input
            .get("purpose")
            .and_then(|v| v.as_str())
            .unwrap_or("New Gold Loan");

        let purpose_enum = match purpose_str {
            "Gold Loan Transfer" => AppointmentPurpose::GoldLoanTransfer,
            "Top-up" => AppointmentPurpose::TopUp,
            "Closure" => AppointmentPurpose::Closure,
            "Consultation" => AppointmentPurpose::Consultation,
            _ => AppointmentPurpose::NewGoldLoan,
        };

        if let Some(ref calendar) = self.calendar {
            let appointment = Appointment {
                id: None,
                customer_name: name.to_string(),
                customer_phone: phone.to_string(),
                branch_id: branch.to_string(),
                date: date.clone(),
                time_slot: time.to_string(),
                purpose: purpose_enum,
                notes: None,
                status: AppointmentStatus::Scheduled,
                confirmation_sent: false,
            };

            match calendar.schedule_appointment(appointment).await {
                Ok(appointment_id) => {
                    let confirmation_sent =
                        calendar.send_confirmation(&appointment_id).await.is_ok();

                    let result = json!({
                        "success": true,
                        "appointment_id": appointment_id,
                        "customer_name": name,
                        "phone_number": phone,
                        "branch_id": branch,
                        "date": date,
                        "time": time,
                        "purpose": purpose_str,
                        "confirmation_sent": confirmation_sent,
                        "calendar_integrated": true,
                        "status": "pending_confirmation",
                        "confirmation_method": "agent_will_call_to_confirm",
                        "next_action": "Agent will call customer to confirm appointment",
                        "message": if confirmation_sent {
                            format!(
                                "Appointment scheduled for {} on {} at {}. Confirmation sent to {}.",
                                name, date, time, phone
                            )
                        } else {
                            format!(
                                "Appointment scheduled for {} on {} at {}. Our team will call to confirm.",
                                name, date, time
                            )
                        }
                    });
                    return Ok(ToolOutput::json(result));
                }
                Err(e) => {
                    tracing::warn!("Calendar integration failed, falling back to local: {}", e);
                }
            }
        }

        let appointment_id = format!(
            "APT{}",
            uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
        );

        let result = json!({
            "success": true,
            "appointment_id": appointment_id,
            "customer_name": name,
            "phone_number": phone,
            "branch_id": branch,
            "date": date,
            "time": time,
            "purpose": purpose_str,
            "confirmation_sent": false,
            "calendar_integrated": false,
            "status": "pending_confirmation",
            "confirmation_method": "agent_will_call_to_confirm",
            "next_action": "Agent will call customer to confirm appointment",
            "message": format!(
                "Appointment scheduled for {} on {} at {}. Our team will call to confirm.",
                name, date, time
            )
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        60
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

/// Get gold price tool
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

// P15 FIX: Removed Default impl - ToolsDomainView is required

/// Human escalation tool
pub struct EscalateToHumanTool {
    on_escalate: Option<Arc<dyn Fn(String, String, String) + Send + Sync>>,
}

impl EscalateToHumanTool {
    pub fn new() -> Self {
        Self { on_escalate: None }
    }

    pub fn with_callback<F>(callback: F) -> Self
    where
        F: Fn(String, String, String) + Send + Sync + 'static,
    {
        Self {
            on_escalate: Some(Arc::new(callback)),
        }
    }
}

#[async_trait]
impl Tool for EscalateToHumanTool {
    fn name(&self) -> &str {
        "escalate_to_human"
    }

    fn description(&self) -> &str {
        "Escalate the conversation to a human agent when the customer requests it or when the AI cannot help"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "reason",
                    PropertySchema::enum_type(
                        "Reason for escalation",
                        vec![
                            "customer_request".into(),
                            "complex_query".into(),
                            "complaint".into(),
                            "technical_issue".into(),
                            "sensitive_matter".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "session_id",
                    PropertySchema::string("Current session ID"),
                    true,
                )
                .property(
                    "customer_phone",
                    PropertySchema::string("Customer phone number"),
                    false,
                )
                .property(
                    "summary",
                    PropertySchema::string("Brief summary of conversation so far"),
                    false,
                )
                .property(
                    "priority",
                    PropertySchema::enum_type(
                        "Escalation priority",
                        vec!["normal".into(), "high".into(), "urgent".into()],
                    )
                    .with_default(json!("normal")),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let reason = input
            .get("reason")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("reason is required"))?;

        let session_id = input
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("session_id is required"))?;

        let customer_phone = input
            .get("customer_phone")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let summary = input
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("No summary provided");

        let priority = input
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("normal");

        let escalation_id = format!(
            "ESC{}",
            uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
        );

        let estimated_wait = match priority {
            "urgent" => "1-2 minutes",
            "high" => "2-5 minutes",
            _ => "5-10 minutes",
        };

        if let Some(ref callback) = self.on_escalate {
            callback(
                escalation_id.clone(),
                session_id.to_string(),
                reason.to_string(),
            );
        }

        tracing::info!(
            escalation_id = %escalation_id,
            session_id = %session_id,
            reason = %reason,
            priority = %priority,
            "Human escalation requested"
        );

        let result = json!({
            "success": true,
            "escalation_id": escalation_id,
            "session_id": session_id,
            "customer_phone": customer_phone,
            "reason": reason,
            "priority": priority,
            "summary": summary,
            "status": "queued",
            "estimated_wait": estimated_wait,
            "queue_position": 1,
            "created_at": Utc::now().to_rfc3339(),
            "message": format!(
                "Your request has been escalated to a human agent. Escalation ID: {}. Estimated wait time: {}. Please hold.",
                escalation_id, estimated_wait
            ),
            "instructions": "A human agent will join this conversation shortly. Please stay on the line."
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        10
    }
}

impl Default for EscalateToHumanTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Send SMS tool
pub struct SendSmsTool {
    sms_service: Option<Arc<dyn voice_agent_persistence::SmsService>>,
}

impl SendSmsTool {
    pub fn new() -> Self {
        Self { sms_service: None }
    }

    pub fn with_sms_service(service: Arc<dyn voice_agent_persistence::SmsService>) -> Self {
        Self {
            sms_service: Some(service),
        }
    }
}

#[async_trait]
impl Tool for SendSmsTool {
    fn name(&self) -> &str {
        "send_sms"
    }

    fn description(&self) -> &str {
        "Send an SMS message to the customer for appointment confirmations, follow-ups, or information sharing"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "phone_number",
                    PropertySchema::string("10-digit mobile number"),
                    true,
                )
                .property(
                    "message_type",
                    PropertySchema::enum_type(
                        "Type of SMS message",
                        vec![
                            "appointment_confirmation".into(),
                            "appointment_reminder".into(),
                            "follow_up".into(),
                            "welcome".into(),
                            "promotional".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "customer_name",
                    PropertySchema::string("Customer name for personalization"),
                    false,
                )
                .property(
                    "custom_message",
                    PropertySchema::string("Custom message text (for follow_up type)"),
                    false,
                )
                .property(
                    "appointment_details",
                    PropertySchema::string("Appointment details (date, time, branch)"),
                    false,
                )
                .property(
                    "session_id",
                    PropertySchema::string("Session ID for tracking"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let phone = input
            .get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        if phone.len() != 10 || !phone.chars().all(|c| c.is_ascii_digit()) {
            return Err(ToolError::invalid_params("phone_number must be 10 digits"));
        }

        let msg_type_str = input
            .get("message_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("message_type is required"))?;

        let customer_name = input
            .get("customer_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Customer");

        let session_id = input.get("session_id").and_then(|v| v.as_str());

        let msg_type = match msg_type_str {
            "appointment_confirmation" => voice_agent_persistence::SmsType::AppointmentConfirmation,
            "appointment_reminder" => voice_agent_persistence::SmsType::AppointmentReminder,
            "follow_up" => voice_agent_persistence::SmsType::FollowUp,
            "welcome" => voice_agent_persistence::SmsType::Welcome,
            "promotional" => voice_agent_persistence::SmsType::Promotional,
            _ => voice_agent_persistence::SmsType::FollowUp,
        };

        let message_text = match msg_type {
            voice_agent_persistence::SmsType::AppointmentConfirmation => {
                let details = input
                    .get("appointment_details")
                    .and_then(|v| v.as_str())
                    .unwrap_or("scheduled date and time");
                format!(
                    "Dear {}, your Kotak Gold Loan appointment is confirmed for {}. Please bring your gold and KYC documents. For queries, call 1800-xxx-xxxx. - Kotak Mahindra Bank",
                    customer_name, details
                )
            }
            voice_agent_persistence::SmsType::AppointmentReminder => {
                let details = input
                    .get("appointment_details")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tomorrow");
                format!(
                    "Reminder: Dear {}, your Kotak Gold Loan appointment is {}. Please bring your gold and KYC documents. - Kotak Mahindra Bank",
                    customer_name, details
                )
            }
            voice_agent_persistence::SmsType::FollowUp => input
                .get("custom_message")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    format!(
                        "Dear {}, thank you for your interest in Kotak Gold Loan. Get up to 75% of gold value at competitive rates. Call 1800-xxx-xxxx or visit your nearest branch. - Kotak Mahindra Bank",
                        customer_name
                    )
                }),
            voice_agent_persistence::SmsType::Welcome => {
                format!(
                    "Welcome to Kotak Mahindra Bank, {}! We're excited to help you with your gold loan needs. For any queries, call 1800-xxx-xxxx. - Kotak Mahindra Bank",
                    customer_name
                )
            }
            voice_agent_persistence::SmsType::Promotional => {
                format!(
                    "Special Offer for {}: Get gold loan at just 10.49%* p.a. with instant disbursement! Visit your nearest Kotak branch or call 1800-xxx-xxxx. T&C apply. - Kotak Mahindra Bank",
                    customer_name
                )
            }
            _ => format!(
                "Dear {}, thank you for contacting Kotak Mahindra Bank. - Kotak Mahindra Bank",
                customer_name
            ),
        };

        let (message_id, status, simulated) = if let Some(ref service) = self.sms_service {
            match service
                .send_sms(phone, &message_text, msg_type, session_id)
                .await
            {
                Ok(result) => (
                    result.message_id.to_string(),
                    result.status.as_str().to_string(),
                    result.simulated,
                ),
                Err(e) => {
                    tracing::warn!("SMS service failed: {}", e);
                    let id = format!(
                        "SMS{}",
                        uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
                    );
                    (id, "failed".to_string(), false)
                }
            }
        } else {
            let id = format!(
                "SMS{}",
                uuid::Uuid::new_v4().to_string()[..8].to_uppercase()
            );
            (id, "simulated_not_sent".to_string(), true)
        };

        let success = status != "failed";

        let result = json!({
            "success": success,
            "message_id": message_id,
            "phone_number": phone,
            "message_type": msg_type_str,
            "message_text": message_text,
            "status": status,
            "simulated": simulated,
            "sent_at": if success { Some(Utc::now().to_rfc3339()) } else { None },
            "message": if success {
                format!("SMS {} to {}.", if simulated { "simulated" } else { "sent" }, phone)
            } else {
                "Failed to send SMS. Please try again.".to_string()
            }
        });

        Ok(ToolOutput::json(result))
    }

    fn timeout_secs(&self) -> u64 {
        30
    }
}

impl Default for SendSmsTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Document checklist tool
pub struct DocumentChecklistTool;

impl DocumentChecklistTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for DocumentChecklistTool {
    fn name(&self) -> &str {
        "get_document_checklist"
    }

    fn description(&self) -> &str {
        "Get the list of documents required for gold loan application based on loan type and customer category"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property(
                    "loan_type",
                    PropertySchema::enum_type(
                        "Type of gold loan",
                        vec![
                            "new_loan".into(),
                            "top_up".into(),
                            "balance_transfer".into(),
                            "renewal".into(),
                        ],
                    ),
                    true,
                )
                .property(
                    "customer_type",
                    PropertySchema::enum_type(
                        "Customer category",
                        vec![
                            "individual".into(),
                            "self_employed".into(),
                            "business".into(),
                            "nri".into(),
                        ],
                    ),
                    false,
                )
                .property(
                    "existing_customer",
                    PropertySchema::boolean("Is an existing Kotak customer"),
                    false,
                ),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let loan_type = input
            .get("loan_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("loan_type is required"))?;

        let customer_type = input
            .get("customer_type")
            .and_then(|v| v.as_str())
            .unwrap_or("individual");

        let existing_customer = input
            .get("existing_customer")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let mut mandatory_docs = vec![
            json!({
                "document": "Valid Photo ID",
                "accepted": ["Aadhaar Card", "PAN Card", "Passport", "Voter ID", "Driving License"],
                "copies": 1,
                "notes": "Original required for verification"
            }),
            json!({
                "document": "Address Proof",
                "accepted": ["Aadhaar Card", "Utility Bill (last 3 months)", "Bank Statement", "Rent Agreement"],
                "copies": 1,
                "notes": "Should match current residence"
            }),
            json!({
                "document": "Passport Size Photographs",
                "copies": 2,
                "notes": "Recent photographs (within 6 months)"
            }),
        ];

        mandatory_docs.push(json!({
            "document": "PAN Card",
            "copies": 1,
            "notes": "Mandatory for loans above ₹50,000"
        }));

        let gold_docs = vec![
            json!({
                "document": "Gold Items",
                "notes": "Bring gold jewelry/items for valuation. Remove any non-gold attachments (stones, pearls)"
            }),
            json!({
                "document": "Gold Purchase Invoice (if available)",
                "notes": "Helps with valuation and authenticity verification"
            }),
        ];

        let additional_docs: Vec<Value> = match loan_type {
            "balance_transfer" => vec![
                json!({
                    "document": "Existing Loan Statement",
                    "notes": "From current lender showing outstanding amount"
                }),
                json!({
                    "document": "Gold Loan Account Details",
                    "notes": "Loan account number and lender details"
                }),
                json!({
                    "document": "NOC from Current Lender",
                    "notes": "May be obtained after approval"
                }),
            ],
            "top_up" => vec![json!({
                "document": "Existing Kotak Gold Loan Details",
                "notes": "Loan account number for top-up"
            })],
            "renewal" => vec![json!({
                "document": "Previous Loan Details",
                "notes": "Loan account number for renewal"
            })],
            _ => vec![],
        };

        let customer_specific: Vec<Value> = match customer_type {
            "self_employed" | "business" => vec![json!({
                "document": "Business Proof",
                "accepted": ["GST Registration", "Shop & Establishment Certificate", "Trade License"],
                "notes": "Any one document for business verification"
            })],
            "nri" => vec![
                json!({
                    "document": "Passport with Valid Visa",
                    "notes": "Required for NRI customers"
                }),
                json!({
                    "document": "NRE/NRO Bank Account Statement",
                    "notes": "Last 6 months statement"
                }),
            ],
            _ => vec![],
        };

        let existing_customer_note = if existing_customer {
            "As an existing Kotak customer, some documents may already be on file. Please bring originals for verification."
        } else {
            "Please bring original documents along with photocopies."
        };

        let result = json!({
            "loan_type": loan_type,
            "customer_type": customer_type,
            "existing_customer": existing_customer,
            "mandatory_documents": mandatory_docs,
            "gold_related": gold_docs,
            "additional_documents": additional_docs,
            "customer_specific_documents": customer_specific,
            "total_documents": mandatory_docs.len() + gold_docs.len() + additional_docs.len() + customer_specific.len(),
            "important_notes": [
                existing_customer_note,
                "Original documents are required for verification at the branch.",
                "Gold items should be free of non-gold attachments for accurate valuation.",
                "Processing time: Same day disbursement subject to document verification."
            ],
            "message": format!(
                "For a {} gold loan, you'll need {} documents. Key documents: Valid ID, Address Proof, PAN Card, and your gold items.",
                loan_type.replace("_", " "),
                mandatory_docs.len() + gold_docs.len() + additional_docs.len() + customer_specific.len()
            )
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for DocumentChecklistTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Competitor comparison tool
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

// P15 FIX: Removed Default impl - ToolsDomainView is required

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
