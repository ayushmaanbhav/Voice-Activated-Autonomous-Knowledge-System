//! Financial Calculation Utilities
//!
//! Domain-agnostic financial calculations for loan products.
//! This is the single source of truth for EMI and related calculations.

/// Calculate EMI using the standard amortization formula.
///
/// EMI = P × r × (1 + r)^n / [(1 + r)^n - 1]
///
/// Where:
/// - P = Principal loan amount
/// - r = Monthly interest rate (annual_rate / 12 / 100)
/// - n = Number of months (tenure)
///
/// # Arguments
/// * `principal` - Principal loan amount (must be positive)
/// * `annual_rate_percent` - Annual interest rate as percentage (e.g., 12.0 for 12%)
/// * `tenure_months` - Loan tenure in months (must be positive)
///
/// # Returns
/// Monthly EMI amount, or 0.0 if inputs are invalid
///
/// # Precision
/// Uses `powi(i32)` for integer month values to maximize floating-point precision.
pub fn calculate_emi(principal: f64, annual_rate_percent: f64, tenure_months: i64) -> f64 {
    // Input validation
    if tenure_months <= 0 || principal <= 0.0 {
        return 0.0;
    }

    let monthly_rate = annual_rate_percent / 100.0 / 12.0;

    // Handle edge case of 0% or negative interest
    if monthly_rate <= 0.0 {
        return principal / tenure_months as f64;
    }

    // Use powi for better precision with integer exponents
    let n = tenure_months as i32;
    let factor = (1.0 + monthly_rate).powi(n);

    // EMI formula: P * r * (1+r)^n / [(1+r)^n - 1]
    principal * monthly_rate * factor / (factor - 1.0)
}

/// Calculate total interest paid over the loan tenure.
///
/// Total Interest = (EMI × n) - Principal
///
/// # Arguments
/// * `principal` - Principal loan amount
/// * `annual_rate_percent` - Annual interest rate as percentage
/// * `tenure_months` - Loan tenure in months
///
/// # Returns
/// Total interest amount paid over the loan tenure
pub fn calculate_total_interest(
    principal: f64,
    annual_rate_percent: f64,
    tenure_months: i64,
) -> f64 {
    let emi = calculate_emi(principal, annual_rate_percent, tenure_months);
    (emi * tenure_months as f64) - principal
}

/// Calculate monthly interest payment (simple interest model).
///
/// Some loan products (like gold loans) use simple interest where only
/// interest is paid monthly and principal is repaid at the end.
///
/// Monthly Interest = Principal × (Annual Rate / 100 / 12)
///
/// # Arguments
/// * `principal` - Principal loan amount
/// * `annual_rate_percent` - Annual interest rate as percentage
///
/// # Returns
/// Monthly interest payment amount
pub fn calculate_simple_monthly_interest(principal: f64, annual_rate_percent: f64) -> f64 {
    if principal <= 0.0 {
        return 0.0;
    }
    principal * annual_rate_percent / 100.0 / 12.0
}

/// Calculate total cost of loan (principal + total interest).
///
/// # Arguments
/// * `principal` - Principal loan amount
/// * `annual_rate_percent` - Annual interest rate as percentage
/// * `tenure_months` - Loan tenure in months
///
/// # Returns
/// Total amount to be repaid (principal + interest)
pub fn calculate_total_repayment(
    principal: f64,
    annual_rate_percent: f64,
    tenure_months: i64,
) -> f64 {
    let emi = calculate_emi(principal, annual_rate_percent, tenure_months);
    emi * tenure_months as f64
}

/// Calculate interest savings when comparing two rates.
///
/// # Arguments
/// * `principal` - Principal loan amount
/// * `rate1_percent` - First interest rate (our rate)
/// * `rate2_percent` - Second interest rate (competitor rate)
/// * `tenure_months` - Loan tenure in months
///
/// # Returns
/// Savings amount (positive if rate1 < rate2)
pub fn calculate_interest_savings(
    principal: f64,
    rate1_percent: f64,
    rate2_percent: f64,
    tenure_months: i64,
) -> f64 {
    let interest1 = calculate_total_interest(principal, rate1_percent, tenure_months);
    let interest2 = calculate_total_interest(principal, rate2_percent, tenure_months);
    interest2 - interest1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_emi() {
        // 1 lakh at 12% for 12 months
        let emi = calculate_emi(100_000.0, 12.0, 12);
        // Expected EMI around 8884.87
        assert!((emi - 8884.87).abs() < 1.0);
    }

    #[test]
    fn test_calculate_emi_zero_principal() {
        assert_eq!(calculate_emi(0.0, 12.0, 12), 0.0);
    }

    #[test]
    fn test_calculate_emi_zero_tenure() {
        assert_eq!(calculate_emi(100_000.0, 12.0, 0), 0.0);
    }

    #[test]
    fn test_calculate_emi_zero_rate() {
        // 1 lakh at 0% for 12 months = 8333.33 per month
        let emi = calculate_emi(100_000.0, 0.0, 12);
        assert!((emi - 8333.33).abs() < 1.0);
    }

    #[test]
    fn test_calculate_total_interest() {
        // 1 lakh at 12% for 12 months
        let interest = calculate_total_interest(100_000.0, 12.0, 12);
        // EMI ~8884.87 * 12 = 106618.44, interest = 6618.44
        assert!((interest - 6618.44).abs() < 1.0);
    }

    #[test]
    fn test_calculate_simple_monthly_interest() {
        // 1 lakh at 12% = 1000 per month simple interest
        let monthly = calculate_simple_monthly_interest(100_000.0, 12.0);
        assert!((monthly - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_calculate_interest_savings() {
        // Savings when switching from 14% to 10%
        let savings = calculate_interest_savings(100_000.0, 10.0, 14.0, 12);
        assert!(savings > 0.0); // Should save money
    }
}
