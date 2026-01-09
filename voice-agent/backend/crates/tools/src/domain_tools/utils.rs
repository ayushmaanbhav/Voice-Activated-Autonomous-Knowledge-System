//! Financial Utility Functions for Domain Tools
//!
//! Re-exports financial calculations from the core crate.
//! This ensures a single source of truth for all financial calculations.

// Re-export from core crate's financial module
pub use voice_agent_core::financial::{calculate_emi, calculate_total_interest};

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
}
