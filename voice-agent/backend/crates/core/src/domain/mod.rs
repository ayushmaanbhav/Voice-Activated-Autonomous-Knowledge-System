//! Domain Abstraction Module
//!
//! Provides generic traits and types for domain-agnostic agent behavior.
//! Specific domains (gold_loan, personal_loan, etc.) implement these traits
//! via configuration, not code.

mod traits;

pub use traits::*;
