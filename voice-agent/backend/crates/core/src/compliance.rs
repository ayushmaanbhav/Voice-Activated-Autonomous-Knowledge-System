//! Compliance checking types
//!
//! For ensuring agent responses follow regulatory requirements
//! and bank policies.

use serde::{Deserialize, Serialize};

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Whether text is compliant
    pub is_compliant: bool,
    /// List of violations found
    pub violations: Vec<ComplianceViolation>,
    /// Required additions (disclaimers, etc.)
    pub required_additions: Vec<RequiredAddition>,
    /// Suggested rewrites
    pub suggested_rewrites: Vec<SuggestedRewrite>,
}

impl ComplianceResult {
    /// Create a compliant result with no violations
    pub fn compliant() -> Self {
        Self {
            is_compliant: true,
            violations: Vec::new(),
            required_additions: Vec::new(),
            suggested_rewrites: Vec::new(),
        }
    }

    /// Create a non-compliant result with violations
    pub fn non_compliant(violations: Vec<ComplianceViolation>) -> Self {
        Self {
            is_compliant: false,
            violations,
            required_additions: Vec::new(),
            suggested_rewrites: Vec::new(),
        }
    }

    /// Add a required addition
    pub fn with_required_addition(mut self, addition: RequiredAddition) -> Self {
        self.required_additions.push(addition);
        self
    }

    /// Check if there are any critical violations
    pub fn has_critical_violations(&self) -> bool {
        self.violations.iter().any(|v| v.severity == Severity::Critical)
    }

    /// Get all violations of a specific severity
    pub fn violations_by_severity(&self, severity: Severity) -> Vec<&ComplianceViolation> {
        self.violations.iter().filter(|v| v.severity == severity).collect()
    }
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    /// Rule identifier (e.g., "RBI-GL-001")
    pub rule_id: String,
    /// Human-readable description
    pub description: String,
    /// Category of violation
    pub category: ViolationCategory,
    /// Severity level
    pub severity: Severity,
    /// Position in text (start, end) - byte offsets
    pub text_span: Option<(usize, usize)>,
    /// The violating text
    pub violating_text: Option<String>,
}

impl ComplianceViolation {
    /// Create a new violation
    pub fn new(
        rule_id: impl Into<String>,
        description: impl Into<String>,
        category: ViolationCategory,
        severity: Severity,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            description: description.into(),
            category,
            severity,
            text_span: None,
            violating_text: None,
        }
    }

    /// Set the text span
    pub fn with_span(mut self, start: usize, end: usize, text: impl Into<String>) -> Self {
        self.text_span = Some((start, end));
        self.violating_text = Some(text.into());
        self
    }
}

/// Violation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationCategory {
    /// RBI regulatory violation
    Regulatory,
    /// Bank policy violation
    Policy,
    /// Misleading or false claims
    MisleadingClaim,
    /// Missing required disclosure
    MissingDisclosure,
    /// Inappropriate language/tone
    InappropriateLanguage,
    /// Competitor disparagement
    CompetitorDisparagement,
    /// Promise that can't be guaranteed
    UnauthorizedPromise,
}

/// Severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Warning - can proceed with caution
    Warning,
    /// Error - should fix before proceeding
    Error,
    /// Critical - must not proceed
    Critical,
}

/// Required addition (disclaimer, disclosure, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredAddition {
    /// Type of addition
    pub addition_type: AdditionType,
    /// The text to add
    pub text: String,
    /// Where to add (beginning, end, inline)
    pub position: AdditionPosition,
    /// Condition for when this is required
    pub condition: Option<String>,
}

impl RequiredAddition {
    /// Create a disclaimer addition
    pub fn disclaimer(text: impl Into<String>) -> Self {
        Self {
            addition_type: AdditionType::Disclaimer,
            text: text.into(),
            position: AdditionPosition::End,
            condition: None,
        }
    }

    /// Create a disclosure addition
    pub fn disclosure(text: impl Into<String>) -> Self {
        Self {
            addition_type: AdditionType::Disclosure,
            text: text.into(),
            position: AdditionPosition::End,
            condition: None,
        }
    }
}

/// Type of required addition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdditionType {
    /// Legal disclaimer
    Disclaimer,
    /// Required disclosure (rates, fees, etc.)
    Disclosure,
    /// Terms and conditions reference
    TermsReference,
    /// Risk warning
    RiskWarning,
}

/// Position for required addition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdditionPosition {
    /// At the beginning
    Beginning,
    /// At the end
    End,
    /// After specific content
    After,
}

/// Suggested rewrite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedRewrite {
    /// Original text
    pub original: String,
    /// Suggested replacement
    pub replacement: String,
    /// Reason for change
    pub reason: String,
    /// Position in original text (start, end)
    pub span: Option<(usize, usize)>,
}

impl SuggestedRewrite {
    /// Create a new suggested rewrite
    pub fn new(
        original: impl Into<String>,
        replacement: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            original: original.into(),
            replacement: replacement.into(),
            reason: reason.into(),
            span: None,
        }
    }

    /// Set the span
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.span = Some((start, end));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliant_result() {
        let result = ComplianceResult::compliant();
        assert!(result.is_compliant);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_non_compliant_result() {
        let violation = ComplianceViolation::new(
            "RBI-GL-001",
            "Missing interest rate disclosure",
            ViolationCategory::MissingDisclosure,
            Severity::Error,
        );
        let result = ComplianceResult::non_compliant(vec![violation]);
        assert!(!result.is_compliant);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_critical_violations() {
        let critical = ComplianceViolation::new(
            "CRIT-001",
            "Critical violation",
            ViolationCategory::Regulatory,
            Severity::Critical,
        );
        let warning = ComplianceViolation::new(
            "WARN-001",
            "Warning",
            ViolationCategory::Policy,
            Severity::Warning,
        );
        let result = ComplianceResult::non_compliant(vec![critical, warning]);
        assert!(result.has_critical_violations());
    }

    #[test]
    fn test_suggested_rewrite() {
        let rewrite = SuggestedRewrite::new(
            "guaranteed lowest rate",
            "competitive rate",
            "Cannot guarantee rates as they are subject to change",
        );
        assert_eq!(rewrite.original, "guaranteed lowest rate");
    }
}
