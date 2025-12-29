//! Rule-based compliance checker

use async_trait::async_trait;
use regex::Regex;
use voice_agent_core::{
    ComplianceChecker, ComplianceResult, ComplianceViolation, Severity,
    ViolationCategory, RequiredAddition, Result,
};
use super::rules::ComplianceRules;

/// Rule-based compliance checker
pub struct RuleBasedComplianceChecker {
    rules: ComplianceRules,
    forbidden_patterns: Vec<CompiledRule>,
    claim_patterns: Vec<CompiledClaimRule>,
    strict_mode: bool,
}

struct CompiledRule {
    pattern: Regex,
    phrase: String,
}

struct CompiledClaimRule {
    pattern: Regex,
    disclaimer: String,
    description: String,
}

impl RuleBasedComplianceChecker {
    /// Create a new checker with rules
    pub fn new(rules: ComplianceRules, strict_mode: bool) -> Self {
        let forbidden_patterns = rules
            .forbidden_phrases
            .iter()
            .filter_map(|phrase| {
                Regex::new(&format!(r"(?i)\b{}\b", regex::escape(phrase)))
                    .ok()
                    .map(|pattern| CompiledRule {
                        pattern,
                        phrase: phrase.clone(),
                    })
            })
            .collect();

        let claim_patterns = rules
            .claims_requiring_disclaimer
            .iter()
            .filter_map(|rule| {
                Regex::new(&rule.pattern)
                    .ok()
                    .map(|pattern| CompiledClaimRule {
                        pattern,
                        disclaimer: rule.disclaimer.clone(),
                        description: rule.description.clone(),
                    })
            })
            .collect();

        Self {
            rules,
            forbidden_patterns,
            claim_patterns,
            strict_mode,
        }
    }

    /// Check for forbidden phrases
    fn check_forbidden(&self, text: &str) -> Vec<ComplianceViolation> {
        self.forbidden_patterns
            .iter()
            .filter_map(|rule| {
                rule.pattern.find(text).map(|m| {
                    ComplianceViolation::new(
                        format!("FORBIDDEN_{}", rule.phrase.to_uppercase().replace(' ', "_")),
                        format!("Forbidden phrase detected: '{}'", rule.phrase),
                        ViolationCategory::MisleadingClaim,
                        Severity::Critical,
                    ).with_span(m.start(), m.end(), m.as_str())
                })
            })
            .collect()
    }

    /// Check for claims requiring disclaimers
    fn check_claims(&self, text: &str) -> (Vec<ComplianceViolation>, Vec<RequiredAddition>) {
        let mut violations = Vec::new();
        let mut additions = Vec::new();

        for rule in &self.claim_patterns {
            if let Some(m) = rule.pattern.find(text) {
                // Check if disclaimer already present
                if !text.contains(&rule.disclaimer) && !rule.disclaimer.is_empty() {
                    violations.push(
                        ComplianceViolation::new(
                            "MISSING_DISCLAIMER".to_string(),
                            format!("Claim '{}' requires disclaimer", rule.description),
                            ViolationCategory::MissingDisclosure,
                            Severity::Warning,
                        ).with_span(m.start(), m.end(), m.as_str())
                    );
                    additions.push(RequiredAddition::disclaimer(&rule.disclaimer));
                }
            }
        }

        (violations, additions)
    }

    /// Check rate accuracy
    fn check_rates(&self, text: &str) -> Vec<ComplianceViolation> {
        let rate_pattern = Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap();
        let mut violations = Vec::new();

        for capture in rate_pattern.captures_iter(text) {
            if let Some(rate_match) = capture.get(1) {
                if let Ok(rate) = rate_match.as_str().parse::<f32>() {
                    if rate < self.rules.rate_rules.min_rate {
                        violations.push(ComplianceViolation::new(
                            "RATE_TOO_LOW".to_string(),
                            format!(
                                "Rate {}% is below minimum {}%",
                                rate, self.rules.rate_rules.min_rate
                            ),
                            ViolationCategory::Regulatory,
                            Severity::Error,
                        ));
                    } else if rate > self.rules.rate_rules.max_rate {
                        violations.push(ComplianceViolation::new(
                            "RATE_TOO_HIGH".to_string(),
                            format!(
                                "Rate {}% exceeds maximum {}%",
                                rate, self.rules.rate_rules.max_rate
                            ),
                            ViolationCategory::Regulatory,
                            Severity::Error,
                        ));
                    }
                }
            }
        }

        violations
    }

    /// Check for competitor disparagement
    fn check_competitor_mentions(&self, text: &str) -> Vec<ComplianceViolation> {
        if self.rules.competitor_rules.allow_disparagement {
            return Vec::new();
        }

        let mut violations = Vec::new();
        let text_lower = text.to_lowercase();

        let disparaging_words = ["bad", "worst", "fraud", "cheat", "scam", "terrible", "avoid"];

        for competitor in &self.rules.competitor_rules.competitors {
            if text_lower.contains(&competitor.to_lowercase()) {
                // Check if any disparaging words are nearby
                for word in &disparaging_words {
                    if text_lower.contains(word) {
                        violations.push(ComplianceViolation::new(
                            "COMPETITOR_DISPARAGEMENT".to_string(),
                            format!(
                                "Potential disparagement of competitor '{}' detected",
                                competitor
                            ),
                            ViolationCategory::CompetitorDisparagement,
                            Severity::Error,
                        ));
                        break;
                    }
                }
            }
        }

        violations
    }
}

#[async_trait]
impl ComplianceChecker for RuleBasedComplianceChecker {
    async fn check(&self, text: &str) -> Result<ComplianceResult> {
        let mut all_violations = Vec::new();
        let mut required_additions = Vec::new();

        // Check forbidden phrases
        all_violations.extend(self.check_forbidden(text));

        // Check claims
        let (claim_violations, claim_additions) = self.check_claims(text);
        all_violations.extend(claim_violations);
        required_additions.extend(claim_additions);

        // Check rates
        all_violations.extend(self.check_rates(text));

        // Check competitor mentions
        all_violations.extend(self.check_competitor_mentions(text));

        // Determine compliance
        let is_compliant = if self.strict_mode {
            all_violations.is_empty()
        } else {
            !all_violations.iter().any(|v| v.severity == Severity::Critical)
        };

        Ok(ComplianceResult {
            is_compliant,
            violations: all_violations,
            required_additions,
            suggested_rewrites: Vec::new(),
        })
    }

    async fn make_compliant(&self, text: &str) -> Result<String> {
        let result = self.check(text).await?;

        if result.is_compliant && result.required_additions.is_empty() {
            return Ok(text.to_string());
        }

        let mut compliant_text = text.to_string();

        // Remove critical violations (replace with safe text)
        for violation in result.violations.iter().filter(|v| v.severity == Severity::Critical) {
            if let Some((start, end)) = violation.text_span {
                compliant_text.replace_range(start..end, "[removed]");
            }
        }

        // Add required disclaimers
        for addition in &result.required_additions {
            if !compliant_text.contains(&addition.text) {
                compliant_text.push(' ');
                compliant_text.push_str(&addition.text);
            }
        }

        Ok(compliant_text)
    }

    fn rules_version(&self) -> &str {
        &self.rules.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rules::default_rules;

    fn create_checker() -> RuleBasedComplianceChecker {
        RuleBasedComplianceChecker::new(default_rules(), false)
    }

    #[tokio::test]
    async fn test_forbidden_phrase() {
        let checker = create_checker();
        let text = "We offer guaranteed approval for your gold loan!";

        let result = checker.check(text).await.unwrap();
        assert!(!result.is_compliant);
        assert!(result.violations.iter().any(|v| v.severity == Severity::Critical));
    }

    #[tokio::test]
    async fn test_clean_text() {
        let checker = create_checker();
        let text = "We offer competitive rates for gold loans.";

        let result = checker.check(text).await.unwrap();
        // May have warnings for rate mentions but should be compliant
        assert!(result.is_compliant || !result.violations.iter().any(|v| v.severity == Severity::Critical));
    }

    #[tokio::test]
    async fn test_rate_validation() {
        let checker = create_checker();

        // Valid rate
        let result = checker.check("Our rate is 10.5%").await.unwrap();
        assert!(result.violations.iter().all(|v| !matches!(v.category, ViolationCategory::Regulatory)));

        // Too low
        let result = checker.check("Our rate is 2%").await.unwrap();
        assert!(result.violations.iter().any(|v| v.rule_id == "RATE_TOO_LOW"));

        // Too high
        let result = checker.check("Our rate is 30%").await.unwrap();
        assert!(result.violations.iter().any(|v| v.rule_id == "RATE_TOO_HIGH"));
    }

    #[tokio::test]
    async fn test_competitor_disparagement() {
        let checker = create_checker();
        let text = "Muthoot is a fraud company, use Kotak instead";

        let result = checker.check(text).await.unwrap();
        assert!(result.violations.iter().any(|v|
            matches!(v.category, ViolationCategory::CompetitorDisparagement)
        ));
    }

    #[tokio::test]
    async fn test_make_compliant() {
        let checker = create_checker();
        let text = "We offer guaranteed approval!";

        let compliant = checker.make_compliant(text).await.unwrap();
        assert!(compliant.contains("[removed]"));
    }
}
