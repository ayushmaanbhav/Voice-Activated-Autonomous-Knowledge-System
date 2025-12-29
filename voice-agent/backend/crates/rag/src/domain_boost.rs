//! Domain-specific term boosting for retrieval
//!
//! Provides boosting for domain-specific terms to improve retrieval relevance.
//! Optimized for the gold loan domain with Kotak-specific terminology.

use std::collections::HashMap;
use parking_lot::RwLock;

/// Domain booster configuration
#[derive(Debug, Clone)]
pub struct DomainBoostConfig {
    /// Base boost multiplier for domain terms
    pub base_boost: f32,
    /// Additional boost for exact matches
    pub exact_match_boost: f32,
    /// Boost for Kotak-specific terms
    pub brand_boost: f32,
    /// Enable category-based boosting
    pub category_boost_enabled: bool,
}

impl Default for DomainBoostConfig {
    fn default() -> Self {
        Self {
            base_boost: 1.5,
            exact_match_boost: 2.0,
            brand_boost: 1.3,
            category_boost_enabled: true,
        }
    }
}

/// Domain term with metadata
#[derive(Debug, Clone)]
pub struct DomainTerm {
    /// The term
    pub term: String,
    /// Category (e.g., "product", "rate", "process")
    pub category: TermCategory,
    /// Boost multiplier
    pub boost: f32,
    /// Related terms for context matching
    pub related: Vec<String>,
}

/// Term category for gold loan domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TermCategory {
    /// Product terms (gold loan, jewel loan)
    Product,
    /// Rate/pricing terms (interest, emi, fees)
    Rate,
    /// Process terms (apply, documents, disbursal)
    Process,
    /// Eligibility terms (criteria, requirements)
    Eligibility,
    /// Brand/competitor terms
    Brand,
    /// Gold-specific terms (purity, weight, karat)
    Gold,
    /// Customer terms (account, kyc)
    Customer,
    /// General/other
    General,
}

impl TermCategory {
    /// Get category boost multiplier
    pub fn boost_multiplier(&self) -> f32 {
        match self {
            Self::Product => 1.5,
            Self::Rate => 1.4,
            Self::Eligibility => 1.3,
            Self::Process => 1.2,
            Self::Brand => 1.5,
            Self::Gold => 1.4,
            Self::Customer => 1.1,
            Self::General => 1.0,
        }
    }
}

/// Boost result for a query
#[derive(Debug, Clone)]
pub struct BoostResult {
    /// Matched domain terms
    pub matched_terms: Vec<MatchedTerm>,
    /// Overall boost factor
    pub total_boost: f32,
    /// Detected categories
    pub categories: Vec<TermCategory>,
    /// Query intent (if determinable)
    pub intent: Option<QueryIntent>,
}

/// A matched domain term
#[derive(Debug, Clone)]
pub struct MatchedTerm {
    /// The term that matched
    pub term: String,
    /// Position in query
    pub position: usize,
    /// Boost applied
    pub boost: f32,
    /// Category
    pub category: TermCategory,
}

/// Detected query intent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryIntent {
    /// Asking about interest rates
    RateInquiry,
    /// Checking eligibility
    EligibilityCheck,
    /// Application process
    ApplicationProcess,
    /// Loan amount calculation
    AmountCalculation,
    /// Branch/location inquiry
    BranchLocator,
    /// Document requirements
    DocumentInquiry,
    /// Competitor comparison
    CompetitorComparison,
    /// Repayment/EMI query
    RepaymentInquiry,
    /// General inquiry
    General,
}

/// Domain booster for gold loan retrieval
pub struct DomainBooster {
    config: DomainBoostConfig,
    /// Domain terms dictionary
    terms: RwLock<HashMap<String, DomainTerm>>,
    /// Intent patterns
    intent_patterns: Vec<(Vec<&'static str>, QueryIntent)>,
}

impl DomainBooster {
    /// Create a new domain booster
    pub fn new(config: DomainBoostConfig) -> Self {
        let booster = Self {
            config,
            terms: RwLock::new(HashMap::new()),
            intent_patterns: Self::default_intent_patterns(),
        };
        booster.load_gold_loan_terms();
        booster
    }

    /// Create with default gold loan configuration
    pub fn gold_loan() -> Self {
        Self::new(DomainBoostConfig::default())
    }

    /// Load gold loan domain terms
    fn load_gold_loan_terms(&self) {
        let terms = vec![
            // Product terms
            DomainTerm {
                term: "gold loan".to_string(),
                category: TermCategory::Product,
                boost: 2.0,
                related: vec!["sona loan".to_string(), "jewel loan".to_string()],
            },
            DomainTerm {
                term: "jewel loan".to_string(),
                category: TermCategory::Product,
                boost: 1.8,
                related: vec!["gold loan".to_string(), "ornament loan".to_string()],
            },
            // Rate terms
            DomainTerm {
                term: "interest rate".to_string(),
                category: TermCategory::Rate,
                boost: 1.8,
                related: vec!["byaj dar".to_string(), "rate".to_string()],
            },
            DomainTerm {
                term: "emi".to_string(),
                category: TermCategory::Rate,
                boost: 1.6,
                related: vec!["kist".to_string(), "installment".to_string()],
            },
            DomainTerm {
                term: "processing fee".to_string(),
                category: TermCategory::Rate,
                boost: 1.5,
                related: vec!["charges".to_string(), "fees".to_string()],
            },
            // Eligibility terms
            DomainTerm {
                term: "eligibility".to_string(),
                category: TermCategory::Eligibility,
                boost: 1.7,
                related: vec!["patrta".to_string(), "qualify".to_string()],
            },
            DomainTerm {
                term: "criteria".to_string(),
                category: TermCategory::Eligibility,
                boost: 1.5,
                related: vec!["requirements".to_string(), "conditions".to_string()],
            },
            // Process terms
            DomainTerm {
                term: "apply".to_string(),
                category: TermCategory::Process,
                boost: 1.5,
                related: vec!["aavedan".to_string(), "application".to_string()],
            },
            DomainTerm {
                term: "disbursal".to_string(),
                category: TermCategory::Process,
                boost: 1.6,
                related: vec!["disbursement".to_string(), "sanction".to_string()],
            },
            DomainTerm {
                term: "prepayment".to_string(),
                category: TermCategory::Process,
                boost: 1.5,
                related: vec!["foreclosure".to_string(), "early payment".to_string()],
            },
            // Gold terms
            DomainTerm {
                term: "purity".to_string(),
                category: TermCategory::Gold,
                boost: 1.6,
                related: vec!["karat".to_string(), "carat".to_string(), "hallmark".to_string()],
            },
            DomainTerm {
                term: "weight".to_string(),
                category: TermCategory::Gold,
                boost: 1.4,
                related: vec!["gram".to_string(), "tola".to_string(), "vajan".to_string()],
            },
            DomainTerm {
                term: "valuation".to_string(),
                category: TermCategory::Gold,
                boost: 1.5,
                related: vec!["assessment".to_string(), "appraisal".to_string()],
            },
            // Brand terms
            DomainTerm {
                term: "kotak".to_string(),
                category: TermCategory::Brand,
                boost: 1.8,
                related: vec!["kotak mahindra".to_string(), "kmb".to_string()],
            },
            DomainTerm {
                term: "muthoot".to_string(),
                category: TermCategory::Brand,
                boost: 1.3,
                related: vec!["muthoot finance".to_string()],
            },
            DomainTerm {
                term: "manappuram".to_string(),
                category: TermCategory::Brand,
                boost: 1.3,
                related: vec!["manappuram finance".to_string()],
            },
            DomainTerm {
                term: "iifl".to_string(),
                category: TermCategory::Brand,
                boost: 1.3,
                related: vec!["iifl finance".to_string()],
            },
            // Customer terms
            DomainTerm {
                term: "kyc".to_string(),
                category: TermCategory::Customer,
                boost: 1.4,
                related: vec!["documents".to_string(), "identity".to_string()],
            },
            DomainTerm {
                term: "account".to_string(),
                category: TermCategory::Customer,
                boost: 1.2,
                related: vec!["khata".to_string(), "savings".to_string()],
            },
        ];

        let mut term_map = self.terms.write();
        for term in terms {
            // Add main term
            term_map.insert(term.term.to_lowercase(), term.clone());
            // Add related terms with lower boost
            for related in &term.related {
                if !term_map.contains_key(&related.to_lowercase()) {
                    term_map.insert(
                        related.to_lowercase(),
                        DomainTerm {
                            term: related.clone(),
                            category: term.category,
                            boost: term.boost * 0.8,
                            related: vec![term.term.clone()],
                        },
                    );
                }
            }
        }
    }

    /// Default intent patterns
    fn default_intent_patterns() -> Vec<(Vec<&'static str>, QueryIntent)> {
        vec![
            // Rate inquiry
            (vec!["interest", "rate"], QueryIntent::RateInquiry),
            (vec!["byaj", "dar"], QueryIntent::RateInquiry),
            (vec!["emi", "calculate"], QueryIntent::RateInquiry),
            (vec!["kitna", "interest"], QueryIntent::RateInquiry),
            // Eligibility
            (vec!["eligible", "loan"], QueryIntent::EligibilityCheck),
            (vec!["eligibility", "criteria"], QueryIntent::EligibilityCheck),
            (vec!["patr", "loan"], QueryIntent::EligibilityCheck),
            (vec!["can", "get", "loan"], QueryIntent::EligibilityCheck),
            // Application
            (vec!["apply", "loan"], QueryIntent::ApplicationProcess),
            (vec!["how", "apply"], QueryIntent::ApplicationProcess),
            (vec!["kaise", "apply"], QueryIntent::ApplicationProcess),
            (vec!["application", "process"], QueryIntent::ApplicationProcess),
            // Amount
            (vec!["loan", "amount"], QueryIntent::AmountCalculation),
            (vec!["kitna", "milega"], QueryIntent::AmountCalculation),
            (vec!["maximum", "loan"], QueryIntent::AmountCalculation),
            (vec!["how", "much", "loan"], QueryIntent::AmountCalculation),
            // Branch
            (vec!["branch", "near"], QueryIntent::BranchLocator),
            (vec!["kahan", "branch"], QueryIntent::BranchLocator),
            (vec!["nearest", "branch"], QueryIntent::BranchLocator),
            // Documents
            (vec!["document", "required"], QueryIntent::DocumentInquiry),
            (vec!["kyc", "document"], QueryIntent::DocumentInquiry),
            (vec!["papers", "needed"], QueryIntent::DocumentInquiry),
            // Competitor
            (vec!["muthoot", "kotak"], QueryIntent::CompetitorComparison),
            (vec!["compare", "loan"], QueryIntent::CompetitorComparison),
            (vec!["better", "than"], QueryIntent::CompetitorComparison),
            (vec!["switch", "from"], QueryIntent::CompetitorComparison),
            // Repayment
            (vec!["repay", "loan"], QueryIntent::RepaymentInquiry),
            (vec!["prepay", "loan"], QueryIntent::RepaymentInquiry),
            (vec!["foreclosure"], QueryIntent::RepaymentInquiry),
            (vec!["chukana", "loan"], QueryIntent::RepaymentInquiry),
        ]
    }

    /// Calculate boost for a query
    pub fn boost(&self, query: &str) -> BoostResult {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        let terms = self.terms.read();
        let mut matched_terms = Vec::new();
        let mut categories = Vec::new();

        // Match single words
        for (pos, word) in words.iter().enumerate() {
            if let Some(domain_term) = terms.get(*word) {
                matched_terms.push(MatchedTerm {
                    term: domain_term.term.clone(),
                    position: pos,
                    boost: domain_term.boost * self.config.base_boost,
                    category: domain_term.category,
                });
                if !categories.contains(&domain_term.category) {
                    categories.push(domain_term.category);
                }
            }
        }

        // Match multi-word terms
        for (term, domain_term) in terms.iter() {
            if term.contains(' ') && query_lower.contains(term) {
                // Avoid double counting if individual words already matched
                if !matched_terms.iter().any(|m| m.term == domain_term.term) {
                    matched_terms.push(MatchedTerm {
                        term: domain_term.term.clone(),
                        position: query_lower.find(term).unwrap_or(0),
                        boost: domain_term.boost * self.config.exact_match_boost,
                        category: domain_term.category,
                    });
                    if !categories.contains(&domain_term.category) {
                        categories.push(domain_term.category);
                    }
                }
            }
        }

        // Detect intent
        let intent = self.detect_intent(&query_lower);

        // Calculate total boost
        let total_boost = if matched_terms.is_empty() {
            1.0
        } else {
            let base: f32 = matched_terms.iter().map(|m| m.boost).sum::<f32>() / matched_terms.len() as f32;
            let category_bonus: f32 = if self.config.category_boost_enabled {
                categories.iter().map(|c| c.boost_multiplier()).sum::<f32>() / categories.len().max(1) as f32
            } else {
                1.0
            };
            base * category_bonus
        };

        BoostResult {
            matched_terms,
            total_boost,
            categories,
            intent,
        }
    }

    /// Detect query intent
    fn detect_intent(&self, query: &str) -> Option<QueryIntent> {
        for (patterns, intent) in &self.intent_patterns {
            if patterns.iter().all(|p| query.contains(p)) {
                return Some(*intent);
            }
        }
        None
    }

    /// Apply boost to search results
    pub fn apply_boost<T>(&self, results: &mut [(T, f32)], query: &str)
    where
        T: AsRef<str>,
    {
        let boost_result = self.boost(query);

        for (doc, score) in results.iter_mut() {
            let doc_text = doc.as_ref().to_lowercase();

            // Calculate document-specific boost
            let mut doc_boost = 1.0f32;

            for matched in &boost_result.matched_terms {
                if doc_text.contains(&matched.term.to_lowercase()) {
                    doc_boost *= matched.boost;
                }
            }

            // Apply brand boost for Kotak mentions
            if doc_text.contains("kotak") {
                doc_boost *= self.config.brand_boost;
            }

            *score *= doc_boost;
        }
    }

    /// Add a custom domain term
    pub fn add_term(&self, term: DomainTerm) {
        let mut terms = self.terms.write();
        terms.insert(term.term.to_lowercase(), term);
    }

    /// Get intent label for display
    pub fn intent_label(intent: QueryIntent) -> &'static str {
        match intent {
            QueryIntent::RateInquiry => "Interest Rate Inquiry",
            QueryIntent::EligibilityCheck => "Eligibility Check",
            QueryIntent::ApplicationProcess => "Application Process",
            QueryIntent::AmountCalculation => "Loan Amount Calculation",
            QueryIntent::BranchLocator => "Branch Locator",
            QueryIntent::DocumentInquiry => "Document Requirements",
            QueryIntent::CompetitorComparison => "Competitor Comparison",
            QueryIntent::RepaymentInquiry => "Repayment/EMI Query",
            QueryIntent::General => "General Inquiry",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_boost() {
        let booster = DomainBooster::gold_loan();
        let result = booster.boost("gold loan interest rate");

        assert!(!result.matched_terms.is_empty());
        assert!(result.total_boost > 1.0);
    }

    #[test]
    fn test_category_detection() {
        let booster = DomainBooster::gold_loan();

        let rate_result = booster.boost("interest rate");
        assert!(rate_result.categories.contains(&TermCategory::Rate));

        let eligibility_result = booster.boost("eligibility criteria");
        assert!(eligibility_result.categories.contains(&TermCategory::Eligibility));
    }

    #[test]
    fn test_intent_detection() {
        let booster = DomainBooster::gold_loan();

        let result = booster.boost("what is the interest rate");
        assert_eq!(result.intent, Some(QueryIntent::RateInquiry));

        let result = booster.boost("am I eligible for loan");
        assert_eq!(result.intent, Some(QueryIntent::EligibilityCheck));

        let result = booster.boost("how to apply for gold loan");
        assert_eq!(result.intent, Some(QueryIntent::ApplicationProcess));
    }

    #[test]
    fn test_competitor_terms() {
        let booster = DomainBooster::gold_loan();
        let result = booster.boost("muthoot vs kotak gold loan");

        assert!(result.categories.contains(&TermCategory::Brand));
        assert_eq!(result.intent, Some(QueryIntent::CompetitorComparison));
    }

    #[test]
    fn test_hindi_terms() {
        let booster = DomainBooster::gold_loan();
        let result = booster.boost("byaj dar kitna hai");

        assert!(result.intent == Some(QueryIntent::RateInquiry));
    }

    #[test]
    fn test_apply_boost() {
        let booster = DomainBooster::gold_loan();

        let mut results = vec![
            ("Kotak gold loan offers competitive rates".to_string(), 0.8f32),
            ("Weather forecast for today".to_string(), 0.9f32),
        ];

        booster.apply_boost(&mut results, "gold loan rate kotak");

        // Kotak gold loan doc should be boosted higher
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_multi_word_terms() {
        let booster = DomainBooster::gold_loan();
        let result = booster.boost("gold loan eligibility");

        // Should match "gold loan" as a phrase
        assert!(result.matched_terms.iter().any(|m| m.term.contains("gold loan") || m.term == "gold"));
    }
}
