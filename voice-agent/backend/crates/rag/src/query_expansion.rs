//! Query Expansion for improved retrieval
//!
//! Expands queries with:
//! - Domain-specific synonyms (gold loan terminology)
//! - Hindi/Hinglish transliterations
//! - Related terms and concepts

use std::collections::HashMap;
use parking_lot::RwLock;

/// Query expansion configuration
#[derive(Debug, Clone)]
pub struct QueryExpansionConfig {
    /// Enable synonym expansion
    pub enable_synonyms: bool,
    /// Enable transliteration expansion (Hindi <-> Roman)
    pub enable_transliteration: bool,
    /// Maximum expansion terms per query term
    pub max_expansions_per_term: usize,
    /// Boost factor for original terms vs expansions
    pub original_term_boost: f32,
    /// Domain for term expansion
    pub domain: String,
}

impl Default for QueryExpansionConfig {
    fn default() -> Self {
        Self {
            enable_synonyms: true,
            enable_transliteration: true,
            max_expansions_per_term: 3,
            original_term_boost: 2.0,
            domain: "gold_loan".to_string(),
        }
    }
}

/// Expanded query result
#[derive(Debug, Clone)]
pub struct ExpandedQuery {
    /// Original query
    pub original: String,
    /// Expanded query terms with weights
    pub terms: Vec<WeightedTerm>,
    /// Whether any expansion occurred
    pub was_expanded: bool,
    /// Expansion statistics
    pub stats: ExpansionStats,
}

/// A weighted query term
#[derive(Debug, Clone)]
pub struct WeightedTerm {
    /// The term
    pub term: String,
    /// Weight/boost factor
    pub weight: f32,
    /// Source of this term
    pub source: TermSource,
}

/// Source of an expanded term
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermSource {
    /// Original query term
    Original,
    /// Synonym expansion
    Synonym,
    /// Transliteration
    Transliteration,
    /// Domain-specific expansion
    Domain,
}

/// Expansion statistics
#[derive(Debug, Clone, Default)]
pub struct ExpansionStats {
    /// Original term count
    pub original_terms: usize,
    /// Synonym expansions added
    pub synonym_expansions: usize,
    /// Transliteration expansions added
    pub transliteration_expansions: usize,
    /// Domain expansions added
    pub domain_expansions: usize,
}

/// Query expander for RAG
pub struct QueryExpander {
    config: QueryExpansionConfig,
    /// Domain synonym dictionary
    synonyms: RwLock<HashMap<String, Vec<String>>>,
    /// Transliteration mappings (Hindi <-> Roman)
    transliterations: RwLock<HashMap<String, Vec<String>>>,
    /// Domain-specific term expansions
    domain_terms: RwLock<HashMap<String, Vec<String>>>,
}

impl QueryExpander {
    /// Create a new query expander
    pub fn new(config: QueryExpansionConfig) -> Self {
        let expander = Self {
            config,
            synonyms: RwLock::new(HashMap::new()),
            transliterations: RwLock::new(HashMap::new()),
            domain_terms: RwLock::new(HashMap::new()),
        };
        expander.load_default_dictionaries();
        expander
    }

    /// Create with default gold loan configuration
    pub fn gold_loan() -> Self {
        Self::new(QueryExpansionConfig {
            domain: "gold_loan".to_string(),
            ..Default::default()
        })
    }

    /// Load default dictionaries for gold loan domain
    fn load_default_dictionaries(&self) {
        // Gold loan synonyms
        let synonyms = vec![
            // Interest/rate terms
            ("interest", vec!["rate", "byaj", "sud"]),
            ("rate", vec!["interest", "percentage", "dar"]),
            ("byaj", vec!["interest", "sud", "rate"]),
            // Loan terms
            ("loan", vec!["karza", "rin", "udhar", "credit"]),
            ("karza", vec!["loan", "rin", "udhar"]),
            ("gold", vec!["sona", "swarna", "jewelry", "jewellery"]),
            ("sona", vec!["gold", "swarna"]),
            // Eligibility
            ("eligibility", vec!["patrta", "qualification", "criteria"]),
            ("eligible", vec!["patr", "qualified", "qualify"]),
            // Amount terms
            ("amount", vec!["rashi", "paisa", "money", "sum"]),
            ("lakh", vec!["lac", "100000"]),
            ("crore", vec!["cr", "10000000"]),
            // Process terms
            ("apply", vec!["aavedan", "application", "request"]),
            ("document", vec!["dastavez", "papers", "kagaz"]),
            ("disburse", vec!["vitrit", "release", "sanction"]),
            // Gold specific
            ("purity", vec!["shudhta", "karat", "carat", "fineness"]),
            ("weight", vec!["vajan", "gram", "tola"]),
            ("hallmark", vec!["certified", "bis", "standard"]),
            // Customer terms
            ("customer", vec!["grahak", "client", "applicant"]),
            ("account", vec!["khata", "savings", "current"]),
            // Competitor terms
            ("muthoot", vec!["muthut", "muthoot finance"]),
            ("manappuram", vec!["manapuram", "manappuram finance"]),
            // EMI/repayment
            ("emi", vec!["installment", "kist", "monthly payment"]),
            ("repay", vec!["chukana", "payment", "return"]),
            ("prepay", vec!["prepayment", "early payment", "foreclosure"]),
        ];

        let mut syn_map = self.synonyms.write();
        for (term, syns) in synonyms {
            syn_map.insert(term.to_string(), syns.iter().map(|s| s.to_string()).collect());
        }

        // Hindi-Roman transliterations
        let transliterations = vec![
            // Common Hindi terms with Roman equivalents
            ("सोना", vec!["sona", "gold"]),
            ("ब्याज", vec!["byaj", "interest"]),
            ("दर", vec!["dar", "rate"]),
            ("कर्ज़ा", vec!["karza", "loan"]),
            ("पात्रता", vec!["patrta", "eligibility"]),
            ("राशि", vec!["rashi", "amount"]),
            ("आवेदन", vec!["aavedan", "apply"]),
            ("दस्तावेज़", vec!["dastavez", "document"]),
            ("ग्राहक", vec!["grahak", "customer"]),
            ("खाता", vec!["khata", "account"]),
            ("किस्त", vec!["kist", "emi"]),
            ("शुद्धता", vec!["shudhta", "purity"]),
            ("वजन", vec!["vajan", "weight"]),
            // Roman Hindi to Devanagari
            ("sona", vec!["सोना", "gold"]),
            ("byaj", vec!["ब्याज", "interest"]),
            ("karza", vec!["कर्ज़ा", "loan"]),
            ("patrta", vec!["पात्रता", "eligibility"]),
        ];

        let mut trans_map = self.transliterations.write();
        for (term, trans) in transliterations {
            trans_map.insert(term.to_string(), trans.iter().map(|s| s.to_string()).collect());
        }

        // Domain-specific expansions
        let domain_terms = vec![
            // Gold loan specific
            ("gold loan", vec!["sona loan", "gold karza", "jewel loan"]),
            ("interest rate", vec!["byaj dar", "rate of interest", "loan rate"]),
            ("eligibility criteria", vec!["patrta", "who can apply", "requirements"]),
            ("loan amount", vec!["kitna milega", "how much", "maximum loan"]),
            ("processing fee", vec!["charges", "fees", "cost"]),
            ("repayment", vec!["chukana", "pay back", "return loan"]),
            // Questions patterns
            ("kya hai", vec!["what is", "क्या है"]),
            ("kitna hai", vec!["how much", "कितना है"]),
            ("kaise", vec!["how to", "कैसे"]),
            ("kahan", vec!["where", "कहाँ"]),
        ];

        let mut domain_map = self.domain_terms.write();
        for (term, expansions) in domain_terms {
            domain_map.insert(term.to_string(), expansions.iter().map(|s| s.to_string()).collect());
        }
    }

    /// Expand a query
    pub fn expand(&self, query: &str) -> ExpandedQuery {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut terms = Vec::new();
        let mut stats = ExpansionStats::default();

        // Add original terms with boost
        for word in &words {
            terms.push(WeightedTerm {
                term: word.to_string(),
                weight: self.config.original_term_boost,
                source: TermSource::Original,
            });
            stats.original_terms += 1;
        }

        // Synonym expansion
        if self.config.enable_synonyms {
            let synonyms = self.synonyms.read();
            for word in &words {
                if let Some(syns) = synonyms.get(*word) {
                    for syn in syns.iter().take(self.config.max_expansions_per_term) {
                        if !terms.iter().any(|t| t.term == *syn) {
                            terms.push(WeightedTerm {
                                term: syn.clone(),
                                weight: 1.0,
                                source: TermSource::Synonym,
                            });
                            stats.synonym_expansions += 1;
                        }
                    }
                }
            }
        }

        // Transliteration expansion
        if self.config.enable_transliteration {
            let trans = self.transliterations.read();
            for word in &words {
                if let Some(translits) = trans.get(*word) {
                    for t in translits.iter().take(self.config.max_expansions_per_term) {
                        if !terms.iter().any(|term| term.term == *t) {
                            terms.push(WeightedTerm {
                                term: t.clone(),
                                weight: 0.8,
                                source: TermSource::Transliteration,
                            });
                            stats.transliteration_expansions += 1;
                        }
                    }
                }
            }
        }

        // Domain-specific multi-word expansion
        let domain_terms = self.domain_terms.read();
        for (pattern, expansions) in domain_terms.iter() {
            if query_lower.contains(pattern) {
                for exp in expansions.iter().take(self.config.max_expansions_per_term) {
                    if !terms.iter().any(|t| t.term == *exp) {
                        terms.push(WeightedTerm {
                            term: exp.clone(),
                            weight: 0.9,
                            source: TermSource::Domain,
                        });
                        stats.domain_expansions += 1;
                    }
                }
            }
        }

        let was_expanded = stats.synonym_expansions > 0
            || stats.transliteration_expansions > 0
            || stats.domain_expansions > 0;

        ExpandedQuery {
            original: query.to_string(),
            terms,
            was_expanded,
            stats,
        }
    }

    /// Get expanded query as a weighted search string
    ///
    /// Format: "term1^2.0 term2^1.0 term3^0.8"
    pub fn expand_to_string(&self, query: &str) -> String {
        let expanded = self.expand(query);

        expanded
            .terms
            .iter()
            .map(|t| {
                if (t.weight - 1.0).abs() < 0.01 {
                    t.term.clone()
                } else {
                    format!("{}^{:.1}", t.term, t.weight)
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Add a custom synonym
    pub fn add_synonym(&self, term: &str, synonyms: &[&str]) {
        let mut syn_map = self.synonyms.write();
        syn_map.insert(
            term.to_lowercase(),
            synonyms.iter().map(|s| s.to_lowercase()).collect(),
        );
    }

    /// Add a custom transliteration
    pub fn add_transliteration(&self, term: &str, transliterations: &[&str]) {
        let mut trans_map = self.transliterations.write();
        trans_map.insert(
            term.to_string(),
            transliterations.iter().map(|s| s.to_string()).collect(),
        );
    }

    /// Add a domain term expansion
    pub fn add_domain_term(&self, term: &str, expansions: &[&str]) {
        let mut domain_map = self.domain_terms.write();
        domain_map.insert(
            term.to_lowercase(),
            expansions.iter().map(|s| s.to_string()).collect(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_expansion() {
        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand("gold loan interest rate");

        assert!(expanded.was_expanded);
        assert!(expanded.terms.len() > 4); // More than just original terms

        // Should contain original terms
        assert!(expanded.terms.iter().any(|t| t.term == "gold"));
        assert!(expanded.terms.iter().any(|t| t.term == "loan"));
        assert!(expanded.terms.iter().any(|t| t.term == "interest"));
        assert!(expanded.terms.iter().any(|t| t.term == "rate"));
    }

    #[test]
    fn test_synonym_expansion() {
        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand("gold eligibility");

        // Should expand "gold" to include "sona"
        assert!(expanded.terms.iter().any(|t| t.term == "sona" && t.source == TermSource::Synonym));
    }

    #[test]
    fn test_transliteration_expansion() {
        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand("sona loan");

        // "sona" should be transliterated
        assert!(expanded.terms.iter().any(|t| t.source == TermSource::Transliteration));
    }

    #[test]
    fn test_domain_expansion() {
        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand("interest rate kya hai");

        // "kya hai" should trigger domain expansion
        assert!(expanded.stats.domain_expansions > 0);
    }

    #[test]
    fn test_expand_to_string() {
        let expander = QueryExpander::gold_loan();
        let query_string = expander.expand_to_string("gold loan");

        // Should contain boosted original terms
        assert!(query_string.contains("gold^2.0"));
        assert!(query_string.contains("loan^2.0"));
    }

    #[test]
    fn test_custom_synonym() {
        let expander = QueryExpander::gold_loan();
        expander.add_synonym("test", &["custom1", "custom2"]);

        let expanded = expander.expand("test query");
        assert!(expanded.terms.iter().any(|t| t.term == "custom1"));
    }

    #[test]
    fn test_no_duplicate_terms() {
        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand("gold sona"); // gold and sona are synonyms

        // Should not have duplicate terms
        let term_strings: Vec<&str> = expanded.terms.iter().map(|t| t.term.as_str()).collect();
        let unique: std::collections::HashSet<&str> = term_strings.iter().copied().collect();
        assert_eq!(term_strings.len(), unique.len());
    }

    #[test]
    fn test_original_term_boost() {
        let expander = QueryExpander::gold_loan();
        let expanded = expander.expand("loan");

        // Original term should have higher weight than expansions
        let original = expanded.terms.iter().find(|t| t.term == "loan" && t.source == TermSource::Original);
        let expansion = expanded.terms.iter().find(|t| t.source == TermSource::Synonym);

        if let (Some(orig), Some(exp)) = (original, expansion) {
            assert!(orig.weight > exp.weight);
        }
    }
}
