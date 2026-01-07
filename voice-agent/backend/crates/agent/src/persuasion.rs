//! P2 FIX: Persuasion Engine for Objection Handling
//!
//! Implements objection handling patterns for sales:
//! - Acknowledge: Validate customer concerns
//! - Reframe: Present alternative perspective
//! - Evidence: Provide supporting facts/data
//! - Value Proposition: Articulate key benefits
//!
//! Supports both English and Hindi responses.
//!
//! P13 FIX: Now supports config-driven objection handling via AgentDomainView.
//! Use `from_view()` constructor to load responses from objections.yaml.
//!
//! ## Phase 2 (Domain-Agnosticism): PersuasionStrategy Trait
//!
//! The `PersuasionStrategy` trait abstracts persuasion/objection handling, allowing
//! domain-agnostic agents to work with any persuasion implementation. This enables:
//! - Domain-specific objection handlers
//! - Testing with mock persuasion engines
//! - Alternative persuasion strategies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use voice_agent_config::domain::AgentDomainView;
use voice_agent_core::Language;

// =============================================================================
// Phase 2: PersuasionStrategy Trait (Domain-Agnostic Abstraction)
// =============================================================================

/// Trait for persuasion/objection handling abstraction
///
/// This trait allows domain-agnostic agents to work with any persuasion
/// implementation. Implementations must be thread-safe (Send + Sync).
///
/// # Example
/// ```ignore
/// // Agent uses trait bound instead of concrete PersuasionEngine
/// pub struct DomainAgent<P: PersuasionStrategy> {
///     persuasion: Arc<P>,
///     // ...
/// }
/// ```
pub trait PersuasionStrategy: Send + Sync {
    /// Handle an objection and return appropriate response
    fn handle_objection(&self, text: &str, language: Language) -> Option<ObjectionResponse>;

    /// Get response for a specific objection type
    fn get_response(
        &self,
        objection_type: ObjectionType,
        language: Language,
    ) -> Option<ObjectionResponse>;

    /// Get value proposition for customer segment
    fn get_value_proposition(&self, segment: &str) -> Option<ValueProposition>;

    /// Get competitor comparison data
    fn get_competitor_comparison(&self, competitor: &str) -> Option<CompetitorComparison>;

    /// Calculate savings for switching from a competitor
    fn calculate_switch_savings(
        &self,
        competitor: &str,
        loan_amount: f64,
    ) -> Option<SwitchSavings>;

    /// Generate a full persuasion script for a scenario
    fn generate_script(
        &self,
        objection_type: ObjectionType,
        language: Language,
        customer_segment: Option<&str>,
    ) -> PersuasionScript;
}

/// Types of objections commonly encountered in gold loan sales
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectionType {
    /// "I'm not sure about this" / "Is it safe?"
    Safety,
    /// "Interest rate is too high"
    InterestRate,
    /// "I don't want to leave my gold"
    GoldSecurity,
    /// "Process seems complicated"
    ProcessComplexity,
    /// "I need to think about it"
    NeedTime,
    /// "I'm happy with my current lender"
    CurrentLenderSatisfaction,
    /// "Hidden charges/fees"
    HiddenCharges,
    /// "Documentation is too much"
    Documentation,
    /// "I don't trust banks"
    TrustIssues,
    /// Unknown/other objection
    Other,
}

impl ObjectionType {
    /// Detect objection type from text
    pub fn detect(text: &str) -> Self {
        let lower = text.to_lowercase();

        // Safety/trust concerns
        if lower.contains("safe")
            || lower.contains("trust")
            || lower.contains("bharosa")
            || lower.contains("डर")
            || lower.contains("सुरक्षित")
        {
            return Self::Safety;
        }

        // Interest rate objection
        if lower.contains("interest")
            || lower.contains("rate")
            || lower.contains("byaj")
            || lower.contains("ब्याज")
            || lower.contains("महंगा")
            || lower.contains("high")
        {
            return Self::InterestRate;
        }

        // Gold security concerns
        if lower.contains("gold") && (lower.contains("leave") || lower.contains("give"))
            || lower.contains("सोना") && (lower.contains("देना") || lower.contains("छोड़"))
            || lower.contains("my gold")
            || lower.contains("मेरा सोना")
        {
            return Self::GoldSecurity;
        }

        // Process complexity
        if lower.contains("complicated")
            || lower.contains("difficult")
            || lower.contains("मुश्किल")
            || lower.contains("complex")
            || lower.contains("time consuming")
        {
            return Self::ProcessComplexity;
        }

        // Need time to decide
        if lower.contains("think")
            || lower.contains("later")
            || lower.contains("sochna")
            || lower.contains("सोचना")
            || lower.contains("बाद में")
            || lower.contains("not now")
        {
            return Self::NeedTime;
        }

        // Happy with current lender
        if lower.contains("happy")
            || lower.contains("satisfied")
            || lower.contains("khush")
            || lower.contains("current")
            || lower.contains("already have")
            || lower.contains("muthoot")
            || lower.contains("manappuram")
            || lower.contains("iifl")
        {
            return Self::CurrentLenderSatisfaction;
        }

        // Hidden charges concern
        if lower.contains("hidden")
            || lower.contains("extra")
            || lower.contains("charges")
            || lower.contains("छुपे")
            || lower.contains("अतिरिक्त")
        {
            return Self::HiddenCharges;
        }

        // Documentation concern
        if lower.contains("document")
            || lower.contains("paper")
            || lower.contains("kagaz")
            || lower.contains("कागज")
            || lower.contains("दस्तावेज")
        {
            return Self::Documentation;
        }

        // Trust issues with banks
        if lower.contains("bank") && (lower.contains("trust") || lower.contains("believe"))
            || lower.contains("बैंक") && lower.contains("भरोसा")
        {
            return Self::TrustIssues;
        }

        Self::Other
    }
}

/// Objection handling response with acknowledge-reframe-evidence pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectionResponse {
    /// Acknowledgment of the concern (validates customer feeling)
    pub acknowledge: String,
    /// Reframe to shift perspective
    pub reframe: String,
    /// Evidence/facts to support the reframe
    pub evidence: String,
    /// Call to action
    pub call_to_action: String,
}

/// Value proposition for gold loans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueProposition {
    /// Headline benefit
    pub headline: String,
    /// Supporting points
    pub points: Vec<String>,
    /// Differentiator from competition
    pub differentiator: String,
    /// Social proof (testimonial, statistic)
    pub social_proof: String,
}

/// Persuasion engine for handling objections
pub struct PersuasionEngine {
    /// Objection handlers per language
    handlers: HashMap<(ObjectionType, Language), ObjectionResponse>,
    /// Value propositions per customer segment
    value_propositions: HashMap<String, ValueProposition>,
    /// Competition comparison data
    competition_data: HashMap<String, CompetitorComparison>,
}

/// Comparison with a competitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorComparison {
    /// Competitor name
    pub name: String,
    /// Their interest rate
    pub their_rate: f64,
    /// Kotak's rate
    pub our_rate: f64,
    /// Monthly savings on 1 lakh
    pub monthly_savings_per_lakh: f64,
    /// Additional benefits we offer
    pub our_advantages: Vec<String>,
}

impl PersuasionEngine {
    /// Create a new persuasion engine with default handlers
    ///
    /// Note: Prefer `from_view()` for config-driven responses.
    pub fn new() -> Self {
        let mut engine = Self {
            handlers: HashMap::new(),
            value_propositions: HashMap::new(),
            competition_data: HashMap::new(),
        };

        engine.register_default_handlers();
        engine.register_value_propositions();
        engine.register_competition_data();

        engine
    }

    /// P13 FIX: Create persuasion engine from config view
    ///
    /// Loads objection handlers and competitor data from YAML configuration.
    /// This is the preferred constructor for production use.
    pub fn from_view(view: &Arc<AgentDomainView>) -> Self {
        let mut engine = Self {
            handlers: HashMap::new(),
            value_propositions: HashMap::new(),
            competition_data: HashMap::new(),
        };

        // Load objection handlers from config
        engine.load_handlers_from_config(view);

        // Load competitor data from config
        engine.load_competition_from_config(view);

        // Value propositions (keep default for now, could be loaded from segments config)
        engine.register_value_propositions();

        engine
    }

    /// P13 FIX: Load objection handlers from config
    fn load_handlers_from_config(&mut self, view: &Arc<AgentDomainView>) {
        let objections_config = view.objections_config();

        // Map config objection types to enum
        let type_mapping: HashMap<&str, ObjectionType> = [
            ("safety", ObjectionType::Safety),
            ("interest_rate", ObjectionType::InterestRate),
            ("gold_security", ObjectionType::GoldSecurity),
            ("process_complexity", ObjectionType::ProcessComplexity),
            ("need_time", ObjectionType::NeedTime),
            ("current_lender_satisfaction", ObjectionType::CurrentLenderSatisfaction),
            ("hidden_charges", ObjectionType::HiddenCharges),
            ("documentation", ObjectionType::Documentation),
            ("trust_issues", ObjectionType::TrustIssues),
        ].into_iter().collect();

        // Load English and Hindi responses for each objection type
        for (config_type, objection_type) in &type_mapping {
            // English
            if let Some(config_response) = objections_config.get_response(config_type, "en") {
                self.handlers.insert(
                    (*objection_type, Language::English),
                    ObjectionResponse {
                        acknowledge: config_response.acknowledge.clone(),
                        reframe: config_response.reframe.clone(),
                        evidence: config_response.evidence.clone(),
                        call_to_action: config_response.call_to_action.clone(),
                    },
                );
            }

            // Hindi
            if let Some(config_response) = objections_config.get_response(config_type, "hi") {
                self.handlers.insert(
                    (*objection_type, Language::Hindi),
                    ObjectionResponse {
                        acknowledge: config_response.acknowledge.clone(),
                        reframe: config_response.reframe.clone(),
                        evidence: config_response.evidence.clone(),
                        call_to_action: config_response.call_to_action.clone(),
                    },
                );
            }
        }

        // Load default/other responses
        if let Some(default_response) = objections_config.get_default_response("en") {
            self.handlers.insert(
                (ObjectionType::Other, Language::English),
                ObjectionResponse {
                    acknowledge: default_response.acknowledge.clone(),
                    reframe: default_response.reframe.clone(),
                    evidence: default_response.evidence.clone(),
                    call_to_action: default_response.call_to_action.clone(),
                },
            );
        }
        if let Some(default_response) = objections_config.get_default_response("hi") {
            self.handlers.insert(
                (ObjectionType::Other, Language::Hindi),
                ObjectionResponse {
                    acknowledge: default_response.acknowledge.clone(),
                    reframe: default_response.reframe.clone(),
                    evidence: default_response.evidence.clone(),
                    call_to_action: default_response.call_to_action.clone(),
                },
            );
        }
    }

    /// P13 FIX: Load competitor comparison data from config
    fn load_competition_from_config(&mut self, view: &Arc<AgentDomainView>) {
        let our_best_rate = view.our_rate_for_amount(500_000.0); // Premium rate

        // Load from config competitors
        let competitors = ["muthoot", "manappuram", "iifl"];
        for competitor_id in competitors {
            if let Some(their_rate) = view.get_competitor_rate(competitor_id) {
                let monthly_savings = (their_rate - our_best_rate) / 100.0 / 12.0 * 100_000.0;

                self.competition_data.insert(
                    competitor_id.to_string(),
                    CompetitorComparison {
                        name: match competitor_id {
                            "muthoot" => "Muthoot Finance".to_string(),
                            "manappuram" => "Manappuram Finance".to_string(),
                            "iifl" => "IIFL Finance".to_string(),
                            _ => competitor_id.to_string(),
                        },
                        their_rate,
                        our_rate: our_best_rate,
                        monthly_savings_per_lakh: monthly_savings,
                        our_advantages: vec![
                            "Bank-level security".to_string(),
                            "RBI-regulated".to_string(),
                            "Doorstep service".to_string(),
                            "Zero foreclosure charges".to_string(),
                        ],
                    },
                );
            }
        }
    }

    /// Register default objection handlers for English and Hindi
    ///
    /// Note: This is a fallback. Prefer config-driven loading via `from_view()`.
    fn register_default_handlers(&mut self) {
        // Safety objection - English
        self.handlers.insert(
            (ObjectionType::Safety, Language::English),
            ObjectionResponse {
                acknowledge: "I completely understand your concern about safety. It's natural to be cautious with something as valuable as gold.".to_string(),
                reframe: "Actually, keeping gold at a bank is much safer than keeping it at home. Kotak has state-of-the-art vaults with 24/7 security, insurance coverage, and a 70-year track record.".to_string(),
                evidence: "We have never had a single case of gold loss or theft. Your gold is insured for its full value and stored in RBI-regulated vaults.".to_string(),
                call_to_action: "Would you like me to explain our security process in more detail?".to_string(),
            }
        );

        // Safety objection - Hindi
        self.handlers.insert(
            (ObjectionType::Safety, Language::Hindi),
            ObjectionResponse {
                acknowledge: "मैं आपकी चिंता पूरी तरह समझता/समझती हूं। सोने जैसी कीमती चीज़ के साथ सावधान रहना स्वाभाविक है।".to_string(),
                reframe: "असल में, बैंक में सोना रखना घर में रखने से ज़्यादा सुरक्षित है। कोटक के पास अत्याधुनिक वॉल्ट हैं जिनमें 24/7 सुरक्षा, बीमा कवरेज और 70 साल का ट्रैक रिकॉर्ड है।".to_string(),
                evidence: "हमारे यहां सोने के नुकसान या चोरी का एक भी मामला नहीं हुआ है। आपका सोना पूरी कीमत के लिए बीमित है और RBI-विनियमित वॉल्ट में रखा जाता है।".to_string(),
                call_to_action: "क्या आप चाहेंगे कि मैं हमारी सुरक्षा प्रक्रिया के बारे में और विस्तार से बताऊं?".to_string(),
            }
        );

        // Interest rate objection - English
        self.handlers.insert(
            (ObjectionType::InterestRate, Language::English),
            ObjectionResponse {
                acknowledge: "I hear you - interest rates are definitely an important factor in choosing a lender.".to_string(),
                reframe: "Here's the thing: our rate at 9.5% for premium customers is actually one of the lowest in the market. NBFCs like Muthoot and Manappuram charge 18-20%.".to_string(),
                evidence: "On a 5 lakh loan, you'd save approximately ₹3,500 per month compared to Muthoot - that's ₹42,000 per year!".to_string(),
                call_to_action: "Shall I calculate the exact savings based on your loan amount?".to_string(),
            }
        );

        // Interest rate objection - Hindi
        self.handlers.insert(
            (ObjectionType::InterestRate, Language::Hindi),
            ObjectionResponse {
                acknowledge: "मैं समझता/समझती हूं - ब्याज दर लेंडर चुनने में ज़रूरी फैक्टर है।".to_string(),
                reframe: "देखिए, प्रीमियम ग्राहकों के लिए हमारी 9.5% दर बाज़ार में सबसे कम में से एक है। Muthoot और Manappuram जैसे NBFC 18-20% चार्ज करते हैं।".to_string(),
                evidence: "5 लाख के लोन पर, आप Muthoot की तुलना में लगभग ₹3,500 प्रति माह बचाएंगे - यानी साल में ₹42,000!".to_string(),
                call_to_action: "क्या मैं आपकी लोन राशि के आधार पर सटीक बचत की गणना करूं?".to_string(),
            }
        );

        // Gold security objection - English
        self.handlers.insert(
            (ObjectionType::GoldSecurity, Language::English),
            ObjectionResponse {
                acknowledge: "I understand - your gold has emotional and financial value. It's natural to feel protective about it.".to_string(),
                reframe: "Think of it this way: your gold stays safely with us while you get the funds you need. And when you repay, you get your exact same gold back - every gram, every piece.".to_string(),
                evidence: "We provide detailed receipts with photos and weight certification. Many customers actually feel relieved knowing their gold is in a secure vault rather than at home.".to_string(),
                call_to_action: "Would you like to visit our branch and see our vault security firsthand?".to_string(),
            }
        );

        // Gold security objection - Hindi
        self.handlers.insert(
            (ObjectionType::GoldSecurity, Language::Hindi),
            ObjectionResponse {
                acknowledge: "मैं समझता/समझती हूं - आपके सोने की भावनात्मक और आर्थिक दोनों कीमत है। इसके प्रति सुरक्षात्मक महसूस करना स्वाभाविक है।".to_string(),
                reframe: "इसे इस तरह सोचिए: आपका सोना हमारे पास सुरक्षित रहता है जबकि आपको ज़रूरी पैसे मिलते हैं। और जब आप चुकाते हैं, तो आपको वही सोना वापस मिलता है - हर ग्राम, हर टुकड़ा।".to_string(),
                evidence: "हम फोटो और वजन प्रमाण पत्र के साथ विस्तृत रसीद देते हैं। कई ग्राहक वास्तव में राहत महसूस करते हैं कि उनका सोना घर के बजाय सुरक्षित वॉल्ट में है।".to_string(),
                call_to_action: "क्या आप हमारी ब्रांच में आकर हमारी वॉल्ट सुरक्षा खुद देखना चाहेंगे?".to_string(),
            }
        );

        // Process complexity objection - English
        self.handlers.insert(
            (ObjectionType::ProcessComplexity, Language::English),
            ObjectionResponse {
                acknowledge: "I completely get it - no one wants to deal with complicated paperwork and long waits.".to_string(),
                reframe: "That's actually one of our biggest strengths! Kotak's gold loan process is designed to be quick and simple. Just bring your gold and one ID proof.".to_string(),
                evidence: "Most customers complete the entire process in under 30 minutes. We've done over 1 lakh gold loans this year with an average disbursal time of just 15 minutes.".to_string(),
                call_to_action: "The process is really straightforward. Would you like me to walk you through the simple steps?".to_string(),
            }
        );

        // Process complexity - Hindi
        self.handlers.insert(
            (ObjectionType::ProcessComplexity, Language::Hindi),
            ObjectionResponse {
                acknowledge: "मैं पूरी तरह समझता/समझती हूं - कोई भी जटिल कागज़ी काम और लंबी प्रतीक्षा नहीं चाहता।".to_string(),
                reframe: "यह असल में हमारी सबसे बड़ी ताकत है! कोटक की गोल्ड लोन प्रक्रिया तेज़ और सरल है। बस अपना सोना और एक ID प्रूफ लाइए।".to_string(),
                evidence: "ज़्यादातर ग्राहक पूरी प्रक्रिया 30 मिनट से कम में पूरी करते हैं। इस साल हमने 1 लाख से ज़्यादा गोल्ड लोन किए हैं, औसत वितरण समय सिर्फ 15 मिनट।".to_string(),
                call_to_action: "प्रक्रिया वास्तव में बहुत सीधी है। क्या मैं आपको सरल चरणों के बारे में बताऊं?".to_string(),
            }
        );

        // Need time objection - English
        self.handlers.insert(
            (ObjectionType::NeedTime, Language::English),
            ObjectionResponse {
                acknowledge: "Of course, taking time to make an informed decision is wise. I respect that.".to_string(),
                reframe: "However, I should mention that our current promotional rate of 9.5% is valid only till month-end. After that, it goes back to 10.5%.".to_string(),
                evidence: "On a 3 lakh loan over 12 months, the difference is about ₹3,000. Many customers who waited ended up paying the higher rate.".to_string(),
                call_to_action: "Would you like me to do a quick eligibility check now? There's no commitment, and you'll have all the information to decide.".to_string(),
            }
        );

        // Need time - Hindi
        self.handlers.insert(
            (ObjectionType::NeedTime, Language::Hindi),
            ObjectionResponse {
                acknowledge: "बिल्कुल, सोच-समझकर फैसला लेना समझदारी है। मैं इसका सम्मान करता/करती हूं।".to_string(),
                reframe: "हालांकि, मुझे बताना चाहिए कि हमारी मौजूदा प्रमोशनल दर 9.5% सिर्फ महीने के अंत तक वैध है। उसके बाद यह वापस 10.5% हो जाएगी।".to_string(),
                evidence: "12 महीने के 3 लाख के लोन पर, अंतर लगभग ₹3,000 का है। कई ग्राहक जिन्होंने इंतज़ार किया, उन्हें ऊंची दर देनी पड़ी।".to_string(),
                call_to_action: "क्या मैं अभी एक त्वरित पात्रता जांच करूं? कोई प्रतिबद्धता नहीं है, और आपके पास फैसला लेने के लिए सारी जानकारी होगी।".to_string(),
            }
        );

        // Current lender satisfaction - English
        self.handlers.insert(
            (ObjectionType::CurrentLenderSatisfaction, Language::English),
            ObjectionResponse {
                acknowledge: "That's great that you have a good relationship with your current lender. Loyalty is valuable.".to_string(),
                reframe: "But have you compared the rates recently? Many customers are surprised to find they're paying 7-8% more than necessary.".to_string(),
                evidence: "We've helped over 50,000 customers switch from Muthoot and Manappuram this year alone, saving them an average of ₹40,000 annually.".to_string(),
                call_to_action: "Would you mind sharing your current interest rate? I can quickly show you the potential savings.".to_string(),
            }
        );

        // Current lender satisfaction - Hindi
        self.handlers.insert(
            (ObjectionType::CurrentLenderSatisfaction, Language::Hindi),
            ObjectionResponse {
                acknowledge: "यह अच्छी बात है कि आपका अपने मौजूदा लेंडर के साथ अच्छा संबंध है। वफादारी मूल्यवान है।".to_string(),
                reframe: "लेकिन क्या आपने हाल ही में दरों की तुलना की है? कई ग्राहक यह जानकर हैरान होते हैं कि वे ज़रूरत से 7-8% ज़्यादा दे रहे हैं।".to_string(),
                evidence: "हमने अकेले इस साल Muthoot और Manappuram से 50,000 से ज़्यादा ग्राहकों को स्विच करने में मदद की है, जिससे उन्हें औसतन ₹40,000 सालाना की बचत हुई।".to_string(),
                call_to_action: "क्या आप अपनी मौजूदा ब्याज दर बता सकते हैं? मैं जल्दी से संभावित बचत दिखा सकता/सकती हूं।".to_string(),
            }
        );

        // Hidden charges - English
        self.handlers.insert(
            (ObjectionType::HiddenCharges, Language::English),
            ObjectionResponse {
                acknowledge: "You're absolutely right to ask about this. Transparency is essential, and unfortunately some lenders aren't upfront about fees.".to_string(),
                reframe: "At Kotak, we're proud of our complete transparency. Our processing fee is just 1%, and there are NO hidden charges - no documentation fee, no valuation fee, no prepayment penalty.".to_string(),
                evidence: "We provide a complete fee breakdown in writing before you sign anything. In fact, RBI regulations require us to disclose all charges, and we go beyond that with our fee guarantee.".to_string(),
                call_to_action: "Would you like me to send you our complete fee structure in writing?".to_string(),
            }
        );

        // Hidden charges - Hindi
        self.handlers.insert(
            (ObjectionType::HiddenCharges, Language::Hindi),
            ObjectionResponse {
                acknowledge: "इस बारे में पूछना बिल्कुल सही है। पारदर्शिता ज़रूरी है, और दुर्भाग्य से कुछ लेंडर फीस के बारे में स्पष्ट नहीं होते।".to_string(),
                reframe: "कोटक में, हम अपनी पूर्ण पारदर्शिता पर गर्व करते हैं। हमारी प्रोसेसिंग फीस सिर्फ 1% है, और कोई छुपे हुए शुल्क नहीं हैं - कोई डॉक्यूमेंटेशन फीस नहीं, कोई वैल्यूएशन फीस नहीं, कोई प्रीपेमेंट पेनल्टी नहीं।".to_string(),
                evidence: "हम आपके साइन करने से पहले लिखित में पूरा फीस विवरण देते हैं। वास्तव में, RBI नियमों के अनुसार हमें सभी शुल्क बताने होते हैं, और हम अपनी फीस गारंटी के साथ उससे भी आगे जाते हैं।".to_string(),
                call_to_action: "क्या आप चाहेंगे कि मैं आपको लिखित में हमारा पूरा फीस स्ट्रक्चर भेजूं?".to_string(),
            }
        );

        // Documentation objection - English
        self.handlers.insert(
            (ObjectionType::Documentation, Language::English),
            ObjectionResponse {
                acknowledge: "I understand - paperwork can be tedious. Nobody wants to gather a pile of documents.".to_string(),
                reframe: "Good news! Kotak gold loan requires minimal documentation. All you need is one photo ID like Aadhaar or PAN, and your gold. That's it!".to_string(),
                evidence: "Unlike other loans that need income proof, bank statements, and employment verification, gold loan uses your gold as security - so minimal paperwork needed.".to_string(),
                call_to_action: "Do you have an Aadhaar card? If yes, you're already eligible. Would you like to proceed?".to_string(),
            }
        );

        // Documentation - Hindi
        self.handlers.insert(
            (ObjectionType::Documentation, Language::Hindi),
            ObjectionResponse {
                acknowledge: "मैं समझता/समझती हूं - कागज़ी काम थकाऊ हो सकता है। कोई भी दस्तावेज़ों का ढेर इकट्ठा नहीं करना चाहता।".to_string(),
                reframe: "अच्छी खबर! कोटक गोल्ड लोन के लिए न्यूनतम दस्तावेज़ चाहिए। बस एक फोटो ID जैसे आधार या PAN, और आपका सोना। बस इतना ही!".to_string(),
                evidence: "अन्य लोन जिनमें इनकम प्रूफ, बैंक स्टेटमेंट, और रोज़गार सत्यापन चाहिए, उनके विपरीत गोल्ड लोन में आपका सोना सिक्योरिटी का काम करता है - इसलिए न्यूनतम कागज़ी काम।".to_string(),
                call_to_action: "क्या आपके पास आधार कार्ड है? अगर हां, तो आप पहले से पात्र हैं। क्या आप आगे बढ़ना चाहेंगे?".to_string(),
            }
        );

        // Trust issues - English
        self.handlers.insert(
            (ObjectionType::TrustIssues, Language::English),
            ObjectionResponse {
                acknowledge: "I appreciate your honesty. Trust has to be earned, and I understand if you've had negative experiences before.".to_string(),
                reframe: "Kotak Mahindra Bank is one of India's most trusted private banks, regulated by RBI. We've been serving customers for over 35 years with complete transparency.".to_string(),
                evidence: "We're rated AA+ by CRISIL, have over 1,600 branches nationwide, and serve more than 40 million customers. You can verify our credentials anytime on RBI's website.".to_string(),
                call_to_action: "Would you like to visit our nearest branch and meet our team in person? Sometimes seeing is believing.".to_string(),
            }
        );

        // Trust issues - Hindi
        self.handlers.insert(
            (ObjectionType::TrustIssues, Language::Hindi),
            ObjectionResponse {
                acknowledge: "मैं आपकी ईमानदारी की सराहना करता/करती हूं। भरोसा कमाना पड़ता है, और मैं समझता/समझती हूं अगर आपके पहले नकारात्मक अनुभव रहे हों।".to_string(),
                reframe: "कोटक महिंद्रा बैंक भारत के सबसे विश्वसनीय निजी बैंकों में से एक है, RBI द्वारा विनियमित। हम 35 से ज़्यादा सालों से पूर्ण पारदर्शिता के साथ ग्राहकों की सेवा कर रहे हैं।".to_string(),
                evidence: "हमें CRISIL द्वारा AA+ रेटिंग मिली है, देशभर में 1,600 से ज़्यादा शाखाएं हैं, और 4 करोड़ से ज़्यादा ग्राहकों की सेवा करते हैं। आप RBI की वेबसाइट पर कभी भी हमारी क्रेडेंशियल्स वेरीफाई कर सकते हैं।".to_string(),
                call_to_action: "क्या आप हमारी नज़दीकी शाखा में आकर हमारी टीम से व्यक्तिगत रूप से मिलना चाहेंगे? कभी-कभी देखकर ही विश्वास होता है।".to_string(),
            }
        );

        // Other/generic objection - English
        self.handlers.insert(
            (ObjectionType::Other, Language::English),
            ObjectionResponse {
                acknowledge: "I appreciate you sharing your concern. It's important that we address any doubts you have.".to_string(),
                reframe: "Let me understand your specific concern better so I can help. What's the main thing holding you back?".to_string(),
                evidence: "Whatever your concern, I want to make sure you have all the facts. We've helped lakhs of customers just like you.".to_string(),
                call_to_action: "Could you tell me more about what's on your mind?".to_string(),
            }
        );

        // Other/generic - Hindi
        self.handlers.insert(
            (ObjectionType::Other, Language::Hindi),
            ObjectionResponse {
                acknowledge: "अपनी चिंता साझा करने के लिए धन्यवाद। आपकी कोई भी शंका दूर करना महत्वपूर्ण है।".to_string(),
                reframe: "मुझे आपकी विशिष्ट चिंता बेहतर समझने दीजिए ताकि मैं मदद कर सकूं। मुख्य बात क्या है जो आपको रोक रही है?".to_string(),
                evidence: "आपकी चिंता जो भी हो, मैं सुनिश्चित करना चाहता/चाहती हूं कि आपके पास सारे तथ्य हों। हमने आप जैसे लाखों ग्राहकों की मदद की है।".to_string(),
                call_to_action: "क्या आप बता सकते हैं कि आपके मन में क्या चल रहा है?".to_string(),
            }
        );
    }

    /// Register value propositions
    fn register_value_propositions(&mut self) {
        // Premium customer value proposition
        self.value_propositions.insert(
            "premium".to_string(),
            ValueProposition {
                headline: "Lowest Interest Rate in the Market".to_string(),
                points: vec![
                    "Interest rate as low as 9.5% p.a.".to_string(),
                    "Loan up to 75% of gold value".to_string(),
                    "15-minute instant disbursal".to_string(),
                    "No prepayment penalty".to_string(),
                    "Doorstep service available".to_string(),
                ],
                differentiator: "Save ₹40,000+ annually compared to NBFCs".to_string(),
                social_proof: "Trusted by 50 lakh+ customers across India".to_string(),
            },
        );

        // First-time borrower value proposition
        self.value_propositions.insert(
            "first_time".to_string(),
            ValueProposition {
                headline: "Your Gold, Your Security, Your Loan".to_string(),
                points: vec![
                    "Simple process - just bring gold and ID".to_string(),
                    "Get cash in minutes, not days".to_string(),
                    "Your gold is fully insured".to_string(),
                    "Flexible repayment options".to_string(),
                    "Zero documentation fee".to_string(),
                ],
                differentiator: "No income proof or bank statements needed".to_string(),
                social_proof: "10 lakh+ first-time borrowers served last year".to_string(),
            },
        );

        // Switcher value proposition
        self.value_propositions.insert(
            "switcher".to_string(),
            ValueProposition {
                headline: "Switch & Save with Zero Hassle".to_string(),
                points: vec![
                    "We handle the entire transfer process".to_string(),
                    "Save 7-8% on interest immediately".to_string(),
                    "Same day balance transfer".to_string(),
                    "Processing fee waived for switchers".to_string(),
                    "Top-up available instantly".to_string(),
                ],
                differentiator: "₹3,500 monthly savings on ₹5 lakh loan".to_string(),
                social_proof: "50,000+ customers switched from Muthoot/Manappuram this year"
                    .to_string(),
            },
        );
    }

    /// Register competition comparison data
    fn register_competition_data(&mut self) {
        self.competition_data.insert(
            "muthoot".to_string(),
            CompetitorComparison {
                name: "Muthoot Finance".to_string(),
                their_rate: 18.0,
                our_rate: 9.5,
                monthly_savings_per_lakh: 708.0, // (18-9.5)/100/12 * 100000
                our_advantages: vec![
                    "Bank-level security vs NBFC".to_string(),
                    "RBI-regulated".to_string(),
                    "Doorstep service".to_string(),
                    "Digital account access".to_string(),
                ],
            },
        );

        self.competition_data.insert(
            "manappuram".to_string(),
            CompetitorComparison {
                name: "Manappuram Finance".to_string(),
                their_rate: 19.0,
                our_rate: 9.5,
                monthly_savings_per_lakh: 791.0,
                our_advantages: vec![
                    "Lower interest rate".to_string(),
                    "Larger branch network".to_string(),
                    "Better customer service rating".to_string(),
                    "No hidden charges guarantee".to_string(),
                ],
            },
        );

        self.competition_data.insert(
            "iifl".to_string(),
            CompetitorComparison {
                name: "IIFL Finance".to_string(),
                their_rate: 17.5,
                our_rate: 9.5,
                monthly_savings_per_lakh: 666.0,
                our_advantages: vec![
                    "Higher LTV ratio".to_string(),
                    "Faster processing".to_string(),
                    "Doorstep gold pickup".to_string(),
                    "Flexible tenure options".to_string(),
                ],
            },
        );
    }

    /// Handle an objection and return appropriate response
    pub fn handle_objection(&self, text: &str, language: Language) -> Option<ObjectionResponse> {
        let objection_type = ObjectionType::detect(text);
        self.get_response(objection_type, language)
    }

    /// Get response for specific objection type
    pub fn get_response(
        &self,
        objection_type: ObjectionType,
        language: Language,
    ) -> Option<ObjectionResponse> {
        // Try exact language match first
        if let Some(response) = self.handlers.get(&(objection_type, language)) {
            return Some(response.clone());
        }

        // Fall back to English
        self.handlers
            .get(&(objection_type, Language::English))
            .cloned()
    }

    /// Get value proposition for customer segment
    pub fn get_value_proposition(&self, segment: &str) -> Option<ValueProposition> {
        self.value_propositions.get(segment).cloned()
    }

    /// Get competition comparison
    pub fn get_competitor_comparison(&self, competitor: &str) -> Option<CompetitorComparison> {
        let lower = competitor.to_lowercase();
        self.competition_data.get(&lower).cloned()
    }

    /// Calculate savings for switching from competitor
    pub fn calculate_switch_savings(
        &self,
        competitor: &str,
        loan_amount: f64,
    ) -> Option<SwitchSavings> {
        let comparison = self.get_competitor_comparison(competitor)?;

        let their_monthly = loan_amount * (comparison.their_rate / 100.0 / 12.0);
        let our_monthly = loan_amount * (comparison.our_rate / 100.0 / 12.0);
        let monthly_savings = their_monthly - our_monthly;
        let annual_savings = monthly_savings * 12.0;

        Some(SwitchSavings {
            competitor_name: comparison.name,
            competitor_rate: comparison.their_rate,
            our_rate: comparison.our_rate,
            monthly_savings,
            annual_savings,
            loan_amount,
        })
    }

    /// Generate a full persuasion script for a scenario
    pub fn generate_script(
        &self,
        objection_type: ObjectionType,
        language: Language,
        customer_segment: Option<&str>,
    ) -> PersuasionScript {
        let response = self
            .get_response(objection_type, language)
            .unwrap_or_else(|| self.get_response(ObjectionType::Other, language).unwrap());

        let value_prop = customer_segment
            .and_then(|s| self.get_value_proposition(s))
            .unwrap_or_else(|| self.get_value_proposition("premium").unwrap());

        PersuasionScript {
            objection_response: response,
            value_proposition: value_prop,
            language,
        }
    }
}

impl Default for PersuasionEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Phase 2: PersuasionStrategy Implementation for PersuasionEngine
// =============================================================================

impl PersuasionStrategy for PersuasionEngine {
    fn handle_objection(&self, text: &str, language: Language) -> Option<ObjectionResponse> {
        PersuasionEngine::handle_objection(self, text, language)
    }

    fn get_response(
        &self,
        objection_type: ObjectionType,
        language: Language,
    ) -> Option<ObjectionResponse> {
        PersuasionEngine::get_response(self, objection_type, language)
    }

    fn get_value_proposition(&self, segment: &str) -> Option<ValueProposition> {
        PersuasionEngine::get_value_proposition(self, segment)
    }

    fn get_competitor_comparison(&self, competitor: &str) -> Option<CompetitorComparison> {
        PersuasionEngine::get_competitor_comparison(self, competitor)
    }

    fn calculate_switch_savings(
        &self,
        competitor: &str,
        loan_amount: f64,
    ) -> Option<SwitchSavings> {
        PersuasionEngine::calculate_switch_savings(self, competitor, loan_amount)
    }

    fn generate_script(
        &self,
        objection_type: ObjectionType,
        language: Language,
        customer_segment: Option<&str>,
    ) -> PersuasionScript {
        PersuasionEngine::generate_script(self, objection_type, language, customer_segment)
    }
}

/// Savings from switching lenders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchSavings {
    pub competitor_name: String,
    pub competitor_rate: f64,
    pub our_rate: f64,
    pub monthly_savings: f64,
    pub annual_savings: f64,
    pub loan_amount: f64,
}

/// Complete persuasion script
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersuasionScript {
    pub objection_response: ObjectionResponse,
    pub value_proposition: ValueProposition,
    pub language: Language,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_objection_detection() {
        assert_eq!(ObjectionType::detect("Is it safe?"), ObjectionType::Safety);
        assert_eq!(
            ObjectionType::detect("Interest rate is too high"),
            ObjectionType::InterestRate
        );
        assert_eq!(
            ObjectionType::detect("I don't want to leave my gold"),
            ObjectionType::GoldSecurity
        );
        assert_eq!(
            ObjectionType::detect("I need to think about it"),
            ObjectionType::NeedTime
        );
        assert_eq!(
            ObjectionType::detect("I'm happy with Muthoot"),
            ObjectionType::CurrentLenderSatisfaction
        );
    }

    #[test]
    fn test_hindi_objection_detection() {
        assert_eq!(ObjectionType::detect("मुझे डर लगता है"), ObjectionType::Safety);
        assert_eq!(
            ObjectionType::detect("ब्याज दर बहुत ज्यादा है"),
            ObjectionType::InterestRate
        );
        assert_eq!(ObjectionType::detect("मुझे सोचना है"), ObjectionType::NeedTime);
    }

    #[test]
    fn test_handle_objection() {
        let engine = PersuasionEngine::new();

        let response = engine.handle_objection("Is it safe?", Language::English);
        assert!(response.is_some());
        let response = response.unwrap();
        assert!(!response.acknowledge.is_empty());
        assert!(!response.reframe.is_empty());
        assert!(!response.evidence.is_empty());
    }

    #[test]
    fn test_handle_objection_hindi() {
        let engine = PersuasionEngine::new();

        let response = engine.handle_objection("ब्याज दर बहुत ज्यादा है", Language::Hindi);
        assert!(response.is_some());
        let response = response.unwrap();
        // Should contain Hindi text
        assert!(response.acknowledge.contains("समझ"));
    }

    #[test]
    fn test_value_proposition() {
        let engine = PersuasionEngine::new();

        let vp = engine.get_value_proposition("premium");
        assert!(vp.is_some());
        let vp = vp.unwrap();
        assert_eq!(vp.points.len(), 5);
        assert!(vp.headline.contains("Lowest"));
    }

    #[test]
    fn test_competitor_comparison() {
        let engine = PersuasionEngine::new();

        let comp = engine.get_competitor_comparison("muthoot");
        assert!(comp.is_some());
        let comp = comp.unwrap();
        assert!(comp.their_rate > comp.our_rate);
    }

    #[test]
    fn test_switch_savings() {
        let engine = PersuasionEngine::new();

        let savings = engine.calculate_switch_savings("muthoot", 500_000.0);
        assert!(savings.is_some());
        let savings = savings.unwrap();
        assert!(savings.monthly_savings > 0.0);
        assert!(savings.annual_savings > 0.0);
        // Should be around 3500 monthly on 5 lakh
        assert!(savings.monthly_savings > 3000.0);
        assert!(savings.monthly_savings < 4000.0);
    }

    #[test]
    fn test_generate_script() {
        let engine = PersuasionEngine::new();

        let script = engine.generate_script(
            ObjectionType::InterestRate,
            Language::English,
            Some("switcher"),
        );

        assert!(!script.objection_response.acknowledge.is_empty());
        assert!(script.value_proposition.headline.contains("Switch"));
    }
}
