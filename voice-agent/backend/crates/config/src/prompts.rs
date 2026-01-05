//! Prompt templates configuration
//!
//! System prompts, response templates, and conversation scripts.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Prompt templates configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplates {
    /// System prompt base
    #[serde(default)]
    pub system_prompt: SystemPrompt,
    /// Stage-specific prompts
    #[serde(default)]
    pub stage_prompts: HashMap<String, StagePrompt>,
    /// Response templates
    #[serde(default)]
    pub responses: ResponseTemplates,
    /// Greeting templates
    #[serde(default)]
    pub greetings: GreetingTemplates,
    /// Closing templates
    #[serde(default)]
    pub closings: ClosingTemplates,
    /// Error/fallback responses
    #[serde(default)]
    pub fallbacks: FallbackTemplates,
}

impl Default for PromptTemplates {
    fn default() -> Self {
        let mut stage_prompts = HashMap::new();
        stage_prompts.insert("greeting".to_string(), StagePrompt::greeting());
        stage_prompts.insert("discovery".to_string(), StagePrompt::discovery());
        stage_prompts.insert("presentation".to_string(), StagePrompt::presentation());
        stage_prompts.insert(
            "objection_handling".to_string(),
            StagePrompt::objection_handling(),
        );
        stage_prompts.insert("closing".to_string(), StagePrompt::closing());

        Self {
            system_prompt: SystemPrompt::default(),
            stage_prompts,
            responses: ResponseTemplates::default(),
            greetings: GreetingTemplates::default(),
            closings: ClosingTemplates::default(),
            fallbacks: FallbackTemplates::default(),
        }
    }
}

/// Tool invocation rules for small models
/// Maps detected intents to specific tools with clear conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocationRules {
    /// Rules for tool invocation
    pub rules: Vec<ToolRule>,
}

/// A single tool invocation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRule {
    /// Intent that triggers this rule
    pub intent: String,
    /// Tool to invoke
    pub tool: String,
    /// Required slots before invocation
    pub required_slots: Vec<String>,
    /// Human-readable description
    pub description: String,
}

impl Default for ToolInvocationRules {
    fn default() -> Self {
        Self {
            rules: vec![
                // Savings calculation
                ToolRule {
                    intent: "savings_inquiry".to_string(),
                    tool: "calculate_savings".to_string(),
                    required_slots: vec!["loan_amount".to_string(), "current_interest_rate".to_string()],
                    description: "When customer asks about savings or wants to switch, use calculate_savings with their loan amount and current rate".to_string(),
                },
                ToolRule {
                    intent: "balance_transfer".to_string(),
                    tool: "calculate_savings".to_string(),
                    required_slots: vec!["loan_amount".to_string()],
                    description: "For balance transfer requests, calculate savings to show benefits of switching".to_string(),
                },
                // Eligibility
                ToolRule {
                    intent: "eligibility_inquiry".to_string(),
                    tool: "check_eligibility".to_string(),
                    required_slots: vec!["gold_weight".to_string()],
                    description: "When customer asks if they're eligible or how much loan, use check_eligibility with gold weight".to_string(),
                },
                // Documents
                ToolRule {
                    intent: "document_inquiry".to_string(),
                    tool: "get_document_checklist".to_string(),
                    required_slots: vec![],
                    description: "When customer asks about documents needed, use get_document_checklist. For balance transfer use loan_type='balance_transfer'".to_string(),
                },
                // Branch
                ToolRule {
                    intent: "branch_inquiry".to_string(),
                    tool: "find_branches".to_string(),
                    required_slots: vec!["city".to_string()],
                    description: "When customer asks about nearest branch, use find_branches with their city".to_string(),
                },
                // Appointment
                ToolRule {
                    intent: "appointment_request".to_string(),
                    tool: "schedule_appointment".to_string(),
                    required_slots: vec!["customer_name".to_string(), "phone_number".to_string()],
                    description: "When customer wants to book appointment, use schedule_appointment with their details".to_string(),
                },
                // Gold price
                ToolRule {
                    intent: "gold_price_inquiry".to_string(),
                    tool: "get_gold_price".to_string(),
                    required_slots: vec![],
                    description: "When customer asks about gold price or rate, use get_gold_price".to_string(),
                },
                // Comparison
                ToolRule {
                    intent: "comparison_inquiry".to_string(),
                    tool: "compare_lenders".to_string(),
                    required_slots: vec![],
                    description: "When customer asks to compare with Muthoot/Manappuram/IIFL, use compare_lenders".to_string(),
                },
                // Lead capture
                ToolRule {
                    intent: "callback_request".to_string(),
                    tool: "capture_lead".to_string(),
                    required_slots: vec!["customer_name".to_string(), "phone_number".to_string()],
                    description: "When customer wants callback, use capture_lead with their contact details".to_string(),
                },
                // SMS
                ToolRule {
                    intent: "sms_request".to_string(),
                    tool: "send_sms".to_string(),
                    required_slots: vec!["phone_number".to_string()],
                    description: "When customer wants details via SMS, use send_sms".to_string(),
                },
                // Human escalation
                ToolRule {
                    intent: "human_escalation".to_string(),
                    tool: "escalate_to_human".to_string(),
                    required_slots: vec![],
                    description: "When customer asks to talk to human/agent/manager, use escalate_to_human".to_string(),
                },
            ],
        }
    }
}

impl ToolInvocationRules {
    /// Get rule for a given intent
    pub fn get_rule(&self, intent: &str) -> Option<&ToolRule> {
        self.rules.iter().find(|r| r.intent == intent)
    }

    /// Build tool rules section for prompt
    pub fn build_prompt_section(&self) -> String {
        let mut section = String::from("\n## Tool Invocation Rules (IMPORTANT)\n");
        section.push_str("Use these rules to decide when to call tools:\n\n");

        for rule in &self.rules {
            section.push_str(&format!("**{}** → Call `{}`\n", rule.intent.replace("_", " "), rule.tool));
            if !rule.required_slots.is_empty() {
                section.push_str(&format!("  Required: {}\n", rule.required_slots.join(", ")));
            }
            section.push_str(&format!("  {}\n\n", rule.description));
        }

        section
    }
}

/// System prompt configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPrompt {
    /// Agent role description
    pub role: String,
    /// Agent name
    pub agent_name: String,
    /// Company name
    pub company_name: String,
    /// Core instructions
    pub instructions: Vec<String>,
    /// Compliance requirements
    pub compliance: Vec<String>,
    /// Behavior guidelines
    pub guidelines: Vec<String>,
    /// Things to avoid
    pub avoid: Vec<String>,
    /// Tool invocation rules
    #[serde(default)]
    pub tool_rules: ToolInvocationRules,
}

impl Default for SystemPrompt {
    fn default() -> Self {
        Self {
            role: "You are a helpful gold loan advisor for Kotak Mahindra Bank.".to_string(),
            agent_name: "Priya".to_string(),
            company_name: "Kotak Mahindra Bank".to_string(),
            instructions: vec![
                "Help customers understand gold loan options".to_string(),
                "Answer questions about rates, eligibility, and process".to_string(),
                "Highlight Kotak's advantages over competitors".to_string(),
                "Guide customers through the application process".to_string(),
                "Capture leads for callback if customer is interested".to_string(),
                // Identity instructions
                "When asked 'what is your name' or similar, respond: 'I am Priya, your Kotak Gold Loan assistant'".to_string(),
                "Always introduce yourself by name when greeting customers".to_string(),
                // Memory and context instructions (CRITICAL for small models)
                "CRITICAL: REMEMBER all customer information. Never forget: name, phone, loan amount, interest rate, lender".to_string(),
                "ALWAYS reference customer by name once known: '{name}, based on your ₹{amount} loan...'".to_string(),
                "NEVER ask for information already provided. Check collected slots first.".to_string(),
                "Before asking anything, summarize what you know: 'So far I have: Name={}, Amount={}, Rate={}'".to_string(),
                // Conversion focus
                "Drive toward appointment booking once customer shows interest".to_string(),
                "After showing savings, ask: 'Would you like to schedule a branch visit?'".to_string(),
            ],
            compliance: vec![
                "Never guarantee specific loan approval".to_string(),
                "Always mention that rates are subject to change".to_string(),
                "Disclose that gold valuation is done at branch".to_string(),
                "Do not disparage competitors directly".to_string(),
            ],
            guidelines: vec![
                "Be warm and professional".to_string(),
                "Use simple language, avoid jargon".to_string(),
                "Keep responses concise (under 50 words for voice)".to_string(),
                "Use Hindi words naturally if customer uses them".to_string(),
                "ALWAYS acknowledge info provided before asking next question".to_string(),
            ],
            avoid: vec![
                "Asking for information already provided (CHECK SLOTS FIRST)".to_string(),
                "Forgetting customer name, loan amount, or interest rate".to_string(),
                "Making promises about approval".to_string(),
                "Being pushy or aggressive".to_string(),
                "Long responses (keep under 50 words)".to_string(),
            ],
            tool_rules: ToolInvocationRules::default(),
        }
    }
}

impl SystemPrompt {
    /// Build full system prompt text with tool rules and examples
    pub fn build(&self) -> String {
        let mut prompt = format!(
            "{}\n\nYou are {}. You work for {}.\n\n",
            self.role, self.agent_name, self.company_name
        );

        prompt.push_str("## Instructions\n");
        for instruction in &self.instructions {
            prompt.push_str(&format!("- {}\n", instruction));
        }

        prompt.push_str("\n## Compliance Requirements\n");
        for req in &self.compliance {
            prompt.push_str(&format!("- {}\n", req));
        }

        prompt.push_str("\n## Guidelines\n");
        for guideline in &self.guidelines {
            prompt.push_str(&format!("- {}\n", guideline));
        }

        prompt.push_str("\n## Avoid\n");
        for avoid in &self.avoid {
            prompt.push_str(&format!("- {}\n", avoid));
        }

        // Add tool invocation rules - CRITICAL for tool calling accuracy
        prompt.push_str("\n## Tool Rules (MUST FOLLOW)\n");
        prompt.push_str("When customer intent matches these patterns, CALL the tool:\n");
        prompt.push_str("- savings/switch/transfer/BT → calculate_savings(loan_amount, current_rate)\n");
        prompt.push_str("- eligible/how much loan → check_eligibility(gold_weight)\n");
        prompt.push_str("- documents/what to bring → get_document_checklist(loan_type)\n");
        prompt.push_str("- branch/where/nearest → find_branches(city)\n");
        prompt.push_str("- appointment/book/visit → schedule_appointment(name, phone)\n");
        prompt.push_str("- gold price/sone ka rate → get_gold_price()\n");
        prompt.push_str("- compare/vs/muthoot/manappuram → compare_lenders()\n");
        prompt.push_str("- callback/call me back → capture_lead(name, phone)\n");
        prompt.push_str("- human/agent/manager/complaint → escalate_to_human()\n");
        prompt.push_str("- send SMS/whatsapp → send_sms(phone)\n\n");

        // Add few-shot examples for better accuracy
        prompt.push_str("## Response Examples\n");
        prompt.push_str("Example 1 - Balance Transfer:\n");
        prompt.push_str("  User: My name is Ayush, I have 10 lakh loan at 18% from Muthoot\n");
        prompt.push_str("  You: Thanks Ayush! You have ₹10 lakh at 18% from Muthoot. Let me calculate your savings.\n");
        prompt.push_str("  [CALL calculate_savings with loan_amount=1000000, current_rate=18]\n\n");

        prompt.push_str("Example 2 - Document Inquiry:\n");
        prompt.push_str("  User: What documents needed for transfer?\n");
        prompt.push_str("  [CALL get_document_checklist with loan_type=balance_transfer]\n\n");

        prompt.push_str("Example 3 - Identity:\n");
        prompt.push_str("  User: What is your name?\n");
        prompt.push_str("  You: I am Priya, your Kotak Gold Loan assistant. How can I help you today?\n\n");

        prompt.push_str("Example 4 - Eligibility:\n");
        prompt.push_str("  User: I have 50 grams gold, how much loan?\n");
        prompt.push_str("  [CALL check_eligibility with gold_weight=50]\n");

        prompt
    }

    /// Build with personalization context
    pub fn build_with_context(&self, customer_name: Option<&str>, segment: Option<&str>) -> String {
        let mut prompt = self.build();

        if let Some(name) = customer_name {
            prompt.push_str(&format!("\n## Customer Context\nCustomer name: {}\n", name));
        }

        if let Some(seg) = segment {
            prompt.push_str(&format!("Customer segment: {}\n", seg));
        }

        prompt
    }
}

/// Stage-specific prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagePrompt {
    /// Stage name
    pub stage: String,
    /// Stage objective
    pub objective: String,
    /// Stage-specific instructions
    pub instructions: Vec<String>,
    /// Questions to ask
    #[serde(default)]
    pub discovery_questions: Vec<String>,
    /// Success criteria for moving to next stage
    pub success_criteria: Vec<String>,
}

impl StagePrompt {
    /// Greeting stage
    pub fn greeting() -> Self {
        Self {
            stage: "greeting".to_string(),
            objective: "Establish rapport and understand initial interest".to_string(),
            instructions: vec![
                "Greet warmly with your name".to_string(),
                "Ask how you can help".to_string(),
                "Listen for initial intent".to_string(),
            ],
            discovery_questions: vec![],
            success_criteria: vec![
                "Customer has stated their need".to_string(),
                "Rapport established".to_string(),
            ],
        }
    }

    /// Discovery stage
    pub fn discovery() -> Self {
        Self {
            stage: "discovery".to_string(),
            objective: "Understand customer needs and current situation".to_string(),
            instructions: vec![
                "Ask about their gold loan needs".to_string(),
                "Understand current loan situation if any".to_string(),
                "Gather gold details if possible".to_string(),
                "Identify pain points with current lender".to_string(),
            ],
            discovery_questions: vec![
                "Do you currently have a gold loan with another lender?".to_string(),
                "What is the approximate weight of gold you want to pledge?".to_string(),
                "What loan amount are you looking for?".to_string(),
                "What is your current interest rate?".to_string(),
                "What concerns do you have with your current lender?".to_string(),
            ],
            success_criteria: vec![
                "Know if customer is new or switcher".to_string(),
                "Have approximate gold weight or loan amount".to_string(),
                "Understand primary motivation".to_string(),
            ],
        }
    }

    /// Presentation stage
    pub fn presentation() -> Self {
        Self {
            stage: "presentation".to_string(),
            objective: "Present Kotak gold loan benefits tailored to customer needs".to_string(),
            instructions: vec![
                "Highlight relevant benefits based on customer segment".to_string(),
                "Show savings calculation if switcher".to_string(),
                "Explain simple process".to_string(),
                "Address implicit concerns".to_string(),
            ],
            discovery_questions: vec![],
            success_criteria: vec![
                "Customer understands key benefits".to_string(),
                "Customer shows interest".to_string(),
                "No major objections raised".to_string(),
            ],
        }
    }

    /// Objection handling stage
    pub fn objection_handling() -> Self {
        Self {
            stage: "objection_handling".to_string(),
            objective: "Address customer concerns and objections".to_string(),
            instructions: vec![
                "Acknowledge the concern first".to_string(),
                "Provide factual response".to_string(),
                "Offer proof points when possible".to_string(),
                "Ask follow-up to confirm resolution".to_string(),
            ],
            discovery_questions: vec![
                "Is there anything else that concerns you?".to_string(),
                "What would help you make a decision?".to_string(),
            ],
            success_criteria: vec![
                "Objection addressed".to_string(),
                "Customer seems satisfied with response".to_string(),
            ],
        }
    }

    /// Closing stage
    pub fn closing() -> Self {
        Self {
            stage: "closing".to_string(),
            objective: "Move customer to next action step".to_string(),
            instructions: vec![
                "Summarize key benefits discussed".to_string(),
                "Offer clear next step".to_string(),
                "Capture contact for callback if needed".to_string(),
                "Thank customer for their time".to_string(),
            ],
            discovery_questions: vec![],
            success_criteria: vec![
                "Customer agrees to next step OR".to_string(),
                "Contact captured for follow-up".to_string(),
            ],
        }
    }
}

/// Response templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTemplates {
    /// Rate inquiry response
    pub rate_inquiry: String,
    /// Eligibility response
    pub eligibility: String,
    /// Process explanation
    pub process: String,
    /// Document requirements
    pub documents: String,
    /// Branch locator
    pub branch_locator: String,
    /// Comparison response
    pub comparison: String,
    /// Safety assurance
    pub safety: String,
}

impl Default for ResponseTemplates {
    fn default() -> Self {
        Self {
            rate_inquiry: "Our gold loan interest rates start from 9.5% per annum, which is among the lowest in the market. The exact rate depends on your loan amount - higher amounts get better rates. Would you like me to calculate your potential savings?".to_string(),
            eligibility: "Gold loan eligibility is simple - you need to be between 21-65 years old with valid ID and address proof. We accept gold ornaments of 18K purity and above. The loan amount depends on your gold's weight and purity.".to_string(),
            process: "The process is quick and simple: 1) Visit any Kotak branch with your gold and ID, 2) We value your gold in 15 minutes, 3) Loan approved and disbursed in 30 minutes. That's it!".to_string(),
            documents: "You just need two documents: 1) ID proof like Aadhaar or PAN, 2) Address proof like utility bill or Aadhaar. If you're an existing Kotak customer, even less documentation is needed.".to_string(),
            branch_locator: "We have over 1,600 branches across India. I can help you find the nearest one. Could you share your city or area?".to_string(),
            comparison: "Compared to NBFCs, Kotak offers significantly lower rates (9.5% vs 18-24%), zero foreclosure charges, and RBI-regulated bank security. Would you like me to show how much you could save?".to_string(),
            safety: "Your gold is stored in RBI-regulated bank-grade vaults with 24/7 security and full insurance coverage. You can even track your gold status through our digital platform. It's much safer than NBFC storage.".to_string(),
        }
    }
}

/// Greeting templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreetingTemplates {
    /// Default greeting
    pub default: String,
    /// Morning greeting
    pub morning: String,
    /// Afternoon greeting
    pub afternoon: String,
    /// Evening greeting
    pub evening: String,
    /// Returning customer greeting
    pub returning: String,
    /// Hindi greeting
    pub hindi: String,
}

impl Default for GreetingTemplates {
    fn default() -> Self {
        Self {
            default: "Hello! I'm {agent_name} from Kotak Mahindra Bank. How can I help you with your gold loan needs today?".to_string(),
            morning: "Good morning! I'm {agent_name} from Kotak Mahindra Bank. How can I assist you today?".to_string(),
            afternoon: "Good afternoon! I'm {agent_name} from Kotak Mahindra Bank. How may I help you?".to_string(),
            evening: "Good evening! I'm {agent_name} from Kotak Mahindra Bank. How can I help you today?".to_string(),
            returning: "Welcome back, {customer_name}! It's great to hear from you again. How can I help you today?".to_string(),
            hindi: "Namaste! Main {agent_name} bol rahi hoon Kotak Mahindra Bank se. Main aapki kaise madad kar sakti hoon?".to_string(),
        }
    }
}

impl GreetingTemplates {
    /// Get greeting for time of day
    pub fn for_time(&self, hour: u32) -> &str {
        match hour {
            0..=11 => &self.morning,
            12..=16 => &self.afternoon,
            _ => &self.evening,
        }
    }

    /// Format greeting with variables
    pub fn format(&self, template: &str, agent_name: &str, customer_name: Option<&str>) -> String {
        let mut result = template.replace("{agent_name}", agent_name);
        if let Some(name) = customer_name {
            result = result.replace("{customer_name}", name);
        }
        result
    }
}

/// Closing templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosingTemplates {
    /// Positive close (interested customer)
    pub positive: String,
    /// Neutral close (needs time)
    pub neutral: String,
    /// Callback request
    pub callback: String,
    /// Thank you close
    pub thank_you: String,
    /// Hindi close
    pub hindi: String,
}

impl Default for ClosingTemplates {
    fn default() -> Self {
        Self {
            positive: "Great! To proceed, you can visit our nearest branch with your gold and documents. I can also arrange a callback from our branch to confirm an appointment. Would you like that?".to_string(),
            neutral: "I understand you need time to think. I'll send you a summary on WhatsApp. Feel free to call us when you're ready - we're here to help!".to_string(),
            callback: "Perfect! I've captured your details. Our branch team will call you within 24 hours to schedule a convenient time. Thank you for considering Kotak!".to_string(),
            thank_you: "Thank you for speaking with Kotak Mahindra Bank. If you have any questions, please call us anytime. Have a great day!".to_string(),
            hindi: "Dhanyawad! Kotak Mahindra Bank se baat karne ke liye. Koi bhi sawal ho toh please call kariye. Aapka din shubh ho!".to_string(),
        }
    }
}

/// Fallback templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackTemplates {
    /// Didn't understand
    pub not_understood: String,
    /// Technical issue
    pub technical_issue: String,
    /// Out of scope
    pub out_of_scope: String,
    /// Need more info
    pub need_more_info: String,
    /// Transfer to human
    pub transfer_human: String,
}

impl Default for FallbackTemplates {
    fn default() -> Self {
        Self {
            not_understood: "I'm sorry, I didn't quite catch that. Could you please rephrase your question?".to_string(),
            technical_issue: "I apologize, but I'm having some technical difficulties. Would you like me to arrange a callback from our team?".to_string(),
            out_of_scope: "I specialize in gold loans. For other banking products, I can connect you with the right team. Would you like that?".to_string(),
            need_more_info: "To help you better, could you share a bit more about what you're looking for?".to_string(),
            transfer_human: "Let me connect you with one of our specialists who can help you better. Please hold for a moment.".to_string(),
        }
    }
}

impl PromptTemplates {
    /// Get stage prompt
    pub fn get_stage_prompt(&self, stage: &str) -> Option<&StagePrompt> {
        self.stage_prompts.get(stage)
    }

    /// Build complete system prompt for a conversation
    pub fn build_system_prompt(&self, stage: Option<&str>, customer_name: Option<&str>) -> String {
        let mut prompt = self.system_prompt.build_with_context(customer_name, None);

        if let Some(stage_name) = stage {
            if let Some(stage_prompt) = self.get_stage_prompt(stage_name) {
                prompt.push_str(&format!(
                    "\n## Current Stage: {}\nObjective: {}\n",
                    stage_prompt.stage, stage_prompt.objective
                ));

                prompt.push_str("Instructions for this stage:\n");
                for instruction in &stage_prompt.instructions {
                    prompt.push_str(&format!("- {}\n", instruction));
                }
            }
        }

        prompt
    }

    /// Get appropriate greeting
    pub fn get_greeting(&self, hour: u32, agent_name: &str, customer_name: Option<&str>) -> String {
        let template = self.greetings.for_time(hour);
        self.greetings.format(template, agent_name, customer_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_templates() {
        let templates = PromptTemplates::default();
        assert!(!templates.system_prompt.instructions.is_empty());
        assert!(!templates.stage_prompts.is_empty());
    }

    #[test]
    fn test_system_prompt_build() {
        let templates = PromptTemplates::default();
        let prompt = templates.system_prompt.build();

        assert!(prompt.contains("Priya"));
        assert!(prompt.contains("Kotak Mahindra Bank"));
        assert!(prompt.contains("Instructions"));
        assert!(prompt.contains("Compliance"));
    }

    #[test]
    fn test_stage_prompts() {
        let templates = PromptTemplates::default();

        assert!(templates.get_stage_prompt("greeting").is_some());
        assert!(templates.get_stage_prompt("discovery").is_some());
        assert!(templates.get_stage_prompt("closing").is_some());
    }

    #[test]
    fn test_greeting_for_time() {
        let greetings = GreetingTemplates::default();

        assert!(greetings.for_time(9).contains("morning"));
        assert!(greetings.for_time(14).contains("afternoon"));
        assert!(greetings.for_time(19).contains("evening"));
    }

    #[test]
    fn test_greeting_format() {
        let greetings = GreetingTemplates::default();

        // Test returning template (has customer_name)
        let formatted = greetings.format(&greetings.returning, "Priya", Some("Raj"));
        assert!(formatted.contains("Raj")); // customer_name is replaced

        // Test default template (has agent_name)
        let formatted = greetings.format(&greetings.default, "Priya", None);
        assert!(formatted.contains("Priya")); // agent_name is replaced
    }

    #[test]
    fn test_build_system_prompt() {
        let templates = PromptTemplates::default();
        let prompt = templates.build_system_prompt(Some("discovery"), Some("Raj"));

        assert!(prompt.contains("discovery"));
        assert!(prompt.contains("Raj"));
    }

    #[test]
    fn test_response_templates() {
        let responses = ResponseTemplates::default();
        assert!(responses.rate_inquiry.contains("9.5%"));
        assert!(responses.safety.contains("RBI"));
    }

    // ============================================
    // TOOL INVOCATION RULES TESTS
    // ============================================

    #[test]
    fn test_tool_rules_default() {
        let rules = ToolInvocationRules::default();
        assert!(!rules.rules.is_empty());
        assert!(rules.rules.len() >= 10); // At least 10 rules
    }

    #[test]
    fn test_tool_rules_get_rule() {
        let rules = ToolInvocationRules::default();

        // Test savings_inquiry rule
        let savings_rule = rules.get_rule("savings_inquiry");
        assert!(savings_rule.is_some());
        assert_eq!(savings_rule.unwrap().tool, "calculate_savings");

        // Test balance_transfer rule
        let bt_rule = rules.get_rule("balance_transfer");
        assert!(bt_rule.is_some());
        assert_eq!(bt_rule.unwrap().tool, "calculate_savings");

        // Test eligibility rule
        let elig_rule = rules.get_rule("eligibility_inquiry");
        assert!(elig_rule.is_some());
        assert_eq!(elig_rule.unwrap().tool, "check_eligibility");

        // Test document_inquiry rule
        let doc_rule = rules.get_rule("document_inquiry");
        assert!(doc_rule.is_some());
        assert_eq!(doc_rule.unwrap().tool, "get_document_checklist");
    }

    #[test]
    fn test_tool_rules_required_slots() {
        let rules = ToolInvocationRules::default();

        // calculate_savings needs loan_amount and current_interest_rate
        let savings_rule = rules.get_rule("savings_inquiry").unwrap();
        assert!(savings_rule.required_slots.contains(&"loan_amount".to_string()));
        assert!(savings_rule.required_slots.contains(&"current_interest_rate".to_string()));

        // check_eligibility needs gold_weight
        let elig_rule = rules.get_rule("eligibility_inquiry").unwrap();
        assert!(elig_rule.required_slots.contains(&"gold_weight".to_string()));

        // schedule_appointment needs name and phone
        let appt_rule = rules.get_rule("appointment_request").unwrap();
        assert!(appt_rule.required_slots.contains(&"customer_name".to_string()));
        assert!(appt_rule.required_slots.contains(&"phone_number".to_string()));

        // escalate_to_human needs no slots
        let escalate_rule = rules.get_rule("human_escalation").unwrap();
        assert!(escalate_rule.required_slots.is_empty());
    }

    #[test]
    fn test_tool_rules_all_intents_covered() {
        let rules = ToolInvocationRules::default();

        // All expected intents should have rules
        let expected_intents = [
            "savings_inquiry",
            "balance_transfer",
            "eligibility_inquiry",
            "document_inquiry",
            "branch_inquiry",
            "appointment_request",
            "gold_price_inquiry",
            "comparison_inquiry",
            "callback_request",
            "sms_request",
            "human_escalation",
        ];

        for intent in expected_intents {
            assert!(rules.get_rule(intent).is_some(), "Missing rule for intent: {}", intent);
        }
    }

    #[test]
    fn test_tool_rules_build_prompt_section() {
        let rules = ToolInvocationRules::default();
        let section = rules.build_prompt_section();

        // Should contain tool names
        assert!(section.contains("calculate_savings"));
        assert!(section.contains("check_eligibility"));
        assert!(section.contains("get_document_checklist"));
        assert!(section.contains("find_branches"));
        assert!(section.contains("escalate_to_human"));

        // Should contain intent keywords
        assert!(section.contains("savings"));
        assert!(section.contains("balance transfer"));
        assert!(section.contains("eligibility"));
        assert!(section.contains("document"));
    }

    #[test]
    fn test_system_prompt_contains_tool_rules() {
        let prompt = SystemPrompt::default();
        let built = prompt.build();

        // Should contain tool rules section
        assert!(built.contains("Tool Rules"));
        assert!(built.contains("calculate_savings"));
        assert!(built.contains("check_eligibility"));

        // Should contain examples
        assert!(built.contains("Example"));
        assert!(built.contains("Ayush"));
    }

    #[test]
    fn test_system_prompt_contains_memory_instructions() {
        let prompt = SystemPrompt::default();
        let built = prompt.build();

        // Should emphasize memory/context retention
        assert!(built.contains("CRITICAL") || built.contains("REMEMBER"));
        assert!(built.contains("NEVER") || built.contains("never"));
    }

    #[test]
    fn test_system_prompt_identity() {
        let prompt = SystemPrompt::default();
        let built = prompt.build();

        // Should contain identity handling
        assert!(built.contains("Priya"));
        assert!(built.contains("name"));
    }
}
