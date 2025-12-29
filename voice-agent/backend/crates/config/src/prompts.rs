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
        stage_prompts.insert("objection_handling".to_string(), StagePrompt::objection_handling());
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
            ],
            compliance: vec![
                "Never guarantee specific loan approval".to_string(),
                "Always mention that rates are subject to change".to_string(),
                "Disclose that gold valuation is done at branch".to_string(),
                "Inform about terms and conditions".to_string(),
                "Do not disparage competitors directly".to_string(),
            ],
            guidelines: vec![
                "Be warm and professional".to_string(),
                "Use simple language, avoid jargon".to_string(),
                "Acknowledge customer concerns before responding".to_string(),
                "Keep responses concise (under 60 words)".to_string(),
                "Use Hindi words naturally if customer uses them".to_string(),
            ],
            avoid: vec![
                "Making promises about approval".to_string(),
                "Sharing personal opinions".to_string(),
                "Discussing non-gold loan products unsolicited".to_string(),
                "Being pushy or aggressive".to_string(),
                "Sharing internal policies or processes".to_string(),
            ],
        }
    }
}

impl SystemPrompt {
    /// Build full system prompt text
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
}
