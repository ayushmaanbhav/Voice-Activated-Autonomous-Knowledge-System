//! Prompt Building and Management
//!
//! Constructs prompts for the gold loan voice agent.

use std::fmt;
use serde::{Deserialize, Serialize};

// P0 FIX: Re-export PersonaConfig from config crate (single source of truth)
pub use voice_agent_config::PersonaConfig;

/// Message role
///
/// P2 FIX: Added Tool role for function calling support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool/function response role
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// P4 FIX: Tool definition for LLM-based tool calling
///
/// Describes a tool that the LLM can request to call.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    /// Tool name (must match the tool registry name)
    pub name: String,
    /// Human-readable description of what the tool does
    pub description: String,
    /// Tool parameters
    pub parameters: Vec<ToolParameter>,
}

/// P4 FIX: Tool parameter definition
#[derive(Debug, Clone)]
pub struct ToolParameter {
    /// Parameter name
    pub name: String,
    /// Parameter description
    pub description: String,
    /// Whether this parameter is required
    pub required: bool,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: Vec::new(),
        }
    }

    /// Add a parameter
    pub fn param(mut self, name: impl Into<String>, description: impl Into<String>, required: bool) -> Self {
        self.parameters.push(ToolParameter {
            name: name.into(),
            description: description.into(),
            required,
        });
        self
    }

    /// Create definitions for gold loan tools
    pub fn gold_loan_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition::new("check_eligibility", "Check if customer is eligible for a gold loan based on their gold weight and purity")
                .param("gold_weight", "Weight of gold in grams", true)
                .param("gold_purity", "Purity of gold (e.g., '22K', '18K')", false),
            ToolDefinition::new("calculate_savings", "Calculate monthly savings when switching from competitor")
                .param("current_lender", "Name of current lender (e.g., 'Muthoot', 'Manappuram')", true)
                .param("current_interest_rate", "Current interest rate in percentage", false)
                .param("current_loan_amount", "Current loan amount in INR", false)
                .param("remaining_tenure_months", "Remaining tenure in months", false),
            ToolDefinition::new("find_branches", "Find nearby Kotak branches")
                .param("city", "City name to search branches in", true)
                .param("area", "Specific area or locality (optional)", false),
            ToolDefinition::new("schedule_callback", "Schedule a callback from branch team")
                .param("phone", "Customer phone number", true)
                .param("preferred_time", "Preferred callback time (e.g., 'morning', 'afternoon')", false),
        ]
    }
}

/// P4 FIX: Parse tool call from LLM response
///
/// Extracts tool calls in the format: `[TOOL_CALL: {"name": "...", "arguments": {...}}]`
pub fn parse_tool_call(response: &str) -> Option<ParsedToolCall> {
    // Look for the tool call pattern
    let start_marker = "[TOOL_CALL:";
    let end_marker = "]";

    let start_idx = response.find(start_marker)?;
    let json_start = start_idx + start_marker.len();

    // Find matching end bracket
    let remaining = &response[json_start..];
    let end_idx = remaining.find(end_marker)?;
    let json_str = remaining[..end_idx].trim();

    // Parse the JSON
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value.get("arguments").cloned().unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    // Extract the text before and after the tool call
    let text_before = response[..start_idx].trim().to_string();
    let text_after = response[json_start + end_idx + 1..].trim().to_string();

    Some(ParsedToolCall {
        name,
        arguments,
        text_before,
        text_after,
    })
}

/// P4 FIX: Parsed tool call from LLM response
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    /// Tool name to call
    pub name: String,
    /// Arguments as JSON value
    pub arguments: serde_json::Value,
    /// Text before the tool call (can be used as partial response)
    pub text_before: String,
    /// Text after the tool call (continuation)
    pub text_after: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }

    /// P2 FIX: Added tool() constructor for function calling responses.
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
        }
    }
}

/// Prompt builder for gold loan agent
pub struct PromptBuilder {
    messages: Vec<Message>,
    persona: PersonaConfig,
}

// PersonaConfig is now imported from voice_agent_config (see re-export above)

impl PromptBuilder {
    /// Create a new prompt builder
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            persona: PersonaConfig::default(),
        }
    }

    /// Set persona configuration
    pub fn with_persona(mut self, persona: PersonaConfig) -> Self {
        self.persona = persona;
        self
    }

    /// Build system prompt for gold loan agent
    pub fn system_prompt(mut self, language: &str) -> Self {
        let persona_traits = self.build_persona_traits();

        let system = format!(
            r#"You are {name}, a friendly and knowledgeable Gold Loan specialist at Kotak Mahindra Bank.

## Your Persona
{traits}

## Your Role
- Help customers understand gold loan products and benefits
- Guide customers through the Switch & Save program
- Answer questions about interest rates, LTV, and documentation
- Address concerns and objections with empathy
- Collect lead information when appropriate

## Communication Guidelines
- Speak naturally and conversationally
- Use {language_style} language naturally
- Keep responses concise (2-3 sentences max for voice)
- Ask one question at a time
- Acknowledge customer concerns before addressing them
- Use the customer's name when known

## Key Product Information
- Interest rates: Starting from 10.5% (vs 18-24% NBFC rates)
- LTV: Up to 75% of gold value
- Processing: Same-day disbursement
- Safety: RBI-regulated bank with insured vault storage
- Bridge loan available for seamless transfer

## Response Format
Respond naturally as if speaking on a phone call. Do not use bullet points, headers, or markdown formatting. Keep responses brief and conversational."#,
            name = self.persona.name,
            traits = persona_traits,
            language_style = if language == "hi" { "Hindi-English (Hinglish)" } else { "English" },
        );

        self.messages.push(Message::system(system));
        self
    }

    /// Build persona traits string
    fn build_persona_traits(&self) -> String {
        let mut traits = Vec::new();

        if self.persona.warmth > 0.7 {
            traits.push("- Warm and welcoming in your tone");
        }
        if self.persona.empathy > 0.8 {
            traits.push("- Highly empathetic, understanding customer concerns");
        }
        if self.persona.formality < 0.4 {
            traits.push("- Casual and friendly, like talking to a trusted friend");
        } else if self.persona.formality > 0.7 {
            traits.push("- Professional and respectful");
        } else {
            traits.push("- Balanced between professional and approachable");
        }
        if self.persona.urgency > 0.6 {
            traits.push("- Gently convey time-sensitive opportunities");
        }

        traits.join("\n")
    }

    /// Add RAG context
    pub fn with_context(mut self, context: &str) -> Self {
        if !context.is_empty() {
            let context_msg = format!(
                "## Relevant Information\n{}\n\nUse this information to answer the customer's question if relevant.",
                context
            );
            self.messages.push(Message::system(context_msg));
        }
        self
    }

    /// Add customer profile
    pub fn with_customer(mut self, name: Option<&str>, segment: Option<&str>, history: Option<&str>) -> Self {
        let mut profile_parts = Vec::new();

        if let Some(n) = name {
            profile_parts.push(format!("Customer name: {}", n));
        }
        if let Some(s) = segment {
            profile_parts.push(format!("Segment: {}", s));
        }
        if let Some(h) = history {
            profile_parts.push(format!("History: {}", h));
        }

        if !profile_parts.is_empty() {
            let profile = format!("## Customer Profile\n{}", profile_parts.join("\n"));
            self.messages.push(Message::system(profile));
        }
        self
    }

    /// Add conversation history
    pub fn with_history(mut self, history: &[Message]) -> Self {
        self.messages.extend(history.iter().cloned());
        self
    }

    /// Add current user message
    pub fn user_message(mut self, message: &str) -> Self {
        self.messages.push(Message::user(message));
        self
    }

    /// Add stage guidance
    pub fn with_stage_guidance(mut self, stage: &str) -> Self {
        let guidance = match stage {
            "greeting" => "Warmly greet the customer and introduce yourself. Build rapport before discussing products.",
            "discovery" => "Ask open questions to understand their gold loan needs and current situation with competitors.",
            "qualification" => "Assess their eligibility and readiness to switch. Understand loan amount and timeline.",
            "presentation" => "Present Kotak's gold loan benefits, focusing on their specific needs and concerns.",
            "objection_handling" => "Address concerns with empathy. Use social proof and guarantees to build confidence.",
            "closing" => "Summarize benefits and guide them to next steps. Create urgency if appropriate.",
            "farewell" => "Thank them warmly and confirm next steps. Leave the door open for future conversations.",
            _ => "",
        };

        if !guidance.is_empty() {
            self.messages.push(Message::system(format!("## Current Stage Guidance\n{}", guidance)));
        }
        self
    }

    /// P4 FIX: Add available tools for LLM-based tool calling
    ///
    /// For Ollama (non-OpenAI models), we inject tool definitions into the system prompt
    /// and instruct the LLM to output tool calls in a specific JSON format.
    ///
    /// The LLM should output: `[TOOL_CALL: {"name": "tool_name", "arguments": {...}}]`
    /// when it determines a tool should be called.
    pub fn with_tools(mut self, tools: &[ToolDefinition]) -> Self {
        if tools.is_empty() {
            return self;
        }

        let mut tool_prompt = String::from(
            r#"## Available Tools

You have access to the following tools. When you need to use a tool to help the customer, output a tool call in this EXACT format:

[TOOL_CALL: {"name": "tool_name", "arguments": {"param1": "value1"}}]

After the tool runs, you will receive the result to incorporate into your response.

Available tools:
"#
        );

        for tool in tools {
            tool_prompt.push_str(&format!(
                "\n### {}\n{}\nParameters:\n",
                tool.name, tool.description
            ));
            for param in &tool.parameters {
                let required = if param.required { " (required)" } else { "" };
                tool_prompt.push_str(&format!(
                    "- {}: {}{}\n",
                    param.name, param.description, required
                ));
            }
        }

        tool_prompt.push_str(
            "\nOnly use tools when the customer's request requires specific calculations or data lookup. For general conversation, respond naturally without tools."
        );

        self.messages.push(Message::system(tool_prompt));
        self
    }

    /// Build final message list
    pub fn build(self) -> Vec<Message> {
        self.messages
    }

    /// Build with context window limit
    ///
    /// P0 FIX: Truncates conversation history to fit within token limit.
    /// Preserves system prompt and most recent messages, removing oldest
    /// non-system messages first.
    pub fn build_with_limit(self, max_tokens: usize) -> Vec<Message> {
        let current_tokens = self.estimate_tokens();

        if current_tokens <= max_tokens {
            return self.messages;
        }

        // Separate system messages (keep all) from conversation history
        let (system_msgs, conv_msgs): (Vec<_>, Vec<_>) = self.messages
            .into_iter()
            .partition(|m| matches!(m.role, Role::System));

        let system_tokens: usize = system_msgs.iter()
            .map(|m| Self::estimate_single_message_tokens(&m.content))
            .sum();

        let available_tokens = max_tokens.saturating_sub(system_tokens);

        // Keep recent messages that fit within limit
        let mut kept_msgs: Vec<Message> = Vec::new();
        let mut used_tokens = 0;

        for msg in conv_msgs.into_iter().rev() {
            let msg_tokens = Self::estimate_single_message_tokens(&msg.content);
            if used_tokens + msg_tokens <= available_tokens {
                kept_msgs.push(msg);
                used_tokens += msg_tokens;
            } else {
                break;
            }
        }

        kept_msgs.reverse();

        // Combine: system messages first, then kept conversation
        let mut result = system_msgs;
        result.extend(kept_msgs);

        tracing::debug!(
            "Context truncated: {} -> {} tokens ({} messages kept)",
            current_tokens,
            system_tokens + used_tokens,
            result.len()
        );

        result
    }

    /// Estimate tokens for a single message content
    fn estimate_single_message_tokens(content: &str) -> usize {
        use unicode_segmentation::UnicodeSegmentation;

        let grapheme_count = content.graphemes(true).count();
        let devanagari_count = content.chars()
            .filter(|c| ('\u{0900}'..='\u{097F}').contains(c))
            .count();

        if devanagari_count > grapheme_count / 3 {
            grapheme_count.max(1) / 2
        } else {
            grapheme_count.max(1) / 4
        }
    }

    /// Get message count
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Estimate token count
    ///
    /// P0 FIX: Improved estimation for Hindi/Devanagari text
    pub fn estimate_tokens(&self) -> usize {
        use unicode_segmentation::UnicodeSegmentation;

        self.messages
            .iter()
            .map(|m| {
                let grapheme_count = m.content.graphemes(true).count();
                let devanagari_count = m.content.chars()
                    .filter(|c| ('\u{0900}'..='\u{097F}').contains(c))
                    .count();

                if devanagari_count > grapheme_count / 3 {
                    grapheme_count.max(1) / 2
                } else {
                    grapheme_count.max(1) / 4
                }
            })
            .sum()
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick response templates
pub struct ResponseTemplates;

impl ResponseTemplates {
    /// Greeting template
    pub fn greeting(name: &str, language: &str) -> String {
        if language == "hi" {
            format!("Namaste! Main {} hoon, Kotak Mahindra Bank se. Aapki madad karne ke liye yahan hoon.", name)
        } else {
            format!("Hello! I'm {} from Kotak Mahindra Bank. I'm here to help you today.", name)
        }
    }

    /// Acknowledgment
    pub fn acknowledge(language: &str) -> String {
        if language == "hi" {
            "Ji bilkul, main samajh sakti hoon.".to_string()
        } else {
            "I understand, that makes sense.".to_string()
        }
    }

    /// Clarification request
    pub fn clarify(language: &str) -> String {
        if language == "hi" {
            "Kya aap thoda aur bata sakte hain?".to_string()
        } else {
            "Could you tell me a bit more about that?".to_string()
        }
    }

    /// Transition to next topic
    pub fn transition(language: &str) -> String {
        if language == "hi" {
            "Achha, ab main aapko batati hoon...".to_string()
        } else {
            "Great, let me tell you about...".to_string()
        }
    }

    /// Closing
    pub fn closing(language: &str) -> String {
        if language == "hi" {
            "Dhanyavaad aapka samay dene ke liye. Koi bhi sawal ho toh zaroor call karein.".to_string()
        } else {
            "Thank you for your time. Please feel free to call if you have any questions.".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_prompt_builder() {
        let messages = PromptBuilder::new()
            .system_prompt("en")
            .user_message("What is your interest rate?")
            .build();

        assert!(messages.len() >= 2);
        assert_eq!(messages[0].role, Role::System);
    }

    #[test]
    fn test_with_context() {
        let messages = PromptBuilder::new()
            .system_prompt("en")
            .with_context("Interest rate is 10.5%")
            .user_message("Tell me about rates")
            .build();

        // Should have system prompt, context, and user message
        assert!(messages.len() >= 3);
    }

    #[test]
    fn test_templates() {
        let greeting = ResponseTemplates::greeting("Priya", "hi");
        assert!(greeting.contains("Namaste"));

        let greeting_en = ResponseTemplates::greeting("Priya", "en");
        assert!(greeting_en.contains("Hello"));
    }

    // P4 FIX: Tool calling tests

    #[test]
    fn test_tool_definition_builder() {
        let tool = ToolDefinition::new("test_tool", "A test tool")
            .param("param1", "First parameter", true)
            .param("param2", "Second parameter", false);

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.parameters.len(), 2);
        assert!(tool.parameters[0].required);
        assert!(!tool.parameters[1].required);
    }

    #[test]
    fn test_gold_loan_tools() {
        let tools = ToolDefinition::gold_loan_tools();

        assert!(tools.len() >= 3);
        assert!(tools.iter().any(|t| t.name == "check_eligibility"));
        assert!(tools.iter().any(|t| t.name == "calculate_savings"));
        assert!(tools.iter().any(|t| t.name == "find_branches"));
    }

    #[test]
    fn test_parse_tool_call_simple() {
        let response = r#"Let me check that for you. [TOOL_CALL: {"name": "check_eligibility", "arguments": {"gold_weight": 50}}]"#;

        let parsed = parse_tool_call(response).expect("Should parse tool call");
        assert_eq!(parsed.name, "check_eligibility");
        assert_eq!(parsed.arguments["gold_weight"], 50);
        assert_eq!(parsed.text_before, "Let me check that for you.");
    }

    #[test]
    fn test_parse_tool_call_no_tool() {
        let response = "Hello! How can I help you today?";
        assert!(parse_tool_call(response).is_none());
    }

    #[test]
    fn test_parse_tool_call_with_text_after() {
        let response = r#"[TOOL_CALL: {"name": "find_branches", "arguments": {"city": "Mumbai"}}] I'll wait for the results."#;

        let parsed = parse_tool_call(response).expect("Should parse tool call");
        assert_eq!(parsed.name, "find_branches");
        assert_eq!(parsed.arguments["city"], "Mumbai");
        assert!(parsed.text_before.is_empty());
        assert_eq!(parsed.text_after, "I'll wait for the results.");
    }

    #[test]
    fn test_with_tools() {
        let tools = ToolDefinition::gold_loan_tools();
        let messages = PromptBuilder::new()
            .system_prompt("en")
            .with_tools(&tools)
            .user_message("Can you check my eligibility?")
            .build();

        // Should have system prompt, tool definitions, and user message
        assert!(messages.len() >= 3);

        // Tool definitions should be in the prompt
        let tool_msg = messages.iter()
            .find(|m| m.content.contains("TOOL_CALL"))
            .expect("Should have tool definitions");
        assert!(tool_msg.content.contains("check_eligibility"));
        assert!(tool_msg.content.contains("calculate_savings"));
    }
}
