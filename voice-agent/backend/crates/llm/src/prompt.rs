//! Prompt Building and Management
//!
//! Constructs prompts for domain-agnostic voice agents.
//! Domain-specific content should be loaded from config.


// P0 FIX: Re-export PersonaConfig from config crate (single source of truth)
pub use voice_agent_config::PersonaConfig;

// P1-1 FIX: Re-export Message and Role from core crate (single source of truth)
// This eliminates duplicate type definitions and ensures consistency
pub use voice_agent_core::llm_types::{Message, Role};

// P0-2 FIX: Use canonical ToolDefinition from core crate (JSON Schema format)
// This ensures compatibility with Claude API native tool_use and other providers
pub use voice_agent_core::llm_types::ToolDefinition;

/// P0-2 FIX: Builder for creating ToolDefinition with JSON Schema parameters
///
/// Makes it ergonomic to build tool definitions with proper JSON Schema format
/// that is compatible with both Claude API (native tool_use) and Ollama (text injection).
///
/// # Example
/// ```ignore
/// let tool = ToolBuilder::new("check_eligibility", "Check loan eligibility")
///     .param("gold_weight", "number", "Weight of gold in grams", true)
///     .param("gold_purity", "string", "Purity (22K, 18K)", false)
///     .string_enum("gold_purity", &["24K", "22K", "18K", "14K"])
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct ToolBuilder {
    name: String,
    description: String,
    properties: serde_json::Map<String, serde_json::Value>,
    required: Vec<String>,
}

impl ToolBuilder {
    /// Create a new tool builder
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            properties: serde_json::Map::new(),
            required: Vec::new(),
        }
    }

    /// Add a parameter with type and description
    pub fn param(
        mut self,
        name: impl Into<String>,
        param_type: &str,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        let name = name.into();
        let mut prop = serde_json::Map::new();
        prop.insert(
            "type".to_string(),
            serde_json::Value::String(param_type.to_string()),
        );
        prop.insert(
            "description".to_string(),
            serde_json::Value::String(description.into()),
        );

        self.properties
            .insert(name.clone(), serde_json::Value::Object(prop));

        if required {
            self.required.push(name);
        }
        self
    }

    /// Add enum constraint to an existing string parameter
    pub fn string_enum(mut self, name: &str, values: &[&str]) -> Self {
        if let Some(prop) = self.properties.get_mut(name) {
            if let Some(obj) = prop.as_object_mut() {
                let enum_values: Vec<serde_json::Value> = values
                    .iter()
                    .map(|v| serde_json::Value::String(v.to_string()))
                    .collect();
                obj.insert("enum".to_string(), serde_json::Value::Array(enum_values));
            }
        }
        self
    }

    /// Add number range constraint
    pub fn number_range(mut self, name: &str, min: Option<f64>, max: Option<f64>) -> Self {
        if let Some(prop) = self.properties.get_mut(name) {
            if let Some(obj) = prop.as_object_mut() {
                if let Some(min_val) = min {
                    obj.insert("minimum".to_string(), serde_json::json!(min_val));
                }
                if let Some(max_val) = max {
                    obj.insert("maximum".to_string(), serde_json::json!(max_val));
                }
            }
        }
        self
    }

    /// Build the ToolDefinition
    pub fn build(self) -> ToolDefinition {
        let parameters = serde_json::json!({
            "type": "object",
            "properties": self.properties,
            "required": self.required,
        });

        ToolDefinition::new(self.name, self.description, parameters)
    }
}

// P16 FIX: gold_loan_tools() REMOVED - tools are now loaded from domain config
// Use voice_agent_config::domain::ToolsConfig::to_tool_definitions() instead
//
// Example usage:
//   let tools_config = master_domain_config.tools.clone();
//   let tools = tools_config.to_tool_definitions();
//   prompt_builder.with_tools(&tools);

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
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

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

// P1-1 FIX: Message constructors are inherited from core crate
// Use Message::system(), Message::user(), Message::assistant(), Message::tool()
// directly from voice_agent_core::llm_types (re-exported above)

/// P13 FIX: Key product facts for system prompt (config-driven)
#[derive(Debug, Clone)]
pub struct ProductFacts {
    /// Our best interest rate (e.g., 9.5)
    pub our_rate: f64,
    /// Typical NBFC rate range (e.g., 18.0)
    pub nbfc_rate_low: f64,
    /// Higher end of NBFC range (e.g., 24.0)
    pub nbfc_rate_high: f64,
    /// LTV percentage (e.g., 75.0)
    pub ltv_percent: f64,
}

impl Default for ProductFacts {
    fn default() -> Self {
        Self {
            our_rate: 10.5,
            nbfc_rate_low: 18.0,
            nbfc_rate_high: 24.0,
            ltv_percent: 75.0,
        }
    }
}

/// Prompt builder for voice agent (domain-agnostic)
pub struct PromptBuilder {
    messages: Vec<Message>,
    persona: PersonaConfig,
    /// P13 FIX: Config-driven product facts
    product_facts: ProductFacts,
}

/// P16 FIX: Brand configuration for config-driven prompts
/// Renamed bank_name to company_name for domain-agnostic design.
#[derive(Debug, Clone, Default)]
pub struct BrandConfig {
    pub agent_name: String,
    pub company_name: String,
    pub agent_role: String,
    pub helpline: String,
}

// PersonaConfig is now imported from voice_agent_config (see re-export above)

impl PromptBuilder {
    /// Create a new prompt builder
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            persona: PersonaConfig::default(),
            product_facts: ProductFacts::default(),
        }
    }

    /// P13 FIX: Set product facts from config
    pub fn with_product_facts(mut self, facts: ProductFacts) -> Self {
        self.product_facts = facts;
        self
    }

    /// Set persona configuration
    pub fn with_persona(mut self, persona: PersonaConfig) -> Self {
        self.persona = persona;
        self
    }

    /// P16 FIX: Build system prompt from config (domain-agnostic)
    ///
    /// Uses PromptsConfig templates instead of hardcoded text.
    /// This is the preferred method for domain-agnostic operation.
    pub fn system_prompt_from_config(
        mut self,
        prompts_config: &voice_agent_config::domain::PromptsConfig,
        brand: &BrandConfig,
        language: &str,
    ) -> Self {
        let persona_traits = prompts_config.build_persona_traits(
            self.persona.warmth,
            self.persona.empathy,
            self.persona.formality,
            self.persona.urgency,
        );

        // Build key facts from product_facts
        // NOTE: For domain-specific key facts (e.g., regulatory info, security features),
        // define them in prompts/system.yaml and use prompts_config.key_facts()
        let key_facts = format!(
            "- Interest rates: Starting from {:.1}% p.a. (vs {:.0}-{:.0}% competitor rates)\n\
             - LTV: Up to {:.0}% of collateral value\n\
             - Processing: Same-day disbursement\n\
             - Regulated financial institution with secure storage",
            self.product_facts.our_rate,
            self.product_facts.nbfc_rate_low,
            self.product_facts.nbfc_rate_high,
            self.product_facts.ltv_percent,
        );

        let system = prompts_config.build_system_prompt(
            &brand.agent_name,
            &brand.company_name,
            &persona_traits,
            language,
            &key_facts,
            &brand.helpline,
        );

        self.messages.push(Message::system(system));
        self
    }

    /// Legacy system_prompt method (deprecated)
    ///
    /// P16 COMPAT: Provides backwards compatibility during migration.
    /// Use system_prompt_from_config() with PromptsConfig for domain-agnostic operation.
    ///
    /// Migration guide:
    ///   let prompts_config = master_domain_config.prompts.clone();
    ///   let brand = BrandConfig { agent_name, company_name, agent_role, helpline };
    ///   prompt_builder.system_prompt_from_config(&prompts_config, &brand, language)
    #[deprecated(note = "Use system_prompt_from_config() for domain-agnostic operation")]
    pub fn system_prompt(mut self, language: &str) -> Self {
        let persona_traits = self.build_persona_traits();

        // Default brand config for backwards compatibility
        let agent_name = "Priya";
        let company_name = "Kotak";

        // Build key facts from product_facts
        let key_facts = format!(
            "- Interest rates: Starting from {:.1}% p.a. (vs {:.0}-{:.0}% competitor rates)\n\
             - LTV: Up to {:.0}% of collateral value\n\
             - Processing: Same-day disbursement\n\
             - Regulated financial institution with secure storage",
            self.product_facts.our_rate,
            self.product_facts.nbfc_rate_low,
            self.product_facts.nbfc_rate_high,
            self.product_facts.ltv_percent,
        );

        let system_content = format!(
            "You are {agent_name}, a helpful voice assistant for {company_name} gold loan services.\n\n\
             ## Your Personality\n{traits}\n\n\
             ## Key Facts\n{facts}\n\n\
             ## Guidelines\n\
             - Be conversational yet professional\n\
             - Keep responses concise (1-2 sentences for voice)\n\
             - Language: {language}\n\
             - Focus on customer needs\n\
             - Never fabricate information",
            agent_name = agent_name,
            company_name = company_name,
            traits = persona_traits,
            facts = key_facts,
            language = language,
        );

        self.messages.insert(0, Message::system(system_content));
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
    pub fn with_customer(
        mut self,
        name: Option<&str>,
        segment: Option<&str>,
        history: Option<&str>,
    ) -> Self {
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

    /// P16 FIX: Add stage guidance from config (domain-agnostic)
    pub fn with_stage_guidance_from_config(
        mut self,
        stage: &str,
        prompts_config: &voice_agent_config::domain::PromptsConfig,
    ) -> Self {
        if let Some(guidance) = prompts_config.get_stage_guidance(stage) {
            let wrapper = prompts_config.build_stage_guidance(guidance);
            if !wrapper.is_empty() {
                self.messages.push(Message::system(wrapper));
            } else {
                self.messages.push(Message::system(format!(
                    "## Current Stage Guidance\n{}",
                    guidance
                )));
            }
        }
        self
    }

    /// Legacy with_stage_guidance method (deprecated)
    ///
    /// P16 COMPAT: Provides backwards compatibility during migration.
    /// Use with_stage_guidance_from_config() with PromptsConfig for domain-agnostic operation.
    ///
    /// Migration guide:
    ///   let prompts_config = master_domain_config.prompts.clone();
    ///   prompt_builder.with_stage_guidance_from_config(stage, &prompts_config)
    #[deprecated(note = "Use with_stage_guidance_from_config() for domain-agnostic operation")]
    pub fn with_stage_guidance(mut self, stage: &str) -> Self {
        // Default stage guidance for backwards compatibility
        let guidance = match stage {
            "greeting" => "Focus on warm welcome and understanding the customer's needs.",
            "qualification" => "Gather information about collateral and loan requirements.",
            "presentation" => "Present relevant loan options and benefits.",
            "objection_handling" => "Address concerns empathetically while highlighting advantages.",
            "closing" => "Guide toward next steps and branch visit.",
            _ => "Be helpful and professional.",
        };

        self.messages.push(Message::system(format!(
            "## Current Stage: {}\n{}",
            stage, guidance
        )));
        self
    }

    /// P0-2 FIX: Add available tools for LLM-based tool calling
    ///
    /// For Ollama (non-native tool support), we inject tool definitions into the system prompt
    /// and instruct the LLM to output tool calls in a specific JSON format.
    ///
    /// For Claude API, tools are passed natively via the API (handled by Claude backend).
    /// This method is primarily used for Ollama text-based tool injection.
    ///
    /// The LLM should output: `[TOOL_CALL: {"name": "tool_name", "arguments": {...}}]`
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
"#,
        );

        for tool in tools {
            tool_prompt.push_str(&format!("\n### {}\n{}\n", tool.name, tool.description));

            // Extract parameters from JSON schema
            if let Some(props) = tool
                .parameters
                .get("properties")
                .and_then(|p| p.as_object())
            {
                let required_params: Vec<&str> = tool
                    .parameters
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                    .unwrap_or_default();

                tool_prompt.push_str("Parameters:\n");
                for (param_name, param_schema) in props {
                    let description = param_schema
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    let param_type = param_schema
                        .get("type")
                        .and_then(|t| t.as_str())
                        .unwrap_or("any");
                    let required = if required_params.contains(&param_name.as_str()) {
                        " (required)"
                    } else {
                        ""
                    };

                    // Include enum values if present
                    let enum_hint = param_schema
                        .get("enum")
                        .and_then(|e| e.as_array())
                        .map(|arr| {
                            let values: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                            format!(" [{}]", values.join(", "))
                        })
                        .unwrap_or_default();

                    tool_prompt.push_str(&format!(
                        "- {} ({}): {}{}{}\n",
                        param_name, param_type, description, required, enum_hint
                    ));
                }
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

    /// P1 FIX: Build as GenerateRequest for core::LanguageModel trait
    ///
    /// Converts the prompt builder's messages to a GenerateRequest that
    /// can be used with the LanguageModel trait from voice_agent_core.
    pub fn build_request(self) -> voice_agent_core::GenerateRequest {
        let core_messages: Vec<voice_agent_core::llm_types::Message> = self
            .messages
            .into_iter()
            .map(Self::convert_message_to_core)
            .collect();

        voice_agent_core::GenerateRequest {
            messages: core_messages,
            ..Default::default()
        }
    }

    /// P1 FIX: Build as GenerateRequest with context window limit
    ///
    /// Combines token-limited message building with GenerateRequest conversion.
    pub fn build_request_with_limit(self, max_tokens: usize) -> voice_agent_core::GenerateRequest {
        let messages = self.build_with_limit_internal(max_tokens);
        let core_messages: Vec<voice_agent_core::llm_types::Message> = messages
            .into_iter()
            .map(Self::convert_message_to_core)
            .collect();

        voice_agent_core::GenerateRequest {
            messages: core_messages,
            ..Default::default()
        }
    }

    /// Convert llm crate Message to core crate Message
    fn convert_message_to_core(m: Message) -> voice_agent_core::llm_types::Message {
        let role = match m.role {
            Role::System => voice_agent_core::llm_types::Role::System,
            Role::User => voice_agent_core::llm_types::Role::User,
            Role::Assistant => voice_agent_core::llm_types::Role::Assistant,
            Role::Tool => voice_agent_core::llm_types::Role::Tool,
        };
        voice_agent_core::llm_types::Message {
            role,
            content: m.content,
            name: None,
            tool_call_id: None,
        }
    }

    /// Internal helper for build_with_limit (also used by build_request_with_limit)
    fn build_with_limit_internal(self, max_tokens: usize) -> Vec<Message> {
        let current_tokens = self.estimate_tokens();

        if current_tokens <= max_tokens {
            return self.messages;
        }

        // Separate system messages (keep all) from conversation history
        let (system_msgs, conv_msgs): (Vec<_>, Vec<_>) = self
            .messages
            .into_iter()
            .partition(|m| matches!(m.role, Role::System));

        let system_tokens: usize = system_msgs
            .iter()
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

    /// Build with context window limit
    ///
    /// P0 FIX: Truncates conversation history to fit within token limit.
    /// Preserves system prompt and most recent messages, removing oldest
    /// non-system messages first.
    pub fn build_with_limit(self, max_tokens: usize) -> Vec<Message> {
        self.build_with_limit_internal(max_tokens)
    }

    /// Estimate tokens for a single message content
    fn estimate_single_message_tokens(content: &str) -> usize {
        use unicode_segmentation::UnicodeSegmentation;

        let grapheme_count = content.graphemes(true).count();
        let devanagari_count = content
            .chars()
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
                let devanagari_count = m
                    .content
                    .chars()
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
///
/// P10 FIX: These templates provide fallback values when LlmDomainView is not available.
/// P16 FIX: Added config-driven methods - prefer using these for domain-agnostic operation.
pub struct ResponseTemplates;

impl ResponseTemplates {
    /// P16 FIX: Greeting from PromptsConfig (domain-agnostic)
    /// Renamed bank_name to company_name for domain-agnostic design.
    pub fn greeting_from_prompts_config(
        prompts_config: &voice_agent_config::domain::PromptsConfig,
        agent_name: &str,
        company_name: &str,
        language: &str,
    ) -> String {
        let template = prompts_config.get_greeting(language);
        template
            .replace("{agent_name}", agent_name)
            .replace("{company_name}", company_name)
            .replace("{bank_name}", company_name) // Legacy support
    }

    /// P16 FIX: Farewell from PromptsConfig (domain-agnostic)
    pub fn farewell_from_prompts_config(
        prompts_config: &voice_agent_config::domain::PromptsConfig,
        helpline: &str,
        language: &str,
    ) -> String {
        let template = prompts_config.get_farewell(language);
        template.replace("{helpline}", helpline)
    }

    // P0 FIX: greeting() REMOVED
    // This deprecated method contained hardcoded greeting text.
    // Use greeting_from_prompts_config() or greeting_from_view() for domain-agnostic operation.
    //
    // Migration guide:
    //   let prompts_config = master_domain_config.prompts.clone();
    //   ResponseTemplates::greeting_from_prompts_config(&prompts_config, agent_name, company_name, language)

    /// P10 FIX: Greeting using domain view configuration
    ///
    /// Uses the hierarchical domain config to get localized greetings.
    /// Falls back to hardcoded values if the config doesn't have a greeting for the language.
    pub fn greeting_from_view(
        view: &voice_agent_config::LlmDomainView,
        name: &str,
        language: &str,
    ) -> String {
        let base_greeting = view.get_greeting(language);
        // The view returns a template-like greeting, personalize with name
        if base_greeting.contains("{name}") {
            base_greeting.replace("{name}", name)
        } else if base_greeting.contains("I'm") || base_greeting.contains("Main") {
            // Already has a name placeholder pattern, return as-is since it's from config
            base_greeting
        } else {
            // Append name if greeting doesn't mention agent name
            format!("{} I'm {}.", base_greeting, name)
        }
    }

    /// P10 FIX: Time-of-day aware greeting using domain view
    pub fn greeting_with_time_from_view(
        view: &voice_agent_config::LlmDomainView,
        name: &str,
        language: &str,
        hour: u32,
    ) -> String {
        let base_greeting = view.get_greeting_with_time(language, hour);
        if base_greeting.contains("{name}") {
            base_greeting.replace("{name}", name)
        } else {
            format!("{} I'm {}.", base_greeting, name)
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

    // P0 FIX: closing() REMOVED
    // This deprecated method contained hardcoded farewell text.
    // Use farewell_from_prompts_config() or closing_from_view() for domain-agnostic operation.
    //
    // Migration guide:
    //   let prompts_config = master_domain_config.prompts.clone();
    //   ResponseTemplates::farewell_from_prompts_config(&prompts_config, helpline, language)

    /// P10 FIX: Closing/farewell using domain view configuration
    ///
    /// Uses the hierarchical domain config to get localized farewells.
    pub fn closing_from_view(
        view: &voice_agent_config::LlmDomainView,
        language: &str,
    ) -> String {
        view.get_farewell(language)
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
        // P0 FIX: Test with user messages only (system_prompt removed)
        let messages = PromptBuilder::new()
            .user_message("What is your interest rate?")
            .build();

        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, Role::User);
    }

    #[test]
    fn test_with_context() {
        // P0 FIX: Test context without deprecated system_prompt
        let messages = PromptBuilder::new()
            .with_context("Interest rate is 10.5%")
            .user_message("Tell me about rates")
            .build();

        // Should have context and user message
        assert!(messages.len() >= 2);
        // First message should be system (context)
        assert_eq!(messages[0].role, Role::System);
    }

    #[test]
    fn test_templates() {
        // P0 FIX: Test non-deprecated response templates
        let ack = ResponseTemplates::acknowledge("hi");
        assert!(ack.contains("samajh"));

        let ack_en = ResponseTemplates::acknowledge("en");
        assert!(ack_en.contains("understand"));

        let clarify = ResponseTemplates::clarify("en");
        assert!(clarify.contains("more"));
    }

    // P0-2 FIX: Tool calling tests updated for JSON Schema format

    #[test]
    fn test_tool_builder() {
        let tool = ToolBuilder::new("test_tool", "A test tool")
            .param("param1", "string", "First parameter", true)
            .param("param2", "number", "Second parameter", false)
            .build();

        assert_eq!(tool.name, "test_tool");
        assert_eq!(tool.description, "A test tool");

        // Verify JSON schema structure
        let props = tool
            .parameters
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(props.contains_key("param1"));
        assert!(props.contains_key("param2"));

        let required = tool.parameters.get("required").unwrap().as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].as_str().unwrap(), "param1");
    }

    #[test]
    fn test_tool_builder_with_enum() {
        let tool = ToolBuilder::new("test_tool", "A test tool")
            .param("status", "string", "Status field", true)
            .string_enum("status", &["active", "inactive", "pending"])
            .build();

        let props = tool
            .parameters
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        let status_schema = props.get("status").unwrap();
        let enum_values = status_schema.get("enum").unwrap().as_array().unwrap();

        assert_eq!(enum_values.len(), 3);
        assert!(enum_values.contains(&serde_json::json!("active")));
    }

    #[test]
    fn test_tool_builder_with_number_range() {
        let tool = ToolBuilder::new("test_tool", "A test tool")
            .param("amount", "number", "Amount field", true)
            .number_range("amount", Some(0.0), Some(1000000.0))
            .build();

        let props = tool
            .parameters
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        let amount_schema = props.get("amount").unwrap();

        assert_eq!(amount_schema.get("minimum").unwrap().as_f64().unwrap(), 0.0);
        assert_eq!(
            amount_schema.get("maximum").unwrap().as_f64().unwrap(),
            1000000.0
        );
    }

    #[test]
    fn test_tool_builder_creates_valid_definition() {
        // P16 FIX: gold_loan_tools() removed - test ToolBuilder directly
        let tool = ToolBuilder::new("check_eligibility", "Check eligibility")
            .param("weight", "number", "Weight in grams", true)
            .param("purity", "string", "Purity level", false)
            .string_enum("purity", &["24K", "22K", "18K", "14K"])
            .number_range("weight", Some(1.0), Some(10000.0))
            .build();

        assert_eq!(tool.name, "check_eligibility");

        // Verify schema structure
        let props = tool
            .parameters
            .get("properties")
            .unwrap()
            .as_object()
            .unwrap();
        assert!(props.contains_key("weight"));
        assert!(props.contains_key("purity"));

        // Verify enum constraint
        let purity_schema = props.get("purity").unwrap();
        assert!(purity_schema.get("enum").is_some());
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
        // P16 FIX: Tools created via ToolBuilder instead of hardcoded gold_loan_tools()
        let tools = vec![
            ToolBuilder::new("check_eligibility", "Check eligibility")
                .param("weight", "number", "Weight in grams", true)
                .build(),
            ToolBuilder::new("calculate_savings", "Calculate savings")
                .param("amount", "number", "Loan amount", true)
                .build(),
        ];

        let messages = PromptBuilder::new()
            .system_prompt("en")
            .with_tools(&tools)
            .user_message("Can you check my eligibility?")
            .build();

        // Should have system prompt, tool definitions, and user message
        assert!(messages.len() >= 3);

        // Tool definitions should be in the prompt
        let tool_msg = messages
            .iter()
            .find(|m| m.content.contains("TOOL_CALL"))
            .expect("Should have tool definitions");
        assert!(tool_msg.content.contains("check_eligibility"));
        assert!(tool_msg.content.contains("calculate_savings"));
        // Should include type info from JSON schema
        assert!(tool_msg.content.contains("(number)") || tool_msg.content.contains("(string)"));
    }
}
