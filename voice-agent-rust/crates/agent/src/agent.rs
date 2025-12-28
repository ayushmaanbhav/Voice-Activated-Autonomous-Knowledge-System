//! Gold Loan Voice Agent
//!
//! Main agent implementation combining all components.

use std::sync::Arc;
use tokio::sync::broadcast;

use voice_agent_llm::{PromptBuilder, Message, Role, OllamaBackend, LlmBackend, LlmConfig};
// P0 FIX: Import PersonaConfig from the single source of truth
use voice_agent_config::PersonaConfig;
use voice_agent_tools::{ToolRegistry, ToolExecutor};
// P1 FIX: Import RAG components for retrieval-augmented generation
use voice_agent_rag::{HybridRetriever, RetrieverConfig, RerankerConfig, VectorStore};

use crate::conversation::{Conversation, ConversationConfig, ConversationEvent, EndReason};
use crate::stage::ConversationStage;
use crate::AgentError;

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Default language
    pub language: String,
    /// Conversation config
    pub conversation: ConversationConfig,
    /// Persona configuration (P0 FIX: now uses consolidated PersonaConfig)
    pub persona: PersonaConfig,
    /// Enable RAG
    pub rag_enabled: bool,
    /// Enable tools
    pub tools_enabled: bool,
    /// P1 FIX: Configurable tool defaults (no more hardcoded values)
    pub tool_defaults: ToolDefaults,
    /// P2 FIX: Context window size in tokens (for LLM prompt truncation)
    pub context_window_tokens: usize,
}

/// P1 FIX: Configurable default values for tool calls
#[derive(Debug, Clone)]
pub struct ToolDefaults {
    /// Default city for branch searches
    pub default_city: String,
    /// Default gold purity for eligibility checks
    pub default_gold_purity: String,
    /// Default competitor interest rate (%)
    pub default_competitor_rate: f64,
    /// Default loan amount for savings calculations
    pub default_loan_amount: u64,
    /// Default remaining tenure (months)
    pub default_tenure_months: u32,
}

impl Default for ToolDefaults {
    fn default() -> Self {
        Self {
            default_city: "Mumbai".to_string(),
            default_gold_purity: "22K".to_string(),
            default_competitor_rate: 18.0,
            default_loan_amount: 100_000,
            default_tenure_months: 12,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            language: "hi".to_string(),
            conversation: ConversationConfig::default(),
            persona: PersonaConfig::default(),
            rag_enabled: true,
            tools_enabled: true,
            tool_defaults: ToolDefaults::default(),
            // P2 FIX: Default context window for typical LLMs (Llama 3, etc.)
            // 4096 tokens leaves room for response generation
            context_window_tokens: 4096,
        }
    }
}

impl AgentConfig {
    /// Get agent name from persona
    pub fn name(&self) -> &str {
        &self.persona.name
    }
}

// P0 FIX: PersonaTraits removed - now uses PersonaConfig from voice_agent_config
// Re-export for backwards compatibility
pub use voice_agent_config::PersonaConfig as PersonaTraits;

/// Agent events
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Response ready
    Response(String),
    /// Thinking/processing
    Thinking,
    /// Tool being called
    ToolCall { name: String },
    /// Tool result
    ToolResult { name: String, success: bool },
    /// Conversation event
    Conversation(ConversationEvent),
    /// Error
    Error(String),
}

/// Gold Loan Voice Agent
pub struct GoldLoanAgent {
    config: AgentConfig,
    conversation: Arc<Conversation>,
    tools: Arc<ToolRegistry>,
    llm: Option<Arc<dyn LlmBackend>>,
    /// P1 FIX: RAG retriever for context augmentation
    retriever: Option<Arc<HybridRetriever>>,
    /// P1 FIX: Vector store for RAG search (optional, can be injected)
    vector_store: Option<Arc<VectorStore>>,
    event_tx: broadcast::Sender<AgentEvent>,
}

impl GoldLoanAgent {
    /// Create a new agent
    pub fn new(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        let conversation = Arc::new(Conversation::new(
            session_id,
            config.conversation.clone(),
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_default_registry());

        // Try to create LLM backend (defaults to Ollama on localhost)
        // P1 FIX: Handle potential creation failure gracefully
        let llm: Option<Arc<dyn LlmBackend>> = OllamaBackend::new(LlmConfig::default())
            .map(|backend| Arc::new(backend) as Arc<dyn LlmBackend>)
            .ok();

        // P1 FIX: Create RAG retriever if enabled
        let retriever = if config.rag_enabled {
            Some(Arc::new(HybridRetriever::new(
                RetrieverConfig::default(),
                RerankerConfig::default(),
            )))
        } else {
            None
        };

        // P1 FIX: Wire LLM to memory for real summarization
        if let Some(ref llm_backend) = llm {
            conversation.memory().set_llm(llm_backend.clone());
        }

        Self {
            config,
            conversation,
            tools,
            llm,
            retriever,
            vector_store: None,
            event_tx,
        }
    }

    /// Create agent with custom LLM backend
    pub fn with_llm(
        session_id: impl Into<String>,
        config: AgentConfig,
        llm: Arc<dyn LlmBackend>,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        let conversation = Arc::new(Conversation::new(
            session_id,
            config.conversation.clone(),
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_default_registry());

        // P1 FIX: Create RAG retriever if enabled
        let retriever = if config.rag_enabled {
            Some(Arc::new(HybridRetriever::new(
                RetrieverConfig::default(),
                RerankerConfig::default(),
            )))
        } else {
            None
        };

        // P1 FIX: Wire LLM to memory for real summarization
        conversation.memory().set_llm(llm.clone());

        Self {
            config,
            conversation,
            tools,
            llm: Some(llm),
            retriever,
            vector_store: None,
            event_tx,
        }
    }

    /// Create agent without LLM (uses mock responses)
    pub fn without_llm(session_id: impl Into<String>, config: AgentConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        let conversation = Arc::new(Conversation::new(
            session_id,
            config.conversation.clone(),
        ));

        let tools = Arc::new(voice_agent_tools::registry::create_default_registry());

        // P1 FIX: Create RAG retriever if enabled
        let retriever = if config.rag_enabled {
            Some(Arc::new(HybridRetriever::new(
                RetrieverConfig::default(),
                RerankerConfig::default(),
            )))
        } else {
            None
        };

        Self {
            config,
            conversation,
            tools,
            llm: None,
            retriever,
            vector_store: None,
            event_tx,
        }
    }

    /// P1 FIX: Set vector store for RAG search
    pub fn with_vector_store(mut self, vector_store: Arc<VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// Subscribe to agent events
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }

    /// Process user input and generate response
    pub async fn process(&self, user_input: &str) -> Result<String, AgentError> {
        // Emit thinking event
        let _ = self.event_tx.send(AgentEvent::Thinking);

        // Add user turn and detect intent
        let intent = self.conversation.add_user_turn(user_input)?;

        // Forward conversation events
        let _ = self.event_tx.send(AgentEvent::Conversation(
            ConversationEvent::IntentDetected(intent.clone())
        ));

        // Check for tool calls based on intent
        let tool_result = if self.config.tools_enabled {
            self.maybe_call_tool(&intent).await?
        } else {
            None
        };

        // Build prompt for LLM
        let response = self.generate_response(user_input, tool_result.as_deref()).await?;

        // Add assistant turn
        self.conversation.add_assistant_turn(&response)?;

        // P1 FIX: Trigger memory summarization in background (non-blocking)
        // This uses the LLM (if available) to summarize conversation history
        let memory = self.conversation.memory_arc();
        tokio::spawn(async move {
            if let Err(e) = memory.summarize_pending_async().await {
                tracing::debug!("Memory summarization skipped: {}", e);
            }
        });

        // Emit response event
        let _ = self.event_tx.send(AgentEvent::Response(response.clone()));

        Ok(response)
    }

    /// Maybe call a tool based on intent
    async fn maybe_call_tool(&self, intent: &crate::intent::DetectedIntent) -> Result<Option<String>, AgentError> {
        let tool_name = match intent.intent.as_str() {
            "eligibility_check" => {
                // Check if we have required slots
                if intent.slots.contains_key("gold_weight") {
                    Some("check_eligibility")
                } else {
                    None
                }
            }
            "switch_lender" => {
                if intent.slots.contains_key("current_lender") {
                    Some("calculate_savings")
                } else {
                    None
                }
            }
            "schedule_visit" => Some("find_branches"),
            _ => None,
        };

        if let Some(name) = tool_name {
            let _ = self.event_tx.send(AgentEvent::ToolCall { name: name.to_string() });

            // Build arguments from slots
            let mut args = serde_json::Map::new();
            for (key, slot) in &intent.slots {
                if let Some(ref value) = slot.value {
                    args.insert(key.clone(), serde_json::json!(value));
                }
            }

            // P1 FIX: Use configurable defaults instead of hardcoded values
            let defaults = &self.config.tool_defaults;

            if name == "check_eligibility" && !args.contains_key("gold_purity") {
                args.insert("gold_purity".to_string(), serde_json::json!(&defaults.default_gold_purity));
            }

            if name == "calculate_savings" {
                if !args.contains_key("current_interest_rate") {
                    args.insert("current_interest_rate".to_string(), serde_json::json!(defaults.default_competitor_rate));
                }
                if !args.contains_key("current_loan_amount") {
                    args.insert("current_loan_amount".to_string(), serde_json::json!(defaults.default_loan_amount));
                }
                if !args.contains_key("remaining_tenure_months") {
                    args.insert("remaining_tenure_months".to_string(), serde_json::json!(defaults.default_tenure_months));
                }
            }

            if name == "find_branches" && !args.contains_key("city") {
                args.insert("city".to_string(), serde_json::json!(&defaults.default_city));
            }

            let result = self.tools.execute(name, serde_json::Value::Object(args)).await;

            let success = result.is_ok();
            let _ = self.event_tx.send(AgentEvent::ToolResult {
                name: name.to_string(),
                success,
            });

            match result {
                Ok(output) => {
                    // Extract text from output
                    let text = output.content.iter()
                        .filter_map(|c| match c {
                            voice_agent_tools::mcp::ContentBlock::Text { text } => Some(text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(Some(text))
                }
                Err(e) => {
                    tracing::warn!("Tool error: {}", e);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Generate response using LLM
    async fn generate_response(
        &self,
        user_input: &str,
        tool_result: Option<&str>,
    ) -> Result<String, AgentError> {
        // Build prompt - P0 FIX: now just clones consolidated PersonaConfig
        let persona = self.config.persona.clone();

        let mut builder = PromptBuilder::new()
            .with_persona(persona)
            .system_prompt(&self.config.language);

        // Add context from memory
        let context = self.conversation.get_context();
        if !context.is_empty() {
            builder = builder.with_context(&context);
        }

        // P1 FIX: Add RAG context if retriever and vector store are available
        if self.config.rag_enabled {
            if let (Some(retriever), Some(vector_store)) = (&self.retriever, &self.vector_store) {
                match retriever.search(user_input, vector_store, None).await {
                    Ok(results) if !results.is_empty() => {
                        let rag_context = results.iter()
                            .take(3) // Limit to top 3 results
                            .map(|r| format!("- {}", r.text))
                            .collect::<Vec<_>>()
                            .join("\n");
                        builder = builder.with_context(&format!("## Relevant Information\n{}", rag_context));
                        tracing::debug!("RAG retrieved {} results for query", results.len());
                    }
                    Ok(_) => {
                        tracing::debug!("RAG returned no results for query");
                    }
                    Err(e) => {
                        tracing::warn!("RAG search failed, continuing without: {}", e);
                    }
                }
            }
        }

        // Add tool result if available
        if let Some(result) = tool_result {
            builder = builder.with_context(&format!("## Tool Result\n{}", result));
        }

        // Add stage guidance
        builder = builder.with_stage_guidance(
            self.conversation.stage().display_name()
        );

        // Add conversation history
        let history: Vec<Message> = self.conversation.get_messages()
            .into_iter()
            .map(|(role, content)| {
                let r = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => Role::System,
                };
                Message { role: r, content }
            })
            .collect();

        builder = builder.with_history(&history);

        // Add current user message
        builder = builder.user_message(user_input);

        // P2 FIX: Use context window limit to truncate conversation history
        // This prevents context overflow errors with long conversations
        let messages = builder.build_with_limit(self.config.context_window_tokens);

        // Try to use LLM backend if available
        if let Some(ref llm) = self.llm {
            // Check if LLM is available
            if llm.is_available().await {
                match llm.generate(&messages).await {
                    Ok(result) => {
                        tracing::debug!(
                            "LLM generated {} tokens in {}ms (TTFT: {}ms)",
                            result.tokens,
                            result.total_time_ms,
                            result.time_to_first_token_ms
                        );
                        return Ok(result.text);
                    }
                    Err(e) => {
                        tracing::warn!("LLM generation failed, falling back to mock: {}", e);
                        // Fall through to mock response
                    }
                }
            } else {
                tracing::debug!("LLM not available, using mock response");
            }
        }

        // Fallback: generate a placeholder response based on intent and stage
        let response = self.generate_mock_response(user_input, tool_result);

        Ok(response)
    }

    /// Generate mock response (placeholder for LLM)
    /// P2 FIX: Language-aware mock responses
    ///
    /// Generates fallback responses based on configured language:
    /// - "hi" or "hi-IN": Hinglish (Hindi + English mix)
    /// - "en" or "en-IN": English
    fn generate_mock_response(&self, _user_input: &str, tool_result: Option<&str>) -> String {
        let stage = self.conversation.stage();

        // If we have tool results, incorporate them
        if let Some(result) = tool_result {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(result) {
                if let Some(msg) = json.get("message").and_then(|m| m.as_str()) {
                    return msg.to_string();
                }
            }
        }

        let name = &self.config.persona.name;
        let is_english = self.config.language.starts_with("en");

        // P2 FIX: Stage-based responses with language awareness
        match stage {
            ConversationStage::Greeting => {
                if is_english {
                    format!(
                        "Hello! I'm {}, calling from Kotak Mahindra Bank. How may I assist you today?",
                        name
                    )
                } else {
                    format!(
                        "Namaste! Main {} hoon, Kotak Mahindra Bank se. Aapki kya madad kar sakti hoon aaj?",
                        name
                    )
                }
            }
            ConversationStage::Discovery => {
                if is_english {
                    "I'd like to understand your needs better. Do you currently have a gold loan with another lender?".to_string()
                } else {
                    "Achha, aap batayein, aapka abhi kahan se gold loan hai? Main aapko dekhti hoon ki hum aapki kaise madad kar sakte hain.".to_string()
                }
            }
            ConversationStage::Qualification => {
                if is_english {
                    "That's helpful. Could you tell me how much gold you have pledged currently? And what interest rate are you paying?".to_string()
                } else {
                    "Bahut achha. Aapke paas kitna gold pledged hai abhi? Aur current rate kya chal raha hai?".to_string()
                }
            }
            ConversationStage::Presentation => {
                if is_english {
                    "At Kotak, we offer just 10.5% interest rate, which is much lower than the 18-20% NBFCs charge. Plus, you get the security of an RBI regulated bank. Would you be interested?".to_string()
                } else {
                    "Dekhiye, Kotak mein aapko sirf 10.5% rate milega, jo NBFC ke 18-20% se bahut kam hai. Aur hamare yahan RBI regulated bank ki security bhi hai. Aap interested hain?".to_string()
                }
            }
            ConversationStage::ObjectionHandling => {
                if is_english {
                    "I understand your concern. We offer a bridge loan facility that makes the transfer process seamless. Your gold is never left unprotected during the transition.".to_string()
                } else {
                    "Main samajh sakti hoon aapki chinta. Lekin dekhiye, hum ek bridge loan dete hain jo aapke transfer process ko seamless banata hai. Aapka gold kabhi bhi unprotected nahi rehta.".to_string()
                }
            }
            ConversationStage::Closing => {
                if is_english {
                    "Shall I schedule an appointment for you? You can visit your nearest branch for gold valuation.".to_string()
                } else {
                    "Toh kya main aapke liye ek appointment schedule kar doon? Aap apne nearest branch mein aa sakte hain gold valuation ke liye.".to_string()
                }
            }
            ConversationStage::Farewell => {
                if is_english {
                    "Thank you for your time! If you have any questions, please don't hesitate to call us. Have a great day!".to_string()
                } else {
                    "Dhanyavaad aapka samay dene ke liye! Agar koi bhi sawal ho toh zaroor call karein. Have a nice day!".to_string()
                }
            }
        }
    }

    /// Get current stage
    pub fn stage(&self) -> ConversationStage {
        self.conversation.stage()
    }

    /// Get conversation reference
    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    /// End conversation
    pub fn end(&self, reason: EndReason) {
        self.conversation.end(reason);
    }

    /// Get agent name
    pub fn name(&self) -> &str {
        &self.config.persona.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_creation() {
        let agent = GoldLoanAgent::new("test-session", AgentConfig::default());

        assert_eq!(agent.name(), "Priya");
        assert_eq!(agent.stage(), ConversationStage::Greeting);
    }

    #[tokio::test]
    async fn test_agent_process() {
        let agent = GoldLoanAgent::new("test", AgentConfig::default());

        let response = agent.process("Hello").await.unwrap();

        assert!(!response.is_empty());
        assert!(response.contains("Namaste") || response.contains("Hello"));
    }

    #[tokio::test]
    async fn test_agent_conversation_flow() {
        let agent = GoldLoanAgent::new("test", AgentConfig::default());

        // Greeting
        let _ = agent.process("Hello").await.unwrap();

        // Should be able to transition to discovery
        agent.conversation().transition_stage(ConversationStage::Discovery).unwrap();

        // Discovery question
        let response = agent.process("I have a loan from Muthoot").await.unwrap();

        assert!(!response.is_empty());
    }

    #[tokio::test]
    async fn test_agent_english_responses() {
        // P2 FIX: Test language-aware mock responses
        let config = AgentConfig {
            language: "en".to_string(),
            ..AgentConfig::default()
        };
        let agent = GoldLoanAgent::without_llm("test-english", config);

        let response = agent.process("Hello").await.unwrap();

        // English mode should produce English response
        assert!(response.contains("Hello") || response.contains("assist"),
            "Expected English response, got: {}", response);
        assert!(!response.contains("Namaste"),
            "Should not contain Hindi greeting in English mode");
    }

    #[tokio::test]
    async fn test_agent_hindi_responses() {
        // P2 FIX: Test language-aware mock responses
        let config = AgentConfig {
            language: "hi".to_string(),
            ..AgentConfig::default()
        };
        let agent = GoldLoanAgent::without_llm("test-hindi", config);

        let response = agent.process("Hello").await.unwrap();

        // Hindi mode should produce Hinglish response
        assert!(response.contains("Namaste") || response.contains("hoon"),
            "Expected Hinglish response, got: {}", response);
    }
}
