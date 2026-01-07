//! Response Generation Methods for DomainAgent
//!
//! This module contains response generation functionality including:
//! - LLM-based response generation
//! - Mock/fallback responses
//! - Stage-aware response adaptation

use super::DomainAgent;
use crate::stage::ConversationStage;
use crate::AgentError;
use voice_agent_core::{FinishReason, ToolDefinition};
use voice_agent_llm::{Message, PromptBuilder, Role};
use voice_agent_rag::QueryContext;
use voice_agent_tools::ToolExecutor;

impl DomainAgent {
    /// Generate response using LLM
    pub(super) async fn generate_response(
        &self,
        user_input: &str,
        tool_result: Option<&str>,
    ) -> Result<String, AgentError> {
        // Build prompt - P0 FIX: now just clones consolidated PersonaConfig
        let persona = self.config.persona.clone();

        let mut builder = PromptBuilder::new()
            .with_persona(persona)
            .system_prompt(&self.config.language);

        // P4 FIX: Add personalization instructions based on detected signals
        // This dynamically adapts the prompt based on customer behavior
        {
            let ctx = self.personalization_ctx.read();
            let personalization_instructions = self.personalization.generate_instructions(&ctx);
            if !personalization_instructions.is_empty() {
                builder = builder.with_context(&format!(
                    "## Personalization Guidance\n{}",
                    personalization_instructions
                ));
                tracing::trace!(
                    instructions_len = personalization_instructions.len(),
                    "Added personalization instructions to prompt"
                );
            }
        }

        // Add context from memory with query-based archival retrieval
        // Phase 10: Use get_context_for_query to include relevant archival memories
        let stage = self.conversation.stage();
        let context_budget = stage.context_budget_tokens();
        let context = self.conversation.get_context_for_query(user_input, context_budget);
        if !context.is_empty() {
            builder = builder.with_context(&context);
        }

        // P1 FIX: Add RAG context if retriever and vector store are available
        // P2 FIX: Use prefetched results if available, otherwise do fresh search
        // P2 FIX: Stage-aware RAG - use rag_context_fraction to determine how much RAG to include
        if self.config.rag_enabled {
            let stage = self.conversation.stage();
            let rag_fraction = stage.rag_context_fraction();

            // Skip RAG entirely for stages that don't need it (greeting, farewell)
            if rag_fraction > 0.0 {
                // Phase 11: Use AgenticRetriever for multi-step retrieval
                if let (Some(agentic_retriever), Some(vector_store)) =
                    (&self.agentic_retriever, &self.vector_store)
                {
                    // First, try to use prefetched results
                    let results = if let Some(prefetched) = self.get_prefetch_results(user_input) {
                        tracing::debug!("Using {} prefetched RAG results", prefetched.len());
                        // Clear cache after use
                        self.clear_prefetch_cache();
                        prefetched
                    } else {
                        // Build query context for agentic retrieval
                        let human_block = self.conversation.agentic_memory().core.human_snapshot();
                        let query_context = QueryContext {
                            // Use conversation context as summary for query rewriting
                            summary: self.conversation.get_context(),
                            stage: Some(stage.display_name().to_string()),
                            entities: human_block
                                .facts
                                .iter()
                                .map(|(k, entry)| (k.clone(), entry.value.clone()))
                                .collect(),
                        };

                        // Use AgenticRetriever for multi-step retrieval
                        match agentic_retriever
                            .search(user_input, vector_store, Some(&query_context))
                            .await
                        {
                            Ok(agentic_result) => {
                                if agentic_result.query_rewritten {
                                    tracing::debug!(
                                        original = %user_input,
                                        rewritten = %agentic_result.final_query,
                                        iterations = agentic_result.iterations,
                                        "Agentic RAG rewrote query (streaming)"
                                    );
                                }
                                agentic_result.results
                            }
                            Err(e) => {
                                tracing::warn!("RAG search failed, continuing without: {}", e);
                                Vec::new()
                            }
                        }
                    };

                    if !results.is_empty() {
                        // P2 FIX: Calculate how many results to include based on stage RAG fraction
                        // Higher fraction = more results (1-5 based on fraction)
                        let max_results = ((rag_fraction * 10.0).ceil() as usize).clamp(1, 5);

                        let rag_context = results
                            .iter()
                            .take(max_results)
                            .map(|r| format!("- {}", r.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        builder = builder
                            .with_context(&format!("## Relevant Information\n{}", rag_context));

                        tracing::debug!(
                            stage = ?stage,
                            rag_fraction = rag_fraction,
                            max_results = max_results,
                            actual_results = results.len().min(max_results),
                            "Stage-aware RAG context added"
                        );
                    } else {
                        tracing::debug!("RAG returned no results for query");
                    }
                }
            } else {
                tracing::trace!(stage = ?stage, "Skipping RAG for stage with rag_fraction=0");
            }
        }

        // Add tool result if available
        if let Some(result) = tool_result {
            builder = builder.with_context(&format!("## Tool Result\n{}", result));
        }

        // Add stage guidance
        builder = builder.with_stage_guidance(self.conversation.stage().display_name());

        // P0 FIX: Detect objections and add persuasion guidance to prompt
        // Uses acknowledge-reframe-evidence pattern from PersuasionEngine
        if let Some(objection_response) = self
            .persuasion
            .handle_objection(user_input, self.user_language)
        {
            let persuasion_guidance = format!(
                "## Objection Handling Guidance\n\
                The customer appears to have a concern. Use this framework:\n\
                1. **Acknowledge**: {}\n\
                2. **Reframe**: {}\n\
                3. **Evidence**: {}\n\
                4. **Call to Action**: {}",
                objection_response.acknowledge,
                objection_response.reframe,
                objection_response.evidence,
                objection_response.call_to_action
            );
            builder = builder.with_context(&persuasion_guidance);

            tracing::debug!(
                objection_type = ?crate::persuasion::ObjectionType::detect(user_input),
                "Detected objection, adding persuasion guidance to prompt"
            );
        }

        // Add conversation history
        let history: Vec<Message> = self
            .conversation
            .get_messages()
            .into_iter()
            .map(|(role, content)| {
                let r = match role.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => Role::System,
                };
                Message {
                    role: r,
                    content,
                    name: None,
                    tool_call_id: None,
                }
            })
            .collect();

        builder = builder.with_history(&history);

        // Add current user message
        builder = builder.user_message(user_input);

        // P2 FIX: Use stage-aware context budget to truncate conversation history
        // Different stages need different amounts of context - early stages need less,
        // presentation/objection handling stages need more for RAG and full history
        let stage = self.conversation.stage();
        let stage_budget = stage.context_budget_tokens();
        // Use the minimum of configured limit and stage-aware budget
        let effective_budget = self.config.context_window_tokens.min(stage_budget);

        tracing::debug!(
            stage = ?stage,
            stage_budget = stage_budget,
            effective_budget = effective_budget,
            "Using stage-aware context budget"
        );

        // P1-2 FIX: Try speculative execution first if enabled and appropriate
        // Speculative doesn't support tool calling, so only use for non-tool responses
        let tool_defs: Vec<ToolDefinition> = if self.config.tools_enabled {
            self.tools
                .list_tools()
                .iter()
                .map(ToolDefinition::from_schema)
                .collect()
        } else {
            Vec::new()
        };

        let has_tools = !tool_defs.is_empty();

        // P1-2 FIX: Use speculative executor when available and no tools needed
        if let Some(ref speculative) = self.speculative {
            if !has_tools {
                // Build messages for speculative executor (uses llm crate's Message type)
                let messages = builder.build_with_limit(effective_budget);

                tracing::debug!(
                    mode = ?self.config.speculative.mode,
                    message_count = messages.len(),
                    "Using speculative executor"
                );

                match speculative.execute(&messages).await {
                    Ok(result) => {
                        tracing::debug!(
                            model_used = ?result.model_used,
                            used_fallback = result.used_fallback,
                            complexity = ?result.complexity_score,
                            tokens = result.generation.tokens,
                            "Speculative execution succeeded"
                        );
                        return Ok(result.text);
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Speculative execution failed, falling back to direct LLM"
                        );
                        // Fall through to direct LLM path
                    }
                }
            } else {
                tracing::debug!("Skipping speculative executor - tool calling required");
            }
        }

        // P1 FIX: Use build_request_with_limit for LanguageModel trait (fallback path)
        // Rebuild the request since speculative may have consumed the builder
        let request = self.build_llm_request(user_input, tool_result).await?;

        // Try to use LLM backend if available
        if let Some(ref llm) = self.llm {
            // Check if LLM is available
            if llm.is_available().await {
                tracing::debug!(
                    tool_count = tool_defs.len(),
                    tools_enabled = self.config.tools_enabled,
                    "Calling LLM with tool definitions"
                );

                // P0-2 FIX: Use generate_with_tools when tools are available
                let result = if has_tools {
                    llm.generate_with_tools(request, &tool_defs).await
                } else {
                    llm.generate(request).await
                };

                match result {
                    Ok(response) => {
                        // P1 FIX: Use GenerateResponse fields (LanguageModel trait)
                        let tokens = response
                            .usage
                            .as_ref()
                            .map(|u| u.completion_tokens)
                            .unwrap_or(0);
                        tracing::debug!(
                            "LLM generated {} tokens, finish_reason={:?}, tool_calls={}",
                            tokens,
                            response.finish_reason,
                            response.tool_calls.len()
                        );

                        // P0-2 FIX: Handle tool calls from LLM
                        if response.finish_reason == FinishReason::ToolCalls
                            && !response.tool_calls.is_empty()
                        {
                            tracing::info!(
                                tool_calls = response.tool_calls.len(),
                                "LLM requested tool calls"
                            );

                            // Execute each tool call and collect results
                            let mut tool_results = Vec::new();
                            for tool_call in &response.tool_calls {
                                let _ = self.event_tx.send(crate::agent_config::AgentEvent::ToolCall {
                                    name: tool_call.name.clone(),
                                });

                                // Convert HashMap arguments to serde_json::Value
                                let args = serde_json::to_value(&tool_call.arguments)
                                    .unwrap_or(serde_json::json!({}));

                                match self.tools.execute(&tool_call.name, args).await {
                                    Ok(output) => {
                                        let _ = self.event_tx.send(
                                            crate::agent_config::AgentEvent::ToolResult {
                                                name: tool_call.name.clone(),
                                                success: true,
                                            },
                                        );

                                        // Extract text from output
                                        let text = output
                                            .content
                                            .iter()
                                            .filter_map(|c| match c {
                                                voice_agent_tools::mcp::ContentBlock::Text {
                                                    text,
                                                } => Some(text.clone()),
                                                _ => None,
                                            })
                                            .collect::<Vec<_>>()
                                            .join("\n");

                                        tool_results.push(format!(
                                            "Tool '{}' result:\n{}",
                                            tool_call.name, text
                                        ));
                                        tracing::debug!(
                                            tool = %tool_call.name,
                                            "Tool execution successful"
                                        );
                                    }
                                    Err(e) => {
                                        let _ = self.event_tx.send(
                                            crate::agent_config::AgentEvent::ToolResult {
                                                name: tool_call.name.clone(),
                                                success: false,
                                            },
                                        );
                                        tool_results.push(format!(
                                            "Tool '{}' failed: {}",
                                            tool_call.name, e
                                        ));
                                        tracing::warn!(
                                            tool = %tool_call.name,
                                            error = %e,
                                            "Tool execution failed"
                                        );
                                    }
                                }
                            }

                            // Recursive call with tool results to get final response
                            // Use Box::pin to avoid infinitely-sized future
                            let combined_results = tool_results.join("\n\n");
                            return Box::pin(
                                self.generate_response(user_input, Some(&combined_results)),
                            )
                            .await;
                        }

                        return Ok(response.text);
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
    pub(super) fn generate_mock_response(&self, _user_input: &str, tool_result: Option<&str>) -> String {
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
                // P13 FIX: Get rates from config if available
                let (our_rate, nbfc_rate) = if let Some(view) = &self.domain_view {
                    (
                        view.our_rate_for_amount(500_000.0),
                        view.get_competitor_rate("muthoot").unwrap_or(18.0),
                    )
                } else {
                    (10.5, 18.0)
                };
                if is_english {
                    format!("At Kotak, we offer just {:.1}% interest rate, which is much lower than the {:.0}-{:.0}% NBFCs charge. Plus, you get the security of an RBI regulated bank. Would you be interested?", our_rate, nbfc_rate, nbfc_rate + 2.0)
                } else {
                    format!("Dekhiye, Kotak mein aapko sirf {:.1}% rate milega, jo NBFC ke {:.0}-{:.0}% se bahut kam hai. Aur hamare yahan RBI regulated bank ki security bhi hai. Aap interested hain?", our_rate, nbfc_rate, nbfc_rate + 2.0)
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
}
