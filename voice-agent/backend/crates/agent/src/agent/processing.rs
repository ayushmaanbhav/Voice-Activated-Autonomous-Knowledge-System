//! Core Processing Methods for DomainAgent
//!
//! This module contains the main processing logic including:
//! - process() - Main turn processing
//! - process_stream() - Streaming turn processing
//! - build_llm_request() - LLM request construction

use futures::StreamExt;

use super::{find_sentence_end, DomainAgent};
use crate::agent_config::AgentEvent;
use crate::conversation::ConversationEvent;
use crate::lead_scoring::{EscalationTrigger, LeadRecommendation};
use crate::memory::{ConversationTurn, TurnRole};
use crate::AgentError;
use voice_agent_core::Language;
use voice_agent_llm::{Message, PromptBuilder, Role};
use voice_agent_rag::QueryContext;

impl DomainAgent {
    /// Process user input and generate response
    ///
    /// P5 FIX: Implements Translate-Think-Translate pattern:
    /// 1. If user language is not English, translate input to English
    /// 2. Process with LLM (which works best in English)
    /// 3. Translate response back to user's language
    pub async fn process(&self, user_input: &str) -> Result<String, AgentError> {
        // Emit thinking event
        let _ = self.event_tx.send(AgentEvent::Thinking);

        // P5 FIX: Translate user input to English if needed
        let english_input = if self.user_language != Language::English {
            if let Some(ref translator) = self.translator {
                match translator
                    .translate(user_input, self.user_language, Language::English)
                    .await
                {
                    Ok(translated) => {
                        tracing::debug!(
                            from = ?self.user_language,
                            original = %user_input,
                            translated = %translated,
                            "Translated user input to English"
                        );
                        translated
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Translation failed, using original input"
                        );
                        user_input.to_string()
                    }
                }
            } else {
                user_input.to_string()
            }
        } else {
            user_input.to_string()
        };

        // Add user turn and detect intent
        let intent = self.conversation.add_user_turn(user_input)?;

        // Add to MemGPT-style agentic memory recall
        let turn = ConversationTurn::new(TurnRole::User, user_input)
            .with_intents(vec![intent.intent.clone()])
            .with_entities(
                intent
                    .slots
                    .iter()
                    .filter_map(|(k, v)| v.value.as_ref().map(|val| (k.clone(), val.clone())))
                    .collect(),
            )
            .with_stage(self.conversation.stage().display_name());
        self.conversation.agentic_memory().add_turn(turn);

        // Log memory state
        let stats = self.conversation.agentic_memory().get_stats();
        tracing::debug!(
            role = "user",
            fifo_tokens = stats.fifo_tokens,
            core_tokens = stats.core_tokens,
            total_turns = self.conversation.agentic_memory().get_recent_turns().len(),
            "Added user turn to agentic memory"
        );

        // Extract and store customer facts from intent slots in core memory
        for (key, slot) in &intent.slots {
            if let Some(ref value) = slot.value {
                let fact_key = match key.as_str() {
                    "gold_weight" | "weight" => Some("gold_weight"),
                    "gold_purity" | "purity" | "karat" => Some("gold_purity"),
                    "loan_amount" | "amount" => Some("loan_amount"),
                    "current_lender" | "lender" => Some("current_lender"),
                    "interest_rate" | "rate" => Some("current_interest_rate"),
                    "city" | "location" => Some("city"),
                    "customer_name" | "name" => {
                        self.set_customer_name(value);
                        None
                    }
                    "phone_number" | "phone" | "mobile" => Some("phone"),
                    _ => None,
                };
                if let Some(k) = fact_key {
                    let _ = self.conversation.agentic_memory().core_memory_append(k, value);
                }
            }
        }

        // Phase 5: Update Dialogue State Tracker with detected intent
        {
            let mut dst = self.dialogue_state.write();
            dst.update(&intent);

            let turn = dst.history().len();
            dst.update_goal_from_intent(&intent.intent, turn);

            tracing::debug!(
                primary_intent = ?dst.state().primary_intent(),
                filled_slots = ?dst.state().filled_slots(),
                goal_id = %dst.goal_id(),
                pending = ?dst.slots_needing_confirmation(),
                "Dialogue state updated"
            );
        }

        // P4 FIX: Process input through personalization engine
        {
            let mut ctx = self.personalization_ctx.write();
            self.personalization.process_input(&mut ctx, user_input);

            if let Some(recent_signal) = ctx.recent_signals(1).first() {
                tracing::debug!(signal = ?recent_signal, "Personalization signal detected");
            }
        }

        // Phase 10: Update lead scoring engine with detected signals
        {
            let mut lead_scoring = self.lead_scoring.write();

            lead_scoring.update_urgency(user_input);

            let slot_values: std::collections::HashMap<String, String> = intent
                .slots
                .iter()
                .filter_map(|(k, v)| v.value.as_ref().map(|val| (k.clone(), val.clone())))
                .collect();

            lead_scoring.update_from_intent(&intent.intent, &slot_values);

            if !slot_values.is_empty() {
                lead_scoring.update_trust(true);
            }

            if let Some(amount_str) = slot_values.get("loan_amount").or(slot_values.get("amount"))
            {
                if let Ok(amount) = amount_str.replace(",", "").parse::<f64>() {
                    if let Some(_trigger) = lead_scoring.check_high_value_loan(amount) {
                        tracing::info!(
                            amount = amount,
                            "High-value loan detected, escalation may be triggered"
                        );
                    }
                }
            }

            tracing::debug!(
                engagement_turns = lead_scoring.signals().engagement_turns,
                has_urgency = lead_scoring.signals().has_urgency_signal,
                provided_contact = lead_scoring.signals().provided_contact_info,
                "Lead scoring signals updated"
            );
        }

        // Forward conversation events
        let _ = self
            .event_tx
            .send(AgentEvent::Conversation(ConversationEvent::IntentDetected(
                intent.clone(),
            )));

        // Check for tool calls based on intent
        let tool_result = if self.config.tools_enabled {
            self.maybe_call_tool(&intent).await?
        } else {
            None
        };

        // Phase 12: Auto-capture lead when we have contact info
        if self.config.tools_enabled {
            let should_capture = {
                let dst = self.dialogue_state.read();
                dst.should_auto_capture_lead()
            };

            if should_capture {
                tracing::info!("Auto-capturing lead with collected contact information");
                let lead_result = self.call_tool_by_name("capture_lead", &intent).await;
                if let Ok(Some(_)) = lead_result {
                    tracing::info!("Lead captured successfully");
                } else {
                    tracing::warn!("Auto lead capture failed or returned empty");
                }
            }
        }

        // Build prompt for LLM
        let english_response = self
            .generate_response(&english_input, tool_result.as_deref())
            .await?;

        // P5 FIX: Translate response back to user's language if needed
        let response = if self.user_language != Language::English {
            if let Some(ref translator) = self.translator {
                match translator
                    .translate(&english_response, Language::English, self.user_language)
                    .await
                {
                    Ok(translated) => {
                        tracing::debug!(
                            to = ?self.user_language,
                            original = %english_response,
                            translated = %translated,
                            "Translated response to user language"
                        );
                        translated
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Response translation failed, using English response"
                        );
                        english_response
                    }
                }
            } else {
                english_response
            }
        } else {
            english_response
        };

        // Add assistant turn
        self.conversation.add_assistant_turn(&response)?;

        // Add to MemGPT-style agentic memory recall
        let assistant_turn = ConversationTurn::new(TurnRole::Assistant, &response)
            .with_stage(self.conversation.stage().display_name());
        self.conversation.agentic_memory().add_turn(assistant_turn);

        // Log memory state
        let stats = self.conversation.agentic_memory().get_stats();
        tracing::debug!(
            role = "assistant",
            fifo_tokens = stats.fifo_tokens,
            core_tokens = stats.core_tokens,
            total_turns = self.conversation.agentic_memory().get_recent_turns().len(),
            "Added assistant turn to agentic memory"
        );

        // P1 FIX: Trigger memory summarization in background
        let memory = self.conversation.memory_arc();
        tokio::spawn(async move {
            if let Err(e) = memory.summarize_pending_async().await {
                tracing::debug!("Memory summarization skipped: {}", e);
            }
        });

        // P2 FIX: Check memory usage and cleanup if needed
        {
            let memory = self.conversation.memory_arc();
            if memory.needs_cleanup() {
                tracing::info!("Memory high watermark exceeded, triggering cleanup");
                memory.cleanup_to_watermark();
            }
        }

        // Check agentic memory compaction
        if self.conversation.agentic_memory().needs_compaction() {
            let stats = self.conversation.agentic_memory().get_stats();
            tracing::debug!(
                core_tokens = stats.core_tokens,
                fifo_tokens = stats.fifo_tokens,
                archival_count = stats.archival_count,
                "Agentic memory high watermark exceeded"
            );
        }

        // Phase 10: Calculate lead score and emit events
        let lead_score = {
            let mut lead_scoring = self.lead_scoring.write();
            lead_scoring.calculate_score()
        };

        // Emit lead score update event
        let _ = self.event_tx.send(AgentEvent::LeadScoreUpdated {
            score: lead_score.total,
            qualification: format!("{:?}", lead_score.qualification),
            classification: format!("{:?}", lead_score.classification),
            conversion_probability: lead_score.conversion_probability,
        });

        tracing::info!(
            score = lead_score.total,
            qualification = ?lead_score.qualification,
            classification = ?lead_score.classification,
            conversion_prob = lead_score.conversion_probability,
            recommendation = ?lead_score.recommendation,
            "Lead score calculated"
        );

        // Check for escalation triggers
        for trigger in &lead_score.escalation_triggers {
            let trigger_str = match trigger {
                EscalationTrigger::ExcessiveObjections { count, threshold } => {
                    format!(
                        "ExcessiveObjections: {} objections (threshold: {})",
                        count, threshold
                    )
                }
                EscalationTrigger::ConversationStalled { turns, threshold } => {
                    format!(
                        "ConversationStalled: {} turns (threshold: {})",
                        turns, threshold
                    )
                }
                EscalationTrigger::HighValueLoan { amount, threshold } => {
                    format!(
                        "HighValueLoan: ₹{:.0} (threshold: ₹{:.0})",
                        amount, threshold
                    )
                }
                EscalationTrigger::CustomerFrustration => "CustomerFrustration".to_string(),
                EscalationTrigger::CustomerRequested => "CustomerRequested".to_string(),
                EscalationTrigger::ComplexQuery => "ComplexQuery".to_string(),
                EscalationTrigger::ComplianceSensitive => "ComplianceSensitive".to_string(),
            };

            let recommendation_str = match &lead_score.recommendation {
                LeadRecommendation::ContinueConversation => "ContinueConversation".to_string(),
                LeadRecommendation::PushForAppointment => "PushForAppointment".to_string(),
                LeadRecommendation::OfferCallback => "OfferCallback".to_string(),
                LeadRecommendation::EscalateNow { reason } => format!("EscalateNow: {}", reason),
                LeadRecommendation::SendFollowUp => "SendFollowUp".to_string(),
                LeadRecommendation::LowPriority => "LowPriority".to_string(),
            };

            tracing::warn!(
                trigger = %trigger_str,
                recommendation = %recommendation_str,
                "Escalation trigger detected"
            );

            let _ = self.event_tx.send(AgentEvent::EscalationTriggered {
                trigger: trigger_str,
                recommendation: recommendation_str,
            });
        }

        // Emit response event
        let _ = self.event_tx.send(AgentEvent::Response(response.clone()));

        Ok(response)
    }

    /// P0-2 FIX: Process user input with streaming LLM output
    pub async fn process_stream(
        &self,
        user_input: &str,
    ) -> Result<tokio::sync::mpsc::Receiver<String>, AgentError> {
        // Emit thinking event
        let _ = self.event_tx.send(AgentEvent::Thinking);

        // P5 FIX: Translate user input to English if needed
        let english_input = if self.user_language != Language::English {
            if let Some(ref translator) = self.translator {
                translator
                    .translate(user_input, self.user_language, Language::English)
                    .await
                    .unwrap_or_else(|_| user_input.to_string())
            } else {
                user_input.to_string()
            }
        } else {
            user_input.to_string()
        };

        // Add user turn and detect intent
        let intent = self.conversation.add_user_turn(user_input)?;

        // P4 FIX: Process through personalization engine
        {
            let mut ctx = self.personalization_ctx.write();
            self.personalization.process_input(&mut ctx, user_input);
        }

        // Forward intent event
        let _ = self
            .event_tx
            .send(AgentEvent::Conversation(ConversationEvent::IntentDetected(
                intent.clone(),
            )));

        // Check for tool calls
        let tool_result = if self.config.tools_enabled {
            self.maybe_call_tool(&intent).await?
        } else {
            None
        };

        // Build prompt
        let prompt_request = self
            .build_llm_request(&english_input, tool_result.as_deref())
            .await?;

        // Create output channel
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(32);

        // Check if LLM is available for streaming
        if let Some(ref llm) = self.llm {
            if llm.is_available().await {
                let mut stream = llm.generate_stream(prompt_request);

                let translator = &self.translator;
                let user_language = self.user_language;
                let terminators = user_language.sentence_terminators();

                let mut buffer = String::new();
                let mut full_response = String::new();

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            buffer.push_str(&chunk.delta);
                            full_response.push_str(&chunk.delta);

                            while let Some(pos) = find_sentence_end(&buffer, terminators) {
                                let sentence = buffer[..=pos].trim().to_string();
                                buffer = buffer[pos + 1..].to_string();

                                if sentence.is_empty() {
                                    continue;
                                }

                                let translated = if user_language != Language::English {
                                    if let Some(ref t) = translator {
                                        t.translate(&sentence, Language::English, user_language)
                                            .await
                                            .unwrap_or(sentence)
                                    } else {
                                        sentence
                                    }
                                } else {
                                    sentence
                                };

                                if tx.send(translated).await.is_err() {
                                    tracing::debug!("Stream receiver dropped");
                                    break;
                                }
                            }

                            if chunk.is_final {
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("LLM stream error: {}", e);
                            break;
                        }
                    }
                }

                // Flush remaining buffer
                if !buffer.trim().is_empty() {
                    let sentence = buffer.trim().to_string();
                    let translated = if user_language != Language::English {
                        if let Some(ref t) = translator {
                            t.translate(&sentence, Language::English, user_language)
                                .await
                                .unwrap_or(sentence)
                        } else {
                            sentence
                        }
                    } else {
                        sentence
                    };
                    let _ = tx.send(translated).await;
                }

                // Update conversation with full response
                let final_response = if user_language != Language::English {
                    if let Some(ref t) = translator {
                        t.translate(&full_response, Language::English, user_language)
                            .await
                            .unwrap_or(full_response.clone())
                    } else {
                        full_response.clone()
                    }
                } else {
                    full_response.clone()
                };

                if let Err(e) = self.conversation.add_assistant_turn(&final_response) {
                    tracing::warn!("Failed to add assistant turn: {}", e);
                }

                let _ = self.event_tx.send(AgentEvent::Response(final_response));

                return Ok(rx);
            }
        }

        // Fallback: No LLM available
        let response = self.generate_mock_response(user_input, tool_result.as_deref());
        self.conversation.add_assistant_turn(&response)?;
        let _ = self.event_tx.send(AgentEvent::Response(response.clone()));

        let _ = tx.send(response).await;

        Ok(rx)
    }

    /// Build LLM request
    pub(super) async fn build_llm_request(
        &self,
        english_input: &str,
        tool_result: Option<&str>,
    ) -> Result<voice_agent_core::GenerateRequest, AgentError> {
        let persona = self.config.persona.clone();

        let mut builder = PromptBuilder::new()
            .with_persona(persona)
            .system_prompt(&self.config.language);

        // Add personalization instructions
        {
            let ctx = self.personalization_ctx.read();
            let instructions = self.personalization.generate_instructions(&ctx);
            if !instructions.is_empty() {
                builder =
                    builder.with_context(&format!("## Personalization Guidance\n{}", instructions));
            }
        }

        // Add memory context with query-based archival retrieval
        let stage = self.conversation.stage();
        let context_budget = stage.context_budget_tokens();
        let context = self
            .conversation
            .get_context_for_query(english_input, context_budget);

        let memory_stats = self.conversation.agentic_memory().get_stats();
        let recent_turns = self.conversation.agentic_memory().get_recent_turns();
        tracing::debug!(
            context_len = context.len(),
            context_budget = context_budget,
            core_tokens = memory_stats.core_tokens,
            fifo_tokens = memory_stats.fifo_tokens,
            archival_count = memory_stats.archival_count,
            recent_turns_count = recent_turns.len(),
            stage = ?stage,
            "Memory context for LLM"
        );

        if tracing::enabled!(tracing::Level::TRACE) {
            let context_preview = if context.len() > 500 {
                format!(
                    "{}...[truncated {} chars]",
                    &context[..500],
                    context.len() - 500
                )
            } else {
                context.clone()
            };
            tracing::trace!(context = %context_preview, "Full memory context");
        }

        if !context.is_empty() {
            builder = builder.with_context(&context);
        }

        // Phase 5 + Phase 12: Add DST state context with goal tracking
        {
            let dst = self.dialogue_state.read();
            let dst_context = dst.state_context();

            if !dst_context.is_empty() && dst_context != "No information collected yet." {
                let dst_section = format!(
                    "## IMPORTANT: Customer Details (Use these for recall)\n{}\n\n## Slots Needing Confirmation\n{}",
                    dst_context,
                    if dst.slots_needing_confirmation().is_empty() {
                        "None".to_string()
                    } else {
                        dst.slots_needing_confirmation().join(", ")
                    }
                );
                builder = builder.with_context(&dst_section);
            }

            let human_block = self.conversation.agentic_memory().core.human_snapshot();
            if !human_block.facts.is_empty() {
                let facts_str = human_block
                    .facts
                    .iter()
                    .map(|(k, entry)| format!("- {}: {}", k, entry.value))
                    .collect::<Vec<_>>()
                    .join("\n");
                builder =
                    builder.with_context(&format!("## Customer Facts from Memory\n{}", facts_str));
            }

            let goal_id = dst.goal_id();
            builder = builder.with_context(&format!("Current Goal: {}", goal_id));

            tracing::debug!(
                goal = %goal_id,
                "Goal context added to prompt"
            );
        }

        // Phase 11: Add RAG context using Agentic RAG
        if self.config.rag_enabled {
            let stage = self.conversation.stage();
            let rag_fraction = stage.rag_context_fraction();

            if rag_fraction > 0.0 {
                if let (Some(agentic_retriever), Some(vector_store)) =
                    (&self.agentic_retriever, &self.vector_store)
                {
                    let results = if let Some(prefetched) = self.get_prefetch_results(english_input)
                    {
                        self.clear_prefetch_cache();
                        prefetched
                    } else {
                        let human_block = self.conversation.agentic_memory().core.human_snapshot();
                        let query_context = QueryContext {
                            summary: self.conversation.get_context(),
                            stage: Some(stage.display_name().to_string()),
                            entities: human_block
                                .facts
                                .iter()
                                .map(|(k, entry)| (k.clone(), entry.value.clone()))
                                .collect(),
                        };

                        match agentic_retriever
                            .search(english_input, vector_store, Some(&query_context))
                            .await
                        {
                            Ok(agentic_result) => {
                                if agentic_result.query_rewritten {
                                    tracing::debug!(
                                        original = %english_input,
                                        rewritten = %agentic_result.final_query,
                                        iterations = agentic_result.iterations,
                                        sufficiency = agentic_result.sufficiency_score,
                                        "Agentic RAG rewrote query"
                                    );
                                }
                                agentic_result.results
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Agentic RAG search failed");
                                vec![]
                            }
                        }
                    };

                    if !results.is_empty() {
                        let max_results = ((rag_fraction * 10.0).ceil() as usize).clamp(1, 5);
                        let rag_context = results
                            .iter()
                            .take(max_results)
                            .map(|r| format!("- {}", r.content))
                            .collect::<Vec<_>>()
                            .join("\n");
                        builder = builder
                            .with_context(&format!("## Relevant Information\n{}", rag_context));
                    }
                }
            }
        }

        // Add tool result
        if let Some(result) = tool_result {
            builder = builder.with_context(&format!("## Tool Result\n{}", result));
        }

        // Add stage guidance
        builder = builder.with_stage_guidance(self.conversation.stage().display_name());

        // Add persuasion guidance
        if let Some(objection_response) = self
            .persuasion
            .handle_objection(english_input, self.user_language)
        {
            let guidance = format!(
                "## Objection Handling Guidance\n\
                1. **Acknowledge**: {}\n\
                2. **Reframe**: {}\n\
                3. **Evidence**: {}\n\
                4. **Call to Action**: {}",
                objection_response.acknowledge,
                objection_response.reframe,
                objection_response.evidence,
                objection_response.call_to_action
            );
            builder = builder.with_context(&guidance);
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

        // Add current message
        builder = builder.user_message(english_input);

        // Build with context budget
        let stage = self.conversation.stage();
        let effective_budget = self
            .config
            .context_window_tokens
            .min(stage.context_budget_tokens());

        Ok(builder.build_request_with_limit(effective_budget))
    }
}
