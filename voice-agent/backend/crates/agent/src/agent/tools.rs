//! Tool Calling Methods for DomainAgent
//!
//! This module contains tool-related functionality including:
//! - Intent-based tool invocation
//! - DST-enriched tool calls
//! - Tool argument mapping and defaults

use super::DomainAgent;
use crate::agent_config::AgentEvent;
use crate::dst::DialogueStateTrait;
use crate::AgentError;
use voice_agent_tools::ToolExecutor;

impl DomainAgent {
    /// Maybe call a tool based on intent
    /// P16 FIX: Now uses config-driven intent-to-tool mappings
    pub(super) async fn maybe_call_tool(
        &self,
        intent: &crate::intent::DetectedIntent,
    ) -> Result<Option<String>, AgentError> {
        // Collect available slot names
        let available_slots: Vec<&str> = intent.slots.keys().map(|s| s.as_str()).collect();

        // P16 FIX: Try config-driven intent-to-tool resolution first
        let tool_name = self.domain_view
            .as_ref()
            .and_then(|view| {
                if view.has_intent_mappings() {
                    view.resolve_tool_for_intent(&intent.intent, &available_slots)
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .or_else(|| {
                // Legacy fallback mappings (used when no config mappings)
                match intent.intent.as_str() {
                    "eligibility_check" => {
                        if intent.slots.contains_key("gold_weight") {
                            Some("check_eligibility".to_string())
                        } else {
                            None
                        }
                    }
                    "switch_lender" => {
                        if intent.slots.contains_key("current_lender") {
                            Some("calculate_savings".to_string())
                        } else {
                            None
                        }
                    }
                    "schedule_visit" => Some("find_locations".to_string()),
                    "capture_lead" | "interested" | "callback_request" => {
                        if intent.slots.contains_key("customer_name")
                            || intent.slots.contains_key("phone_number")
                        {
                            Some("capture_lead".to_string())
                        } else {
                            None
                        }
                    }
                    "schedule_appointment" | "book_appointment" | "visit_branch" => {
                        if intent.slots.contains_key("preferred_date")
                            || intent.slots.contains_key("branch_id")
                        {
                            Some("schedule_appointment".to_string())
                        } else {
                            Some("find_locations".to_string())
                        }
                    }
                    "gold_price" | "check_gold_price" | "price_inquiry" | "current_rate" => {
                        Some("get_price".to_string())
                    }
                    "escalate" | "human_agent" | "speak_to_person" | "talk_to_human" | "real_person" => {
                        Some("escalate_to_human".to_string())
                    }
                    "send_sms" | "send_message" | "text_me" | "send_details" | "sms_info" => {
                        Some("send_sms".to_string())
                    }
                    _ => None,
                }
            });

        if let Some(name) = tool_name {
            let _ = self.event_tx.send(AgentEvent::ToolCall {
                name: name.to_string(),
            });

            // Build arguments from slots
            let mut args = serde_json::Map::new();
            for (key, slot) in &intent.slots {
                if let Some(ref value) = slot.value {
                    args.insert(key.clone(), serde_json::json!(value));
                }
            }

            // P16 FIX: Use config-driven tool defaults
            // First try config-driven defaults, then fall back to hardcoded defaults
            if let Some(view) = self.domain_view.as_ref() {
                // Apply argument name mappings from config
                if let Some(arg_mapping) = view.get_argument_mapping(&name) {
                    let keys: Vec<String> = args.keys().cloned().collect();
                    for slot_name in keys {
                        if let Some(arg_name) = arg_mapping.get(&slot_name) {
                            if !args.contains_key(arg_name) {
                                if let Some(value) = args.remove(&slot_name) {
                                    args.insert(arg_name.clone(), value);
                                }
                            }
                        }
                    }
                }

                // Apply defaults from config
                if let Some(tool_defaults) = view.get_tool_defaults(&name) {
                    for (arg_name, default_value) in tool_defaults {
                        if !args.contains_key(arg_name) {
                            args.insert(arg_name.clone(), default_value.clone());
                        }
                    }
                }
            }

            // Legacy fallback: Use hardcoded defaults from config struct if no domain_view
            let defaults = &self.config.tool_defaults;

            if name == "check_eligibility" && !args.contains_key("collateral_variant") && !args.contains_key("gold_purity") {
                args.insert(
                    "collateral_variant".to_string(),
                    serde_json::json!(&defaults.default_gold_purity),
                );
            }

            if name == "calculate_savings" {
                if !args.contains_key("current_interest_rate") {
                    args.insert(
                        "current_interest_rate".to_string(),
                        serde_json::json!(defaults.default_competitor_rate),
                    );
                }
                if !args.contains_key("current_loan_amount") {
                    args.insert(
                        "current_loan_amount".to_string(),
                        serde_json::json!(defaults.default_loan_amount),
                    );
                }
                if !args.contains_key("remaining_tenure_months") {
                    args.insert(
                        "remaining_tenure_months".to_string(),
                        serde_json::json!(defaults.default_tenure_months),
                    );
                }
            }

            if (name == "find_branches" || name == "find_locations") && !args.contains_key("city") {
                args.insert(
                    "city".to_string(),
                    serde_json::json!(&defaults.default_city),
                );
            }

            // P4 FIX: Handle capture_lead tool arguments
            if name == "capture_lead" {
                // Map slot names to tool parameter names
                if args.contains_key("name") && !args.contains_key("customer_name") {
                    if let Some(v) = args.remove("name") {
                        args.insert("customer_name".to_string(), v);
                    }
                }
                if args.contains_key("phone") && !args.contains_key("phone_number") {
                    if let Some(v) = args.remove("phone") {
                        args.insert("phone_number".to_string(), v);
                    }
                }
                // Default interest level based on intent confidence
                if !args.contains_key("interest_level") {
                    let level = if intent.confidence > 0.8 {
                        "High"
                    } else {
                        "Medium"
                    };
                    args.insert("interest_level".to_string(), serde_json::json!(level));
                }
            }

            // P4 FIX: Handle schedule_appointment tool arguments
            if name == "schedule_appointment" {
                // Map slot names to tool parameter names
                if args.contains_key("name") && !args.contains_key("customer_name") {
                    if let Some(v) = args.remove("name") {
                        args.insert("customer_name".to_string(), v);
                    }
                }
                if args.contains_key("phone") && !args.contains_key("phone_number") {
                    if let Some(v) = args.remove("phone") {
                        args.insert("phone_number".to_string(), v);
                    }
                }
                if args.contains_key("date") && !args.contains_key("preferred_date") {
                    if let Some(v) = args.remove("date") {
                        args.insert("preferred_date".to_string(), v);
                    }
                }
                if args.contains_key("time") && !args.contains_key("preferred_time") {
                    if let Some(v) = args.remove("time") {
                        args.insert("preferred_time".to_string(), v);
                    }
                }
                if args.contains_key("branch") && !args.contains_key("branch_id") {
                    if let Some(v) = args.remove("branch") {
                        args.insert("branch_id".to_string(), v);
                    }
                }
            }

            let result = self
                .tools
                .execute(&name, serde_json::Value::Object(args))
                .await;

            let success = result.is_ok();
            let _ = self.event_tx.send(AgentEvent::ToolResult {
                name: name.to_string(),
                success,
            });

            match result {
                Ok(output) => {
                    // Extract text from output
                    let text = output
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            voice_agent_tools::mcp::ContentBlock::Text { text } => {
                                Some(text.clone())
                            }
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

    /// Call a tool by name using DST state for arguments (Phase 12 - proactive tool triggering)
    pub(super) async fn call_tool_by_name(
        &self,
        tool_name: &str,
        intent: &crate::intent::DetectedIntent,
    ) -> Result<Option<String>, AgentError> {
        let _ = self.event_tx.send(AgentEvent::ToolCall {
            name: tool_name.to_string(),
        });

        // Build arguments from DST state (more complete than just current intent slots)
        let mut args = serde_json::Map::new();

        // First, add slots from the current intent
        for (key, slot) in &intent.slots {
            if let Some(ref value) = slot.value {
                args.insert(key.clone(), serde_json::json!(value));
            }
        }

        // Then, enrich with DST state values that may have been collected over multiple turns
        {
            let dst = self.dialogue_state.read();
            let state = dst.state();

            // Map DST slot names to tool argument names
            // Uses generic get_slot_value() for domain-agnostic slot access
            if let Some(val) = state.customer_name() {
                args.entry("customer_name".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.phone_number() {
                args.entry("phone_number".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.location() {
                args.entry("city".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.get_slot_value("gold_weight") {
                args.entry("gold_weight".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.get_slot_value("gold_purity") {
                args.entry("gold_purity".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.get_slot_value("loan_amount") {
                args.entry("loan_amount".to_string())
                    .or_insert(serde_json::json!(val));
                args.entry("current_loan_amount".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.get_slot_value("current_lender") {
                args.entry("current_lender".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.get_slot_value("current_interest_rate") {
                args.entry("current_interest_rate".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.get_slot_value("loan_tenure") {
                args.entry("remaining_tenure_months".to_string())
                    .or_insert(serde_json::json!(val));
            }
        }

        // P16 FIX: Apply config-driven defaults
        if let Some(view) = self.domain_view.as_ref() {
            // Apply argument name mappings from config
            if let Some(arg_mapping) = view.get_argument_mapping(tool_name) {
                let keys: Vec<String> = args.keys().cloned().collect();
                for slot_name in keys {
                    if let Some(arg_name) = arg_mapping.get(&slot_name) {
                        if !args.contains_key(arg_name) {
                            if let Some(value) = args.remove(&slot_name) {
                                args.insert(arg_name.clone(), value);
                            }
                        }
                    }
                }
            }

            // Apply defaults from config
            if let Some(tool_defaults) = view.get_tool_defaults(tool_name) {
                for (arg_name, default_value) in tool_defaults {
                    if !args.contains_key(arg_name) {
                        args.insert(arg_name.clone(), default_value.clone());
                    }
                }
            }
        }

        // Legacy fallback: Use hardcoded defaults
        let defaults = &self.config.tool_defaults;

        if tool_name == "check_eligibility" && !args.contains_key("collateral_variant") && !args.contains_key("gold_purity") {
            args.insert(
                "collateral_variant".to_string(),
                serde_json::json!(&defaults.default_gold_purity),
            );
        }

        if tool_name == "calculate_savings" {
            if !args.contains_key("current_interest_rate") {
                args.insert(
                    "current_interest_rate".to_string(),
                    serde_json::json!(defaults.default_competitor_rate),
                );
            }
            if !args.contains_key("current_loan_amount") {
                args.insert(
                    "current_loan_amount".to_string(),
                    serde_json::json!(defaults.default_loan_amount),
                );
            }
            if !args.contains_key("remaining_tenure_months") {
                args.insert(
                    "remaining_tenure_months".to_string(),
                    serde_json::json!(defaults.default_tenure_months),
                );
            }
        }

        if (tool_name == "find_branches" || tool_name == "find_locations") && !args.contains_key("city") {
            args.insert(
                "city".to_string(),
                serde_json::json!(&defaults.default_city),
            );
        }

        if tool_name == "capture_lead" && !args.contains_key("interest_level") {
            // Default interest level to High for proactive capture
            args.insert("interest_level".to_string(), serde_json::json!("High"));
        }

        tracing::debug!(
            tool = tool_name,
            args = ?args,
            "Calling tool proactively with DST state"
        );

        let result = self
            .tools
            .execute(tool_name, serde_json::Value::Object(args))
            .await;

        let success = result.is_ok();
        let _ = self.event_tx.send(AgentEvent::ToolResult {
            name: tool_name.to_string(),
            success,
        });

        match result {
            Ok(output) => {
                let text = output
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        voice_agent_tools::mcp::ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                Ok(Some(text))
            }
            Err(e) => {
                tracing::warn!("Proactive tool error: {}", e);
                Ok(None)
            }
        }
    }
}
