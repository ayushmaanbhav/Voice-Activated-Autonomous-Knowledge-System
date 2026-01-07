//! Tool Calling Methods for DomainAgent
//!
//! This module contains tool-related functionality including:
//! - Intent-based tool invocation
//! - DST-enriched tool calls
//! - Tool argument mapping and defaults

use super::DomainAgent;
use crate::agent_config::AgentEvent;
use crate::AgentError;
use voice_agent_tools::ToolExecutor;

impl DomainAgent {
    /// Maybe call a tool based on intent
    pub(super) async fn maybe_call_tool(
        &self,
        intent: &crate::intent::DetectedIntent,
    ) -> Result<Option<String>, AgentError> {
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
            // P4 FIX: Add intent mappings for CRM/Calendar integrations
            "capture_lead" | "interested" | "callback_request" => {
                // Capture lead when customer shows interest
                if intent.slots.contains_key("customer_name")
                    || intent.slots.contains_key("phone_number")
                {
                    Some("capture_lead")
                } else {
                    None
                }
            }
            "schedule_appointment" | "book_appointment" | "visit_branch" => {
                // Schedule appointment when customer wants to visit
                if intent.slots.contains_key("preferred_date")
                    || intent.slots.contains_key("branch_id")
                {
                    Some("schedule_appointment")
                } else {
                    // If no specific date/branch, first find branches
                    Some("find_branches")
                }
            }
            // P1 FIX: Add missing tool intent mappings
            "gold_price" | "check_gold_price" | "price_inquiry" | "current_rate" => {
                // Gold price inquiry - no required slots
                Some("get_gold_price")
            }
            "escalate" | "human_agent" | "speak_to_person" | "talk_to_human" | "real_person" => {
                // Escalation to human agent - no required slots
                Some("escalate_to_human")
            }
            "send_sms" | "send_message" | "text_me" | "send_details" | "sms_info" => {
                // Send SMS - phone_number slot is optional (can use customer's registered number)
                Some("send_sms")
            }
            _ => None,
        };

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

            // P1 FIX: Use configurable defaults instead of hardcoded values
            let defaults = &self.config.tool_defaults;

            if name == "check_eligibility" && !args.contains_key("gold_purity") {
                args.insert(
                    "gold_purity".to_string(),
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

            if name == "find_branches" && !args.contains_key("city") {
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
                .execute(name, serde_json::Value::Object(args))
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
            if let Some(val) = state.gold_weight_grams() {
                args.entry("gold_weight".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.gold_purity() {
                args.entry("gold_purity".to_string())
                    .or_insert(serde_json::json!(val.to_string()));
            }
            if let Some(val) = state.loan_amount() {
                args.entry("loan_amount".to_string())
                    .or_insert(serde_json::json!(val));
                args.entry("current_loan_amount".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.current_lender() {
                args.entry("current_lender".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.current_interest_rate() {
                args.entry("current_interest_rate".to_string())
                    .or_insert(serde_json::json!(val));
            }
            if let Some(val) = state.loan_tenure() {
                args.entry("remaining_tenure_months".to_string())
                    .or_insert(serde_json::json!(val));
            }
        }

        // Apply defaults based on tool type
        let defaults = &self.config.tool_defaults;

        if tool_name == "check_eligibility" && !args.contains_key("gold_purity") {
            args.insert(
                "gold_purity".to_string(),
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

        if tool_name == "find_branches" && !args.contains_key("city") {
            args.insert(
                "city".to_string(),
                serde_json::json!(&defaults.default_city),
            );
        }

        if tool_name == "capture_lead" {
            // Default interest level to High for proactive capture
            if !args.contains_key("interest_level") {
                args.insert("interest_level".to_string(), serde_json::json!("High"));
            }
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
