//! MCP JSON-RPC Server Endpoint
//!
//! P2 FIX: Exposes tools via standard MCP JSON-RPC 2.0 protocol.
//! This allows external MCP clients to interact with the voice agent's tools.

use axum::{extract::State, Json};
use voice_agent_tools::{
    mcp::{methods, JsonRpcError, JsonRpcRequest, JsonRpcResponse, ToolCallParams},
    ToolExecutor,
};

use crate::state::AppState;

/// MCP JSON-RPC endpoint handler
///
/// POST /mcp
///
/// Handles MCP protocol requests:
/// - tools/list: List available tools with schemas
/// - tools/call: Execute a tool with arguments
pub async fn handle_mcp_request(
    State(state): State<AppState>,
    Json(request): Json<JsonRpcRequest>,
) -> Json<JsonRpcResponse> {
    // Validate JSON-RPC version
    if request.jsonrpc != "2.0" {
        return Json(JsonRpcResponse::error(
            request.id,
            JsonRpcError {
                code: -32600,
                message: "Invalid Request: jsonrpc must be \"2.0\"".to_string(),
                data: None,
            },
        ));
    }

    let response = match request.method.as_str() {
        methods::TOOLS_LIST => handle_tools_list(&state, &request),
        methods::TOOLS_CALL => handle_tools_call(&state, &request).await,
        _ => JsonRpcResponse::error(
            request.id.clone(),
            JsonRpcError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
                data: None,
            },
        ),
    };

    Json(response)
}

/// Handle tools/list request
fn handle_tools_list(state: &AppState, request: &JsonRpcRequest) -> JsonRpcResponse {
    let tools = state.tools.list_tools();

    let tool_schemas: Vec<serde_json::Value> = tools
        .into_iter()
        .map(|tool| {
            serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        })
        .collect();

    JsonRpcResponse::success(
        request
            .id
            .clone()
            .unwrap_or(voice_agent_tools::mcp::RequestId::Number(0)),
        serde_json::json!({
            "tools": tool_schemas
        }),
    )
}

/// Handle tools/call request
async fn handle_tools_call(state: &AppState, request: &JsonRpcRequest) -> JsonRpcResponse {
    // Parse call params
    let params: ToolCallParams = match &request.params {
        Some(p) => match serde_json::from_value(p.clone()) {
            Ok(params) => params,
            Err(e) => {
                return JsonRpcResponse::error(
                    request.id.clone(),
                    JsonRpcError {
                        code: -32602,
                        message: format!("Invalid params: {}", e),
                        data: None,
                    },
                );
            },
        },
        None => {
            return JsonRpcResponse::error(
                request.id.clone(),
                JsonRpcError {
                    code: -32602,
                    message: "Missing params for tools/call".to_string(),
                    data: None,
                },
            );
        },
    };

    // Execute the tool
    match state.tools.execute(&params.name, params.arguments).await {
        Ok(output) => {
            // Convert ToolOutput to MCP response format
            let content: Vec<serde_json::Value> = output
                .content
                .into_iter()
                .map(|block| match block {
                    voice_agent_tools::mcp::ContentBlock::Text { text } => {
                        serde_json::json!({
                            "type": "text",
                            "text": text
                        })
                    },
                    voice_agent_tools::mcp::ContentBlock::Image { data, mime_type } => {
                        serde_json::json!({
                            "type": "image",
                            "data": data,
                            "mimeType": mime_type
                        })
                    },
                    voice_agent_tools::mcp::ContentBlock::Resource { uri, mime_type } => {
                        serde_json::json!({
                            "type": "resource",
                            "resource": {
                                "uri": uri,
                                "mimeType": mime_type
                            }
                        })
                    },
                    voice_agent_tools::mcp::ContentBlock::Audio {
                        data,
                        mime_type,
                        sample_rate,
                        duration_ms,
                    } => {
                        serde_json::json!({
                            "type": "audio",
                            "data": data,
                            "mimeType": mime_type,
                            "sampleRate": sample_rate,
                            "durationMs": duration_ms
                        })
                    },
                })
                .collect();

            JsonRpcResponse::success(
                request
                    .id
                    .clone()
                    .unwrap_or(voice_agent_tools::mcp::RequestId::Number(0)),
                serde_json::json!({
                    "content": content,
                    "isError": output.is_error
                }),
            )
        },
        Err(tool_error) => JsonRpcResponse::from_tool_error(request.id.clone(), tool_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_rpc_request_parsing() {
        let json = r#"{
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        }"#;

        let request: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.jsonrpc, "2.0");
        assert_eq!(request.method, "tools/list");
    }

    #[test]
    fn test_tool_call_params_parsing() {
        let json = r#"{
            "name": "calculate_loan_eligibility",
            "arguments": {
                "gold_weight_grams": 100,
                "purity": "22K"
            }
        }"#;

        let params: ToolCallParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.name, "calculate_loan_eligibility");
    }
}
