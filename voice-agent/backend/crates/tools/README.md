# voice-agent-tools

> MCP-compatible tool interface with domain-specific tools

---

## Overview

The `tools` crate implements the Model Context Protocol (MCP) for LLM tool calling:

- **MCP Protocol** - JSON-RPC based tool interface
- **Tool Registry** - Dynamic tool registration
- **Gold Loan Tools** - Domain-specific business tools
- **Integrations** - CRM, Calendar, SMS connectors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOOLS ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌─────────────────────┐                              │
│                        │   Tool Registry     │                              │
│                        │   (Configurable)    │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                         │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│         ▼                         ▼                         ▼              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Gold Loan     │    │   MCP Protocol  │    │  Integrations   │        │
│  │     Tools       │    │   (JSON-RPC)    │    │  (CRM, etc.)    │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│         │                         │                         │              │
│         ▼                         ▼                         ▼              │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│  │ Calculator  │         │ Tool Schema │         │ Lead Capture│          │
│  │ Branch Find │         │ Validation  │         │ Appointment │          │
│  │ Eligibility │         │ JSON-RPC    │         │ SMS Sender  │          │
│  └─────────────┘         └─────────────┘         └─────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Available Tools

| Tool | Purpose | Parameters |
|------|---------|------------|
| `calculate_savings` | Calculate savings vs competitor | current_lender, loan_amount, tenure |
| `check_eligibility` | Check loan eligibility | age, gold_weight, gold_purity |
| `find_branch` | Find nearest branch | city, pincode, doorstep_required |
| `schedule_appointment` | Book appointment | customer_name, phone, datetime, branch_id |
| `capture_lead` | Capture lead info | name, phone, loan_amount, notes |
| `get_gold_price` | Current gold price | purity (optional) |
| `send_sms` | Send SMS notification | phone, template, params |
| `escalate_to_human` | Transfer to human agent | reason, priority |

---

## Basic Usage

```rust
use voice_agent_tools::{ToolRegistry, ToolExecutor};
use voice_agent_tools::gold_loan::{
    SavingsCalculatorTool, BranchLocatorTool, EligibilityCheckTool
};

// Create registry with tools
let mut registry = ToolRegistry::new();
registry.register(SavingsCalculatorTool::new(&domain_config));
registry.register(BranchLocatorTool::new(&branch_config));
registry.register(EligibilityCheckTool::new(&eligibility_config));

// Execute tool call from LLM
let result = registry.execute(
    "calculate_savings",
    json!({
        "current_lender": "Muthoot",
        "loan_amount": 500000,
        "tenure_months": 12
    }),
).await?;

println!("{}", result); // JSON response
```

---

## Tool Definitions

### Savings Calculator

```rust
use voice_agent_tools::gold_loan::SavingsCalculatorTool;

let tool = SavingsCalculatorTool::new(&domain_config);

let result = tool.execute(json!({
    "current_lender": "Muthoot",
    "loan_amount": 500000,
    "tenure_months": 12
})).await?;

// Result:
// {
//   "current_rate": 18.0,
//   "kotak_rate": 10.5,
//   "monthly_savings": 3125,
//   "annual_savings": 37500,
//   "tenure_savings": 37500
// }
```

### Branch Locator

```rust
use voice_agent_tools::gold_loan::BranchLocatorTool;

let tool = BranchLocatorTool::new(&branch_config);

let result = tool.execute(json!({
    "city": "Mumbai",
    "doorstep_required": true
})).await?;

// Result:
// {
//   "branches": [
//     {
//       "id": "MUM001",
//       "name": "Andheri West Branch",
//       "address": "Shop 12, Andheri West...",
//       "phone": "022-12345678",
//       "doorstep_available": true
//     }
//   ]
// }
```

---

## MCP Protocol

### Tool Schema (JSON Schema)

```rust
use voice_agent_tools::mcp::{ToolSchema, InputSchema, PropertySchema};

let schema = ToolSchema {
    name: "calculate_savings".into(),
    description: "Calculate savings when switching from competitor".into(),
    input_schema: InputSchema {
        properties: vec![
            PropertySchema {
                name: "current_lender".into(),
                schema_type: "string".into(),
                description: "Current lender (Muthoot, Manappuram, IIFL)".into(),
                required: true,
                enum_values: Some(vec!["Muthoot", "Manappuram", "IIFL", "Other"]),
            },
            PropertySchema {
                name: "loan_amount".into(),
                schema_type: "number".into(),
                description: "Loan amount in INR".into(),
                required: true,
                minimum: Some(10000.0),
                maximum: Some(25000000.0),
            },
        ],
    },
};
```

### JSON-RPC Interface

```rust
use voice_agent_tools::mcp::{JsonRpcRequest, JsonRpcResponse};

// Incoming request
let request = JsonRpcRequest {
    jsonrpc: "2.0".into(),
    id: RequestId::Number(1),
    method: "tools/call".into(),
    params: Some(json!({
        "name": "calculate_savings",
        "arguments": {
            "current_lender": "Muthoot",
            "loan_amount": 500000
        }
    })),
};

// Execute
let response = mcp_server.handle(request).await?;
```

---

## Tool Registry

### Configuration-Based

```rust
use voice_agent_tools::registry::create_registry_with_config;

let registry = create_registry_with_config(&config)?;

// Available tools based on config
let tools = registry.list_tools();
for tool in tools {
    println!("{}: {}", tool.name, tool.description);
}
```

### With Integrations

```rust
use voice_agent_tools::registry::create_registry_with_integrations;
use voice_agent_tools::integrations::{CrmIntegration, CalendarIntegration};

let registry = create_registry_with_integrations(
    &config,
    &IntegrationConfig {
        crm: Some(Box::new(SalesforceCrm::new(&crm_config)?)),
        calendar: Some(Box::new(GoogleCalendar::new(&calendar_config)?)),
    },
)?;
```

---

## Integrations

### CRM Integration

```rust
use voice_agent_tools::integrations::{CrmIntegration, CrmLead};

#[async_trait]
impl CrmIntegration for SalesforceCrm {
    async fn create_lead(&self, lead: CrmLead) -> Result<String> {
        // Create lead in Salesforce
    }

    async fn update_lead(&self, id: &str, lead: CrmLead) -> Result<()> {
        // Update existing lead
    }
}
```

### Calendar Integration

```rust
use voice_agent_tools::integrations::{CalendarIntegration, Appointment};

#[async_trait]
impl CalendarIntegration for GoogleCalendar {
    async fn get_available_slots(&self, date: Date) -> Result<Vec<TimeSlot>> {
        // Get available slots
    }

    async fn book_appointment(&self, appointment: Appointment) -> Result<String> {
        // Book appointment
    }
}
```

---

## Configuration

```yaml
# config/tools.yaml
tools:
  calculate_savings:
    enabled: true
    competitor_rates:
      muthoot: 18.0
      manappuram: 19.0
      iifl: 17.5

  find_branch:
    enabled: true
    data_source: "config/domain.yaml"

  schedule_appointment:
    enabled: true
    min_advance_hours: 24
    max_advance_days: 7

  capture_lead:
    enabled: true
    crm_integration: "salesforce"

  send_sms:
    enabled: true
    provider: "simulated"  # or "twilio", "gupshup"

integrations:
  crm:
    type: "salesforce"
    endpoint: "https://api.salesforce.com"
    # api_key via VOICE_AGENT__INTEGRATIONS__CRM__API_KEY

  calendar:
    type: "google"
    # credentials via environment
```
