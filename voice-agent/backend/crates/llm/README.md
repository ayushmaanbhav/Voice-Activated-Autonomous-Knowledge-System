# voice-agent-llm

> Multi-provider LLM integration with speculative execution

---

## Overview

The `llm` crate provides unified access to multiple LLM providers:

- **Ollama** - Local LLM serving (default)
- **Claude** - Anthropic API with native tool_use
- **OpenAI** - OpenAI API compatible
- **Speculative Decoding** - SLM + LLM for faster responses

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LLM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌─────────────────────┐                              │
│                        │    LLM Factory      │                              │
│                        │  (Provider Router)  │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                         │
│         ┌─────────────────────────┼─────────────────────────┐              │
│         │                         │                         │              │
│         ▼                         ▼                         ▼              │
│  ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│  │   Ollama    │          │   Claude    │          │   OpenAI    │        │
│  │   Backend   │          │   Backend   │          │   Backend   │        │
│  └─────────────┘          └─────────────┘          └─────────────┘        │
│                                   │                                         │
│                                   ▼                                         │
│                        ┌─────────────────────┐                              │
│                        │   Speculative       │                              │
│                        │    Executor         │                              │
│                        └──────────┬──────────┘                              │
│                                   │                                         │
│                    ┌──────────────┼──────────────┐                         │
│                    ▼                             ▼                         │
│             ┌─────────────┐             ┌─────────────┐                    │
│             │  SLM Draft  │─ ─ ─ ─ ─ ─ ▶│ LLM Verify  │                    │
│             │  (1.5B Q4)  │             │  (7B Q4)    │                    │
│             └─────────────┘             └─────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Basic Usage

```rust
use voice_agent_llm::{LlmFactory, LlmProvider, LlmProviderConfig};

// Create LLM instance
let llm = LlmFactory::create(LlmProviderConfig {
    provider: LlmProvider::Ollama,
    model: "qwen2.5:7b".into(),
    endpoint: "http://localhost:11434".into(),
    temperature: 0.7,
    max_tokens: 200,
})?;

// Generate response
let response = llm.generate(&[
    Message::system("You are a helpful assistant."),
    Message::user("What are the interest rates for gold loans?"),
]).await?;

println!("{}", response.text);
```

---

## Streaming

```rust
use voice_agent_llm::streaming::StreamingGenerator;

let stream = llm.generate_stream(&messages).await;

while let Some(chunk) = stream.next().await {
    match chunk? {
        StreamChunk::Token(text) => print!("{}", text),
        StreamChunk::ToolCall(call) => execute_tool(call),
        StreamChunk::Done(reason) => break,
    }
}
```

---

## Providers

### Ollama (Local)

```rust
use voice_agent_llm::backend::OllamaBackend;

let backend = OllamaBackend::new(OllamaConfig {
    endpoint: "http://localhost:11434".into(),
    model: "qwen2.5:7b".into(),
})?;
```

### Claude (Anthropic)

```rust
use voice_agent_llm::claude::{ClaudeBackend, ClaudeConfig, ClaudeModel};

let backend = ClaudeBackend::new(ClaudeConfig {
    api_key: std::env::var("ANTHROPIC_API_KEY")?,
    model: ClaudeModel::Claude3Haiku,
})?;

// Native tool_use support (not text-based parsing)
let response = backend.generate_with_tools(&messages, &tools).await?;
```

### OpenAI

```rust
use voice_agent_llm::backend::{OpenAIBackend, OpenAIConfig};

let backend = OpenAIBackend::new(OpenAIConfig {
    api_key: std::env::var("OPENAI_API_KEY")?,
    model: "gpt-4o".into(),
})?;
```

---

## Speculative Execution

For lower latency, run a small model (SLM) in parallel with a large model (LLM):

```rust
use voice_agent_llm::speculative::{SpeculativeExecutor, SpeculativeMode};

let executor = SpeculativeExecutor::new(SpeculativeConfig {
    slm: slm_backend,    // Qwen 1.5B
    llm: llm_backend,    // Qwen 7B
    mode: SpeculativeMode::SlmFirst,
})?;

let result = executor.generate(&messages).await?;
// Uses SLM if sufficient, falls back to LLM for complex queries
```

### Speculative Modes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SPECULATIVE EXECUTION MODES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODE             BEHAVIOR                         USE CASE                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  SlmFirst         SLM drafts response              Simple queries (70%)    │
│                   LLM verifies if needed           Lower latency           │
│                                                                             │
│  RaceParallel     Both run in parallel             Complex queries         │
│                   First to finish wins             Minimum latency         │
│                                                                             │
│  HybridStreaming  SLM streams tokens               Best quality            │
│                   LLM corrects in background       Low perceived latency   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tool Calling

```rust
use voice_agent_llm::prompt::{ToolDefinition, ToolBuilder};

let tools = vec![
    ToolBuilder::new("calculate_savings")
        .description("Calculate savings when switching from competitor")
        .add_param("current_lender", "string", "Current lender name")
        .add_param("loan_amount", "number", "Loan amount in INR")
        .build(),

    ToolBuilder::new("find_branch")
        .description("Find nearest Kotak branch")
        .add_param("city", "string", "City name")
        .add_param("pincode", "string", "Pincode (optional)")
        .build(),
];

let response = llm.generate_with_tools(&messages, &tools).await?;

for call in response.tool_calls {
    println!("Tool: {} Args: {:?}", call.name, call.arguments);
}
```

---

## Prompt Building

```rust
use voice_agent_llm::prompt::PromptBuilder;

let prompt = PromptBuilder::new()
    .with_persona(&persona_config)
    .with_system_prompt(&domain_config.system_prompt)
    .with_context(&rag_documents)
    .with_memory(&conversation_memory)
    .with_tools(&available_tools)
    .build();

let response = llm.generate(&prompt.messages).await?;
```

---

## Configuration

```yaml
agent:
  llm:
    provider: "ollama"
    model: "qwen2.5:7b"
    endpoint: "http://localhost:11434"
    temperature: 0.7
    max_tokens: 200

    # Speculative execution
    speculative:
      enabled: true
      mode: "slm_first"
      slm_model: "qwen2.5:1.5b"
      verification_threshold: 0.8

    # Fallback chain
    fallback:
      - provider: "ollama"
        model: "qwen2.5:7b"
      - provider: "claude"
        model: "claude-3-haiku"
```
