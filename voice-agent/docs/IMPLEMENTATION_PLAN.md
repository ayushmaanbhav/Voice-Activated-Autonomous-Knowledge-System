# Implementation Plan: Small Model Agent Improvements

> Based on research in `SMALL_MODEL_AGENT_RESEARCH.md`
> Focus areas: 1, 3, 4, 5, 7, 8

## Current State Analysis

### What's Already Implemented
| Component | Status | File |
|-----------|--------|------|
| LLM Backend | Ollama with qwen2.5:1.5b-instruct-q4_K_M | `crates/llm/src/backend.rs` |
| Memory | Basic hierarchical (working/episodic/semantic) | `crates/agent/src/memory.rs` |
| Embeddings | ONNX-based, 384 dim | `crates/rag/src/embeddings.rs` |
| MCP Protocol | JSON-RPC 2.0 compliant | `crates/tools/src/mcp.rs` |
| DST | Basic intent detection only | `crates/agent/src/intent.rs` |
| Context Compression | NOT IMPLEMENTED | - |

---

## Phase 0: Verification (Priority: Critical)

### 0.1 Verify Qwen2.5-1.5B Q4_K_M Configuration
**Goal**: Confirm model is correctly loaded and quantized

**Tasks**:
- [ ] Verify Ollama has model: `ollama list | grep qwen2.5:1.5b`
- [ ] Check actual quantization: `ollama show qwen2.5:1.5b-instruct-q4_K_M --modelfile`
- [ ] Benchmark baseline: measure tokens/sec, TTFT, memory usage
- [ ] Test KV cache reuse (existing implementation)
- [ ] Document baseline metrics in `benchmarks/baseline_metrics.md`

**Verification Script**:
```bash
# Check model details
ollama show qwen2.5:1.5b-instruct-q4_K_M

# Test inference speed
time curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:1.5b-instruct-q4_K_M",
  "prompt": "What is a gold loan?",
  "stream": false
}' | jq .eval_count,.eval_duration
```

**Success Criteria**:
- Model uses Q4_K_M quantization (4-bit, K-quants, medium)
- Inference: >15 tokens/sec on CPU, >50 tokens/sec on GPU
- TTFT: <500ms for simple prompts
- Memory: <2GB VRAM/RAM for model

---

## Phase 1: Small Language Model Optimization (Section 1)

### 1.1 Qwen2.5 Best Practices Implementation
**Based on**: Qwen2.5 Technical Report, Function Calling docs

**Tasks**:
- [ ] Review current prompt template against Qwen2.5 chat template
- [ ] Implement Qwen-style function calling format (not ReAct stopwords)
- [ ] Add `think: false` parameter for non-reasoning mode (already done)
- [ ] Consider upgrade path to Qwen3-1.7B when available

**Files to Modify**:
- `crates/llm/src/prompt.rs` - Prompt formatting
- `crates/llm/src/backend.rs` - Already has `think: false`
- `crates/agent/src/agent.rs` - Tool calling integration

### 1.2 Token Estimation for Hindi/Multilingual
**Current**: Simple 4-chars-per-token estimate
**Research**: Hindi/Devanagari needs ~2 chars per token

**Already Implemented** in `backend.rs:128-149`:
```rust
// Count Devanagari characters (U+0900 to U+097F)
let devanagari_count = text.chars()
    .filter(|c| ('\u{0900}'..='\u{097F}').contains(c))
    .count();
```

**Tasks**:
- [ ] Verify token estimation accuracy with actual Qwen2.5 tokenizer
- [ ] Add benchmarks comparing estimated vs actual token counts
- [ ] Consider integrating actual tokenizer for precise counts

---

## Phase 2: Memory & Context Management (Section 3)

### 2.1 MemGPT-Style Memory Architecture
**Research**: MemGPT paper, A-MEM paper

**Current Architecture**:
```
working_memory (Vec<MemoryEntry>)  → Recent turns
episodic_memory (VecDeque<EpisodicSummary>) → Summaries
semantic_memory (HashMap<String, SemanticFact>) → Facts
```

**Target Architecture (MemGPT-inspired)**:
```
┌─────────────────────────────────────────────────────┐
│                    Main Context                      │
│  ┌─────────────┬─────────────────┬───────────────┐  │
│  │   System    │  Working Context │   FIFO Queue  │  │
│  │ Instructions│   (Core Memory)  │  (Recent 4-6) │  │
│  └─────────────┴─────────────────┴───────────────┘  │
└─────────────────────────────────────────────────────┘
                        ↕ Function Calls
┌─────────────────────────────────────────────────────┐
│                 External Context                     │
│  ┌──────────────────┬───────────────────────────┐   │
│  │ Archival Storage │    Recall Storage         │   │
│  │ (Vector DB/Long) │  (Conversation Search)    │   │
│  └──────────────────┴───────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

**Tasks**:
- [ ] Implement `CoreMemory` struct with:
  - `human_block`: Customer info, preferences, facts
  - `persona_block`: Agent personality, guidelines
- [ ] Add memory manipulation functions:
  - `core_memory_append(key, value)`
  - `core_memory_replace(key, old, new)`
  - `archival_memory_insert(content)`
  - `archival_memory_search(query, k)`
  - `conversation_search(query, k)`
- [ ] Implement automatic memory compaction when above high watermark
- [ ] Add memory persistence to Qdrant (archival) and SQLite (recall)

**Files to Create/Modify**:
- `crates/agent/src/memory.rs` - Major refactor
- `crates/agent/src/memory/core.rs` - New: Core memory blocks
- `crates/agent/src/memory/archival.rs` - New: Vector DB integration
- `crates/agent/src/memory/recall.rs` - New: Conversation search

### 2.2 A-MEM Zettelkasten-Style Linking
**Research**: A-MEM paper (NeurIPS 2025)

**Tasks**:
- [ ] Add memory note structure:
  ```rust
  struct MemoryNote {
      id: Uuid,
      content: String,
      context_description: String,
      keywords: Vec<String>,
      tags: Vec<String>,
      links: Vec<Uuid>,  // Related notes
      created_at: DateTime<Utc>,
      updated_at: DateTime<Utc>,
  }
  ```
- [ ] Implement dynamic linking on memory addition
- [ ] Add relevance-based retrieval with link traversal

### 2.3 Context Window Watermarks
**Already Partially Implemented**:
- `high_watermark_tokens: 3000`
- `low_watermark_tokens: 2000`
- `max_context_tokens: 4000`
- `cleanup_to_watermark()` function exists

**Tasks**:
- [ ] Add proactive summarization before hitting high watermark
- [ ] Implement sliding window with overlap for long conversations
- [ ] Add metrics for context utilization monitoring

---

## Phase 3: Context Compression (Section 4)

### 3.1 Implement LLMLingua-Style Compression
**Research**: LLMLingua, LongLLMLingua papers

**Goal**: Compress context by 4-20x while maintaining quality

**Approach Options**:

#### Option A: LLM-Based Summarization (Simpler)
Use the existing LLM to summarize older context:
```rust
async fn compress_context(&self, entries: &[MemoryEntry]) -> String {
    let prompt = "Summarize this conversation, keeping key facts...";
    self.llm.generate(prompt).await
}
```
**Pro**: No new dependencies. **Con**: Uses LLM tokens.

#### Option B: Token-Level Compression (LLMLingua-style)
- Train/use small model to identify unimportant tokens
- Remove low-importance tokens while preserving meaning
**Pro**: Faster, no LLM calls. **Con**: Needs compression model.

#### Option C: Semantic Chunking + Selective Inclusion
- Chunk context semantically
- Embed chunks, select most relevant for current query
**Pro**: Works well with RAG. **Con**: May miss global context.

**Recommended**: Start with Option A (already partially implemented in `summarize_pending_async`), then add Option C for RAG integration.

**Tasks**:
- [ ] Enhance `summarize_pending_async()` with better prompts
- [ ] Add compression ratio tracking
- [ ] Implement selective context injection based on query relevance
- [ ] Add config for compression aggressiveness

**Files to Modify**:
- `crates/agent/src/memory.rs` - Enhance summarization
- `crates/rag/src/compressor.rs` - Add semantic selection

### 3.2 RAG Context Compaction
**Current**: `crates/rag/src/compressor.rs` exists but may be basic

**Tasks**:
- [ ] Implement Anthropic-style contextual chunk descriptions
- [ ] Add chunk relevance scoring
- [ ] Implement "late chunking" approach for embeddings

---

## Phase 4: Vector DB & Embeddings (Section 5)

### 4.1 Upgrade to e5-small Embeddings
**Research**: e5-small has 100% Top-5 accuracy, 16ms latency

**Current Config** (default.yaml):
```yaml
models:
  embeddings: "models/embeddings/e5-multilingual.onnx"
rag:
  vector_dim: 1024  # qwen3-embedding:0.6b dimension
```

**Issue**: Config says 1024 dim but may be using 384 dim model

**Tasks**:
- [ ] Verify actual embedding model and dimensions
- [ ] Download e5-small-v2 ONNX model if not present
- [ ] Update `EmbeddingConfig` to match model:
  ```rust
  EmbeddingConfig {
      embedding_dim: 384,  // e5-small
      output_name: "sentence_embedding".to_string(),
      // ...
  }
  ```
- [ ] Re-index Qdrant collection with correct dimensions
- [ ] Benchmark embedding latency (target: <20ms)

**Files to Modify**:
- `config/default.yaml` - Fix vector_dim
- `crates/rag/src/embeddings.rs` - Verify output tensor name

### 4.2 Semantic Chunking Implementation
**Research**: 30-50% higher retrieval precision with semantic chunking

**Current**: Unknown chunking strategy

**Tasks**:
- [ ] Implement sentence-based chunking with overlap
- [ ] Add semantic boundary detection (topic shifts)
- [ ] Configure 10-20% overlap between chunks
- [ ] Add chunk metadata (source, position, surrounding context)

**Files to Create**:
- `crates/rag/src/chunker.rs` - Semantic chunking module

### 4.3 Hybrid Retrieval (Dense + Sparse)
**Current Config**:
```yaml
dense_top_k: 20
sparse_top_k: 20
dense_weight: 0.7
```

**Tasks**:
- [ ] Verify BM25/sparse retrieval is working
- [ ] Implement Reciprocal Rank Fusion (RRF) properly
- [ ] Add domain-specific stopwords for Hindi/gold loan
- [ ] Benchmark hybrid vs dense-only retrieval

### 4.4 Late Chunking for Embeddings
**Research**: Jina AI late chunking paper

**Tasks**:
- [ ] Implement late chunking:
  1. Embed full document with long-context model
  2. Extract per-chunk embeddings from positions
- [ ] Compare retrieval quality vs standard chunking

---

## Phase 5: Dialogue State Tracking (Section 7)

### 5.1 Implement Domain-Slot DST
**Research**: LDST paper, ACL 2024 DST paper

**Current**: Basic intent detection in `crates/agent/src/intent.rs`

**Gold Loan Domain Slots**:
```rust
struct GoldLoanDialogueState {
    // Customer Info
    customer_name: Option<String>,
    phone_number: Option<String>,
    preferred_language: Language,

    // Loan Requirements
    gold_weight_grams: Option<f32>,
    gold_purity: Option<GoldPurity>,  // 24K, 22K, 18K
    desired_loan_amount: Option<f64>,
    loan_tenure_months: Option<u32>,

    // Intent Tracking
    primary_intent: Intent,
    secondary_intents: Vec<Intent>,

    // Stage Tracking
    current_stage: ConversationStage,
    completed_stages: Vec<ConversationStage>,

    // Extracted Information
    objections: Vec<Objection>,
    competitor_mentions: Vec<String>,
    urgency_level: UrgencyLevel,

    // Confirmation Status
    slots_confirmed: HashSet<String>,
    slots_pending: HashSet<String>,
}
```

**Tasks**:
- [ ] Define slot schema for gold loan domain
- [ ] Implement slot extraction from utterances
- [ ] Add slot confirmation tracking
- [ ] Implement state update rules
- [ ] Add dialogue state persistence

**Files to Create/Modify**:
- `crates/agent/src/dst.rs` - New: Dialogue State Tracking
- `crates/agent/src/dst/slots.rs` - Slot definitions
- `crates/agent/src/dst/extractor.rs` - NLU slot extraction

### 5.2 Multi-Turn Tracking
**Tasks**:
- [ ] Track slot values across turns
- [ ] Handle slot corrections ("actually, it's 50 grams, not 40")
- [ ] Implement slot carryover for confirmed values
- [ ] Add confidence tracking per slot

### 5.3 DST-Guided Response Generation
**Tasks**:
- [ ] Use dialogue state to guide LLM prompts
- [ ] Add missing slot prompting
- [ ] Implement clarification question generation

---

## Phase 6: MCP Tool Use (Section 8)

### 6.1 Enhance Tool Schema Validation
**Research**: MCP-Universe benchmark, Berkeley BFCL

**Current**: Basic MCP protocol in `crates/tools/src/mcp.rs`

**Already Implemented** (P3 FIX):
- Type validation
- Enum value validation
- Range validation

**Tasks**:
- [ ] Add comprehensive schema validation tests
- [ ] Implement nested object validation
- [ ] Add array item validation
- [ ] Test against MCP-Universe benchmark scenarios

### 6.2 Qwen2.5 Function Calling Format
**Research**: Qwen Function Calling docs

**Warning from Research**:
> "For reasoning models like Qwen3, it is not recommended to use tool call templates based on stopwords (such as ReAct)"

**Current Format**: Unknown - need to verify

**Tasks**:
- [ ] Review current tool call prompt format
- [ ] Implement Qwen2.5 native function calling format:
  ```
  <|im_start|>system
  You are a helpful assistant with access to the following tools:
  {tools_json}
  <|im_end|>
  <|im_start|>user
  {query}
  <|im_end|>
  <|im_start|>assistant
  ```
- [ ] Parse tool call responses correctly
- [ ] Handle multi-tool calls in single response

**Files to Modify**:
- `crates/llm/src/prompt.rs` - Add tool formatting
- `crates/agent/src/agent.rs` - Tool call parsing

### 6.3 Tool Execution Pipeline
**Tasks**:
- [ ] Add tool execution timeout handling
- [ ] Implement progress reporting for long tools
- [ ] Add tool result caching for idempotent operations
- [ ] Implement tool retry logic

### 6.4 Gold Loan Specific Tools
**Existing Tools** (from `crates/tools/src/gold_loan.rs`):
- Loan calculation
- Interest rate lookup
- Branch finder

**Tasks**:
- [ ] Add appointment scheduling tool
- [ ] Add document checklist tool
- [ ] Add competitor comparison tool
- [ ] Add SMS notification tool (via MCP server)

---

## Implementation Timeline

### Week 1: Foundation & Verification
- [ ] Phase 0: Verify Qwen2.5 model configuration
- [ ] Phase 4.1: Verify/fix embedding dimensions
- [ ] Establish baseline benchmarks

### Week 2: Memory Architecture
- [ ] Phase 2.1: MemGPT-style memory refactor
- [ ] Phase 2.3: Context watermark improvements

### Week 3: Compression & Embeddings
- [ ] Phase 3.1: Context compression
- [ ] Phase 4.2: Semantic chunking
- [ ] Phase 4.3: Hybrid retrieval verification

### Week 4: DST & Tools
- [ ] Phase 5.1-5.2: Dialogue State Tracking
- [ ] Phase 6.1-6.2: Tool schema & format

### Week 5: Integration & Testing
- [ ] Phase 5.3: DST-guided responses
- [ ] Phase 6.3-6.4: Tool execution & domain tools
- [ ] End-to-end testing
- [ ] Performance benchmarks

---

## Success Metrics

### Latency
| Metric | Baseline | Target |
|--------|----------|--------|
| TTFT | ? | <300ms |
| Total Response | ? | <1s |
| Embedding | ? | <20ms |
| RAG Retrieval | ? | <100ms |

### Quality
| Metric | Baseline | Target |
|--------|----------|--------|
| Intent Accuracy | ? | >90% |
| Slot Extraction | ? | >85% |
| Tool Call Success | ? | >95% |
| Hallucination Rate | ? | <5% |

### Memory
| Metric | Baseline | Target |
|--------|----------|--------|
| Context Utilization | ? | <80% of limit |
| Memory Overhead | ? | <100MB per session |
| Compression Ratio | N/A | >4x |

---

## Testing Strategy

### Unit Tests
- Memory management functions
- Slot extraction
- Tool schema validation
- Context compression

### Integration Tests
- Full conversation flow with DST
- RAG retrieval accuracy
- Tool execution pipeline

### Benchmark Tests
- Latency measurements
- Token counting accuracy
- Memory efficiency

---

## Dependencies to Add

```toml
# Cargo.toml additions

# For better Hindi tokenization
unicode-segmentation = "1.10"  # Already added

# For semantic chunking
punkt = "0.1"  # Sentence tokenization

# For DST slot extraction
regex = "1.10"

# For improved embeddings (if using Hugging Face)
# tokenizers = "0.15"  # Already present
```

---

## Notes

### Key Research Insights to Remember

1. **Context Rot**: Accuracy degrades beyond 70-80% of context window
2. **Compression**: LLMLingua achieves 20x compression with minimal loss
3. **Embeddings**: e5-small beats larger models for retrieval
4. **Memory**: A-MEM's Zettelkasten linking improves recall
5. **Tools**: Avoid ReAct stopwords with reasoning models
6. **DST**: Domain-slot instruction tuning beats zero-shot

### Risks

1. **Model Upgrade**: Qwen3 may have different tool calling format
2. **Memory Bloat**: Need aggressive compaction for long sessions
3. **Hindi Accuracy**: Tokenization affects context estimation

### Open Questions

1. Should we use Ollama's native embeddings or separate ONNX model?
2. Is Qdrant sparse index properly configured?
3. What's the actual compression ratio of current summarization?
