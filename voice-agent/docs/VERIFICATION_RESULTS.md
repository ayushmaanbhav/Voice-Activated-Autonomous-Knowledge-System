# Model Verification Results

> Verified: January 2025

## Qwen2.5-1.5B Q4_K_M

### Model Information
| Property | Value | Status |
|----------|-------|--------|
| Model ID | qwen2.5:1.5b-instruct-q4_K_M | ✓ |
| Architecture | qwen2 | ✓ |
| Parameters | 1.5B | ✓ |
| Quantization | **Q4_K_M** | ✓ |
| Context Length | 32,768 tokens | ✓ |
| Embedding Length | 1536 | - |
| Tool Calling | Supported | ✓ |

### Benchmark Results
```
Test: "What is a gold loan? Answer in one sentence."
- Response: "A gold loan is a type of personal financing product that allows
  individuals to borrow money against the value of their gold jewelry or
  certificates."
- Tokens generated: 27
- Eval duration: 969ms
- Speed: 27.85 tokens/sec (CPU)
- Total time: 1.44s
```

### Q4_K_M Quantization Details
- **Q4**: 4-bit quantization
- **K**: K-quants (optimal grouping)
- **M**: Medium precision (balance of size/quality)
- Memory: ~986 MB on disk
- Expected quality loss: <1% from FP16

---

## Embedding Model: Qwen3-Embedding:0.6B

### Model Information
| Property | Value | Status |
|----------|-------|--------|
| Model ID | qwen3-embedding:0.6b | ✓ |
| Architecture | qwen3 | ✓ |
| Parameters | 595.78M | ✓ |
| Quantization | Q8_0 | ✓ |
| Embedding Dimension | **1024** | ✓ |
| Context Length | 32,768 tokens | ✓ |

### Config Match
```yaml
# config/default.yaml
rag:
  vector_dim: 1024  # ✓ Matches model
```

### Instruction Format (Implemented)
```
# For queries (better retrieval):
Instruct: Given a user query about banking products or gold loans, retrieve
relevant information that answers the query
Query:<user query>

# For documents:
<plain text, no instruction>
```

---

## Configuration Summary

### Current LLM Config (`config/default.yaml`)
```yaml
agent:
  model: "qwen2.5:1.5b-instruct-q4_K_M"  # ✓ Correct
  temperature: 0.7                         # ✓ Good for conversation
  max_tokens: 200                          # ✓ Short responses
  llm:
    provider: "ollama"                     # ✓
    model: "qwen2.5:1.5b-instruct-q4_K_M"  # ✓
    endpoint: "http://localhost:11434"     # ✓
```

### Current LLM Backend Config (`crates/llm/src/backend.rs`)
```rust
impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model: "qwen2.5:1.5b-instruct-q4_K_M".to_string(),  // ✓
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            timeout: Duration::from_secs(120),
            stream: true,
            max_retries: 3,
            keep_alive: "5m".to_string(),  // ✓ Model stays loaded
        }
    }
}
```

### Key Features Already Implemented
1. **KV Cache Reuse**: `generate_with_session()` maintains context
2. **Think Mode Disabled**: `think: Some(false)` for faster responses
3. **Model Keep-Alive**: 5 minutes to avoid cold starts
4. **Retry Logic**: Exponential backoff for transient failures
5. **Hindi Token Estimation**: Unicode-aware estimation

---

## Comparison: Research Recommendations vs Current

| Recommendation | Research | Current | Status |
|----------------|----------|---------|--------|
| Model | Qwen2.5-1.5B | qwen2.5:1.5b-instruct-q4_K_M | ✓ Match |
| Quantization | INT4/Q4 | Q4_K_M | ✓ Match |
| Embeddings | e5-small (384d) | qwen3-embedding (1024d) | ✓ Better |
| Context Window | 4K-8K effective | 32K available | ✓ Good |
| Tool Calling | Qwen native format | Implemented | ⚠️ Verify format |
| KV Cache | Recommended | Implemented | ✓ |

---

## Available Models (Ollama)

```
NAME                             SIZE
qwen3-embedding:0.6b             639 MB    (embeddings)
qwen2.5:1.5b-instruct-q4_K_M     986 MB    (current LLM)
qwen3:1.7b-q4_K_M                1.4 GB    (upgrade option)
qwen3:4b-instruct-2507-q4_K_M    2.5 GB    (high quality option)
qwen3-embedding:4b-q4_K_M        2.5 GB    (larger embeddings)
```

### Upgrade Path
1. **Current**: qwen2.5:1.5b-instruct-q4_K_M (986 MB)
2. **Recommended**: qwen3:1.7b-q4_K_M (1.4 GB) - Native MCP, hybrid reasoning
3. **Premium**: qwen3:4b-instruct-2507-q4_K_M (2.5 GB) - Highest quality

---

## Performance Baseline

### Inference Latency (CPU)
| Metric | Value |
|--------|-------|
| TTFT (Time to First Token) | ~500ms |
| Token Generation | ~28 tokens/sec |
| Response (27 tokens) | ~1.4s total |

### Target Improvements
| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| TTFT | ~500ms | <300ms | Prompt optimization |
| Response Time | ~1.4s | <1s | Smaller responses + caching |
| Tokens/sec | 28 | 28 | Limited by hardware |

---

## Action Items from Verification

### Immediate (Phase 0)
- [x] Verify model is Q4_K_M quantized
- [x] Verify embedding dimensions match config
- [x] Confirm tool calling capability
- [ ] Test KV cache reuse latency improvement
- [ ] Benchmark Hindi text inference

### Short-term (Phase 1-2)
- [ ] Test Qwen3:1.7b upgrade (if better tool calling)
- [ ] Verify prompt template matches Qwen2.5 format
- [ ] Add token count accuracy tests

### Medium-term (Phase 3-6)
- [ ] Implement context compression
- [ ] Add dialogue state tracking
- [ ] Enhance MCP tool format
