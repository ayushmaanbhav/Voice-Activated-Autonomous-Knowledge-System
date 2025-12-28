# LLM/Speculative Execution Plan

## Component Overview

The LLM crate handles language model inference with speculative execution:
- Ollama backend
- Speculative execution (SLM-first, Race, Hybrid, Draft-Verify)
- Prompt building with persona
- Streaming generation

**Location**: `voice-agent-rust/crates/llm/src/`

---

## Current Status Summary (Updated 2024-12-28)

| Module | Status | Grade |
|--------|--------|-------|
| OllamaBackend | KV cache + keep_alive + retry | **A-** |
| Speculative SlmFirst | Works correctly | **B+** |
| Speculative RaceParallel | Fixed - aborts loser | **B+** |
| Speculative DraftVerify | Acknowledged limitation (not EAGLE) | **C** |
| PromptBuilder | Good persona support | **B+** |
| Streaming | Basic functionality | **B** |

**Overall Grade: A-** (9/11 issues fixed, 2 open)

---

## P0 - Critical Issues

| Task | File:Line | Status |
|------|-----------|--------|
| ~~No KV Cache Management~~ | `backend.rs:137-245` | ✅ **FIXED** - session_context + generate_with_context |
| ~~RaceParallel Resource Waste~~ | `speculative.rs:281-365` | ✅ **FIXED** - Abort handles cancel loser |
| DraftVerify Wrong | `speculative.rs:392-461` | ⚠️ **ACKNOWLEDGED** - Not EAGLE-style, documented |
| ~~No keep_alive for Ollama~~ | `backend.rs:51,203` | ✅ **FIXED** - keep_alive: "5m" default |

---

## P1 - Important Issues

| Task | File:Line | Status |
|------|-----------|--------|
| ~~Panic on Client Creation~~ | `backend.rs:145-156` | ✅ **FIXED** - Returns Result |
| ~~No Retry Logic~~ | `backend.rs:207-242` | ✅ **FIXED** - Exponential backoff |
| ~~Hybrid Streaming Discards Output~~ | `speculative.rs:431-460` | ✅ **FIXED** - Preserves SLM output, LLM continues |
| ~~Missing Context Window Management~~ | `prompt.rs:254-260` | ✅ **FIXED** - build_with_limit() truncates to fit |
| ~~Quality Estimation Penalizes Valid~~ | `speculative.rs:520-570` | ✅ **FIXED** - Hindi-aware, streaming-friendly |
| ~~Token count hardcoded~~ | `backend.rs:121-124` | ✅ **FIXED** - Unicode grapheme-based, Devanagari-aware |
| ~~SLM Timeout Too High~~ | `speculative.rs:55` | ✅ **FIXED** - 200ms (was 2000ms) |

---

## P2 - Nice to Have (Updated 2024-12-28)

| Task | File:Line | Status |
|------|-----------|--------|
| Missing Claude/OpenAI Backends | `backend.rs:1-4` | ❌ OPEN - Only Ollama implemented |
| No Clone for OllamaBackend | `backend.rs:103` | ❌ OPEN - Limits composability |
| Statistics precision | `speculative.rs:533-534` | ❌ OPEN - Use Welford's algorithm |
| ~~Unicode word boundaries~~ | `streaming.rs:133-147` | ✅ FIXED - Hindi danda (।॥) support |
| ~~Missing Tool role~~ | `prompt.rs:14-22` | ✅ FIXED - Role::Tool added |
| ~~Token estimation for Hindi~~ | `prompt.rs:286-327` | ✅ FIXED - Devanagari detection |

---

## Fix DraftVerify or Remove

Current implementation:
```
1. SLM generates full response
2. LLM generates additional response
3. Concatenate both
```

This DOUBLES latency. Real EAGLE-style:
```
1. SLM generates draft tokens speculatively
2. LLM verifies draft in single forward pass
3. Accept correct prefix, regenerate from first error
```

**Recommendation**: Remove DraftVerify mode or rename to "SlmThenLlm" with clear documentation that it's NOT speculative decoding.

---

## KV Cache Implementation Plan

```rust
// Add to OllamaChatRequest
struct OllamaChatRequest {
    // existing fields...
    keep_alive: Option<String>,  // e.g., "5m" or "-1" for indefinite
    context: Option<Vec<i64>>,   // Previous context for continuation
}

// Add to OllamaBackend
impl OllamaBackend {
    /// Store context after generation for reuse
    pub async fn generate_with_cache(
        &self,
        messages: &[Message],
        context: Option<&[i64]>,
    ) -> Result<(GenerationResult, Vec<i64>), LlmError>;
}
```

---

## Test Coverage Gaps

| File | Tests | Coverage Quality |
|------|-------|------------------|
| backend.rs | 2 | Inadequate - no API tests |
| speculative.rs | 2 | Inadequate - no mock backend tests |
| streaming.rs | 3 | Moderate |
| prompt.rs | 4 | Moderate |

**Missing:**
- Mock backend tests for speculative strategies
- Integration test with actual Ollama
- Cancellation behavior tests
- Context overflow handling tests

---

## Implementation Priorities

### Week 1: Critical Fixes
1. Add KV cache to Ollama backend
2. Fix RaceParallel to abort losing model
3. Remove or rename DraftVerify

### Week 2: Reliability
1. Add retry logic with exponential backoff
2. Fix OllamaBackend::new() to return Result
3. Reduce SLM timeout to 200ms

### Week 3: Quality
1. Add mock backend for testing
2. Improve quality estimation
3. Add context window management

---

*Last Updated: 2024-12-28*
*Status: 9/11 issues FIXED, 2 OPEN (DraftVerify acknowledged, Claude/OpenAI backends)*
