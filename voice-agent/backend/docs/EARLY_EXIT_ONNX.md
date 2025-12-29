# Early Exit with ONNX: Current Limitation and Future Solutions

## P0 Documentation: Why Early-Exit Doesn't Work with ONNX

### The Problem

The `EarlyExitReranker` in `crates/rag/src/reranker.rs` is designed to support
early-exit inference, where the model can terminate early if confident about
its prediction. However, **this feature is not functional with standard ONNX models**.

### Technical Background

Early-exit inference requires:
1. Access to intermediate layer outputs (hidden states)
2. Ability to pause inference mid-execution
3. Evaluation of exit criteria between layers

ONNX Runtime limitations:
1. **Monolithic graph execution**: ONNX compiles the model as a single optimized graph
2. **No intermediate outputs**: Standard exports only return final logits
3. **No pause mechanism**: The runtime executes the entire graph atomically

### Current Behavior

```rust
// In reranker.rs - run_with_early_exit()
fn run_with_early_exit(&self, ...) -> Result<(f32, Option<usize>), RagError> {
    // Runs FULL model - no actual early exit
    let outputs = self.session.run(...)?;

    // Returns None for exit_layer because we can't exit early
    Ok((score, None))
}
```

The `should_exit()` function is marked `#[allow(dead_code)]` because no caller
provides `LayerOutput` data - it would only work if we had per-layer outputs.

### Alternative: Cascaded Reranking

Instead of layer-level early exit, we implement **cascaded reranking**:

```
Query → Pre-filter (BM25/keyword) → Top candidates → Full model → Final ranking
         ↓ eliminates ~70%           ↓ only ~30%      ↓ highest quality
```

Benefits:
- Similar latency reduction (skip ~70% of documents)
- Works with any ONNX model
- No custom export required

### Future: Enabling True Early Exit

#### Option 1: Custom ONNX Export

Modify the PyTorch export to include hidden state outputs:

```python
class RerankerWithHiddenStates(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True  # Enable hidden states
        )

        # Return logits + all hidden states
        return (outputs.logits, *outputs.hidden_states)

# Export with multiple outputs
torch.onnx.export(
    wrapped_model,
    (dummy_input_ids, dummy_attention),
    "reranker_with_hidden.onnx",
    output_names=["logits"] + [f"hidden_{i}" for i in range(13)],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
    }
)
```

Then in Rust:
```rust
fn run_with_early_exit(&self, ...) -> Result<(f32, Option<usize>), RagError> {
    let outputs = self.session.run(...)?;

    // Now we have hidden states!
    for layer in 0..12 {
        let hidden = outputs.get(&format!("hidden_{}", layer))?;
        let (score, confidence) = self.classify_from_hidden(hidden)?;

        if self.should_exit(confidence, layer) {
            return Ok((score, Some(layer)));
        }
    }

    let final_logits = outputs.get("logits")?;
    Ok((self.compute_score(final_logits), None))
}
```

**Drawback**: All hidden states are computed even if we exit early. The graph
still runs completely; we just ignore later outputs.

#### Option 2: Split ONNX Models

Export each transformer block as a separate ONNX file:

```
reranker_embed.onnx      # Embedding layer
reranker_layer_0.onnx    # Transformer block 0
reranker_layer_1.onnx    # Transformer block 1
...
reranker_layer_11.onnx   # Transformer block 11
reranker_classifier.onnx # Classification head
```

Then in Rust:
```rust
fn run_with_early_exit(&self, input: &Tensor) -> Result<(f32, Option<usize>), RagError> {
    let mut hidden = self.embed_session.run(input)?;

    for (i, layer_session) in self.layer_sessions.iter().enumerate() {
        hidden = layer_session.run(hidden)?;

        // Probe hidden state for confidence
        let (score, confidence) = self.probe_hidden(&hidden)?;
        if confidence >= self.config.threshold && i >= self.config.min_layer {
            return Ok((score, Some(i)));
        }
    }

    let logits = self.classifier_session.run(hidden)?;
    Ok((self.compute_score(logits), None))
}
```

**Benefits**: True early exit - skipped layers never execute.
**Drawbacks**: Complex setup, model management overhead, potential accuracy loss
from breaking optimization across layers.

#### Option 3: Use Candle Instead of ONNX

The Candle framework allows native Rust model execution with full control:

```rust
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

fn run_with_early_exit(&self, tokens: &Tensor) -> Result<(f32, Option<usize>)> {
    let embeddings = self.model.embeddings(tokens)?;
    let mut hidden = embeddings;

    for (i, layer) in self.model.encoder.layers.iter().enumerate() {
        hidden = layer.forward(&hidden)?;

        // Check exit condition after each layer
        let (score, confidence) = self.probe_classifier.forward(&hidden)?;
        if self.should_exit(confidence, i) {
            return Ok((score, Some(i)));
        }
    }

    let logits = self.classifier.forward(&hidden)?;
    Ok((self.compute_score(logits), None))
}
```

**Benefits**:
- True early exit with actual compute savings
- Native Rust with no FFI overhead
- Full control over execution

**Drawbacks**:
- Requires porting model weights (safetensors format)
- May be slower than ONNX Runtime for full runs
- Limited operator support compared to ONNX

### Recommendation

For the gold loan voice agent use case:

1. **Short term**: Keep cascaded reranking (current implementation)
   - Works well, provides ~60-70% latency reduction
   - No model changes required

2. **Medium term**: Consider Candle migration
   - The project already uses Candle for STT/TTS
   - Would enable true early-exit
   - Better latency for streaming scenarios

3. **Long term**: Investigate split ONNX approach
   - Maximum flexibility
   - Best latency for high-throughput scenarios

### Related Files

- `crates/rag/src/reranker.rs` - EarlyExitReranker implementation
- `crates/rag/src/retriever.rs` - Cascaded reranking integration
- `plans/03-rag-plan.md` - RAG component planning

### References

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [Candle Framework](https://github.com/huggingface/candle)
- [Early Exit Paper: DeeBERT](https://arxiv.org/abs/2004.12993)
- [Right Tool for the Job: Matching Model and Instance Complexities](https://arxiv.org/abs/2004.00436)
