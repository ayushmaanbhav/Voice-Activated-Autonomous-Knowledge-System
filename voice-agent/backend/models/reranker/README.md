# Cross-Encoder Reranker Model

## Recommended: ms-marco-MiniLM

```bash
pip install transformers optimum
optimum-cli export onnx --model cross-encoder/ms-marco-MiniLM-L-6-v2 models/reranker/minilm
```

## Alternative: Multilingual (for Hindi)

```bash
optimum-cli export onnx --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 models/reranker/mmarco
```

## Export with Early Exit Support

For true layer-by-layer early exit, you need a model that exposes intermediate layers.
This requires custom ONNX export:

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", output_hidden_states=True)

# Export with all hidden states
torch.onnx.export(
    model,
    (dummy_input,),
    "reranker_with_layers.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"] + [f"hidden_state_{i}" for i in range(7)],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, ...}
)
```

Note: The current implementation uses cascaded reranking (pre-filter + full model)
which provides similar speedups without requiring custom models.
