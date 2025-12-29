# Embedding Model for RAG

## Recommended: e5-small (fast, good quality)

```bash
pip install optimum
optimum-cli export onnx --model intfloat/e5-small-v2 models/embedding/e5-small
```

## Alternative: Multilingual e5

```bash
optimum-cli export onnx --model intfloat/multilingual-e5-small models/embedding/me5-small
```

## Usage

Place the exported model in models/embedding/ with structure:
- model.onnx
- tokenizer.json
- tokenizer_config.json
