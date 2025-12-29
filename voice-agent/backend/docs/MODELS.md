# Voice Agent ONNX Models

This document describes the ONNX models required for the Voice Agent and how to obtain/export them.

## Overview

| Component | Model | Purpose | Size | Latency Target | Status |
|-----------|-------|---------|------|----------------|--------|
| VAD | Silero VAD | Voice activity detection | 1.5MB | <5ms | ✅ Ready |
| STT | IndicConformer 600M | Hindi/English speech-to-text | 4.8GB | <100ms | ✅ Ready (ONNX included) |
| TTS | IndicF5 | Hindi text-to-speech | 2.7GB | <100ms first chunk | ⚠️ Needs ONNX conversion |
| Reranker | MiniLM Cross-encoder | Document reranking | ~25MB | <20ms | 🔧 Manual export |
| Embedding | e5-small | Dense embeddings for RAG | ~100MB | <50ms | 🔧 Manual export |

## Quick Start

```bash
# Download models
./scripts/download_models.sh

# Clone models via SSH (requires HuggingFace SSH key setup)
cd models/stt && git clone git@hf.co:ai4bharat/indic-conformer-600m-multilingual indicconformer
cd models/tts && git clone git@hf.co:ai4bharat/IndicF5

# Set models path
export MODELS_PATH=./models
```

## Model Details

### 1. Silero VAD (Voice Activity Detection)

**Status**: ✅ Auto-downloaded

The Silero VAD model detects speech segments in audio. It's lightweight and fast.

```bash
# Already downloaded by script
ls models/vad/silero_vad.onnx
```

**Input**: 16kHz mono audio (512 samples = 32ms chunks)
**Output**: Speech probability (0.0 - 1.0)

### 2. IndicConformer (Speech-to-Text)

**Status**: ✅ ONNX files included in repository

The IndicConformer 600M multilingual model from AI4Bharat is the recommended STT for Hindi. The model comes with pre-exported ONNX files.

```bash
# Clone the model (requires HuggingFace SSH access)
cd models/stt
git clone git@hf.co:ai4bharat/indic-conformer-600m-multilingual indicconformer

# Model structure:
# models/stt/indicconformer/
# ├── assets/
# │   ├── encoder.onnx          # 2.9MB - Main encoder
# │   ├── ctc_decoder.onnx      # 23MB - CTC decoder
# │   ├── joint_post_net_hi.onnx # 648KB - Hindi post-processing
# │   └── ... (other language post-nets)
# ├── model_onnx.py             # ONNX inference code
# └── config.json
```

**Usage in Python** (from model's README):
```python
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    trust_remote_code=True
)
# Use model.transcribe() for inference
```

**References:**
- [HuggingFace Model](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [ONNX Export Discussion](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual/discussions/5)

#### Alternative: Indic-Seamless (Future)

For future use, the Indic-Seamless model provides better streaming support:
```bash
# Requires HuggingFace approval
git clone git@hf.co:ai4bharat/indic-seamless
```

### 3. IndicF5 TTS (Text-to-Speech)

**Status**: ⚠️ Downloaded, needs ONNX conversion

IndicF5 is a high-quality polyglot TTS model supporting 11 Indian languages including Hindi.

```bash
# Clone the model
cd models/tts
git clone git@hf.co:ai4bharat/IndicF5

# Model structure:
# models/tts/IndicF5/
# ├── model.safetensors    # 1.4GB - Main model weights
# ├── checkpoints/vocab.txt # Vocabulary
# ├── prompts/             # Reference audio samples
# └── f5_tts/              # Model code
```

#### ONNX Conversion

IndicF5 requires conversion to ONNX for use in Rust. This is based on the [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) project.

**Warning**: ONNX conversion is experimental. Some users report noise output issues.
See: https://huggingface.co/ai4bharat/IndicF5/discussions/2

```bash
# Clone the conversion tools
git clone https://github.com/DakeQQ/F5-TTS-ONNX.git scripts/F5-TTS-ONNX

# Install dependencies
pip install f5-tts vocos torchaudio jieba pypinyin pydub soundfile omegaconf

# Run conversion (requires manual path configuration)
python scripts/convert_indicf5_onnx.py
```

The conversion produces 3 ONNX files:
1. `F5_Preprocess.onnx` - Audio preprocessing, STFT, mel spectrogram
2. `F5_Transformer.onnx` - Main DiT transformer (largest)
3. `F5_Decode.onnx` - Vocos vocoder for audio synthesis

#### Native Rust Implementation (Candle)

IndicF5 SafeTensors can be loaded directly in Candle. Requires implementing:

1. **DiT blocks** (Diffusion Transformer) - `candle-nn` attention + feedforward
2. **ConvNeXt V2 blocks** - custom Module with depthwise conv
3. **Flow Matching / Sway Sampling** - ODE solver for diffusion
4. **Vocos vocoder** - mel-spectrogram to audio

```rust
use candle_core::{Device, DType};
use candle_nn::VarBuilder;

// Load IndicF5 weights
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(
        "models/tts/IndicF5/model.safetensors",
        DType::F32,
        &Device::Cpu
    )?
};

// Initialize F5-TTS model (requires custom implementation)
let model = IndicF5::load(vb)?;
```

Benefits over ONNX:
- No conversion noise issues
- Native Rust performance
- Direct HuggingFace integration

#### Alternative: Piper TTS (Simpler)

For simpler Hindi TTS without quality issues, use Piper:
```bash
# Auto-downloaded by download_models.sh
ls models/tts/hi_IN-swara-medium.onnx
```

**Input**: Phoneme sequence (IPA)
**Output**: 22050Hz audio samples

### 4. Cross-Encoder Reranker

**Status**: 🔧 Manual export required

The reranker uses a cross-encoder model to score query-document relevance.

```bash
# Export using Python (recommended)
python3 << 'EOF'
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
output_dir = "models/reranker/minilm"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

os.makedirs(output_dir, exist_ok=True)

dummy_input = tokenizer([("query", "document")], padding=True, truncation=True, return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    f"{output_dir}/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"}
    },
    opset_version=14
)

tokenizer.save_pretrained(output_dir)
print(f"Model exported to {output_dir}")
EOF
```

### 5. Embedding Model (RAG)

**Status**: 🔧 Manual export required

Dense embeddings for semantic search.

```bash
# Export using Python
python3 << 'EOF'
import os
import torch
from transformers import AutoModel, AutoTokenizer

model_name = "intfloat/e5-small-v2"
output_dir = "models/embedding/e5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

os.makedirs(output_dir, exist_ok=True)

dummy_input = tokenizer(["query: sample text"], padding=True, truncation=True, return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    f"{output_dir}/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"}
    },
    opset_version=14
)

tokenizer.save_pretrained(output_dir)
print(f"Model exported to {output_dir}")
EOF
```

For multilingual (Hindi + English):
```bash
# Use multilingual-e5-small instead
model_name = "intfloat/multilingual-e5-small"
output_dir = "models/embedding/me5-small"
```

## Configuration

Set the model paths in your configuration:

```toml
[models]
vad_path = "models/vad/silero_vad.onnx"
stt_path = "models/stt/indicconformer"
tts_path = "models/tts/IndicF5_ONNX"  # or "models/tts/hi_IN-swara-medium.onnx"
reranker_path = "models/reranker/minilm"
embedding_path = "models/embedding/e5-small"
```

Or via environment variables:

```bash
export VAD_MODEL_PATH=models/vad/silero_vad.onnx
export STT_MODEL_PATH=models/stt/indicconformer
export TTS_MODEL_PATH=models/tts/IndicF5_ONNX
export RERANKER_MODEL_PATH=models/reranker/minilm
export EMBEDDING_MODEL_PATH=models/embedding/e5-small
```

## Inference Backends

### Option 1: ONNX Runtime (Primary)

Used for models with pre-exported ONNX files or complex architectures.

| Model | ONNX Runtime Version | Execution Provider | Notes |
|-------|---------------------|-------------------|-------|
| All | 2.0.0-rc.9+ | CPU | Default, always works |
| All | 2.0.0-rc.9+ | CUDA | Requires CUDA 11.x |
| All | 2.0.0-rc.9+ | TensorRT | Best GPU performance |

### Option 2: Candle (SafeTensors)

[Candle](https://github.com/huggingface/candle) is Hugging Face's Rust ML framework. It can load SafeTensors directly without ONNX conversion.

**Supported models:**
- BERT embeddings (e5-small, multilingual-e5-small)
- Whisper (alternative STT)
- MetaVoice/Parler-TTS (alternative TTS)

**Requires custom implementation:**
- Cross-encoder classification head (reranker) - add classification layer on top of BERT
- IndicConformer (custom architecture) - implement Conformer blocks
- IndicF5 (F5-TTS architecture) - implement DiT + ConvNeXt V2 + Flow Matching

All SafeTensors can be loaded; the architecture just needs to be implemented in Rust using `candle-nn` primitives.

```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::bert::BertModel;
use hf_hub::api::sync::Api;

// Load BERT from HuggingFace Hub
let api = Api::new()?;
let repo = api.model("intfloat/e5-small-v2".to_string());
let weights = repo.get("model.safetensors")?;

// Load into Candle
let device = Device::Cpu;
let vb = candle_nn::VarBuilder::from_safetensors(weights, candle_core::DType::F32, &device)?;
let model = BertModel::load(vb, &config)?;
```

**Workspace dependencies (Cargo.toml):**
```toml
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
safetensors = "0.4"
hf-hub = "0.3"
```

## Latency Optimization

1. **Use INT8 Quantization** for faster inference:
   ```bash
   python -m onnxruntime.quantization.quantize \
       --model model.onnx \
       --output model_int8.onnx \
       --quant_format QDQ
   ```

2. **Use ONNX Runtime Graph Optimization**:
   - Set `GraphOptimizationLevel::Level3` in code

3. **Use IOBinding** for zero-copy GPU inference

4. **Batch Processing** for embedding/reranking

## Troubleshooting

### Model Loading Fails

```
Error: Failed to load model: Invalid model
```

- Verify ONNX file is not corrupted
- Check ONNX opset version compatibility
- Try reconverting with latest optimum

### Out of Memory

- Use INT8 quantized models
- Reduce batch size
- Enable memory arena shrinking

### Slow Performance

- Enable GPU execution provider
- Check model is actually using ONNX Runtime (not Python fallback)
- Profile with ONNX Runtime profiler

### IndicF5 Noise Output

If IndicF5 ONNX produces noise:
- Verify reference audio quality
- Check sample rate matches (24kHz)
- Try Piper TTS as fallback
- Run PyTorch model via Python service as workaround
