#!/bin/bash
# Download ONNX models for Voice Agent
#
# This script downloads pre-trained models required for the voice agent:
# - IndicConformer (STT) - Hindi/English speech recognition
# - Piper/IndicF5 (TTS) - Hindi text-to-speech
# - Cross-encoder (Reranker) - Document reranking
# - Silero VAD - Voice activity detection
#
# Usage:
#   ./scripts/download_models.sh [--all|--stt|--tts|--reranker|--vad]

set -e

MODELS_DIR="${MODELS_DIR:-./models}"
HUGGINGFACE_CACHE="${HUGGINGFACE_CACHE:-$HOME/.cache/huggingface}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create models directory structure
create_dirs() {
    log_info "Creating model directories..."
    mkdir -p "$MODELS_DIR/stt"
    mkdir -p "$MODELS_DIR/tts"
    mkdir -p "$MODELS_DIR/reranker"
    mkdir -p "$MODELS_DIR/vad"
    mkdir -p "$MODELS_DIR/embedding"
}

# Download Silero VAD model
download_vad() {
    log_info "Downloading Silero VAD model..."
    local vad_url="https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
    local vad_path="$MODELS_DIR/vad/silero_vad.onnx"

    if [ -f "$vad_path" ]; then
        log_warn "VAD model already exists at $vad_path"
        return
    fi

    curl -L -o "$vad_path" "$vad_url"
    log_info "VAD model downloaded to $vad_path"
}

# Download IndicConformer STT model
download_stt() {
    log_info "Setting up STT model..."
    local stt_path="$MODELS_DIR/stt"

    # Check if Python and required packages are available
    if command -v python3 &> /dev/null; then
        log_info "Attempting to download IndicConformer model from HuggingFace..."

        python3 << 'PYTHON_SCRIPT' || log_warn "Auto-download failed, manual setup required"
import sys
try:
    from huggingface_hub import hf_hub_download
    import os

    models_dir = os.environ.get('MODELS_DIR', './models')
    stt_path = os.path.join(models_dir, 'stt')
    os.makedirs(stt_path, exist_ok=True)

    # Download the Hindi model
    print("Downloading ai4b_indicconformer_hi.nemo from HuggingFace...")
    hf_hub_download(
        repo_id='ai4bharat/indic-conformer-600m-multilingual',
        filename='ai4b_indicconformer_hi.nemo',
        local_dir=stt_path
    )
    print(f"Downloaded to {stt_path}/ai4b_indicconformer_hi.nemo")
    print("\nNext step: Convert to ONNX using:")
    print("  python scripts/convert_indicconformer.py")

except ImportError:
    print("huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)
except Exception as e:
    print(f"Download failed: {e}")
    sys.exit(1)
PYTHON_SCRIPT
    fi

    # Create conversion script
    cat > "$MODELS_DIR/../scripts/convert_indicconformer.py" << 'EOF'
#!/usr/bin/env python3
"""Convert IndicConformer .nemo model to ONNX format."""

import os
import sys

def main():
    models_dir = os.environ.get('MODELS_DIR', './models')
    nemo_path = os.path.join(models_dir, 'stt', 'ai4b_indicconformer_hi.nemo')
    onnx_path = os.path.join(models_dir, 'stt', 'indicconformer_hi.onnx')

    if not os.path.exists(nemo_path):
        print(f"Error: {nemo_path} not found")
        print("Run: ./scripts/download_models.sh --stt")
        sys.exit(1)

    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("Required packages not installed. Run:")
        print("  pip install nemo_toolkit[asr] torch")
        sys.exit(1)

    print(f"Loading model from {nemo_path}...")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(nemo_path)
    model.eval()

    print(f"Exporting to {onnx_path}...")
    # NeMo has built-in ONNX export
    model.export(onnx_path)

    print(f"Model exported to {onnx_path}")

    # Optimize if onnxruntime is available
    try:
        from onnxruntime.transformers import optimizer
        print("Optimizing model...")
        optimized = optimizer.optimize_model(onnx_path, model_type='bert')
        opt_path = onnx_path.replace('.onnx', '_optimized.onnx')
        optimized.save_model_to_file(opt_path)
        print(f"Optimized model saved to {opt_path}")
    except ImportError:
        print("onnxruntime not installed, skipping optimization")

if __name__ == '__main__':
    main()
EOF
    chmod +x "$MODELS_DIR/../scripts/convert_indicconformer.py"

    cat > "$stt_path/README.md" << 'EOF'
# IndicConformer STT Model

## Quick Setup

1. Download the .nemo model (auto or manual):
   ```bash
   ./scripts/download_models.sh --stt
   ```

2. Convert to ONNX:
   ```bash
   pip install nemo_toolkit[asr] torch onnxruntime
   python scripts/convert_indicconformer.py
   ```

## Manual Download

If auto-download fails, manually download from:
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual

Place the .nemo file at: models/stt/ai4b_indicconformer_hi.nemo

## References

- [HuggingFace Model](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [ONNX Export Discussion](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual/discussions/5)
- [AI4Bharat GitHub](https://github.com/AI4Bharat/IndicConformer)
EOF

    log_info "STT setup complete. See $stt_path/README.md for conversion steps."
}

# Download TTS model (Piper Hindi voice)
download_tts() {
    log_info "Downloading TTS model (Piper Hindi)..."
    local tts_path="$MODELS_DIR/tts"

    # Piper Hindi voice
    local voice_name="hi_IN-swara-medium"
    local model_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/swara/medium/hi_IN-swara-medium.onnx"
    local config_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/swara/medium/hi_IN-swara-medium.onnx.json"

    if [ -f "$tts_path/$voice_name.onnx" ]; then
        log_warn "TTS model already exists at $tts_path/$voice_name.onnx"
    else
        log_info "Downloading Piper Hindi voice..."
        curl -L -o "$tts_path/$voice_name.onnx" "$model_url" || {
            log_warn "Failed to download from HuggingFace. Creating placeholder..."
            touch "$tts_path/$voice_name.onnx.placeholder"
        }
        curl -L -o "$tts_path/$voice_name.onnx.json" "$config_url" || true
    fi

    # Create G2P config for Hindi
    cat > "$tts_path/g2p_config.json" << 'EOF'
{
    "language": "hi",
    "phoneme_set": "ipa",
    "mappings": {
        "schwa_deletion": true,
        "nukta_handling": true,
        "gemination": true
    },
    "rules_file": "hindi_g2p_rules.txt"
}
EOF

    log_info "TTS model setup complete at $tts_path"
}

# Download cross-encoder reranker model
download_reranker() {
    log_info "Setting up reranker model..."
    local reranker_path="$MODELS_DIR/reranker"

    cat > "$reranker_path/README.md" << 'EOF'
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
EOF

    log_warn "Reranker model requires manual setup. See $reranker_path/README.md"
}

# Download embedding model for RAG
download_embedding() {
    log_info "Setting up embedding model..."
    local embed_path="$MODELS_DIR/embedding"

    cat > "$embed_path/README.md" << 'EOF'
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
EOF

    log_warn "Embedding model requires manual setup. See $embed_path/README.md"
}

# Main function
main() {
    local target="${1:-all}"

    create_dirs

    case "$target" in
        all)
            download_vad
            download_stt
            download_tts
            download_reranker
            download_embedding
            ;;
        vad)
            download_vad
            ;;
        stt)
            download_stt
            ;;
        tts)
            download_tts
            ;;
        reranker)
            download_reranker
            ;;
        embedding)
            download_embedding
            ;;
        *)
            echo "Usage: $0 [--all|--stt|--tts|--reranker|--vad|--embedding]"
            exit 1
            ;;
    esac

    echo ""
    log_info "Model download complete!"
    echo ""
    echo "Models directory: $MODELS_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Review README files in each model directory"
    echo "2. Download/export models that require manual setup"
    echo "3. Set MODELS_PATH environment variable or update config"
    echo ""
}

main "$@"
