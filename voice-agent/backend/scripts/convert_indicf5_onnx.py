#!/usr/bin/env python3
"""
Convert IndicF5 TTS model to ONNX format.

Based on: https://github.com/DakeQQ/F5-TTS-ONNX

This script exports IndicF5 to 3 separate ONNX models:
1. F5_Preprocess.onnx - Audio preprocessing, STFT, mel spectrogram
2. F5_Transformer.onnx - Main DiT transformer model
3. F5_Decode.onnx - Vocos vocoder for audio synthesis

WARNING: IndicF5 ONNX conversion is experimental. Users have reported
noise output issues. See: https://huggingface.co/ai4bharat/IndicF5/discussions/2

Usage:
    pip install f5-tts vocos torchaudio jieba pypinyin pydub soundfile omegaconf
    python scripts/convert_indicf5_onnx.py
"""

import os
import sys

# Check if F5-TTS-ONNX scripts are available
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
F5_ONNX_DIR = os.path.join(SCRIPT_DIR, "F5-TTS-ONNX", "Export_ONNX", "F5_TTS")

if not os.path.exists(F5_ONNX_DIR):
    print("F5-TTS-ONNX not found. Cloning...")
    os.system(f"git clone https://github.com/DakeQQ/F5-TTS-ONNX.git {os.path.join(SCRIPT_DIR, 'F5-TTS-ONNX')} --depth 1")

# Configuration for IndicF5
INDICF5_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "tts", "IndicF5", "model.safetensors")
INDICF5_VOCAB_PATH = os.path.join(SCRIPT_DIR, "..", "models", "tts", "IndicF5", "checkpoints", "vocab.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "models", "tts", "IndicF5_ONNX")

def check_requirements():
    """Check if required packages are installed."""
    required = ['torch', 'torchaudio', 'onnxruntime', 'soundfile', 'omegaconf']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True

def main():
    print("=" * 60)
    print("IndicF5 ONNX Conversion")
    print("=" * 60)

    if not check_requirements():
        sys.exit(1)

    if not os.path.exists(INDICF5_MODEL_PATH):
        print(f"Error: IndicF5 model not found at {INDICF5_MODEL_PATH}")
        print("Run: cd models/tts && git clone git@hf.co:ai4bharat/IndicF5")
        sys.exit(1)

    print(f"\nModel: {INDICF5_MODEL_PATH}")
    print(f"Vocab: {INDICF5_VOCAB_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("IMPORTANT NOTES")
    print("=" * 60)
    print("""
1. IndicF5 ONNX conversion requires modifying the F5-TTS-ONNX scripts
   to use Hindi text processing instead of Chinese pinyin.

2. The F5-TTS-ONNX scripts replace system-installed vocos and f5_tts
   packages with modified versions. Re-install after conversion.

3. Known issues: Some users report noise output from ONNX model.
   See: https://huggingface.co/ai4bharat/IndicF5/discussions/2

4. For production use, consider:
   - Using PyTorch model with TorchScript instead of ONNX
   - Running IndicF5 as a Python service called from Rust via IPC
   - Using the faster Piper TTS for basic Hindi TTS

Steps to manually convert:
1. cd scripts/F5-TTS-ONNX/Export_ONNX/F5_TTS
2. Edit Export_F5.py:
   - Set F5_safetensors_path to IndicF5 model.safetensors
   - Set vocab_path to IndicF5 vocab.txt
   - Modify convert_char_to_pinyin() for Hindi text processing
3. Download Vocos: https://huggingface.co/charactr/vocos-mel-24khz
4. Run: python Export_F5.py
""")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create a placeholder config
    config = {
        "model_type": "indicf5",
        "onnx_files": {
            "preprocess": "F5_Preprocess.onnx",
            "transformer": "F5_Transformer.onnx",
            "decode": "F5_Decode.onnx"
        },
        "sample_rate": 24000,
        "hop_length": 256,
        "n_mels": 100,
        "nfe_steps": 32,
        "status": "requires_manual_conversion"
    }

    import json
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nCreated config at {OUTPUT_DIR}/config.json")
    print("Manual conversion required - see notes above.")

if __name__ == "__main__":
    main()
