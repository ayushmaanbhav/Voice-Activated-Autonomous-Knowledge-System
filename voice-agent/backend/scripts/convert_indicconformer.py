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
