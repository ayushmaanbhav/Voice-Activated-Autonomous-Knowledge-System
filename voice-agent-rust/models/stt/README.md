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
