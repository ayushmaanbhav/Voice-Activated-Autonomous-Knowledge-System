# IndicConformer STT Integration Guide

## P0 FIX: Real Model Integration Required

The current STT implementation in `crates/pipeline/src/stt/streaming.rs` has placeholder vocabulary
and requires proper model integration to function correctly.

## Current Issues

1. **Fake Vocabulary**: Lines 136-148 generate placeholder tokens `<0>`, `<1>`, etc.
2. **No Real Tokenizer**: Missing SentencePiece/BPE tokenizer for IndicConformer
3. **Missing Model Files**: No ONNX model download/verification

## Required Model Files

### IndicConformer (AI4Bharat)

Download from: https://github.com/AI4Bharat/IndicConformer

```
models/
  stt/
    indicconformer/
      model.onnx           # Exported ONNX model (~500MB)
      tokenizer.model      # SentencePiece tokenizer
      vocab.txt            # Vocabulary file (8000+ tokens)
      config.json          # Model configuration
```

### Export from NeMo

```python
# Export IndicConformer to ONNX
import nemo.collections.asr as nemo_asr

# Load pretrained model
model = nemo_asr.models.EncDecCTCModel.from_pretrained(
    "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"
)

# Export to ONNX
model.export("indicconformer.onnx")
```

## Integration Steps

### Step 1: Vocabulary Loader

Replace the placeholder in `streaming.rs:136-148`:

```rust
fn load_vocab(engine: &SttEngine) -> Result<Vec<String>, PipelineError> {
    match engine {
        SttEngine::IndicConformer => {
            // Load real vocabulary from file
            let vocab_path = std::env::var("INDICCONFORMER_VOCAB")
                .unwrap_or_else(|_| "models/stt/indicconformer/vocab.txt".to_string());

            let content = std::fs::read_to_string(&vocab_path)
                .map_err(|e| PipelineError::Model(format!("Failed to load vocab: {}", e)))?;

            Ok(content.lines().map(|s| s.to_string()).collect())
        }
        // ... other engines
    }
}
```

### Step 2: SentencePiece Tokenizer

Add dependency in `Cargo.toml`:
```toml
sentencepiece = "0.11"  # Or use tokenizers crate
```

Create tokenizer wrapper:
```rust
use sentencepiece::SentencePieceProcessor;

pub struct IndicTokenizer {
    processor: SentencePieceProcessor,
}

impl IndicTokenizer {
    pub fn new(model_path: &str) -> Result<Self, Error> {
        let processor = SentencePieceProcessor::open(model_path)?;
        Ok(Self { processor })
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        self.processor.decode_pieces(ids).unwrap_or_default()
    }
}
```

### Step 3: ONNX Model Input Schema

IndicConformer expects:
- **Input**: `audio` - Float32[batch, time] - Raw audio samples at 16kHz
- **Output**: `logits` - Float32[batch, frames, vocab_size] - CTC logits

Verify with:
```bash
python -c "import onnx; m = onnx.load('model.onnx'); print(m.graph.input, m.graph.output)"
```

### Step 4: Streaming Inference

The current chunked processing is correct, but needs:
1. Proper feature extraction (Mel spectrogram) before ONNX inference
2. CTC decoding with language model (optional but improves accuracy)

```rust
// Add mel spectrogram extraction
fn extract_features(&self, audio: &[f32]) -> Array2<f32> {
    // 80-dim mel filterbank, 25ms window, 10ms hop
    // Use mel_spec crate or implement manually
}
```

## Environment Variables

```bash
# Model paths
export INDICCONFORMER_MODEL=/path/to/model.onnx
export INDICCONFORMER_VOCAB=/path/to/vocab.txt
export INDICCONFORMER_TOKENIZER=/path/to/tokenizer.model

# Optional: Language hint
export STT_LANGUAGE=hi  # Hindi (default)
```

## Testing

```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    #[ignore = "requires model files"]
    fn test_indicconformer_hindi() {
        let stt = StreamingStt::new(
            std::env::var("INDICCONFORMER_MODEL").unwrap(),
            SttConfig {
                engine: SttEngine::IndicConformer,
                language: Some("hi".to_string()),
                ..Default::default()
            }
        ).unwrap();

        // Load test audio
        let audio = load_test_audio("tests/audio/hindi_sample.wav");

        // Process
        for chunk in audio.chunks(1600) {
            stt.process(chunk).unwrap();
        }

        let result = stt.finalize();
        assert!(!result.text.is_empty());
        assert!(result.text.contains("नमस्ते")); // Example Hindi word
    }
}
```

## Latency Considerations

Target: **<100ms** for STT processing

| Component | Budget | Notes |
|-----------|--------|-------|
| Audio buffering | 10ms | 160 samples at 16kHz |
| Feature extraction | 5ms | Mel spectrogram |
| ONNX inference | 60ms | GPU recommended |
| CTC decoding | 10ms | Greedy or beam search |
| Post-processing | 5ms | Normalization, punctuation |
| **Total** | **90ms** | Within budget |

## Hindi/Hinglish Support

IndicConformer supports:
- Pure Hindi (Devanagari script)
- Hinglish (Hindi written in Roman script)
- Code-switching (Hindi-English mix)

Configure via:
```rust
SttConfig {
    language: Some("hi".to_string()),  // Hindi
    // or
    language: Some("hi-en".to_string()),  // Hinglish/code-switch
}
```

## Fallback to Whisper

If IndicConformer fails, fall back to Whisper:

```rust
impl StreamingStt {
    pub fn with_fallback(primary: SttConfig, fallback: SttConfig) -> Self {
        // Try IndicConformer first, Whisper as backup
    }
}
```

## References

- [AI4Bharat IndicConformer](https://github.com/AI4Bharat/IndicConformer)
- [NeMo ASR Models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html)
- [ONNX Runtime Rust](https://github.com/pykeio/ort)
