# IndicF5 TTS Integration Guide

## P0 FIX: Real Model Integration Required

The current TTS implementation in `crates/pipeline/src/tts/streaming.rs` has incorrect input schema
and requires proper phoneme conversion for IndicF5 to function correctly.

## Current Issues

1. **Wrong Input Schema**: Lines 223-230 use raw character codes instead of phoneme IDs
2. **Missing Phoneme Conversion**: No grapheme-to-phoneme (G2P) for Hindi/Indian languages
3. **Missing Speaker Embeddings**: IndicF5 requires speaker embeddings for voice cloning
4. **Incorrect ONNX Inputs**: Model expects different tensor names and shapes

## Required Model Files

### IndicF5 (AI4Bharat)

Download from: https://github.com/AI4Bharat/Indic-TTS

```
models/
  tts/
    indicf5/
      model.onnx           # Exported ONNX model (~200MB)
      phoneme_map.json     # Phoneme to ID mapping
      speaker_embeddings/  # Pre-computed speaker embeddings
        female_hindi.npy
        male_hindi.npy
      config.json          # Model configuration
```

### Export from Original

```python
# Export IndicF5 to ONNX
import torch
from indicf5 import IndicF5

model = IndicF5.from_pretrained("ai4bharat/indicf5_hindi")
model.eval()

# Export with proper input names
dummy_phonemes = torch.zeros(1, 100, dtype=torch.long)
dummy_speaker = torch.zeros(1, 256)
dummy_lengths = torch.tensor([100])

torch.onnx.export(
    model,
    (dummy_phonemes, dummy_speaker, dummy_lengths),
    "indicf5.onnx",
    input_names=["phoneme_ids", "speaker_embedding", "lengths"],
    output_names=["audio"],
    dynamic_axes={
        "phoneme_ids": {0: "batch", 1: "seq"},
        "audio": {0: "batch", 1: "time"}
    }
)
```

## Integration Steps

### Step 1: Grapheme-to-Phoneme (G2P)

Hindi text must be converted to phonemes before TTS:

```rust
use std::collections::HashMap;

pub struct HindiG2P {
    /// Devanagari to IPA mapping
    char_map: HashMap<char, Vec<&'static str>>,
}

impl HindiG2P {
    pub fn new() -> Self {
        let mut char_map = HashMap::new();

        // Vowels
        char_map.insert('अ', vec!["ə"]);
        char_map.insert('आ', vec!["aː"]);
        char_map.insert('इ', vec!["ɪ"]);
        char_map.insert('ई', vec!["iː"]);
        char_map.insert('उ', vec!["ʊ"]);
        char_map.insert('ऊ', vec!["uː"]);
        char_map.insert('ए', vec!["eː"]);
        char_map.insert('ऐ', vec!["ɛː"]);
        char_map.insert('ओ', vec!["oː"]);
        char_map.insert('औ', vec!["ɔː"]);

        // Consonants
        char_map.insert('क', vec!["k"]);
        char_map.insert('ख', vec!["kʰ"]);
        char_map.insert('ग', vec!["ɡ"]);
        char_map.insert('घ', vec!["ɡʱ"]);
        // ... complete mapping

        Self { char_map }
    }

    pub fn convert(&self, text: &str) -> Vec<String> {
        let mut phonemes = Vec::new();
        for c in text.chars() {
            if let Some(phones) = self.char_map.get(&c) {
                phonemes.extend(phones.iter().map(|s| s.to_string()));
            } else if c.is_ascii_alphabetic() {
                // English letters - use CMU dict or similar
                phonemes.push(c.to_string());
            }
        }
        phonemes
    }
}
```

### Step 2: Phoneme ID Mapping

Replace character IDs with phoneme IDs:

```rust
pub struct PhonemeMapper {
    phoneme_to_id: HashMap<String, i64>,
    id_to_phoneme: HashMap<i64, String>,
}

impl PhonemeMapper {
    pub fn load(path: &str) -> Result<Self, Error> {
        let content = std::fs::read_to_string(path)?;
        let map: HashMap<String, i64> = serde_json::from_str(&content)?;

        let reverse: HashMap<i64, String> = map.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        Ok(Self {
            phoneme_to_id: map,
            id_to_phoneme: reverse,
        })
    }

    pub fn encode(&self, phonemes: &[String]) -> Vec<i64> {
        phonemes.iter()
            .filter_map(|p| self.phoneme_to_id.get(p).copied())
            .collect()
    }
}
```

### Step 3: Fix ONNX Input Schema

Update `synthesize_chunk` in `streaming.rs`:

```rust
fn synthesize_chunk(&self, chunk: &TextChunk) -> Result<Vec<f32>, PipelineError> {
    let session = self.session.as_ref()
        .ok_or_else(|| PipelineError::Model("No model loaded".to_string()))?;

    // Step 1: Convert text to phonemes
    let g2p = HindiG2P::new();
    let phonemes = g2p.convert(&chunk.text);

    // Step 2: Map phonemes to IDs
    let phoneme_ids: Vec<i64> = self.phoneme_mapper.encode(&phonemes);

    // Step 3: Create input tensors
    let seq_len = phoneme_ids.len();

    let phoneme_input = Array2::from_shape_vec(
        (1, seq_len),
        phoneme_ids,
    ).map_err(|e| PipelineError::Tts(e.to_string()))?;

    // Speaker embedding (256-dim for IndicF5)
    let speaker_embedding = Array2::from_shape_vec(
        (1, 256),
        self.speaker_embedding.clone(),
    ).map_err(|e| PipelineError::Tts(e.to_string()))?;

    // Sequence lengths
    let lengths = Array2::from_shape_vec(
        (1, 1),
        vec![seq_len as i64],
    ).map_err(|e| PipelineError::Tts(e.to_string()))?;

    // Run inference
    let outputs = session.run(ort::inputs![
        "phoneme_ids" => phoneme_input.view(),
        "speaker_embedding" => speaker_embedding.view(),
        "lengths" => lengths.view(),
    ].map_err(|e| PipelineError::Model(e.to_string()))?)
    .map_err(|e| PipelineError::Model(e.to_string()))?;

    let audio = outputs
        .get("audio")
        .ok_or_else(|| PipelineError::Model("Missing audio output".to_string()))?
        .try_extract_tensor::<f32>()
        .map_err(|e| PipelineError::Model(e.to_string()))?;

    Ok(audio.view().iter().copied().collect())
}
```

### Step 4: Speaker Embedding Management

```rust
pub struct SpeakerManager {
    embeddings: HashMap<String, Vec<f32>>,
    default_speaker: String,
}

impl SpeakerManager {
    pub fn load(dir: &str) -> Result<Self, Error> {
        let mut embeddings = HashMap::new();

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.path().extension() == Some("npy".as_ref()) {
                let name = entry.path().file_stem()
                    .unwrap().to_string_lossy().to_string();
                let data = Self::load_npy(&entry.path())?;
                embeddings.insert(name, data);
            }
        }

        Ok(Self {
            embeddings,
            default_speaker: "female_hindi".to_string(),
        })
    }

    pub fn get(&self, name: &str) -> Option<&[f32]> {
        self.embeddings.get(name).map(|v| v.as_slice())
    }
}
```

## Environment Variables

```bash
# Model paths
export INDICF5_MODEL=/path/to/model.onnx
export INDICF5_PHONEMES=/path/to/phoneme_map.json
export INDICF5_SPEAKERS=/path/to/speaker_embeddings/

# Voice selection
export TTS_VOICE=female_hindi  # or male_hindi
export TTS_SAMPLE_RATE=22050
```

## Testing

```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    #[ignore = "requires model files"]
    fn test_indicf5_hindi() {
        let tts = StreamingTts::new(
            std::env::var("INDICF5_MODEL").unwrap(),
            TtsConfig {
                engine: TtsEngine::IndicF5,
                sample_rate: 22050,
                voice_id: Some("female_hindi".to_string()),
                ..Default::default()
            }
        ).unwrap();

        let (tx, mut rx) = mpsc::channel(10);
        tts.start("नमस्ते, मैं प्रिया हूँ", tx);

        let mut audio_samples = Vec::new();
        while let Some(event) = rx.blocking_recv() {
            match event {
                TtsEvent::Audio { samples, .. } => {
                    audio_samples.extend(samples.iter());
                }
                TtsEvent::Complete => break,
                _ => {}
            }
        }

        assert!(!audio_samples.is_empty());
        // Should be ~2 seconds of audio at 22050 Hz
        assert!(audio_samples.len() > 22050);
    }
}
```

## Latency Considerations

Target: **<100ms** for first audio chunk

| Component | Budget | Notes |
|-----------|--------|-------|
| G2P conversion | 5ms | Lookup table |
| Phoneme encoding | 2ms | HashMap lookup |
| ONNX inference | 80ms | GPU recommended |
| Audio post-processing | 5ms | Normalization |
| **First chunk** | **~90ms** | Within budget |

## Word-Level Streaming

IndicF5 can generate audio word-by-word for low latency:

```rust
impl StreamingTts {
    pub fn synthesize_word_stream(&self, text: &str) -> impl Iterator<Item = TtsEvent> {
        let words: Vec<&str> = text.split_whitespace().collect();

        words.into_iter().enumerate().map(|(idx, word)| {
            let audio = self.synthesize_chunk(&TextChunk {
                text: word.to_string(),
                word_indices: vec![idx],
                is_final: idx == words.len() - 1,
                can_pause: true,
            }).unwrap();

            TtsEvent::Audio {
                samples: audio.into(),
                text: word.to_string(),
                word_indices: vec![idx],
                is_final: idx == words.len() - 1,
            }
        })
    }
}
```

## Prosody Control

IndicF5 supports prosody modification:

```rust
pub struct ProsodyConfig {
    /// Speaking rate (0.5 = half speed, 2.0 = double speed)
    pub rate: f32,
    /// Pitch shift in semitones (-12 to +12)
    pub pitch: f32,
    /// Energy/volume (0.5 to 2.0)
    pub energy: f32,
}

impl StreamingTts {
    pub fn set_prosody(&mut self, config: ProsodyConfig) {
        self.prosody = config;
        // Applied during synthesis via scales tensor
    }
}
```

## Fallback to Piper

If IndicF5 fails, fall back to Piper TTS:

```rust
impl StreamingTts {
    pub fn with_fallback(primary: TtsConfig, fallback: TtsConfig) -> Self {
        // Try IndicF5 first, Piper as backup
    }
}
```

## References

- [AI4Bharat Indic-TTS](https://github.com/AI4Bharat/Indic-TTS)
- [IndicF5 Paper](https://arxiv.org/abs/2310.04076)
- [Piper TTS](https://github.com/rhasspy/piper) (fallback)
- [ONNX Runtime Rust](https://github.com/pykeio/ort)
