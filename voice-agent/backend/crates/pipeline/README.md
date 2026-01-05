<![CDATA[# voice-agent-pipeline

> Real-time audio processing pipeline: VAD → STT → Turn Detection → TTS

---

## Overview

The `pipeline` crate provides the complete audio processing stack for voice conversations. It handles:

- **Voice Activity Detection (VAD)** - Detect speech vs silence
- **Speech-to-Text (STT)** - Transcribe audio to text
- **Turn Detection** - Know when user is done speaking
- **Text-to-Speech (TTS)** - Generate natural speech
- **Barge-In Handling** - Allow user interruptions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AUDIO IN                                                                   │
│     │                                                                       │
│     ▼                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     VOICE ACTIVITY DETECTION                         │   │
│  │                                                                      │   │
│  │   ┌───────────────┐          ┌───────────────┐                      │   │
│  │   │  Silero VAD   │    OR    │  MagicNet VAD │                      │   │
│  │   │  (ONNX, 2MB)  │          │  (10ms frames)│                      │   │
│  │   └───────────────┘          └───────────────┘                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       SPEECH-TO-TEXT                                 │   │
│  │                                                                      │   │
│  │   ┌───────────────────┐       ┌───────────────────┐                 │   │
│  │   │  IndicConformer   │       │     Streaming     │                 │   │
│  │   │  (22 languages)   │───────│    Transcription  │                 │   │
│  │   └───────────────────┘       └───────────────────┘                 │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TURN DETECTION                                  │   │
│  │                                                                      │   │
│  │   ┌───────────────┐          ┌───────────────┐                      │   │
│  │   │  VAD-Based    │    +     │   Semantic    │                      │   │
│  │   │  (Silence)    │          │  (SmolLM2)    │                      │   │
│  │   └───────────────┘          └───────────────┘                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│            TRANSCRIPT         ▼          RESPONSE                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      TEXT-TO-SPEECH                                  │   │
│  │                                                                      │   │
│  │   ┌───────────────┐          ┌───────────────┐                      │   │
│  │   │   IndicF5     │    OR    │    Piper      │                      │   │
│  │   │ (Indian TTS)  │          │  (Fallback)   │                      │   │
│  │   └───────────────┘          └───────────────┘                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                             │
│                               ▼                                             │
│                          AUDIO OUT                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Modules

### VAD (Voice Activity Detection)

```rust
use voice_agent_pipeline::vad::{SileroVad, VadEngine, VadResult};

let vad = SileroVad::new("models/vad/silero_vad.onnx")?;

// Process audio frame
let result = vad.process(&audio_frame).await?;
match result {
    VadResult::Speech { probability } => println!("Speech detected: {}", probability),
    VadResult::Silence => println!("Silence"),
}
```

**Available Engines:**
- `SileroVad` - Production-ready, ONNX-based
- `MagicNetVad` - Lower latency (10ms frames)

### STT (Speech-to-Text)

```rust
use voice_agent_pipeline::stt::{IndicConformerStt, SttEngine};

let stt = IndicConformerStt::new(IndicConformerConfig {
    model_path: "models/stt/indicconformer.onnx".into(),
    language: Language::Hindi,
})?;

// Streaming transcription
let transcript_stream = stt.transcribe_stream(audio_stream);
while let Some(result) = transcript_stream.next().await {
    println!("Partial: {}", result.text);
}
```

**Features:**
- 22 Indian language support via IndicConformer
- Streaming partial results
- Enhanced decoder with confidence scoring
- Hallucination prevention

### Turn Detection

```rust
use voice_agent_pipeline::turn_detection::{HybridTurnDetector, TurnDetectionResult};

let detector = HybridTurnDetector::new(TurnDetectionConfig {
    semantic_enabled: true,
    endpoint_threshold: 0.85,
    min_silence_ms: 500,
})?;

// Check if user is done speaking
let result = detector.detect(&transcript, &audio_state).await?;
if result.is_end_of_turn {
    // User finished, generate response
}
```

**Modes:**
- **VAD-Only** - Simple silence-based detection
- **Semantic** - SmolLM2-135M for natural turn boundaries
- **Hybrid** - Best of both (recommended)

### TTS (Text-to-Speech)

```rust
use voice_agent_pipeline::tts::{TtsEngine, StreamingTts};

let tts = StreamingTts::new(TtsConfig {
    model: TtsModel::IndicF5,
    voice_id: "hindi-female".into(),
    sample_rate: SampleRate::Hz22050,
})?;

// Word-level streaming
let audio_stream = tts.synthesize_stream(text_stream);
while let Some(audio) = audio_stream.next().await {
    // Send audio immediately
    send_to_client(audio)?;
}
```

**Features:**
- Word-level streaming (low latency)
- Multiple voices per language
- IndicF5 for Indian languages
- Piper fallback for robustness

---

## Orchestrator

The `VoicePipeline` orchestrates all components:

```rust
use voice_agent_pipeline::{VoicePipeline, PipelineConfig, PipelineEvent};

let pipeline = VoicePipeline::new(PipelineConfig {
    vad: VadConfig::default(),
    stt: SttConfig::indicconformer(),
    turn_detection: TurnDetectionConfig::hybrid(),
    tts: TtsConfig::indicf5(),
    barge_in: BargeInConfig::sentence_boundary(),
})?;

// Process audio stream
let event_stream = pipeline.process(audio_stream);
while let Some(event) = event_stream.next().await {
    match event {
        PipelineEvent::VadState(is_speaking) => { /* ... */ }
        PipelineEvent::PartialTranscript(text) => { /* ... */ }
        PipelineEvent::FinalTranscript(text) => { /* ... */ }
        PipelineEvent::AudioOutput(audio) => { /* ... */ }
        PipelineEvent::BargeIn => { /* ... */ }
    }
}
```

---

## Latency Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE LATENCY BUDGET                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Component          Target    Actual    Technique                          │
│  ───────────────────────────────────────────────────────────────────────   │
│  VAD Frame          10ms      8ms       10ms frame processing              │
│  STT Streaming      100ms     95ms      Partial results + prefetch         │
│  Turn Detection     30ms      25ms      SmolLM2-135M semantic              │
│  TTS First Audio    60ms      55ms      Word-level streaming               │
│  ───────────────────────────────────────────────────────────────────────   │
│  TOTAL              200ms     183ms     Before agent processing            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Flags

```toml
[features]
default = ["onnx"]
onnx = ["ort"]                          # ONNX Runtime
candle = ["candle-core", "candle-nn"]   # Pure Rust ML
noise-suppression = ["nnnoiseless"]     # RNNoise
```

---

## Model Files

| Model | Path | Size | Purpose |
|-------|------|------|---------|
| Silero VAD | `models/vad/silero_vad.onnx` | 2MB | Voice detection |
| IndicConformer | `models/stt/indicconformer.onnx` | 600MB | Hindi STT |
| SmolLM2-135M | `models/turn_detection/smollm2.onnx` | 270MB | Turn detection |
| IndicF5 | `models/tts/indicf5/` | 500MB | Indian TTS |
| Piper | `models/tts/piper/` | 50MB | Fallback TTS |
]]>