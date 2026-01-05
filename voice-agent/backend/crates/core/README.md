<![CDATA[# voice-agent-core

> Foundation traits, types, and language support for the Voice Agent platform

---

## Overview

The `core` crate provides the foundational abstractions that all other crates depend on. It defines:

- **13 Core Traits** - Interfaces for all pluggable components
- **22 Indian Languages** - Full language and script support
- **Audio Types** - Frames, encodings, sample rates
- **Conversation Types** - Turns, stages, customer profiles
- **Compliance Types** - PII, regulatory violations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CORE CRATE MODULES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                            TRAITS                                     │  │
│  │                                                                      │  │
│  │  SpeechToText │ TextToSpeech │ VoiceActivityDetector                │  │
│  │  LanguageModel │ Retriever │ TextProcessor │ FrameProcessor         │  │
│  │  ConversationFSM │ ComplianceChecker │ PIIRedactor │ Translator     │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │    Audio    │  │Conversation │  │   Language  │  │   Compliance    │   │
│  │    Types    │  │    Types    │  │   Support   │  │     Types       │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Traits

### SpeechToText

```rust
#[async_trait]
pub trait SpeechToText: Send + Sync + 'static {
    async fn transcribe(&self, audio: &AudioFrame) -> Result<TranscriptFrame>;

    fn transcribe_stream(
        &self,
        audio: impl Stream<Item = AudioFrame> + Send,
    ) -> impl Stream<Item = Result<TranscriptFrame>> + Send;

    fn supported_languages(&self) -> &[Language];
}
```

### TextToSpeech

```rust
#[async_trait]
pub trait TextToSpeech: Send + Sync + 'static {
    async fn synthesize(&self, text: &str, config: &VoiceConfig) -> Result<AudioFrame>;

    fn synthesize_stream(
        &self,
        text: impl Stream<Item = String> + Send,
        config: &VoiceConfig,
    ) -> impl Stream<Item = Result<AudioFrame>> + Send;
}
```

### LanguageModel

```rust
#[async_trait]
pub trait LanguageModel: Send + Sync + 'static {
    async fn generate(&self, messages: &[Message], tools: &[ToolDefinition])
        -> Result<GenerateResponse>;

    fn generate_stream(&self, messages: &[Message])
        -> impl Stream<Item = Result<StreamChunk>> + Send;
}
```

---

## Language Support

### 22 Indian Languages

| Code | Language | Script | Status |
|------|----------|--------|--------|
| `en` | English | Latin | ✅ Full |
| `hi` | Hindi | Devanagari | ✅ Full |
| `ta` | Tamil | Tamil | ✅ Full |
| `te` | Telugu | Telugu | ✅ Full |
| `kn` | Kannada | Kannada | ✅ Full |
| `ml` | Malayalam | Malayalam | ✅ Full |
| `bn` | Bengali | Bengali | ✅ Full |
| `mr` | Marathi | Devanagari | ✅ Full |
| `gu` | Gujarati | Gujarati | ✅ Full |
| `pa` | Punjabi | Gurmukhi | ✅ Full |
| `or` | Odia | Odia | ✅ Full |
| `as` | Assamese | Bengali | ✅ Full |
| `brx` | Bodo | Devanagari | ✅ Full |
| `doi` | Dogri | Devanagari | ✅ Full |
| `kok` | Konkani | Devanagari | ✅ Full |
| `ks` | Kashmiri | Arabic/Devanagari | ✅ Full |
| `mai` | Maithili | Devanagari | ✅ Full |
| `mni` | Manipuri | Bengali/Meetei | ✅ Full |
| `ne` | Nepali | Devanagari | ✅ Full |
| `sa` | Sanskrit | Devanagari | ✅ Full |
| `sat` | Santali | Ol Chiki | ✅ Full |
| `sd` | Sindhi | Arabic/Devanagari | ✅ Full |
| `ur` | Urdu | Arabic | ✅ Full |

### Script Detection

```rust
pub fn detect_script(text: &str) -> Script {
    // Detects: Devanagari, Bengali, Tamil, Telugu, Kannada,
    // Malayalam, Gujarati, Odia, Gurmukhi, Arabic, Ol Chiki
}

pub fn normalize_indic_numerals(text: &str) -> String {
    // Converts: ०१२, ০১২, ௦௧௨, ౦౧౨ → 012
}
```

---

## Core Types

### AudioFrame

```rust
pub struct AudioFrame {
    pub data: Vec<i16>,
    pub sample_rate: SampleRate,
    pub channels: Channels,
    pub timestamp_ms: u64,
}

pub enum SampleRate {
    Hz8000,
    Hz16000,
    Hz22050,
    Hz44100,
    Hz48000,
}
```

### Conversation Types

```rust
pub enum ConversationStage {
    Greeting,
    Exploration,
    ValueProposition,
    ObjectionHandling,
    Closing,
    FollowUp,
}

pub struct Turn {
    pub role: TurnRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, Value>,
}
```

### Compliance Types

```rust
pub enum PIIType {
    PersonName,
    PhoneNumber,
    Aadhaar,
    PAN,
    Email,
    BankAccount,
    // ... more
}

pub struct PIIEntity {
    pub pii_type: PIIType,
    pub text: String,
    pub span: (usize, usize),
    pub confidence: f32,
}
```

---

## Usage

```rust
use voice_agent_core::{
    traits::{SpeechToText, TextToSpeech, LanguageModel},
    audio::{AudioFrame, SampleRate, Channels},
    language::Language,
    conversation::{Turn, ConversationStage},
};

// Implement traits for your components
impl SpeechToText for MySTTEngine {
    async fn transcribe(&self, audio: &AudioFrame) -> Result<TranscriptFrame> {
        // Implementation
    }
}
```

---

## Dependencies

- `async-trait` - Async trait support
- `futures` - Stream trait
- `serde` - Serialization
- `chrono` - Timestamps
- `uuid` - Unique identifiers
]]>