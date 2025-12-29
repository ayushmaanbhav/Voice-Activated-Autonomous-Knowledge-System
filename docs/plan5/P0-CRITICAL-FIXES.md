# P0 Critical Fixes - Immediate Action Required

These issues must be fixed for full functionality.

---

## 1. InMemorySessionStore Touch Bug

### Problem
`touch()` sets timestamp to 0 instead of current time.

### Location
- `crates/server/src/session.rs:117-120`

### Fix

```rust
// OLD
async fn touch(&self, id: &str) -> Result<(), ServerError> {
    if let Some(meta) = self.metadata.write().get_mut(id) {
        meta.last_activity_ms = 0;  // BUG!
    }
    Ok(())
}

// NEW
async fn touch(&self, id: &str) -> Result<(), ServerError> {
    if let Some(meta) = self.metadata.write().get_mut(id) {
        meta.last_activity_ms = chrono::Utc::now().timestamp_millis() as u64;
    }
    Ok(())
}
```

---

## 2. gRPC Translation Stub

### Problem
Translation returns original text instead of translated text when ONNX models unavailable.

### Location
- `crates/text_processing/src/translation/grpc.rs:125-143`

### Options

#### Option A: Implement HTTP Client (Recommended)

```rust
async fn call_service(&self, text: &str, from: Language, to: Language) -> Result<String> {
    let client = reqwest::Client::new();
    let response = client
        .post(&self.endpoint)
        .json(&serde_json::json!({
            "text": text,
            "source_lang": from.code(),
            "target_lang": to.code(),
        }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| TextProcessingError::TranslationError(e.to_string()))?;

    let result: TranslationResponse = response.json().await
        .map_err(|e| TextProcessingError::TranslationError(e.to_string()))?;

    Ok(result.translated_text)
}
```

#### Option B: Return Error Instead of Silent Failure

```rust
async fn call_service(&self, text: &str, from: Language, to: Language) -> Result<String> {
    Err(TextProcessingError::TranslationError(
        "Translation service not configured. Set TRANSLATION_ENDPOINT or use ONNX models.".to_string()
    ))
}
```

#### Option C: Clear User Warning

```rust
async fn call_service(&self, text: &str, from: Language, to: Language) -> Result<String> {
    tracing::warn!(
        from = %from.code(),
        to = %to.code(),
        "Translation service unavailable - returning original text"
    );
    Ok(format!("[UNTRANSLATED: {}]", text))  // Make it obvious
}
```

---

## 3. PersuasionEngine Not Integrated

### Problem
`PersuasionEngine` is implemented but NOT integrated into the agent's `process()` flow.

### Location
- `crates/agent/src/agent.rs` - Missing integration
- `crates/agent/src/persuasion.rs` - Implementation exists

### Fix

Add to `generate_response()` in `agent.rs` around line 680-700:

```rust
// When stage is ObjectionHandling, use PersuasionEngine
if self.conversation.stage() == ConversationStage::ObjectionHandling {
    let persuasion = PersuasionEngine::new();
    let detected = ObjectionType::detect(user_input);

    if let Some(objection_type) = detected {
        if let Some(response) = persuasion.get_response(&objection_type, &language) {
            builder = builder.with_context(&format!(
                "## Objection Handling Guidance\n\
                 Acknowledge: {}\n\
                 Reframe: {}\n\
                 Evidence: {}\n\
                 Call to Action: {}",
                response.acknowledge,
                response.reframe,
                response.evidence,
                response.call_to_action
            ));
        }
    }
}
```

---

## 4. WebRTC Transport Not Exposed

### Problem
The transport crate has full WebRTC implementation, but server has no signaling endpoint.

### Location
- `crates/server/src/http.rs` - Missing endpoint
- `crates/transport/src/webrtc.rs` - Implementation exists

### Fix

Add to `http.rs`:

```rust
// Add routes
.route("/api/webrtc/offer", post(webrtc_offer))
.route("/api/webrtc/candidate", post(ice_candidate))

// Handler
async fn webrtc_offer(
    State(state): State<AppState>,
    Json(offer): Json<WebRtcOffer>,
) -> Result<Json<WebRtcAnswer>, ServerError> {
    let mut transport = WebRtcTransport::new(WebRtcConfig::default());
    let answer = transport.accept(&offer.sdp).await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    // Store transport in session
    state.sessions.attach_transport(&offer.session_id, transport).await?;

    Ok(Json(WebRtcAnswer { sdp: answer }))
}

async fn ice_candidate(
    State(state): State<AppState>,
    Json(candidate): Json<IceCandidate>,
) -> Result<StatusCode, ServerError> {
    let session = state.sessions.get(&candidate.session_id).await?
        .ok_or(ServerError::NotFound)?;

    session.transport.add_ice_candidate(&candidate.candidate).await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(StatusCode::OK)
}
```

---

## 5. LlmBackend Trait Mismatch

### Problem
`LlmBackend` trait in llm crate does not implement core's `LanguageModel` trait.

### Location
- `crates/llm/src/backend.rs:101-148`
- `crates/core/src/traits/llm.rs:25-85`

### Fix

Add adapter implementation:

```rust
// In crates/llm/src/backend.rs

use voice_agent_core::traits::LanguageModel;
use voice_agent_core::llm_types::{GenerateRequest, GenerateResponse};

impl LanguageModel for OllamaBackend {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, voice_agent_core::Error> {
        // Convert GenerateRequest to Messages
        let messages: Vec<Message> = request.messages.iter().map(|m| {
            Message {
                role: m.role.clone(),
                content: m.content.clone(),
            }
        }).collect();

        let result = LlmBackend::generate(self, &messages).await
            .map_err(|e| voice_agent_core::Error::Llm(e.to_string()))?;

        Ok(GenerateResponse {
            content: result.text,
            finish_reason: result.stop_reason,
            usage: None,
        })
    }

    fn context_size(&self) -> usize {
        self.config.context_size.unwrap_or(4096)
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        LlmBackend::estimate_tokens(self, text)
    }
}
```

---

## Verification Checklist

After applying fixes:

- [ ] `cargo check` passes without errors
- [ ] `cargo test` passes
- [ ] Session cleanup works correctly (touch bug fixed)
- [ ] Translation failures are handled gracefully
- [ ] PersuasionEngine responses appear in objection handling
- [ ] WebRTC signaling endpoints accessible
- [ ] LLM can be used via LanguageModel trait

---

## Testing Commands

```bash
# Run tests
cargo test -p voice-agent-server
cargo test -p voice-agent-text-processing
cargo test -p voice-agent-agent
cargo test -p voice-agent-llm

# Manual verification
curl -X POST http://localhost:3000/api/webrtc/offer -d '{"session_id":"test","sdp":"..."}'
```
