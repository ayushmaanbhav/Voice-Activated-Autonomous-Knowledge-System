//! Push-to-Talk API Endpoint
//!
//! Simplified audio processing without VAD/turn detection.
//! User records audio, sends it, and receives response.
//!
//! Flow:
//! 1. Receive audio (webm/opus, base64 encoded)
//! 2. Convert to PCM 16kHz mono
//! 3. STT via Rust IndicConformer
//! 4. LLM processing via DomainAgent (with RAG + tools)
//! 5. TTS response generation
//! 6. Return transcripts and audio

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::{Event, Sse}},
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use futures::stream::Stream;
use std::convert::Infallible;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::state::AppState;

// Pre-compiled regex patterns for markdown stripping (compiled once at startup)
static RE_HEADERS: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"(?m)^#{1,6}\s*").unwrap());
static RE_BOLD_ASTERISK: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\*\*([^*]+)\*\*").unwrap());
static RE_BOLD_UNDERSCORE: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"__([^_]+)__").unwrap());
static RE_ITALIC: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\*([^*\n]+)\*").unwrap());
static RE_INLINE_CODE: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"`([^`]+)`").unwrap());
static RE_BULLET_POINTS: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"(?m)^[\s]*[-*+]\s+").unwrap());
static RE_NUMBERED_LIST: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"(?m)^\s*\d+\.\s+").unwrap());
static RE_LINKS: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\[([^\]]+)\]\([^)]+\)").unwrap());
static RE_MULTIPLE_NEWLINES: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\n{3,}").unwrap());

// Service URLs loaded from environment with fallback defaults
static WHISPER_SERVICE_URL: Lazy<String> = Lazy::new(|| {
    std::env::var("STT_URL").unwrap_or_else(|_| "http://127.0.0.1:8091".to_string())
});
static TTS_SERVICE_URL: Lazy<String> = Lazy::new(|| {
    std::env::var("TTS_URL").unwrap_or_else(|_| "http://127.0.0.1:8092".to_string())
});

use voice_agent_pipeline::stt::{IndicConformerStt, IndicConformerConfig};

/// STT pool size (number of concurrent STT instances)
/// Can be overridden via STT_POOL_SIZE env var
static STT_POOL_SIZE: Lazy<usize> = Lazy::new(|| {
    std::env::var("STT_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2) // Default to 2 concurrent instances
});

/// STT Pool - allows multiple concurrent STT requests
/// Uses a channel to distribute IndicConformerStt instances
struct SttPool {
    sender: tokio::sync::mpsc::Sender<IndicConformerStt>,
    receiver: Mutex<tokio::sync::mpsc::Receiver<IndicConformerStt>>,
}

/// Lazy-initialized STT pool for concurrent request handling
static STT_POOL: Lazy<Mutex<Option<SttPool>>> = Lazy::new(|| Mutex::new(None));

/// Request for push-to-talk processing
#[derive(Debug, Deserialize)]
pub struct PttRequest {
    /// Base64 encoded audio data
    pub audio: String,
    /// Audio format (webm, wav, etc.)
    pub audio_format: String,
    /// Language code (hi, en, ta, etc.)
    pub language: String,
    /// Optional session ID for conversation continuity
    /// If provided, reuses existing session; otherwise creates new one
    #[serde(default)]
    pub session_id: Option<String>,
}

/// Response from push-to-talk processing
#[derive(Debug, Serialize)]
pub struct PttResponse {
    /// User's transcribed text (raw from STT)
    pub user_text: String,
    /// User's text after grammar correction
    pub user_text_corrected: Option<String>,
    /// User's text before translation (if translated)
    pub user_text_original: Option<String>,
    /// Assistant's response text (in user's language)
    pub assistant_text: String,
    /// Assistant's text before translation (if translated)
    pub assistant_text_original: Option<String>,
    /// Base64 encoded audio response
    pub audio_response: Option<String>,
    /// Audio response format
    pub audio_format: Option<String>,
    /// Processing metrics
    pub metrics: PttMetrics,
    /// Current processing phase (for UI status updates)
    pub phase: String,
    /// Session ID for conversation continuity (send back in next request)
    pub session_id: String,
}

/// Processing metrics with per-phase timing
#[derive(Debug, Serialize, Default)]
pub struct PttMetrics {
    pub stt_ms: u64,
    pub grammar_ms: u64,
    pub llm_ms: u64,
    pub tts_ms: u64,
    pub total_ms: u64,
}

/// Check if language is English
fn is_english(language: &str) -> bool {
    matches!(language.to_lowercase().as_str(), "en" | "english")
}

/// Call faster-whisper HTTP service for English STT
async fn transcribe_with_whisper(audio: &[f32], language: &str) -> Result<String, String> {
    // Convert f32 samples to bytes
    let audio_bytes: Vec<u8> = audio
        .iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    let audio_b64 = BASE64.encode(&audio_bytes);

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/transcribe", &*WHISPER_SERVICE_URL))
        .json(&serde_json::json!({
            "audio": audio_b64,
            "language": language,
            "sample_rate": 16000
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Whisper service request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Whisper service error {}: {}", status, body));
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse whisper response: {}", e))?;

    let text = result["text"].as_str().unwrap_or("").to_string();
    let proc_time = result["processing_time_seconds"].as_f64().unwrap_or(0.0);

    tracing::info!(
        "Faster-whisper transcribed in {:.2}s: '{}'",
        proc_time,
        if text.len() > 100 { &text[..100] } else { &text }
    );

    Ok(text)
}

/// Initialize STT pool if not already initialized
/// Creates multiple IndicConformerStt instances for concurrent processing
fn init_stt_pool(language: &str) -> Result<(), String> {
    let mut pool_guard = STT_POOL.lock();
    if pool_guard.is_some() {
        return Ok(()); // Already initialized
    }

    let pool_size = *STT_POOL_SIZE;
    tracing::info!(
        pool_size = pool_size,
        language = language,
        "Initializing IndicConformer STT pool for PTT..."
    );

    let (tx, rx) = tokio::sync::mpsc::channel(pool_size);
    let model_dir = std::path::Path::new("models/stt/indicconformer");

    // Create pool_size STT instances
    for i in 0..pool_size {
        let config = IndicConformerConfig {
            language: language.to_string(),
            ..Default::default()
        };
        match IndicConformerStt::new(model_dir, config) {
            Ok(stt) => {
                if tx.try_send(stt).is_err() {
                    tracing::warn!("Failed to add STT instance {} to pool", i);
                }
            }
            Err(e) => {
                return Err(format!("Failed to initialize STT instance {}: {}", i, e));
            }
        }
    }

    *pool_guard = Some(SttPool {
        sender: tx,
        receiver: Mutex::new(rx),
    });

    tracing::info!(
        pool_size = pool_size,
        "IndicConformer STT pool initialized for PTT"
    );
    Ok(())
}

/// Acquire an STT instance from the pool (waits if all instances are busy)
async fn acquire_stt() -> Result<IndicConformerStt, String> {
    loop {
        // Try to acquire from pool
        {
            let pool_guard = STT_POOL.lock();
            if let Some(pool) = pool_guard.as_ref() {
                let mut rx = pool.receiver.lock();
                match rx.try_recv() {
                    Ok(stt) => return Ok(stt),
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                        return Err("STT pool channel disconnected".to_string());
                    }
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        // All instances busy, will retry after sleep
                    }
                }
            } else {
                return Err("STT pool not initialized".to_string());
            }
        }
        // Wait before retrying (all instances are busy)
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
}

/// Release an STT instance back to the pool
fn release_stt(stt: IndicConformerStt) {
    let pool_guard = STT_POOL.lock();
    if let Some(pool) = pool_guard.as_ref() {
        let _ = pool.sender.try_send(stt);
    }
}

/// Strip markdown formatting for TTS output
/// Removes headers (##), bold (**), italic (*), bullets, etc.
/// Uses pre-compiled static regex patterns for performance.
fn strip_markdown_for_tts(text: &str) -> String {
    let mut result = text.to_string();

    // Remove headers (## Header)
    result = RE_HEADERS.replace_all(&result, "").to_string();

    // Remove bold (**text** or __text__)
    result = RE_BOLD_ASTERISK.replace_all(&result, "$1").to_string();
    result = RE_BOLD_UNDERSCORE.replace_all(&result, "$1").to_string();

    // Remove italic (*text* or _text_) - bold (**) already removed above
    // Simple pattern works since bold was already stripped
    result = RE_ITALIC.replace_all(&result, "$1").to_string();

    // Remove inline code (`code`)
    result = RE_INLINE_CODE.replace_all(&result, "$1").to_string();

    // Remove bullet points (- item or * item) at start of lines
    result = RE_BULLET_POINTS.replace_all(&result, "").to_string();

    // Remove numbered lists (1. item)
    result = RE_NUMBERED_LIST.replace_all(&result, "").to_string();

    // Remove links [text](url) -> text
    result = RE_LINKS.replace_all(&result, "$1").to_string();

    // Remove multiple newlines
    result = RE_MULTIPLE_NEWLINES.replace_all(&result, "\n\n").to_string();

    // Trim whitespace
    result.trim().to_string()
}

/// Call IndicF5 TTS HTTP service to synthesize speech
async fn synthesize_with_tts(text: &str, language: &str) -> Result<(String, String), String> {
    // Strip markdown formatting for cleaner TTS output
    let clean_text = strip_markdown_for_tts(text);

    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/synthesize", &*TTS_SERVICE_URL))
        .json(&serde_json::json!({
            "text": clean_text,
            "language": language
        }))
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("TTS service request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("TTS service error {}: {}", status, body));
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse TTS response: {}", e))?;

    let audio = result["audio"].as_str().unwrap_or("").to_string();
    let format = result["format"].as_str().unwrap_or("wav").to_string();
    let proc_time = result["processing_time_seconds"].as_f64().unwrap_or(0.0);
    let duration = result["duration_seconds"].as_f64().unwrap_or(0.0);

    tracing::info!(
        "IndicF5 TTS synthesized in {:.2}s, audio duration: {:.2}s",
        proc_time,
        duration
    );

    Ok((audio, format))
}

/// Handle push-to-talk request
pub async fn handle_ptt(
    State(state): State<AppState>,
    Json(request): Json<PttRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let mut metrics = PttMetrics::default();

    tracing::info!(
        "PTT request: language={}, audio_format={}, audio_size={}",
        request.language,
        request.audio_format,
        request.audio.len()
    );

    // 1. Decode audio from base64
    let audio_bytes = match BASE64.decode(&request.audio) {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Invalid base64 audio: {}", e) })),
            );
        }
    };

    // 2. Convert audio to PCM 16kHz mono f32
    let pcm_f32 = match convert_to_pcm_f32(&audio_bytes, &request.audio_format).await {
        Ok(pcm) => pcm,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Audio conversion failed: {}", e) })),
            );
        }
    };

    tracing::info!("Converted audio to PCM f32: {} samples", pcm_f32.len());

    // 3. Initialize STT and run based on language
    let stt_start = std::time::Instant::now();
    let use_english = is_english(&request.language);

    let stt_text = if use_english {
        // Use faster-whisper HTTP service for English
        match transcribe_with_whisper(&pcm_f32, &request.language).await {
            Ok(text) => text,
            Err(e) => {
                tracing::error!("Faster-whisper STT error: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": format!("Whisper STT failed: {}", e) })),
                );
            }
        }
    } else {
        // Use Rust IndicConformer STT for Indian languages (from pool)
        if let Err(e) = init_stt_pool(&request.language) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            );
        }

        // Acquire STT instance from pool (waits if all busy)
        let stt = match acquire_stt().await {
            Ok(stt) => stt,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e })),
                );
            }
        };

        // Reset STT state before processing new audio
        stt.reset();

        tracing::info!(
            pcm_samples = pcm_f32.len(),
            duration_secs = pcm_f32.len() as f32 / 16000.0,
            "IndicConformer processing audio"
        );

        // Process the audio
        if let Err(e) = stt.process(&pcm_f32) {
            tracing::error!("IndicConformer STT process error: {}", e);
        }

        // Finalize to get transcript
        let final_result = stt.finalize();
        tracing::info!(
            text_len = final_result.text.len(),
            text = %final_result.text,
            confidence = final_result.confidence,
            "IndicConformer finalized transcript"
        );

        // Release STT instance back to pool
        release_stt(stt);

        final_result.text
    };
    metrics.stt_ms = stt_start.elapsed().as_millis() as u64;

    tracing::info!("STT result ({}): '{}'", if use_english { "faster-whisper" } else { "IndicConformer" }, stt_text);

    if stt_text.is_empty() {
        let no_speech_msg = if use_english {
            "I didn't hear anything. Please speak again."
        } else {
            "मुझे कुछ सुनाई नहीं दिया। कृपया फिर से बोलें।"
        };
        // Preserve session_id if provided, or generate new one
        let session_id = request.session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "user_text": "",
                "user_text_corrected": null,
                "user_text_original": null,
                "assistant_text": no_speech_msg,
                "assistant_text_original": if use_english { serde_json::Value::Null } else { serde_json::json!("I didn't hear anything. Please speak again.") },
                "audio_response": null,
                "metrics": metrics,
                "phase": "complete",
                "session_id": session_id
            })),
        );
    }

    // 4. Deterministic phonetic error correction (SymSpell + confusion rules)
    let grammar_start = std::time::Instant::now();
    let (corrected, corrections) = state.phonetic_corrector.correct(&stt_text);
    let corrected_text = if !corrections.is_empty() {
        tracing::info!(
            original = stt_text.as_str(),
            corrected = corrected.as_str(),
            corrections = ?corrections.iter().map(|c| format!("{}->{}({})", c.original, c.corrected, c.rule)).collect::<Vec<_>>(),
            "Phonetic correction applied"
        );
        Some(corrected)
    } else {
        None
    };
    metrics.grammar_ms = grammar_start.elapsed().as_millis() as u64;

    // Use corrected text for LLM if available
    let text_for_llm = corrected_text.as_ref().unwrap_or(&stt_text);

    // 5. Call LLM via Agent pipeline (with RAG + tools)
    let llm_start = std::time::Instant::now();
    let (llm_response, session_id) = match process_with_agent(
        &state,
        text_for_llm,
        &request.language,
        request.session_id.as_deref(),
    ).await {
        Ok((response, sid)) => (response, sid),
        Err(e) => {
            tracing::error!("Agent processing failed: {}", e);
            // Fallback to basic acknowledgment - generate new session_id
            let fallback_sid = uuid::Uuid::new_v4().to_string();
            (format_fallback_response(text_for_llm, &request.language), fallback_sid)
        }
    };
    metrics.llm_ms = llm_start.elapsed().as_millis() as u64;

    tracing::info!("LLM response: '{}'", llm_response);

    // 6. Generate TTS via IndicF5 service
    let tts_start = std::time::Instant::now();
    let audio_response = match synthesize_with_tts(&llm_response, &request.language).await {
        Ok((audio_b64, format)) => {
            tracing::info!("TTS generated {} bytes of {} audio", audio_b64.len(), format);
            Some(audio_b64)
        }
        Err(e) => {
            tracing::warn!("TTS generation failed: {}", e);
            None
        }
    };
    let audio_format = if audio_response.is_some() { Some("wav".to_string()) } else { None };
    metrics.tts_ms = tts_start.elapsed().as_millis() as u64;

    metrics.total_ms = start.elapsed().as_millis() as u64;

    // 7. Return response
    let response = PttResponse {
        user_text: stt_text.clone(),
        user_text_corrected: corrected_text,
        user_text_original: None,
        assistant_text: llm_response,
        assistant_text_original: None,
        audio_response,
        audio_format,
        metrics,
        phase: "complete".to_string(),
        session_id,
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Convert audio to PCM 16kHz mono f32 samples
async fn convert_to_pcm_f32(audio_bytes: &[u8], format: &str) -> Result<Vec<f32>, String> {
    match format {
        "webm" | "opus" => convert_webm_to_pcm_f32(audio_bytes).await,
        "wav" => extract_wav_pcm_f32(audio_bytes),
        "pcm" => {
            // Assume PCM16 little-endian
            let samples: Vec<f32> = audio_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();
            Ok(samples)
        }
        _ => Err(format!("Unsupported audio format: {}", format)),
    }
}

/// Convert WebM/Opus to PCM f32 using ffmpeg
async fn convert_webm_to_pcm_f32(webm_bytes: &[u8]) -> Result<Vec<f32>, String> {
    use tokio::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Create unique temp file names using timestamp + random component
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let unique_id = format!("{}_{}", std::process::id(), timestamp);
    let input_path = format!("/tmp/ptt_input_{}.webm", unique_id);
    let output_path = format!("/tmp/ptt_output_{}.raw", unique_id);

    // Write input file
    tokio::fs::write(&input_path, webm_bytes)
        .await
        .map_err(|e| format!("Failed to write temp file: {}", e))?;

    // Run ffmpeg to convert to raw PCM16
    let output = Command::new("ffmpeg")
        .args([
            "-y",
            "-i", &input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            &output_path,
        ])
        .output()
        .await
        .map_err(|e| format!("FFmpeg failed: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!("FFmpeg error: {}", stderr);
        // Cleanup
        let _ = tokio::fs::remove_file(&input_path).await;
        let _ = tokio::fs::remove_file(&output_path).await;
        return Err(format!("FFmpeg conversion failed: {}", stderr));
    }

    // Read output as PCM16 bytes
    let pcm_bytes = tokio::fs::read(&output_path)
        .await
        .map_err(|e| format!("Failed to read output: {}", e))?;

    // Cleanup
    let _ = tokio::fs::remove_file(&input_path).await;
    let _ = tokio::fs::remove_file(&output_path).await;

    // Convert PCM16 to f32
    let samples: Vec<f32> = pcm_bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            sample as f32 / 32768.0
        })
        .collect();

    Ok(samples)
}

/// Extract PCM f32 data from WAV file
fn extract_wav_pcm_f32(wav_bytes: &[u8]) -> Result<Vec<f32>, String> {
    // Simple WAV header parsing
    if wav_bytes.len() < 44 {
        return Err("WAV file too short".to_string());
    }

    // Find 'data' chunk
    let mut pos = 12; // Skip RIFF header
    while pos + 8 < wav_bytes.len() {
        let chunk_id = &wav_bytes[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            wav_bytes[pos + 4],
            wav_bytes[pos + 5],
            wav_bytes[pos + 6],
            wav_bytes[pos + 7],
        ]) as usize;

        if chunk_id == b"data" {
            let data_start = pos + 8;
            let data_end = data_start + chunk_size;
            if data_end <= wav_bytes.len() {
                // Convert PCM16 to f32
                let pcm_bytes = &wav_bytes[data_start..data_end];
                let samples: Vec<f32> = pcm_bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect();
                return Ok(samples);
            }
        }

        pos += 8 + chunk_size;
    }

    Err("Could not find data chunk in WAV".to_string())
}

/// Process user input through the full Agent pipeline (LLM + RAG + tools)
/// Returns (response_text, session_id) for conversation continuity
async fn process_with_agent(
    state: &AppState,
    user_text: &str,
    language: &str,
    existing_session_id: Option<&str>,
) -> Result<(String, String), String> {
    use voice_agent_agent::AgentConfig;

    // Try to reuse existing session if provided
    let session = if let Some(sid) = existing_session_id {
        if let Some(existing) = state.sessions.get(sid) {
            tracing::info!(
                session_id = %sid,
                "Reusing existing PTT session for conversation continuity"
            );
            existing
        } else {
            tracing::warn!(
                session_id = %sid,
                "Session not found, creating new one"
            );
            create_new_session(state, language)?
        }
    } else {
        create_new_session(state, language)?
    };

    let session_id = session.id.clone();

    tracing::info!(
        session_id = %session_id,
        language = ?language,
        "Processing PTT request"
    );

    // Process through agent pipeline
    let response = session
        .agent
        .process(user_text)
        .await
        .map_err(|e| format!("Agent processing failed: {}", e))?;

    // Don't remove session - keep it for conversation continuity
    // Sessions will be cleaned up by timeout/explicit end

    Ok((response, session_id))
}

/// Create a new agent session
fn create_new_session(
    state: &AppState,
    language: &str,
) -> Result<std::sync::Arc<crate::session::Session>, String> {
    use voice_agent_agent::AgentConfig;

    let mut config = AgentConfig::default();
    config.language = language.to_string();

    let session = state
        .sessions
        .create_with_full_integration(
            config,
            state.vector_store.clone(),
            Some(state.tools.clone()),
        )
        .map_err(|e| format!("Failed to create session: {}", e))?;

    tracing::info!(
        session_id = %session.id,
        language = ?language,
        "Created new PTT session"
    );

    Ok(session)
}

/// Fallback response when agent processing fails
fn format_fallback_response(user_text: &str, language: &str) -> String {
    if language == "hi" {
        format!("आपने कहा: '{}'. कृपया थोड़ी देर बाद पुनः प्रयास करें।", user_text)
    } else {
        format!("You said: '{}'. Please try again in a moment.", user_text)
    }
}

/// Language-specific greeting messages
fn get_greeting(language: &str) -> (&'static str, &'static str) {
    // Returns (greeting in target language, English translation)
    match language.to_lowercase().as_str() {
        "en" | "english" => (
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
        "hi" | "hindi" => (
            "नमस्ते! मैं आपका कोटक गोल्ड लोन सहायक हूं। आज मैं आपकी कैसे मदद कर सकता हूं?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
        "ta" | "tamil" => (
            "வணக்கம்! நான் உங்கள் கோடக் கோல்ட் லோன் உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
        "te" | "telugu" => (
            "నమస్కారం! నేను మీ కోటక్ గోల్డ్ లోన్ అసిస్టెంట్. ఈ రోజు నేను మీకు ఎలా సహాయం చేయగలను?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
        "kn" | "kannada" => (
            "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಕೋಟಕ್ ಗೋಲ್ಡ್ ಲೋನ್ ಸಹಾಯಕ. ಇಂದು ನಾನು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಹುದು?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
        "ml" | "malayalam" => (
            "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ കോട്ടക് ഗോൾഡ് ലോൺ അസിസ്റ്റന്റ് ആണ്. ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കാം?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
        _ => (
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?",
            "Hello! I'm your Kotak Gold Loan assistant. How can I help you today?"
        ),
    }
}

/// Request for greeting
#[derive(Debug, Deserialize)]
pub struct GreetingRequest {
    /// Language code (hi, en, ta, etc.)
    pub language: String,
}

/// Response with greeting
#[derive(Debug, Serialize)]
pub struct GreetingResponse {
    /// Greeting text in the requested language
    pub greeting: String,
    /// English translation of the greeting
    pub greeting_english: String,
    /// Language code
    pub language: String,
}

/// Get language-specific greeting
pub async fn get_greeting_handler(
    Json(request): Json<GreetingRequest>,
) -> impl IntoResponse {
    let (greeting, greeting_english) = get_greeting(&request.language);

    (
        StatusCode::OK,
        Json(GreetingResponse {
            greeting: greeting.to_string(),
            greeting_english: greeting_english.to_string(),
            language: request.language,
        }),
    )
}

/// Request for translating messages
#[derive(Debug, Deserialize)]
pub struct TranslateRequest {
    /// Messages to translate
    pub messages: Vec<TranslateMessage>,
    /// Target language code (hi, en, ta, etc.)
    pub target_language: String,
    /// Source language code (optional, auto-detect if not provided)
    pub source_language: Option<String>,
}

/// A single message to translate
#[derive(Debug, Deserialize)]
pub struct TranslateMessage {
    /// Message ID (for correlation in response)
    pub id: String,
    /// Text to translate
    pub text: String,
    /// Role (user or assistant)
    pub role: String,
}

/// Response with translated messages
#[derive(Debug, Serialize)]
pub struct TranslateResponse {
    /// Translated messages
    pub messages: Vec<TranslatedMessage>,
    /// Target language code
    pub target_language: String,
    /// Source language code
    pub source_language: String,
}

/// A translated message
#[derive(Debug, Serialize)]
pub struct TranslatedMessage {
    /// Message ID (matches request)
    pub id: String,
    /// Translated text
    pub text: String,
    /// Original text (for reference)
    pub original: String,
    /// Role (user or assistant)
    pub role: String,
}

/// Translate messages to a target language
pub async fn translate_handler(
    State(state): State<AppState>,
    Json(request): Json<TranslateRequest>,
) -> impl IntoResponse {
    use voice_agent_core::Language;

    let source_lang_str = request.source_language.clone().unwrap_or_else(|| "en".to_string());

    tracing::info!(
        "Translate request: {} messages, {} -> {}",
        request.messages.len(),
        source_lang_str,
        request.target_language
    );

    // If source and target are the same, just return the original messages
    if source_lang_str == request.target_language {
        let messages: Vec<TranslatedMessage> = request.messages
            .into_iter()
            .map(|m| TranslatedMessage {
                id: m.id,
                text: m.text.clone(),
                original: m.text,
                role: m.role,
            })
            .collect();

        return (
            StatusCode::OK,
            Json(TranslateResponse {
                messages,
                target_language: request.target_language,
                source_language: source_lang_str,
            }),
        );
    }

    // Parse language codes to Language enum
    let source_lang = match Language::from_str_loose(&source_lang_str) {
        Some(lang) => lang,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(TranslateResponse {
                    messages: vec![],
                    target_language: request.target_language,
                    source_language: source_lang_str,
                }),
            );
        }
    };

    let target_lang = match Language::from_str_loose(&request.target_language) {
        Some(lang) => lang,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(TranslateResponse {
                    messages: vec![],
                    target_language: request.target_language,
                    source_language: source_lang_str,
                }),
            );
        }
    };

    // Translate each message
    let mut translated_messages = Vec::new();

    for msg in request.messages {
        let translated_text = match state.translator.translate(
            &msg.text,
            source_lang,
            target_lang,
        ).await {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!(
                    "Translation failed for message {}: {}. Using original.",
                    msg.id, e
                );
                // Fallback to original text
                msg.text.clone()
            }
        };

        translated_messages.push(TranslatedMessage {
            id: msg.id,
            text: translated_text,
            original: msg.text,
            role: msg.role,
        });
    }

    (
        StatusCode::OK,
        Json(TranslateResponse {
            messages: translated_messages,
            target_language: request.target_language,
            source_language: source_lang_str,
        }),
    )
}

/// Health check for PTT service
pub async fn ptt_health() -> impl IntoResponse {
    // Check if STT model exists
    let model_path = std::path::Path::new("models/stt/indicconformer/assets/encoder.onnx");
    let mask_path = std::path::Path::new("models/stt/indicconformer/assets/language_masks.json");
    let stt_ok = model_path.exists() && mask_path.exists();

    if stt_ok {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "stt_backend": "rust_indicconformer",
                "model_path": model_path.to_string_lossy(),
                "mask_loaded": mask_path.exists()
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "error",
                "error": "STT model not found",
                "expected_model": model_path.to_string_lossy(),
                "expected_mask": mask_path.to_string_lossy()
            })),
        )
    }
}

/// Request for STT-only testing
#[derive(Debug, Deserialize)]
pub struct SttTestRequest {
    /// Base64 encoded WAV audio
    pub audio: String,
    /// Language code (hi, en, etc.)
    pub language: String,
}

/// STT-only endpoint for testing (no LLM/TTS)
pub async fn handle_stt_test(
    Json(request): Json<SttTestRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    tracing::info!(
        "STT test request: language={}, audio_size={}",
        request.language,
        request.audio.len()
    );

    // 1. Decode audio from base64
    let audio_bytes = match BASE64.decode(&request.audio) {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Invalid base64 audio: {}", e) })),
            );
        }
    };

    // 2. Convert audio to PCM 16kHz mono f32
    let pcm_f32 = match convert_to_pcm_f32(&audio_bytes, "wav").await {
        Ok(pcm) => pcm,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Audio conversion failed: {}", e) })),
            );
        }
    };

    tracing::info!("Converted to {} PCM samples", pcm_f32.len());

    // 3. Run STT
    let stt_start = std::time::Instant::now();
    let transcription = if is_english(&request.language) {
        // Use Whisper for English
        match transcribe_with_whisper(&pcm_f32, &request.language).await {
            Ok(text) => text,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": format!("Whisper STT failed: {}", e) })),
                );
            }
        }
    } else {
        // Use Rust IndicConformer for Indian languages (from pool)
        if let Err(e) = init_stt_pool(&request.language) {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e })),
            );
        }

        // Acquire STT instance from pool
        let stt = match acquire_stt().await {
            Ok(stt) => stt,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e })),
                );
            }
        };

        // Reset STT state
        stt.reset();

        // Process the audio
        if let Err(e) = stt.process(&pcm_f32) {
            tracing::error!("IndicConformer STT process error: {}", e);
        }

        // Finalize
        let result = stt.finalize();

        // Release STT instance back to pool
        release_stt(stt);

        result.text
    };
    let stt_time = stt_start.elapsed();

    let total_time = start.elapsed();

    tracing::info!(
        "STT test completed: text='{}', stt_time={:?}, total_time={:?}",
        transcription,
        stt_time,
        total_time
    );

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "text": transcription,
            "language": request.language,
            "stt_ms": stt_time.as_millis(),
            "total_ms": total_time.as_millis()
        })),
    )
}

/// SSE event types for streaming PTT
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PttEvent {
    /// User's transcribed text (sent as soon as STT completes)
    UserText {
        text: String,
        corrected: Option<String>,
        session_id: String,
    },
    /// Assistant's response text (sent when LLM completes)
    AssistantText {
        text: String,
    },
    /// Audio response ready (sent when TTS completes)
    AudioReady {
        audio: String,
        format: String,
    },
    /// Processing complete with metrics
    Complete {
        metrics: PttMetrics,
    },
    /// Error occurred
    Error {
        message: String,
    },
}

/// Handle push-to-talk request with SSE streaming
/// This endpoint streams events as processing progresses:
/// 1. user_text - sent immediately after STT completes
/// 2. assistant_text - sent when LLM responds
/// 3. audio_ready - sent when TTS completes
/// 4. complete - sent with final metrics
pub async fn handle_ptt_stream(
    State(state): State<AppState>,
    Json(request): Json<PttRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(10);

    // Spawn async task to process and send events
    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let mut metrics = PttMetrics::default();

        // Helper to send event
        let send_event = |tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>, event: PttEvent| {
            let data = serde_json::to_string(&event).unwrap_or_default();
            let _ = tx.try_send(Ok(Event::default().data(data)));
        };

        tracing::info!(
            "PTT stream request: language={}, audio_format={}, audio_size={}",
            request.language,
            request.audio_format,
            request.audio.len()
        );

        // 1. Decode audio from base64
        let audio_bytes = match BASE64.decode(&request.audio) {
            Ok(bytes) => bytes,
            Err(e) => {
                send_event(&tx, PttEvent::Error { message: format!("Invalid base64 audio: {}", e) });
                return;
            }
        };

        // 2. Convert audio to PCM 16kHz mono f32
        let pcm_f32 = match convert_to_pcm_f32(&audio_bytes, &request.audio_format).await {
            Ok(pcm) => pcm,
            Err(e) => {
                send_event(&tx, PttEvent::Error { message: format!("Audio conversion failed: {}", e) });
                return;
            }
        };

        // 3. STT
        let stt_start = std::time::Instant::now();
        let use_english = is_english(&request.language);

        let stt_text = if use_english {
            match transcribe_with_whisper(&pcm_f32, &request.language).await {
                Ok(text) => text,
                Err(e) => {
                    send_event(&tx, PttEvent::Error { message: format!("STT failed: {}", e) });
                    return;
                }
            }
        } else {
            // Use Rust IndicConformer STT for Indian languages (from pool)
            if let Err(e) = init_stt_pool(&request.language) {
                send_event(&tx, PttEvent::Error { message: e });
                return;
            }

            // Acquire STT instance from pool
            let stt = match acquire_stt().await {
                Ok(stt) => stt,
                Err(e) => {
                    send_event(&tx, PttEvent::Error { message: e });
                    return;
                }
            };

            stt.reset();

            if let Err(e) = stt.process(&pcm_f32) {
                tracing::error!("IndicConformer STT process error: {}", e);
            }

            let final_result = stt.finalize();

            // Release STT instance back to pool
            release_stt(stt);

            final_result.text
        };
        metrics.stt_ms = stt_start.elapsed().as_millis() as u64;

        if stt_text.is_empty() {
            let session_id = request.session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            send_event(&tx, PttEvent::UserText {
                text: String::new(),
                corrected: None,
                session_id: session_id.clone(),
            });
            send_event(&tx, PttEvent::AssistantText {
                text: if use_english {
                    "I didn't hear anything. Please speak again.".to_string()
                } else {
                    "मुझे कुछ सुनाई नहीं दिया। कृपया फिर से बोलें।".to_string()
                },
            });
            send_event(&tx, PttEvent::Complete { metrics });
            return;
        }

        // 4. Grammar correction
        let grammar_start = std::time::Instant::now();
        let (corrected, corrections) = state.phonetic_corrector.correct(&stt_text);
        let corrected_text = if !corrections.is_empty() {
            tracing::info!(
                original = stt_text.as_str(),
                corrected = corrected.as_str(),
                "Phonetic correction applied"
            );
            Some(corrected)
        } else {
            None
        };
        metrics.grammar_ms = grammar_start.elapsed().as_millis() as u64;

        // Create/get session
        let session_id = request.session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // Send user text immediately!
        send_event(&tx, PttEvent::UserText {
            text: stt_text.clone(),
            corrected: corrected_text.clone(),
            session_id: session_id.clone(),
        });

        let text_for_llm = corrected_text.as_ref().unwrap_or(&stt_text);

        // 5. LLM processing
        let llm_start = std::time::Instant::now();
        let llm_response = match process_with_agent(
            &state,
            text_for_llm,
            &request.language,
            Some(&session_id),
        ).await {
            Ok((response, _sid)) => response,
            Err(e) => {
                tracing::error!("Agent processing failed: {}", e);
                format_fallback_response(text_for_llm, &request.language)
            }
        };
        metrics.llm_ms = llm_start.elapsed().as_millis() as u64;

        // Send assistant text
        send_event(&tx, PttEvent::AssistantText {
            text: llm_response.clone(),
        });

        // 6. TTS
        let tts_start = std::time::Instant::now();
        if let Ok((audio_b64, format)) = synthesize_with_tts(&llm_response, &request.language).await {
            send_event(&tx, PttEvent::AudioReady {
                audio: audio_b64,
                format,
            });
        }
        metrics.tts_ms = tts_start.elapsed().as_millis() as u64;

        metrics.total_ms = start.elapsed().as_millis() as u64;

        // Send completion
        send_event(&tx, PttEvent::Complete { metrics });
    });

    Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::default())
}
