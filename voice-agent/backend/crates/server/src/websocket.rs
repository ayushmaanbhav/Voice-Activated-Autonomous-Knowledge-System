//! WebSocket Handler
//!
//! Real-time audio streaming and conversation.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::Response,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc;

use voice_agent_core::{AudioFrame, Channels, Frame, SampleRate};
use voice_agent_pipeline::{create_noise_suppressor, PipelineConfig, PipelineEvent, VoicePipeline};

use crate::rate_limit::RateLimiter;
use crate::session::Session;
use crate::state::AppState;

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WsMessage {
    /// Audio data (base64 encoded)
    Audio {
        data: String,
    },
    /// Text input
    Text {
        content: String,
    },
    /// Transcript update
    Transcript {
        text: String,
        is_final: bool,
    },
    /// Agent response
    Response {
        text: String,
    },
    /// Agent audio response
    ResponseAudio {
        data: String,
    },
    /// Status update
    Status {
        state: String,
        stage: String,
    },
    /// Error
    Error {
        message: String,
    },
    /// Ping/Pong
    Ping,
    Pong,
    /// Session info
    SessionInfo {
        session_id: String,
    },
    /// End session
    EndSession,
}

/// WebSocket handler
pub struct WebSocketHandler;

impl WebSocketHandler {
    /// Handle WebSocket upgrade
    pub async fn handle(
        ws: WebSocketUpgrade,
        State(state): State<AppState>,
        Path(session_id): Path<String>,
    ) -> Result<Response, axum::http::StatusCode> {
        // Get or create session
        let session = state
            .sessions
            .get(&session_id)
            .ok_or(axum::http::StatusCode::NOT_FOUND)?;

        // Create rate limiter for this connection
        // P1 FIX: Use RwLock for hot-reload support
        let rate_limit_config = state.config.read().server.rate_limit.clone();
        let rate_limiter = RateLimiter::new(rate_limit_config);

        Ok(ws.on_upgrade(move |socket| Self::handle_socket(socket, session, state, rate_limiter)))
    }

    /// Handle WebSocket connection
    async fn handle_socket(
        socket: WebSocket,
        session: Arc<Session>,
        state: AppState,
        rate_limiter: RateLimiter,
    ) {
        // P2 FIX: Get text processing components from state
        let text_processing = state.text_processing.clone();
        let text_simplifier = state.text_simplifier.clone();
        let (sender, mut receiver) = socket.split();

        // Wrap sender in Arc<Mutex> for sharing across tasks
        let sender = Arc::new(tokio::sync::Mutex::new(sender));

        // Wrap rate limiter in Arc<Mutex> for thread-safe access
        let rate_limiter = Arc::new(tokio::sync::Mutex::new(rate_limiter));

        // Send session info
        {
            let info = WsMessage::SessionInfo {
                session_id: session.id.clone(),
            };
            let mut s = sender.lock().await;
            let _ = s
                .send(Message::Text(serde_json::to_string(&info).unwrap()))
                .await;

            // Send initial status
            let status = WsMessage::Status {
                state: "active".to_string(),
                stage: session.agent.stage().display_name().to_string(),
            };
            let _ = s
                .send(Message::Text(serde_json::to_string(&status).unwrap()))
                .await;
        }

        // Subscribe to agent events
        let mut agent_events = session.agent.subscribe();

        // Create channels for audio processing
        let (audio_tx, mut audio_rx) = mpsc::channel::<Vec<u8>>(100);

        // Create voice pipeline for audio processing
        // P0 FIX: Wire text processing (grammar, PII, compliance) to pipeline
        // P2 FIX: Wire noise suppression for cleaner audio input
        let noise_suppressor: Arc<dyn voice_agent_core::AudioProcessor> =
            Arc::from(create_noise_suppressor(16000)); // 16kHz input
        let pipeline = match VoicePipeline::simple(PipelineConfig::default()) {
            Ok(p) => {
                let p = p
                    .with_text_processor(text_processing.clone())
                    .with_noise_suppressor(noise_suppressor);
                Some(Arc::new(tokio::sync::Mutex::new(p)))
            },
            Err(e) => {
                tracing::warn!(
                    "Failed to create voice pipeline: {}, using text-only mode",
                    e
                );
                None
            },
        };

        // Spawn audio processor task - receives audio and feeds to pipeline
        let session_clone = session.clone();
        let pipeline_clone = pipeline.clone();

        let audio_task = tokio::spawn(async move {
            let mut frame_count: u64 = 0;

            while let Some(audio_data) = audio_rx.recv().await {
                session_clone.touch();

                // Convert raw PCM bytes to f32 samples (assuming 16-bit PCM)
                let samples: Vec<f32> = audio_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect();

                if samples.is_empty() {
                    continue;
                }

                // Create audio frame
                let frame =
                    AudioFrame::new(samples, SampleRate::Hz16000, Channels::Mono, frame_count);
                frame_count += 1;

                // Process through pipeline if available
                if let Some(ref pipeline) = pipeline_clone {
                    let pipeline_guard = pipeline.lock().await;

                    if let Err(e) = pipeline_guard.process_audio(frame).await {
                        tracing::debug!("Pipeline processing error: {}", e);
                    }
                }
            }
        });

        // Spawn pipeline event handler task
        let session_for_pipeline = session.clone();
        let sender_for_pipeline = sender.clone();
        let pipeline_for_tts = pipeline.clone(); // P0 FIX: Clone for TTS synthesis
                                                 // P2 FIX: Clone text processing for pipeline event handler
        let text_processing_for_pipeline = text_processing.clone();
        let text_simplifier_for_pipeline = text_simplifier.clone();

        #[allow(unused_mut)]
        let pipeline_event_task = if let Some(ref pipeline) = pipeline {
            let mut pipeline_events = pipeline.lock().await.subscribe();
            Some(tokio::spawn(async move {
                while let Ok(event) = pipeline_events.recv().await {
                    match event {
                        PipelineEvent::PartialTranscript(transcript) => {
                            // Send partial transcript to client
                            let msg = WsMessage::Transcript {
                                text: transcript.text,
                                is_final: false,
                            };
                            let json = serde_json::to_string(&msg).unwrap();
                            let mut s = sender_for_pipeline.lock().await;
                            let _ = s.send(Message::Text(json)).await;
                        },
                        PipelineEvent::FinalTranscript(transcript) => {
                            let text = transcript.text.clone();

                            // Send final transcript to client
                            let msg = WsMessage::Transcript {
                                text: text.clone(),
                                is_final: true,
                            };
                            let json = serde_json::to_string(&msg).unwrap();
                            let mut s = sender_for_pipeline.lock().await;
                            let _ = s.send(Message::Text(json)).await;
                            drop(s); // Release lock before async operations

                            // Process through agent
                            if !text.trim().is_empty() {
                                // P2 FIX: Process user input through text processing pipeline
                                // (grammar correction, PII detection)
                                let processed_input = match text_processing_for_pipeline
                                    .process(&text)
                                    .await
                                {
                                    Ok(result) => {
                                        if result.pii_detected {
                                            tracing::info!("PII detected in user input, redacted");
                                        }
                                        result.processed
                                    },
                                    Err(e) => {
                                        tracing::warn!(
                                            "Text processing failed: {}, using raw input",
                                            e
                                        );
                                        text.clone()
                                    },
                                };

                                // P0-2 FIX: Use streaming agent response with streaming TTS
                                // Note: process_stream blocks until complete (LLM stream lifetime limitation)
                                // Spawn the entire flow to not block the pipeline event handler
                                let session = session_for_pipeline.clone();
                                let sender = sender_for_pipeline.clone();
                                let text_simplifier = text_simplifier_for_pipeline.clone();
                                let pipeline = pipeline_for_tts.clone();

                                tokio::spawn(async move {
                                    let user_language = session.agent.user_language();

                                    match session.agent.process_stream(&processed_input).await {
                                        Ok(mut chunk_rx) => {
                                            // P0-2 FIX: Use speak_streaming() for lower latency TTS
                                            if let Some(ref pipeline) = pipeline {
                                                let p = pipeline.lock().await;

                                                // Create channel to forward to TTS
                                                let (tts_tx, tts_rx) = mpsc::channel::<String>(32);

                                                // Start streaming TTS first
                                                match p.speak_streaming(tts_rx, user_language).await
                                                {
                                                    Ok(mut audio_rx) => {
                                                        drop(p); // Release pipeline lock

                                                        // Spawn task to handle audio output frames
                                                        let sender_for_audio = sender.clone();
                                                        tokio::spawn(async move {
                                                            while let Some(frame) =
                                                                audio_rx.recv().await
                                                            {
                                                                if let Frame::AudioOutput(
                                                                    audio_frame,
                                                                ) = frame
                                                                {
                                                                    // Convert f32 samples to i16 PCM bytes
                                                                    let pcm_bytes: Vec<u8> =
                                                                        audio_frame
                                                                            .samples
                                                                            .iter()
                                                                            .flat_map(|&sample| {
                                                                                let clamped =
                                                                                    sample.clamp(
                                                                                        -1.0, 1.0,
                                                                                    );
                                                                                let i16_sample =
                                                                                    (clamped
                                                                                        * 32767.0)
                                                                                        as i16;
                                                                                i16_sample
                                                                                    .to_le_bytes()
                                                                                    .to_vec()
                                                                            })
                                                                            .collect();

                                                                    // Base64 encode and send
                                                                    let audio_data =
                                                                        BASE64.encode(&pcm_bytes);
                                                                    let msg =
                                                                        WsMessage::ResponseAudio {
                                                                            data: audio_data,
                                                                        };
                                                                    let json =
                                                                        serde_json::to_string(&msg)
                                                                            .unwrap();
                                                                    let mut s = sender_for_audio
                                                                        .lock()
                                                                        .await;
                                                                    if let Err(e) = s
                                                                        .send(Message::Text(json))
                                                                        .await
                                                                    {
                                                                        tracing::debug!("Failed to send streaming TTS audio: {}", e);
                                                                        break;
                                                                    }
                                                                }
                                                            }
                                                        });

                                                        // Forward chunks to client and TTS
                                                        while let Some(chunk) =
                                                            chunk_rx.recv().await
                                                        {
                                                            // Send to client
                                                            let resp = WsMessage::Response {
                                                                text: chunk.clone(),
                                                            };
                                                            let json = serde_json::to_string(&resp)
                                                                .unwrap();
                                                            let mut s = sender.lock().await;
                                                            let _ =
                                                                s.send(Message::Text(json)).await;
                                                            drop(s);

                                                            // Simplify and send to TTS
                                                            let simplified =
                                                                text_simplifier.simplify(&chunk);
                                                            let _ = tts_tx.send(simplified).await;
                                                        }

                                                        tracing::debug!(
                                                            "Streaming response complete"
                                                        );
                                                    },
                                                    Err(e) => {
                                                        drop(p);
                                                        tracing::warn!("speak_streaming failed: {}, using text-only", e);

                                                        // Fallback: just stream text
                                                        while let Some(chunk) =
                                                            chunk_rx.recv().await
                                                        {
                                                            let resp =
                                                                WsMessage::Response { text: chunk };
                                                            let json = serde_json::to_string(&resp)
                                                                .unwrap();
                                                            let mut s = sender.lock().await;
                                                            let _ =
                                                                s.send(Message::Text(json)).await;
                                                        }
                                                    },
                                                }
                                            } else {
                                                // No pipeline - just stream text responses
                                                while let Some(chunk) = chunk_rx.recv().await {
                                                    let resp = WsMessage::Response { text: chunk };
                                                    let json =
                                                        serde_json::to_string(&resp).unwrap();
                                                    let mut s = sender.lock().await;
                                                    let _ = s.send(Message::Text(json)).await;
                                                }
                                            }
                                        },
                                        Err(e) => {
                                            tracing::error!("Agent streaming error: {}", e);
                                        },
                                    }
                                });
                            }
                        },
                        PipelineEvent::VadStateChanged(state) => {
                            use voice_agent_pipeline::VadState;
                            let (ws_state, stage) = match state {
                                VadState::Speech => ("listening", "speech_active"),
                                VadState::Silence => ("idle", "silence"),
                                VadState::SpeechStart => ("listening", "speech_detected"),
                                VadState::SpeechEnd => ("processing", "speech_ended"),
                            };
                            let msg = WsMessage::Status {
                                state: ws_state.to_string(),
                                stage: stage.to_string(),
                            };
                            let json = serde_json::to_string(&msg).unwrap();
                            let mut s = sender_for_pipeline.lock().await;
                            let _ = s.send(Message::Text(json)).await;
                        },
                        PipelineEvent::Error(e) => {
                            tracing::error!("Pipeline error: {}", e);
                        },
                        PipelineEvent::TtsAudio {
                            samples,
                            text: _,
                            is_final: _,
                        } => {
                            // P0 FIX: Send TTS audio to client
                            // Convert f32 samples to i16 PCM bytes
                            let pcm_bytes: Vec<u8> = samples
                                .iter()
                                .flat_map(|&sample| {
                                    // Clamp to [-1.0, 1.0] and convert to i16
                                    let clamped = sample.clamp(-1.0, 1.0);
                                    let i16_sample = (clamped * 32767.0) as i16;
                                    i16_sample.to_le_bytes().to_vec()
                                })
                                .collect();

                            // Base64 encode and send
                            let audio_data = BASE64.encode(&pcm_bytes);
                            let msg = WsMessage::ResponseAudio { data: audio_data };
                            let json = serde_json::to_string(&msg).unwrap();
                            let mut s = sender_for_pipeline.lock().await;
                            if let Err(e) = s.send(Message::Text(json)).await {
                                tracing::debug!("Failed to send TTS audio: {}", e);
                            }
                        },
                        _ => {},
                    }
                }
            }))
        } else {
            None
        };

        // Spawn event forwarder task
        let sender_clone = sender.clone();

        let event_task = tokio::spawn(async move {
            while let Ok(event) = agent_events.recv().await {
                let msg = match event {
                    voice_agent_agent::AgentEvent::Response(text) => {
                        Some(WsMessage::Response { text })
                    },
                    voice_agent_agent::AgentEvent::Thinking => Some(WsMessage::Status {
                        state: "thinking".to_string(),
                        stage: "processing".to_string(),
                    }),
                    voice_agent_agent::AgentEvent::Error(e) => {
                        Some(WsMessage::Error { message: e })
                    },
                    _ => None,
                };

                if let Some(msg) = msg {
                    let json = serde_json::to_string(&msg).unwrap();
                    let mut s = sender_clone.lock().await;
                    let _ = s.send(Message::Text(json)).await;
                }
            }
        });

        // Clone rate limiter for main loop
        let rate_limiter_main = rate_limiter.clone();

        // Main message loop
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Text(text)) => {
                    // Check rate limit for messages
                    {
                        let mut limiter = rate_limiter_main.lock().await;
                        if let Err(e) = limiter.check_message() {
                            tracing::warn!("Rate limit exceeded: {}", e);
                            let err = WsMessage::Error {
                                message: format!("Rate limit exceeded: {}", e),
                            };
                            let mut s = sender.lock().await;
                            let _ = s
                                .send(Message::Text(serde_json::to_string(&err).unwrap()))
                                .await;
                            continue;
                        }
                    }

                    session.touch();

                    if let Ok(ws_msg) = serde_json::from_str::<WsMessage>(&text) {
                        match ws_msg {
                            WsMessage::Text { content } => {
                                // P2 FIX: Process user input through text processing pipeline
                                let processed_input = match text_processing.process(&content).await
                                {
                                    Ok(result) => {
                                        if result.pii_detected {
                                            tracing::info!(
                                                "PII detected in direct text input, redacted"
                                            );
                                        }
                                        result.processed
                                    },
                                    Err(e) => {
                                        tracing::warn!(
                                            "Text processing failed: {}, using raw input",
                                            e
                                        );
                                        content.clone()
                                    },
                                };

                                // Process text input
                                match session.agent.process(&processed_input).await {
                                    Ok(response) => {
                                        let resp = WsMessage::Response { text: response };
                                        let json = serde_json::to_string(&resp).unwrap();
                                        let mut s = sender.lock().await;
                                        let _ = s.send(Message::Text(json)).await;

                                        // Send status update
                                        let status = WsMessage::Status {
                                            state: "active".to_string(),
                                            stage: session.agent.stage().display_name().to_string(),
                                        };
                                        let _ = s
                                            .send(Message::Text(
                                                serde_json::to_string(&status).unwrap(),
                                            ))
                                            .await;
                                    },
                                    Err(e) => {
                                        let err = WsMessage::Error {
                                            message: e.to_string(),
                                        };
                                        let mut s = sender.lock().await;
                                        let _ = s
                                            .send(Message::Text(
                                                serde_json::to_string(&err).unwrap(),
                                            ))
                                            .await;
                                    },
                                }
                            },
                            WsMessage::Ping => {
                                let pong = WsMessage::Pong;
                                let mut s = sender.lock().await;
                                let _ = s
                                    .send(Message::Text(serde_json::to_string(&pong).unwrap()))
                                    .await;
                            },
                            WsMessage::Audio { data } => {
                                // Decode base64 audio data and send to processor
                                match BASE64.decode(&data) {
                                    Ok(audio_bytes) => {
                                        // Check rate limit for audio data
                                        let mut limiter = rate_limiter_main.lock().await;
                                        if let Err(e) = limiter.check_audio(audio_bytes.len()) {
                                            tracing::warn!(
                                                "Audio rate limit exceeded: {} bytes",
                                                audio_bytes.len()
                                            );
                                            let err = WsMessage::Error {
                                                message: format!("Rate limit exceeded: {}", e),
                                            };
                                            let mut s = sender.lock().await;
                                            let _ = s
                                                .send(Message::Text(
                                                    serde_json::to_string(&err).unwrap(),
                                                ))
                                                .await;
                                            continue;
                                        }
                                        drop(limiter); // Release lock before sending
                                        let _ = audio_tx.send(audio_bytes).await;
                                    },
                                    Err(e) => {
                                        tracing::warn!("Failed to decode audio data: {}", e);
                                    },
                                }
                            },
                            WsMessage::EndSession => {
                                session.close();
                                break;
                            },
                            _ => {},
                        }
                    }
                },
                Ok(Message::Binary(data)) => {
                    // Check rate limit for audio data
                    {
                        let mut limiter = rate_limiter_main.lock().await;
                        if let Err(e) = limiter.check_audio(data.len()) {
                            tracing::warn!("Audio rate limit exceeded: {} bytes", data.len());
                            let err = WsMessage::Error {
                                message: format!("Rate limit exceeded: {}", e),
                            };
                            let mut s = sender.lock().await;
                            let _ = s
                                .send(Message::Text(serde_json::to_string(&err).unwrap()))
                                .await;
                            continue;
                        }
                    }

                    // Raw binary audio data (PCM)
                    let _ = audio_tx.send(data).await;
                },
                Ok(Message::Ping(data)) => {
                    let mut s = sender.lock().await;
                    let _ = s.send(Message::Pong(data)).await;
                },
                Ok(Message::Close(_)) => break,
                Err(e) => {
                    tracing::error!("WebSocket error: {}", e);
                    break;
                },
                _ => {},
            }
        }

        // Cleanup
        audio_task.abort();
        event_task.abort();
        if let Some(task) = pipeline_event_task {
            task.abort();
        }

        tracing::info!("WebSocket closed for session: {}", session.id);
    }
}

/// Create new session endpoint
pub async fn create_session(
    State(state): State<AppState>,
) -> Result<axum::Json<serde_json::Value>, axum::http::StatusCode> {
    let config = voice_agent_agent::AgentConfig::default();

    // P0 FIX: Pass vector store AND tools to enable full integration in agent
    // This ensures the agent uses the persistence-wired tool registry from AppState
    // instead of creating its own default registry without persistence.
    match state.sessions.create_with_full_integration(
        config,
        state.vector_store.clone(),
        Some(state.tools.clone()),
    ) {
        Ok(session) => {
            // P2-3 FIX: Persist session metadata to configured store
            if let Err(e) = state.persist_session(&session).await {
                tracing::warn!(session_id = %session.id, error = %e, "Failed to persist session metadata");
                // Continue anyway - session is functional even if persistence fails
            } else {
                tracing::debug!(
                    session_id = %session.id,
                    distributed = state.is_distributed_sessions(),
                    rag_enabled = state.vector_store.is_some(),
                    tools_wired = true,
                    "Session persisted with full integration"
                );
            }

            Ok(axum::Json(serde_json::json!({
                "session_id": session.id,
                "websocket_url": format!("/ws/{}", session.id),
                "rag_enabled": state.vector_store.is_some(),
                "tools_wired": true,
            })))
        },
        Err(_) => Err(axum::http::StatusCode::SERVICE_UNAVAILABLE),
    }
}
