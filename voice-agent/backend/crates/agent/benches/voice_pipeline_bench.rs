//! Performance Benchmarks for Voice Agent Pipeline
//!
//! P5 FIX: Task 5.7 - Performance benchmarking suite
//!
//! Run with: cargo bench -p voice-agent-agent --bench voice_pipeline_bench

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::HashMap;

// =============================================================================
// Audio Resampling Benchmarks
// =============================================================================

fn bench_audio_resampling(c: &mut Criterion) {
    use voice_agent_core::audio::{AudioFrame, SampleRate, Channels};

    let mut group = c.benchmark_group("audio_resampling");

    // Test different frame sizes
    for frame_size in [160, 320, 480, 960, 1920].iter() {
        let samples: Vec<f32> = (0..*frame_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let frame = AudioFrame::new(samples, SampleRate::Hz16000, Channels::Mono, 0);

        group.throughput(Throughput::Elements(*frame_size as u64));
        group.bench_with_input(
            BenchmarkId::new("rubato_16k_to_22k", frame_size),
            &frame,
            |b, frame| {
                b.iter(|| {
                    frame.resample(SampleRate::Hz22050)
                })
            },
        );
    }

    // 16kHz to 8kHz (downsampling)
    let samples: Vec<f32> = (0..960).map(|i| (i as f32 * 0.01).sin()).collect();
    let frame = AudioFrame::new(samples, SampleRate::Hz16000, Channels::Mono, 0);
    group.bench_function("rubato_16k_to_8k_960", |b| {
        b.iter(|| frame.resample(SampleRate::Hz8000))
    });

    // 48kHz to 16kHz (common WebRTC scenario)
    let samples_48k: Vec<f32> = (0..2880).map(|i| (i as f32 * 0.01).sin()).collect();
    let frame_48k = AudioFrame::new(samples_48k, SampleRate::Hz48000, Channels::Mono, 0);
    group.bench_function("rubato_48k_to_16k_2880", |b| {
        b.iter(|| frame_48k.resample(SampleRate::Hz16000))
    });

    group.finish();
}

// =============================================================================
// VAD Benchmarks (using VoiceSession which has VAD built in)
// =============================================================================

fn bench_vad(c: &mut Criterion) {
    use voice_agent_agent::{VoiceSession, VoiceSessionConfig};

    let mut group = c.benchmark_group("vad");

    // Create session for VAD testing
    let config = VoiceSessionConfig::default();
    if let Ok(session) = VoiceSession::new("bench-vad", config) {
        // Silence
        let silence = vec![0.0f32; 512];
        group.bench_function("detect_silence_512", |b| {
            b.iter(|| session.detect_voice_activity(&silence))
        });

        // Speech-like signal
        let speech: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        group.bench_function("detect_speech_512", |b| {
            b.iter(|| session.detect_voice_activity(&speech))
        });

        // Larger frame
        let speech_large: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        group.bench_function("detect_speech_1024", |b| {
            b.iter(|| session.detect_voice_activity(&speech_large))
        });
    }

    group.finish();
}

// =============================================================================
// Intent Detection Benchmarks
// =============================================================================

fn bench_intent_detection(c: &mut Criterion) {
    use voice_agent_agent::IntentDetector;

    let mut group = c.benchmark_group("intent_detection");
    let detector = IntentDetector::new();

    // Short English query
    group.bench_function("short_english", |b| {
        b.iter(|| detector.detect("I want a gold loan"))
    });

    // Long English query
    group.bench_function("long_english", |b| {
        b.iter(|| detector.detect(
            "I am currently paying 18% interest on my gold loan from Muthoot Finance \
             and I heard Kotak offers better rates. Can you tell me how much I could save \
             if I transfer my loan of 5 lakh rupees?"
        ))
    });

    // Hindi query
    group.bench_function("hindi_query", |b| {
        b.iter(|| detector.detect("मुझे गोल्ड लोन चाहिए"))
    });

    // Mixed Hindi-English (code-switching)
    group.bench_function("hinglish_query", |b| {
        b.iter(|| detector.detect("Mujhe 5 lakh ka gold loan chahiye, Muthoot se transfer karna hai"))
    });

    group.finish();
}

// =============================================================================
// RAG Benchmarks
// =============================================================================

fn bench_rag(c: &mut Criterion) {
    use voice_agent_rag::{RetrieverConfig, RerankerConfig};

    let mut group = c.benchmark_group("rag");

    // Config creation benchmark (frequent operation)
    group.bench_function("retriever_config_create", |b| {
        b.iter(|| RetrieverConfig::default())
    });

    group.bench_function("reranker_config_create", |b| {
        b.iter(|| RerankerConfig::default())
    });

    // Config conversion from settings (P5 task)
    use voice_agent_config::RagConfig;

    let rag_config = RagConfig::default();
    group.bench_function("retriever_config_from_settings", |b| {
        b.iter(|| {
            let _: RetrieverConfig = (&rag_config).into();
        })
    });

    group.bench_function("reranker_config_from_settings", |b| {
        b.iter(|| {
            let _: RerankerConfig = (&rag_config).into();
        })
    });

    group.finish();
}

// =============================================================================
// Tool Execution Benchmarks
// =============================================================================

fn bench_tools(c: &mut Criterion) {
    use voice_agent_tools::{EligibilityCheckTool, SavingsCalculatorTool, Tool};
    use serde_json::json;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("tools");

    // Eligibility check tool
    let eligibility_tool = EligibilityCheckTool::new();
    let eligibility_input = json!({
        "gold_weight_grams": 50.0,
        "gold_purity": "22K"
    });

    group.bench_function("eligibility_check", |b| {
        b.to_async(&rt).iter(|| async {
            eligibility_tool.execute(eligibility_input.clone()).await
        })
    });

    // Savings calculator tool
    let savings_tool = SavingsCalculatorTool::new();
    let savings_input = json!({
        "current_loan_amount": 500000.0,
        "current_interest_rate": 18.0,
        "remaining_tenure_months": 12,
        "current_lender": "Muthoot"
    });

    group.bench_function("savings_calculator", |b| {
        b.to_async(&rt).iter(|| async {
            savings_tool.execute(savings_input.clone()).await
        })
    });

    // Tool timeout retrieval (new P5 feature)
    group.bench_function("get_tool_timeout", |b| {
        b.iter(|| eligibility_tool.timeout_secs())
    });

    group.finish();
}

// =============================================================================
// Memory Management Benchmarks
// =============================================================================

fn bench_memory(c: &mut Criterion) {
    use voice_agent_agent::{ConversationMemory, MemoryConfig, MemoryEntry};

    let mut group = c.benchmark_group("memory");

    // Add entry benchmark
    let config = MemoryConfig::default();
    let memory = ConversationMemory::new(config);

    group.bench_function("add_entry", |b| {
        let entry = MemoryEntry {
            role: "user".to_string(),
            content: "मुझे गोल्ड लोन की जानकारी चाहिए".to_string(),
            timestamp_ms: 100,
            stage: Some("Discovery".to_string()),
            intents: vec!["loan_inquiry".to_string()],
            entities: HashMap::new(),
        };
        b.iter(|| memory.add(entry.clone()))
    });

    // Get context benchmark (after adding entries)
    let config2 = MemoryConfig::default();
    let memory2 = ConversationMemory::new(config2);

    // Pre-populate
    for i in 0..20 {
        memory2.add(MemoryEntry {
            role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
            content: format!("Test message {}", i),
            timestamp_ms: i as u64 * 100,
            stage: None,
            intents: vec![],
            entities: HashMap::new(),
        });
    }
    memory2.add_fact("customer_name", "Rajesh Kumar", 0.95);
    memory2.add_fact("loan_amount", "500000", 0.9);

    group.bench_function("get_context_populated", |b| {
        b.iter(|| memory2.get_context())
    });

    group.bench_function("get_stats", |b| {
        b.iter(|| memory2.get_stats())
    });

    // P1 FIX: Cleanup benchmark
    group.bench_function("needs_cleanup_check", |b| {
        b.iter(|| memory2.needs_cleanup())
    });

    group.finish();
}

// =============================================================================
// Stage Management Benchmarks
// =============================================================================

fn bench_stage_manager(c: &mut Criterion) {
    use voice_agent_agent::{StageManager, ConversationStage, TransitionReason};

    let mut group = c.benchmark_group("stage_manager");

    group.bench_function("create_and_check", |b| {
        b.iter(|| {
            let manager = StageManager::new();
            let _ = manager.current();
            let _ = manager.stage_completed();
        })
    });

    group.bench_function("transition", |b| {
        let manager = StageManager::new();
        manager.record_turn(); // Complete greeting stage

        b.iter(|| {
            // Note: This may fail on subsequent iterations due to already being in Discovery,
            // but we're measuring the transition attempt performance
            let _ = manager.transition(ConversationStage::Discovery, TransitionReason::NaturalFlow);
        })
    });

    group.bench_function("suggest_next", |b| {
        let manager = StageManager::new();
        manager.record_turn();

        b.iter(|| manager.suggest_next())
    });

    // RAG timing strategy (P4 fix)
    use voice_agent_agent::RagTimingStrategy;

    group.bench_function("rag_timing_should_prefetch", |b| {
        let strategy = RagTimingStrategy::StageAware;
        b.iter(|| {
            strategy.should_prefetch(0.8, ConversationStage::Presentation)
        })
    });

    // Context budget calculation
    group.bench_function("context_budget_tokens", |b| {
        b.iter(|| {
            let stage = ConversationStage::Presentation;
            stage.context_budget_tokens()
        })
    });

    // RAG context fraction
    group.bench_function("rag_context_fraction", |b| {
        b.iter(|| {
            let stage = ConversationStage::ObjectionHandling;
            stage.rag_context_fraction()
        })
    });

    group.finish();
}

// =============================================================================
// Voice Session Benchmarks
// =============================================================================

fn bench_voice_session(c: &mut Criterion) {
    use voice_agent_agent::{VoiceSession, VoiceSessionConfig};

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("voice_session");

    // Session creation
    group.bench_function("create_session", |b| {
        let mut counter = 0;
        b.iter(|| {
            counter += 1;
            let config = VoiceSessionConfig::default();
            VoiceSession::new(&format!("bench-{}", counter), config)
        })
    });

    // Audio processing (when session is active)
    let config = VoiceSessionConfig::default();
    if let Ok(session) = VoiceSession::new("bench-audio", config) {
        rt.block_on(async { session.start().await.ok() });

        let audio_chunk: Vec<f32> = (0..320).map(|i| (i as f32 * 0.1).sin() * 0.3).collect();

        group.bench_function("process_audio_320", |b| {
            b.to_async(&rt).iter(|| async {
                session.process_audio(&audio_chunk).await
            })
        });
    }

    group.finish();
}

// =============================================================================
// Criterion Groups and Main
// =============================================================================

criterion_group!(
    benches,
    bench_audio_resampling,
    bench_vad,
    bench_intent_detection,
    bench_rag,
    bench_tools,
    bench_memory,
    bench_stage_manager,
    bench_voice_session,
);

criterion_main!(benches);
