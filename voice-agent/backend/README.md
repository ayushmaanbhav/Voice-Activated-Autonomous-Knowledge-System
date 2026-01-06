# Voice Agent Backend

> Pure Rust Voice/Text/Chat Agent Backend

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange?logo=rust)](https://www.rust-lang.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime%202.x-blue)](https://onnxruntime.ai/)
[![Tokio](https://img.shields.io/badge/Tokio-Async-green)](https://tokio.rs/)

---

## Overview

The backend is organized as a Rust workspace with 11 specialized crates that work together to provide a production-grade voice agent system.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CRATE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────────┐                                │
│                              │   server    │  HTTP/WS Entry Point           │
│                              └──────┬──────┘                                │
│                                     │                                       │
│                    ┌────────────────┼────────────────┐                     │
│                    ▼                ▼                ▼                     │
│              ┌─────────┐      ┌─────────┐      ┌─────────┐                 │
│              │  agent  │      │ pipeline│      │transport│                 │
│              └────┬────┘      └────┬────┘      └────┬────┘                 │
│                   │                │                │                       │
│         ┌────────┬┴────────┐      │                │                       │
│         ▼        ▼         ▼      ▼                ▼                       │
│    ┌────────┐ ┌────────┐ ┌────────────┐     ┌───────────┐                  │
│    │  rag   │ │  llm   │ │   tools    │     │persistence│                  │
│    └────┬───┘ └────┬───┘ └─────┬──────┘     └─────┬─────┘                  │
│         │          │           │                  │                        │
│         └────────┬─┴───────────┘                  │                        │
│                  ▼                                │                        │
│         ┌────────────────┐                        │                        │
│         │text_processing │                        │                        │
│         └───────┬────────┘                        │                        │
│                 │                                 │                        │
│                 └──────────────┬──────────────────┘                        │
│                                ▼                                           │
│                         ┌───────────┐                                      │
│                         │  config   │                                      │
│                         └─────┬─────┘                                      │
│                               ▼                                            │
│                         ┌───────────┐                                      │
│                         │   core    │  Foundation Layer                    │
│                         └───────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Crates Overview

| Crate | Description | Key Features |
|-------|-------------|--------------|
| **core** | Foundation traits & types | 13 traits, 22 languages, audio types |
| **config** | Configuration management | YAML/TOML, hot-reload, domain config |
| **pipeline** | Audio processing | VAD, STT, TTS, turn detection, barge-in |
| **agent** | Conversation orchestration | DST, memory, lead scoring, stages |
| **rag** | Retrieval system | Hybrid search, reranking, caching |
| **llm** | LLM integration | Multi-provider, speculative decoding |
| **text_processing** | NLP pipeline | Grammar, translation, PII, compliance |
| **tools** | MCP tool interface | JSON-RPC, gold loan tools |
| **transport** | Audio transport | WebRTC, WebSocket, Opus codec |
| **persistence** | Data storage | ScyllaDB, audit logs, sessions |
| **server** | HTTP/WS server | Axum, metrics, auth |

---

## Quick Start

### Prerequisites

```bash
# ONNX Runtime
export ORT_LIB_LOCATION=/path/to/onnxruntime
export LIBRARY_PATH=$ORT_LIB_LOCATION/lib

# Qdrant (for RAG)
docker run -d -p 6333:6333 qdrant/qdrant

# Ollama (for LLM)
ollama pull qwen2.5:7b
```

### Build

```bash
# Development build
cargo build

# Release build with optimizations
cargo build --release

# Run tests
cargo test --workspace

# Run with logging
RUST_LOG=info cargo run --release
```

### Configuration

```bash
# Default configuration
cat config/default.yaml

# Override with environment variables
VOICE_AGENT__SERVER__PORT=8081 cargo run

# Use production config
cargo run -- --config config/production.yaml
```

---

## Directory Structure

```
backend/
├── Cargo.toml              # Workspace definition
├── Cargo.lock              # Locked dependencies
├── rustfmt.toml            # Formatting config
│
├── crates/                 # Rust crates
│   ├── core/               # Foundation (see crates/core/README.md)
│   ├── config/             # Configuration (see crates/config/README.md)
│   ├── pipeline/           # Audio pipeline (see crates/pipeline/README.md)
│   ├── agent/              # Agent orchestration
│   ├── rag/                # Retrieval system
│   ├── llm/                # LLM providers
│   ├── text_processing/    # NLP modules
│   ├── tools/              # MCP tools
│   ├── transport/          # Audio transport
│   ├── persistence/        # Data storage
│   └── server/             # HTTP server
│
├── config/                 # Configuration files
│   ├── default.yaml        # Default settings
│   ├── domain.yaml         # Domain-specific config
│   └── production.yaml     # Production overrides
│
├── knowledge/              # RAG knowledge base
│   ├── manifest.yaml       # Knowledge index
│   ├── products.yaml       # Product info
│   ├── rates.yaml          # Interest rates
│   └── ...                 # Other knowledge files
│
├── models/                 # ML model files (git-lfs)
│   ├── vad/                # Voice activity detection
│   ├── stt/                # Speech-to-text
│   ├── tts/                # Text-to-speech
│   ├── embeddings/         # Embedding models
│   └── reranker/           # Reranking models
│
└── scripts/                # Build & utility scripts
    └── load_knowledge.py   # Knowledge base loader
```

---

## Feature Flags

The workspace uses feature flags to control optional dependencies:

```toml
# Pipeline features
voice-agent-pipeline = { path = "crates/pipeline", features = ["onnx", "candle"] }

# RAG features
voice-agent-rag = { path = "crates/rag", features = ["candle"] }

# Server features
voice-agent-server = { path = "crates/server", features = ["webrtc", "telemetry"] }
```

### Available Features

| Crate | Feature | Description |
|-------|---------|-------------|
| **pipeline** | `onnx` | ONNX Runtime inference |
| **pipeline** | `candle` | Pure Rust Candle inference |
| **pipeline** | `noise-suppression` | RNNoise audio cleanup |
| **rag** | `onnx` | ONNX embeddings |
| **rag** | `candle` | Candle embeddings |
| **server** | `webrtc` | WebRTC support (~200 deps) |
| **server** | `telemetry` | OpenTelemetry tracing |
| **text_processing** | `onnx` | ONNX translation |
| **text_processing** | `candle` | Candle translation |

---

## Dependencies

### Core Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| tokio | 1.x | Async runtime |
| axum | 0.7 | Web framework |
| ort | 2.0-rc | ONNX Runtime |
| candle-core | 0.8 | Pure Rust ML |
| qdrant-client | 1.x | Vector search |
| tantivy | 0.22 | BM25 search |
| scylla | 0.14 | Database driver |
| tracing | 0.1 | Structured logging |

### Model Dependencies

| Model | Format | Size | Purpose |
|-------|--------|------|---------|
| silero_vad | ONNX | 2MB | Voice detection |
| indicconformer | ONNX | 600MB | Indian STT |
| indicf5 | SafeTensors | 500MB | Indian TTS |
| e5-multilingual | ONNX | 278MB | Embeddings |
| bge-reranker-v2-m3 | ONNX | 500MB | Reranking |

---

## API Endpoints

### HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/ready` | Readiness check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/api/session` | Create session |
| POST | `/api/ptt/:session_id` | Push-to-talk |

### WebSocket Endpoints

| Path | Description |
|------|-------------|
| `/ws/:session_id` | Bidirectional audio stream |

### WebRTC Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/webrtc/:session_id/offer` | SDP offer |
| POST | `/api/webrtc/:session_id/answer` | SDP answer |
| POST | `/api/webrtc/:session_id/ice` | ICE candidate |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Log level |
| `VOICE_AGENT__SERVER__PORT` | `8080` | HTTP port |
| `VOICE_AGENT__AGENT__LLM__ENDPOINT` | `http://localhost:11434` | Ollama endpoint |
| `VOICE_AGENT__RAG__QDRANT_ENDPOINT` | `http://localhost:6333` | Qdrant endpoint |
| `ORT_LIB_LOCATION` | - | ONNX Runtime path |

---

## Development

### Code Style

```bash
# Format code
cargo fmt --all

# Lint code
cargo clippy --workspace --all-features

# Check types
cargo check --workspace --all-features
```

### Testing

```bash
# Run all tests
cargo test --workspace

# Run specific crate tests
cargo test -p voice-agent-pipeline

# Run with output
cargo test --workspace -- --nocapture
```

### Benchmarks

```bash
# Run benchmarks
cargo bench --workspace
```

---

## Deployment

### Docker Build

```dockerfile
FROM rust:1.75-slim AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/voice-agent-server /usr/local/bin/
COPY --from=builder /app/config /etc/voice-agent/config
COPY --from=builder /app/models /var/lib/voice-agent/models
CMD ["voice-agent-server"]
```

### Systemd Service

```ini
[Unit]
Description=Voice Agent Server
After=network.target

[Service]
Type=simple
User=voice-agent
ExecStart=/usr/local/bin/voice-agent-server
Restart=always
Environment=RUST_LOG=info

[Install]
WantedBy=multi-user.target
```

---

## Observability

### Logging

```bash
# Structured JSON logging (production)
VOICE_AGENT__OBSERVABILITY__LOG_JSON=true cargo run

# Debug logging
RUST_LOG=voice_agent=debug cargo run
```

### Metrics

Prometheus metrics available at `/metrics`:

- `conversations_started_total` - Total conversations
- `stt_latency_seconds` - STT processing time
- `llm_latency_seconds` - LLM generation time
- `active_sessions` - Current active sessions

### Tracing

OpenTelemetry traces (when enabled):

```bash
# Enable OTEL tracing
VOICE_AGENT__OBSERVABILITY__TRACING_ENABLED=true \
VOICE_AGENT__OBSERVABILITY__OTLP_ENDPOINT=http://localhost:4317 \
cargo run
```

---

## License

Proprietary. See [LICENSE](../../LICENSE) for details.
