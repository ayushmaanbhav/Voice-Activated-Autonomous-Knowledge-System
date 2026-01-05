<![CDATA[# Voice Agent Module

> Complete Voice/Text/Chat Agent Application

---

## Overview

The voice-agent module contains the complete application:

- **Backend** - Rust-based server with 11 specialized crates
- **Frontend** - Web-based voice interface (React/Vite)
- **Scripts** - Build and development utilities

```
voice-agent/
├── backend/                 # Rust server (see backend/README.md)
│   ├── crates/              # 11 Rust crates
│   ├── config/              # Configuration files
│   ├── knowledge/           # RAG knowledge base
│   └── models/              # ML model files
│
├── frontend/                # Web interface
│   ├── src/                 # React components
│   └── dist/                # Built assets
│
└── scripts/                 # Build & dev scripts
    ├── build-backend.sh     # Build backend
    ├── start-backend.sh     # Start server
    └── dev.sh               # Development mode
```

---

## Quick Start

### Prerequisites

```bash
# Rust 1.75+
rustup default stable

# Node.js 18+ (for frontend)
node --version

# ONNX Runtime
export ORT_LIB_LOCATION=/path/to/onnxruntime

# Qdrant (vector search)
docker run -d -p 6333:6333 qdrant/qdrant

# Ollama (local LLM)
ollama pull qwen2.5:7b
```

### Build & Run

```bash
# Backend
cd backend
./scripts/build-backend.sh
RUST_LOG=info cargo run --release

# Frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Access

- **API Server**: http://localhost:8080
- **Frontend**: http://localhost:5173
- **Metrics**: http://localhost:9090/metrics

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VOICE AGENT APPLICATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           FRONTEND                                   │   │
│  │                                                                      │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │   │
│  │   │   Voice     │  │    Chat     │  │   Session   │                │   │
│  │   │   Capture   │  │   Widget    │  │   Manager   │                │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘                │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                          WebSocket / WebRTC                                 │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           BACKEND                                    │   │
│  │                                                                      │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  Pipeline: VAD → STT → Turn Detection → TTS                 │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                    │                                │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  Agent: DST → Memory → RAG → LLM → Tools → Response         │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                       │
│                    ▼               ▼               ▼                       │
│              ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│              │ Qdrant  │    │ScyllaDB │    │ Ollama  │                    │
│              │ Vectors │    │ Storage │    │  LLM    │                    │
│              └─────────┘    └─────────┘    └─────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Backend

See [backend/README.md](backend/README.md) for details.

### Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `POST /api/session` | Create new session |
| `POST /api/ptt/:id` | Push-to-talk audio |
| `WS /ws/:id` | WebSocket audio stream |

### Configuration

```bash
# Environment variables
export RUST_LOG=info
export VOICE_AGENT__SERVER__PORT=8080
export VOICE_AGENT__AGENT__LLM__ENDPOINT=http://localhost:11434
export VOICE_AGENT__RAG__QDRANT_ENDPOINT=http://localhost:6333

# Or use config file
cargo run -- --config config/production.yaml
```

---

## Frontend

React-based web interface with:

- **Voice Capture** - WebRTC audio streaming
- **Chat Widget** - Text-based fallback
- **Session Manager** - Connection handling
- **Audio Playback** - TTS output

### Development

```bash
cd frontend
npm install
npm run dev    # Development server
npm run build  # Production build
```

### Environment

```bash
# .env
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080/ws
```

---

## Scripts

### Build Backend

```bash
./scripts/build-backend.sh
```

Builds the backend with ONNX Runtime support.

### Start Backend

```bash
./scripts/start-backend.sh
```

Starts the backend server with appropriate environment.

### Development Mode

```bash
./scripts/dev.sh
```

Runs both backend and frontend in development mode with hot reload.

### Watch Mode

```bash
./scripts/watch-backend.sh
```

Watches for changes and rebuilds automatically.

---

## Knowledge Base

The knowledge base lives in `backend/knowledge/`:

```
knowledge/
├── manifest.yaml      # Index of all files
├── products.yaml      # Product info (EN + Hindi)
├── rates.yaml         # Interest rates
├── objections.yaml    # Objection handling
├── eligibility.yaml   # Eligibility criteria
├── faqs.yaml          # FAQs
└── ...
```

### Loading Knowledge

```bash
cd backend
python scripts/load_knowledge.py
```

This indexes all knowledge into Qdrant.

---

## Model Files

Models are stored in `backend/models/` (use git-lfs for large files):

| Directory | Model | Size |
|-----------|-------|------|
| `vad/` | Silero VAD | 2MB |
| `stt/` | IndicConformer | 600MB |
| `tts/` | IndicF5, Piper | 500MB |
| `embeddings/` | E5-Multilingual | 278MB |
| `reranker/` | BGE-Reranker | 500MB |

---

## Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  voice-agent:
    build: ./backend
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - RUST_LOG=info
      - VOICE_AGENT__RAG__QDRANT_ENDPOINT=http://qdrant:6333
      - VOICE_AGENT__AGENT__LLM__ENDPOINT=http://ollama:11434
    depends_on:
      - qdrant
      - ollama

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
```

---

## Monitoring

### Prometheus Metrics

```
# conversations_started_total
# stt_latency_seconds
# llm_latency_seconds
# active_sessions
# tool_calls_total
```

### Grafana Dashboard

Import the dashboard from `docs/grafana/voice-agent-dashboard.json`.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| ONNX Runtime not found | Set `ORT_LIB_LOCATION` and `LIBRARY_PATH` |
| Qdrant connection failed | Ensure Qdrant is running on port 6333 |
| Ollama not responding | Run `ollama serve` and `ollama pull qwen2.5:7b` |
| High latency | Check model quantization and CPU cores |

### Debug Logging

```bash
RUST_LOG=voice_agent=debug,tower_http=debug cargo run
```
]]>