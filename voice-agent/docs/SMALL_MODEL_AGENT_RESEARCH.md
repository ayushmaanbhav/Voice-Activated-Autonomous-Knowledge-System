# Research Summary: Small Model Voice/Chat Agents

> Research compiled: January 2025
> Focus: Managing voice agents with small models (Qwen2.5:1.5B, Whisper-tiny)

## Table of Contents
1. [Small Language Models](#1-small-language-models-qwen25-phi-3-etc)
2. [Whisper for Hindi/Indic Languages](#2-whisper-for-hindiindic-languages)
3. [Memory & Context Management](#3-memory--context-management)
4. [Context Compression for Small Models](#4-context-compression-for-small-models)
5. [Vector DB & Embeddings](#5-vector-db--embeddings)
6. [Hallucination Reduction & Accuracy](#6-hallucination-reduction--accuracy)
7. [Dialogue State Tracking](#7-dialogue-state-tracking)
8. [MCP Tool Use](#8-mcp-tool-use)
9. [Voice Agent Turn Detection](#9-voice-agent-turn-detection)
10. [Sales Conversion & Customer Support](#10-sales-conversion--customer-support)
11. [Context Engineering](#11-context-engineering-anthropic)
12. [Edge Deployment & Latency](#12-edge-deployment--latency)

---

## 1. Small Language Models (Qwen2.5, Phi-3, etc.)

### Qwen2.5 Technical Report
- **Models**: 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B parameters
- **Training data**: 18 trillion tokens (up from 7T in Qwen2)
- Small models (0.5B, 1.5B, 3B) maintain strong performance across benchmarks
- **Source**: [Qwen2.5 Technical Report - arXiv](https://arxiv.org/abs/2412.15115)

### Qwen2.5-1M (Long Context Extension)
- Context lengths expanded progressively: 65,536 → 131,072 → 262,144 → 1M tokens
- Training data: 75% sequences at max length, 25% shorter sequences
- **Source**: [Qwen2.5-1M Technical Report](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/Qwen2_5_1M_Technical_Report.pdf)

### Qwen3 (Successor - May 2025)
- Native MCP support and robust function-calling
- BFCL (Berkeley Function Calling Leaderboard) score: **68.2**
- Qwen3-4B can rival Qwen2.5-72B-Instruct performance
- Hybrid thinking/non-thinking modes in unified framework
- **Source**: [Qwen3 Technical Report - arXiv](https://arxiv.org/abs/2505.09388)

### Small LLM Performance Trends
- SLMs improved by **13.5%** from 2022-2024 vs 7.5% for LLMs
- "A small model can do 80-90% of what the mega model can do at 1/100th the cost"
- Phi-3, Qwen 2.5 outperform open-source Llama 3.1 7B
- **Source**: [Kommunicate - Small Language Models](https://www.kommunicate.io/blog/faster-small-language-models/)

### Quantization for Small Models
- INT4 quantization preserves 99% of FP16 accuracy for 7B/8B models
- Degradation observed on ultra-long contexts (128K tokens)
- **Source**: [arXiv - Whisper Quantization Analysis](https://arxiv.org/html/2503.09905v1)

---

## 2. Whisper for Hindi/Indic Languages

### Fine-Tuning Strategies for Low-Resource ASR
- **57% improvement** in Whisper-Tiny with fine-tuning on Hindi (OpenSLR-69)
- All five fine-tuning strategies significantly enhance low-resource ASR
- Whisper versions: Tiny, Small, Base, Medium, Large
- **Source**: [EURASIP Journal - Whisper Fine-tuning](https://link.springer.com/article/10.1186/s13636-024-00349-3)

### Hindi-Specific Models
- **IndicWhisper**, **IndicConformer** from AI4Bharat support 22 languages
- 300,000 hours raw speech, 6,000 hours transcribed data
- Use **Indic Normalization** (retains diacritics) instead of default Whisper normalization
- **Sources**:
  - [AI4Bharat ASR](https://ai4bharat.iitm.ac.in/areas/asr)
  - [Collabora - Whisper for Hindi](https://www.collabora.com/news-and-blog/news-and-events/breaking-language-barriers-fine-tuning-whisper-for-hindi.html)

### Prompt-Tuning Research (Dec 2024)
- Language family information in prompts improves accuracy
- New tokenizer reduces inference time without losing accuracy
- **Source**: [arXiv - Enhancing Whisper for Indian Languages](https://arxiv.org/html/2412.19785v1)

### Comparative Performance
- W2V2-BERT and XLSR-53 outperform other models for Indo-Aryan languages
- Higher error rates for Dravidian languages (complex phonology, agglutinative morphology)
- **Source**: [EURASIP - Comparative ASR Analysis](https://link.springer.com/article/10.1186/s13636-025-00395-5)

### Available Models
- `vasista22/whisper-hindi-small` on HuggingFace
- KathBath dataset for Hindi ASR training
- **Source**: [HuggingFace - whisper-hindi-small](https://huggingface.co/vasista22/whisper-hindi-small)

---

## 3. Memory & Context Management

### MemGPT: LLMs as Operating Systems (2023)
**Key Paper**: Virtual context management inspired by OS memory hierarchies

**Architecture**:
- **Main context**: System instructions + working context + FIFO queue
- **External context**: Archival storage + recall storage databases
- Uses function calls to move data between memory tiers

**Memory Blocks**:
- "Human" block: user info, preferences, facts, context
- "Persona" block: agent's self-concept, personality, behavioral guidelines

**Evaluation Domains**:
1. Document analysis (documents exceeding context window)
2. Multi-session chat (long-term interactions)

**Source**: [MemGPT Paper - arXiv](https://arxiv.org/abs/2310.08560)

### A-MEM: Agentic Memory (NeurIPS 2025)
**Key Innovation**: Dynamic memory organization using **Zettelkasten method**

**Features**:
- Creates interconnected knowledge networks through indexing/linking
- Memory attributes: contextual descriptions, keywords, tags
- New memories trigger updates to historical memory representations
- Superior performance across 6 foundation models

**Source**: [A-MEM Paper - arXiv](https://arxiv.org/abs/2502.12110)

### Hierarchical Memory Architecture (Agentic RAG)
```
Short-Term Memory  → Current context window
Session Memory     → Rolling conversation buffer
Long-Term Memory   → Vector database (Qdrant, Weaviate)
External Memory    → Knowledge graph or data lake
Reflective Memory  → Self-summaries of prior interactions
```

**Source**: [Weaviate - Context Engineering](https://weaviate.io/blog/context-engineering)

### Mem0 Memory Orchestration
- Manages memory lifecycle: extraction → storage → retrieval
- Unified APIs for episodic, semantic, procedural, associative memories
- Automatic filtering to prevent memory bloat
- **Source**: [AWS - Mem0 Integration](https://aws.amazon.com/blogs/database/build-persistent-memory-for-agentic-ai-applications-with-mem0-open-source-amazon-elasticache-for-valkey-and-amazon-neptune-analytics/)

### Context Rot Problem
- As context window tokens increase, recall accuracy decreases
- Effective limit often 70-80% of nominal limit
- **Source**: [IBM Research - Context Windows](https://research.ibm.com/blog/larger-context-window)

---

## 4. Context Compression for Small Models

### LLMLingua Series (Microsoft)
- **20x compression** with minimal performance loss
- Uses GPT2-small or LLaMA-7B to identify/remove unimportant tokens
- Compressed prompts difficult for humans but effective for LLMs
- **Source**: [Microsoft LLMLingua](https://github.com/microsoft/LLMLingua)

### LLMLingua-2
- BERT-level encoder trained via GPT-4 distillation
- **3-6x faster** than original LLMLingua
- Excels in task-agnostic compression
- Better handling of out-of-domain data

### LongLLMLingua for Long Contexts
- **21.4% performance boost** with 4x fewer tokens (GPT-3.5-Turbo)
- **94% cost reduction** on LooGLE benchmark
- Uses LLaMA-2-7B-Chat for compression
- **Source**: [ACL 2024 - LongLLMLingua](https://aclanthology.org/2024.acl-long.91/)

### Soft Prompt Compression (SPC-LLM)
- Combines soft prompt compression with natural language summarization
- Distills texts into summaries integrated via trainable soft prompts
- Extends effective context window
- **Source**: [ACM - SPC-LLM](https://dl.acm.org/doi/fullHtml/10.1145/3677779.3677794)

### Rate-Distortion Framework (NeurIPS 2024)
- Analyzes rate-distortion trade-off for prompt compression
- Shortening sequence length reduces time and memory costs
- **Source**: [NeurIPS 2024 - Fundamental Limits](https://deepcomm.github.io/jekyll/pixyll/2024/07/26/prompt-compression/)

### Related Methods
- **Gist Tokens** (NeurIPS 2023): Learning to compress prompts
- **In-context Autoencoder** (ICLR 2024): Context compression in LLMs
- **RECOMP**: Compression and selective augmentation for RAG

---

## 5. Vector DB & Embeddings

### Semantic Chunking Best Practices
- Break documents into sentences, group with surrounding sentences
- Compare semantic distance to identify topic shifts
- Recursive/semantic chunkers with **10-20% overlap**
- **30-50% higher retrieval precision** vs naive fixed sizing
- **Source**: [Pinecone - Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)

### Small Embedding Model Benchmarks

| Model | Top-5 Accuracy | Latency | Notes |
|-------|---------------|---------|-------|
| e5-small | 100% | 16ms | 7x faster than qwen3-0.6b |
| e5-base-instruct | 100% | <30ms | Production-ready |
| MiniLM | High | Fast | Good for text |
| BGE-M3 | High | Moderate | Hybrid-ready |

- e5-small: **44% higher accuracy** than qwen3-0.6b
- **Source**: [AIMultiple - Embedding Benchmarks](https://research.aimultiple.com/open-source-embedding-models/)

### Late Chunking (Jina AI - Oct 2023)
- First open-source 8K context embedding model
- Embeds full context before chunking
- Vectors for "the city" contain links to "Berlin" from context
- **Source**: [Jina AI - Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)

### Contextual Retrieval (Anthropic 2024)
- Claude generates contextualized description for each chunk
- Description appended to chunk before embedding
- Addresses context loss from document splitting
- **Source**: [Stack Overflow - Chunking in RAG](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)

### Hybrid Retrieval
- Combine dense (BGE) + sparse (TF-IDF/BM25) embeddings
- Sparse and dense capture different relevance features
- Optimize combination based on use case
- **Source**: [Databricks - Embedding Finetuning](https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning)

### Vector Store Optimization
- Use HNSW or IVF-PQ algorithms (FAISS/Qdrant/Pinecone)
- Monitor recall@k, latency, index memory
- Re-embed on model upgrades
- Reranking with Cohere ReRank or ColBERT
- **Source**: [Medium - RAG Optimization](https://medium.com/@adnanmasood/optimizing-chunking-embedding-and-vectorization-for-retrieval-augmented-generation-ea3b083b68f7)

### Domain-Specific Fine-tuning
- Smaller finetuned model can outperform larger alternatives
- Better distinction between relevant/irrelevant documents
- Enhanced generation quality with domain-tailored responses
- **Source**: [Dataworkz - Domain Embedding Evaluation](https://www.dataworkz.com/blog/evaluating-voyage-embedding-models/)

---

## 6. Hallucination Reduction & Accuracy

### RAG Grounding Strategies
- **96% hallucination reduction** by combining RAG + RLHF + guardrails (Stanford 2024)
- **Source**: [Voiceflow - Prevent Hallucinations](https://www.voiceflow.com/blog/prevent-llm-hallucinations)

### MEGA-RAG Framework (2024)
- Multi-source evidence: dense (FAISS) + keyword (BM25) + knowledge graphs
- Cross-encoder reranker for semantic relevance
- Discrepancy-aware refinement module
- **40%+ reduction** in hallucination rates
- **Source**: [MEGA-RAG - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12540348/)

### Small Retriever + Small LLM
- Well-trained small retriever encoder reduces required LLM size
- Less resource-intensive deployments
- Improved out-of-domain generalization
- **Source**: [arXiv - Reducing Hallucination in Structured Outputs](https://arxiv.org/abs/2404.08189)

### DataGemma (Google 2024)
- Grounds LLMs in 240 billion data points (UN, WHO, CDC)
- **RIG**: Fact-checks during generation
- **RAG**: Integrates context before generation
- **Source**: [Medium - RIG/RAG](https://sbagency.medium.com/reduce-llms-hallucinations-use-of-rig-rag-38ff9a5370ef)

### Retrieval-Based Systems for Zero Hallucination
- Offline-constructed QA knowledge base
- Match queries with pre-existing questions
- No generation process = zero hallucination
- **Source**: [arXiv - Context-Aware Question Generation](https://arxiv.org/html/2410.12444v2)

### Hallucination Sources in RAG (Review Paper)
**Retrieval Phase**:
- Data source problems
- Query formulation issues
- Retriever limitations
- Strategy problems

**Generation Phase**:
- Context noise
- Context conflict
- "Lost in the middle" effect

**Source**: [MDPI - Hallucination Mitigation Review](https://www.mdpi.com/2227-7390/13/5/856)

---

## 7. Dialogue State Tracking

### LLM-Driven DST (LDST)
- Achieves ChatGPT-level performance with smaller open-source models
- **Domain-slot instruction tuning** for zero-shot/few-shot settings
- Based on smaller foundation models
- **Source**: [arXiv - LLM-driven DST](https://arxiv.org/abs/2310.14970)

### LLM-Backed User-Agent Simulation (ACL 2024)
- GPT-4 simulates user-agent interaction for training data generation
- Generates thousands of dialogues with DST labels
- Two-stage fine-tuning on LLaMA 2
- Outperforms real-data-only baseline
- **Source**: [ACL 2024 - DST Enhancement](https://aclanthology.org/2024.acl-long.473/)

### Zero-Shot DST Evaluation (2024)
- Two-dimensional evaluation: accuracy + completeness
- Uses GPT-4 for evaluation
- **Source**: [Papers with Code - DST](https://paperswithcode.com/task/dialogue-state-tracking/latest)

### Accountable DST for AI Agents (2025)
- Enables reliable estimation of AI agent errors
- Guides LLM decoder in generating more accurate actions
- **Source**: [Papers with Code - Latest DST](https://paperswithcode.com/task/dialogue-state-tracking/latest)

### Multi-Turn Conversation Evaluation Survey
- Covers 479 papers (2017-2024)
- Key targets: task success, response quality, user experience
- Venues: ICLR, NeurIPS, AAAI, NAACL, EMNLP, ACL
- **Source**: [arXiv - Multi-Turn Evaluation Survey](https://arxiv.org/html/2503.22458v1)

### DARD: Multi-Domain Dialogue System
- Multi-agent system for multi-domain dialogs
- Handles diverse user intents, entity types, domain knowledge
- **Source**: [arXiv Survey](https://arxiv.org/html/2503.22458v1)

---

## 8. MCP Tool Use

### Model Context Protocol Overview
- Introduced by Anthropic (November 2024)
- "USB-C for AI" - universal standard for connecting AI to external data
- **Source**: [MCP Ecosystem Overview](https://rickxie.cn/blog/MCP/)

### MCP Architecture
**Core Primitives**:
- **Resources**: Read-only data blobs
- **Prompts**: Templated messages/workflows
- **Tools**: Functions the model can call

**Source**: [Preprints - MCP Survey](https://www.preprints.org/manuscript/202504.0245)

### MCP-Universe Benchmark (Salesforce 2025)
First comprehensive MCP evaluation benchmark:
- **6 domains**: Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, Web Searching
- **11 MCP servers** evaluated
- Tests realistic, hard tasks
- **Source**: [MCP-Universe - arXiv](https://arxiv.org/pdf/2508.14704)

### MCP-Radar (2025)
- Multi-dimensional benchmark for tool use capabilities
- **Source**: [MCP Benchmarks](https://arxiv.org/pdf/2508.07575)

### Berkeley Function Calling Leaderboard (BFCL)
- First comprehensive LLM function calling evaluation
- Representative of enterprise workflow use-cases
- Qwen3-32B leads at **68.2**
- **Source**: [Berkeley BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)

### Qwen-Agent Framework
- Function call template for Qwen2.5 series and QwQ-32B
- Tool-Integrated Reasoning for Qwen2.5-Math
- **Avoid ReAct-style stopwords** for reasoning models (may output in thought section)
- **Source**: [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)

### Using MCP with Local LLMs
- Any LLM supporting function calling can use MCP
- Works with small Llama 3.2 models
- **Source**: [Medium - MCP with Local LLM](https://medium.com/predict/using-the-model-context-protocol-mcp-with-a-local-llm-e398d6f318c3)

### Guided-Structured Templates (2025)
- Pre-execution structured reasoning
- Enhances interpretability and performance
- **Source**: [arXiv - Guided Templates](https://arxiv.org/html/2509.18076v1)

---

## 9. Voice Agent Turn Detection

### VAD vs Semantic Turn Detection

| Approach | Method | Limitation |
|----------|--------|------------|
| Basic VAD | Silence detection | Misses "I understand, but..." |
| Semantic VAD | Word probability scoring | More compute |
| Hybrid | Audio + semantic analysis | Best accuracy |

**Source**: [Retell AI - VAD vs Turn-Taking](https://www.retellai.com/blog/vad-vs-turn-taking-end-point-in-conversational-ai)

### OpenAI Semantic VAD
- Classifier scores probability user is done speaking
- Low probability → wait for timeout
- High probability → respond immediately
- "Ummm..." → longer timeout; definitive statement → no wait
- **Source**: [OpenAI VAD Guide](https://platform.openai.com/docs/guides/realtime-vad)

### LiveKit End-of-Utterance (EOU) Model
- **135M parameter** transformer (SmolLM v2 based)
- Runs on CPU in real-time
- 4-turn sliding context window
- **Source**: [LiveKit - EOU Detection](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection/)

### TEN VAD (Agora)
- Designed for full-duplex voice communication
- Detects natural turn-taking cues
- Enables contextually aware interruptions
- Faster than Silero VAD (reduces latency by hundreds of ms)
- **Source**: [Agora - TEN VAD](https://www.agora.io/en/blog/making-voice-ai-agents-more-human-with-ten-vad-and-turn-detection/)

### Krisp Turn-Taking Model
- Audio-only, 6M weights
- Designed for Voice AI Agents
- **Source**: [Krisp - Turn Taking](https://krisp.ai/blog/turn-taking-for-voice-ai/)

### Full-Duplex Dialogue Control Tokens
- `<|S-S|>`, `<|C-S|>`, `<|C-L|>`, `<|S-L|>`
- Distinguish complete queries vs spurious barge-ins
- Robust turn switching management
- **Source**: [OpenReview - Full-Duplex](https://openreview.net/pdf?id=QbLbXz8Idp)

### Latency Targets
- **<300ms**: Feels natural
- **300ms-1.5s**: Acceptable
- **>1.5s**: Degrades experience rapidly
- **Source**: [Twilio - Voice Agent Latency](https://www.twilio.com/en-us/blog/developers/best-practices/guide-core-latency-ai-voice-agents)

### VAD Evaluation Metrics
- **FEC** (Front End Clipping): Speech cut off at beginning
- **MSC** (Mid Speech Clipping): Interrupted during conversation
- **NDS** (Noise Detected as Speech): Noise vs speech distinction
- **Source**: [Picovoice - VAD Guide](https://picovoice.ai/blog/complete-guide-voice-activity-detection-vad/)

---

## 10. Sales Conversion & Customer Support

### Key Performance Metrics
- **35-40% reduction** in agent time with automated routing
- Response time: 2-3 days → **<4 minutes**
- Cold email reply rates down ~50% in past 2 years
- By 2026: **$80B reduction** in agent labor costs (Gartner)
- **Source**: [Salesforce - Conversational AI](https://www.salesforce.com/artificial-intelligence/what-is-conversational-ai/)

### AI Sales Agent Capabilities
- Handle sales process with minimal human intervention
- NLP + customer data + ML for human-like conversations
- Analyze information and take independent action
- Eliminate non-revenue tasks for sales reps
- **Source**: [Nextiva - Conversational AI for Sales](https://www.nextiva.com/blog/conversational-ai-for-sales.html)

### Lead Qualification Features
- Personalized AI keeps prospects engaged
- Reduces drop-off, accelerates sales cycles
- Instant lead qualification
- Live rep handoff when needed
- **Source**: [Salesloft Drift](https://www.salesloft.com/platform/drift)

### Domain-Specific Banking SLMs
- Fine-tune Phi-3, Gemma, Mistral 7B, or Llama on domain data
- Or build SLM from scratch with domain-specific training
- Use RAG for up-to-date information (avoids fine-tuning costs)
- **Source**: [Infosys - SLMs in Financial Services](https://www.infosys.com/iki/perspectives/small-language-models-financial-services.html)

### Bank of America "Erica" Example
- Assists with routine banking tasks
- Financial advice and proactive insights
- Real-time assistance and personalized recommendations
- Increased customer engagement
- **Source**: [Medium - Custom LLMs in Fintech](https://mustafa-najoom.medium.com/how-custom-llms-are-saving-the-day-in-fintech-813e1f5b1301)

### FinBEN Benchmark (Summer 2024)
- 36 datasets, 24 tasks
- Categories: information extraction, risk management, decision making, text generation
- **Source**: [arXiv - Domain-specific LLMs](https://arxiv.org/abs/2401.02981)

### Dialogue Control Architecture
- Dialogue control modules for natural conversation progression
- Task prediction mechanisms for anticipating user intentions
- NLU for intent comprehension and error accommodation
- NLG for coherent, persuasive responses
- **Source**: [DigitalOcean - Conversational AI Platforms](https://www.digitalocean.com/resources/articles/conversational-ai-platforms)

---

## 11. Context Engineering (Anthropic)

### Definition
Context = set of tokens included when sampling from LLM
Goal = optimize token utility against LLM constraints

**Source**: [Anthropic - Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

### Layered Cognitive Model
```
Meta-Context        → Agent identity, tone, persona, confidence thresholds
Operational Context → Task, user intent, tools available, constraints
Domain Context      → Industry-specific knowledge, business rules
Historical Context  → Condensed interaction memory
Environmental       → System state, live data feeds, time awareness
```

**Source**: [Medium - Context Engineering](https://medium.com/@kuldeep.paul08/context-engineering-optimizing-llm-memory-for-production-ai-agents-6a7c9165a431)

### Context Rot
- Recall accuracy decreases as context tokens increase
- Effective limit: 70-80% of nominal context window
- All models exhibit this characteristic
- Context must be treated as finite resource with diminishing returns
- **Source**: [IBM Research - Context Windows](https://research.ibm.com/blog/larger-context-window)

### Google ADK Context Compaction
- LLM summarizes older events over sliding window
- Writes summary as new "compaction" event
- Prunes/de-prioritizes summarized raw events
- Every sub-agent sees **minimum context required**
- **Source**: [Google - Multi-Agent Framework](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)

### Design Principles
1. Separate storage from presentation (Sessions vs working context)
2. Every model call sees minimum required context
3. Durable state distinct from per-call views

### Context Extension Techniques
- **PI** (Position Interpolation): Input position indices
- **YaRN, LongRoPE**: RoPE frequency schedule
- **PSC**: Small modules at embedding level
- Brief fine-tuning (100-1000 steps) to adapt to wider context
- **Source**: [HuggingFace - LongRoPE](https://huggingface.co/papers/2402.13753)

### Recurrent Context Compression (RCC)
- **32x compression** with minimal loss (BLEU4 ~0.95)
- Enables 32K-long reconstruction within modest resources
- **Source**: [Emergent Mind - 32K Context](https://www.emergentmind.com/topics/32k-context-window)

---

## 12. Edge Deployment & Latency

### Production Voice Agent Benchmarks
| Component | Latency |
|-----------|---------|
| ASR | ~0.05s |
| TTS | ~0.28s |
| LLM | ~0.67s |
| **Total** | **0.94s** |

Under 1-second threshold considered acceptable.
- **Source**: [arXiv - Low-Latency Voice Agents](https://arxiv.org/html/2508.04721v1)

### Quantization Benefits
- INT4 quantization: **40% latency reduction**
- First-token latency **<1 second** with ARM KleidiAI
- ~25 tokens/sec (vs <1 token/sec for FP16)
- **Source**: [Edge LLM Deployment Review](https://www.rohan-paul.com/p/edge-deployment-of-llms-and-ml-models)

### AWS Local Zones
- Place FM inference closer to end users
- Dramatically reduces TTFT vs Regional deployments
- Tested with Meta Llama 3.2-3B
- **Source**: [AWS - Edge Inference](https://aws.amazon.com/blogs/machine-learning/reduce-conversational-ai-response-time-through-inference-at-the-edge-with-aws-local-zones/)

### Colocation Strategy
- GPUs + telephony in global PoPs
- Round-trip time <200ms between speech and inference
- Brings compute to voice traffic origin
- **Source**: [Telnyx - Voice AI Latency](https://telnyx.com/resources/voice-ai-agents-compared-latency)

### Hybrid Architecture
- Edge-driven for inference (privacy, latency)
- Cloud-supported for model improvements
- Speech engines: Vosk, Kaldi for local processing
- **Source**: [MDPI - Edge Speech-to-Text](https://www.mdpi.com/2078-2489/16/8/685)

### Speculative Decoding
- Small "draft" model generates future tokens
- Large model verifies; if correct, skip ahead
- Otherwise fall back to normal generation
- **Source**: [Deepgram - AI Agent Considerations](https://deepgram.com/learn/considerations-for-building-ai-agents)

### TTS Performance
- Murf Falcon: **55ms model latency**, 130ms time-to-first-audio
- Consistent across geographies via edge deployment
- **Source**: [Murf Falcon](https://murf.ai/falcon)

### Optimization Tips
1. Reuse connections (especially for LLM)
2. Prefer streaming APIs
3. Avoid DNS in critical path
4. Use WebRTC or Apache Kafka for real-time streaming
5. Focus on TCP handshakes and DNS optimization
- **Source**: [Cresta - Voice Agent Latency](https://cresta.com/blog/engineering-for-real-time-voice-agent-latency)

---

## Key Takeaways for Voice Agent Implementation

### Model Selection
1. **LLM**: Qwen2.5-1.5B or Qwen3-0.6B/1.7B with INT4 quantization
2. **ASR**: Fine-tune Whisper-small on Hindi with IndicNLP normalization
3. **Embeddings**: e5-small (100% top-5 accuracy, 16ms latency)

### Architecture
4. **Memory**: MemGPT-style tiers (working context + vector DB + archival)
5. **Compression**: LLMLingua-2 for context efficiency
6. **Turn Detection**: Semantic (not just silence-based VAD)

### Quality
7. **Accuracy**: RAG + guardrails for hallucination prevention
8. **DST**: Domain-slot instruction tuning for dialogue state tracking
9. **Tools**: MCP integration with function calling

### Performance
10. **Target Latency**: <1s total (ASR 50ms + LLM 670ms + TTS 280ms)
11. **Quantization**: INT4 for 40% latency reduction
12. **Edge**: Colocation for <200ms round-trip

---

## References by Topic

### Core Papers
- [MemGPT - arXiv](https://arxiv.org/abs/2310.08560)
- [A-MEM - arXiv](https://arxiv.org/abs/2502.12110)
- [Qwen2.5 - arXiv](https://arxiv.org/abs/2412.15115)
- [LongLLMLingua - ACL 2024](https://aclanthology.org/2024.acl-long.91/)

### Benchmarks
- [Berkeley BFCL](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
- [MCP-Universe](https://arxiv.org/pdf/2508.14704)
- [FinBEN](https://arxiv.org/abs/2401.02981)

### Implementation Guides
- [Anthropic Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Google ADK Framework](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)
- [Qwen Function Calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
