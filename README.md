# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system with FastAPI backend and Streamlit frontend enabling intelligent document Q&A through advanced processing, OCR capabilities, and multimodal AI-powered retrieval.

## ✨ Key Features

- 🤖 **OpenAI Integration**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo
- 🖼️ **Multimodal Search**: Dual-collection retrieval (text + images) with GPT-4o Mini Vision AI captioning
- 📄 **Hybrid Chunking**: 5-stage pipeline combining layout-aware + semantic chunking (Phase 04 complete)
- 🧠 **Smart Processing**: 5-tier PDF strategies, streaming CSV pipeline, OCR with 125+ languages
- 💾 **Vector Database**: Qdrant integration with dual collections (text + images)
- 🔄 **Real-time Chat**: Context-aware conversations with RAG or LLM-only modes
- 🗂️ **Data Management**: Collection CRUD, adaptive pagination, metadata inspection
- 🌐 **REST API**: FastAPI backend with SSE streaming, health checks, CORS support

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (environment management)
- Docker & Docker Compose (for Qdrant)

### Installation

```bash
# 1. Clone and setup
git clone <repository-url>
cd rag_chatbot

# 2. Setup conda environment
conda create -n rag_chatbot python=3.11 -y
conda activate rag_chatbot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your OpenAI API key

# 5. Start Qdrant
docker-compose up -d

# 6. Run FastAPI backend
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Run Streamlit frontend (new terminal)
streamlit run app.py
```

- Backend API: `http://localhost:8000` (Swagger docs at `/docs`)
- Frontend UI: `http://localhost:8501`

### Basic Workflow

1. **Setup**: Configure API keys in the sidebar
2. **Upload**: Add PDF/CSV files via Upload page; select processing strategy
3. **Vision Config**: Set caption failure mode (Graceful/Strict/Skip) for multimodal PDFs
4. **Chat**: Query your documents; results include relevant text and images
5. **Manage**: Inspect collections and pagination via Data Management

## 📋 Document Processing

### Supported Formats

- **PDF**: Docling-powered processing with EasyOCR (en, vi), TableFormer tables, GPU acceleration
- **CSV**: Streaming processing with column-based grouping and memory optimization

### Processing Features

- **Docling PDF Strategy**: Modern PDF processing with OCR, table extraction, and image handling
- **Token-Aware Chunking**: 512 max tokens optimized for RAG retrieval performance
- **Post-Processing Overlap**: 50 tokens overlap for context continuity
- **Vision API Integration**: GPT-4o Mini image captioning with MD5 caching

### Multimodal Vision Features

- **AI Captioning**: GPT-4o Mini Vision generates descriptive captions for extracted images
- **Caption Caching**: MD5-based caching for >80% cost savings on duplicate images

## ⚙️ Configuration

### Environment Variables (.env)

```bash
OPENAI_API_KEY=sk-...
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## 🏗️ Architecture

```
┌───────────────────┐     HTTP      ┌───────────────────┐
│   Streamlit UI    │ ───────────►  │   FastAPI API     │
│ (localhost:8501)  │               │ (localhost:8000)  │
├───────────────────┤               ├───────────────────┤
│  StreamlitAPI     │               │  /api/v1/chat     │
│  Client (httpx)   │               │  /api/v1/rag      │
└───────────────────┘               │  /api/v1/health   │
                                    └─────────┬─────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                   ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │ChatService  │    │RAGService   │    │SessionSvc   │
                   │(Orchestrate)│    │(Retrieval)  │    │(State)      │
                   └─────────────┘    └─────────────┘    └─────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │ QueryRouter │         │ Qdrant DB   │
       │ (LLM-based) │         │ (Vector)    │
       └─────────────┘         └─────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat/query` | POST | Synchronous chat with RAG |
| `/api/v1/chat/query/stream` | POST | SSE streaming chat |
| `/api/v1/rag/search` | POST | Vector search without LLM |
| `/api/v1/health` | GET | Health check with Qdrant status |

## 🔧 Development

The project uses a modular architecture with Strategy, Factory, Singleton, and Dependency Injection patterns.

- `api/`: FastAPI backend (routers, services, models)
- `backend/`: Core logic (document processing, vision, embeddings, LLMs)
- `ui/`: Streamlit components and API client
- `config/`: Application constants and defaults
- `tests/`: Pytest suite (API, services, UI client)

## 📚 Documentation

Comprehensive documentation available in `docs/`:

- **[Project Overview & PDR](docs/project-overview-pdr.md)**: Features, requirements, architecture
- **[Codebase Summary](docs/codebase-summary.md)**: Component details, data flows
- **[Code Standards](docs/code-standards.md)**: Development guidelines, patterns
- **[System Architecture](docs/system-architecture.md)**: Layer diagrams, integration points
- **[Project Roadmap](docs/project-roadmap.md)**: Version history, planned features

## 🛠️ Technology Stack

**Core**: Python 3.11+, FastAPI 0.128+, Streamlit 1.29+, Qdrant 1.15.0
**AI/ML**: OpenAI (GPT-4o/embeddings/Vision), LangChain 0.1+
**Processing**: Docling 2.0+, EasyOCR, pandas, Pillow, imagehash, tiktoken
**HTTP**: httpx (async-capable), uvicorn ASGI server
**Infrastructure**: Docker Compose, conda environment management
