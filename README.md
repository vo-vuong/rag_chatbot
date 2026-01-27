# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system with FastAPI backend and Streamlit frontend enabling intelligent document Q&A through Docling-powered processing (PDF/DOCX), multimodal AI retrieval, and token-aware chunking. **Powered by LangGraph agentic workflow with ReAct pattern.**

## Key Features

- **LangGraph Agent**: Agentic RAG with ReAct workflow, tool-calling, and session persistence
- **OpenAI Integration**: GPT-4o, GPT-4o Mini
- **Reranking**: Cohere Rerank API (Multilingual v3.0) for high-precision retrieval
- **Multimodal Search**: Dual-collection retrieval (text + images) with GPT-4o Mini Vision AI captioning
- **Token-Aware Chunking**: Docling HybridChunker
- **Smart Processing**: PDF/DOCX via Docling, streaming CSV pipeline, OCR with EasyOCR
- **Vector Database**: Qdrant integration with dual collections (text + images)
- **Real-time Chat**: Context-aware conversations with RAG, query routing
- **Data Management**: Collection CRUD, adaptive pagination, metadata inspection
- **REST API**: FastAPI backend with CORS support

## Quick Start

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
2. **Upload**: Add PDF/CSV/DOCX files via Upload page with two-step workflow:
   - **Preview**: Process files and review extracted chunks/images (no save)
   - **Save**: Commit approved data to Qdrant vector database
3. **Vision Config**: Set caption failure mode (Graceful/Strict/Skip) for multimodal PDFs
4. **Chat**: Query your documents; results include relevant text and images with source attribution
5. **Manage**: Inspect collections and pagination via Data Management

## Document Processing

### Supported Formats

- **PDF**: Docling-powered processing with EasyOCR (en, vi), TableFormer tables, GPU acceleration
- **DOCX**: Docling-powered processing with same chunking and image captioning as PDF
- **CSV**: Streaming processing with column-based grouping and memory optimization

### Processing Features

- **Docling Document Strategy**: Modern PDF/DOCX processing with OCR, table extraction, and image handling
- **Token-Aware Chunking**: 512 max tokens optimized for RAG retrieval performance
- **Post-Processing Overlap**: 50 tokens overlap for context continuity
- **Vision API Integration**: GPT-4o Mini image captioning with MD5 caching

### Multimodal Vision Features

- **AI Captioning**: GPT-4o Mini Vision generates descriptive captions for extracted images
- **Caption Caching**: MD5-based caching for >80% cost savings on duplicate images

## Configuration

### Environment Variables (.env)

```bash
OPENAI_API_KEY=sk-...
COHERE_API_KEY=... (Optional - for Reranking)
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Architecture

```
┌───────────────────┐     HTTP      ┌───────────────────┐
│   Streamlit UI    │ ───────────►  │   FastAPI API     │
│ (localhost:8501)  │               │ (localhost:8000)  │
├───────────────────┤               ├───────────────────┤
│  StreamlitAPI     │               │  /api/v1/chat     │
│  Client (httpx)   │               │  /api/v1/rag      │
└───────────────────┘               │  /api/v1/upload   │
                                    │  /api/v1/health   │
                                    └─────────┬─────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
          ┌─────────────────┐       ┌─────────────┐          ┌─────────────┐
          │  AgentService   │       │RAGService   │          │UploadSvc    │
          │  (LangGraph     │       │(Retrieval + │          │(2-Step)     │
          │   ReAct Agent)  │       │ Reranking)  │          │             │
          └─────────────────┘       └─────────────┘          └─────────────┘
                    │                        │                        │
          ┌─────────┴─────────┐              │            ┌───────────┴───────────┐
          ▼                   ▼              ▼            ▼                       ▼
    ┌───────────┐       ┌───────────┐  ┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
    │LangGraph  │       │Retrieval  │  │SessionSvc   │  │DocumentProcessor │  │ Qdrant DB   │
    │StateGraph │       │Tool       │  │(State)      │  │(Docling+Vision)  │  │(Text+Image) │
    └───────────┘       └───────────┘  └─────────────┘  └──────────────────┘  └─────────────┘
                                        ▲
                                        │
                                 ┌─────────────┐
                                 │Cohere Rerank│
                                 │(API)        │
                                 └─────────────┘
```

### API Endpoints

| Endpoint                       | Method | Description                               |
| ------------------------------ | ------ | ----------------------------------------- |
| `/api/v1/chat/query`           | POST   | LangGraph agent chat with RAG             |
| `/api/v1/chat/query/stream`    | POST   | SSE streaming chat via LangGraph          |
| `/api/v1/chat/history/{id}`    | GET    | Get conversation history                  |
| `/api/v1/rag/search`           | POST   | Vector search without LLM                 |
| `/api/v1/upload/preview`       | POST   | Process file and return preview (no save) |
| `/api/v1/upload/save`          | POST   | Save processed chunks/images to Qdrant    |
| `/api/v1/health`               | GET    | Health check with Qdrant status           |

## Development

The project uses a modular architecture with Strategy, Factory, Singleton, and Dependency Injection patterns.

### Project Structure

- `api/`: FastAPI REST API layer (routers, services, models)
- `backend/`: Core processing engine (document processing, vision, embeddings, LLMs)
  - `backend/agent/`: LangGraph agent module (workflows, tools, nodes)
- `ui/`: Streamlit web interface and API client
- `config/`: Application constants, logging, and prompts
- `rag_evaluation/`: RAG evaluation framework (Hit@K, Recall@K, Precision@K, F1@K, MRR@K, Faithfulness, Response Relevancy, Context Precision, Context Recall, Answer Correctness)

### Design Patterns

- **Strategy Pattern**: Document processors, embeddings, LLMs
- **Singleton Pattern**: SessionManager, PromptManager
- **Factory Pattern**: EmbeddingFactory, LLMFactory, Graph factory
- **Dependency Injection**: FastAPI dependencies
- **Lazy Initialization**: Docling converter, Image captioner
- **ReAct Pattern**: LangGraph agent workflow (Agent → Tools → Agent → END)

## Technology Stack

**Core**: Python 3.11+, FastAPI 0.128+, Streamlit 1.29+, Qdrant 1.15.0
**AI/ML**: OpenAI (GPT-4o/embeddings/Vision), LangChain 0.1+, LangGraph 0.2+
**Processing**: Docling 2.0+, EasyOCR, pandas, Pillow, imagehash, tiktoken
**HTTP**: httpx (async-capable), uvicorn ASGI server
**Infrastructure**: Docker Compose, conda environment management

## RAG Evaluation

Evaluate retrieval and generation performance with extensible metrics framework.

```bash
# Run all metrics (retrieval + generation)
conda activate rag_chatbot && python -m rag_evaluation --metric all --k 5

# Run all retrieval metrics only
conda activate rag_chatbot && python -m rag_evaluation --metric all_retrieval --k 5

# Run all generation metrics only
conda activate rag_chatbot && python -m rag_evaluation --metric all_generation --k 5

# Run specific metrics
conda activate rag_chatbot && python -m rag_evaluation --metric hit recall precision f1 mrr --k 5

# Run Faithfulness (generation metric)
conda activate rag_chatbot && python -m rag_evaluation --metric faithfulness --k 5

# Run Response Relevancy (generation metric)
conda activate rag_chatbot && python -m rag_evaluation --metric response_relevancy --k 5

# Run Context Precision (generation metric)
conda activate rag_chatbot && python -m rag_evaluation --metric context_precision --k 5

# Run Context Recall (generation metric)
conda activate rag_chatbot && python -m rag_evaluation --metric context_recall --k 5

# Run Answer Correctness (generation metric)
conda activate rag_chatbot && python -m rag_evaluation --metric answer_correctness --k 5

# List available metrics
conda activate rag_chatbot && python -m rag_evaluation --list-metrics
```

See [rag_evaluation/RAG_EVAL_DOCUMENTATION.md](rag_evaluation/RAG_EVAL_DOCUMENTATION.md) for details on adding custom metrics.
