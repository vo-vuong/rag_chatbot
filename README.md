# RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system built with Streamlit enabling intelligent document Q&A through advanced processing, OCR capabilities, and multimodal AI-powered retrieval.

## ✨ Key Features

- 🤖 **Multi-LLM Support**: OpenAI (GPT-4o/Mini), Google Gemini Pro
- 🖼️ **Multimodal Search**: Dual-collection retrieval (text + images) with GPT-4o Mini Vision AI captioning
- 📄 **Hybrid Chunking**: 5-stage pipeline combining layout-aware + semantic chunking (Phase 04 complete)
- 🧠 **Smart Processing**: 5-tier PDF strategies, streaming CSV pipeline, OCR with 125+ languages
- 💾 **Vector Database**: Qdrant integration with dual collections (text + images)
- 🔄 **Real-time Chat**: Context-aware conversations with RAG or LLM-only modes
- 🗂️ **Data Management**: Collection CRUD, adaptive pagination, metadata inspection

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (environment management)
- Docker & Docker Compose (for Qdrant)
- Tesseract OCR (system-level installation)

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

# 6. Run application
streamlit run app.py
```

Visit `http://localhost:8501` to access the application.

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
GEMINI_API_KEY=...      # Optional
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## 🏗️ Architecture

```
┌───────────────────┐
│   Streamlit UI    │ (Chat, Upload, Data Management)
├───────────────────┤
│  Session Manager  │ (Singleton state & Multimodal settings)
├───────────────────┤
│ Document Processor│ (Orchestrator with PDF/CSV strategies)
├───────────────────┤
│   Vision Service  │ (GPT-4o Mini Vision & Caption Cache)
├───────────────────┤
│  Vector Database  │ (Qdrant: Text & Image collections)
└───────────────────┘
```

## 🔧 Development

The project uses a modular architecture with Strategy, Factory, and Singleton patterns.

- `backend/`: Core logic (document processing, vision, embeddings, LLMs)
- `ui/`: Streamlit components and page routing
- `config/`: Application constants and defaults
- `tests/`: Pytest suite (focused on vision module)

## 📚 Documentation

Comprehensive documentation available in `docs/`:

- **[Project Overview & PDR](docs/project-overview-pdr.md)**: Features, requirements, architecture
- **[Codebase Summary](docs/codebase-summary.md)**: Component details, data flows
- **[Code Standards](docs/code-standards.md)**: Development guidelines, patterns
- **[System Architecture](docs/system-architecture.md)**: Layer diagrams, integration points
- **[Project Roadmap](docs/project-roadmap.md)**: Version history, planned features

## 🛠️ Technology Stack

**Core**: Python 3.11+, Streamlit 1.29+, Qdrant 1.12.5, LangChain 0.1+
**AI/ML**: OpenAI (GPT-4o/embeddings/Vision), Google Gemini, sentence-transformers
**Processing**: Docling 2.0+, EasyOCR, pandas, pytesseract, Pillow, imagehash, tiktoken
**Infrastructure**: Docker Compose, conda environment management
