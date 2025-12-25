# RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that enables intelligent document Q&A through advanced processing, OCR capabilities, and multimodal AI-powered retrieval.

## ✨ Key Features

- 🤖 **Multi-LLM Support**: OpenAI (GPT-4o/Mini), Google Gemini (implemented), and planned Ollama support
- 🖼️ **Multimodal Search**: Dual-collection retrieval (text + images) with GPT-4o Mini Vision captioning
- 📄 **Advanced Document Processing**: 5-tier PDF strategy and streaming CSV pipeline with semantic chunking
- 🧠 **Semantic Chunking**: Embedding-based chunking using LangChain for coherent text segmentation
- 💾 **Vector Database**: Qdrant integration for efficient similarity search and collection management
- 🔤 **OCR Integration**: Tesseract OCR with 125+ language support including English and Vietnamese
- 🔄 **Real-time Chat**: Context-aware conversations with RAG or LLM-only modes
- 🗂️ **Data Management**: Collection CRUD operations and paginated data exploration
- ⚙️ **UI Optimization**: Real-time progress tracking, cost metrics, and configurable failure modes

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
- **PDF**: Multi-tier processing (Auto, Fast, Hi-Res, OCR) with image extraction and AI captioning
- **CSV**: Streaming processing with column-based grouping and memory optimization

### Processing Strategies
- **Auto**: Intelligent strategy detection based on document content
- **Fast**: Quick text extraction for text-based PDFs
- **High Resolution**: OCR-enabled processing for image-based PDFs
- **OCR Only**: Force OCR for scanned documents
- **Fallback**: Basic extraction using pdfplumber

### Multimodal Vision Features
- **AI Captioning**: GPT-4o Mini Vision generates descriptive captions for extracted images
- **Caption Caching**: MD5-based caching for >80% cost savings on duplicate images
- **Cost Tracking**: Real-time display of Vision API costs during upload
- **Failure Modes**: Configurable handling (Graceful/Strict/Skip) for captioning errors

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

Full documentation is available in the `docs/` folder.
