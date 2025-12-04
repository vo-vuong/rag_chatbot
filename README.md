# RAG Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that enables intelligent document Q&A through advanced processing, OCR capabilities, and AI-powered retrieval.

## ✨ Key Features

- 🤖 **Multi-LLM Support**: OpenAI models (GPT-4o, GPT-4o Mini, GPT-4 Turbo) with planned Gemini and local Ollama support
- 📄 **Advanced Document Processing**: PDF and CSV support with intelligent OCR and semantic chunking
- 💾 **Vector Database**: Qdrant integration for efficient similarity search and collection management
- 🔤 **OCR Integration**: Tesseract OCR with 125+ language support including English and Vietnamese
- 🔄 **Real-time Chat**: Context-aware conversations with RAG or LLM-only modes
- 🗂️ **Data Management**: Create, explore, and manage vector collections with advanced filtering
- ⚙️ **Multi-tier Processing**: Auto-detection with robust fallback mechanisms

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda (environment management)
- Docker & Docker Compose (for Qdrant)

### Installation

```bash
# 1. Clone and setup
git clone <repository-url>
cd rag_chatbot

# 2. Create conda environment
conda create -n rag-chatbot python=3.9 -y
conda activate rag-chatbot

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

1. **Setup**: Configure OpenAI API key in sidebar
2. **Upload**: Add PDF/CSV files via Upload page
3. **Process**: Choose processing strategy (Auto, Fast, High-Res, OCR)
4. **Chat**: Ask questions on main chat page
5. **Manage**: Use Data Management to explore collections

## 📋 Document Processing

### Supported Formats
- **PDF**: Advanced processing with OCR, semantic chunking, and metadata extraction
- **CSV**: Intelligent processing with column-based grouping, enhanced chunking, and tab-based UI

### Processing Strategies
- **Auto**: Intelligent strategy detection (recommended)
- **Fast**: Quick text extraction for text-based PDFs
- **High Resolution**: OCR-enabled processing for image-based PDFs
- **OCR Only**: Force OCR processing for scanned documents
- **Fallback**: Basic extraction with pdfplumber

### CSV Processing Features
- **Column-based Grouping**: Intelligent chunking by selected columns
- **Memory Optimization**: Streaming processing for large CSV files
- **Enhanced UI**: Tab-based interface (replaced nested expanders)
- **File Pointer Management**: Robust file handling with seek(0) operations
- **Performance Monitoring**: Real-time processing statistics and benchmarking

## ⚙️ Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_chatbot_collection
DEFAULT_LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.7
```

### LLM Configuration
- **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo, GPT-3.5 Turbo
- **Gemini**: Google Gemini models (planned)
- **Local**: Ollama-hosted models (planned)

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │
├─────────────────┤
│ Session Manager │  (Singleton state management)
├─────────────────┤
│  Document Proc  │  (Strategy pattern with OCR)
├─────────────────┤
│   Vector DB     │  (Qdrant with collections)
├─────────────────┤
│   LLM Services  │  (OpenAI, Gemini, Ollama)
└─────────────────┘
```

## 📚 Documentation

- **[Project Overview & PDR](docs/project-overview-pdr.md)**: Comprehensive requirements and specifications
- **[Codebase Summary](docs/codebase-summary.md)**: Detailed component documentation
- **[Code Standards](docs/code-standards.md)**: Development guidelines and best practices
- **[System Architecture](docs/system-architecture.md)**: Technical architecture and design patterns

## 🐳 Docker Management

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f qdrant

# Stop services
docker-compose down

# Restart Qdrant
docker-compose restart qdrant
```

## 📊 Usage Tips

### PDF Processing Best Practices
- **Text-based PDFs**: Use "Fast Processing" for speed
- **Image-based PDFs**: Use "High Resolution" for quality
- **Scanned documents**: Use "OCR Processing" for extraction
- **Large files**: Allow extra time for OCR processing

### Search Strategies
- **Vector Search**: Best for semantic similarity
- **Keywords Search**: Traditional text matching
- **Hybrid Search**: Combines both approaches

## 🔧 Development

### Project Structure
```
rag_chatbot/
├── app.py                    # Main Streamlit app
├── backend/                  # Core business logic
│   ├── session_manager.py     # State management
│   ├── document_processor.py  # Document processing
│   ├── chunking/            # Chunking strategies
│   ├── llms/                # LLM integrations
│   ├── embeddings/          # Vector embeddings
│   ├── ocr/                 # OCR services
│   └── vector_db/           # Qdrant integration
├── ui/                       # UI components
├── config/                    # Configuration
└── docs/                      # Documentation
```
