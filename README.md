# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit that allows users to upload documents, process them using advanced chunking strategies with OCR capabilities, and ask questions answered using AI-powered retrieval from document chunks.

## Features

- 🤖 **LLM Support**: OpenAI models (with planned support for Gemini and local Ollama models)
- 📄 **Advanced Document Processing**: Support for CSV and PDF files with intelligent processing strategies
- 🔍 **Intelligent Chunking**: Advanced chunking strategies including semantic chunking for PDFs and title-based segmentation
- 💾 **Vector Database**: Qdrant integration for efficient document retrieval with comprehensive collection management
- 🔤 **OCR Integration**: Tesseract OCR with 125+ language support and cross-platform compatibility
- 🌐 **Multi-language**: Support for English and Vietnamese with intelligent language detection
- 🎯 **Vector Search**: Qdrant-based similarity search for document retrieval with multiple search strategies
- 📝 **Prompt Management**: Template system for customizable system, RAG, and chat prompts
- 🔄 **Real-time Chat**: Context-aware conversations with history and RAG or LLM-only modes
- 🗂️ **Collection Management**: Create, view, and delete vector collections through dedicated UI
- 📊 **Data Exploration**: Browse collection data points with pagination, search, and filtering
- 🔎 **Advanced Search**: Content-based filtering across stored documents with pagination
- 🖼️ **Image Extraction**: Automatic image extraction from PDFs with permanent storage
- ⚙️ **Multi-tier Processing**: Automatic strategy detection with fallback mechanisms for robust document processing

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Main App** (`app.py`): Streamlit entry point with chat-first architecture
- **Session Manager** (`backend/session_manager.py`): Singleton pattern for state management
- **UI Components** (`ui/`): Modular interface components
- **Backend Services** (`backend/`): Core business logic including LLMs, embeddings, and vector DB

## Prerequisites

- Python 3.8+
- Conda (for environment management)
- Docker and Docker Compose (for Qdrant vector database)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag_chatbot
```

### 2. Create Conda Environment

```bash
# Create new conda environment
conda create -n rag-chatbot python=3.9 -y

# Activate the environment
conda activate rag-chatbot
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install OCR system dependencies (recommended for PDF processing)
conda install -c conda-forge tesseract poppler -y

# Verify installation
pip list | grep -E "(streamlit|qdrant|langchain|openai)"
```

### 4. Set Up Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
# Add your OpenAI API key and other settings
```

### 5. Start Qdrant Vector Database

```bash
# Start Qdrant container using Docker Compose
docker-compose up -d

# Verify Qdrant is running
docker-compose logs -f qdrant
```

## Usage

### Running the Application

```bash
# Activate conda environment
conda activate rag-chatbot

# Run the Streamlit application
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`.

### Basic Workflow

1. **Configure LLM**: Set up your OpenAI API key in the sidebar
2. **Select Language**: Choose document language (English by default, or Vietnamese)
3. **Upload Documents**: Navigate to the Upload page to add PDF and CSV files
4. **Select Processing Strategy**: Choose appropriate PDF processing strategy (Auto, Fast, High-Resolution, OCR, or Fallback)
5. **Process Documents**: Select chunking strategy and process your files
6. **Save to Vector DB**: Store processed chunks in Qdrant for retrieval
7. **Chat**: Ask questions about your uploaded documents on the main chat page
8. **Manage Collections**: Use the Data Management page to create, view, and manage vector collections
9. **Explore Data**: Browse and search through stored document data with advanced filtering

### PDF Processing Best Practices

- **Text-based PDFs**: Use "Fast Processing" for quicker results
- **Image-based PDFs**: Use "High Resolution" with OCR for best results
- **Scanned Documents**: Use "OCR Processing" when text is embedded in images
- **Large Files**: Processing may take longer for files >10MB due to OCR operations
- **Image Storage**: Extracted images are saved to the `./figures/` directory automatically

### Supported Document Types

- **PDF files** (.pdf): Advanced processing with OCR and semantic chunking
- **CSV files** (.csv): Text-based processing with simple chunking strategies

### Document Processing Strategies

The system uses a multi-tier processing strategy with automatic detection and fallbacks:

#### PDF Processing Strategies

- **Auto (Recommended)**: Automatically detects optimal processing strategy
- **Fast Processing**: Quick text extraction for text-based PDFs
- **High Resolution**: Advanced processing with OCR for image-based PDFs
- **OCR Processing**: Force OCR processing for scanned documents
- **Basic Fallback**: Simple text extraction with pdfplumber

#### Chunking Options

- **No Chunking**: Keep text as-is without splitting
- **Simple Split**: Split text by sentences using punctuation marks (., !, ?)
- **Semantic Chunking** (PDF only): Intelligent title-based segmentation using `chunk_by_title`

#### OCR Configuration

- **Multi-language Support**: 125+ languages including English and Vietnamese
- **Cross-platform**: Windows, macOS, and Linux compatibility
- **Automatic Detection**: Intelligent text vs image PDF detection
- **Graceful Fallbacks**: Processing continues even if OCR fails

## Configuration

### Environment Variables

Configure these variables in your `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_SERVER=http://localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_chatbot_collection
```

### LLM Options

- **OpenAI**: Currently available models including GPT-4o, GPT-4o Mini, GPT-4 Turbo, and GPT-3.5 Turbo
- **Gemini**: Planned support for Google's Gemini models
- **Local Ollama**: Planned support for local models via Ollama

### Embedding Options

- **OpenAI**: text-embedding-3-small (currently supported)
- **Local Models**: Planned support for sentence-transformer models for English and Vietnamese

### Search Strategies

- **Vector Search**: Semantic similarity search using embeddings
- **Keywords Search**: Traditional keyword-based search
- **Hyde Search**: Hypothetical document embeddings for improved retrieval

## Development

### Docker Management

```bash
# Start Qdrant container
docker-compose up -d

# Stop Qdrant container
docker-compose down

# Restart Qdrant container
docker-compose restart qdrant

# View Qdrant logs
docker-compose logs -f qdrant
```

## Project Structure

```
rag_chatbot/
├── app.py                          # Main Streamlit application
├── backend/                        # Core business logic and services
│   ├── session_manager.py         # Singleton state management
│   ├── document_processor.py      # Main document processing orchestrator
│   ├── collection_management.py   # Vector collection management
│   ├── chunking/                  # Intelligent chunking strategies
│   │   └── semantic_chunker.py    # Semantic chunking implementation
│   ├── embeddings/                # Embedding strategies and factories
│   ├── llms/                      # LLM integrations and factories
│   ├── ocr/                       # OCR integration and configuration
│   │   └── tesseract_ocr.py       # Tesseract OCR wrapper
│   ├── prompts/                   # Prompt template system
│   ├── strategies/                # Document processing strategies
│   │   ├── interfaces.py          # Strategy interfaces
│   │   ├── pdf_strategy.py        # PDF processing strategy
│   │   └── results.py             # Processing result classes
│   ├── services/                  # Core business services
│   └── vector_db/                 # Qdrant integration
│       └── qdrant_manager.py      # Qdrant client and collection management
├── ui/                            # Streamlit UI components
│   ├── chat_main.py              # Chat interface
│   ├── data_upload.py            # Document upload UI with advanced processing
│   ├── data_management.py        # Modular collection management and data exploration UI
│   └── sidebar_navigation.py     # Navigation sidebar
├── config/                        # Configuration constants
├── figures/                       # Extracted images from PDFs
├── plans/                        # Workflow planning documents
├── qdrant_storage/               # Local vector database storage
├── requirements.txt              # Python dependencies
├── docker-compose.yml           # Qdrant container configuration
├── .env.example                  # Environment variables template
├── README.md                     # This file
├── CLAUDE.md                     # Project development guide
└── backend/CLAUDE.md             # Backend architecture documentation
```
