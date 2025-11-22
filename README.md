# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit that allows users to upload documents, process them using various chunking strategies, and ask questions answered using AI-powered retrieval from document chunks.

## Features

- 🤖 **LLM Support**: OpenAI models (with planned support for Gemini and local Ollama models)
- 📄 **Advanced Document Processing**: Support for CSV and PDF files with OCR capabilities
- 🔍 **Intelligent Chunking**: Semantic chunking strategies with title-based segmentation
- 🔬 **OCR Integration**: Tesseract OCR for extracting text from images in PDFs
- 💾 **Vector Database**: Qdrant integration for efficient document retrieval with comprehensive collection management
- 🌐 **Multi-language**: Support for English and Vietnamese
- 🎯 **Vector Search**: Qdrant-based similarity search for document retrieval
- 📝 **Prompt Management**: Template system for customizable prompts
- 🔄 **Real-time Chat**: Context-aware conversations with history
- 🗂️ **Collection Management**: Create, view, and delete vector collections through dedicated UI
- 📊 **Data Exploration**: Browse collection data points with pagination, search, and filtering
- 🔎 **Advanced Search**: Content-based filtering across stored documents with pagination
- 🛠️ **Cross-Platform**: Works on Windows, macOS, and Linux with automatic dependency detection

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
- Git

### OCR System Requirements (Optional, for PDF image processing)

- **Tesseract OCR**: Automatic installation via conda, or manual installation
- **Poppler**: PDF rendering library (automatically installed with conda)
- **4GB+ RAM**: Recommended for processing large PDFs with images

> **Note**: The system automatically detects and configures OCR components. If Tesseract or Poppler are not available, PDF processing will gracefully fallback to text-only extraction.

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
# Install Python dependencies (includes OCR components)
pip install -r requirements.txt

# Optional: Install OCR system dependencies via conda (recommended)
conda install -c conda-forge tesseract poppler -y

# Verify installation
pip list | grep -E "(streamlit|qdrant|langchain|openai|unstructured|tesseract)"
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
3. **Upload Documents**: Navigate to the Upload page to add CSV and PDF files
4. **Process Documents**: Select chunking strategy and process your files
   - **PDF files**: Automatic OCR processing for images, semantic chunking
   - **CSV files**: Text-based processing with simple chunking
5. **Save to Vector DB**: Store processed chunks in Qdrant for retrieval
6. **Chat**: Ask questions about your uploaded documents on the main chat page
7. **Manage Collections**: Use the Data Management page to create, view, and manage vector collections
8. **Explore Data**: Browse and search through stored document data with advanced filtering

### Supported Document Types

- **CSV files (.csv)**: Structured data with automatic text extraction
- **PDF files (.pdf)**: Advanced processing with OCR capabilities
  - Text-based PDFs: Direct text extraction
  - Image-based PDFs: OCR text recognition using Tesseract
  - Mixed PDFs: Combined text and image processing

### Document Chunking Strategies

- **No Chunking**: Keep text as-is without splitting
- **Simple Split**: Split text by sentences using punctuation marks (., !, ?)
- **Semantic Chunking**: Content-aware segmentation using title-based structure (PDFs)
  - Preserves document hierarchy and structure
  - Intelligent section detection
  - Context-aware chunk boundaries

### OCR Capabilities

- **Automatic Detection**: System automatically detects and configures OCR components
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **Multi-Language**: Support for 125+ languages including English and Vietnamese
- **Graceful Fallback**: Continues processing even if OCR components are missing
- **Performance Optimized**: Efficient processing of large PDF files

For detailed OCR setup instructions, see [setup_read_image_from_pdf.md](setup_read_image_from_pdf.md).

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

## Development

### Code Formatting

This project uses Black for code formatting. Format your code before committing:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Clear Cache

If you encounter issues with Streamlit caching:

```bash
# Clear Streamlit cache
streamlit cache clear
```

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
├── backend/                        # Core business logic
│   ├── session_manager.py         # Singleton state management
│   ├── collection_management.py   # Vector collection management
│   ├── document_processor.py      # Advanced document processing with OCR
│   ├── embeddings/                # Embedding strategies
│   ├── llms/                      # LLM integrations
│   ├── prompts/                   # Prompt template system
│   └── vector_db/                 # Qdrant integration
│       └── qdrant_manager.py      # Qdrant client and collection management
├── ui/                            # Streamlit UI components
│   ├── chat_main.py              # Chat interface
│   ├── data_upload.py            # Document upload UI (CSV + PDF support)
│   ├── data_management.py        # Modular collection management and data exploration UI
│   └── sidebar_navigation.py     # Navigation sidebar
├── config/                        # Configuration constants
├── plans/                        # Workflow planning documents
├── qdrant_storage/               # Local vector database storage
├── requirements.txt              # Python dependencies (includes OCR components)
├── docker-compose.yml           # Qdrant container configuration
├── .env.example                  # Environment variables template
├── setup_read_image_from_pdf.md  # Comprehensive OCR setup guide
└── README.md                     # This file
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**

   - Ensure Docker is running
   - Check if Qdrant container is up: `docker-compose ps`
   - Verify port 6333 is available

2. **OpenAI API Errors**

   - Check your API key in the sidebar or `.env` file
   - Verify you have sufficient API credits
   - Check network connectivity

3. **PDF Processing Issues**

   - **Tesseract OCR Errors**: See [setup_read_image_from_pdf.md](setup_read_image_from_pdf.md) for detailed troubleshooting
   - **"Unable to get page count"**: Install Poppler: `conda install -c conda-forge poppler`
   - **"eng.traineddata not found"**: Install Tesseract with language data: `conda install -c conda-forge tesseract`
   - **Processing timeouts**: Reduce file size or ensure sufficient RAM (4GB+ recommended)
   - **OCR warnings**: Normal fallback behavior - processing continues with text extraction

4. **CSV File Upload Issues**

   - Ensure CSV files are properly formatted
   - Check for encoding issues (UTF-8 recommended)
   - Verify file size is reasonable (under 100MB recommended)

5. **Memory Issues**

   - Restart Qdrant container: `docker-compose restart qdrant`
   - Clear Streamlit cache: `streamlit cache clear`
   - Reduce PDF file size or process in smaller batches

6. **Conda Environment Issues**
   - Ensure you're using the correct conda environment: `conda activate rag-chatbot`
   - Verify Python version: `python --version`

### OCR-Specific Troubleshooting

For detailed OCR setup and troubleshooting, including platform-specific installations and configuration, refer to the comprehensive [PDF Processing Setup Guide](setup_read_image_from_pdf.md).

#### Quick OCR Verification

```bash
# Check Tesseract installation
tesseract --version
tesseract --list-langs

# Verify Python integration
conda activate rag-chatbot
python -c "import pytesseract; print('✅ OCR ready')"
```

### Logs and Debugging

- Check console output for detailed logging
- Use the SessionManager status methods for system validation
- Monitor Qdrant logs: `docker-compose logs -f qdrant`
