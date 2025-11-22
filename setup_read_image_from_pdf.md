# PDF Image Processing Setup Guide

This document provides a comprehensive guide for reading images from PDF files using OCR (Optical Character Recognition). It covers the libraries, tools, configurations, and deployment considerations for different platforms.

## Table of Contents

1. [Overview](#overview)
2. [Core Libraries](#core-libraries)
3. [System Requirements](#system-requirements)
4. [Installation by Platform](#installation-by-platform)
5. [Configuration Details](#configuration-details)
6. [Cross-Platform Detection](#cross-platform-detection)
7. [Deployment Considerations](#deployment-considerations)
8. [Troubleshooting](#troubleshooting)
9. [Testing & Validation](#testing--validation)

## Overview

The PDF image processing system uses a multi-layer approach to extract text from images embedded in PDF files:

### Processing Pipeline
```
PDF File → Unstructured Library → Image Extraction → Tesseract OCR → Text Extraction → Document Chunks
```

### Key Features
- **Intelligent Fallback System**: Graceful degradation when OCR components are missing
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **Multiple Installation Methods**: Supports conda, pip, system packages
- **Semantic Chunking**: Content-aware segmentation of extracted text
- **Metadata Preservation**: Page numbers, file information, processing methods

## Core Libraries

### Primary Processing Library
- **Unstructured** (`unstructured[pdf]==0.18.20`)
  - Purpose: Intelligent document parsing and image extraction
  - Features: High-resolution processing, table detection, OCR integration
  - Strategy: `hi_res` with `extract_images_in_pdf=True`

### OCR Engine
- **Tesseract OCR** (`tesseract`)
  - Purpose: Optical character recognition from images
  - Version: 5.5.1 (recommended)
  - Language Support: 125+ languages including English (`eng`)

### Python Wrapper
- **pytesseract** (`pytesseract==0.3.13`)
  - Purpose: Python interface for Tesseract OCR
  - Features: Image preprocessing, language configuration

### Fallback Libraries
- **pdfplumber** (`pdfplumber`)
  - Purpose: PDF text extraction when OCR fails
  - Features: Table extraction, metadata preservation

### Support Libraries
- **pikepdf** - PDF processing backend
- **langchain** - Document processing framework
- **pandas** - Data manipulation and storage

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM (8GB+ recommended for large PDFs)
- 1GB disk space for OCR language data

### Platform-Specific Requirements

#### Windows
- Windows 10/11 (x64)
- Microsoft Visual C++ Redistributable
- Administrative access for system installation

#### macOS
- macOS 10.15+ (Catalina or newer)
- Xcode Command Line Tools
- Homebrew (recommended)

#### Linux
- Ubuntu 18.04+, CentOS 7+, or equivalent
- Build tools (gcc, make)
- Package manager (apt, yum, dnf)

## Installation by Platform

### Method 1: Conda (Recommended for Development)

```bash
# Create conda environment
conda create -n rag_chatbot python=3.9 -y
conda activate rag_chatbot

# Install Python dependencies
conda install -c conda-forge poppler tesseract -y
pip install unstructured[pdf]==0.18.20 pdfplumber pytesseract
```

### Method 2: Manual Installation

#### Windows

**Option A: Installer**
1. Download Tesseract from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run installer with "English" language data selected
3. Add Tesseract to PATH during installation

**Option B: Chocolatey**
```powershell
# Run as Administrator
choco install tesseract
```

**Option C: Conda**
```bash
conda install -c conda-forge tesseract poppler
```

#### macOS

**Option A: Homebrew (Recommended)**
```bash
brew install tesseract poppler
```

**Option B: MacPorts**
```bash
sudo port install tesseract poppler
```

#### Linux

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng poppler-utils
```

**CentOS/RHEL:**
```bash
sudo yum install tesseract tesseract-langpack-eng poppler-utils
```

**Fedora:**
```bash
sudo dnf install tesseract tesseract-langpack-eng poppler-utils
```

### Python Dependencies

```bash
pip install unstructured[pdf]==0.18.20 \
            pdfplumber \
            pytesseract \
            pikepdf \
            langchain \
            pandas
```

## Configuration Details

### Environment Variables

#### Critical Variables
- **TESSDATA_PREFIX**: Points to tessdata directory containing language files
- **PATH**: Must include Tesseract executable directory

#### Example Configuration
```bash
# Windows
TESSDATA_PREFIX=C:\Program Files\Tesseract-OCR\tessdata
PATH=%PATH%;C:\Program Files\Tesseract-OCR

# macOS/Linux
TESSDATA_PREFIX=/usr/local/share/tessdata
PATH=$PATH:/usr/local/bin
```

### Language Data Files

Tesseract requires traineddata files for each language:

#### English Location Examples
```
Windows: C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata
macOS:   /usr/local/share/tessdata/eng.traineddata
Linux:   /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata
```

#### Conda Environment Locations
```
Windows: C:\Users\user\miniconda3\envs\rag_chatbot\share\tessdata\eng.traineddata
macOS:   /Users/user/miniconda3/envs/rag_chatbot/share/tessdata/eng.traineddata
Linux:   /home/user/miniconda3/envs/rag_chatbot/share/tessdata/eng.traineddata
```

## Cross-Platform Detection

### Current Machine Configuration

```python
# This system's configuration
System: Windows
Platform: win32
Tesseract Version: 5.5.1
Installation Method: Conda Environment
Tesseract Path: Available in PATH
TESSDATA_PREFIX: C:\Users\vovuo\miniconda3\envs\rag_chatbot\share\tessdata
Language Data: 125 languages available (including eng)
```

### Automatic Detection Strategy

The application implements a robust detection system:

```python
def configure_tesseract():
    """Multi-layer Tesseract detection for cross-platform deployment."""

    # Priority 1: PATH Detection
    if tesseract_in_path():
        return configure_tessdata()

    # Priority 2: Platform-Specific Detection
    if windows():
        return windows_detection()
    elif macos():
        return macos_detection()
    elif linux():
        return linux_detection()

    # Priority 3: Graceful Fallback
    return disable_ocr_with_warning()
```

### Platform Detection Methods

#### Windows Detection
- Registry reading (official installations)
- Environment variables (`ProgramFiles`, `ProgramFiles(x86)`)
- Conda environment (`CONDA_PREFIX`)
- Python environment (`sys.prefix`)

#### macOS Detection
- Homebrew installation (`/opt/homebrew/bin/tesseract`)
- MacPorts (`/opt/local/bin/tesseract`)
- System installation (`/usr/bin/tesseract`)
- Conda environment

#### Linux Detection
- System packages (`/usr/bin/tesseract`)
- Snap packages (`/snap/bin/tesseract`)
- Conda environment
- Local compilation (`/usr/local/bin/tesseract`)

## Deployment Considerations

### Production Environment Setup

#### Docker Configuration
```dockerfile
# Example Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
```

#### Cloud Deployment (AWS/GCP/Azure)

**Key Considerations:**
- Install Tesseract in system packages
- Ensure language data is included
- Set environment variables
- Handle PATH configuration

**AWS Lambda Example:**
```bash
# Include Tesseract binaries in deployment package
mkdir -p lambda/tessdata
cp /usr/share/tesseract-ocr/4.00/tessdata/* lambda/tessdata/
cp /usr/bin/tesseract lambda/
export TESSDATA_PREFIX=/var/task/tessdata
```

#### Container Orchestration (Kubernetes)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: tesseract-config
data:
  TESSDATA_PREFIX: "/usr/share/tesseract-ocr/4.00/tessdata"
---
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
      - name: rag-chatbot
        env:
        - name: TESSDATA_PREFIX
          valueFrom:
            configMapKeyRef:
              name: tesseract-config
              key: TESSDATA_PREFIX
```

### Environment Variable Management

#### Development (.env)
```bash
TESSDATA_PREFIX=/usr/local/share/tessdata
PYTHONPATH=/app
LOG_LEVEL=INFO
```

#### Production (Docker/Cloud)
```bash
TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
PATH=/usr/local/bin:$PATH
PYTHONUNBUFFERED=1
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Unable to get page count. Is poppler installed?"
**Cause**: Poppler library missing
**Solution**:
```bash
# Conda
conda install -c conda-forge poppler

# System packages
sudo apt install poppler-utils  # Linux
brew install poppler              # macOS
```

#### 2. "Tesseract OCR not available"
**Cause**: Tesseract not installed or not in PATH
**Solution**:
```bash
# Verify installation
tesseract --version

# Check PATH
echo $PATH  # Linux/macOS
echo %PATH% # Windows

# Add to PATH if missing
export PATH=$PATH:/usr/local/bin  # Linux/macOS
```

#### 3. "Error opening data file ./eng.traineddata"
**Cause**: TESSDATA_PREFIX not set correctly
**Solution**:
```bash
# Find tessdata directory
find / -name "eng.traineddata" 2>/dev/null

# Set environment variable
export TESSDATA_PREFIX=/path/to/tessdata
```

#### 4. "CompositeElement object has no attribute 'elements'"
**Cause**: Version compatibility issue with unstructured
**Solution**: Update to specified version:
```bash
pip install unstructured[pdf]==0.18.20
```

### Debug Commands

#### Verify Installation
```bash
# Check Tesseract
tesseract --version
tesseract --list-langs

# Check Python integration
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"

# Check Poppler
pdftoppm --version  # Linux/macOS
```

#### Test Document Processing
```python
from backend.document_processor import create_document_processor

# Create processor and test
processor = create_document_processor()
print(f"Supported extensions: {processor.get_supported_extensions()}")
print(f"Tesseract available: {processor.TESSERACT_AVAILABLE}")
```

## Testing & Validation

### Unit Tests
```python
def test_tesseract_configuration():
    """Test Tesseract OCR configuration."""
    import os
    import pytesseract

    # Check version
    version = pytesseract.get_tesseract_version()
    assert version >= (5, 0, 0)

    # Check languages
    result = subprocess.run(['tesseract', '--list-langs'],
                          capture_output=True, text=True)
    assert 'eng' in result.stdout

    # Check tessdata
    tessdata_path = os.environ.get('TESSDATA_PREFIX')
    eng_path = os.path.join(tessdata_path, 'eng.traineddata')
    assert os.path.exists(eng_path)
```

### Integration Tests
```python
def test_pdf_processing():
    """Test end-to-end PDF processing."""
    from backend.document_processor import create_document_processor

    processor = create_document_processor()

    # Test with image-based PDF
    with open('test_image_pdf.pdf', 'rb') as f:
        content = f.read()
        result = processor.process_file(content, 'test.pdf', language='English')

    assert len(result) > 0
    assert result.iloc[0]['content'].strip() != ""
```

### Performance Benchmarks
- **Small PDF (< 1MB)**: 2-5 seconds processing time
- **Medium PDF (1-10MB)**: 5-30 seconds processing time
- **Large PDF (> 10MB)**: 30+ seconds, memory intensive

### Recommended Monitoring
```python
# Log processing metrics
logger.info(f"PDF: {file_name}, Size: {len(file_content)} bytes, "
           f"Elements: {len(elements)}, Chunks: {len(chunks)}, "
           f"Time: {processing_time:.2f}s")
```

## Best Practices

### Development
1. **Use conda environments** for consistent dependencies
2. **Test on multiple platforms** before deployment
3. **Validate OCR configuration** during startup
4. **Implement graceful fallbacks** for missing dependencies

### Production
1. **Dockerize applications** with all dependencies
2. **Set explicit version numbers** for reproducibility
3. **Monitor OCR performance** and resource usage
4. **Implement health checks** for external dependencies

### Security
1. **Validate file types** before processing
2. **Implement size limits** for uploaded files
3. **Scan uploaded files** for malware
4. **Isolate processing** in sandboxed environments

## Quick Start Commands

### Verify Current Setup
```bash
# Check all components
conda activate rag_chatbot

# Test Tesseract
tesseract --version
tesseract --list-langs

# Test Python integration
python -c "import pytesseract; print('✅ OCR ready')"

# Check environment
echo "TESSDATA_PREFIX: $TESSDATA_PREFIX"
```

### Test PDF Processing
```bash
# Run the application
conda activate rag_chatbot
streamlit run app.py

# Upload a PDF with images
# Check logs for OCR processing
# Verify text extraction from images
```

### Troubleshoot Issues
```bash
# Reset environment
conda deactivate
conda activate rag_chatbot

# Reinstall dependencies
conda install -c conda-forge tesseract poppler -y
pip install unstructured[pdf]==0.18.20 --force-reinstall

# Test configuration
python -c "
from backend.document_processor import TESSERACT_AVAILABLE
print(f'Tesseract configured: {TESSERACT_AVAILABLE}')
"
```

This setup guide provides everything needed to configure PDF image processing on any platform, with detailed troubleshooting and deployment considerations.