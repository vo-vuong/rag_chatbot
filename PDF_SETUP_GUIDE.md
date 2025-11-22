# PDF Processing Setup Guide

This guide helps you set up PDF processing capabilities for the RAG chatbot.

## Complete Setup (Recommended)

### Method 1: Full Automatic Installation
```bash
# Activate your environment
conda activate rag_chatbot

# Install all required dependencies
conda install -c conda-forge poppler tesseract
pip install unstructured[pdf]==0.18.20 pdfplumber pytesseract
```

### Method 2: Manual Installation
If you encounter issues, follow these steps:

#### Step 1: Install Poppler via Conda
```bash
conda activate rag_chatbot
conda install -c conda-forge poppler
```

#### Step 2: Install Python Libraries
```bash
conda activate rag_chatbot
pip install unstructured[pdf]==0.18.20
pip install pdfplumber pytesseract
```

#### Step 3: Verify Installation
```bash
conda activate rag_chatbot
python -c "from unstructured.partition.pdf import partition_pdf; print('✅ Unstructured PDF processing available')"
python -c "import pdfplumber; print('✅ pdfplumber available')"
python -c "import pytesseract; print('✅ pytesseract available')"
tesseract --version
```

## How PDF Processing Works

The system uses a **multi-tier fallback strategy**:

1. **Primary**: Unstructured with hi-res strategy (requires Poppler)
2. **Fallback 1**: Unstructured with fast strategy (no Poppler required)
3. **Fallback 2**: pdfplumber (independent library)

### Processing Features

- **Intelligent Text Extraction**: Preserves document structure and layout
- **Semantic Chunking**: Content-aware segmentation
- **Metadata Enrichment**: Page numbers, file types, processing methods
- **Multilingual Support**: English and Vietnamese
- **Error Recovery**: Graceful fallback handling

## Troubleshooting

### Common Issues and Solutions

#### Issue: "Unable to get page count. Is poppler installed and in PATH?"
**Solution**: Install Poppler
```bash
conda activate rag_chatbot
conda install -c conda-forge poppler
```

#### Issue: "Unstructured import error"
**Solution**: Reinstall unstructured
```bash
conda activate rag_chatbot
pip uninstall unstructured
pip install unstructured[pdf]==0.18.20
```

#### Issue: "pdfplumber not available"
**Solution**: Install pdfplumber
```bash
conda activate rag_chatbot
pip install pdfplumber
```

#### Issue: "tesseract is not installed or it's not in your PATH"
**Solution**: Install Tesseract OCR
```bash
conda activate rag_chatbot
conda install -c conda-forge tesseract
pip install pytesseract
```

#### Issue: OCR processing is very slow
**Solution**: This is normal for OCR processing. The system will automatically fall back to faster methods:
- Try the "fast" strategy instead of "hi_res"
- Check if you really need OCR (most text-based PDFs don't)
- The system automatically uses fallbacks when OCR fails

#### Issue: Large PDF processing is slow
**Solution**: The system automatically falls back to faster processing methods:
- Try processing smaller PDF files first
- Check system resources (memory, disk space)
- The system will use fast strategy if hi-res fails

### Error Messages and Meanings

- **"Poppler not available, falling back to fast strategy"**: Normal fallback, processing will continue
- **"Tesseract OCR not available, falling back to fast strategy without OCR"**: OCR missing, will process text-only content
- **"Attempting pdfplumber as fallback"**: Unstructured failed, trying alternative method
- **"PDF processing failed"**: All methods failed, check PDF file integrity
- **"Missing dependency detected"**: System identified missing components and is using alternatives

## Testing Your Setup

### Test with the Application
1. Start the application: `conda activate rag_chatbot && streamlit run app.py`
2. Go to the Upload page
3. Select language (English/Vietnamese)
4. Upload a PDF file
5. Check the processing results

### Test Programmatically
```python
from backend.document_processor import create_document_processor

# Create processor
processor = create_document_processor()
print(f"Supported extensions: {processor.get_supported_extensions()}")

# Test with a PDF file (replace with your file path)
with open('test.pdf', 'rb') as f:
    pdf_content = f.read()
    result = processor.process_file(pdf_content, 'test.pdf', language='English')
    print(f"Processed {len(result)} chunks")
```

## Supported PDF Features

### Text Extraction
- **Document Structure**: Titles, paragraphs, lists
- **Table Recognition**: Basic table structure detection
- **Page Layout**: Understanding of columns and sections
- **Clean Text**: Removal of extra whitespace and special characters

### Metadata Extraction
- **Page Numbers**: Accurate page tracking
- **File Information**: Source file, processing timestamps
- **Processing Method**: Which library was used for extraction
- **Chunk Information**: Length, type, and position details

### Chunking Strategies
- **Semantic**: Content-aware segmentation (when possible)
- **Basic**: Sentence and paragraph-based splitting
- **No Chunking**: Preserve original structure

## Performance Tips

1. **File Size**: Large PDFs may take longer to process
2. **Complex Layouts**: Highly formatted PDFs may need more processing time
3. **Memory Usage**: The system uses efficient streaming for large files
4. **Fallback Speed**: Fast strategy processes quicker but with less accuracy

## Next Steps

Once PDF processing is working:
1. Upload both PDF and CSV files together
2. Test with different PDF formats (text, scanned, mixed)
3. Experiment with different languages
4. Check the quality of extracted text chunks

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify your PDF files are not corrupted
3. Try with a simple text-based PDF first
4. Check the application logs for detailed error messages

The system is designed to be robust and will provide helpful error messages to guide you through any issues.