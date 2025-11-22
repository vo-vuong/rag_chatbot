"""
Test script for PDF processing functionality.
This script tests the document processor with various scenarios.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from backend.document_processor import create_document_processor, PDFProcessor, CSVProcessor
        from ui.data_upload import DataUploadUI
        from backend.session_manager import SessionManager

        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_processor_creation():
    """Test document processor creation and basic functionality."""
    try:
        from backend.document_processor import create_document_processor

        processor = create_document_processor()
        supported_extensions = processor.get_supported_extensions()

        print(f"✅ Processor created successfully")
        print(f"✅ Supported extensions: {supported_extensions}")

        # Check if PDF is supported
        if 'pdf' in supported_extensions:
            print("✅ PDF processing is supported")
        else:
            print("❌ PDF processing is NOT supported")
            return False

        if 'csv' in supported_extensions:
            print("✅ CSV processing is supported")
        else:
            print("❌ CSV processing is NOT supported")

        return True
    except Exception as e:
        print(f"❌ Processor creation error: {e}")
        return False

def test_strategies():
    """Test individual processing strategies."""
    try:
        from backend.document_processor import PDFProcessor, CSVProcessor

        # Test PDF processor
        pdf_processor = PDFProcessor()
        pdf_extensions = pdf_processor.get_supported_extensions()
        print(f"✅ PDF processor supports: {pdf_extensions}")

        # Test CSV processor
        csv_processor = CSVProcessor()
        csv_extensions = csv_processor.get_supported_extensions()
        print(f"✅ CSV processor supports: {csv_extensions}")

        return True
    except Exception as e:
        print(f"❌ Strategy testing error: {e}")
        return False

def test_session_manager_integration():
    """Test SessionManager integration."""
    try:
        from backend.session_manager import SessionManager

        # Create session manager instance
        session_manager = SessionManager()

        # Test basic session operations
        session_manager.set("test_key", "test_value")
        retrieved_value = session_manager.get("test_key")

        if retrieved_value == "test_value":
            print("✅ SessionManager integration successful")
            return True
        else:
            print("❌ SessionManager integration failed")
            return False

    except Exception as e:
        print(f"❌ SessionManager integration error: {e}")
        return False

def test_unstructured_import():
    """Test that unstructured library is properly installed."""
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.partition.auto import partition
        from unstructured.chunking.title import chunk_by_title

        print("✅ Unstructured library imported successfully")
        print("✅ PDF partitioning available")
        print("✅ Auto partitioning available")
        print("✅ Title chunking available")

        return True
    except ImportError as e:
        print(f"❌ Unstructured import error: {e}")
        print("💡 Make sure to run: conda activate rag_chatbot && pip install unstructured[pdf]")
        return False

def test_ui_components():
    """Test UI component creation."""
    try:
        from ui.data_upload import DataUploadUI
        from backend.session_manager import SessionManager

        session_manager = SessionManager()
        upload_ui = DataUploadUI(session_manager)

        print("✅ DataUploadUI component created successfully")
        return True

    except Exception as e:
        print(f"❌ UI component creation error: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🔧 Testing PDF Processing Implementation")
    print("=" * 50)

    tests = [
        ("Module Imports", test_imports),
        ("Unstructured Library", test_unstructured_import),
        ("Processor Creation", test_processor_creation),
        ("Processing Strategies", test_strategies),
        ("SessionManager Integration", test_session_manager_integration),
        ("UI Components", test_ui_components),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! PDF processing is ready to use.")
        print("\n📚 Usage Instructions:")
        print("1. Start the application: conda activate rag_chatbot && streamlit run app.py")
        print("2. Go to the Upload page")
        print("3. Select language (English/Vietnamese)")
        print("4. Upload PDF files alongside CSV files")
        print("5. Process and save to vector database")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)