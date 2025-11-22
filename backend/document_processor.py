"""
Document processing module for RAG chatbot.

This module provides a unified interface for processing different file types
including CSV and PDF files, with extensible architecture for future formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import pandas as pd
from io import BytesIO
import uuid
from datetime import datetime
import os
import subprocess
import platform

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PDF processing imports
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition.auto import partition
    from unstructured.chunking.title import chunk_by_title
    from unstructured.cleaners.core import clean_extra_whitespace, clean_ligatures

    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    UNSTRUCTURED_AVAILABLE = False
    logger.warning(f"Unstructured library not fully available: {e}")

# Fallback PDF processing
try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# Configure Tesseract OCR for cross-platform deployment
def configure_tesseract():
    """Configure Tesseract OCR with dynamic path detection for cross-platform deployment."""
    try:
        import platform
        import subprocess
        import os
        import sys

        logger.info("Configuring Tesseract OCR...")

        # First, try to find tesseract in PATH (works for most installations)
        if _try_tesseract_in_path():
            return _configure_tessdata()

        # If not in PATH, try platform-specific detection
        if platform.system() == "Windows":
            if _try_windows_tesseract_detection():
                return _configure_tessdata()
        elif platform.system() == "Darwin":  # macOS
            if _try_macos_tesseract_detection():
                return _configure_tessdata()
        elif platform.system() == "Linux":
            if _try_linux_tesseract_detection():
                return _configure_tessdata()

        logger.warning("Tesseract not found. OCR functionality will be disabled.")
        return False

    except Exception as e:
        logger.warning(f"Failed to configure Tesseract: {e}")
        return False


def _try_tesseract_in_path() -> bool:
    """Try to run tesseract from system PATH."""
    try:
        result = subprocess.run(["tesseract", "--version"],
                              capture_output=True, check=True,
                              timeout=5, text=True)
        logger.info(f"✅ Tesseract found in PATH: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _try_windows_tesseract_detection() -> bool:
    """Try to find Tesseract on Windows systems."""
    try:
        import winreg

        # Check registry for Tesseract installation
        common_locations = []

        # Try to read from registry (official Windows installations)
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Tesseract-OCR") as key:
                install_path = winreg.QueryValueEx(key, "InstallPath")[0]
                common_locations.append(os.path.join(install_path, "tesseract.exe"))
        except (FileNotFoundError, OSError):
            pass

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Wow6432Node\Tesseract-OCR") as key:
                install_path = winreg.QueryValueEx(key, "InstallPath")[0]
                common_locations.append(os.path.join(install_path, "tesseract.exe"))
        except (FileNotFoundError, OSError):
            pass

        # Add common installation locations
        program_files = [os.environ.get("ProgramFiles", "C:\\Program Files"),
                        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")]

        for pf in program_files:
            common_locations.extend([
                os.path.join(pf, "Tesseract-OCR", "tesseract.exe"),
                os.path.join(pf, "Tesseract", "tesseract.exe")
            ])

        # Try conda/pip installations
        if "CONDA_PREFIX" in os.environ:
            conda_env = os.environ["CONDA_PREFIX"]
            common_locations.extend([
                os.path.join(conda_env, "Scripts", "tesseract.exe"),
                os.path.join(conda_env, "Library", "bin", "tesseract.exe")
            ])

        # Try python environment Scripts directory
        python_scripts = os.path.join(sys.prefix, "Scripts")
        if os.path.exists(python_scripts):
            common_locations.append(os.path.join(python_scripts, "tesseract.exe"))

        # Test each potential location
        for tesseract_path in common_locations:
            if os.path.exists(tesseract_path):
                try:
                    result = subprocess.run([tesseract_path, "--version"],
                                          capture_output=True, check=True,
                                          timeout=5, text=True)
                    logger.info(f"✅ Tesseract found at: {tesseract_path}")

                    # Add to PATH for future calls
                    tesseract_dir = os.path.dirname(tesseract_path)
                    if tesseract_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = tesseract_dir + os.pathsep + os.environ.get("PATH", "")

                    return True
                except subprocess.CalledProcessError:
                    continue

        return False

    except ImportError:
        # winreg not available, fall back to common locations
        return _try_fallback_windows_detection()


def _try_fallback_windows_detection() -> bool:
    """Fallback Windows detection without registry access."""
    common_locations = []

    # Common installation paths
    program_files = [os.environ.get("ProgramFiles", "C:\\Program Files"),
                    os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")]

    for pf in program_files:
        common_locations.extend([
            os.path.join(pf, "Tesseract-OCR", "tesseract.exe"),
            os.path.join(pf, "Tesseract", "tesseract.exe")
        ])

    # Try conda environments
    if "CONDA_PREFIX" in os.environ:
        conda_env = os.environ["CONDA_PREFIX"]
        common_locations.extend([
            os.path.join(conda_env, "Scripts", "tesseract.exe"),
            os.path.join(conda_env, "Library", "bin", "tesseract.exe")
        ])

    # Test each location
    for tesseract_path in common_locations:
        if os.path.exists(tesseract_path):
            try:
                result = subprocess.run([tesseract_path, "--version"],
                                      capture_output=True, check=True,
                                      timeout=5, text=True)
                logger.info(f"✅ Tesseract found at: {tesseract_path}")

                # Add to PATH
                tesseract_dir = os.path.dirname(tesseract_path)
                if tesseract_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = tesseract_dir + os.pathsep + os.environ.get("PATH", "")

                return True
            except subprocess.CalledProcessError:
                continue

    return False


def _try_macos_tesseract_detection() -> bool:
    """Try to find Tesseract on macOS systems."""
    common_locations = [
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "/usr/bin/tesseract",
        "/opt/local/bin/tesseract"  # MacPorts
    ]

    # Try conda environments
    if "CONDA_PREFIX" in os.environ:
        conda_env = os.environ["CONDA_PREFIX"]
        common_locations.extend([
            os.path.join(conda_env, "bin", "tesseract")
        ])

    for tesseract_path in common_locations:
        if os.path.exists(tesseract_path):
            try:
                result = subprocess.run([tesseract_path, "--version"],
                                      capture_output=True, check=True,
                                      timeout=5, text=True)
                logger.info(f"✅ Tesseract found at: {tesseract_path}")
                return True
            except subprocess.CalledProcessError:
                continue

    return False


def _try_linux_tesseract_detection() -> bool:
    """Try to find Tesseract on Linux systems."""
    common_locations = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/snap/bin/tesseract"  # Snap installation
    ]

    # Try conda environments
    if "CONDA_PREFIX" in os.environ:
        conda_env = os.environ["CONDA_PREFIX"]
        common_locations.extend([
            os.path.join(conda_env, "bin", "tesseract")
        ])

    for tesseract_path in common_locations:
        if os.path.exists(tesseract_path):
            try:
                result = subprocess.run([tesseract_path, "--version"],
                                      capture_output=True, check=True,
                                      timeout=5, text=True)
                logger.info(f"✅ Tesseract found at: {tesseract_path}")
                return True
            except subprocess.CalledProcessError:
                continue

    return False


def _configure_tessdata() -> bool:
    """Configure TESSDATA_PREFIX by finding tessdata directory."""
    import sys

    # Priority 1: Check if TESSDATA_PREFIX is already set and valid
    if os.environ.get("TESSDATA_PREFIX"):
        tessdata_dir = os.environ["TESSDATA_PREFIX"]
        if os.path.exists(os.path.join(tessdata_dir, "eng.traineddata")):
            logger.info(f"✅ Using existing TESSDATA_PREFIX: {tessdata_dir}")
            return True
        else:
            logger.warning(f"TESSDATA_PREFIX set but eng.traineddata not found at: {tessdata_dir}")

    # Priority 2: Try conda environment tessdata (most reliable for packaged apps)
    conda_tessdata_paths = []
    if "CONDA_PREFIX" in os.environ:
        conda_env = os.environ["CONDA_PREFIX"]
        conda_tessdata_paths.extend([
            os.path.join(conda_env, "share", "tessdata"),
            os.path.join(conda_env, "lib", "tessdata"),
            os.path.join(conda_env, "Library", "bin", "tessdata")  # Windows conda
        ])

    # Priority 3: Try Python environment tessdata
    python_tessdata = os.path.join(sys.prefix, "share", "tessdata")
    conda_tessdata_paths.append(python_tessdata)

    # Priority 4: Try system locations based on OS
    import platform
    if platform.system() == "Windows":
        # Try to find tessdata relative to tesseract.exe
        try:
            result = subprocess.run(["where", "tesseract"],
                                  capture_output=True, text=True, check=True)
            tesseract_path = result.stdout.strip().split('\n')[0]
            if tesseract_path:
                tesseract_dir = os.path.dirname(tesseract_path)
                conda_tessdata_paths.append(os.path.join(tesseract_dir, "tessdata"))
        except subprocess.CalledProcessError:
            pass

        # Common Windows locations
        program_files = [os.environ.get("ProgramFiles", "C:\\Program Files"),
                        os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")]
        for pf in program_files:
            conda_tessdata_paths.extend([
                os.path.join(pf, "Tesseract-OCR", "tessdata"),
                os.path.join(pf, "Tesseract", "tessdata")
            ])
    elif platform.system() == "Darwin":  # macOS
        conda_tessdata_paths.extend([
            "/usr/local/share/tessdata",
            "/opt/homebrew/share/tessdata",
            "/usr/share/tesseract-ocr/4.00/tessdata"
        ])
    elif platform.system() == "Linux":
        conda_tessdata_paths.extend([
            "/usr/share/tesseract-ocr/4.00/tessdata",
            "/usr/share/tesseract-ocr/tessdata",
            "/usr/local/share/tessdata"
        ])

    # Test each tessdata location
    for tessdata_path in conda_tessdata_paths:
        if os.path.exists(tessdata_path) and os.path.exists(os.path.join(tessdata_path, "eng.traineddata")):
            os.environ["TESSDATA_PREFIX"] = tessdata_path
            logger.info(f"✅ TESSDATA_PREFIX set to: {tessdata_path}")
            return True

    logger.warning("❌ Could not find tessdata directory with eng.traineddata")
    logger.info("💡 To enable OCR, ensure Tesseract is installed with English language data")
    return False

# Configure Tesseract at module import time
TESSERACT_AVAILABLE = configure_tesseract()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessorStrategy(ABC):
    """Abstract base class for document processing strategies."""

    @abstractmethod
    def process(self, file_content: bytes, file_name: str, **kwargs) -> pd.DataFrame:
        """
        Process file content and return a standardized DataFrame.

        Args:
            file_content: Raw file content as bytes
            file_name: Original file name
            **kwargs: Additional processing parameters

        Returns:
            DataFrame with standardized columns: ['doc_id', 'content', 'metadata']
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass


class PDFProcessor(DocumentProcessorStrategy):
    """PDF document processor using Unstructured for intelligent text extraction."""

    def __init__(self):
        self.supported_extensions = ["pdf"]

    def process(self, file_content: bytes, file_name: str, **kwargs) -> pd.DataFrame:
        """
        Process PDF file using Unstructured with intelligent partitioning.

        Args:
            file_content: PDF file content as bytes
            file_name: Original file name
            **kwargs: Additional parameters (language, chunking_strategy, etc.)

        Returns:
            DataFrame with extracted text and metadata
        """
        try:
            language = kwargs.get("language", "English")
            chunking_strategy = kwargs.get("chunking_strategy", "semantic")

            logger.info(
                f"Processing PDF: {file_name} with strategy: {chunking_strategy}"
            )
            logger.info(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX', 'Not set')}")
            logger.info(f"Tesseract available: {TESSERACT_AVAILABLE}")

            # Process PDF using Unstructured with fallback strategy
            try:
                # Try high-resolution strategy first (requires Poppler)
                elements = partition_pdf(
                    file=BytesIO(file_content),
                    strategy="hi_res",  # Use high-resolution processing for better accuracy
                    infer_table_structure=True,  # Extract table structure
                    extract_images_in_pdf=True,  # Extract images for OCR processing
                    languages=[self._get_language_code(language)],
                    include_page_breaks=True,
                    starting_page_number=1,
                )
            except Exception as unstructured_error:
                error_msg = str(unstructured_error).lower()

                # Debug: Log the actual error
                logger.warning(f"Unstructured processing failed: {unstructured_error}")

                # Check for OCR/Tesseract errors
                if "tesseract" in error_msg or "ocr" in error_msg or "traineddata" in error_msg:
                    logger.warning(
                        f"Tesseract OCR not available ({unstructured_error}), falling back to fast strategy without OCR"
                    )

                    # Provide specific guidance for missing traineddata
                    if "traineddata" in error_msg:
                        logger.warning(
                            "Tesseract language data missing. To enable OCR:\n"
                            "1. Install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                            "2. Ensure English language data is installed during setup\n"
                            "3. Or use conda: conda install -c conda-forge tesseract -y"
                        )

                    # Fallback to fast strategy without OCR
                    elements = partition_pdf(
                        file=BytesIO(file_content),
                        strategy="fast",  # Use fast strategy that doesn't require OCR
                        languages=[self._get_language_code(language)],
                        include_page_breaks=True,
                        starting_page_number=1,
                    )

                # Check for Poppler errors
                elif "poppler" in error_msg or "page count" in error_msg:
                    logger.warning(
                        "Poppler not available, falling back to fast strategy"
                    )
                    # Fallback to fast strategy (doesn't require Poppler)
                    elements = partition_pdf(
                        file=BytesIO(file_content),
                        strategy="fast",  # Use fast strategy that doesn't require Poppler
                        languages=[self._get_language_code(language)],
                        include_page_breaks=True,
                        starting_page_number=1,
                    )

                # Check for other dependency errors
                elif "not installed" in error_msg or "not in PATH" in error_msg:
                    logger.warning(
                        f"Missing dependency detected: {unstructured_error}. Using pdfplumber fallback."
                    )
                    raise  # Let the outer exception handler use pdfplumber

                else:
                    raise  # Re-raise if it's not a recognized dependency error

            logger.info(f"Extracted {len(elements)} elements from PDF")

            # Apply text cleaning
            for element in elements:
                element.text = clean_extra_whitespace(element.text)
                element.text = clean_ligatures(element.text)

            # Apply chunking strategy with error handling
            try:
                if chunking_strategy == "semantic":
                    chunks = self._semantic_chunking(elements, file_name)
                elif chunking_strategy == "basic":
                    chunks = self._basic_chunking(elements, file_name)
                else:
                    chunks = self._no_chunking(elements, file_name)
            except Exception as chunking_error:
                logger.warning(
                    f"Semantic chunking failed, falling back to basic chunking: {str(chunking_error)}"
                )
                try:
                    chunks = self._basic_chunking(elements, file_name)
                except Exception as basic_chunking_error:
                    logger.warning(
                        f"Basic chunking also failed, using no chunking: {str(basic_chunking_error)}"
                    )
                    chunks = self._no_chunking(elements, file_name)

            # Create DataFrame
            df = pd.DataFrame(chunks)

            logger.info(f"Created {len(chunks)} chunks from PDF: {file_name}")
            return df

        except Exception as e:
            logger.error(
                f"Error processing PDF {file_name} with Unstructured: {str(e)}"
            )

            # Try pdfplumber as fallback
            if PDFPLUMBER_AVAILABLE:
                try:
                    logger.info("Attempting pdfplumber as fallback...")
                    return self._process_with_pdfplumber_fallback(
                        file_content, file_name, chunking_strategy
                    )
                except Exception as pdfplumber_error:
                    logger.error(
                        f"pdfplumber fallback also failed: {str(pdfplumber_error)}"
                    )

            # All methods failed
            error_msg = f"PDF processing failed: {str(e)}"
            return pd.DataFrame(
                [
                    {
                        "doc_id": str(uuid.uuid4()),
                        "content": error_msg,
                        "metadata": {
                            "source_file": file_name,
                            "file_type": "pdf",
                            "processing_error": True,
                            "error_message": error_msg,
                            "suggestion": "Ensure Poppler is installed or try a different PDF file.",
                        },
                    }
                ]
            )

    def _semantic_chunking(self, elements: List, file_name: str) -> List[Dict]:
        """Apply semantic chunking using title-based segmentation."""
        chunks = chunk_by_title(
            elements,
            max_characters=1500,
            new_after_n_chars=1200,
            combine_text_under_n_chars=400,
            multipage_sections=True,
        )

        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract page numbers from metadata - handle different chunk types
            page_numbers = []

            # Handle CompositeElement (has .elements attribute)
            if hasattr(chunk, "elements") and hasattr(chunk.elements, "__iter__"):
                page_numbers = list(
                    set(
                        [
                            el.metadata.page_number
                            for el in chunk.elements
                            if hasattr(el, "metadata")
                            and hasattr(el.metadata, "page_number")
                        ]
                    )
                )
            # Handle individual elements
            elif hasattr(chunk, "metadata") and hasattr(chunk.metadata, "page_number"):
                page_numbers = [chunk.metadata.page_number]

            # Default page number if none found
            if not page_numbers:
                page_numbers = [1]

            processed_chunks.append(
                {
                    "doc_id": str(uuid.uuid4()),
                    "content": chunk.text,
                    "metadata": {
                        "source_file": file_name,
                        "file_type": "pdf",
                        "chunk_index": i,
                        "page_numbers": page_numbers,
                        "element_count": (
                            len(chunk.elements)
                            if hasattr(chunk, "elements")
                            and hasattr(chunk.elements, "__len__")
                            else 1
                        ),
                        "chunk_type": "semantic",
                        "chunk_length": len(chunk.text),
                        "extraction_timestamp": datetime.now().isoformat(),
                        "element_type": type(chunk).__name__,
                    },
                }
            )

        return processed_chunks

    def _basic_chunking(self, elements: List, file_name: str) -> List[Dict]:
        """Apply basic chunking by character count."""
        chunks = []
        current_chunk = ""
        chunk_index = 0
        current_page = 1

        for element in elements:
            element_text = element.text.strip()
            if not element_text:
                continue

            # Add element to current chunk
            if len(current_chunk) + len(element_text) > 1000 and current_chunk:
                # Save current chunk
                chunks.append(
                    {
                        "doc_id": str(uuid.uuid4()),
                        "content": current_chunk.strip(),
                        "metadata": {
                            "source_file": file_name,
                            "file_type": "pdf",
                            "chunk_index": chunk_index,
                            "page_numbers": [current_page],
                            "chunk_type": "basic",
                            "chunk_length": len(current_chunk),
                            "extraction_timestamp": datetime.now().isoformat(),
                        },
                    }
                )

                current_chunk = element_text
                chunk_index += 1
            else:
                if not current_chunk:
                    current_chunk = element_text
                else:
                    current_chunk += "\n\n" + element_text

            # Update page number if available
            if hasattr(element.metadata, "page_number"):
                current_page = element.metadata.page_number

        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append(
                {
                    "doc_id": str(uuid.uuid4()),
                    "content": current_chunk.strip(),
                    "metadata": {
                        "source_file": file_name,
                        "file_type": "pdf",
                        "chunk_index": chunk_index,
                        "page_numbers": [current_page],
                        "chunk_type": "basic",
                        "chunk_length": len(current_chunk),
                        "extraction_timestamp": datetime.now().isoformat(),
                    },
                }
            )

        return chunks

    def _no_chunking(self, elements: List, file_name: str) -> List[Dict]:
        """Create a single chunk with all content."""
        all_text = "\n\n".join([el.text.strip() for el in elements if el.text.strip()])
        page_numbers = list(
            set(
                [
                    el.metadata.page_number
                    for el in elements
                    if hasattr(el.metadata, "page_number")
                ]
            )
        )

        return [
            {
                "doc_id": str(uuid.uuid4()),
                "content": all_text,
                "metadata": {
                    "source_file": file_name,
                    "file_type": "pdf",
                    "chunk_index": 0,
                    "page_numbers": page_numbers,
                    "element_count": len(elements),
                    "chunk_type": "no_chunking",
                    "chunk_length": len(all_text),
                    "extraction_timestamp": datetime.now().isoformat(),
                },
            }
        ]

    def _get_language_code(self, language: str) -> str:
        """Convert language name to ISO code for Unstructured."""
        language_map = {
            "English": "eng",
            "Vietnamese": "vie",
            "French": "fra",
            "German": "deu",
            "Spanish": "spa",
        }
        return language_map.get(language, "eng")

    def _process_with_pdfplumber_fallback(
        self, file_content: bytes, file_name: str, chunking_strategy: str
    ) -> pd.DataFrame:
        """Process PDF using pdfplumber as fallback."""
        import pdfplumber

        with pdfplumber.open(BytesIO(file_content)) as pdf:
            chunks = []

            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    if chunking_strategy == "no_chunking":
                        # Treat each page as one chunk
                        chunks.append(
                            {
                                "doc_id": str(uuid.uuid4()),
                                "content": text.strip(),
                                "metadata": {
                                    "source_file": file_name,
                                    "file_type": "pdf",
                                    "chunk_index": len(chunks),
                                    "page_numbers": [page_num + 1],
                                    "chunk_type": "page",
                                    "chunk_length": len(text.strip()),
                                    "extraction_timestamp": datetime.now().isoformat(),
                                    "processing_method": "pdfplumber_fallback",
                                },
                            }
                        )
                    else:
                        # Simple paragraph-based chunking
                        paragraphs = text.split("\n\n")
                        for paragraph in paragraphs:
                            paragraph = paragraph.strip()
                            if (
                                paragraph and len(paragraph) > 20
                            ):  # Filter very short lines
                                chunks.append(
                                    {
                                        "doc_id": str(uuid.uuid4()),
                                        "content": paragraph,
                                        "metadata": {
                                            "source_file": file_name,
                                            "file_type": "pdf",
                                            "chunk_index": len(chunks),
                                            "page_numbers": [page_num + 1],
                                            "chunk_type": "paragraph",
                                            "chunk_length": len(paragraph),
                                            "extraction_timestamp": datetime.now().isoformat(),
                                            "processing_method": "pdfplumber_fallback",
                                        },
                                    }
                                )

        logger.info(
            f"Extracted {len(chunks)} chunks from PDF using pdfplumber fallback"
        )
        return pd.DataFrame(chunks)

    def get_supported_extensions(self) -> List[str]:
        return self.supported_extensions


class CSVProcessor(DocumentProcessorStrategy):
    """CSV document processor using pandas for structured data."""

    def __init__(self):
        self.supported_extensions = ["csv"]

    def process(self, file_content: bytes, file_name: str, **kwargs) -> pd.DataFrame:
        """
        Process CSV file using pandas.

        Args:
            file_content: CSV file content as bytes
            file_name: Original file name
            **kwargs: Additional parameters

        Returns:
            DataFrame with CSV data and metadata
        """
        try:
            logger.info(f"Processing CSV: {file_name}")

            # Read CSV from bytes
            df = pd.read_csv(BytesIO(file_content))

            # Create document IDs
            doc_ids = [str(uuid.uuid4()) for _ in range(len(df))]
            df["doc_id"] = doc_ids

            logger.info(f"Processed {len(df)} rows from CSV: {file_name}")
            return df

        except Exception as e:
            logger.error(f"Error processing CSV {file_name}: {str(e)}")
            # Return empty DataFrame with error information
            return pd.DataFrame(
                [
                    {
                        "doc_id": str(uuid.uuid4()),
                        "content": f"Error processing CSV: {str(e)}",
                        "metadata": {
                            "source_file": file_name,
                            "file_type": "csv",
                            "processing_error": True,
                            "error_message": str(e),
                        },
                    }
                ]
            )

    def get_supported_extensions(self) -> List[str]:
        return self.supported_extensions


class DocumentProcessor:
    """Main document processor that routes to appropriate strategy."""

    def __init__(self):
        self.strategies = {"pdf": PDFProcessor(), "csv": CSVProcessor()}
        self.supported_extensions = []
        for strategy in self.strategies.values():
            self.supported_extensions.extend(strategy.get_supported_extensions())

    def process_file(
        self, file_content: bytes, file_name: str, **kwargs
    ) -> pd.DataFrame:
        """
        Process a file using the appropriate strategy.

        Args:
            file_content: Raw file content as bytes
            file_name: Original file name with extension
            **kwargs: Additional processing parameters

        Returns:
            DataFrame with processed content and metadata
        """
        # Extract file extension
        file_extension = file_name.split(".")[-1].lower()

        # Get appropriate strategy
        if file_extension not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file extension: {file_extension}. Supported: {self.supported_extensions}"
            )

        strategy = self.strategies.get(file_extension)
        if not strategy:
            raise ValueError(f"No strategy found for file type: {file_extension}")

        return strategy.process(file_content, file_name, **kwargs)

    def get_supported_extensions(self) -> List[str]:
        """Return list of all supported file extensions."""
        return self.supported_extensions

    def add_strategy(self, extension: str, strategy: DocumentProcessorStrategy):
        """Add a new processing strategy for a file extension."""
        self.strategies[extension] = strategy
        self.supported_extensions.append(extension)


# Factory function for easy instantiation
def create_document_processor() -> DocumentProcessor:
    """Create and return a DocumentProcessor instance."""
    return DocumentProcessor()


# Utility function for processing multiple files
def process_multiple_files(
    files_data: List[Tuple[bytes, str]], **kwargs
) -> pd.DataFrame:
    """
    Process multiple files and combine results into a single DataFrame.

    Args:
        files_data: List of tuples (file_content, file_name)
        **kwargs: Additional processing parameters

    Returns:
        Combined DataFrame with all processed content
    """
    processor = create_document_processor()
    all_data = []

    for file_content, file_name in files_data:
        try:
            df = processor.process_file(file_content, file_name, **kwargs)
            all_data.append(df)
        except Exception as e:
            logger.error(f"Failed to process {file_name}: {str(e)}")
            # Add error entry
            error_data = pd.DataFrame(
                [
                    {
                        "doc_id": str(uuid.uuid4()),
                        "content": f"Error processing file {file_name}: {str(e)}",
                        "metadata": {
                            "source_file": file_name,
                            "processing_error": True,
                            "error_message": str(e),
                        },
                    }
                ]
            )
            all_data.append(error_data)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage and testing
    processor = create_document_processor()
    print(f"Supported extensions: {processor.get_supported_extensions()}")
