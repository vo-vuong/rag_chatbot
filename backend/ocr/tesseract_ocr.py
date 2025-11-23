"""
Tesseract OCR integration with cross-platform support.

This module provides comprehensive OCR functionality with automatic detection
and configuration of Tesseract installations across different platforms.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import pytesseract
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logger = logging.getLogger(__name__)

# Supported OCR languages with their Tesseract codes
SUPPORTED_LANGUAGES = {
    "en": "eng",  # English
    "vi": "vie",  # Vietnamese
    "auto": "eng+vie",  # Auto-detect with both English and Vietnamese
}

# Default OCR language
DEFAULT_OCR_LANG = "eng+vie"


class TesseractOCR:
    """
    Tesseract OCR integration with cross-platform support and automatic configuration.

    This class provides a high-level interface for OCR operations with automatic
    detection of Tesseract installations, cross-platform support, and multi-language
    capabilities.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        tesseract_path: Optional[str] = None,
    ):
        """
        Initialize Tesseract OCR with automatic configuration.

        Args:
            languages: List of language codes for OCR (e.g., ['en', 'vi'])
            tesseract_path: Custom path to Tesseract executable
        """
        self.languages = languages or ["en", "vi"]
        self.tesseract_path = tesseract_path
        self.is_configured = False
        self.tesseract_version = None
        self.supported_languages = set()

        # Configure Tesseract on initialization
        self._configure_tesseract()

    def _configure_tesseract(self) -> bool:
        """
        Configure Tesseract with cross-platform detection.

        Returns:
            True if configuration was successful, False otherwise
        """
        try:
            # Set custom Tesseract path if provided
            if self.tesseract_path:
                if os.path.exists(self.tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                    logger.info(f"Using custom Tesseract path: {self.tesseract_path}")
                else:
                    logger.warning(
                        f"Custom Tesseract path not found: {self.tesseract_path}"
                    )
                    return self._detect_tesseract()

            return self._detect_tesseract()

        except Exception as e:
            logger.error(f"Failed to configure Tesseract: {e}")
            return False

    def _detect_tesseract(self) -> bool:
        """
        Detect Tesseract installation across different platforms.

        Returns:
            True if Tesseract was detected and configured successfully
        """
        system = platform.system().lower()

        # Try common Tesseract paths based on platform
        common_paths = self._get_common_tesseract_paths(system)

        # Add current conda environment to PATH
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            conda_paths = [
                os.path.join(conda_prefix, "Library", "bin", "tesseract.exe"),
                os.path.join(conda_prefix, "bin", "tesseract"),
                os.path.join(conda_prefix, "Scripts", "tesseract.exe"),
            ]
            common_paths.extend(conda_paths)

        # Try to find Tesseract in common paths
        for path in common_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Found Tesseract at: {path}")
                return self._validate_tesseract_installation()

        # Try to find Tesseract in PATH
        try:
            tesseract_cmd = shutil.which("tesseract")
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                logger.info(f"Found Tesseract in PATH: {tesseract_cmd}")
                return self._validate_tesseract_installation()
        except Exception as e:
            logger.debug(f"Error searching Tesseract in PATH: {e}")

        logger.warning("Tesseract not found. OCR functionality will be limited.")
        return False

    def _get_common_tesseract_paths(self, system: str) -> List[str]:
        """
        Get common Tesseract installation paths for the current platform.

        Args:
            system: Current operating system

        Returns:
            List of common Tesseract paths
        """
        if system == "windows":
            return [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\tools\tesseract\tesseract.exe",
                os.path.expanduser(
                    r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
                ),
            ]
        elif system == "darwin":  # macOS
            return [
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
                "/usr/bin/tesseract",
                "/opt/local/bin/tesseract",
            ]
        else:  # Linux and others
            return [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/snap/bin/tesseract",
                "/opt/tesseract/bin/tesseract",
            ]

    def _validate_tesseract_installation(self) -> bool:
        """
        Validate Tesseract installation and get version information.

        Returns:
            True if installation is valid, False otherwise
        """
        try:
            # Test Tesseract version
            result = subprocess.run(
                [pytesseract.pytesseract.tesseract_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Extract version from output
                version_line = result.stdout.strip().split('\n')[0]
                if "tesseract" in version_line.lower():
                    version_parts = version_line.split()
                    if len(version_parts) >= 2:
                        self.tesseract_version = version_parts[1]
                        logger.info(f"Tesseract version: {self.tesseract_version}")

                # Get supported languages
                self._get_supported_languages()

                # Configure TESSDATA_PREFIX if needed
                self._configure_tessdata()

                self.is_configured = True
                return True
            else:
                logger.error(f"Tesseract validation failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Tesseract validation timed out")
            return False
        except Exception as e:
            logger.error(f"Error validating Tesseract installation: {e}")
            return False

    def _get_supported_languages(self) -> None:
        """
        Get list of languages supported by the Tesseract installation.

        Populates self.supported_languages with available language codes.
        """
        try:
            result = subprocess.run(
                [pytesseract.pytesseract.tesseract_cmd, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0:
                languages = set()
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header line
                    lang = line.strip()
                    if lang:
                        languages.add(lang)

                self.supported_languages = languages
                logger.info(
                    f"Found {len(languages)} supported languages: {sorted(languages)}"
                )
            else:
                logger.warning(f"Failed to get supported languages: {result.stderr}")

        except Exception as e:
            logger.warning(f"Error getting supported languages: {e}")
            self.supported_languages = {"eng"}  # Default to English

    def _configure_tessdata(self) -> None:
        """
        Configure TESSDATA_PREFIX environment variable for language data.
        """
        # Try to find tessdata directory
        tessdata_paths = self._get_tessdata_paths()

        for path in tessdata_paths:
            if os.path.exists(path):
                os.environ["TESSDATA_PREFIX"] = path
                logger.info(f"Set TESSDATA_PREFIX to: {path}")
                return

        logger.debug("TESSDATA_PREFIX not configured, using default locations")

    def _get_tessdata_paths(self) -> List[str]:
        """
        Get possible tessdata directory paths.

        Returns:
            List of possible tessdata paths
        """
        system = platform.system().lower()
        tesseract_dir = os.path.dirname(pytesseract.pytesseract.tesseract_cmd)

        paths = []

        if system == "windows":
            paths.extend(
                [
                    os.path.join(tesseract_dir, "tessdata"),
                    r"C:\Program Files\Tesseract-OCR\tessdata",
                    r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
                ]
            )
        else:
            paths.extend(
                [
                    "/usr/share/tesseract-ocr/4.00/tessdata",
                    "/usr/share/tesseract-ocr/tessdata",
                    "/usr/local/share/tessdata",
                    os.path.join(tesseract_dir, "share", "tessdata"),
                ]
            )

        # Add conda environment tessdata
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            paths.extend(
                [
                    os.path.join(conda_prefix, "share", "tessdata"),
                    os.path.join(conda_prefix, "Library", "share", "tessdata"),
                ]
            )

        return paths

    def get_language_code(self, languages: List[str]) -> str:
        """
        Convert language codes to Tesseract language string.

        Args:
            languages: List of language codes (e.g., ['en', 'vi'])

        Returns:
            Tesseract language string (e.g., 'eng+vie')
        """
        lang_codes = []
        for lang in languages:
            if lang in SUPPORTED_LANGUAGES:
                lang_codes.append(SUPPORTED_LANGUAGES[lang])
            else:
                # Assume it's already a Tesseract code
                lang_codes.append(lang)

        return "+".join(lang_codes) if lang_codes else DEFAULT_OCR_LANG

    def perform_ocr(
        self,
        image: Union[str, Path, Image.Image],
        languages: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Perform OCR on an image.

        Args:
            image: Image to process (path, Path object, or PIL Image)
            languages: List of languages for OCR (overrides instance languages)
            **kwargs: Additional pytesseract parameters

        Returns:
            Extracted text from the image

        Raises:
            RuntimeError: If Tesseract is not configured
            ValueError: If image is invalid
        """
        if not self.is_configured:
            raise RuntimeError(
                "Tesseract is not configured. OCR functionality unavailable."
            )

        # Use provided languages or instance languages
        ocr_languages = languages or self.languages
        lang_string = self.get_language_code(ocr_languages)

        # Validate image
        if isinstance(image, (str, Path)):
            if not os.path.exists(image):
                raise ValueError(f"Image file not found: {image}")
            image_path = str(image)
        elif isinstance(image, Image.Image):
            # Save PIL image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                image.save(temp_file.name)
                image_path = temp_file.name
                temp_file_created = True
        else:
            raise ValueError("Image must be a file path or PIL Image object")

        try:
            # Set default OCR parameters
            ocr_config = {
                "lang": lang_string,
                "config": "--psm 6 --oem 3",  # Page segmentation mode and OCR engine mode
                **kwargs,
            }

            # Perform OCR
            logger.debug(f"Performing OCR with languages: {lang_string}")
            start_time = time.time()

            text = pytesseract.image_to_string(image_path, **ocr_config)

            processing_time = time.time() - start_time
            logger.debug(
                f"OCR completed in {processing_time:.2f} seconds, extracted {len(text)} characters"
            )

            return text.strip()

        finally:
            # Clean up temporary file if created
            if 'temp_file_created' in locals():
                try:
                    os.unlink(image_path)
                except Exception:
                    pass

    def get_ocr_info(self) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Get information about the OCR configuration.

        Returns:
            Dictionary containing OCR configuration information
        """
        return {
            "configured": self.is_configured,
            "version": self.tesseract_version,
            "executable": (
                pytesseract.pytesseract.tesseract_cmd if self.is_configured else None
            ),
            "supported_languages": sorted(list(self.supported_languages)),
            "current_languages": self.languages,
            "tessdata_prefix": os.environ.get("TESSDATA_PREFIX"),
        }

    def test_ocr(
        self, test_image: Optional[Union[str, Path, Image.Image]] = None
    ) -> bool:
        """
        Test OCR functionality with a sample image.

        Args:
            test_image: Optional custom test image

        Returns:
            True if OCR test was successful, False otherwise
        """
        if not self.is_configured:
            return False

        try:
            # Create a simple test image if none provided
            if test_image is None:
                test_image = self._create_test_image()

            # Perform OCR test
            text = self.perform_ocr(test_image, languages=["en"])

            # Check if we got any reasonable output
            success = len(text) > 0

            if success:
                logger.info("OCR test passed successfully")
            else:
                logger.warning("OCR test returned no text")

            return success

        except Exception as e:
            logger.error(f"OCR test failed: {e}")
            return False

    def _create_test_image(self) -> Image.Image:
        """
        Create a simple test image for OCR testing.

        Returns:
            PIL Image with test text
        """
        try:
            # Create a simple image with text
            img = Image.new('RGB', (400, 100), color='white')
            draw = ImageDraw.Draw(img)

            # Try to use a simple font
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except Exception:
                font = ImageFont.load_default()

            # Draw test text
            draw.text((10, 30), "TEST OCR 123", fill='black', font=font)

            return img

        except ImportError:
            # If PIL drawing components are not available, create a simple image
            logger.warning("Limited PIL features available, creating simple test image")
            return Image.new('RGB', (100, 100), color='white')


# Global OCR instance
_tesseract_instance: Optional[TesseractOCR] = None


def get_tesseract_ocr(languages: Optional[List[str]] = None) -> TesseractOCR:
    """
    Get or create a Tesseract OCR instance.

    Args:
        languages: List of languages for OCR

    Returns:
        TesseractOCR instance
    """
    global _tesseract_instance

    if _tesseract_instance is None:
        _tesseract_instance = TesseractOCR(languages=languages)

    return _tesseract_instance


def is_ocr_available() -> bool:
    """
    Check if OCR functionality is available.

    Returns:
        True if Tesseract is configured and available
    """
    try:
        ocr = get_tesseract_ocr()
        return ocr.is_configured
    except Exception:
        return False
