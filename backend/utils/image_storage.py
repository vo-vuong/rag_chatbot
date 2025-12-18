"""
Enhanced Image Storage Utility for PDF Processing

This utility provides comprehensive image storage capabilities including:
- Multi-format support (PNG, JPEG, WebP)
- Image optimization and compression
- Structured storage with metadata
- Error handling and logging
- Performance monitoring

Created: 2025-12-17
Phase: 01 - Enhanced Image Storage Utility
"""

import os
import io
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
from dataclasses import dataclass, asdict
from datetime import datetime
from PIL import Image, ImageOps
import pi_heif
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for stored images."""
    filename: str
    original_format: str
    stored_format: str
    width: int
    height: int
    size_bytes: int
    optimized_size_bytes: int
    compression_ratio: float
    storage_path: str
    hash_md5: str
    page_number: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    extraction_timestamp: str = ""
    mime_type: str = ""
    color_mode: str = ""
    has_transparency: bool = False


@dataclass
class StorageStats:
    """Statistics for image storage operations."""
    total_images: int = 0
    total_original_size: int = 0
    total_optimized_size: int = 0
    total_space_saved: int = 0
    average_compression: float = 0.0
    processing_time_seconds: float = 0.0
    failed_count: int = 0
    formats_processed: Dict[str, int] = None

    def __post_init__(self):
        if self.formats_processed is None:
            self.formats_processed = {}


class ImageStorageError(Exception):
    """Custom exception for image storage errors."""
    pass


class ImageValidationError(ImageStorageError):
    """Exception raised for image validation failures."""
    pass


class ImageOptimizationError(ImageStorageError):
    """Exception raised for image optimization failures."""
    pass


class ImageStorageUtility:
    """
    Enhanced utility for storing and optimizing images extracted from PDFs.

    Features:
    - Multi-format support (PNG, JPEG, WebP, HEIC)
    - Automatic optimization and compression
    - Duplicate detection and prevention
    - Structured directory storage
    - Comprehensive metadata tracking
    - Performance monitoring
    - Error handling and recovery
    """

    # Supported formats and their quality settings
    SUPPORTED_FORMATS = {
        'PNG': {'quality': None, 'optimize': True, 'compress_level': 6},
        'JPEG': {'quality': 85, 'optimize': True, 'progressive': True},
        'WEBP': {'quality': 80, 'optimize': True, 'method': 6},
        'HEIC': {'quality': 85, 'optimize': True},
        'HEIF': {'quality': 85, 'optimize': True},
    }

    # Default maximum dimensions for resizing
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048

    # File size limits (in bytes)
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

    def __init__(
        self,
        base_storage_path: str,
        max_workers: int = 4,
        enable_optimization: bool = True,
        create_subdirs: bool = True
    ):
        """
        Initialize the image storage utility.

        Args:
            base_storage_path: Base directory for image storage
            max_workers: Number of worker threads for parallel processing
            enable_optimization: Whether to enable image optimization
            create_subdirs: Whether to create subdirectories for organization
        """
        self.base_path = Path(base_storage_path)
        self.max_workers = max_workers
        self.enable_optimization = enable_optimization
        self.create_subdirs = create_subdirs

        # Register HEIF opener
        pi_heif.register_heif_opener()

        # Create base directories
        self._create_directory_structure()

        # Initialize stats tracking
        self.stats = StorageStats()
        self._start_time = None

        # Configure logger
        self._setup_logging()

        logger.info(f"ImageStorageUtility initialized with path: {self.base_path}")

    def _create_directory_structure(self) -> None:
        """Create the necessary directory structure for image storage."""
        directories = [
            self.base_path,
            self.base_path / "images",
            self.base_path / "metadata",
            self.base_path / "temp",
        ]

        if self.create_subdirs:
            # Create date-based subdirectories
            today = datetime.now().strftime("%Y-%m-%d")
            directories.extend([
                self.base_path / "images" / today,
                self.base_path / "metadata" / today,
            ])

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.base_path / "image_storage.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def _generate_storage_path(
        self,
        image_hash: str,
        format_type: str,
        create_date_dir: bool = True
    ) -> Path:
        """
        Generate a storage path for an image based on its hash.

        Args:
            image_hash: MD5 hash of the image content
            format_type: Image format (PNG, JPEG, etc.)
            create_date_dir: Whether to create date-based subdirectories

        Returns:
            Path object for the storage location
        """
        # Use first 2 characters for subdirectory to avoid too many files in one dir
        subdir = image_hash[:2]

        if create_date_dir and self.create_subdirs:
            today = datetime.now().strftime("%Y-%m-%d")
            return self.base_path / "images" / today / subdir / f"{image_hash}.{format_type.lower()}"
        else:
            return self.base_path / "images" / subdir / f"{image_hash}.{format_type.lower()}"

    def _generate_metadata_path(self, image_hash: str, create_date_dir: bool = True) -> Path:
        """
        Generate a metadata storage path for an image.

        Args:
            image_hash: MD5 hash of the image content
            create_date_dir: Whether to create date-based subdirectories

        Returns:
            Path object for the metadata location
        """
        subdir = image_hash[:2]

        if create_date_dir and self.create_subdirs:
            today = datetime.now().strftime("%Y-%m-%d")
            return self.base_path / "metadata" / today / subdir / f"{image_hash}.json"
        else:
            return self.base_path / "metadata" / subdir / f"{image_hash}.json"

    def _calculate_image_hash(self, image_data: bytes) -> str:
        """
        Calculate MD5 hash of image data.

        Args:
            image_data: Raw image data

        Returns:
            MD5 hash as hexadecimal string
        """
        return hashlib.md5(image_data).hexdigest()

    def _validate_image(self, image: Image.Image) -> None:
        """
        Validate an image object.

        Args:
            image: PIL Image object to validate

        Raises:
            ImageValidationError: If image is invalid
        """
        if image is None:
            raise ImageValidationError("Image is None")

        if image.size[0] <= 0 or image.size[1] <= 0:
            raise ImageValidationError(f"Invalid image dimensions: {image.size}")

        # Check for modes that might cause issues
        if image.mode not in ['RGB', 'RGBA', 'L', 'P', 'CMYK']:
            logger.warning(f"Unusual image mode: {image.mode}")

    def _optimize_image(
        self,
        image: Image.Image,
        target_format: str = 'PNG'
    ) -> Tuple[Image.Image, Dict]:
        """
        Optimize an image for storage.

        Args:
            image: PIL Image object to optimize
            target_format: Target format for optimization

        Returns:
            Tuple of (optimized_image, optimization_info)

        Raises:
            ImageOptimizationError: If optimization fails
        """
        try:
            optimization_info = {
                'original_size': image.size,
                'original_mode': image.mode,
                'operations': []
            }

            # Convert to RGB if necessary (for JPEG)
            if target_format.upper() == 'JPEG' and image.mode in ['RGBA', 'LA', 'P']:
                # Create white background for transparent images
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                if image.mode == 'LA':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
                optimization_info['operations'].append('converted_to_rgb_with_bg')

            # Resize if too large
            if image.size[0] > self.MAX_WIDTH or image.size[1] > self.MAX_HEIGHT:
                image.thumbnail((self.MAX_WIDTH, self.MAX_HEIGHT), Image.Resampling.LANCZOS)
                optimization_info['operations'].append('resized')
                optimization_info['new_size'] = image.size

            # Auto-orient based on EXIF
            image = ImageOps.exif_transpose(image)
            optimization_info['operations'].append('exif_transpose')

            # Strip metadata for privacy and size reduction
            if 'exif' in image.info:
                del image.info['exif']
            if 'icc_profile' in image.info:
                del image.info['icc_profile']

            optimization_info['final_size'] = image.size
            optimization_info['final_mode'] = image.mode

            return image, optimization_info

        except Exception as e:
            raise ImageOptimizationError(f"Failed to optimize image: {str(e)}")

    def _save_image(
        self,
        image: Image.Image,
        storage_path: Path,
        format_type: str,
        optimization_info: Optional[Dict] = None
    ) -> int:
        """
        Save an image to storage with optimization.

        Args:
            image: PIL Image object to save
            storage_path: Path where to save the image
            format_type: Image format
            optimization_info: Dictionary with optimization details

        Returns:
            Size of the saved file in bytes

        Raises:
            ImageStorageError: If saving fails
        """
        try:
            # Create parent directories if they don't exist
            storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Get format-specific save parameters
            save_params = self.SUPPORTED_FORMATS.get(format_type.upper(), {})

            # Save with optimization
            with io.BytesIO() as output:
                if format_type.upper() == 'PNG':
                    image.save(output, format='PNG', **{k: v for k, v in save_params.items() if v is not None})
                elif format_type.upper() == 'JPEG':
                    image.save(output, format='JPEG', **{k: v for k, v in save_params.items() if v is not None})
                elif format_type.upper() == 'WEBP':
                    image.save(output, format='WEBP', **{k: v for k, v in save_params.items() if v is not None})
                else:
                    # Fallback to original format
                    image.save(output, format=format_type)

                # Write to file
                output.seek(0)
                with open(storage_path, 'wb') as f:
                    f.write(output.getvalue())

                return len(output.getvalue())

        except Exception as e:
            raise ImageStorageError(f"Failed to save image to {storage_path}: {str(e)}")

    def _save_metadata(self, metadata: ImageMetadata, metadata_path: Path) -> None:
        """
        Save image metadata to a JSON file.

        Args:
            metadata: ImageMetadata object to save
            metadata_path: Path where to save the metadata
        """
        try:
            # Create parent directories if they don't exist
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert metadata to dictionary and save as JSON
            metadata_dict = asdict(metadata)

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save metadata to {metadata_path}: {str(e)}")
            # Don't raise - metadata saving failure shouldn't stop the main process

    def _create_metadata(
        self,
        image: Image.Image,
        image_hash: str,
        storage_path: Path,
        original_size: int,
        optimized_size: int,
        page_number: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        original_filename: Optional[str] = None
    ) -> ImageMetadata:
        """
        Create metadata for a stored image.

        Args:
            image: PIL Image object
            image_hash: MD5 hash of the image
            storage_path: Path where the image is stored
            original_size: Original file size in bytes
            optimized_size: Optimized file size in bytes
            page_number: PDF page number (if applicable)
            bbox: Bounding box coordinates (if applicable)
            original_filename: Original filename

        Returns:
            ImageMetadata object
        """
        compression_ratio = (original_size - optimized_size) / original_size if original_size > 0 else 0

        # Determine MIME type
        format_to_mime = {
            'PNG': 'image/png',
            'JPEG': 'image/jpeg',
            'JPG': 'image/jpeg',
            'WEBP': 'image/webp',
            'HEIC': 'image/heic',
            'HEIF': 'image/heif'
        }

        return ImageMetadata(
            filename=original_filename or f"{image_hash}.png",
            original_format=image.format or 'UNKNOWN',
            stored_format=storage_path.suffix.upper().replace('.', ''),
            width=image.width,
            height=image.height,
            size_bytes=original_size,
            optimized_size_bytes=optimized_size,
            compression_ratio=compression_ratio,
            storage_path=str(storage_path),
            hash_md5=image_hash,
            page_number=page_number,
            bbox=bbox,
            extraction_timestamp=datetime.now().isoformat(),
            mime_type=format_to_mime.get(image.format or '', 'application/octet-stream'),
            color_mode=image.mode,
            has_transparency=image.mode in ['RGBA', 'LA'] or 'transparency' in image.info
        )

    @lru_cache(maxsize=128)
    def _check_duplicate(self, image_hash: str) -> Optional[Path]:
        """
        Check if an image with the given hash already exists.

        Args:
            image_hash: MD5 hash to check

        Returns:
            Path to existing image if found, None otherwise
        """
        # Get today's date for consistent path checking
        today = datetime.now().strftime("%Y-%m-%d")

        # Check both with and without date subdirectories
        for format_type in ['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif']:
            # Check with date subdirectory first
            path_with_date = self.base_path / "images" / today / image_hash[:2] / f"{image_hash}.{format_type}"
            if path_with_date.exists():
                return path_with_date

            # Check without date subdirectory
            path_without_date = self.base_path / "images" / image_hash[:2] / f"{image_hash}.{format_type}"
            if path_without_date.exists():
                return path_without_date

        return None

    def store_image(
        self,
        image_data: Union[bytes, BinaryIO, Image.Image],
        original_filename: Optional[str] = None,
        page_number: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        target_format: str = 'PNG',
        force_optimization: bool = None
    ) -> Tuple[Path, ImageMetadata]:
        """
        Store an image with optimization and metadata.

        Args:
            image_data: Image data as bytes, file-like object, or PIL Image
            original_filename: Original filename
            page_number: PDF page number (if applicable)
            bbox: Bounding box coordinates (if applicable)
            target_format: Target format for storage
            force_optimization: Override default optimization setting

        Returns:
            Tuple of (storage_path, metadata)

        Raises:
            ImageStorageError: If storage fails
        """
        try:
            # Track processing time
            if self._start_time is None:
                self._start_time = datetime.now()

            # Validate input
            if image_data is None:
                raise ImageValidationError("Image data cannot be None")

            # Convert input to PIL Image
            if isinstance(image_data, Image.Image):
                image = image_data
                original_bytes = b''
            else:
                # Handle bytes or file-like object
                if isinstance(image_data, bytes):
                    original_bytes = image_data
                else:
                    # Read from file-like object
                    image_data.seek(0)
                    original_bytes = image_data.read()

                # Create PIL Image from bytes
                image = Image.open(io.BytesIO(original_bytes))

            # Validate the image
            self._validate_image(image)

            # Calculate hash - use a consistent method for PIL Images
            if isinstance(image_data, Image.Image):
                # Convert PIL Image to bytes for hash calculation
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                hash_bytes = buffer.getvalue()
            else:
                hash_bytes = original_bytes

            image_hash = self._calculate_image_hash(hash_bytes)

            # Check for duplicates
            existing_path = self._check_duplicate(image_hash)
            if existing_path:
                logger.info(f"Found duplicate image: {image_hash} at {existing_path}")
                self.stats.total_images += 1
                # Load existing metadata and return with new context if provided
                existing_metadata = self.get_image_metadata(image_hash)
                if existing_metadata:
                    # Update context info if provided
                    if page_number is not None:
                        existing_metadata.page_number = page_number
                    if bbox is not None:
                        existing_metadata.bbox = bbox
                    if original_filename:
                        existing_metadata.filename = original_filename
                return existing_path, existing_metadata

            # Determine if optimization should be applied
            should_optimize = force_optimization if force_optimization is not None else self.enable_optimization

            # Optimize image if requested
            if should_optimize:
                optimized_image, optimization_info = self._optimize_image(image, target_format)
            else:
                optimized_image = image
                optimization_info = {'operations': ['no_optimization']}

            # Generate storage path
            storage_path = self._generate_storage_path(image_hash, target_format)

            # Save image
            optimized_size = self._save_image(optimized_image, storage_path, target_format, optimization_info)

            # Calculate original size
            if isinstance(image_data, Image.Image):
                # For PIL Images, estimate original size from dimensions
                buffer = io.BytesIO()
                image.save(buffer, format='PNG', optimize=False)
                original_size = len(buffer.getvalue())
            else:
                original_size = len(original_bytes) if original_bytes else optimized_size

            # Create metadata
            metadata = self._create_metadata(
                optimized_image,
                image_hash,
                storage_path,
                int(original_size),
                optimized_size,
                page_number,
                bbox,
                original_filename
            )

            # Save metadata
            metadata_path = self._generate_metadata_path(image_hash)
            self._save_metadata(metadata, metadata_path)

            # Update statistics
            self._update_stats(metadata, target_format)

            logger.info(f"Stored image: {storage_path} (original: {original_size:,} bytes, "
                       f"optimized: {optimized_size:,} bytes, compression: {metadata.compression_ratio:.1%})")

            return storage_path, metadata

        except Exception as e:
            self.stats.failed_count += 1
            logger.error(f"Failed to store image: {str(e)}")
            raise ImageStorageError(f"Failed to store image: {str(e)}")

    def store_images_batch(
        self,
        images_list: List[Tuple[Union[bytes, BinaryIO, Image.Image], dict]],
        target_format: str = 'PNG'
    ) -> List[Tuple[Path, ImageMetadata]]:
        """
        Store multiple images in parallel.

        Args:
            images_list: List of tuples (image_data, metadata_dict)
            target_format: Target format for storage

        Returns:
            List of tuples (storage_path, metadata)
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all storage tasks
            future_to_index = {
                executor.submit(
                    self.store_image,
                    image_data,
                    target_format=target_format,
                    **metadata_dict
                ): idx
                for idx, (image_data, metadata_dict) in enumerate(images_list)
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to store image in batch: {str(e)}")
                    results.append((None, None))

        return results

    def _update_stats(self, metadata: ImageMetadata, format_type: str) -> None:
        """
        Update storage statistics.

        Args:
            metadata: Image metadata
            format_type: Image format type
        """
        self.stats.total_images += 1
        self.stats.total_original_size += metadata.size_bytes
        self.stats.total_optimized_size += metadata.optimized_size_bytes
        self.stats.total_space_saved += (metadata.size_bytes - metadata.optimized_size_bytes)

        # Update format statistics
        format_key = format_type.upper()
        self.stats.formats_processed[format_key] = self.stats.formats_processed.get(format_key, 0) + 1

    def get_storage_stats(self) -> StorageStats:
        """
        Get current storage statistics.

        Returns:
            StorageStats object with current statistics
        """
        if self._start_time:
            self.stats.processing_time_seconds = (datetime.now() - self._start_time).total_seconds()

        if self.stats.total_images > 0:
            self.stats.average_compression = (
                self.stats.total_space_saved / self.stats.total_original_size
            )

        return self.stats

    def get_image_metadata(self, image_hash: str) -> Optional[ImageMetadata]:
        """
        Retrieve metadata for a stored image.

        Args:
            image_hash: MD5 hash of the image

        Returns:
            ImageMetadata object if found, None otherwise
        """
        # Try with date subdirectory first
        metadata_path = self._generate_metadata_path(image_hash, True)
        if not metadata_path.exists():
            # Try without date subdirectory
            metadata_path = self._generate_metadata_path(image_hash, False)

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            return ImageMetadata(**metadata_dict)
        except Exception as e:
            logger.error(f"Failed to load metadata for {image_hash}: {str(e)}")
            return None

    def cleanup_temp_files(self) -> None:
        """Clean up temporary files."""
        temp_dir = self.base_path / "temp"
        if temp_dir.exists():
            for file_path in temp_dir.glob("*"):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {file_path}: {str(e)}")

    def validate_storage_integrity(self) -> Dict[str, List[str]]:
        """
        Validate the integrity of stored images and metadata.

        Returns:
            Dictionary with 'missing_images', 'missing_metadata', and 'corrupted' lists
        """
        issues = {
            'missing_images': [],
            'missing_metadata': [],
            'corrupted': []
        }

        # Scan all image files
        images_dir = self.base_path / "images"
        if images_dir.exists():
            for image_path in images_dir.rglob("*"):
                if image_path.is_file() and image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif']:
                    # Get hash from filename
                    image_hash = image_path.stem

                    # Check if metadata exists
                    metadata = self.get_image_metadata(image_hash)
                    if metadata is None:
                        issues['missing_metadata'].append(str(image_path))
                    else:
                        # Verify file size matches metadata
                        try:
                            actual_size = image_path.stat().st_size
                            if actual_size != metadata.optimized_size_bytes:
                                issues['corrupted'].append(str(image_path))
                        except Exception:
                            issues['corrupted'].append(str(image_path))

        # Scan all metadata files
        metadata_dir = self.base_path / "metadata"
        if metadata_dir.exists():
            for metadata_path in metadata_dir.rglob("*.json"):
                if metadata_path.is_file():
                    # Get hash from filename
                    image_hash = metadata_path.stem

                    # Check if image exists
                    image_path = None
                    for format_ext in ['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif']:
                        potential_path = self._generate_storage_path(image_hash, format_ext, False)
                        if potential_path.exists():
                            image_path = potential_path
                            break

                    if image_path is None:
                        issues['missing_images'].append(str(metadata_path))

        return issues

    def export_storage_report(self, output_path: Optional[str] = None) -> str:
        """
        Export a comprehensive storage report.

        Args:
            output_path: Path to save the report (optional)

        Returns:
            Report content as string
        """
        stats = self.get_storage_stats()
        integrity_issues = self.validate_storage_integrity()

        report_lines = [
            "Image Storage Report",
            "=" * 50,
            f"Generated: {datetime.now().isoformat()}",
            f"Storage Path: {self.base_path}",
            "",
            "Storage Statistics:",
            f"- Total Images: {stats.total_images:,}",
            f"- Total Original Size: {stats.total_original_size / (1024*1024):.2f} MB",
            f"- Total Optimized Size: {stats.total_optimized_size / (1024*1024):.2f} MB",
            f"- Total Space Saved: {stats.total_space_saved / (1024*1024):.2f} MB",
            f"- Average Compression: {stats.average_compression:.1%}",
            f"- Processing Time: {stats.processing_time_seconds:.2f} seconds",
            f"- Failed Count: {stats.failed_count}",
            "",
            "Formats Processed:",
        ]

        for format_type, count in stats.formats_processed.items():
            report_lines.append(f"- {format_type}: {count:,}")

        report_lines.extend([
            "",
            "Integrity Issues:",
            f"- Missing Images: {len(integrity_issues['missing_images'])}",
            f"- Missing Metadata: {len(integrity_issues['missing_metadata'])}",
            f"- Corrupted Files: {len(integrity_issues['corrupted'])}",
        ])

        report_content = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Storage report saved to: {output_path}")

        return report_content


# Convenience function for quick image storage
def store_image_quick(
    image_data: Union[bytes, BinaryIO, Image.Image],
    storage_path: str,
    filename: Optional[str] = None,
    optimize: bool = True
) -> Tuple[str, ImageMetadata]:
    """
    Quick function to store an image with default settings.

    Args:
        image_data: Image data to store
        storage_path: Base storage path
        filename: Original filename
        optimize: Whether to optimize the image

    Returns:
        Tuple of (stored_file_path, metadata)
    """
    utility = ImageStorageUtility(storage_path, enable_optimization=optimize)
    return utility.store_image(image_data, original_filename=filename)