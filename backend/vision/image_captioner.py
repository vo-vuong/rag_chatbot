"""
Image captioning service using GPT-4o Mini Vision API.

Provides caption generation for images with parallel processing,
retry logic, and cost tracking capabilities.
"""

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI

from .caption_cache import CaptionCache
from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """Generate captions for images using GPT-4o Mini Vision API."""

    # Pricing (as of 2024-12, input/output per 1K tokens)
    PRICING_INPUT = 0.00015
    PRICING_OUTPUT = 0.0006

    BRIEF_CAPTION_PROMPT = """Describe this image briefly in 1-2 sentences.
Focus on:
- Main content (chart, diagram, photo, etc.)
- Key data or information shown
- Chart type if applicable (bar, line, pie, etc.)
Be concise and factual."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 100,
        temperature: float = 0.3,
        detail_mode: str = "low",
        enable_cache: bool = True,
        daily_budget: Optional[float] = None
    ):
        """
        Initialize image captioner.

        Args:
            api_key: OpenAI API key
            model: Vision model name (default: gpt-4o-mini)
            max_tokens: Max tokens in caption (100 = ~50 words)
            temperature: Sampling temperature (0.3 = more consistent)
            detail_mode: Image detail level ("low" or "high")
            enable_cache: Enable caption caching (default: True)
            daily_budget: Daily cost budget in USD (None = unlimited)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.detail_mode = detail_mode
        self.cache = CaptionCache() if enable_cache else None
        self.cost_tracker = CostTracker(daily_budget=daily_budget)

        logger.info(
            f"ImageCaptioner initialized: model={model}, "
            f"detail={detail_mode}, max_tokens={max_tokens}, cache={enable_cache}, "
            f"daily_budget=${daily_budget}"
        )

    def caption_image(
        self,
        image_path: str,
        max_tokens: Optional[int] = None,
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Generate caption for an image with retry logic.

        Args:
            image_path: Path to image file
            max_tokens: Override default max_tokens
            custom_prompt: Override default prompt

        Returns:
            Tuple of (caption_text, cost_usd)

        Raises:
            ImageCaptioningError: If all retry attempts fail
        """
        try:
            # Check cache first
            if self.cache:
                cached_caption = self.cache.get(image_path)
                if cached_caption:
                    return cached_caption, 0.0  # No cost for cached

            # Validate image exists
            if not Path(image_path).exists():
                raise ImageCaptioningError(
                    f"Image file not found: {image_path}",
                    image_path=image_path
                )

            # Encode image
            base64_image = self._encode_image_to_base64(image_path)

            # Call API with retry
            caption, cost = self._retry_with_backoff(
                image_path=image_path,
                base64_image=base64_image,
                prompt=custom_prompt or self.BRIEF_CAPTION_PROMPT,
                max_tokens=max_tokens or self.max_tokens
            )

            # Cache the result
            if self.cache:
                self.cache.set(image_path, caption, cost)

            # Track cost
            if cost > 0:  # Don't track cached results
                self.cost_tracker.add_cost(cost)

            logger.info(
                f"Captioned {Path(image_path).name}: "
                f"'{caption[:50]}...' (cost: ${cost:.6f})"
            )

            return caption, cost

        except ImageCaptioningError:
            raise
        except Exception as e:
            raise ImageCaptioningError(
                f"Unexpected error captioning {image_path}: {str(e)}",
                image_path=image_path
            )

    def caption_images_batch(
        self,
        image_paths: List[str],
        max_workers: int = 10,
        failure_mode: str = "graceful"
    ) -> List[Tuple[str, str, float]]:
        """
        Caption multiple images in parallel.

        Args:
            image_paths: List of image file paths
            max_workers: Max concurrent API calls (default: 10)
            failure_mode: "strict" | "graceful" | "skip"
                - strict: Raise on any failure
                - graceful: Use fallback caption on failure
                - skip: Skip failed images

        Returns:
            List of (image_path, caption, cost) tuples

        Raises:
            ImageCaptioningError: If failure_mode="strict" and any caption fails
        """
        results = []
        total_cost = 0.0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self.caption_image, path): path
                for path in image_paths
            }

            # Collect results
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    caption, cost = future.result()
                    results.append((path, caption, cost))
                    total_cost += cost

                    # Check budget after each image
                    if self.cost_tracker.is_over_budget():
                        logger.error("Daily budget exceeded, stopping caption generation")
                        raise ImageCaptioningError(
                            "Daily Vision API budget exceeded",
                            error_code="BUDGET_EXCEEDED"
                        )
                except ImageCaptioningError as e:
                    # Re-raise budget exceeded errors
                    if hasattr(e, 'error_code') and e.error_code == "BUDGET_EXCEEDED":
                        raise

                    failed_count += 1
                    error_msg = f"Failed to caption {path}: {str(e)}"

                    if failure_mode == "strict":
                        logger.error(f"STRICT MODE: {error_msg}")
                        raise ImageCaptioningError(
                            f"Image captioning failed for {path}. "
                            f"Upload aborted (strict mode enabled).",
                            image_path=path
                        )
                    elif failure_mode == "graceful":
                        logger.warning(f"GRACEFUL MODE: {error_msg}")
                        fallback_caption = "Image (caption unavailable)"
                        results.append((path, fallback_caption, 0.0))
                    elif failure_mode == "skip":
                        logger.info(f"SKIP MODE: Skipping {path}")
                        continue
                    else:
                        raise ValueError(f"Invalid failure_mode: {failure_mode}")
                except Exception as e:
                    failed_count += 1
                    error_msg = f"Failed to caption {path}: {str(e)}"

                    if failure_mode == "strict":
                        logger.error(f"STRICT MODE: {error_msg}")
                        raise ImageCaptioningError(
                            f"Image captioning failed for {path}. "
                            f"Upload aborted (strict mode enabled).",
                            image_path=path
                        )
                    elif failure_mode == "graceful":
                        logger.warning(f"GRACEFUL MODE: {error_msg}")
                        fallback_caption = "Image (caption unavailable)"
                        results.append((path, fallback_caption, 0.0))
                    elif failure_mode == "skip":
                        logger.info(f"SKIP MODE: Skipping {path}")
                        continue
                    else:
                        raise ValueError(f"Invalid failure_mode: {failure_mode}")

        logger.info(
            f"Batch captioning complete: {len(results)} images, "
            f"${total_cost:.4f} total cost, {failed_count} failed"
        )

        return results

    def get_cost_stats(self) -> dict:
        """Get cost tracking statistics."""
        return self.cost_tracker.get_stats()

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Read and encode image as base64."""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise ImageCaptioningError(
                f"Failed to encode image {image_path}: {str(e)}",
                image_path=image_path
            )

    def _call_vision_api(
        self,
        base64_image: str,
        prompt: str,
        max_tokens: int
    ) -> Tuple[str, int, int]:
        """
        Call Vision API.

        Returns:
            Tuple of (caption, input_tokens, output_tokens)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": self.detail_mode
                            }
                        }
                    ]
                }
            ],
            max_tokens=max_tokens,
            temperature=self.temperature
        )

        caption = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return caption, input_tokens, output_tokens

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        cost = (
            (input_tokens / 1000 * self.PRICING_INPUT) +
            (output_tokens / 1000 * self.PRICING_OUTPUT)
        )
        return cost

    def _retry_with_backoff(
        self,
        image_path: str,
        base64_image: str,
        prompt: str,
        max_tokens: int,
        max_attempts: int = 3,
        backoff_delays: List[float] = None
    ) -> Tuple[str, float]:
        """
        Retry API call with exponential backoff.

        Args:
            image_path: Path for error messages
            base64_image: Encoded image data
            prompt: Caption prompt
            max_tokens: Max tokens in response
            max_attempts: Max retry attempts (default: 3)
            backoff_delays: Delay between retries in seconds (default: [1, 2, 4])

        Returns:
            Tuple of (caption, cost)

        Raises:
            ImageCaptioningError: If all attempts fail
        """
        if backoff_delays is None:
            backoff_delays = [1.0, 2.0, 4.0]

        last_error = None

        for attempt in range(max_attempts):
            try:
                caption, input_tokens, output_tokens = self._call_vision_api(
                    base64_image, prompt, max_tokens
                )
                cost = self._calculate_cost(input_tokens, output_tokens)
                return caption, cost

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Categorize error
                error_str = str(e).lower()
                if "rate_limit" in error_str or "rate limit" in error_str:
                    error_code = "RATE_LIMIT"
                elif "timeout" in error_str:
                    error_code = "TIMEOUT"
                elif "invalid" in error_str:
                    error_code = "INVALID_IMAGE"
                else:
                    error_code = "UNKNOWN"

                if attempt < max_attempts - 1:
                    delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
                    logger.warning(
                        f"Vision API error (attempt {attempt + 1}/{max_attempts}): {error_type} - {e}",
                        extra={
                            "error_type": error_type,
                            "error_code": error_code,
                            "image_path": image_path,
                            "retry_delay": delay
                        }
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_attempts} attempts failed for {image_path}: {error_type} - {e}",
                        extra={
                            "error_type": error_type,
                            "error_code": error_code,
                            "image_path": image_path
                        }
                    )

        raise ImageCaptioningError(
            f"Failed to caption {image_path} after {max_attempts} attempts: {last_error}",
            image_path=image_path,
            error_code=error_code if 'error_code' in locals() else "UNKNOWN",
            retry_count=max_attempts
        )


class ImageCaptioningError(Exception):
    """Raised when image captioning fails."""

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        error_code: Optional[str] = None,
        retry_count: int = 0
    ):
        super().__init__(message)
        self.image_path = image_path
        self.error_code = error_code
        self.retry_count = retry_count

    def to_dict(self) -> dict:
        """Convert to dictionary for structured logging."""
        return {
            "message": str(self),
            "image_path": self.image_path,
            "error_code": self.error_code,
            "retry_count": self.retry_count
        }
