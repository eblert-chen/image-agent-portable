"""Stability AI — Stable Image Ultra model service.

Endpoint: POST https://api.stability.ai/v2beta/stable-image/generate/ultra
Auth: Bearer <STABILITY_API_KEY>
Content-Type: multipart/form-data
"""

from __future__ import annotations

import logging
import os

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

STABILITY_ENDPOINT = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

ASPECT_RATIOS = ["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
OUTPUT_FORMATS = ["png", "jpeg", "webp"]
STYLE_PRESETS = [
    "3d-model", "analog-film", "anime", "cinematic", "comic-book",
    "digital-art", "enhance", "fantasy-art", "isometric", "line-art",
    "low-poly", "modeling-compound", "neon-punk", "origami",
    "photographic", "pixel-art", "tile-texture",
]


class StabilityUltraModel(BaseImageModel):
    """Stable Image Ultra — highest quality text-to-image generation."""

    model_name = "stable-ultra"

    async def generate(
        self,
        prompt: str,
        *,
        num_images: int = 4,
        width: int = 1024,
        height: int = 1024,
        seed: int | None = None,
        **kwargs,
    ) -> list[GeneratedImage]:
        api_key = os.getenv("STABILITY_API_KEY", "")
        if not api_key:
            logger.warning("STABILITY_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        # Map width/height to closest supported aspect ratio
        aspect = self._resolve_aspect(width, height)
        output_format = kwargs.get("output_format", "png")
        negative_prompt = kwargs.get("negative_prompt", "")
        style_preset = kwargs.get("style_preset")

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(num_images):
                try:
                    form_data = {
                        "prompt": (None, prompt),
                        "output_format": (None, output_format),
                        "aspect_ratio": (None, aspect),
                    }
                    if negative_prompt:
                        form_data["negative_prompt"] = (None, negative_prompt)
                    if style_preset:
                        form_data["style_preset"] = (None, style_preset)
                    # Use random seed per image when no fixed seed
                    img_seed = (seed or 0) + i if seed else 0
                    if img_seed:
                        form_data["seed"] = (None, str(img_seed))

                    resp = await client.post(
                        STABILITY_ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Accept": "application/json",
                        },
                        files=form_data,
                    )
                    if resp.status_code != 200:
                        logger.error(
                            "Stability API error %d: %s",
                            resp.status_code,
                            resp.text[:500],
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    b64 = data.get("image", "")
                    actual_seed = data.get("seed", img_seed)
                    finish = data.get("finish_reason", "")

                    if finish == "CONTENT_FILTERED":
                        logger.warning("Stability image %d filtered by content policy", i)

                    url = f"data:image/{output_format};base64,{b64}" if b64 else ""
                    results.append(GeneratedImage(
                        url=url or f"https://placehold.co/{width}x{height}?text=stable-ultra+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=actual_seed,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error("Stability Ultra generation %d failed: %s (%s)", i, exc, type(exc).__name__)
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=stable-ultra+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results

    @staticmethod
    def _resolve_aspect(width: int, height: int) -> str:
        """Map pixel dimensions to the closest supported aspect ratio."""
        ratio = width / height
        candidates = {
            "1:1": 1.0,
            "4:5": 0.8,
            "5:4": 1.25,
            "2:3": 0.6667,
            "3:2": 1.5,
            "9:16": 0.5625,
            "16:9": 1.7778,
            "9:21": 0.4286,
            "21:9": 2.3333,
        }
        best = "1:1"
        best_diff = float("inf")
        for ar, ar_ratio in candidates.items():
            diff = abs(ratio - ar_ratio)
            if diff < best_diff:
                best_diff = diff
                best = ar
        return best
