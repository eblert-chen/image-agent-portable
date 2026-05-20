"""Doubao-Seedream 5.0 model service (Volcengine Ark).

Endpoint: POST https://ark.cn-beijing.volces.com/api/v3/images/generations
Auth: Bearer <API_KEY>
"""

from __future__ import annotations

import logging
import os

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

DOUBAO_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
DOUBAO_MODEL = "doubao-seedream-5-0-260128"


class DoubaoSeedreamModel(BaseImageModel):
    """Doubao-Seedream 5.0 Lite — Volcengine Ark image generation."""

    model_name = "doubao-seedream"

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
        api_key = os.getenv("DOUBAO_API_KEY", "")
        if not api_key:
            logger.warning("DOUBAO_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        # Map resolution to size enum: 2K (≤2048) or 4K (>2048)
        max_dim = max(width, height)
        size = "4K" if max_dim > 2048 else "2K"
        response_format = kwargs.get("response_format", "url")
        watermark = kwargs.get("watermark", True)

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(num_images):
                try:
                    body: dict = {
                        "model": DOUBAO_MODEL,
                        "prompt": prompt,
                        "size": size,
                        "response_format": response_format,
                        "sequential_image_generation": "disabled",
                        "stream": False,
                        "watermark": watermark,
                    }
                    if seed is not None:
                        body["seed"] = seed + i

                    resp = await client.post(
                        DOUBAO_ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                    if resp.status_code != 200:
                        logger.error(
                            "Doubao API error %d: %s",
                            resp.status_code,
                            resp.text[:500],
                        )
                    resp.raise_for_status()
                    data = resp.json()

                    # Response can be {"data": [{"url": "..."}]} or {"data": [{"b64_json": "..."}]}
                    items = data.get("data", [])
                    if items:
                        img_data = items[0]
                        url = img_data.get("url", "")
                        if not url and "b64_json" in img_data:
                            url = f"data:image/jpeg;base64,{img_data['b64_json']}"
                        actual_seed = img_data.get("seed", seed or 0)
                    else:
                        url = ""
                        actual_seed = seed or 0

                    results.append(GeneratedImage(
                        url=url or f"https://placehold.co/{width}x{height}?text=doubao+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=actual_seed,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error(
                        "Doubao generation %d failed: %s (%s)", i, exc, type(exc).__name__
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=doubao+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results
