"""Anime / illustration image generation service.

Supports SDXL-Anime, Anything-v5, and Niji-style endpoints.
"""

from __future__ import annotations

import time

import httpx

from config import settings
from services.sdxl import BaseImageModel
from state import GeneratedImage


class AnimeModel(BaseImageModel):
    """Anime-style image generator (SDXL-Anime / Anything / Niji)."""

    model_name = "sdxl-anime"

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
        cfg = settings.image
        if not cfg.api_key:
            return self._placeholder_images(prompt, num_images, width, height, seed)

        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            results: list[GeneratedImage] = []
            for i in range(num_images):
                resp = await client.post(
                    cfg.anime_endpoint,
                    headers={"Authorization": f"Token {cfg.api_key}"},
                    json={
                        "input": {
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "num_outputs": 1,
                            "seed": (seed or int(time.time())) + i,
                        }
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                output = data.get("output", [])
                url = output[0] if isinstance(output, list) and output else str(output)
                results.append(
                    GeneratedImage(
                        url=url,
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    )
                )
            return results
