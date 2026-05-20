"""Image-generation service layer for SDXL, Flux, and general models.

Every generator implements the same protocol:

    async generate(prompt: str, **kwargs) -> list[GeneratedImage]

The real implementations call external REST APIs.
Mock implementations (used when no API key) return placeholder images so the pipeline is always runnable.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod

import httpx

from config import settings
from state import GeneratedImage


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class BaseImageModel(ABC):
    """Every image generator implements this interface."""

    model_name: str = "base"

    @abstractmethod
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
        ...

    def _placeholder_images(
        self,
        prompt: str,
        num: int,
        width: int,
        height: int,
        seed: int | None,
    ) -> list[GeneratedImage]:
        """Generate placeholder image records for offline / mock mode."""
        return [
            GeneratedImage(
                url=f"https://placehold.co/{width}x{height}?text={self.model_name}+{i+1}",
                model=self.model_name,
                prompt_used=prompt,
                seed=(seed or 0) + i,
                width=width,
                height=height,
            )
            for i in range(num)
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Real REST-based generators
# ═══════════════════════════════════════════════════════════════════════════

class SDXLModel(BaseImageModel):
    """Stability AI SDXL / SD3 via REST."""

    model_name = "sdxl"

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
                    cfg.sdxl_endpoint,
                    headers={
                        "Authorization": f"Bearer {cfg.api_key}",
                        "Content-Type": "multipart/form-data",
                    },
                    data={"prompt": prompt, "output_format": "jpeg"},
                    files={"none": b""},
                )
                resp.raise_for_status()
                data = resp.json()
                results.append(
                    GeneratedImage(
                        url=data.get("image", data.get("url", "")),
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or int(time.time())) + i,
                        width=width,
                        height=height,
                    )
                )
            return results


class FluxModel(BaseImageModel):
    """Flux (Black Forest Labs) via Replicate REST API."""

    model_name = "flux"

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
                    cfg.flux_endpoint,
                    headers={
                        "Authorization": f"Token {cfg.api_key}",
                        "Content-Type": "application/json",
                    },
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
                # Replicate returns output as list of URLs
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
