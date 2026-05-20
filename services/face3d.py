"""3D Face Reconstruction pipeline service.

MICA → EMOCA → DECA pipeline wrapper.
Provides a mock interface so the system runs without a real GPU pipeline.
"""

from __future__ import annotations

import time

import httpx

from config import settings
from services.sdxl import BaseImageModel
from state import GeneratedImage


class Face3DModel(BaseImageModel):
    """3D face reconstruction via MICA → EMOCA → DECA.

    In production this calls a dedicated GPU micro-service.
    The mock mode returns placeholder rendered face images.
    """

    model_name = "face3d"

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

        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
                resp = await client.post(
                    cfg.face3d_endpoint,
                    json={
                        "prompt": prompt,
                        "num_outputs": num_images,
                        "seed": seed or int(time.time()),
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return [
                    GeneratedImage(
                        url=img["url"],
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=img.get("seed"),
                        width=img.get("width", width),
                        height=img.get("height", height),
                    )
                    for img in data["images"]
                ]
        except (httpx.ConnectError, httpx.TimeoutException):
            # Face3D service is optional — fall back to mock gracefully
            return self._placeholder_images(prompt, num_images, width, height, seed)
