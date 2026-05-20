"""Pollinations.ai image generation service.

Endpoint (GET, returns binary image):
  https://image.pollinations.ai/prompt/{prompt}?model=...&width=...&height=...

Free models (no auth): flux, zimage, turbo
Premium models (auth required): gptimage, kontext, seedream, nanobanana, etc.
Full model list: GET /image/models
"""

from __future__ import annotations

import logging
import os
from urllib.parse import quote, urlencode

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

POLLINATIONS_BASE = "https://image.pollinations.ai"
DEFAULT_MODEL = "zimage"

# Premium models that require auth token
PREMIUM_MODELS = {
    "gptimage", "gptimage-large", "gpt-image-2",
    "kontext",
    "seedream", "seedream5", "seedream-pro",
    "nanobanana", "nanobanana-2", "nanobanana-pro",
    "grok-imagine", "grok-imagine-pro",
    "wan-image", "wan-image-pro",
    "qwen-image",
    "klein",
    "p-image", "p-image-edit",
    "nova-canvas",
}


class PollinationsModel(BaseImageModel):
    """Pollinations.ai — unified image generation API."""

    model_name = "pollinations"

    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or os.getenv("POLLINATIONS_MODEL", DEFAULT_MODEL)

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
        # Auth: only needed for premium models
        api_key = os.getenv("POLLINATIONS_API_KEY", "")
        needs_auth = self._model_id in PREMIUM_MODELS

        params: dict = {
            "model": self._model_id,
            "width": str(width),
            "height": str(height),
        }
        if seed is not None and seed >= 0:
            params["seed"] = str(seed)
        if kwargs.get("enhance"):
            params["enhance"] = "true"
        if kwargs.get("negative_prompt"):
            params["negative_prompt"] = kwargs["negative_prompt"]
        if kwargs.get("nologo"):
            params["nologo"] = "true"

        encoded_prompt = quote(prompt, safe="")
        url = f"{POLLINATIONS_BASE}/prompt/{encoded_prompt}?{urlencode(params)}"

        headers: dict = {}
        if needs_auth:
            if not api_key:
                logger.warning(
                    "POLLINATIONS_API_KEY not set — premium model '%s' may not work",
                    self._model_id,
                )
            headers["Authorization"] = f"Bearer {api_key}"

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            for i in range(num_images):
                try:
                    # Append per-image seed offset if seed is set
                    img_url = url
                    if seed is not None and seed >= 0:
                        img_url = url + f"&seed={seed + i}"

                    resp = await client.get(img_url, headers=headers)
                    if resp.status_code == 401 or resp.status_code == 403:
                        logger.error(
                            "[Pollinations] Auth error %d — premium model '%s' requires valid token",
                            resp.status_code, self._model_id,
                        )
                        return self._placeholder_images(prompt, num_images, width, height, seed)

                    if resp.status_code != 200:
                        logger.error(
                            "[Pollinations] HTTP %d generating image %d: %.300s",
                            resp.status_code, i + 1, resp.text,
                        )

                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "")
                    if content_type.startswith("image/"):
                        import base64
                        b64 = base64.b64encode(resp.content).decode("ascii")
                        fmt = "jpeg" if "jpeg" in content_type else "png"
                        data_url = f"data:image/{fmt};base64,{b64}"
                        results.append(GeneratedImage(
                            url=data_url,
                            model=self._model_id,
                            prompt_used=prompt,
                            seed=(seed or 0) + i,
                            width=width,
                            height=height,
                        ))
                    else:
                        # Might be a JSON error or plain text
                        logger.warning(
                            "[Pollinations] Unexpected content-type '%s' for image %d: %.200s",
                            content_type, i + 1, resp.text,
                        )
                        results.append(GeneratedImage(
                            url=f"https://placehold.co/{width}x{height}?text=pollinations+{i+1}",
                            model=self._model_id,
                            prompt_used=prompt,
                            seed=(seed or 0) + i,
                            width=width,
                            height=height,
                        ))
                except Exception as exc:
                    logger.error(
                        "[Pollinations] Image %d failed: %s (%s)",
                        i, exc, type(exc).__name__,
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=pollinations+error+{i+1}",
                        model=self._model_id,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results
