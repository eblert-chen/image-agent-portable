"""Baidu Qianfan Qwen-Image model service (百度千帆 通义千问图像).

Endpoint: POST https://qianfan.baidubce.com/v2/images/generations
Auth: Bearer <ACCESS_TOKEN>
"""

from __future__ import annotations

import logging
import os

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

QIANFAN_ENDPOINT = "https://qianfan.baidubce.com/v2/images/generations"
QIANFAN_MODEL = "qwen-image"


class QianfanModel(BaseImageModel):
    """Baidu Qianfan Qwen-Image — 百度千帆平台通义千问图像生成."""

    model_name = "qianfan-qwen"

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
        api_key = os.getenv("QIANFAN_API_KEY", "")
        if not api_key:
            logger.warning("QIANFAN_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(num_images):
                try:
                    body: dict = {
                        "model": QIANFAN_MODEL,
                        "prompt": prompt,
                    }
                    if width and height:
                        body["size"] = f"{width}x{height}"

                    resp = await client.post(
                        QIANFAN_ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )

                    if resp.status_code != 200:
                        logger.error(
                            "Qianfan API error %d: %s",
                            resp.status_code,
                            resp.text[:500],
                        )
                        resp.raise_for_status()

                    data = resp.json()
                    items = data.get("data", [])
                    if items:
                        img_data = items[0]
                        url = img_data.get("url", "")
                        if not url and "b64_json" in img_data:
                            url = f"data:image/jpeg;base64,{img_data['b64_json']}"
                    else:
                        url = ""

                    results.append(GeneratedImage(
                        url=url or f"https://placehold.co/{width}x{height}?text=qianfan+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=seed,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error(
                        "Qianfan generation %d failed: %s (%s)", i, exc, type(exc).__name__
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=qianfan+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=None,
                        width=width,
                        height=height,
                    ))
        return results
