"""Zhipu GLM-Image / CogView model service.

API: https://open.bigmodel.cn/api/paas/v4/images/generations
Models: glm-image, cogview-4-250304, cogview-4, cogview-3-flash
"""

from __future__ import annotations

import logging
import os
import time

import httpx
import jwt

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

GLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/images/generations"


def _generate_zhipu_token(api_key: str, exp_seconds: int = 3600) -> str:
    """Generate a JWT token for Zhipu API authentication."""
    key_id, secret = api_key.split(".", 1)
    now_ms = int(time.time() * 1000)
    payload = {
        "api_key": key_id,
        "exp": now_ms + exp_seconds * 1000,
        "timestamp": now_ms,
    }
    return jwt.encode(payload, secret, algorithm="HS256", headers={"alg": "HS256", "sign_type": "SIGN"})

# glm-image 推荐尺寸 (宽高均为32的倍数, 1024-2048px, ≤2^22 px)
GLM_SIZES = ["1280x1280", "1568x1056", "1056x1568", "1472x1088", "1088x1472", "1728x960", "960x1728"]

# 其他CogView模型推荐尺寸 (16的倍数, 512-2048px, ≤2^21 px)
COGVIEW_SIZES = ["1024x1024", "768x1344", "864x1152", "1344x768", "1152x864", "1440x720", "720x1440"]


class GLMImageModel(BaseImageModel):
    """Zhipu GLM-Image / CogView image generation."""

    model_name = "glm-image"

    def __init__(self, variant: str = "glm-image") -> None:
        super().__init__()
        self.variant = variant  # glm-image | cogview-4 | cogview-4-250304 | cogview-3-flash
        self.model_name = variant

    async def generate(
        self,
        prompt: str,
        *,
        num_images: int = 4,
        width: int = 1280,
        height: int = 1280,
        seed: int | None = None,
        **kwargs,
    ) -> list[GeneratedImage]:
        api_key = os.getenv("ZHIPU_API_KEY", "")
        if not api_key:
            logger.warning("ZHIPU_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        token = _generate_zhipu_token(api_key)

        # Clamp to valid GLM size (round to nearest 32 for glm-image, 16 for others)
        if self.variant == "glm-image":
            width = max(1024, min(2048, (width // 32) * 32))
            height = max(1024, min(2048, (height // 32) * 32))
        else:
            width = max(512, min(2048, (width // 16) * 16))
            height = max(512, min(2048, (height // 16) * 16))
        size = f"{width}x{height}"

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=90.0) as client:
            for i in range(num_images):
                try:
                    resp = await client.post(
                        GLM_ENDPOINT,
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.variant,
                            "prompt": prompt,
                            "size": size,
                        },
                    )
                    if resp.status_code != 200:
                        logger.error("GLM API error %d: %s", resp.status_code, resp.text[:500])
                    resp.raise_for_status()
                    data = resp.json()
                    images = data.get("data", [])
                    url = images[0]["url"] if images else ""
                    results.append(GeneratedImage(
                        url=url or f"https://placehold.co/{width}x{height}?text={self.variant}+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error("GLM generation %d failed: %s", i, exc)
                    try:
                        if hasattr(exc, 'response') and exc.response is not None:
                            logger.error("GLM error body: %s", exc.response.text[:500])
                    except Exception:
                        pass
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text={self.variant}+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results
