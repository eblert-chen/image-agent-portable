"""Alibaba DashScope WAN2.6 Image model service.

Endpoint: POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
Auth: Bearer <DASHSCOPE_API_KEY>

Pure text-to-image uses enable_interleave=true with SSE streaming.
Image-editing mode (with reference images) uses enable_interleave=false.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


class Wan26ImageModel(BaseImageModel):
    """Alibaba DashScope WAN2.6 Image generation."""

    model_name = "wan2.6-image"

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
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            logger.warning("DASHSCOPE_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        max_dim = max(width, height)
        enable_interleave = True if not kwargs.get("reference_images") else False

        if enable_interleave:
            size = f"{width}*{height}"
        elif max_dim <= 1024:
            size = "1K"
        elif max_dim <= 2048:
            size = "2K"
        else:
            size = "4K"

        watermark = kwargs.get("watermark", False)
        prompt_extend = kwargs.get("prompt_extend", True)
        ref_images = kwargs.get("reference_images")

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for i in range(num_images):
                try:
                    content: list[dict] = [{"text": prompt}]
                    if ref_images:
                        for ref in ref_images:
                            content.append({"image": ref})

                    body: dict = {
                        "model": self.model_name,
                        "input": {
                            "messages": [{"role": "user", "content": content}]
                        },
                        "parameters": {
                            "n": 1,
                            "size": size,
                            "watermark": watermark,
                            "prompt_extend": prompt_extend,
                            "enable_interleave": enable_interleave,
                        },
                    }
                    if seed is not None:
                        body["parameters"]["seed"] = seed + i

                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    }
                    if enable_interleave:
                        headers["X-DashScope-Sse"] = "enable"
                        body["parameters"]["stream"] = True

                    resp = await client.post(
                        DASHSCOPE_ENDPOINT,
                        headers=headers,
                        json=body,
                    )
                    if resp.status_code != 200:
                        logger.error(
                            "WAN2.6 API error %d: %s",
                            resp.status_code,
                            resp.text[:500],
                        )
                    resp.raise_for_status()

                    url = ""
                    content_type = resp.headers.get("content-type", "")

                    if "text/event-stream" in content_type:
                        url = _parse_sse_image(resp.text)
                    else:
                        logger.info("WAN2.6 JSON resp: %.500s", resp.text)
                        try:
                            data = resp.json()
                            url = _extract_url_from_response(data)
                        except Exception:
                            logger.warning("WAN2.6 failed to parse JSON response")

                    results.append(GeneratedImage(
                        url=url or f"https://placehold.co/{width}x{height}?text=wan26+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error(
                        "WAN2.6 generation %d failed: %s (%s)",
                        i, exc, type(exc).__name__,
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=wan26+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results


def _parse_sse_image(text: str) -> str:
    """Parse SSE event stream and extract the last image URL."""
    last_url = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = re.sub(r"^data:\s*", "", line)
            if payload and payload != "[DONE]":
                try:
                    data = json.loads(payload)
                    url = _extract_url_from_response(data)
                    if url:
                        last_url = url
                except json.JSONDecodeError:
                    continue
    return last_url


def _extract_url_from_response(data: dict) -> str:
    try:
        output = data.get("output", {})
        choices = output.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", [])
            if content:
                return content[0].get("image", "")
    except Exception:
        pass
    return ""
