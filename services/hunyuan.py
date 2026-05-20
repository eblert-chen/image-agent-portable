"""Tencent Hunyuan (混元) Image model service — hy-image-v3.0.

Async submit + poll pattern:
  POST /v1/api/image/submit  →  {id: "xxx"}
  POST /v1/api/image/query   →  poll until SUCCESS / FAILED
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

HUNYUAN_BASE = "https://tokenhub.tencentmaas.com/v1/api/image"
POLL_INTERVAL = 3
MAX_POLL_SECONDS = 300


class HunyuanImageModel(BaseImageModel):
    """Tencent Hunyuan Image v3.0 — async submit + query."""

    model_name = "hy-image-v3.0"

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
        api_key = os.getenv("HUNYUAN_API_KEY", "")
        if not api_key:
            logger.warning("HUNYUAN_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=float(MAX_POLL_SECONDS + 30)) as client:
            for i in range(num_images):
                try:
                    # Step 1: submit
                    submit_resp = await client.post(
                        f"{HUNYUAN_BASE}/submit",
                        headers=headers,
                        json={"model": self.model_name, "prompt": prompt},
                    )
                    submit_resp.raise_for_status()
                    submit_data = submit_resp.json()
                    task_id = submit_data.get("id") or submit_data.get("task_id")
                    if not task_id:
                        raise ValueError(f"No task id in response: {submit_data}")

                    logger.info("[Hunyuan] Task %s submitted · prompt=%.60s", task_id, prompt)

                    # Step 2: poll
                    image_url = await self._poll(client, headers, task_id)

                    results.append(GeneratedImage(
                        url=image_url or f"https://placehold.co/{width}x{height}?text=hunyuan+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error(
                        "[Hunyuan] Generation %d failed: %s (%s)",
                        i, exc, type(exc).__name__,
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=hunyuan+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results

    async def _poll(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        task_id: str,
    ) -> str | None:
        deadline = asyncio.get_event_loop().time() + MAX_POLL_SECONDS
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                resp = await client.post(
                    f"{HUNYUAN_BASE}/query",
                    headers=headers,
                    json={"model": self.model_name, "id": task_id},
                )
                resp.raise_for_status()
                data = resp.json()
                status = str(data.get("status") or data.get("state", "")).lower()

                if status in ("success", "completed", "done", "succeeded"):
                    # Standard response: data[0].url
                    data_list = data.get("data") or data.get("images") or []
                    if isinstance(data_list, list) and data_list:
                        url = data_list[0].get("url")
                    if not url:
                        url = data.get("url") or (data.get("result", {}).get("url") if isinstance(data.get("result"), dict) else None)
                    if url:
                        logger.info("[Hunyuan] Task %s completed", task_id)
                        return url
                    logger.warning("[Hunyuan] Task %s success but no image URL: %s", task_id, data)
                    return None

                if status in ("failed", "error", "cancelled"):
                    logger.error("[Hunyuan] Task %s failed: %s", task_id, data)
                    return None

                logger.debug("[Hunyuan] Task %s status=%s — polling...", task_id, status)
            except httpx.HTTPError as exc:
                logger.warning("[Hunyuan] Poll error for %s: %s", task_id, exc)

        logger.error("[Hunyuan] Task %s timed out after %ds", task_id, MAX_POLL_SECONDS)
        return None
