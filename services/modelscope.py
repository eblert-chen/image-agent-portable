"""ModelScope API Inference image generation service.

Async polling pattern:
  POST /v1/images/generations  →  task_id
  GET  /v1/tasks/{task_id}     →  poll until SUCCEED / FAILED

Supported models (ModelScope Model-Id):
  - Qwen/Qwen-Image
  - Qwen/Qwen-Image-2512
  - Tongyi-MAI/Z-Image-Turbo
  - modelscope/Nexus-Gen
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

MODELSCOPE_BASE = "https://api-inference.modelscope.cn"
DEFAULT_MODEL = "Qwen/Qwen-Image"
POLL_INTERVAL = 3  # seconds between polls
MAX_POLL_SECONDS = 300  # 5-minute timeout


class ModelScopeModel(BaseImageModel):
    """ModelScope API Inference — async image generation."""

    model_name = "modelscope"

    def __init__(self, model_id: str | None = None):
        self._model_id = model_id or os.getenv("MODELSCOPE_MODEL_ID", DEFAULT_MODEL)
        self.model_name = model_id or self.model_name  # report actual model name

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
        api_key = os.getenv("MODELSCOPE_API_KEY", "")
        if not api_key:
            logger.warning("MODELSCOPE_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Clamp size to valid values
        size = f"{width}x{height}"

        async with httpx.AsyncClient(timeout=float(MAX_POLL_SECONDS + 30)) as client:
            results: list[GeneratedImage] = []
            for i in range(num_images):
                try:
                    body: dict = {
                        "model": self._model_id,
                        "prompt": prompt,
                        "size": size,
                    }
                    if seed is not None:
                        body["seed"] = seed + i

                    # Step 1: submit async task
                    submit_resp = await client.post(
                        f"{MODELSCOPE_BASE}/v1/images/generations",
                        headers={**headers, "X-ModelScope-Async-Mode": "true"},
                        json=body,
                    )
                    submit_resp.raise_for_status()
                    task_id = submit_resp.json()["task_id"]
                    logger.info(
                        "[ModelScope] Task %s submitted · model=%s size=%s",
                        task_id, self._model_id, size,
                    )

                    # Step 2: poll for result
                    image_url = await self._poll_task(client, headers, task_id)

                    results.append(GeneratedImage(
                        url=image_url or f"https://placehold.co/{width}x{height}?text=modelscope+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=seed or 0,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error(
                        "[ModelScope] Generation %d failed: %s (%s)",
                        i, exc, type(exc).__name__,
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=modelscope+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
            return results

    async def _poll_task(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        task_id: str,
    ) -> str | None:
        """Poll task status until SUCCEED or FAILED. Returns image URL or None."""
        poll_headers = {**headers, "X-ModelScope-Task-Type": "image_generation"}
        deadline = asyncio.get_event_loop().time() + MAX_POLL_SECONDS

        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                resp = await client.get(
                    f"{MODELSCOPE_BASE}/v1/tasks/{task_id}",
                    headers=poll_headers,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("task_status", "")

                if status == "SUCCEED":
                    images = data.get("output_images", [])
                    if images:
                        logger.info("[ModelScope] Task %s succeeded · %d images", task_id, len(images))
                        return images[0]
                    logger.warning("[ModelScope] Task %s succeeded but no output_images", task_id)
                    return None

                if status == "FAILED":
                    logger.error("[ModelScope] Task %s failed: %s", task_id, data.get("error", "unknown"))
                    return None

                logger.debug("[ModelScope] Task %s status=%s — polling...", task_id, status)
            except httpx.HTTPError as exc:
                logger.warning("[ModelScope] Poll error for %s: %s", task_id, exc)

        logger.error("[ModelScope] Task %s timed out after %ds", task_id, MAX_POLL_SECONDS)
        return None
