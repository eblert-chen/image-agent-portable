"""RunningHub API service — Flux2-Klein-9B-Consistency and other hosted models.

RunningHub provides cloud-hosted AI model inference with an API key.
Model: Flux2-Klein-9B-Consistency (post ID: 2028302502973677570)
"""

from __future__ import annotations

import logging
import os
import time

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

RUNNINGHUB_API = "https://www.runninghub.ai/api/v1"


class Flux2KleinModel(BaseImageModel):
    """Flux2-Klein-9B-Consistency via RunningHub cloud API.

    Sign up at https://studio.aigate.cc to get API credits.
    Set RUNNINGHUB_API_KEY env var to authenticate.
    """

    model_name = "flux2-klein"

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
        api_key = os.getenv("RUNNINGHUB_API_KEY", "")
        if not api_key:
            logger.warning("RUNNINGHUB_API_KEY not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=180.0) as client:
            for i in range(num_images):
                try:
                    # Step 1: submit task
                    submit_resp = await client.post(
                        f"{RUNNINGHUB_API}/task/submit",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model_id": "2028302502973677570",
                            "inputs": {
                                "prompt": prompt,
                                "width": width,
                                "height": height,
                                "seed": (seed or int(time.time())) + i,
                                "num_inference_steps": 28,
                                "guidance_scale": 3.5,
                            },
                        },
                    )
                    submit_resp.raise_for_status()
                    task = submit_resp.json()
                    task_id = task.get("task_id") or task.get("id")

                    if not task_id:
                        raise ValueError(f"No task_id in response: {task}")

                    # Step 2: poll for result (non-blocking, up to 120s)
                    url = await self._poll_task(client, api_key, task_id)
                    results.append(GeneratedImage(
                        url=url,
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error("Flux2-Klein task %d failed: %s", i, exc)
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=klein+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=(seed or 0) + i,
                        width=width,
                        height=height,
                    ))
        return results

    async def _poll_task(self, client: httpx.AsyncClient, api_key: str, task_id: str) -> str:
        """Poll RunningHub task status until complete, return image URL."""
        max_wait = 120
        interval = 2
        elapsed = 0

        while elapsed < max_wait:
            await _async_sleep(interval)
            elapsed += interval

            try:
                status_resp = await client.get(
                    f"{RUNNINGHUB_API}/task/status/{task_id}",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                status_resp.raise_for_status()
                status = status_resp.json()

                state = status.get("status") or status.get("state") or ""
                if state in ("completed", "success", "done"):
                    # Extract image URL from response
                    outputs = status.get("outputs") or status.get("output") or {}
                    url = outputs.get("image_url") or outputs.get("url") or outputs.get("images", [None])[0]
                    if isinstance(url, list):
                        url = url[0]
                    if url:
                        return url
                    raise ValueError(f"No image URL in completed task: {status}")

                if state in ("failed", "error", "cancelled"):
                    raise RuntimeError(f"Task failed: {status}")
            except httpx.HTTPError:
                continue  # retry on transient errors

        raise TimeoutError(f"Task {task_id} timed out after {max_wait}s")


async def _async_sleep(seconds: float) -> None:
    """Async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)
