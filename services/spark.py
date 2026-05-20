"""iFlytek Spark TTI model service (讯飞星火 图文生成).

Endpoint: POST http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti
Auth: HMAC-SHA256 signature with APP_ID + API_KEY + API_SECRET
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import httpx

from services.sdxl import BaseImageModel
from state import GeneratedImage

logger = logging.getLogger(__name__)

SPARK_HOST = "spark-api.cn-huabei-1.xf-yun.com"
SPARK_URL = "http://spark-api.cn-huabei-1.xf-yun.com/v2.1/tti"


# ---------------------------------------------------------------------------
# HMAC-SHA256 auth — required by iFlytek Spark API
# ---------------------------------------------------------------------------

def _build_auth_url(api_key: str, api_secret: str) -> str:
    """Build the authenticated URL with host / date / authorization query params."""
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))
    path = "/v2.1/tti"

    signature_origin = f"host: {SPARK_HOST}\ndate: {date}\nPOST {path} HTTP/1.1"
    signature_sha = hmac.new(
        api_secret.encode("utf-8"),
        signature_origin.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    signature_b64 = base64.b64encode(signature_sha).decode("utf-8")

    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature_b64}"'
    )
    authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode("utf-8")

    params = urlencode({"host": SPARK_HOST, "date": date, "authorization": authorization})
    return f"{SPARK_URL}?{params}"


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class SparkTTIModel(BaseImageModel):
    """iFlytek Spark TTI — 讯飞星火图文生成."""

    model_name = "spark-tti"

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
        app_id = os.getenv("SPARK_APP_ID", "")
        api_key = os.getenv("SPARK_API_KEY", "")
        api_secret = os.getenv("SPARK_API_SECRET", "")

        if not all([app_id, api_key, api_secret]):
            logger.warning("Spark credentials not set — using placeholder")
            return self._placeholder_images(prompt, num_images, width, height, seed)

        results: list[GeneratedImage] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(num_images):
                try:
                    url = _build_auth_url(api_key, api_secret)
                    body = {
                        "header": {"app_id": app_id, "uid": "image-agent"},
                        "parameter": {
                            "chat": {
                                "domain": "general",
                                "temperature": 0.5,
                                "max_tokens": 4096,
                            }
                        },
                        "payload": {
                            "message": {
                                "text": [{"role": "user", "content": prompt}]
                            }
                        },
                    }

                    resp = await client.post(
                        url,
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )

                    if resp.status_code != 200:
                        logger.error(
                            "Spark API error %d: %s",
                            resp.status_code,
                            resp.text[:500],
                        )
                        resp.raise_for_status()

                    data = resp.json()
                    code = data.get("header", {}).get("code", -1)
                    if code != 0:
                        logger.error("Spark API code %d: %s", code, data)
                        raise RuntimeError(f"Spark API error code {code}")

                    text_choices = data.get("payload", {}).get("choices", {}).get("text", [])
                    if text_choices:
                        img_b64 = text_choices[0].get("content", "")
                        url_out = f"data:image/png;base64,{img_b64}"
                    else:
                        url_out = ""

                    results.append(GeneratedImage(
                        url=url_out or f"https://placehold.co/{width}x{height}?text=spark+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=None,
                        width=width,
                        height=height,
                    ))
                except Exception as exc:
                    logger.error(
                        "Spark generation %d failed: %s (%s)", i, exc, type(exc).__name__
                    )
                    results.append(GeneratedImage(
                        url=f"https://placehold.co/{width}x{height}?text=spark+error+{i+1}",
                        model=self.model_name,
                        prompt_used=prompt,
                        seed=None,
                        width=width,
                        height=height,
                    ))
        return results
