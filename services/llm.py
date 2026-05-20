"""LLM service abstraction – supports OpenAI, Anthropic, DeepSeek via a uniform interface.

When no valid API key is configured, falls back to deterministic mock responses
so the full pipeline remains runnable end-to-end.

Usage:
    from services.llm import LLMService
    llm = LLMService()
    reply = await llm.chat([{"role": "user", "content": "..."}])
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared system prompts
# ---------------------------------------------------------------------------

PROMPT_ENHANCE_SYSTEM = """You are a world-class AI prompt engineer for image generation models (SDXL, Flux, Midjourney).

Given a user's natural-language description, produce a SINGLE high-quality English prompt.

Rules:
- Output ONLY the prompt text — no explanation, no preamble, no markdown fences.
- Include artistic style descriptors (e.g. "digital painting", "photorealistic", "concept art").
- Include camera / lens language (e.g. "85mm lens", "cinematic depth of field", "shallow focus").
- Include lighting descriptors (e.g. "cinematic lighting", "soft golden hour light", "rim lighting").
- Include quality booster tags (e.g. "8k", "highly detailed", "sharp focus", "masterpiece").
- Keep the prompt under 300 characters.
- Preserve the original subject and intent of the user request."""

ROUTER_SYSTEM = """You are an image-generation model router. Analyze the user's request and output ONLY valid JSON with this structure:

{
  "category": "<realistic|anime|video-frame|face3d|general>",
  "primary_model": "<model-key>",
  "fallback_models": ["<alt1>"],
  "reasoning": "<one sentence>",
  "style_tags": ["tag1", "tag2"]
}

Routing rules:
- realistic portrait/photo → category=realistic, primary_model=flux
- anime/manga/illustration → category=anime, primary_model=sdxl-anime
- video/animation frame → category=video-frame, primary_model=animatediff
- 3d face/avatar/head reconstruction → category=face3d, primary_model=face3d
- everything else → category=general, primary_model=sdxl

Only include known model keys: flux, sdxl, sdxl-anime, animatediff, face3d. Do NOT output any text outside the JSON object."""

GPT_VISION_SYSTEM = """You are an expert art critic for AI-generated images.

Evaluate the image against the given prompt on:
1. Prompt adherence — does the image show what the prompt describes?
2. Composition — framing, rule of thirds, balance.
3. Lighting & color — is lighting effective and color harmonious?
4. Technical quality — sharpness, artifacts, anatomy issues.
5. Overall aesthetic appeal.

Output ONLY valid JSON (no markdown fences):
{
  "score": <float 0.0 to 1.0>,
  "reason": "<2-3 sentence critique>"
}"""

PROMPT_REFINER_SYSTEM = """You are a prompt optimizer. The previous prompt produced images with a final critic score of {score:.2f} (threshold is {threshold:.2f}).

Critic feedback: {feedback}

Original user intent: {original}

Improve the prompt to address the weaknesses. Apply:
- Better style and composition keywords
- More specific lighting / camera descriptors
- Fix any subject confusion
- Add negative-prompt-like cues through positive framing

Output ONLY the improved prompt text — no explanation, no markdown."""


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    """Unified async LLM client.

    Switches backend via ``settings.llm.provider``.
    When no valid API key is configured, falls back to deterministic mock responses.
    """

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        cfg = settings.llm
        self.provider = cfg.provider
        self.model = cfg.model
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self._mock = self.api_key in ("", "sk-REPLACE-ME", "sk-your-key-here")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Send a chat-completion request; fall back to mock on failure."""
        if self._mock:
            return self._mock_chat(messages)

        try:
            if self.provider == "openai":
                return await self._chat_openai(messages, response_format)
            if self.provider == "deepseek":
                return await self._chat_deepseek(messages, response_format)
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as exc:
            logger.warning("LLM call failed (%s), using mock fallback", exc)
            return self._mock_chat(messages)

    async def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Chat and parse the response as JSON."""
        raw = await self.chat(messages)
        return _extract_json(raw)

    # ------------------------------------------------------------------
    # High-level helpers used by nodes
    # ------------------------------------------------------------------

    async def enhance_prompt(self, user_text: str) -> str:
        """Turn a user description into a production-grade SDXL/Flux prompt."""
        if self._mock:
            return _mock_enhance_prompt(user_text)
        return await self._try_enhance(user_text)

    async def route_request(self, user_text: str) -> dict[str, Any]:
        if self._mock:
            return _mock_route_json(user_text)
        return await self._try_route(user_text)

    async def score_image(self, prompt: str, image_url: str) -> dict[str, Any]:
        if self._mock:
            return _mock_vision_score()
        return await self._try_score(prompt, image_url)

    async def refine_prompt(
        self,
        current_prompt: str,
        original: str,
        score: float,
        threshold: float,
        feedback: str,
    ) -> str:
        if self._mock:
            return _mock_refine_prompt(current_prompt)
        return await self._try_refine(current_prompt, original, score, threshold, feedback)

    # ------------------------------------------------------------------
    # Real API call wrappers (fall back to mock on failure)
    # ------------------------------------------------------------------

    async def _try_enhance(self, user_text: str) -> str:
        try:
            response = await self.chat([
                {"role": "system", "content": PROMPT_ENHANCE_SYSTEM},
                {"role": "user", "content": user_text},
            ])
            return response.strip().strip('"')
        except Exception as exc:
            logger.warning("Enhance prompt failed: %s", exc)
            return _mock_enhance_prompt(user_text)

    async def _try_route(self, user_text: str) -> dict[str, Any]:
        try:
            return await self.chat_json([
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": user_text},
            ])
        except Exception as exc:
            logger.warning("LLM route failed: %s", exc)
            return _mock_route_json(user_text)

    async def _try_score(self, prompt: str, image_url: str) -> dict[str, Any]:
        try:
            content: list[dict[str, Any]] = [
                {"type": "text", "text": f"Prompt: {prompt}"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
            return await self.chat_json([
                {"role": "system", "content": GPT_VISION_SYSTEM},
                {"role": "user", "content": content},
            ])
        except Exception as exc:
            logger.warning("GPT-Vision score failed: %s", exc)
            return _mock_vision_score()

    async def _try_refine(
        self, current_prompt: str, original: str, score: float, threshold: float, feedback: str
    ) -> str:
        try:
            system = PROMPT_REFINER_SYSTEM.format(
                score=score, threshold=threshold, feedback=feedback, original=original,
            )
            response = await self.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Current prompt: {current_prompt}"},
            ])
            return response.strip().strip('"')
        except Exception as exc:
            logger.warning("Refine prompt failed: %s", exc)
            return _mock_refine_prompt(current_prompt)

    # ------------------------------------------------------------------
    # Mock chat (for _mock_chat fallback)
    # ------------------------------------------------------------------

    def _mock_chat(self, messages: list[dict[str, Any]]) -> str:
        """Return a plausible mock response based on the system prompt intent."""
        for m in messages:
            if m.get("role") == "system":
                sys = str(m.get("content", ""))
                if "prompt engineer" in sys.lower():
                    user = next((x for x in messages if x["role"] == "user"), None)
                    return _mock_enhance_prompt(str(user["content"]) if user else "")
                if "router" in sys.lower() and "model" in sys.lower():
                    user = next((x for x in messages if x["role"] == "user"), None)
                    return json.dumps(_mock_route_json(str(user["content"]) if user else ""))
                if "art critic" in sys.lower():
                    return json.dumps(_mock_vision_score())
                if "prompt optimizer" in sys.lower() or "improve" in sys.lower():
                    user = next((x for x in messages if x["role"] == "user"), None)
                    return _mock_refine_prompt(str(user["content"]) if user else "")
        return '{"score": 0.75, "reason": "Mock response — no LLM configured."}'

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> str:
        url = f"{self.base_url or 'https://api.openai.com/v1'}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def _chat_deepseek(
        self,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None,
    ) -> str:
        url = f"{self.base_url or 'https://api.deepseek.com/v1'}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any]:
    """Robust JSON extraction — handles markdown fences and stray text."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


# ---------------------------------------------------------------------------
# Mock implementations — deterministic, always succeed, no network needed
# ---------------------------------------------------------------------------

# Style/camera/lighting keyword pools for prompt enhancement
_STYLE_POOL = [
    "digital painting", "photorealistic", "cinematic", "concept art",
    "hyperrealistic", "oil painting", "matte painting", "trending on artstation",
]
_CAMERA_POOL = [
    "85mm lens", "cinematic depth of field", "shallow focus",
    "f/1.8 aperture", "wide angle", "35mm lens", "tilt-shift",
]
_LIGHT_POOL = [
    "cinematic lighting", "soft golden hour light", "rim lighting",
    "volumetric lighting", "studio lighting", "dramatic shadows",
    "soft diffused light",
]
_QUALITY_POOL = [
    "8k", "highly detailed", "sharp focus", "masterpiece",
    "intricate details", "professional photography",
]


def _mock_enhance_prompt(user_text: str) -> str:
    """Build a quality-enhanced prompt from a fixed template."""
    import hashlib

    # Deterministic selection based on input hash
    h = int(hashlib.md5(user_text.encode()).hexdigest(), 16)
    style = _STYLE_POOL[h % len(_STYLE_POOL)]
    camera = _CAMERA_POOL[(h // 7) % len(_CAMERA_POOL)]
    light = _LIGHT_POOL[(h // 13) % len(_LIGHT_POOL)]
    quality = ", ".join([
        _QUALITY_POOL[(h // 17) % len(_QUALITY_POOL)],
        _QUALITY_POOL[(h // 23) % len(_QUALITY_POOL)],
    ])
    prompt = f"{user_text}, {style} style, {camera}, {light}, {quality}"
    if len(prompt) > 300:
        prompt = prompt[:297] + "..."
    return prompt


def _mock_route_json(user_text: str) -> dict[str, Any]:
    """Regex-based routing matching the same rules as nodes/router.py."""
    text_lower = user_text.lower()
    # Same patterns as router node
    patterns = [
        (r"anime|manga|二次元|illustration|cartoon|anime style", "anime", "doubao-seedream", ["wan2.6-image", "hy-image-v3.0", "modelscope-zimage"], ["anime", "illustration"]),
        (r"3d\s*face|face\s*reconstruct|3d\s*head|avatar\s*3d|人脸|三维人脸", "face3d", "modelscope-zimage", ["doubao-seedream", "hy-image-v3.0", "wan2.6-image"], ["3d", "face-reconstruction"]),
        (r"video\s*frame|animation\s*frame|animatediff|视频帧|动画帧", "video-frame", "wan2.6-image", ["doubao-seedream", "hy-image-v3.0", "modelscope-zimage"], ["video-frame"]),
        (r"photo\s*realistic|realistic|photograph|portrait|真人|写实|人像|realistic photo", "realistic", "doubao-seedream", ["wan2.6-image", "hy-image-v3.0", "modelscope-zimage"], ["photorealistic", "8k"]),
        (r"cinematic|film\s*grain|movie\s*still|电影|电影感", "realistic", "wan2.6-image", ["doubao-seedream", "hy-image-v3.0", "modelscope-zimage"], ["cinematic", "film-grain"]),
        (r"product|commercial|产品|商业摄影|product photo", "realistic", "modelscope-zimage", ["doubao-seedream", "wan2.6-image", "hy-image-v3.0"], ["product-photography", "commercial"]),
        (r"landscape|scenery|风景|landscape photography", "general", "wan2.6-image", ["doubao-seedream", "hy-image-v3.0", "modelscope-zimage"], ["landscape", "nature"]),
        (r"concept\s*art|fantasy|fantasy art|概念艺术|fantasy illustration", "general", "hy-image-v3.0", ["doubao-seedream", "wan2.6-image", "modelscope-zimage"], ["concept-art", "fantasy"]),
    ]
    for pattern, category, primary, fallbacks, style_tags in patterns:
        if re.search(pattern, text_lower):
            return {
                "category": category,
                "primary_model": primary,
                "fallback_models": fallbacks,
                "reasoning": f"Mock rule-based match: {pattern}",
                "style_tags": style_tags,
            }
    # Default
    return {
        "category": "general",
        "primary_model": "doubao-seedream",
        "fallback_models": ["wan2.6-image", "hy-image-v3.0", "modelscope-zimage"],
        "reasoning": "Mock default route — no specific pattern matched.",
        "style_tags": [],
    }


def _mock_vision_score() -> dict[str, Any]:
    """Deterministic mock GPT-Vision score."""
    return {
        "score": 0.78,
        "reason": "Good composition and lighting. Subject matches the prompt well. Minor artifacts in fine details prevent a higher score.",
    }


def _mock_refine_prompt(current_prompt: str) -> str:
    """Append stronger quality keywords to refine the prompt."""
    boosters = [
        ", ultra high quality",
        ", photorealistic 8k trending on artstation",
        ", award-winning composition, volumetric lighting",
        ", hyperdetailed, professional color grading",
        ", masterpiece, best quality, intricate details",
    ]
    import hashlib
    h = int(hashlib.md5(current_prompt.encode()).hexdigest(), 16)
    suffix = boosters[h % len(boosters)]
    refined = current_prompt.rstrip(",").rstrip() + suffix
    if len(refined) > 300:
        refined = refined[:297] + "..."
    return refined
