"""Model Routing Node.

Hybrid router: rule-based fast-path + LLM fallback for ambiguous requests.

Outputs a RouterDecision stored in state.
"""

from __future__ import annotations

import logging
import re

from state import AgentState, RouterDecision
from services.llm import LLMService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based routing — fast, deterministic, no LLM call needed
# ---------------------------------------------------------------------------

RULE_TABLE: list[tuple[str, str, list[str], list[str]]] = [
    # (keyword regex, category, primary, fallbacks, style_tags)
    # 4 configured models: doubao-seedream, modelscope-zimage, wan2.6-image, hy-image-v3.0
    (r"anime|manga|二次元|illustration|cartoon|anime style", "anime", "doubao-seedream", ["wan2.6-image", "hy-image-v3.0", "modelscope-zimage"], ["anime", "illustration"]),
    (r"3d\s*face|face\s*reconstruct|3d\s*head|avatar\s*3d|人脸|三维人脸", "face3d", "modelscope-zimage", ["doubao-seedream", "hy-image-v3.0", "wan2.6-image"], ["3d", "face-reconstruction"]),
    (r"video\s*frame|animation\s*frame|animatediff|视频帧|动画帧", "video-frame", "wan2.6-image", ["doubao-seedream", "hy-image-v3.0", "modelscope-zimage"], ["video-frame"]),
    (r"photo\s*realistic|realistic|photograph|portrait|真人|写实|人像|realistic photo", "realistic", "doubao-seedream", ["wan2.6-image", "hy-image-v3.0", "modelscope-zimage"], ["photorealistic", "8k"]),
    (r"cinematic|film\s*grain|movie\s*still|电影|电影感", "realistic", "wan2.6-image", ["doubao-seedream", "hy-image-v3.0", "modelscope-zimage"], ["cinematic", "film-grain"]),
    (r"product|commercial|产品|商业摄影|product photo", "realistic", "modelscope-zimage", ["doubao-seedream", "wan2.6-image", "hy-image-v3.0"], ["product-photography", "commercial"]),
    (r"landscape|scenery|风景|landscape photography", "general", "wan2.6-image", ["doubao-seedream", "hy-image-v3.0", "modelscope-zimage"], ["landscape", "nature"]),
    (r"concept\s*art|fantasy|fantasy art|概念艺术|fantasy illustration", "general", "hy-image-v3.0", ["doubao-seedream", "wan2.6-image", "modelscope-zimage"], ["concept-art", "fantasy"]),
    (r"chinese|中国风|国风|水墨|古风|traditional\s*chinese", "general", "spark-tti", ["doubao-seedream", "wan2.6-image", "hy-image-v3.0", "modelscope-zimage"], ["chinese-style", "traditional"]),
]


def _rule_route(user_text: str) -> RouterDecision | None:
    """Try to match user text against deterministic rules."""
    text_lower = user_text.lower()
    for pattern, category, primary, fallbacks, style_tags in RULE_TABLE:
        if re.search(pattern, text_lower):
            return RouterDecision(
                category=category,
                primary_model=primary,
                fallback_models=fallbacks,
                reasoning=f"Rule-based match: {pattern}",
                style_tags=style_tags,
            )
    return None


async def _llm_route(user_text: str) -> RouterDecision:
    """Use LLM to classify the request and pick a model."""
    llm = LLMService()
    result = await llm.route_request(user_text)
    return RouterDecision(
        category=result.get("category", "general"),
        primary_model=result.get("primary_model", "doubao-seedream"),
        fallback_models=result.get("fallback_models", ["wan2.6-image", "hy-image-v3.0", "modelscope-zimage", "spark-tti"]),
        reasoning=result.get("reasoning", ""),
        style_tags=result.get("style_tags", []),
    )


# ---------------------------------------------------------------------------
# Node entry-point
# ---------------------------------------------------------------------------

async def router_node(state: AgentState) -> dict:
    """Determine which model(s) to use for the given prompt.

    Strategy: rule-first, LLM-fallback.
    """
    prompt: str = state.get("enhanced_prompt", "") or state.get("user_prompt", "")
    logger.info("[RouterNode] Routing for: %s", prompt[:80])

    # 1. Try rule-based routing
    decision = _rule_route(prompt)

    # 2. Fall back to LLM if rules didn't match
    if decision is None:
        logger.info("[RouterNode] No rule match — using LLM router")
        decision = await _llm_route(prompt)

    logger.info(
        "[RouterNode] → category=%s primary=%s fallbacks=%s",
        decision.category,
        decision.primary_model,
        decision.fallback_models,
    )

    return {"router_decision": decision}
