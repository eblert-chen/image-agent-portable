"""Prompt Enhancement Node.

Takes the userʼs raw natural-language input and produces a high-quality
SDXL / Flux prompt enriched with style, camera, and lighting keywords.
"""

from __future__ import annotations

import logging

from state import AgentState
from services.llm import LLMService

logger = logging.getLogger(__name__)


async def prompt_enhance_node(state: AgentState) -> dict:
    """Enhance user prompt → SDXL/Flux-ready prompt.

    Called as the first node in the LangGraph pipeline.
    """
    user_prompt: str = state.get("user_prompt", "")
    logger.info("[PromptNode] Enhancing: %s", user_prompt[:80])

    llm = LLMService()
    enhanced = await llm.enhance_prompt(user_prompt)

    logger.info("[PromptNode] Enhanced → %s", enhanced[:120])
    return {"enhanced_prompt": enhanced}
