"""Decision Node — the control-flow gate in the LangGraph pipeline.

Decides whether to:
  - STOP  (score >= threshold, or max rounds reached)
  - RETRY (refine prompt and loop back to router)
"""

from __future__ import annotations

import logging

from config import settings
from state import AgentState
from services.llm import LLMService

logger = logging.getLogger(__name__)


async def decision_node(state: AgentState) -> dict:
    """Evaluate current state and decide: stop or retry.

    Returns updates including `should_stop` boolean.
    If retrying, also returns a refined `enhanced_prompt`.
    """
    best_score: float = state.get("best_score", 0.0)
    round_number: int = state.get("round_number", 1)
    max_rounds: int = state.get("max_rounds", settings.critic.max_retry_rounds)
    threshold: float = state.get("pass_threshold", settings.critic.pass_threshold)
    scored = state.get("images_with_scores", [])
    original_user_prompt: str = state.get("user_prompt", "")
    current_prompt: str = state.get("enhanced_prompt", "")

    logger.info(
        "[DecisionNode] Round %d/%d — best_score=%.3f threshold=%.2f",
        round_number, max_rounds, best_score, threshold,
    )

    # --- Check stop conditions ---
    if best_score >= threshold:
        logger.info("[DecisionNode] PASSED — score %.3f >= %.2f", best_score, threshold)
        return {"should_stop": True}

    if round_number >= max_rounds:
        logger.info("[DecisionNode] MAX ROUNDS — stopping after %d attempts", round_number)
        return {"should_stop": True}

    # --- Refine prompt for retry ---
    feedback = _build_feedback(scored)
    llm = LLMService()

    refined = await llm.refine_prompt(
        current_prompt=current_prompt,
        original=original_user_prompt,
        score=best_score,
        threshold=threshold,
        feedback=feedback,
    )

    logger.info("[DecisionNode] RETRY — refined prompt: %.100s", refined)

    return {
        "should_stop": False,
        "enhanced_prompt": refined,
        "round_number": round_number + 1,
    }


def _build_feedback(scored: list) -> str:
    """Aggregate critic feedback into a concise summary for the refiner."""
    if not scored:
        return "No scores available."
    # Sort by score descending
    sorted_scores = sorted(scored, key=lambda s: s.final_score, reverse=True)
    lines = []
    for i, s in enumerate(sorted_scores[:3], 1):
        lines.append(
            f"Image #{i}: score={s.final_score:.2f} "
            f"(CLIP={s.clip_score:.2f} Aesthetic={s.aesthetic_score:.2f} "
            f"GPT={s.gpt_vision_score:.2f}) — {s.gpt_vision_reason}"
        )
    return "\n".join(lines)
