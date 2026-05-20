"""Critic Scoring Node.

Evaluates every generated image using three scorers:
  1. CLIP score (text-image similarity)
  2. Aesthetic score (LAION predictor)
  3. GPT-4o Vision score

Computes weighted final score per image and selects the best one.
"""

from __future__ import annotations

import asyncio
import logging

from config import settings
from state import AgentState, GeneratedImage, ImageScores
from services.llm import LLMService
from services.scorer import score_clip, score_aesthetic

logger = logging.getLogger(__name__)


async def _score_single_image(prompt: str, image: GeneratedImage) -> ImageScores:
    """Run all three scorers on one image concurrently."""
    llm = LLMService()

    # Run CLIP, Aesthetic, GPT-Vision in parallel
    clip_task = score_clip(prompt, image.url)
    aesthetic_task = score_aesthetic(image.url)
    vision_task = llm.score_image(prompt, image.url)

    clip, aesthetic, vision_result = await asyncio.gather(
        clip_task, aesthetic_task, vision_task, return_exceptions=True,
    )

    # Handle exceptions gracefully
    if isinstance(clip, Exception):
        logger.warning("CLIP failed for %s: %s", image.url, clip)
        clip = 0.5
    if isinstance(aesthetic, Exception):
        logger.warning("Aesthetic failed for %s: %s", image.url, aesthetic)
        aesthetic = 0.5
    if isinstance(vision_result, Exception):
        logger.warning("GPT-Vision failed for %s: %s", image.url, vision_result)
        vision_result = {"score": 0.5, "reason": "Vision API unavailable"}

    return ImageScores(
        image_url=image.url,
        clip_score=float(clip),
        aesthetic_score=float(aesthetic),
        gpt_vision_score=float(vision_result.get("score", 0.5)),
        gpt_vision_reason=str(vision_result.get("reason", "")),
    )


async def critic_node(state: AgentState) -> dict:
    """Score all generated images and pick the best one.

    Updates state with:
    - images_with_scores: all scored images
    - best_image: the highest-scoring image
    - best_score: its final_score
    """
    prompt: str = state.get("enhanced_prompt", "")
    images: list[GeneratedImage] = state.get("generated_images", [])
    current_best_score: float = state.get("best_score", 0.0)
    current_best_image: GeneratedImage | None = state.get("best_image")
    round_number: int = state.get("round_number", 1)

    logger.info("[CriticNode] Round %d — scoring %d images", round_number, len(images))

    if not images:
        logger.warning("[CriticNode] No images to score")
        return {}

    # Score all images concurrently
    scored: list[ImageScores] = []
    tasks = [_score_single_image(prompt, img) for img in images]
    scored = list(await asyncio.gather(*tasks))

    # Find best in this round
    round_best = max(scored, key=lambda s: s.final_score)

    logger.info(
        "[CriticNode] Round %d best: score=%.3f (CLIP=%.3f Aesthetic=%.3f GPT=%.3f)",
        round_number,
        round_best.final_score,
        round_best.clip_score,
        round_best.aesthetic_score,
        round_best.gpt_vision_score,
    )

    # Update global best if this round is better
    best_image = current_best_image
    best_score = current_best_score
    best_round = state.get("best_round", 0)

    if round_best.final_score > best_score:
        # Find the GeneratedImage corresponding to the best score
        for img in images:
            if img.url == round_best.image_url:
                best_image = img
                break
        best_score = round_best.final_score
        best_round = round_number

    return {
        "images_with_scores": scored,
        "best_image": best_image,
        "best_score": best_score,
        "best_round": best_round,
    }
