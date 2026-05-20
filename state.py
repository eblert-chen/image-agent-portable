"""LangGraph agent state definition.

Uses a typed dictionary so every node reads/writes well-defined keys.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Sequence

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# Single-image record produced by a generator
# ---------------------------------------------------------------------------
@dataclass
class GeneratedImage:
    """One generated image with its metadata."""

    url: str                         # URL or local path
    model: str                       # e.g. "flux", "sdxl-anime"
    prompt_used: str
    seed: int | None = None
    width: int = 1024
    height: int = 1024


# ---------------------------------------------------------------------------
# Scores record for a single image
# ---------------------------------------------------------------------------
@dataclass
class ImageScores:
    """Scores assigned by one critic round to a single image."""

    image_url: str
    clip_score: float = 0.0
    aesthetic_score: float = 0.0
    gpt_vision_score: float = 0.0
    gpt_vision_reason: str = ""

    @property
    def final_score(self) -> float:
        from config import settings

        c = settings.critic
        return (
            c.clip_weight * self.clip_score
            + c.aesthetic_weight * self.aesthetic_score
            + c.gpt_vision_weight * self.gpt_vision_score
        )


# ---------------------------------------------------------------------------
# Router decision output
# ---------------------------------------------------------------------------
@dataclass
class RouterDecision:
    """What the router decided."""

    category: str                   # realistic | anime | video-frame | face3d | general
    primary_model: str              # e.g. "flux", "sdxl-anime"
    fallback_models: list[str] = field(default_factory=list)
    reasoning: str = ""
    style_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph AgentState
# ---------------------------------------------------------------------------
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """The shared state flowing through every LangGraph node."""

    # ---- Conversation ----
    messages: Annotated[list[BaseMessage], add_messages]

    # ---- Pipeline inputs / outputs ----
    user_prompt: str                          # original user input
    enhanced_prompt: str                      # after prompt-enhancement node
    router_decision: RouterDecision            # routing result
    generated_images: list[GeneratedImage]     # images from current round
    images_with_scores: list[ImageScores]      # scored images for current round
    best_image: GeneratedImage | None          # best image of all rounds
    best_score: float                         # best score across all rounds
    best_round: int                           # which round gave the best
    image_width: int                          # requested image width
    image_height: int                         # requested image height

    # ---- Loop control ----
    round_number: int                         # current retry round (1-indexed)
    max_rounds: int                           # max allowed rounds
    pass_threshold: float                     # score threshold to stop early
    should_stop: bool                         # decision node sets this
    enable_scoring: bool                      # when False, skip critic+decision and stop after generate

    # ---- Errors & fallbacks ----
    errors: list[str]
    model_fallback_used: str | None

    # ---- Metadata ----
    run_id: str
    started_at: str
