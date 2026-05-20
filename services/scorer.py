"""Mock scoring service — portable edition.

CLIP and Aesthetic scoring require torch + open-clip-torch (~2.5 GB).
The portable edition uses deterministic mock scores so the critic pipeline
functions without GPU dependencies. GPT-Vision scoring (API-based) provides
the actual quality signal.

To restore real CLIP scoring: pip install open-clip-torch torch torchvision
and replace this file with the full version from the main project.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def score_clip(prompt: str, image_url: str) -> float:
    """Deterministic mock CLIP score — no ML dependency."""
    return 0.72


async def score_aesthetic(image_url: str) -> float:
    """Deterministic mock aesthetic score — no ML dependency."""
    return 0.68
