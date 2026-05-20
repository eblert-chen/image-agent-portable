"""Central configuration for Image Generation Agent System.

All settings are overridable via environment variables or a .env file.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Resolve project root (where this file lives, or exe dir when frozen)
# ---------------------------------------------------------------------------
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMConfig:
    """LLM provider settings – swap provider / model / key here."""

    provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | deepseek
    model: str = os.getenv("LLM_MODEL", "gpt-4o")
    api_key: str = os.getenv("OPENAI_API_KEY", "sk-REPLACE-ME")
    base_url: str | None = os.getenv("LLM_BASE_URL")
    temperature: float = 0.7
    max_tokens: int = 2048


# ---------------------------------------------------------------------------
# Image-generation backends
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageGenConfig:
    """Image generation API configuration."""

    sdxl_endpoint: str = os.getenv("SDXL_ENDPOINT", "https://api.stability.ai/v2beta/stable-image/generate/sd3")
    flux_endpoint: str = os.getenv("FLUX_ENDPOINT", "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions")
    anime_endpoint: str = os.getenv("ANIME_ENDPOINT", "https://api.replicate.com/v1/models/stability-ai/sdxl-anime/predictions")
    face3d_endpoint: str = os.getenv("FACE3D_ENDPOINT", "http://localhost:8001/api/v1/face3d")
    api_key: str = os.getenv("IMAGE_GEN_API_KEY", "")
    num_images_per_prompt: int = 4
    default_width: int = 1024
    default_height: int = 1024
    timeout_seconds: int = 120


# ---------------------------------------------------------------------------
# Critic / scoring
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CriticConfig:
    """Weights and thresholds for the critic scoring pipeline."""

    clip_weight: float = 0.0
    aesthetic_weight: float = 0.0
    gpt_vision_weight: float = 1.0
    pass_threshold: float = 0.80
    max_retry_rounds: int = 3


# ---------------------------------------------------------------------------
# Composite config – single import for everything
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    image: ImageGenConfig = field(default_factory=ImageGenConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    debug: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")


settings = Settings()
