"""Image Generation Node.

Invokes the selected model and stores results in state.
Supports model fallback: if the primary model fails, tries fallback_models in order.
"""

from __future__ import annotations

import asyncio
import logging

from config import settings
from state import AgentState, GeneratedImage, RouterDecision
from services.sdxl import SDXLModel, FluxModel
from services.anime import AnimeModel
from services.face3d import Face3DModel
from services.runninghub import Flux2KleinModel
from services.glm import GLMImageModel
from services.stability import StabilityUltraModel
from services.doubao import DoubaoSeedreamModel
from services.modelscope import ModelScopeModel
from services.pollinations import PollinationsModel
from services.wan26 import Wan26ImageModel
from services.hunyuan import HunyuanImageModel
from services.spark import SparkTTIModel
from services.qianfan import QianfanModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry — add new models here
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "sdxl": SDXLModel,
    "flux": FluxModel,
    "sdxl-anime": AnimeModel,
    "face3d": Face3DModel,
    "flux2-klein": Flux2KleinModel,
    "glm-image": GLMImageModel,
    "cogview-4": lambda: GLMImageModel("cogview-4"),
    "cogview-4-250304": lambda: GLMImageModel("cogview-4-250304"),
    "cogview-3-flash": lambda: GLMImageModel("cogview-3-flash"),
    "stable-ultra": StabilityUltraModel,
    "doubao-seedream": DoubaoSeedreamModel,
    "modelscope": ModelScopeModel,
    "modelscope-zimage": lambda: ModelScopeModel("Tongyi-MAI/Z-Image-Turbo"),
    "modelscope-nexus": lambda: ModelScopeModel("modelscope/Nexus-Gen"),
    "modelscope-flux": lambda: ModelScopeModel("black-forest-labs/FLUX.2-dev"),
    "modelscope-sdxl": lambda: ModelScopeModel("HingXuan/WAI-illustrious-SDXL-v17"),
    "pollinations-sana": lambda: PollinationsModel("sana"),
    "pollinations-flux": lambda: PollinationsModel("sana"),
    "pollinations-zimage": lambda: PollinationsModel("sana"),
    "pollinations-turbo": lambda: PollinationsModel("sana"),
    "pollinations-gptimage": lambda: PollinationsModel("gptimage"),
    "pollinations-gptimage2": lambda: PollinationsModel("gpt-image-2"),
    "pollinations-seedream5": lambda: PollinationsModel("seedream5"),
    "pollinations-nanobanana": lambda: PollinationsModel("nanobanana-pro"),
    "pollinations-kontext": lambda: PollinationsModel("kontext"),
    "pollinations-grok": lambda: PollinationsModel("grok-imagine"),
    "wan2.6-image": Wan26ImageModel,
    "hy-image-v3.0": HunyuanImageModel,
    "spark-tti": SparkTTIModel,
    "qianfan-qwen": QianfanModel,
}


# ---------------------------------------------------------------------------
# Node entry-point
# ---------------------------------------------------------------------------

async def generate_node(state: AgentState) -> dict:
    """Run all routed models concurrently — 1 image per model.

    Instead of generating 4 similar images from a single model,
    we fire every model in the router's list (primary + fallbacks)
    in parallel, each producing exactly 1 image. The critic then
    scores all results to pick the best one.
    """
    prompt: str = state.get("enhanced_prompt", "")
    decision: RouterDecision | None = state.get("router_decision")
    errors: list[str] = list(state.get("errors", []))

    if not decision:
        decision = RouterDecision(
            category="general",
            primary_model="doubao-seedream",
            fallback_models=["wan2.6-image", "hy-image-v3.0", "modelscope-zimage"],
        )

    # Fixed 6-model ensemble for connectivity check
    model_keys = ["doubao-seedream", "wan2.6-image", "hy-image-v3.0", "modelscope-zimage", "spark-tti", "qianfan-qwen"]

    gen_width = state.get("image_width", settings.image.default_width)
    gen_height = state.get("image_height", settings.image.default_height)

    logger.info(
        "[GenerateNode] Firing %d models in parallel: %s",
        len(model_keys), ", ".join(model_keys),
    )

    async def _try_model(model_key: str) -> tuple[str, list[GeneratedImage], str | None]:
        """Generate 1 image from a single model. Returns (key, images, error)."""
        model_cls = MODEL_REGISTRY.get(model_key)
        if model_cls is None:
            return model_key, [], f"Unknown model key: {model_key}"

        try:
            model = model_cls()
            imgs = await model.generate(
                prompt, num_images=1, width=gen_width, height=gen_height,
            )
            if imgs:
                logger.info("[GenerateNode] %s → 1 image", model_key)
                return model_key, imgs, None
            else:
                return model_key, [], f"{model_key} returned 0 images"
        except Exception as exc:
            return model_key, [], f"{model_key}: {exc}"

    # Launch all models concurrently
    results = await asyncio.gather(
        *[_try_model(k) for k in model_keys], return_exceptions=True,
    )

    # Collect images and errors
    all_images: list[GeneratedImage] = []
    for result in results:
        if isinstance(result, Exception):
            errors.append(f"Model task crashed: {result}")
            continue
        _key, imgs, err = result
        if imgs:
            all_images.extend(imgs)
        if err:
            errors.append(err)

    if not all_images:
        logger.warning("[GenerateNode] All models failed — using emergency placeholders")
        placeholder = SDXLModel()
        all_images = placeholder._placeholder_images(
            prompt, 1, gen_width, gen_height, None,
        )

    logger.info("[GenerateNode] Collected %d images from %d models", len(all_images), len(model_keys))

    if all_images:
        from stats import record_usage
        for img in all_images:
            asyncio.create_task(record_usage(img.model, 1))

    return {
        "generated_images": all_images,
        "errors": errors,
        "model_fallback_used": None,
    }
