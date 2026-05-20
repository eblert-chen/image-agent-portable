"""LangGraph workflow definition for the Image Generation Agent.

Graph topology:

    prompt_enhance → router → generate → critic → decision
                          ↑                              │
                          └────── (retry loop) ──────────┘
                                                        │
                                                     END (stop)

The decision node controls the loop:
  - If should_stop=True  → END
  - If should_stop=False → router (with refined prompt)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from config import settings

from nodes.prompt import prompt_enhance_node
from nodes.router import router_node
from nodes.generator import generate_node
from nodes.critic import critic_node
from nodes.decision import decision_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional edge: where to go after decision?
# ---------------------------------------------------------------------------

def _after_generate(state: AgentState) -> str:
    """Skip critic+decision when scoring is disabled."""
    if state.get("enable_scoring", True):
        return "critic"
    else:
        logger.info("[Graph] Scoring disabled → END")
        return END


def _after_decision(state: AgentState) -> str:
    """Return the next node name based on the decision outcome."""
    if state.get("should_stop", True):
        logger.info("[Graph] Decision → END")
        return END
    else:
        logger.info("[Graph] Decision → router (retry)")
        return "router"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the LangGraph state graph."""

    workflow = StateGraph(AgentState)

    # --- Add nodes ---
    workflow.add_node("prompt_enhance", prompt_enhance_node)
    workflow.add_node("router", router_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("decision", decision_node)

    # --- Wire edges ---
    workflow.set_entry_point("prompt_enhance")

    workflow.add_edge("prompt_enhance", "router")
    workflow.add_edge("router", "generate")
    workflow.add_conditional_edges(
        "generate",
        _after_generate,
        {"critic": "critic", END: END},
    )
    workflow.add_edge("critic", "decision")

    # Conditional branching from decision
    workflow.add_conditional_edges(
        "decision",
        _after_decision,
        {
            "router": "router",
            END: END,
        },
    )

    # --- Compile with in-memory checkpointing ---
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    logger.info("[Graph] Workflow compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Singleton graph instance
# ---------------------------------------------------------------------------

_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    """Return the compiled graph (lazy singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

async def run_pipeline(
    user_prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    max_retries: int | None = None,
    pass_threshold: float | None = None,
    enable_scoring: bool = True,
) -> dict:
    """Run the full pipeline and return the final state as a dict.

    This is the primary entry point called by the FastAPI layer.
    """
    graph = get_graph()

    max_rounds = max_retries if max_retries is not None else settings.critic.max_retry_rounds
    threshold = pass_threshold if pass_threshold is not None else settings.critic.pass_threshold

    # Build the initial state
    initial_state: AgentState = {
        "messages": [],
        "user_prompt": user_prompt,
        "enhanced_prompt": "",
        "generated_images": [],
        "images_with_scores": [],
        "best_image": None,
        "best_score": 0.0,
        "best_round": 0,
        "round_number": 1,
        "max_rounds": max_rounds,
        "pass_threshold": threshold,
        "should_stop": False,
        "enable_scoring": enable_scoring,
        "errors": [],
        "model_fallback_used": None,
        "run_id": uuid.uuid4().hex[:12],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "image_width": width,
        "image_height": height,
    }

    # A unique thread_id enables checkpointing across runs
    config = {"configurable": {"thread_id": initial_state["run_id"]}}

    logger.info("[Graph] Starting pipeline run_id=%s", initial_state["run_id"])

    # Stream through the graph
    final_state: dict = {}
    async for event in graph.astream(initial_state, config):
        # event is a dict like {"node_name": {state_updates}}
        for node_name, updates in event.items():
            logger.debug("[Graph] Node '%s' completed", node_name)
            final_state.update(updates)

    # Write each generated image to the generations table
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    category = final_state.get("router_decision")
    category_name = category.category if category else None
    scored_images: dict[str, any] = {}
    for s in final_state.get("images_with_scores", []):
        scored_images[s.image_url] = s
    try:
        from db import insert_generation
        for img in final_state.get("generated_images", []):
            si = scored_images.get(img.url)
            await insert_generation(
                prompt=user_prompt,
                model=img.model,
                image_base64=img.url,
                enhanced_prompt=final_state.get("enhanced_prompt", ""),
                category=category_name,
                width=img.width,
                height=img.height,
                seed=img.seed,
                best_score=si.final_score if si else None,
                clip_score=si.clip_score if si else None,
                aesthetic_score=si.aesthetic_score if si else None,
                gpt_vision_score=si.gpt_vision_score if si else None,
                source="agent",
                created_at=now_iso,
            )
    except Exception as exc:
        logger.warning("Failed to write generation to DB: %s", exc)

    # Build a clean response
    best_img = final_state.get("best_image")
    return {
        "run_id": initial_state["run_id"],
        "user_prompt": user_prompt,
        "enhanced_prompt": final_state.get("enhanced_prompt", ""),
        "router_decision": _serialize_router(final_state.get("router_decision")),
        "best_image": _serialize_image(best_img),
        "best_score": final_state.get("best_score", 0.0),
        "best_round": final_state.get("best_round", 0),
        "total_rounds": final_state.get("round_number", 1),
        "all_images": [_serialize_image(img) for img in final_state.get("generated_images", [])],
        "all_scores": [
            {
                "image_url": s.image_url,
                "clip_score": round(s.clip_score, 4),
                "aesthetic_score": round(s.aesthetic_score, 4),
                "gpt_vision_score": round(s.gpt_vision_score, 4),
                "final_score": round(s.final_score, 4),
                "gpt_vision_reason": s.gpt_vision_reason,
            }
            for s in final_state.get("images_with_scores", [])
        ],
        "errors": final_state.get("errors", []),
        "model_fallback_used": final_state.get("model_fallback_used"),
    }


def _serialize_router(decision) -> dict | None:
    if decision is None:
        return None
    return {
        "category": decision.category,
        "primary_model": decision.primary_model,
        "fallback_models": decision.fallback_models,
        "reasoning": decision.reasoning,
        "style_tags": decision.style_tags,
    }


def _serialize_image(img) -> dict | None:
    if img is None:
        return None
    return {
        "url": img.url,
        "model": img.model,
        "prompt_used": img.prompt_used,
        "seed": img.seed,
        "width": img.width,
        "height": img.height,
    }


# ---------------------------------------------------------------------------
# Direct generation runner (skip agent pipeline)
# ---------------------------------------------------------------------------

async def run_direct(
    prompt: str,
    model: str = "sdxl",
    width: int = 1024,
    height: int = 1024,
    num_images: int = 4,
    enhance_prompt: bool = False,
    api_key: str | None = None,
) -> dict:
    """Direct image generation — no router, no critic, no retry loop."""
    from services.llm import LLMService
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

    # Optional prompt enhancement
    final_prompt = prompt
    if enhance_prompt:
        llm = LLMService()
        final_prompt = await llm.enhance_prompt(prompt)

    # Clamp resolution per model limits
    MODEL_MAX_DIM: dict[str, int] = {"flux2-klein": 1024, "face3d": 1024, "stable-ultra": 1024, "spark-tti": 2048, "qianfan-qwen": 2048, "doubao-seedream": 2048, "modelscope": 2048, "modelscope-zimage": 2048, "modelscope-nexus": 2048, "modelscope-flux": 2048, "modelscope-sdxl": 2048, "pollinations-flux": 2048, "pollinations-zimage": 2048, "pollinations-turbo": 2048, "pollinations-gptimage": 2048, "pollinations-gptimage2": 2048, "pollinations-seedream5": 2048, "pollinations-nanobanana": 2048, "pollinations-kontext": 2048, "pollinations-grok": 2048, "wan2.6-image": 2048, "hy-image-v3.0": 2048}
    limit = MODEL_MAX_DIM.get(model, 2048)
    width = min(width, limit)
    height = min(height, limit)

    # Inject API key for cloud models
    if api_key:
        import os
        if model in ("glm-image", "cogview-4", "cogview-3-flash"):
            os.environ["ZHIPU_API_KEY"] = api_key
        elif model == "flux2-klein":
            os.environ["RUNNINGHUB_API_KEY"] = api_key
        elif model == "stable-ultra":
            os.environ["STABILITY_API_KEY"] = api_key
        elif model == "doubao-seedream":
            os.environ["DOUBAO_API_KEY"] = api_key
        elif model in ("modelscope", "modelscope-zimage", "modelscope-nexus", "modelscope-flux"):
            os.environ["MODELSCOPE_API_KEY"] = api_key
        elif model.startswith("pollinations-"):
            os.environ["POLLINATIONS_API_KEY"] = api_key
        elif model == "wan2.6-image":
            os.environ["DASHSCOPE_API_KEY"] = api_key
        elif model == "hy-image-v3.0":
            os.environ["HUNYUAN_API_KEY"] = api_key
        elif model == "spark-tti":
            os.environ["SPARK_APP_ID"] = api_key
        elif model == "qianfan-qwen":
            os.environ["QIANFAN_API_KEY"] = api_key

    # Pick model
    registry = {"sdxl": SDXLModel, "flux": FluxModel, "sdxl-anime": AnimeModel, "face3d": Face3DModel, "flux2-klein": Flux2KleinModel, "stable-ultra": StabilityUltraModel, "doubao-seedream": DoubaoSeedreamModel, "glm-image": GLMImageModel, "cogview-4": lambda: GLMImageModel("cogview-4"), "cogview-4-250304": lambda: GLMImageModel("cogview-4-250304"), "cogview-3-flash": lambda: GLMImageModel("cogview-3-flash"), "modelscope": ModelScopeModel, "modelscope-zimage": lambda: ModelScopeModel("Tongyi-MAI/Z-Image-Turbo"), "modelscope-nexus": lambda: ModelScopeModel("modelscope/Nexus-Gen"), "modelscope-flux": lambda: ModelScopeModel("black-forest-labs/FLUX.2-dev"), "modelscope-sdxl": lambda: ModelScopeModel("HingXuan/WAI-illustrious-SDXL-v17"), "pollinations-flux": lambda: PollinationsModel("flux"), "pollinations-zimage": lambda: PollinationsModel("zimage"), "pollinations-turbo": lambda: PollinationsModel("turbo"), "pollinations-gptimage": lambda: PollinationsModel("gptimage"), "pollinations-gptimage2": lambda: PollinationsModel("gpt-image-2"), "pollinations-seedream5": lambda: PollinationsModel("seedream5"), "pollinations-nanobanana": lambda: PollinationsModel("nanobanana-pro"), "pollinations-kontext": lambda: PollinationsModel("kontext"), "pollinations-grok": lambda: PollinationsModel("grok-imagine"), "wan2.6-image": Wan26ImageModel, "hy-image-v3.0": HunyuanImageModel, "spark-tti": SparkTTIModel, "qianfan-qwen": QianfanModel}
    cls = registry.get(model, SDXLModel)
    gen = cls()

    images = await gen.generate(final_prompt, num_images=num_images, width=width, height=height)

    if images:
        import asyncio
        from stats import record_usage
        asyncio.create_task(record_usage(model, len(images)))
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            from db import insert_generation
            for img in images:
                await insert_generation(
                    prompt=prompt,
                    model=model,
                    image_base64=img.url,
                    enhanced_prompt=final_prompt if enhance_prompt else "",
                    width=img.width,
                    height=img.height,
                    seed=img.seed,
                    source="direct",
                    created_at=now_iso,
                )
        except Exception as exc:
            logger.warning("Failed to write generation to DB: %s", exc)

    return {
        "images": images,
        "model_used": model,
        "prompt_used": final_prompt,
    }
