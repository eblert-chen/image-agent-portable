"""FastAPI application — Image Generation Agent System.

Endpoints:
  POST /api/v1/generate  —  Run the full agent pipeline
  GET  /api/v1/health    —  Health check
  GET  /api/v1/docs      —  Swagger UI (auto)

Usage:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path

from graph import run_pipeline, run_direct
from stats import get_daily_stats
from db import (
    init_db, insert_task, get_task, list_tasks, update_task_status, delete_task,
    insert_generation, list_generations, get_generation, delete_generation,
    get_generation_filter_counts,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("image-agent")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    logger.info("Image Generation Agent System starting up ...")
    await init_db()
    # Load stored API keys from DB into environment
    from key_manager import load_keys_from_db
    await load_keys_from_db()
    # Recover orphaned tasks left in "running" state from previous crash/restart
    from db import update_task_status
    all_tasks = await list_tasks()
    for t in all_tasks:
        if t.get("status") == "running":
            logger.warning("Recovering orphaned task %s → failed", t.get("task_id"))
            await update_task_status(t["task_id"], "failed", error="Server restarted — task lost")
    yield
    logger.info("Image Generation Agent System shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Image Generation Agent System",
    description="Industrial-grade image generation agent with prompt enhancement, multi-model routing, and automatic critic-based optimization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Incoming generation request."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language description of the image to generate.",
        examples=["A cyberpunk samurai standing in a rainy Tokyo alley at night"],
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum refinement rounds if score is below threshold.",
    )
    pass_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Score threshold to stop early (0.0-1.0).",
    )
    width: int = Field(
        default=1024,
        ge=512,
        le=2048,
        description="Image width in pixels.",
    )
    height: int = Field(
        default=1024,
        ge=512,
        le=2048,
        description="Image height in pixels.",
    )
    enable_scoring: bool = Field(
        default=True,
        description="Enable critic scoring and retry loop. When disabled, generates once without evaluation.",
    )


class GenerateResponse(BaseModel):
    """Final pipeline response."""

    run_id: str
    user_prompt: str
    enhanced_prompt: str
    router_decision: dict | None
    best_image: dict | None
    best_score: float
    best_round: int
    total_rounds: int
    all_images: list[dict]
    all_scores: list[dict]
    errors: list[str]
    model_fallback_used: str | None
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Task Queue — models, storage, processor
# ---------------------------------------------------------------------------

class TaskItem(BaseModel):
    """Single task in a batch submission."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_retries: int = Field(default=3, ge=1, le=5)
    pass_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    enable_scoring: bool = Field(default=True)
    source: str = Field(default="agent", description="agent | direct | local")
    model: str = Field(default="", description="Model key for direct/local tasks")
    num_images: int = Field(default=4, ge=1, le=8)


class BatchTaskRequest(BaseModel):
    """Batch task submission payload."""
    tasks: list[TaskItem] = Field(..., min_length=1, max_length=20)
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)


# Async task queue (storage backed by SQLite via db.py)
task_queue: asyncio.Queue = asyncio.Queue()
_processor_started: bool = False


async def _process_task_queue():
    """Background coroutine — processes tasks sequentially from the queue."""
    logger.info("Task queue processor started")
    while True:
        task_id = await task_queue.get()
        task = await get_task(task_id)
        if task is None:
            task_queue.task_done()
            continue
        await update_task_status(task_id, "running")
        logger.info("Queue processing task %s [%s]: %.80s", task_id, task.get("source", "agent"), task.get("prompt", ""))
        try:
            if task.get("source") in ("direct", "local"):
                raw = await run_direct(
                    task["prompt"],
                    model=task.get("model", "sdxl"),
                    width=task["width"],
                    height=task["height"],
                    num_images=task.get("num_images", 4),
                )
                # Normalize to same shape as run_pipeline result
                imgs = raw.get("images", [])
                best = imgs[0] if imgs else None
                result = {
                    "run_id": task_id,
                    "user_prompt": task["prompt"],
                    "enhanced_prompt": raw.get("prompt_used", ""),
                    "router_decision": {"primary_model": task.get("model", "")} if task.get("model") else None,
                    "best_image": {"url": best.url, "model": best.model, "seed": best.seed, "width": best.width, "height": best.height} if best else None,
                    "best_score": 1.0,
                    "best_round": 1,
                    "total_rounds": 1,
                    "all_images": [{"url": img.url, "model": img.model, "seed": img.seed, "width": img.width, "height": img.height} for img in imgs],
                    "all_scores": [],
                    "errors": [],
                    "model_fallback_used": None,
                }
            else:
                result = await run_pipeline(
                    task["prompt"],
                    width=task["width"],
                    height=task["height"],
                    max_retries=task["max_retries"],
                    pass_threshold=task["pass_threshold"],
                    enable_scoring=task["enable_scoring"],
                )
            await update_task_status(task_id, "completed", result=result)
            logger.info("Task %s completed · score=%.3f", task_id, result.get("best_score", 0))
        except Exception as exc:
            logger.exception("Task %s failed", task_id)
            await update_task_status(task_id, "failed", error=str(exc))
        task_queue.task_done()


def _ensure_processor():
    """Start the queue processor if not already running."""
    global _processor_started
    if not _processor_started:
        asyncio.create_task(_process_task_queue())
        _processor_started = True


# ---------------------------------------------------------------------------
# Task Queue — API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/tasks")
async def create_tasks(req: BatchTaskRequest):
    """Submit a batch of prompts to the task queue."""
    _ensure_processor()
    created: list[dict] = []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for item in req.tasks:
        tid = uuid.uuid4().hex[:12]
        await insert_task(
            task_id=tid,
            prompt=item.prompt,
            width=req.width,
            height=req.height,
            max_retries=item.max_retries,
            pass_threshold=item.pass_threshold,
            enable_scoring=item.enable_scoring,
            source=item.source,
            model=item.model,
            num_images=item.num_images,
            status="pending",
            created_at=now,
        )
        created.append({"task_id": tid, "prompt": item.prompt[:100], "source": item.source, "status": "pending"})
        task_queue.put_nowait(tid)

    logger.info("Batch enqueued: %d tasks", len(created))
    return {"enqueued": len(created), "tasks": created}


@app.get("/api/v1/tasks")
async def api_list_tasks():
    """Return all tasks, newest first."""
    return await list_tasks()


@app.get("/api/v1/tasks/{task_id}")
async def api_get_task(task_id: str):
    """Return a single task by ID."""
    task = await get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.delete("/api/v1/tasks/{task_id}")
async def api_delete_task(task_id: str):
    """Delete a task (only if not currently running)."""
    task = await get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete a running task")
    await delete_task(task_id)
    return {"deleted": task_id}


# ---------------------------------------------------------------------------
# Static files (GUI)
# ---------------------------------------------------------------------------

import sys as _sys
if getattr(_sys, 'frozen', False):
    STATIC_DIR = Path(_sys._MEIPASS) / "static"
else:
    STATIC_DIR = Path(__file__).parent / "static"
    STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    """Serve the web GUI."""
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
async def health():
    """Liveness probe."""
    return {"status": "healthy", "service": "image-generation-agent"}


@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Run the full image-generation agent pipeline.

    Flow: prompt enhance → route → generate → critic → (retry or finish).
    """
    t0 = time.perf_counter()

    try:
        result = await run_pipeline(
            req.prompt,
            width=req.width,
            height=req.height,
            max_retries=req.max_retries,
            pass_threshold=req.pass_threshold,
            enable_scoring=req.enable_scoring,
        )
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.perf_counter() - t0) * 1000

    return GenerateResponse(
        run_id=result["run_id"],
        user_prompt=result["user_prompt"],
        enhanced_prompt=result["enhanced_prompt"],
        router_decision=result["router_decision"],
        best_image=result["best_image"],
        best_score=result["best_score"],
        best_round=result["best_round"],
        total_rounds=result["total_rounds"],
        all_images=result["all_images"],
        all_scores=result["all_scores"],
        errors=result["errors"],
        model_fallback_used=result["model_fallback_used"],
        elapsed_ms=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Direct / Free model generation (skip agent pipeline)
# ---------------------------------------------------------------------------

class DirectGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(default="sdxl", description="Model key: sdxl, flux, sdxl-anime, face3d, etc.")
    width: int = Field(default=1024, ge=512, le=2048)
    height: int = Field(default=1024, ge=512, le=2048)
    num_images: int = Field(default=4, ge=1, le=8)
    enhance_prompt: bool = Field(default=False, description="Optionally run prompt enhancement before generation")
    api_key: str | None = Field(default=None, description="API key for cloud models (e.g. RunningHub)")


class DirectGenerateResponse(BaseModel):
    images: list[dict]
    model_used: str
    prompt_used: str
    elapsed_ms: float


@app.post("/api/v1/generate/direct", response_model=DirectGenerateResponse)
async def generate_direct(req: DirectGenerateRequest):
    """Direct generation — user picks model, no routing, no critic loop."""
    t0 = time.perf_counter()

    try:
        result = await run_direct(
            prompt=req.prompt,
            model=req.model,
            width=req.width,
            height=req.height,
            num_images=req.num_images,
            enhance_prompt=req.enhance_prompt,
            api_key=req.api_key,
        )
    except Exception as exc:
        logger.exception("Direct generation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.perf_counter() - t0) * 1000

    return DirectGenerateResponse(
        images=[{
            "url": img.url,
            "model": img.model,
            "seed": img.seed,
            "width": img.width,
            "height": img.height,
        } for img in result["images"]],
        model_used=result["model_used"],
        prompt_used=result["prompt_used"],
        elapsed_ms=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Generations (image + prompt history)
# ---------------------------------------------------------------------------

@app.get("/api/v1/generations")
async def generations_list(
    page: int = 1,
    page_size: int = 50,
    model: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    source: str | None = None,
):
    """List generated images with pagination and filters. Returns thumbnail-only,
    use /api/v1/generations/{id} for full base64."""
    items, total = await list_generations(
        page=page, page_size=min(page_size, 200),
        model=model, category=category,
        date_from=date_from, date_to=date_to, source=source,
    )
    filter_counts = await get_generation_filter_counts()
    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": min(page_size, 200),
        "filters": filter_counts,
    }


@app.get("/api/v1/generations/export")
async def export_generations(
    model: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    ids: str | None = None,
):
    """Export selected generations as a ZIP file with images + metadata.json."""
    import base64
    import io
    import zipfile
    from datetime import datetime, timezone

    # Determine which records to export
    if ids:
        id_list = [int(x) for x in ids.split(",") if x.strip().isdigit()]
        records = []
        for gid in id_list:
            r = await get_generation(gid)
            if r:
                records.append(r)
    else:
        records, _ = await list_generations(
            page=1, page_size=9999,
            model=model, category=category,
            date_from=date_from, date_to=date_to,
        )

    if not records:
        raise HTTPException(status_code=404, detail="No generations found for export")

    buf = io.BytesIO()
    metadata_images: list[dict] = []

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in records:
            model_name = rec["model"] or "unknown"
            date = (rec["created_at"] or "unknown")[:10]
            safe_prompt = (rec["prompt"] or "untitled")[:40].replace("/", "_").replace("\\", "_")
            filename = f"{model_name}/{date}/{safe_prompt}_{rec['id']:04d}.jpg"

            # Decode base64 and write as binary
            try:
                header, b64 = rec["image_base64"].split(",", 1)
                img_bytes = base64.b64decode(b64)
                zf.writestr(filename, img_bytes)
            except Exception:
                continue

            metadata_images.append({
                "id": rec["id"],
                "prompt": rec["prompt"],
                "enhanced_prompt": rec.get("enhanced_prompt"),
                "model": rec["model"],
                "category": rec.get("category"),
                "width": rec.get("width"),
                "height": rec.get("height"),
                "best_score": rec.get("best_score"),
                "filename": filename,
                "source": rec.get("source"),
                "created_at": rec.get("created_at"),
            })

        meta = {
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_images": len(metadata_images),
            "images": metadata_images,
        }
        zf.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))

    buf.seek(0)
    date_label = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="export_{date_label}.zip"'},
    )


@app.get("/api/v1/generations/{generation_id}")
async def get_generation_detail(generation_id: int):
    """Return full generation record including base64 image data."""
    rec = await get_generation(generation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return rec


@app.delete("/api/v1/generations/{generation_id}")
async def delete_generation_record(generation_id: int):
    """Delete a single generation record."""
    ok = await delete_generation(generation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Generation not found")
    return {"deleted": generation_id}


# ---------------------------------------------------------------------------
# Usage statistics
# ---------------------------------------------------------------------------

@app.get("/api/v1/stats/daily")
async def daily_stats():
    """Return per-model daily usage statistics."""
    return get_daily_stats()


# ---------------------------------------------------------------------------
# Settings — API key management
# ---------------------------------------------------------------------------

class KeyUpdateRequest(BaseModel):
    keys: dict[str, str] = Field(..., description="Dict of env var name → value")


@app.get("/api/v1/settings/keys")
async def get_api_keys():
    """Return stored API keys (values masked). Also return list of known keys."""
    from db import get_settings
    stored = await get_settings()
    from key_manager import KEY_MAP
    masked = {}
    for k in KEY_MAP:
        v = stored.get(k, "")
        if v:
            masked[k] = v[:4] + "***" + v[-4:] if len(v) > 8 else "***"
        else:
            masked[k] = ""
    return {
        "keys": masked,
        "known_keys": list(KEY_MAP.keys()),
    }


@app.post("/api/v1/settings/keys")
async def set_api_keys(req: KeyUpdateRequest):
    """Save API keys to DB and reload into environment."""
    from db import set_settings
    from key_manager import KEY_MAP

    allowed = set(KEY_MAP.keys())
    filtered = {k: v for k, v in req.keys.items() if k in allowed}
    await set_settings(filtered)

    # Immediately inject into os.environ for running services
    import os
    for k, v in filtered.items():
        if v:
            os.environ[KEY_MAP[k]] = v
        elif k in os.environ:
            del os.environ[k]

    logger.info("Updated %d API key(s)", len(filtered))
    return {"saved": len(filtered), "keys": list(filtered.keys())}


# ---------------------------------------------------------------------------
# Main entry-point for `python app.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # PyInstaller: pass app object directly, no reload
    if getattr(sys, 'frozen', False):
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
